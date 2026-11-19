// 語音辨識 Worker：在背景執行 Whisper，避免畫面卡住
import { pipeline, env } from './lib/transformers.min.js';

env.allowLocalModels = false;
env.useBrowserCache = true;
env.backends.onnx.wasm.wasmPaths = new URL('./lib/', self.location.href).href;

let asr = null;
let lastFile = '';
let loadedKey = '';

async function load(model, useGpu) {
  const key = model + (useGpu ? ':gpu' : ':cpu');
  if (asr && loadedKey === key) return useGpu;
  const progress_callback = (p) => {
    if (p.file) lastFile = p.file;
    if (p.status === 'progress' && p.total) {
      self.postMessage({ type: 'download', file: p.file, loaded: p.loaded, total: p.total });
    }
  };
  if (useGpu) {
    const adapter = self.navigator.gpu ? await self.navigator.gpu.requestAdapter().catch(() => null) : null;
    if (!adapter) throw new Error('GPU_UNAVAILABLE');
    try {
      asr = await pipeline('automatic-speech-recognition', model, {
        device: 'webgpu',
        dtype: { encoder_model: 'fp32', decoder_model_merged: 'q4' },
        progress_callback,
      });
      loadedKey = key;
      return true;
    } catch (e) {
      throw new Error('GPU_UNAVAILABLE');
    }
  }
  asr = await pipeline('automatic-speech-recognition', model, { device: 'wasm', dtype: 'q8', progress_callback });
  loadedKey = model + ':cpu';
  return false;
}

self.onmessage = async (ev) => {
  // 每個 Worker 只處理一段（約 30 秒），處理完由主程式關掉，以釋放記憶體
  const { audio, model, useGpu, offset } = ev.data;
  try {
    let gpu;
    try { gpu = await load(model, useGpu); }
    catch (e) {
      if (String(e.message) === 'GPU_UNAVAILABLE') throw e;
      throw new Error(`下載或載入模型失敗（${lastFile || '程式庫'}）：${e.message || e}`);
    }
    self.postMessage({ type: 'loaded', gpu });
    const chunkDur = audio.length / 16000;
    const out = await asr(audio, { language: 'japanese', task: 'transcribe', return_timestamps: true });
    const segs = [];
    for (const c of out.chunks || []) {
      const text = (c.text || '').trim();
      if (!text) continue;
      let [s, e] = c.timestamp;
      s = (s ?? 0) + offset;
      e = (e ?? chunkDur) + offset;
      if (e <= s) e = s + 1.5;
      segs.push({ start: s, end: Math.min(e, offset + chunkDur), ja: text });
    }
    self.postMessage({ type: 'segments', segs });
  } catch (e) {
    self.postMessage({ type: 'error', text: String(e && e.message || e) });
  }
};
