// 語音辨識 Worker：在背景執行 Whisper，避免畫面卡住
import { pipeline, env } from './lib/transformers.min.js';

env.allowLocalModels = false;
env.useBrowserCache = true;
env.backends.onnx.wasm.wasmPaths = new URL('./lib/', self.location.href).href;

let asr = null;
let loadedKey = '';

async function load(model, useGpu) {
  const key = model + (useGpu ? ':gpu' : ':cpu');
  if (asr && loadedKey === key) return useGpu;
  const progress_callback = (p) => {
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
  const { audio, model, useGpu } = ev.data;
  try {
    const gpu = await load(model, useGpu);
    self.postMessage({ type: 'loaded', gpu });
    const SR = 16000, WIN = 30 * SR;
    const total = audio.length;
    for (let start = 0; start < total; start += WIN) {
      const chunk = audio.subarray(start, Math.min(total, start + WIN));
      if (chunk.length < SR * 0.5) break;
      const out = await asr(chunk, { language: 'japanese', task: 'transcribe', return_timestamps: true });
      const offset = start / SR;
      const chunkDur = chunk.length / SR;
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
      self.postMessage({ type: 'segments', segs, done: Math.min(total, start + WIN) / total });
    }
    self.postMessage({ type: 'finished' });
  } catch (e) {
    self.postMessage({ type: 'error', text: String(e && e.message || e) });
  }
};
