# 由 colab.ipynb 在執行時下載並執行；表單參數定義在筆記本裡
import os, sys, subprocess, shutil, time, glob

def sh(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-800:])
    return r.stdout

print("🔖 版本：2026-11-19 v10（上字幕時只輸出字幕版）")
print("① 安裝需要的工具（第一次約 1 分鐘）…")
if not os.path.exists("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"):
    sh("apt-get -qq update && apt-get -qq install -y fonts-noto-cjk > /dev/null")
try:
    import faster_whisper, deep_translator  # noqa
except ImportError:
    sh(f"{sys.executable} -m pip install -q faster-whisper deep-translator")

def 補GPU函式庫():
    """faster-whisper 需要 CUDA 12 的 cuBLAS / cuDNN 9；Colab 環境不一定有，缺的話自動補裝並預先載入"""
    import ctypes
    try:
        ctypes.CDLL("libcublas.so.12", mode=ctypes.RTLD_GLOBAL)
        ctypes.CDLL("libcudnn_ops.so.9", mode=ctypes.RTLD_GLOBAL)
        return True
    except OSError:
        pass
    try:
        sh(f"{sys.executable} -m pip install -q nvidia-cublas-cu12 'nvidia-cudnn-cu12>=9,<10'")
        import site
        roots = site.getsitepackages() + [site.getusersitepackages()]
        libs = []
        for r in roots:
            libs += sorted(glob.glob(os.path.join(r, "nvidia", "cublas", "lib", "libcublasLt.so.12*")))
            libs += sorted(glob.glob(os.path.join(r, "nvidia", "cublas", "lib", "libcublas.so.12*")))
            libs += sorted(glob.glob(os.path.join(r, "nvidia", "cudnn", "lib", "libcudnn*.so.9*")))
        for lib in libs:
            try:
                ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass
        ctypes.CDLL("libcublas.so.12", mode=ctypes.RTLD_GLOBAL)
        return True
    except Exception as e:
        print(f"  （GPU 函式庫補裝失敗：{e}）")
        return False

import numpy as np
from faster_whisper import WhisperModel
from deep_translator import GoogleTranslator

def 秒數(t):
    t = t.strip()
    if not t:
        return None
    parts = [float(x) for x in t.split(":")]
    s = 0.0
    for p in parts:
        s = s * 60 + p
    return s

def ts(t):
    ms = int(round(max(0.0, t) * 1000))
    h, ms = divmod(ms, 3600000); m, ms = divmod(ms, 60000); s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def 讀聲音(path):
    out = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vn", "-f", "s16le", "-ac", "1", "-ar", "16000", "-"],
                         capture_output=True)
    if not out.stdout:
        raise RuntimeError("讀不到影片的聲音：" + out.stderr.decode("utf-8", "ignore")[-300:])
    return np.frombuffer(out.stdout, np.int16).astype(np.float32) / 32768.0

def 翻譯(lines):
    """一次送一批（較快、較不會被擋）；Google 失敗就改用 MyMemory；兩個都失敗保留日文"""
    from deep_translator import MyMemoryTranslator
    google = GoogleTranslator(source="ja", target="zh-TW")
    backup = MyMemoryTranslator(source="ja-JP", target="zh-TW")
    out = list(lines)
    用備用 = False
    BATCH = 20
    for i in range(0, len(lines), BATCH):
        group = lines[i:i + BATCH]
        done = False
        if not 用備用:
            try:
                res = google.translate("\n".join(t.replace("\n", " ") for t in group)).split("\n")
                if len(res) == len(group):
                    out[i:i + BATCH] = [r.strip() for r in res]; done = True
            except Exception as e:
                print(f"  Google 翻譯失敗（{str(e)[:80]}），改用 MyMemory…")
                用備用 = True
        if not done:
            for k, t in enumerate(group):
                for tr in ([backup] if 用備用 else [google, backup]):
                    try:
                        out[i + k] = tr.translate(t).strip(); break
                    except Exception:
                        continue
        print(f"  翻譯 {min(i + BATCH, len(lines))}/{len(lines)} 句完成")
    for z in out:
        print("   ", z)
    return out

def AI模型():
    # 預先壓縮好的 4-bit 版本：下載量小、直接載入
    return "unsloth/Qwen2.5-7B-Instruct-bnb-4bit" if "7B" in 翻譯方式 else "unsloth/Qwen2.5-14B-Instruct-bnb-4bit"
_ai = {}

def 載入AI():
    if "m" in _ai:
        return _ai["tok"], _ai["m"]
    print(f"  載入 AI 翻譯模型 {AI模型().split('/')[-1]}（第一次會下載，約 3–8 分鐘）…")
    sh(f"{sys.executable} -m pip install -q -U bitsandbytes accelerate opencc-python-reimplemented")
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    tok = AutoTokenizer.from_pretrained(AI模型())
    m = AutoModelForCausalLM.from_pretrained(AI模型(), device_map="cuda", torch_dtype=torch.float16)
    m.eval()
    _ai["tok"], _ai["m"] = tok, m
    return tok, m

def 解析(text, n):
    import re
    got = {}
    for line in text.splitlines():
        mm = re.match(r"\s*(\d+)\s*[\.、:：）)]\s*(.*)", line)
        if mm:
            k = int(mm.group(1))
            if 1 <= k <= n and k not in got and mm.group(2).strip():
                got[k] = mm.group(2).strip()
    return [got.get(i + 1) for i in range(n)]

SYSTEM = ("你是專業的日翻中影片字幕譯者，負責把日文字幕翻成自然、口語、通順的台灣繁體中文。\n"
          "翻譯原則：\n"
          "1. 影片多半是訪談或對話，提問者與受訪者一來一往；請依前後文判斷每句是誰說的、在回答什麼。\n"
          "2. 只有日文本身是問句（か、の？、？結尾等）才翻成問句；直述句不要翻成問句。\n"
          "3. 日文常省略主詞，請依上下文補成通順的中文，但不要加入原文沒有的資訊。\n"
          "4. 俚語、委婉說法要依語境理解意思，不要照字面直譯（例如在性相關語境中，「出る」「イク」是射精或高潮的意思）。\n"
          "5. 語音辨識可能有錯字，請依上下文推測正確意思。\n"
          "6. 保持編號與行數完全一致，每行一句，只輸出「編號. 譯文」，不要加任何說明。")

def AI翻譯(lines):
    import torch, importlib
    tok, m = 載入AI()          # 這一步會安裝 opencc，所以要先執行
    importlib.invalidate_caches()
    try:
        from opencc import OpenCC
        轉繁 = OpenCC("s2twp").convert
    except Exception:
        try:
            sh(f"{sys.executable} -m pip install -q opencc-python-reimplemented")
            importlib.invalidate_caches()
            from opencc import OpenCC
            轉繁 = OpenCC("s2twp").convert
        except Exception:
            print("  （簡轉繁工具載入失敗，譯文可能有少數簡體字）")
            轉繁 = lambda x: x
    out = [None] * len(lines)
    B = 12
    for i in range(0, len(lines), B):
        group = lines[i:i + B]
        前文 = "\n".join(f"{lines[k]} → {out[k] or ''}" for k in range(max(0, i - 10), i))
        後文 = "\n".join(lines[i + B:i + B + 5])
        user = ((f"【前文與已完成的翻譯，僅供參考】\n{前文}\n\n" if 前文 else "") +
                (f"【後文，僅供參考】\n{後文}\n\n" if 後文 else "") +
                "【請翻譯以下字幕】\n" + "\n".join(f"{k + 1}. {t}" for k, t in enumerate(group)))
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
        prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        enc = tok(prompt, return_tensors="pt").to(m.device)
        n_in = enc["input_ids"].shape[1]
        with torch.no_grad():
            gen = m.generate(**enc, max_new_tokens=min(2048, 90 * len(group)), do_sample=False,
                             repetition_penalty=1.05, pad_token_id=tok.eos_token_id)
        res = 解析(tok.decode(gen[0][n_in:], skip_special_tokens=True), len(group))
        for k, r in enumerate(res):
            if r:
                out[i + k] = 轉繁(r)
        print(f"  AI 翻譯 {min(i + B, len(lines))}/{len(lines)} 句完成")
    缺 = [k for k, z in enumerate(out) if not z]
    if 缺:
        print(f"  有 {len(缺)} 句 AI 沒翻出來，改用線上翻譯補上…")
        補 = 翻譯([lines[k] for k in 缺])
        for k, z in zip(缺, 補):
            out[k] = z
    for z in out:
        print("   ", z)
    return out

def 燒字幕(video, srt, out_path, height):
    work = "/content/_work"; os.makedirs(work, exist_ok=True)
    shutil.copy(srt, f"{work}/sub.srt")
    fs = 20  # 以 288 行高為基準，任何解析度都會自動等比例縮放
    style = f"FontName=Noto Sans CJK TC,FontSize={fs},Outline=2,Shadow=0,MarginV=18"
    r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", os.path.abspath(video),
                        "-vf", f"subtitles=sub.srt:fontsdir=/usr/share/fonts/opentype/noto:force_style='{style}'",
                        "-c:v", "libx264", "-preset", "veryfast", "-crf", "21", "-c:a", "aac", "-b:a", "160k",
                        "-movflags", "+faststart", os.path.abspath(out_path)], cwd=work, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("燒字幕失敗：" + r.stderr[-500:])

def 辨識(model_holder, audio):
    def run(m):
        segs = []
        for s in m.transcribe(audio, language="ja", vad_filter=True, beam_size=5)[0]:
            t = s.text.strip()
            if t:
                segs.append((s.start, s.end, t)); print(f"  {ts(s.start)[3:8]}  {t}")
        return segs
    try:
        return run(model_holder["m"])
    except Exception as e:
        if model_holder.get("gpu") and ("cublas" in str(e).lower() or "cudnn" in str(e).lower() or "cuda" in str(e).lower()):
            print(f"  ⚠️ GPU 無法使用（{e}），改用 CPU 繼續（會比較慢）…")
            model_holder["m"] = WhisperModel("small" if 模型 == "large-v3-turbo" else 模型, device="cpu", compute_type="int8")
            model_holder["gpu"] = False
            return run(model_holder["m"])
        raise

def 處理(path, model):
    base = os.path.splitext(os.path.basename(path))[0]
    a, b = 秒數(開始), 秒數(結束)
    src = path
    if a is not None or b is not None:
        print(f"② 剪出指定片段 {開始 or '開頭'} ~ {結束 or '結尾'}…")
        cut = f"/content/{base}_片段.mp4"
        cmd = ["ffmpeg", "-v", "error", "-y"]
        if a is not None: cmd += ["-ss", str(a)]
        cmd += ["-i", path]
        if b is not None: cmd += ["-t", str(b - (a or 0))]
        cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-c:a", "aac", cut]
        subprocess.run(cmd, check=True)
        src = cut
    print("③ 辨識日文語音…")
    segs = 辨識(model, 讀聲音(src))
    if not segs:
        print("沒有辨識到日文語音。"); return []
    ja = [s[2] for s in segs]
    zh = None
    if 翻譯方式.startswith("AI"):
        print(f"④ AI 翻譯成繁體中文（共 {len(segs)} 句）…")
        try:
            zh = AI翻譯(ja)
        except Exception as e:
            import traceback
            print(f"  ⚠️ AI 翻譯失敗：{type(e).__name__}: {str(e)[:300]}")
            print("  " + traceback.format_exc().strip().splitlines()[-3][:200])
            print("  改用線上翻譯…")
    if zh is None:
        print(f"④ 線上翻譯成繁體中文（共 {len(segs)} 句）…")
        zh = 翻譯(ja)
    srt = f"/content/{base}_中文字幕.srt"
    with open(srt, "w", encoding="utf-8") as f:
        for i, ((st, en, _), z) in enumerate(zip(segs, zh), 1):
            f.write(f"{i}\n{ts(st)} --> {ts(en)}\n{z}\n\n")
    print("⑤ 把字幕燒進影片…")
    h = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=height",
                        "-of", "csv=p=0", src], capture_output=True, text=True).stdout.strip()
    mp4 = f"/content/{base}_中文字幕.mp4"
    燒字幕(src, srt, mp4, int(h) if h.isdigit() else 0)
    return [mp4, srt]

def 備用取得影片網址(url):
    """yt-dlp 抓不到時（例如敏感內容要登入），改問 FxTwitter / vxTwitter 要影片網址"""
    import re, requests
    m = re.search(r"/status(?:es)?/(\d+)", url)
    if not m:
        return []
    tid = m.group(1)
    found = []
    try:
        j = requests.get(f"https://api.fxtwitter.com/status/{tid}", timeout=20).json()
        tw = j.get("tweet") or {}
        for src in [tw, tw.get("quote") or {}]:
            for v in ((src.get("media") or {}).get("videos") or []):
                if v.get("url"): found.append(v["url"])
    except Exception as e:
        print(f"  （FxTwitter 失敗：{str(e)[:80]}）")
    if not found:
        try:
            j = requests.get(f"https://api.vxtwitter.com/Twitter/status/{tid}", timeout=20).json()
            for src in [j, j.get("qrt") or {}]:
                for mm in (src.get("media_extended") or []):
                    if mm.get("type") in ("video", "gif") and mm.get("url"): found.append(mm["url"])
        except Exception as e:
            print(f"  （vxTwitter 失敗：{str(e)[:80]}）")
    return list(dict.fromkeys(found))

def 下載網址影片(url):
    """先用 yt-dlp 下載；抓不到再用備用服務取得影片網址"""
    sh(f"{sys.executable} -m pip install -q -U yt-dlp")
    import yt_dlp, requests
    d = "/content/下載"
    shutil.rmtree(d, ignore_errors=True); os.makedirs(d, exist_ok=True)   # 每次清空，避免同一支影片被當成已下載
    opts = {"outtmpl": f"{d}/X_%(uploader_id)s_%(id)s_%(autonumber)s.%(ext)s",
            "format": "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/bv*+ba/b",
            "merge_output_format": "mp4", "quiet": True, "no_warnings": True}
    try:
        with yt_dlp.YoutubeDL(opts) as y:
            y.download([url])
        got = sorted(glob.glob(f"{d}/*.mp4"))
        if got:
            return got
    except Exception as e:
        print(f"  yt-dlp 抓不到（{str(e)[:120]}），改用備用方式…")
    vids = 備用取得影片網址(url)
    if not vids:
        raise RuntimeError("備用方式也找不到影片。這則推文可能沒有影片（例如是轉推、引用或圖片），或影片已被刪除。")
    out = []
    for i, v in enumerate(vids, 1):
        p = f"{d}/X_{i}.mp4"
        with requests.get(v, stream=True, timeout=60, headers={"User-Agent": "Mozilla/5.0"}) as r:
            r.raise_for_status()
            with open(p, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
        out.append(p)
    return out

def 主程式():
    import torch
    gpu = torch.cuda.is_available()
    if not gpu:
        print("⚠️ 目前沒有 GPU，會很慢。請按上方「執行階段」→「變更執行階段類型」→ 選 T4 GPU，再執行一次。")
    from google.colab import files
    paths, 原始影片 = [], []
    if 影片網址.strip():
        print("📥 從網址下載影片…")
        try:
            paths = 下載網址影片(影片網址.strip())
            原始影片 = list(paths)
            for p in paths:
                print(f"  已下載：{os.path.basename(p)}（{os.path.getsize(p)/1048576:.1f} MB）")
        except Exception as e:
            msg = str(e)
            if any(k in msg.lower() for k in ["login", "authentication", "nsfw", "sensitive", "age", "log in"]):
                print("  ⚠️ 這支影片要登入 X 才看得到（例如被標成敏感內容），工具無法直接下載。")
                print("     請先用 SaveTwitter 等網站把影片存到手機，再用下面的「選擇檔案」上傳。")
            else:
                print(f"  ⚠️ 下載失敗：{msg[:300]}")
                print("     請確認網址是推文連結（x.com/…/status/…），或改用下面的「選擇檔案」上傳。")
        if paths and 只下載不上字幕:
            for p in paths:
                files.download(p)
            if 存到雲端硬碟:
                from google.colab import drive
                drive.mount("/content/drive")
                dd = "/content/drive/MyDrive/日文影片中文字幕"; os.makedirs(dd, exist_ok=True)
                for p in paths: shutil.copy(p, dd)
                print("☁️ 已存到 Google 雲端硬碟：我的雲端硬碟 / 日文影片中文字幕")
            print("✅ 下載完成！"); return
    if not paths:
        print("\n👉 按下面的「選擇檔案」上傳影片（可多選）")
        up = files.upload()
        if not up:
            print("沒有選擇影片。"); return
        paths = [os.path.abspath(fn) for fn in up]
    if gpu:
        print("檢查 GPU 函式庫…")
        gpu = 補GPU函式庫()
    name = 模型 if gpu else ("small" if 模型 == "large-v3-turbo" else 模型)
    print(f"\n載入辨識模型 {name}（第一次會下載，約 1–3 分鐘）…")
    model = {"m": WhisperModel(name, device="cuda" if gpu else "cpu", compute_type="float16" if gpu else "int8"), "gpu": gpu}
    results = []   # 沒勾「只下載不上字幕」時，只給有字幕的版本
    for fn in paths:
        print(f"\n===== {os.path.basename(fn)} =====")
        try:
            results += 處理(fn, model)
        except Exception as e:
            print(f"❌ 處理失敗：{e}")
    if not results:
        return
    if 存到雲端硬碟:
        from google.colab import drive
        drive.mount("/content/drive")
        d = "/content/drive/MyDrive/日文影片中文字幕"; os.makedirs(d, exist_ok=True)
        for p in results: shutil.copy(p, d)
        print(f"\n☁️ 已存到 Google 雲端硬碟：我的雲端硬碟 / 日文影片中文字幕")
    print("\n✅ 完成！下載檔案：")
    for p in results:
        print("  ", os.path.basename(p))
        files.download(p)

主程式()
