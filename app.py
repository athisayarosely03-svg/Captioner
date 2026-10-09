import os, re, uuid, threading, subprocess
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory, send_file

BASE = Path(__file__).parent
UPLOADS = BASE / "uploads"
UPLOADS.mkdir(exist_ok=True)

# ---- Engine: Groq cloud (seconds, large-v3 accuracy) if a key is set, else local ----
GROQ_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL = os.getenv("GROQ_MODEL", "whisper-large-v3")
# Small hint that helps Whisper with Tamil + English mixed speech
HINT = os.getenv("WHISPER_HINT", "வணக்கம், this is a Tamil and English video. Let's start.")

LOCAL_MODEL = os.getenv("WHISPER_MODEL", "medium")
DEVICE = os.getenv("WHISPER_DEVICE", "auto")
COMPUTE = os.getenv("WHISPER_COMPUTE", "int8")

app = Flask(__name__, static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024**3
jobs = {}
_model = None
_model_lock = threading.Lock()


def get_model():
    global _model
    with _model_lock:
        if _model is None:
            from faster_whisper import WhisperModel
            _model = WhisperModel(LOCAL_MODEL, device=DEVICE, compute_type=COMPUTE,
                                  cpu_threads=os.cpu_count() or 4)
        return _model


# ---------- audio: pull a small 16 kHz mono wav out of the video ----------
def extract_audio(src, dst):
    import av
    with av.open(str(src)) as inp:
        a = next((s for s in inp.streams if s.type == "audio"), None)
        if a is None:
            raise RuntimeError("This file has no audio track.")
        with av.open(str(dst), "w", format="wav") as out:
            o = out.add_stream("pcm_s16le", rate=16000, layout="mono")
            rs = av.AudioResampler(format="s16", layout="mono", rate=16000)
            for frame in inp.decode(a):
                for f in rs.resample(frame):
                    out.mux(o.encode(f))
            for f in rs.resample(None):
                out.mux(o.encode(f))
            out.mux(o.encode(None))


# ---------- turn word timings into readable caption lines ----------
def group_words(words, max_chars=42, max_dur=4.5, gap=0.7):
    caps, cur = [], []

    def flush():
        if cur:
            text = " ".join(w["word"] for w in cur).strip()
            if text:
                caps.append({"start": round(cur[0]["start"], 2),
                             "end": round(cur[-1]["end"], 2), "text": text})
            cur.clear()

    for w in words:
        w = {"word": w["word"].strip(), "start": w["start"], "end": w["end"]}
        if not w["word"]:
            continue
        if cur:
            too_long = len(" ".join(x["word"] for x in cur)) + 1 + len(w["word"]) > max_chars
            if too_long or w["start"] - cur[-1]["end"] > gap or w["end"] - cur[0]["start"] > max_dur:
                flush()
        cur.append(w)
        if re.search(r"[.?!।]$", w["word"]):
            flush()
    flush()
    return caps


# ---------- engines ----------
def run_groq(job, wav, lang):
    import requests
    data = [("model", GROQ_MODEL), ("response_format", "verbose_json"), ("temperature", "0"),
            ("timestamp_granularities[]", "word"), ("timestamp_granularities[]", "segment")]
    if lang:
        data.append(("language", lang))
    if HINT and lang != "en":
        data.append(("prompt", HINT))
    with open(wav, "rb") as f:
        r = requests.post("https://api.groq.com/openai/v1/audio/transcriptions",
                          headers={"Authorization": f"Bearer {GROQ_KEY}"},
                          files={"file": ("audio.wav", f, "audio/wav")}, data=data, timeout=300)
    if r.status_code != 200:
        try:
            msg = r.json()["error"]["message"]
        except Exception:
            msg = r.text[:200]
        raise RuntimeError(f"Groq error {r.status_code}: {msg}")
    j = r.json()
    job["duration"] = j.get("duration", 0)
    job["language"] = j.get("language", lang or "?")
    words = [{"word": w["word"], "start": w["start"], "end": w["end"]} for w in j.get("words", [])]
    if words:
        caps = group_words(words)
    else:  # fallback: use Whisper's own segments
        caps = [{"start": round(s["start"], 2), "end": round(s["end"], 2), "text": s["text"].strip()}
                for s in j.get("segments", []) if s.get("no_speech_prob", 0) < 0.8 and s["text"].strip()]
    job["segments"].extend(caps)


def run_local(job, wav, lang):
    job["status"] = "loading"
    model = get_model()
    job["status"] = "transcribing"
    segs, info = model.transcribe(str(wav), language=lang or None, task="transcribe",
                                  vad_filter=True, beam_size=1, word_timestamps=True,
                                  condition_on_previous_text=False,
                                  initial_prompt=(HINT if lang != "en" else None))
    job["duration"], job["language"] = info.duration, info.language
    for s in segs:
        words = [{"word": w.word, "start": w.start, "end": w.end} for w in (s.words or [])]
        job["segments"].extend(group_words(words) if words else
                               [{"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()}])


def run_job(job_id, path, lang):
    job = jobs[job_id]
    try:
        job["status"] = "extracting"
        wav = Path(path).parent / "audio.wav"
        extract_audio(path, wav)
        if GROQ_KEY:
            job["status"] = "transcribing"
            run_groq(job, wav, lang)
        else:
            run_local(job, wav, lang)
        (Path(path).parent / "captions.txt").write_text(
            "\n".join(s["text"] for s in job["segments"]), encoding="utf-8")
        job["status"] = "done"
    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)


# ---------- routes ----------
@app.get("/")
def index():
    return send_from_directory("static", "index.html")


@app.get("/api/info")
def info():
    return jsonify(engine="groq" if GROQ_KEY else "local", model=GROQ_MODEL if GROQ_KEY else LOCAL_MODEL)


@app.post("/api/upload")
def upload():
    f = request.files.get("video")
    if not f:
        return jsonify(error="No video uploaded"), 400
    job_id = uuid.uuid4().hex[:12]
    folder = UPLOADS / job_id
    folder.mkdir()
    path = folder / ("input" + Path(f.filename or "v.mp4").suffix.lower())
    f.save(path)
    jobs[job_id] = {"status": "queued", "segments": [], "duration": 0, "path": str(path)}
    threading.Thread(target=run_job, args=(job_id, path, request.form.get("language", "")), daemon=True).start()
    return jsonify(job=job_id)


@app.get("/api/job/<job_id>")
def job_status(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify(error="Unknown job"), 404
    return jsonify({k: v for k, v in job.items() if k != "path"})


def srt_time(t):
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


@app.post("/api/burn")
def burn():
    data = request.get_json(force=True)
    job = jobs.get(data.get("job"))
    if not job:
        return jsonify(error="Unknown job"), 404
    folder = Path(job["path"]).parent
    srt = "\n".join(f"{i}\n{srt_time(s['start'])} --> {srt_time(s['end'])}\n{s['text']}\n"
                    for i, s in enumerate(data.get("segments", []), 1))
    (folder / "subs.srt").write_text(srt, encoding="utf-8")
    style = "Fontname=Noto Sans Tamil,Fontsize=20,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,Outline=2,MarginV=36"
    cmd = ["ffmpeg", "-y", "-i", Path(job["path"]).name, "-vf",
           f"subtitles=subs.srt:force_style='{style}'", "-c:a", "copy", "captioned.mp4"]
    try:
        r = subprocess.run(cmd, cwd=folder, capture_output=True, text=True)
    except FileNotFoundError:
        return jsonify(error="ffmpeg is not installed, so burning in is unavailable. Use the .srt instead."), 500
    if r.returncode != 0:
        return jsonify(error="ffmpeg failed: " + r.stderr[-300:]), 500
    return send_file(folder / "captioned.mp4", as_attachment=True, download_name="captioned.mp4")


if __name__ == "__main__":
    if not GROQ_KEY:  # warm the local model so the first upload doesn't wait
        threading.Thread(target=get_model, daemon=True).start()
    print("Engine:", "Groq " + GROQ_MODEL if GROQ_KEY else "local " + LOCAL_MODEL)
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
