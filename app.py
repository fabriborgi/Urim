"""
CleanWave Backend - YouTube audio downloader with profanity muting.
Uses yt-dlp for downloading, faster-whisper for transcription, and pydub for audio processing.

faster-whisper uses CTranslate2 instead of PyTorch, reducing memory usage from ~1.5GB to ~300MB.
"""

import os
import uuid
import tempfile
import logging
import re
import threading
import gc

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import yt_dlp
from faster_whisper import WhisperModel
from pydub import AudioSegment

from profanity_words import SINGLE_WORD_PROFANITY, MULTI_WORD_PROFANITY

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TEMP_DIR = tempfile.gettempdir()
WHISPER_MODEL_SIZE = os.environ.get("WHISPER_MODEL", "tiny")

# ---------------------------------------------------------------------------
# Lazy-load Whisper model in background (so gunicorn can bind port immediately)
# ---------------------------------------------------------------------------
_model = None
_model_lock = threading.Lock()
_model_ready = threading.Event()


def _load_model_background():
    """Load faster-whisper model in a background thread."""
    global _model
    try:
        logger.info(f"[bg] Loading faster-whisper model '{WHISPER_MODEL_SIZE}' ...")
        _model = WhisperModel(
            WHISPER_MODEL_SIZE,
            device="cpu",
            compute_type="int8",      # int8 quantization = less RAM
            cpu_threads=2,
            num_workers=1,
        )
        logger.info(f"[bg] Model '{WHISPER_MODEL_SIZE}' loaded successfully.")
    except Exception as e:
        logger.error(f"[bg] Failed to load model: {e}", exc_info=True)
    finally:
        _model_ready.set()


# Start loading immediately in background
threading.Thread(target=_load_model_background, daemon=True).start()


def get_model():
    """Get the Whisper model, waiting for it to load if necessary."""
    _model_ready.wait(timeout=300)
    if _model is None:
        raise RuntimeError("Whisper model failed to load. Server may be low on memory.")
    return _model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def validate_youtube_url(url: str) -> bool:
    """Check that the URL looks like a valid YouTube link."""
    patterns = [
        r'(https?://)?(www\.)?youtube\.com/watch\?v=[\w-]+',
        r'(https?://)?(www\.)?youtu\.be/[\w-]+',
        r'(https?://)?(www\.)?youtube\.com/shorts/[\w-]+',
        r'(https?://)?music\.youtube\.com/watch\?v=[\w-]+',
        r'(https?://)?(www\.)?youtube\.com/embed/[\w-]+',
    ]
    return any(re.match(p, url.strip()) for p in patterns)


def extract_video_id(url: str) -> str:
    """Extract just the video ID from various YouTube URL formats."""
    patterns = [
        r'(?:https?://)?(?:www\.)?youtube\.com/watch\?v=([\w-]+)',
        r'(?:https?://)?(?:www\.)?youtu\.be/([\w-]+)',
        r'(?:https?://)?(?:www\.)?youtube\.com/shorts/([\w-]+)',
        r'(?:https?://)?music\.youtube\.com/watch\?v=([\w-]+)',
        r'(?:https?://)?(?:www\.)?youtube\.com/embed/([\w-]+)',
    ]
    for p in patterns:
        m = re.search(p, url.strip())
        if m:
            return m.group(1)
    return None


def download_audio(url: str, output_path: str):
    """Download audio from YouTube in WAV format using yt-dlp."""
    video_id = extract_video_id(url)
    if not video_id:
        raise ValueError("Could not extract video ID from URL")

    clean_url = f"https://www.youtube.com/watch?v={video_id}"
    logger.info(f"Downloading from: {clean_url}")

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_path,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '128',
        }],
        'quiet': True,
        'no_warnings': True,
        'noplaylist': True,
        'socket_timeout': 30,
        'retries': 3,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(clean_url, download=True)
        title = info.get('title', 'audio')

    # yt-dlp appends .wav to the output path
    wav_path = output_path + '.wav'
    if not os.path.exists(wav_path):
        wav_path = output_path
    if not os.path.exists(wav_path):
        raise FileNotFoundError("Downloaded file not found")

    return wav_path, title


def transcribe_audio(audio_path: str) -> list:
    """
    Transcribe audio using faster-whisper and return word-level timestamps.
    Returns a list of dicts: [{"word": "...", "start": float, "end": float}, ...]
    """
    model = get_model()
    segments, info = model.transcribe(
        audio_path,
        word_timestamps=True,
        language=None,          # auto-detect
        beam_size=1,            # beam_size=1 = greedy, faster and less RAM
        best_of=1,
        vad_filter=True,        # skip silence = faster
        vad_parameters=dict(
            min_silence_duration_ms=500,
        ),
    )

    logger.info(f"Detected language: {info.language} (prob={info.language_probability:.2f})")

    words = []
    for segment in segments:
        if segment.words:
            for w in segment.words:
                words.append({
                    "word": w.word.strip(),
                    "start": w.start,
                    "end": w.end,
                })

    return words


def find_profanity(words: list) -> list:
    """
    Given word-level output, identify profanity timestamps.
    Returns list of (start_ms, end_ms) tuples for sections to mute.
    """
    mute_ranges = []

    for i, w in enumerate(words):
        clean = re.sub(r'[^\w\s]', '', w["word"].lower().strip())
        if clean in SINGLE_WORD_PROFANITY:
            start_ms = max(0, int(w["start"] * 1000) - 50)
            end_ms = int(w["end"] * 1000) + 50
            mute_ranges.append((start_ms, end_ms))

    # Check multi-word phrases
    for phrase in MULTI_WORD_PROFANITY:
        phrase_words = phrase.split()
        phrase_len = len(phrase_words)
        for i in range(len(words) - phrase_len + 1):
            window = [
                re.sub(r'[^\w\s]', '', words[i + j]["word"].lower().strip())
                for j in range(phrase_len)
            ]
            if window == phrase_words:
                start_ms = max(0, int(words[i]["start"] * 1000) - 50)
                end_ms = int(words[i + phrase_len - 1]["end"] * 1000) + 50
                mute_ranges.append((start_ms, end_ms))

    if not mute_ranges:
        return []

    # Merge overlapping ranges
    mute_ranges.sort()
    merged = [mute_ranges[0]]
    for start, end in mute_ranges[1:]:
        if start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

    return merged


def mute_sections(audio_path: str, mute_ranges: list, output_format: str) -> str:
    """Mute the specified time ranges in the audio file."""
    audio = AudioSegment.from_wav(audio_path)

    for start_ms, end_ms in mute_ranges:
        duration = end_ms - start_ms
        silence = AudioSegment.silent(duration=duration)
        audio = audio[:start_ms] + silence + audio[end_ms:]

    output_path = audio_path.rsplit('.', 1)[0] + f'_clean.{output_format}'

    if output_format == 'mp3':
        audio.export(output_path, format='mp3', bitrate='192k')
    else:
        audio.export(output_path, format='wav')

    return output_path


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route('/api/health', methods=['GET'])
def health():
    """Health check — returns immediately even if model is still loading."""
    model_status = "ready" if _model is not None else "loading"
    return jsonify({"status": "ok", "model": model_status})


@app.route('/api/process', methods=['POST'])
def process_video():
    """
    Main endpoint.
    Expects JSON: { "url": "https://youtube.com/...", "format": "mp3" | "wav" }
    Returns the processed audio file.
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON body provided"}), 400

    url = data.get("url", "").strip()
    output_format = data.get("format", "mp3").lower()

    if not url:
        return jsonify({"error": "URL is required"}), 400

    if not validate_youtube_url(url):
        return jsonify({"error": "Invalid YouTube URL"}), 400

    if output_format not in ("mp3", "wav"):
        return jsonify({"error": "Format must be 'mp3' or 'wav'"}), 400

    job_id = str(uuid.uuid4())[:8]
    raw_path = os.path.join(TEMP_DIR, f"cw_{job_id}")

    try:
        # Step 1: Download
        logger.info(f"[{job_id}] Downloading audio ...")
        wav_path, title = download_audio(url, raw_path)
        logger.info(f"[{job_id}] Downloaded: {title}")

        # Step 2: Transcribe
        logger.info(f"[{job_id}] Transcribing ...")
        words = transcribe_audio(wav_path)
        logger.info(f"[{job_id}] Transcription complete: {len(words)} words")

        # Step 3: Detect profanity
        mute_ranges = find_profanity(words)
        logger.info(f"[{job_id}] Found {len(mute_ranges)} profanity sections to mute")

        # Step 4: Mute and export
        if mute_ranges:
            logger.info(f"[{job_id}] Muting profanity ...")
            output_path = mute_sections(wav_path, mute_ranges, output_format)
        else:
            logger.info(f"[{job_id}] No profanity detected, converting format ...")
            audio = AudioSegment.from_wav(wav_path)
            output_path = wav_path.rsplit('.', 1)[0] + f'_clean.{output_format}'
            if output_format == 'mp3':
                audio.export(output_path, format='mp3', bitrate='192k')
            else:
                audio.export(output_path, format='wav')

        safe_title = re.sub(r'[^\w\s-]', '', title)[:60].strip()
        download_name = f"{safe_title} (Clean).{output_format}"

        logger.info(f"[{job_id}] Done. Sending file.")

        # Force garbage collection to free memory before sending
        gc.collect()

        return send_file(
            output_path,
            as_attachment=True,
            download_name=download_name,
            mimetype='audio/mpeg' if output_format == 'mp3' else 'audio/wav',
        )

    except Exception as e:
        logger.error(f"[{job_id}] Error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

    finally:
        # Cleanup temp files to free disk/memory
        for suffix in ['', '.wav', '_clean.mp3', '_clean.wav']:
            p = raw_path + suffix
            if os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass
        gc.collect()


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
