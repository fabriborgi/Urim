"""
CleanWave Backend - YouTube audio downloader with profanity muting.
Uses yt-dlp for downloading, Whisper for transcription, and pydub for audio processing.
"""

import os
import uuid
import tempfile
import logging
import re
from pathlib import Path

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import yt_dlp
import whisper
from pydub import AudioSegment

from profanity_words import SINGLE_WORD_PROFANITY, MULTI_WORD_PROFANITY

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TEMP_DIR = tempfile.gettempdir()
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "base")

# Load Whisper model once at startup
logger.info(f"Loading Whisper model '{WHISPER_MODEL}' ...")
model = whisper.load_model(WHISPER_MODEL)
logger.info("Whisper model loaded.")

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
    ]
    return any(re.match(p, url.strip()) for p in patterns)


def download_audio(url: str, output_path: str) -> str:
    """Download audio from YouTube in WAV format using yt-dlp."""
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_path,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'quiet': True,
        'no_warnings': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        title = info.get('title', 'audio')

    # yt-dlp may append .wav
    wav_path = output_path + '.wav' if not output_path.endswith('.wav') else output_path
    if not os.path.exists(wav_path):
        # try without extension change
        wav_path = output_path
    return wav_path, title


def transcribe_audio(audio_path: str) -> list:
    """
    Transcribe audio using Whisper and return word-level timestamps.
    Returns a list of dicts: [{"word": "...", "start": float, "end": float}, ...]
    """
    result = model.transcribe(
        audio_path,
        word_timestamps=True,
        language=None,  # auto-detect
    )

    words = []
    for segment in result.get("segments", []):
        for w in segment.get("words", []):
            words.append({
                "word": w["word"].strip(),
                "start": w["start"],
                "end": w["end"],
            })
    return words


def find_profanity(words: list) -> list:
    """
    Given Whisper word-level output, identify profanity timestamps.
    Returns list of (start_ms, end_ms) tuples for sections to mute.
    """
    mute_ranges = []

    for i, w in enumerate(words):
        clean = re.sub(r'[^\w\s]', '', w["word"].lower().strip())
        if clean in SINGLE_WORD_PROFANITY:
            # Add a small buffer (50ms) around the word for clean muting
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

    # Merge overlapping ranges
    if not mute_ranges:
        return []

    mute_ranges.sort()
    merged = [mute_ranges[0]]
    for start, end in mute_ranges[1:]:
        if start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

    return merged


def mute_sections(audio_path: str, mute_ranges: list, output_format: str) -> str:
    """
    Mute the specified time ranges in the audio file.
    Returns path to the processed file.
    """
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
    return jsonify({"status": "ok"})


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
            # No profanity found — just convert to desired format
            logger.info(f"[{job_id}] No profanity detected, converting format ...")
            audio = AudioSegment.from_wav(wav_path)
            output_path = wav_path.rsplit('.', 1)[0] + f'_clean.{output_format}'
            if output_format == 'mp3':
                audio.export(output_path, format='mp3', bitrate='192k')
            else:
                audio.export(output_path, format='wav')

        # Sanitize filename
        safe_title = re.sub(r'[^\w\s-]', '', title)[:60].strip()
        download_name = f"{safe_title} (Clean).{output_format}"

        logger.info(f"[{job_id}] Done. Sending file.")
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
        # Cleanup temp files
        for suffix in ['', '.wav', '_clean.mp3', '_clean.wav']:
            p = raw_path + suffix
            if os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
