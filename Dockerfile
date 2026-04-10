FROM python:3.11-slim

# Install ffmpeg (required by pydub and yt-dlp)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Expose port
EXPOSE 5000

# Run with gunicorn
# --timeout 300  : allow long audio processing
# --workers 1    : single worker to minimize memory
# --preload      : share model across forks (not needed with 1 worker, but good practice)
CMD gunicorn app:app \
    --bind 0.0.0.0:${PORT:-5000} \
    --timeout 300 \
    --workers 1 \
    --threads 2
