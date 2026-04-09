FROM python:3.11-slim

# Install ffmpeg
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

# Run with gunicorn (timeout 300s for long processing)
CMD gunicorn app:app --bind 0.0.0.0:${PORT:-5000} --timeout 300 --workers 1
