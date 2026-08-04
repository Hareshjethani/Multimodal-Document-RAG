FROM python:3.11-slim

WORKDIR /app

# System dependencies (Pillow, torch etc need these)
RUN apt-get update && apt-get install -y \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY app.py .

# Render provides $PORT at runtime — app.py reads it dynamically
EXPOSE 7860

CMD ["python", "app.py"]