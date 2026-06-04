FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV ASR_BACKEND=mock
ENV TTS_BACKEND=mock
ENV LLM_BACKEND=rules

EXPOSE 7860

CMD ["python", "frontend/gradio_app.py"]
