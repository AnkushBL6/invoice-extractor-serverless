FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

# Install in specific order to avoid conflicts
RUN pip install --no-cache-dir packaging wheel

RUN pip install --no-cache-dir \
    runpod \
    transformers==4.55.4 \
    peft \
    accelerate \
    pillow \
    safetensors \
    sentencepiece

# Install unsloth last (it has specific dependencies)
RUN pip install --no-cache-dir unsloth

# Install bitsandbytes separately
RUN pip install --no-cache-dir bitsandbytes

COPY handler.py .

ENV PYTHONUNBUFFERED=1

CMD ["python", "-u", "handler.py"]
