FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

WORKDIR /app

# Install Python
RUN apt-get update && apt-get install -y python3.11 python3-pip git

# Install PyTorch first
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other deps
RUN pip3 install --no-cache-dir \
    runpod \
    transformers==4.55.4 \
    accelerate \
    peft \
    pillow \
    safetensors \
    sentencepiece \
    bitsandbytes

# Install unsloth
RUN pip3 install unsloth

COPY handler.py .

ENV PYTHONUNBUFFERED=1

CMD ["python3", "-u", "handler.py"]
