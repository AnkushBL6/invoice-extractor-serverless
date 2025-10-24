FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel

WORKDIR /app

# Clean and install in one layer to save space
RUN pip install --no-cache-dir \
    runpod \
    transformers==4.55.4 \
    accelerate \
    peft \
    bitsandbytes \
    pillow \
    safetensors \
    sentencepiece && \
    pip install --no-cache-dir unsloth && \
    pip cache purge && \
    rm -rf /root/.cache/pip

COPY handler.py .

ENV PYTHONUNBUFFERED=1

CMD ["python", "-u", "handler.py"]
