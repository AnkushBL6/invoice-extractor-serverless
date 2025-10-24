FROM unsloth/unsloth:latest

WORKDIR /app

# Only install runpod (unsloth already has everything else)
RUN pip install --no-cache-dir runpod

COPY handler.py .

CMD ["python", "-u", "handler.py"]
