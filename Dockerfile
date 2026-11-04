# Container for Hugging Face Spaces (Docker SDK) or any container host.
# The LLM provider is chosen at run time by environment variables / Space secrets:
#   OPENAI_API_KEY (secret), OPENAI_BASE_URL, CHAT_MODEL
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.cache \
    EMBED_PROVIDER=local

RUN useradd -m -u 1000 user
WORKDIR /app

# CPU-only torch keeps the image small; then the project's own requirements.
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

COPY . .
RUN mkdir -p /app/.cache && chown -R user:user /app
USER user

# Build the clause-aware index once, at image build time (downloads the MiniLM model).
RUN python -m scripts.build_index

EXPOSE 7860
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "7860"]
