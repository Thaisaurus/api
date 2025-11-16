FROM python:alpine

RUN apk add -U uv
RUN uv pip install --system nltk

WORKDIR /app
RUN python -c "$(printf "import nltk\nnltk.download('vader_lexicon')\nnltk.download('wordnet')")"

RUN uv pip install --system requests numpy qdrant-client fastapi uvicorn httpx[http2] async-lru
COPY ./api.py /app/api.py

CMD ["python3", "/app/api.py"]

