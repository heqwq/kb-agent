FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONUTF8=1

WORKDIR /app
COPY . .

RUN python scripts/make_analytics_db.py \
    && python -m unittest discover -s tests \
    && python -m kb_agent.cli ingest \
    && python -m kb_agent.cli eval \
    && python -m kb_agent.cli ask "一线城市出差住宿标准每晚多少钱？"

CMD ["python", "-m", "kb_agent.cli", "eval"]
