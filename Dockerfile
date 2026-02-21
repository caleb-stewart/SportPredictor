# syntax=docker/dockerfile:1

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update -qq && \
    apt-get install --no-install-recommends -y curl postgresql-client && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app
RUN chmod +x /app/bin/docker-entrypoint

EXPOSE 3141

ENTRYPOINT ["/app/bin/docker-entrypoint"]
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "3141"]
