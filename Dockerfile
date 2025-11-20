FROM python:3.9-slim

WORKDIR /app

COPY backend/requirements.txt .

RUN python -m venv venv && \
    /app/venv/bin/pip install --no-cache-dir --upgrade pip && \
    /app/venv/bin/pip install --no-cache-dir -r requirements.txt

COPY backend .

EXPOSE 8000

CMD ["/app/venv/bin/python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]