FROM python:3.9-slim

WORKDIR /app

COPY backend/requirements.txt .

RUN python -m venv venv && \
    /app/venv/bin/pip install --no-cache-dir --upgrade pip && \
    /app/venv/bin/pip install --no-cache-dir -r requirements.txt

COPY backend .

EXPOSE 8000

CMD ["/app/venv/bin/uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# Docker commands
#  sudo docker build -t data-science-life-cycle .
#  sudo docker run -p 8000:8000 data-science-life-cycle 