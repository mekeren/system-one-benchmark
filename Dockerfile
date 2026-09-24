FROM python:3.11-slim

WORKDIR /app

# Sistem gereksinimleri (build araçları ve curl)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

EXPOSE 8001 8002 8003 8150

CMD ["python", "services/ui_gateway.py"]
