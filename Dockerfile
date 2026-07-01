FROM python:3.10-slim

WORKDIR /app

# Install system dependencies required by LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy directories over to preserve layout
COPY data/ ./data/
COPY models/ ./models/
COPY app/ ./app/

EXPOSE 8000

ENV PYTHONPATH=/app

CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8000"]