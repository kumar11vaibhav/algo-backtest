# Build stage
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    python3-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY config/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Final stage
FROM python:3.11-slim as paper_trader

WORKDIR /app

# Copy from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy source code
COPY src/ ./src/

# Create necessary directories
RUN mkdir -p logs data paper_trade_logs

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TZ=Asia/Kolkata
ENV PYTHONPATH=/app

# Run the paper trading script
WORKDIR /app/src
CMD ["python", "run_paper_trading.py"]
