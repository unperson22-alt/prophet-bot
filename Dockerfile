FROM python:3.11-slim

# git нужен для pip install git+https://... (ai-office-shared)
RUN apt-get update && apt-get install -y --no-install-recommends git     && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "bot.py"]
