FROM python:3.10-slim

WORKDIR /app

# system deps for pillow, etc.
RUN apt-get update && apt-get install -y build-essential libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

COPY dist ./dist
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
