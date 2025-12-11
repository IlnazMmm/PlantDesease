FROM python:3.10-slim

WORKDIR /app

COPY requirements.migrations.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

ENV PYTHONUNBUFFERED=1

CMD ["python", "-m", "app.migrate"]
