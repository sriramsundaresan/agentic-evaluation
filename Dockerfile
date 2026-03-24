FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

# Azure App Service sets PORT env var; app.py reads it
ENV PORT=8000
CMD ["python", "app.py"]
