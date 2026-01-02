FROM python:3.10-slim
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt  /app/
RUN pip install -r requirements.txt
COPY app.py /app/
COPY models/* /app/models/ 
COPY templates /app/templates
CMD ["python", "app.py"]