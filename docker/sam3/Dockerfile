FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y git ffmpeg libgl1 \
 && rm -rf /var/lib/apt/lists/*

COPY docker/sam3/requirements-sam3.txt .
RUN pip install --no-cache-dir -r docker/sam3/requirements-sam3.txt

COPY . .
RUN pip install --no-cache-dir -e .

EXPOSE 8001
CMD ["uvicorn", "carpet_search.sam3_server:app", "--host", "0.0.0.0", "--port", "8001"]
