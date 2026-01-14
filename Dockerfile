
FROM python:3.11-slim-bullseye


RUN apt-get update && apt-get install -y \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*


WORKDIR /app


COPY . .

RUN pip install --no-cache-dir -r requirements.txt


EXPOSE 5000


CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app", "--timeout", "120"]