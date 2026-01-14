# 1. Use Python 3.11 (Bullseye is stable)
FROM python:3.11-slim-bullseye

# 2. Install system dependencies
# "git" fixes Exit 128
# "libgl..." fixes the visual library errors
RUN apt-get update && apt-get install -y \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 3. Set the working directory inside the container
WORKDIR /app

# 4. Copy all files from your computer to the container
COPY . .

# 5. Install the Python libraries
RUN pip install --no-cache-dir -r requirements.txt

# 6. Expose the port
EXPOSE 5000

# 7. Start the application
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app", "--timeout", "120"]