# 1. Use a lightweight version of Python 3.9
FROM python:3.9-slim



# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy all files from your computer to the container
COPY . .

# 4. Install the Python libraries you listed in requirements.txt
# --no-cache-dir keeps the file size small
RUN pip install --no-cache-dir -r requirements.txt

# 5. Expose the port that Flask will run on
EXPOSE 5000

# 5. Start the application
# "gunicorn" is a production server (faster than standard Flask)
# "app:app" means: look in file 'app.py' for the variable 'app'
# "--timeout 120" gives your model 2 minutes to load before giving up (crucial for AI)
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app", "--timeout", "120"]