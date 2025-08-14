# Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Copy requirements.txt before other files for better caching
COPY requirements.txt .

RUN python -m pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
COPY . .

EXPOSE 80

# Run the application using uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]
