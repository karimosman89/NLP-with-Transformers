# MLflow Dockerfile
FROM python:3.9-slim

# Set environment variables to prevent Python from writing .pyc files and buffering output
ENV PYTHONUNBUFFERED=1



# Create a directory for MLflow to store data
RUN mkdir -p /mlflow

WORKDIR /mlflow
# Copy the requirements file
COPY requirements.txt .  

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt
# Expose the port MLflow will run on
EXPOSE 5000

# Command to run MLflow
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"]
