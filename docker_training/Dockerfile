# Python runtime as a parent image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app  

# Set the working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the src directory
COPY ../src /app/src

# Copy the pipeline_mlflow.py script from src/pipeline
COPY ../src/model/train_mlflow.py /app/src/model/train_mlflow.py

# Copy the data_files directory
COPY ../data_files /app/data_files

# Set environment variables for training
ENV MODEL_NAME=ifs_Random_Forest
ENV LOCAL_DATA_PATH=/app/data_files/
ENV MODEL_OUTPUT_PATH=/app/models/
ENV MLFLOW_TRACKING_URI=http://mlflow:5000

# run the training script
CMD ["python", "/app/src/model/train_mlflow.py"]