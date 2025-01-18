# Use the official Python 3.12 base image
FROM python:3.12-slim

# Set environment variables to prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libssl-dev \
    libffi-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install pip-tools for managing lock files
RUN pip install --no-cache-dir pip-tools

# Copy the lock file and install dependencies
COPY requirements.lock /app/requirements.lock

# Install dependencies from the lock file
RUN pip install --no-cache-dir -r requirements.lock

# Copy the application code into the container
COPY . /app

# Set the default command to run your application, allowing it to be overwritten
CMD ["python", "-m", "main"]
