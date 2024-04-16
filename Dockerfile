# app/Dockerfile

# Use the official Python image
FROM python:3.9-slim

# Set environment variable for setting UTF-8 encoding
ENV PYTHONUNBUFFERED=1

# Set the working directory to /app
WORKDIR /app

# Install dependencies that may be needed for your application
RUN apt-get update && apt-get install -y \
libgl1-mesa-glx \
build-essential \
cmake \
curl \
software-properties-common \
&& rm -rf /var/lib/apt/lists/*

# Copy executable files into /app/web_proctoring
COPY . .

# Install project dependencies
RUN pip3 install -r requirements.txt

# Set the port that your Streamlit application will use
EXPOSE 8501

# Run your Streamlit application when the container starts
CMD ["streamlit", "run", "web_proctoring/main.py", "--server.port=8501", "--server.address=0.0.0.0"]