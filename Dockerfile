# Use a base image with Python installed
FROM python:3.10.12

# Set the working directory in the container
WORKDIR /app

# Copy all Python files and requirements.txt into the container
COPY *.py /app/
COPY requirements.txt /app/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set the entry point to the main Python file
CMD ["python", "main.py"]
