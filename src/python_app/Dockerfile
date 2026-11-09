# Python container for the Titanic logistic regression model
FROM python:3.11-slim

# Keep Python output unbuffered so every print shows up in the terminal immediately
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DATA_DIR=/app/data

WORKDIR /app

# Install pinned dependencies first (cached layer)
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the script
COPY titanic.py .

# The data is NOT baked into the image; it is mounted at /app/data at run time
CMD ["python", "titanic.py"]
