# Use Python as the base image
FROM python:3.11-slim

# Install dependencies
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Copy the source code and requirements files
COPY src src/
COPY requirements.txt requirements.txt
COPY requirements_dev.txt requirements_dev.txt
COPY README.md README.md
COPY pyproject.toml pyproject.toml

# Install Python dependencies with pip cache enabled
RUN --mount=type=cache,target=/root/.cache/pip pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install -r requirements_dev.txt

# Set the working directory
WORKDIR /src/sven_project_day_2

# Set PYTHONPATH to include the src directory
ENV PYTHONPATH=/src

# Define the command to run the training script
CMD ["python", "train.py"]
