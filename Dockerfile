# FROM python:3.11-slim
# WORKDIR /app
# COPY requirements.txt /app/
# ENV PYTHONUNBUFFERED=1
# # minimal deps needed by your code
# RUN pip install --no-cache-dir -r requirements.txt
# COPY *.py /app/
# COPY GasProperties.csv /app/
# CMD ["python", "evaluate_models.py"]
FROM python:3.11-slim

# System libs sometimes needed by torch/plotting
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxext6 libxrender1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY *.py /app/
COPY GasProperties.csv /app/
COPY requirements.txt /app/

# IMPORTANT: requirements must point to CUDA wheels, not CPU
RUN pip install --no-cache-dir -r requirements.txt

ENV PYTHONUNBUFFERED=1
CMD ["python", "/app/evaluate_models.py"]
