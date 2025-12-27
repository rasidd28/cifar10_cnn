# ------------------------------------------------------------
# SageMaker PyTorch Training Container
# ------------------------------------------------------------

# 1. Base image (PyTorch + CUDA support)
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

# 2. Set working directory inside container
WORKDIR /opt/ml/code

# 3. Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy training script into container
COPY train.py .

# 5. Environment variables required by SageMaker
#ENV SAGEMAKER_PROGRAM train.py
#ENV SAGEMAKER_SUBMIT_DIRECTORY /opt/ml/code

# 6. Default command (SageMaker overrides this)
ENTRYPOINT ["python", "train.py"]
