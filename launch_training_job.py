import sagemaker
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput

# ---------- BASIC SETUP ----------
region = "us-east-1"
role = sagemaker.get_execution_role()
session = sagemaker.Session()

# ---------- ECR IMAGE ----------
image_uri = "884040630640.dkr.ecr.us-east-1.amazonaws.com/cifar10-sagemaker:latest"

# ---------- S3 DATA ----------
dataset_s3_uri = "s3://rahul-cifar10-sagemaker-2025/dataset.zip"

# ---------- ESTIMATOR ----------
estimator = Estimator(
    image_uri=image_uri,
    role=role,
    instance_count=1,
    instance_type="ml.g4dn.xlarge",  # GPU (recommended)
    volume_size=50,
    max_run=4 * 60 * 60,  # 4 hours
    input_mode="File",
    output_path=f"s3://rahul-cifar10-sagemaker-2025/output/",
    sagemaker_session=session
)

# ---------- TRAINING INPUT ----------
train_input = TrainingInput(
    s3_data=dataset_s3_uri,
    destination="/opt/ml/input/data/dataset"
)

# ---------- START TRAINING ----------
estimator.fit({"dataset": train_input})

print("🚀 Training job started successfully!")
