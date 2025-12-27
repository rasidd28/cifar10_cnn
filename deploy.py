from sagemaker.pytorch import PyTorchModel
import sagemaker

role = sagemaker.get_execution_role()

model = PyTorchModel(
    model_data="s3://rahul-cifar10-sagemaker-2025/output/cifar10-sagemaker-2025-12-25-10-59-17-079/output/model.tar.gz",
    role=role,
    framework_version="2.0",
    py_version="py310",
    entry_point="inference.py",
    source_dir="."
)

predictor = model.deploy(
    instance_type="ml.m5.large",
    initial_instance_count=1,
    endpoint_name="cifar10-endpoint"
)

print("DONE")
