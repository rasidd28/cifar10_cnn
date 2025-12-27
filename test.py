import boto3
from botocore.config import Config
from PIL import Image
import io

endpoint_name = "cifar10-endpoint"

config = Config(
    read_timeout=120,
    connect_timeout=30
)

runtime = boto3.client(
    "sagemaker-runtime",
    region_name="us-east-1",
    config=config
)

image_path = ""#add your test image location

# 🔹 Load and resize image BEFORE sending
img = Image.open(image_path).convert("RGB")
img = img.resize((32, 32))

buf = io.BytesIO()
img.save(buf, format="PNG")
image_bytes = buf.getvalue()

response = runtime.invoke_endpoint(
    EndpointName=endpoint_name,
    ContentType="application/octet-stream",
    Body=image_bytes
)

print(response["Body"].read().decode())

