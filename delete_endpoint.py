import boto3

sm = boto3.client("sagemaker", region_name="us-east-1")

# delete endpoint (if exists)
try:
    sm.delete_endpoint(EndpointName="cifar10-endpoint")
except:
    pass

# delete endpoint config
try:
    sm.delete_endpoint_config(EndpointConfigName="cifar10-endpoint")
except:
    pass

print("OLD ENDPOINT REMOVED")
