import boto3
from dotenv import load_dotenv
load_dotenv()

client = boto3.client("bedrock-runtime", region_name="us-east-1")

try:
    response = client.converse(
        modelId="mistral.mistral-small-2402-v1:0",
        messages=[{"role": "user", "content": [{"text": "Hi"}]}]
    )
    print("SUCCESS:", response["output"]["message"]["content"][0]["text"])
except Exception as e:
    print("ERROR:", type(e).__name__, str(e)[:200])