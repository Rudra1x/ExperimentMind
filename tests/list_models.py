import boto3
from dotenv import load_dotenv
load_dotenv()

client = boto3.client('bedrock', region_name='us-east-1')
models = client.list_foundation_models()

for m in models['modelSummaries']:
    if ('ON_DEMAND' in m.get('inferenceTypesSupported', []) and
        'TEXT' in m.get('outputModalities', [])):
        print(f"{m['providerName']:15} | {m['modelId']}")