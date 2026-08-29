import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.base import create_agent

def test_bedrock():
    print("Testing Bedrock connection...")
    agent = create_agent(
        system_prompt="You are a test agent. Respond with exactly one sentence."
    )
    response = agent("Say: ExperimentMind is online and ready.")
    print(f"Response: {response}")
    print("Bedrock connection OK")

if __name__ == "__main__":
    test_bedrock()