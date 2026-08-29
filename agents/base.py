import os
from dotenv import load_dotenv
from strands import Agent
from strands.models.ollama import OllamaModel

load_dotenv()

def get_model():
    return OllamaModel(
        model_id="qwen2.5:3b-instruct",
        host="http://localhost:11434",
    )

def create_agent(system_prompt: str, tools: list = None) -> Agent:
    return Agent(
        model=get_model(),
        system_prompt=system_prompt,
        tools=tools or [],
    )