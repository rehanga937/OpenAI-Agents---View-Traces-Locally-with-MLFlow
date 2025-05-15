from dotenv import load_dotenv, find_dotenv, dotenv_values
load_dotenv(find_dotenv()) # read local .env file

import os

import mlflow
import asyncio
from agents import Agent, Runner, set_default_openai_client, OpenAIChatCompletionsModel
from openai import AsyncOpenAI


# artifact folders
os.makedirs("azuretraces", exist_ok=True)
os.makedirs("mlartifacts", exist_ok=True)
os.makedirs("mlruns", exist_ok=True)

openai_client = AsyncOpenAI(api_key=os.getenv("openai_openai_key"))
set_default_openai_client(openai_client)

# Enable auto tracing for OpenAI Agents SDK
mlflow.openai.autolog()
mlflow.set_tracking_uri(uri="") # when this is blank, it will save to the default folder in the project directory
mlflow.set_experiment("OpenAI Agent")


# Define a simple multi-agent workflow
spanish_agent = Agent(
    name="Spanish agent",
    instructions="You only speak Spanish.",
    model=OpenAIChatCompletionsModel(
        model="gpt-4o-mini",
        openai_client=openai_client
    )
)
english_agent = Agent(
    name="English agent",
    instructions="You only speak English",
    model=OpenAIChatCompletionsModel(
        model="gpt-4o-mini",
        openai_client=openai_client
    )
)
triage_agent = Agent(
    name="Triage agent",
    instructions="Handoff to the appropriate agent based on the language of the request.",
    handoffs=[spanish_agent, english_agent],
    model=OpenAIChatCompletionsModel(
        model="gpt-4o-mini",
        openai_client=openai_client
    )
)


async def main():
    result = await Runner.run(triage_agent, input="Hola, ¿cómo estás?")
    print("Answer:")
    print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())