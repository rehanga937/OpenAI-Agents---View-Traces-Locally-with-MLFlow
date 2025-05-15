from dotenv import load_dotenv, find_dotenv, dotenv_values
load_dotenv(find_dotenv()) # read local .env file

import os

import mlflow
import asyncio
from agents import Agent, Runner, set_default_openai_client, OpenAIChatCompletionsModel, set_trace_processors
from openai import AsyncAzureOpenAI


# setup an Azure OpenAI client
openai_client = AsyncAzureOpenAI(
    api_key=os.getenv('AZURE_OPENAI_API_KEY'),
    api_version=os.getenv('AZURE_OPENAI_API_VERSION'),
    azure_endpoint=os.getenv('AZURE_OPENAI_BASE_URL')
)
set_default_openai_client(openai_client)

set_trace_processors([]) # Setting this to a blank array removes whatever default trace processor that tries to send all the traces to the OpenAI trace dashboard (https://platform.openai.com/traces)
mlflow.openai.autolog(disable=False, log_traces=True, silent=True) # for some reason setting trace processes to a blank array gives a warning/error (that does not affect the program), so I set silent to True

# mlflow.set_tracking_uri(uri="file:./azuretraces") # when this is blank, it will save to the default folder in the project directory. Currently it will save everything to the azuretraces folder.
mlflow.set_tracking_uri(uri="http://127.0.0.1:5000") # This is where my MLFlow UI is hosted.
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

# Run the agents system with tracing
@mlflow.trace(name="trace1")
async def main():
    result = await Runner.run(triage_agent, input="Hola, ¿cómo estás?")
    print("Answer:")
    print(result.final_output)

    result = await Runner.run(triage_agent, input="Hi how are u")
    print("Answer 2:")
    print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())

