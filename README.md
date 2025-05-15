# About
This project demos the usage of MLFlow for tracing OpenAI Agents when using Azure OpenAI, and viewing the traces **completely locally** (without relying on https://platform.openai.com/traces or any other online service).

There're 2 files:
- withOpenAI.py - I didn't actively pursue this as my main target was Azure OpenAI
- withAzureOpenAI.py

# Initial Setup
## Packages
Use requirements.txt or requirements_frozen.txt
## Required Environmental Variables
- AZURE_OPENAI_BASE_URL
- AZURE_OPENAI_API_KEY
- AZURE_OPENAI_API_VERSION
- openai_openai_key - if you want to mess with withOpenAI.py

# Setting Up and Running the Demo
For a fresh clean start, ensure that the 3 folders azuretraces, mlartifacts and mlruns exist but are empty.

In a new terminal, run:
```bash
mlflow ui
```
This will launch the mlflow UI server locally (the IP will be shown in the terminal). Keep this terminal running.

Run withAzureOpenAI.py in another terminal. This will ask 2 questions from the agent system and the traces will be logged under 'trace1'.
The trace can then be viewed on the local mlflow UI server similar to how traces are displayed on OpenAI's trace portal.

The experiments view - you may have to switch to a different part within the UI or refresh it a bit for stuff to show up.
![Experiements View on MLFlow UI](https://github.com/rehanga937/OpenAI-Agents---View-Traces-Locally-with-MLFlow/blob/main/images/experiments-traces.png)

Viewing the trace:
![Viewing the Trace](https://github.com/rehanga937/OpenAI-Agents---View-Traces-Locally-with-MLFlow/blob/main/images/trace1.png)

