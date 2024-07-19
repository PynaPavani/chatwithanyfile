from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_community.chat_models import ChatOllama
from langchain_community.llms import Ollama

query='what is the highest value of talach?'

agent = create_csv_agent(
    Ollama(temperature=0.7,model='mistral'),
    "./data/heart.csv",
    verbose=True,
    allow_dangerous_code=True
)
agent.run(query)