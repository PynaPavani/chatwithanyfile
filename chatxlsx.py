from langchain.agents.agent_types import AgentType
from langchain_community.chat_models import ChatOllama
from langchain_community.llms import Ollama
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
import pandas as pd

query='what are the columns present?'


# Define a function to create Pandas DataFrame agent from a CSV file.
def create_pd_agent(query,file):
    # Initiate a connection to the LLM from Azure OpenAI Service via LangChain.
    llm = Ollama(temperature=0.7,model='llama3')

    # Create a Pandas DataFrame agent from the CSV file.
    agent=create_pandas_dataframe_agent(llm, file, verbose=True,allow_dangerous_code=True,agent_executor_kwargs={"handle_parsing_errors": True})
    response=agent.run(query)
    return response


file=pd.read_excel('./data/Iris.xlsx')

if __name__=='__main__':
    response=create_pd_agent(query=query,file=file)

