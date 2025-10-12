import os
from dotenv import load_dotenv

from langchain_cohere.chat_models import ChatCohere
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.utilities.sql_database import SQLDatabase
from langchain.agents import AgentType

load_dotenv()

class Agent:
    def __init__(self):
        db = SQLDatabase.from_uri(os.environ.get("DB_URI"))

        llm = ChatCohere(
            cohere_api_key=os.environ.get("CO_API_KEY"), 
            model="command-a-03-2025",
            temperature=0
        )

        self.agent = create_sql_agent(
            llm=llm, 
            db=db, 
            verbose=True, 
            handle_parsing_errors=True, 
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION
        )

    def invoke(self, query: str) -> str:
        response = self.agent.invoke(query)
        print(f"llm response: {response}")
        return response