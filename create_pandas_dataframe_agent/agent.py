from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_cohere.chat_models import ChatCohere
import matplotlib.pyplot as plt
import pandas as pd
import os
from dotenv import load_dotenv


load_dotenv()

class Agent:
    def __init__(self):
        df = pd.read_csv(
            "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/ZNoKMJ9rssJn-QbJ49kOzA/student-mat.csv"
        )

        llm = ChatCohere(
            cohere_api_key=os.environ.get("CO_API_KEY"), 
            model="command-a-03-2025"
        )

        self.agent = create_pandas_dataframe_agent(
            llm=llm,
            df=df,
            verbose=False,
            return_intermediate_steps=True  # set return_intermediate_steps=True so that model could return code that it comes up with to generate the chart
        )

    def invoke(self, query: str) -> str:
        response = self.agent.invoke(query)
        final_answer = response["output"]
        return final_answer