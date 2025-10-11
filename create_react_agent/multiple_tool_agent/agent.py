import os
from dotenv import load_dotenv
from typing import List, Tuple

from langgraph.prebuilt import create_react_agent
from langchain_cohere.chat_models import ChatCohere

from tools.add_numbers_tool import add_numbers
from tools.divide_numbers_tool import divide_numbers
from tools.multiply_numbers_tool import multiply_numbers
from tools.subtract_numbers_tool import subtract_numbers

load_dotenv()

class Agent:
    def get_agent(self):
        llm = ChatCohere(
            cohere_api_key=os.environ.get("CO_API_KEY"), 
            model="command-a-03-2025"
        )

        add_agent = create_react_agent(
            model=llm,
            tools=[add_numbers, subtract_numbers, multiply_numbers, divide_numbers],
            prompt="You are a helpful mathmatical assistant that can perform various operations. Use the tools precisely and explan your reasoning clearly."
        )
        return add_agent

    def invoke(self, messages: List[Tuple[str, str]]) -> str:
        agent = self.get_agent()

        response = agent.invoke(
            {"messages": messages}
        )
        final_answer = response["messages"][-1].content
        return final_answer