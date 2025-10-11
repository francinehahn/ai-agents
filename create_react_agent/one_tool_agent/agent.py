import os
from dotenv import load_dotenv
from typing import List, Tuple

from langgraph.prebuilt import create_react_agent
from langchain_cohere.chat_models import ChatCohere

from sum_numbers_tool import sum_numbers_with_complex_output

load_dotenv()

class Agent:
    def invoke(self, messages: List[Tuple[str, str]]) -> str:
        llm = ChatCohere(
            cohere_api_key=os.environ.get("CO_API_KEY"), 
            model="command-a-03-2025"
        )

        add_agent = create_react_agent(
            model=llm,
            tools=[sum_numbers_with_complex_output],
            prompt="You are a helpful mathmatical assistant that can perform various operations. Use the tools precisely and explan your reasoning clearly."
        )

        response = add_agent.invoke(
            {"messages": messages}
        )
        final_answer = response["messages"][-1].content
        return final_answer