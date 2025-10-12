import os
from dotenv import load_dotenv

from langchain_cohere.chat_models import ChatCohere
from langchain_core.messages import HumanMessage, ToolMessage

from tools.add_numbers_tool import add_numbers
from tools.divide_numbers_tool import divide_numbers
from tools.multiply_numbers_tool import multiply_numbers
from tools.subtract_numbers_tool import subtract_numbers

load_dotenv()

class Agent:
    def __init__(self):
        llm = ChatCohere(
            cohere_api_key=os.environ.get("CO_API_KEY"), 
            model="command-a-03-2025"
        )

        self.agent = llm.bind_tools(tools=[add_numbers, subtract_numbers, divide_numbers, multiply_numbers])

        self.tool_map = {
            "add_numbers": add_numbers, 
            "subtract_numbers": subtract_numbers,
            "multiply_numbers": multiply_numbers,
            "divide_numbers": divide_numbers
        }

    def invoke(self, query: str) -> str:
        # Step 1: Initial user message
        chat_history = [HumanMessage(content=query)]
        print(f"chat history: {chat_history}\n\n")

        while True:
            # Step 2: LLM chooses tool
            response = self.agent.invoke(chat_history)
            print(f"llm response: {response}")

            if not response.tool_calls:
                return response.content # Direct response, no tool needed
            
            # Step 3: Handle tool call
            tool_name = response.tool_calls[0]["name"]
            tool_args = response.tool_calls[0]["args"]
            tool_call_id = response.tool_calls[0]["id"]

            # Step 4: Call tool manually
            tool_result = self.tool_map[tool_name].invoke(tool_args)

            # Step 5: Send result back to LLM
            tool_message = ToolMessage(content=str(tool_result), tool_call_id=tool_call_id)
            chat_history.extend([response, tool_message])
