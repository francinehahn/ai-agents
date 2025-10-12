from agent import Agent
from langchain_core.messages import HumanMessage
from tools.add_numbers_tool import add_numbers
from tools.divide_numbers_tool import divide_numbers
from tools.multiply_numbers_tool import multiply_numbers
from tools.subtract_numbers_tool import subtract_numbers

# use agent directly
agent = Agent()
chat_history = [HumanMessage(content="Divide 9, 3 and 1 and then get the result of the division and subtract it from 10")]
resp = agent.invoke(messages=chat_history)
print(f"resp: {resp}")
