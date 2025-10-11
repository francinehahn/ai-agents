from agent import Agent
from langchain.agents import AgentExecutor
from langchain_core.messages import HumanMessage
from tools.add_numbers_tool import add_numbers
from tools.divide_numbers_tool import divide_numbers
from tools.multiply_numbers_tool import multiply_numbers
from tools.subtract_numbers_tool import subtract_numbers

# use agent directly
agent = Agent()
messages = [("human", "Divide 9, 3 and 1 and then get the result of the division and subtract it from 10")]
resp = agent.invoke(messages=messages)
print(f"resp 1: {resp}")

agent = Agent()
tools = [add_numbers, divide_numbers, multiply_numbers, subtract_numbers]
agent_executor = AgentExecutor(agent=agent.get_agent(), tools=tools, verbose=True, handle_parsing_errors=True)
agent_executor.agent.stream_runnable = False
result=agent_executor.invoke(input={"messages": messages})
print(f"resp 2: {result}")