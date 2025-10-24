from agent import ReflectionAgent
from langchain.schema import HumanMessage

inputs = {"messages": [HumanMessage(content="Write a LinkedIn post on getting a software developer job at IBM under 160 characters")]}

resp = ReflectionAgent().call(inputs=inputs)
print(resp.get("messages")[-1])