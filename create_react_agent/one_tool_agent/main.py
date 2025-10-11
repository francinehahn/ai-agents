from agent import Agent

agent = Agent()

messages = [("human", "Add the numbers -10, -20, -30")]

response = agent.invoke(messages=messages)

print(response)