from agent import Agent

agent = Agent()
query = "Divide 9, 3 and 1 and then get the result of the division and subtract it from 10"
resp = agent.invoke(query=query)
print(resp)