from agent import Agent

agent = Agent()

response = agent.invoke(query="How many rows of data are in this file?")
print(f"How many rows of data are in this file? {response}")

response = agent.invoke("Give me all the data where student's age is over 18 years old.")
print(f"Give me all the data where student's age is over 18 years old.\n{response}")

response = agent.invoke("Plot the gender count with bars.")
