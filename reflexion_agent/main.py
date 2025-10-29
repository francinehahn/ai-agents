from agent import ReflexionAgent

agent = ReflexionAgent()

query = "I'm pre-diabetic and need to lower my blood sugar, and I have heart issues. What breakfast foods should I eat and avoid"
response = agent.invoke(query=query)
print(f"final response: {response}")