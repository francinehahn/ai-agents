from agent import ClothingRecommendationAgent
from langchain_core.messages import HumanMessage

agent = ClothingRecommendationAgent()

inputs = {"messages": [HumanMessage(content="What's the weather like in Zurich, and what should I wear based on the temperature?")]}
response = agent.invoke(input=inputs)

print(response)