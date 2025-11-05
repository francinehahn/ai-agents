from multi_agent import MultiAgent

def main():
    agent = MultiAgent()
    inputs = {"topic": "Latest Generative AI breakthroughs"}
    response = agent.invoke(inputs=inputs)
    print(response)

if __name__ == "__main__":
    main()