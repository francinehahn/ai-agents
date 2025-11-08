from agent import Agent

def main():
    agent = Agent()
    input_message = "Can you explain what a neural network is?"
    resp = agent.invoke(input_message=input_message)
    return resp

if __name__ == "__main__":
    resp = main()
    print(resp)