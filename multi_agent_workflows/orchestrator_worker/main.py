from agent import Agent


def main():
    agent = Agent()
    resp = agent.invoke(user_input={"meals": "Steak and eggs, tacos, and chili"})
    return resp

if __name__ == "__main__":
    resp = main()
    print(resp)