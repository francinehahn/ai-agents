from agent import Agent

def main():
    input_text = {
        "text": "Good morning! I hope you have a wonderful day."
    }

    agent = Agent()
    resp = agent.invoke(input_text=input_text)

    return resp

if __name__ == "__main__":
    resp = main()
    print(resp)