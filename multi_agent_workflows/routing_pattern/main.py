from agent import Agent

def main(user_input):
    agent = Agent()
    result = agent.invoke(input_text=user_input)
    return result

if __name__ == "__main__":
    # TRANSLATION
    input_text = {
        "user_input": "Can you translate this sentence: I love programming?"
    }

    result = main(user_input=input_text)
    print(f"TRANSLATION--------\n{result}\n\n")

    # SUMMARY
    input_text = {
        "user_input": "Can you summarize this sentence: I love programming so much it is the best thing ever. All I want to do is programming"
    }

    result = main(user_input=input_text)
    print(f"SUMMARY-----------\n{result}\n\n")