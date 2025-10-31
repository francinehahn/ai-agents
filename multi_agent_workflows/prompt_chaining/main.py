from agent import Agent

def main():
    agent = Agent()

    input_state = {
        "job_description": "We are looking for a data scientist with experience in machine learning, NLP, and Python. Prior work with large datasets and experience deploying models into production is required."
    }

    result = agent.invoke(input_state)
    
    return result

if __name__ == "__main__":
    result = main()
    print(result)