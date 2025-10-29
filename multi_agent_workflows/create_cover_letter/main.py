from multi_agent_workflows.create_cover_letter.agent import MultiAgentWorkflow

def main():
    agent = MultiAgentWorkflow()

    input_state = {
        "job_description": "We are looking for a data scientist with experience in machine learning, NLP, and Python. Prior work with large datasets and experience deploying models into production is required."
    }

    result = agent.invoke(input_state)
    
    return result

result = main()
print(result)