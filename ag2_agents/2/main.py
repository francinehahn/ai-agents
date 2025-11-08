from agent import HealthCareAgent

def main():
    input_message = "I have a stomach ache, diarrhea, and vomiting. I haven't eaten anything different in the last few days that could have caused this. What could it be?"
    agent = HealthCareAgent()
    resp = agent.invoke(input_message=input_message)
    return resp

if __name__ == "__main__":
    resp = main()
    print(resp)
