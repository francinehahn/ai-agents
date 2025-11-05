from multi_agent import MultiAgent

def main():
    agent = MultiAgent()
    
    while True: 
        user_input = input("\nYour question: ").lower()
        if user_input == 'exit':
            print("Thank you for chatting. Have a great day!")
            break
        
        if not user_input:
            print("Please type a question.")
            continue

        try:
            # Here we use our more advanced, task-centric crew
            result_task_centric = agent.task_centric_crew().kickoff(inputs={'customer_query': user_input})
            print("\n--- The Daily Dish Assistant ---")
            print(result_task_centric)
            print("--------------------------------")
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()