import os
from dotenv import load_dotenv
from autogen import ConversableAgent, GroupChat, GroupChatManager
from openai import OpenAI

load_dotenv()

class HealthCareAgent:
    def __init__(self):
        llm_config = {"config_list": [{"model": os.environ.get("OPEN_AI_MODEL"), "api_key": os.environ.get("OPEN_AI_KEY")}]}

        self.patient_agent = ConversableAgent(
            name="patient", 
            system_message="You describe symptoms and ask for medical help.", 
            llm_config=llm_config
        )

        self.diagnosis_agent = ConversableAgent(
            name="diagnosis", 
            system_message="You analyze symptoms and provide a possible diagnosis. Summarize key points in one response.", 
            llm_config=llm_config
        )

        self.pharmacy_agent = ConversableAgent(
            name="pharmacy", 
            system_message="You recommend medications based on diagnosis. Only respond once.", 
            llm_config=llm_config
        )

        self.consultation_agent = ConversableAgent(
            name="consultation", 
            system_message="You determine if a doctor's visit is required. Provide a final summary with clear next steps. IMPORTANT: End your response with 'CONSULTATION_COMPLETE' to signal the end of the conversation.", 
            llm_config=llm_config
        )

    def invoke(self, input_message: str):
        groupchat = GroupChat(
            agents=[self.diagnosis_agent, self.pharmacy_agent, self.consultation_agent],  # Patient only initiates
            messages=[], 
            max_round=5,  # Limits conversation to 5 rounds
            speaker_selection_method="round_robin"  # Ensures structured conversation flow
        )

        manager = GroupChatManager(name="manager", groupchat=groupchat)

        print("\n🩺 Diagnosing symptoms...")
        response = self.patient_agent.initiate_chat(
            manager, 
            message=input_message,
        )

        return response