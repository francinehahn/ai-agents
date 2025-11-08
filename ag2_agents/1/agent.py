import os
from dotenv import load_dotenv
from autogen import ConversableAgent, AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
from autogen.llm_config import LLMConfig
import json
import time
import random

load_dotenv()

class Agent:
    def __init__(self):
        llm_config = LLMConfig({
            "model": os.environ.get("OPEN_AI_MODEL"),
            "api_key": os.environ.get("OPEN_AI_KEY")
        })

        with llm_config:
            # Create the student agent (asks questions)
            self.student = ConversableAgent(
                name="student",
                system_message="You are a curious student. You ask clear, specific questions to learn new concepts.",
                human_input_mode="NEVER"  # disables manual input during chat
            )

            # Create the tutor agent (responds with beginner-friendly answers)
            self.tutor = ConversableAgent(
                name="tutor",
                system_message="You are a helpful tutor who provides clear and concise explanations suitable for a beginner.",
                human_input_mode="NEVER"
            )

    def invoke(self, input_message: str):
        chat_result = self.student.initiate_chat(
            recipient=self.tutor,                                # who the student is talking to
            message=input_message,  # the student's question
            max_turns=2,                                     # total number of back-and-forth messages
            summary_method="reflection_with_llm"            # generate a final summary using LLM
        )
        return chat_result