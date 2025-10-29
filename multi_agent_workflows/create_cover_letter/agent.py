import os
from dotenv import load_dotenv
from typing import TypedDict

from langchain_cohere.chat_models import ChatCohere
from langgraph.graph import END, StateGraph

load_dotenv()

class ChainState(TypedDict):
    job_description: str
    resume_summary: str
    cover_letter: str

class MultiAgentWorkflow:
    def __init__(self):
        self.llm = ChatCohere(
            cohere_api_key=os.environ.get("CO_API_KEY"), 
            model="command-a-03-2025"
        )

    def _generate_cover_letter(self, state: ChainState) -> ChainState:
        prompt = f"""
        You're a cover letter writing assistant. Using the resume summary below, write a professional and personalized cover letter for the following job.

        Resume Summary:
        {state['resume_summary']}

        Job Description:
        {state['job_description']}
        """

        response = self.llm.invoke(prompt)

        return {**state, "cover_letter": response.content}

    def _generate_resume_summary(self, state: ChainState) -> ChainState:
        prompt = f"""
        You're a resume assistant. Read the following job description and summarize the key qualifications and experience the ideal candidate should have, phrased as if from the perspective of a strong applicant's resume summary.

        Job Description:
        {state['job_description']}
        """

        response = self.llm.invoke(prompt)

        return {**state, "resume_summary": response.content}
    
    def invoke(self, input_state):
        workflow = StateGraph(ChainState)
        workflow.add_node("generate_resume_summary", self._generate_resume_summary)
        workflow.add_node("generate_cover_letter", self._generate_cover_letter)
        workflow.set_entry_point("generate_resume_summary")
        workflow.add_edge("generate_resume_summary", "generate_cover_letter")
        workflow.set_finish_point("generate_cover_letter")
        
        app = workflow.compile()
        result = app.invoke(input_state)
        
        return result["resume_summary"]