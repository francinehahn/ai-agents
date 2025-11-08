import os
from dotenv import load_dotenv
from crewai_tools import SerperDevTool, PDFSearchTool
from crewai import Agent, Task, Crew, Process, LLM

load_dotenv()

class MultiAgent():
    def __init__(self):
        self.web_search_tool = SerperDevTool()

        self.pdf_search_tool = PDFSearchTool(
            pdf="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/7vgNfis17dQfjHAiIKkBOg/The-Daily-Dish-FAQ.pdf",
            config=dict(
                embedder=dict(
                    provider="huggingface",
                    config=dict(
                        model="sentence-transformers/all-MiniLM-L6-v2"
                    )
                )
            )
        )
                
        self.llm = LLM(
            api_key=os.environ.get("OPEN_AI_KEY"),
            model=os.environ.get("OPEN_AI_MODEL")
        )
    
    def _faq_search_task(self):
        return Task(
            description="Search the restaurant's FAQ PDF for information related to the customer's query: '{customer_query}'.",
            expected_output="A snippet of the most relevant information from the PDF, or a statement that the information was not found.",
            tools=[self.pdf_search_tool], # Tool assigned directly to the task
            agent=self._task_centric_agent()
        )

    def _response_drafting_task(self):
        return Task(
            description="Using the information gathered from the FAQ search, draft a friendly and comprehensive response to the customer's query: '{customer_query}'.",
            expected_output="The final, customer-facing response.",
            agent=self._task_centric_agent(),
            context=[self._faq_search_task()]
        )

    def _task_centric_agent(self):
        return Agent(
            role="Customer Service Specialist",
            goal="Provide exceptional customer service by following a multi-step process to answer customer questions accurately.",
            backstory="""You are an AI assistant for 'The Daily Dish'.
            You are an expert at following instructions. You will be given a sequence of tasks to complete.
            For each task, you will be provided with the specific tool needed to accomplish it.
            Your job is to execute each task diligently and pass the results to the next step.""",
            tools=[], # The agent is not given any tools directly
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )

    def task_centric_crew(self):
        return Crew(
            agents=[self._task_centric_agent()],
            tasks=[self._faq_search_task(), self._response_drafting_task()],
            process=Process.sequential,
            verbose=True
        )