import os
from dotenv import load_dotenv
from crewai_tools import SerperDevTool
from crewai import Agent, Task, Crew, Process, LLM

load_dotenv()

class MultiAgent():
    def __init__(self):
        self.search_tool = SerperDevTool()

        self.llm = LLM(
            api_key=os.environ.get("OPEN_AI_KEY"),
            model=os.environ.get("OPEN_AI_MODEL")
        )

    def _writer_agent(self):
        return Agent(
            role='Tech Content Strategist',
            goal='Craft well-structured and engaging content based on research findings',
            backstory="""You are a skilled content strategist known for translating 
            complex topics into clear and compelling narratives. Your writing makes 
            information accessible and engaging for a wide audience.""",
            verbose=True,
            llm = self.llm,
            allow_delegation=True
        )

    def _researcher_agent(self):
        return Agent(
            role='Senior Research Analyst',
            goal='Uncover cutting-edge information and insights on any subject with comprehensive analysis',
            backstory="""You are an expert researcher with extensive experience in gathering, analyzing, and synthesizing information across multiple domains. 
            Your analytical skills allow you to quickly identify key trends, separate fact from opinion, and produce insightful reports on any topic. 
            You excel at finding reliable sources and extracting valuable information efficiently.""",
            verbose=True,
            allow_delegation=False,
            llm = self.llm,
            tools=[self.search_tool]
        )
    
    def _research_task(self):
        return Task(
            description="Analyze the major {topic}, identifying key trends and technologies. Provide a detailed report on their potential impact.",
            agent=self._researcher_agent(),
            expected_output="A detailed report on {topic}, including trends, emerging technologies, and their impact."
        )

    def _writer_task(self):
        return Task(
            description="Create an engaging blog post based on the research findings about {topic}. Tailor the content for a tech-savvy audience, ensuring clarity and interest.",
            agent=self._writer_agent(),
            expected_output="A 4-paragraph blog post on {topic}, written clearly and engagingly for tech enthusiasts."
        )

    def invoke(self, inputs):
        crew = Crew(
            agents=[self._researcher_agent(), self._writer_agent()],
            tasks=[self._research_task(), self._writer_task()],
            process=Process.sequential,
            verbose=True 
        )
        result = crew.kickoff(inputs=inputs)
        return result
