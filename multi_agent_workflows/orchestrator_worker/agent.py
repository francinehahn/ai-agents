import os
from dotenv import load_dotenv
from typing import TypedDict, List, Annotated
from pydantic import BaseModel, Field
import operator

from langchain_cohere.chat_models import ChatCohere
from langchain_core.prompts import ChatPromptTemplate
from langgraph.types import Send
from langgraph.graph import START, END, StateGraph

load_dotenv()

class Dish(BaseModel):
    name: str = Field(
        description="Name of the dish (for example, Spaghetti Bolognese, Chicken Curry)."
    )
    ingredients: List[str] = Field(
        description="List of ingredients needed for this dish, separated by commas."
    )
    location: str = Field(
        description="The cuisine or cultural origin of the dish (for example, Italian, Indian, Mexican)."
    )

class Dishes(BaseModel):
    sections: List[Dish] = Field(
        description="A list of grocery sections, one for each dish, with ingredients."
    )

class WorkerState(TypedDict):
    section: Dish
    completed_menu: Annotated[list, operator.add] # list with addition operators between elements

class State(TypedDict):
    meals: str  # The user's input listing the meals to prepare
    sections: List[Dish] # One section per meal/dish with ingredients
    completed_menu: Annotated[List[str], operator.add]  # Worker written dish guide chunks
    final_meal_guide: str  # Fully compiled, readable menu

class Agent:
    def __init__(self):
        self.llm = ChatCohere(
            cohere_api_key=os.environ.get("CO_API_KEY"), 
            model="command-a-03-2025"
        )

    def _chef_worker(self, state: WorkerState):
        """Worker node that generates the cooking instructions for one meal section."""

        chef_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a world-class chef from {location}.\n\n"
                "Please introduce yourself briefly and present a detailed walkthrough for preparing the dish the user wants.\n"
                "Your response should include:\n"
                "- Start with hello with your name and culinary background\n"
                "- A clear list of preparation steps\n"
                "- A full explanation of the cooking process\n\n"
            ),
            (
                "human",
                "I want to prepare the dish {name}. Use the following ingredients: {ingredients}."
            )
        ])

        chef_pipe = chef_prompt | self.llm

        # Use the language model to generate a meal preparation plan
        # The model receives the dish name, location, and ingredients from the current section
        meal_plan = chef_pipe.invoke({
            "name": state["section"].name,
            "location": state["section"].location,
            "ingredients": state["section"].ingredients
        })

        # Return the generated meal plan wrapped in a list under completed_sections
        # This will be merged into the main state using operator.add in LangGraph
        resp = {"completed_menu": [meal_plan.content]}
        print(f"chef worker response: {resp}\n\n")
        return resp

    def _orchestrator(self, state: State):
        """Orchestrator that generates a structured dish list from the given meals."""

        # construct a prompt template
        dish_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are an assistant that generates a structured grocery list for meals that the user wants to prepare.\n\n"
                "For each meal, return a section with:\n"
                "- the name of the dish\n"
                "- a comma-separated list of ingredients needed for that dish.\n"
                "- the cuisine or cultural origin of the food"
            ),
            (
                "human",
                "I would like to prepare the following meals: {meals}"
            )
        ])

        # use LCEL to pipe the prompt to an LLM with a structured output of Dishes
        planner_pipe = dish_prompt | self.llm.with_structured_output(Dishes)

        # use the planner_pipe LLM to break the user's meal list into structured dish sections
        dish_descriptions = planner_pipe.invoke({"meals": state["meals"]})

        # return the list of dish sections to be passed to worker nodes
        resp = {"sections": dish_descriptions.sections}
        print(f"orchestrator response: {resp}\n\n")
        return resp
    
    def _assign_workers(self, state: State):
        """Assign a worker to each section in the plan"""

        # Kick off section writing in parallel via Send() API
        resp = [Send("chef_worker", {"section": s}) for s in state["sections"]]
        print(f"assigning the workers: {resp}\n\n")
        return resp
    
    def _synthesizer(self, state: State):
        """Synthesize full report from sections"""

        # list of completed sections
        completed_sections = state["completed_menu"]

        # format completed section to str to use as context for final sections
        completed_menu = "\n\n---\n\n".join(completed_sections)

        return {"final_meal_guide": completed_menu}
    
    def invoke(self, user_input):
        # instantiate the builder
        orchestrator_worker_builder = StateGraph(State)

        # add the nodes
        orchestrator_worker_builder.add_node("orchestrator", self._orchestrator)
        orchestrator_worker_builder.add_node("chef_worker", self._chef_worker)
        orchestrator_worker_builder.add_node("synthesizer", self._synthesizer)

        orchestrator_worker_builder.add_conditional_edges(
            "orchestrator", self._assign_workers, ["chef_worker"] # source node, routing function, list of allowed targets
        )

        # add the edges, connections between nodes
        orchestrator_worker_builder.add_edge(START, "orchestrator")
        orchestrator_worker_builder.add_edge("chef_worker", "synthesizer")
        orchestrator_worker_builder.add_edge("synthesizer", END)

        # compile the builder to get a complete workflow executable
        orchestrator_worker = orchestrator_worker_builder.compile()

        state = orchestrator_worker.invoke(user_input)

        return state