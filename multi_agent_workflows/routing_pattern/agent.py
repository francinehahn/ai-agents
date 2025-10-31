import os
from dotenv import load_dotenv
from typing import TypedDict
from pydantic import Field, BaseModel

from langchain_cohere.chat_models import ChatCohere
from langgraph.graph import END, StateGraph

load_dotenv()


class RouterState(TypedDict):
    user_input: str
    task_type: str
    output: str

class Router(BaseModel):
    role: str = Field(..., description="Decide whether the user wants to summarize a passage  ouput 'summarize'  or translate text into French oupput translate.")
    

class Agent:
    def __init__(self):
        self.llm = ChatCohere(
            cohere_api_key=os.environ.get("CO_API_KEY"), 
            model="command-a-03-2025"
        )

        self.llm_router = self.llm.bind_tools([Router])

    def _router_node(self, state: RouterState) -> RouterState:
        routing_prompt = f"""
        You are an AI task classifier.
        
        Decide whether the user wants to:
        - "summarize" a passage
        - or "translate" text into French
        
        Respond with just one word: 'summarize' or 'translate'.
        
        User Input: "{state['user_input']}"
        """

        response = self.llm_router.invoke(routing_prompt)

        return {**state, "task_type": response.tool_calls[0]['args']['role']} # This becomes the next node's name!
    
    def _router(self, state: RouterState) -> str:
        return state['task_type']
    
    def _summarize_node(self, state: RouterState) -> RouterState:
        prompt = f"Please summarize the following passage:\n\n{state['user_input']}"
        response = self.llm.invoke(prompt)
        
        return {**state, "task_type": "summarize", "output": response.content}
    
    def _translate_node(self, state: RouterState) -> RouterState:
        prompt = f"Translate the following text to French:\n\n{state['user_input']}"
        response = self.llm.invoke(prompt)

        return {**state, "task_type": "translate", "output": response.content}
    
    def invoke(self, input_text):
        workflow = StateGraph(RouterState)
        workflow.add_node("router", self._router_node)
        workflow.add_node("summarize", self._summarize_node)
        workflow.add_node("translate", self._translate_node)

        workflow.set_entry_point("router")

        workflow.add_conditional_edges("router", self._router, {
            "summarize": "summarize",
            "translate": "translate"
        })

        workflow.set_finish_point("summarize")
        workflow.set_finish_point("translate")

        app = workflow.compile()

        result = app.invoke(input_text)

        return result