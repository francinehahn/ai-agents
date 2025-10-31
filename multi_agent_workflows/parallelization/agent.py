import os
from dotenv import load_dotenv
from typing import TypedDict

from langchain_cohere.chat_models import ChatCohere
from langgraph.graph import START, END, StateGraph

load_dotenv()

class State(TypedDict):
    text: str
    french: str
    spanish: str
    japanese: str
    combined_output: str

class Agent:
    def __init__(self):
        self.llm = ChatCohere(
            cohere_api_key=os.environ.get("CO_API_KEY"), 
            model="command-a-03-2025"
        )

    def _translate_french(self, state: State) -> dict:
        print("Translating to french...")
        response = self.llm.invoke(f"Translate the following text to French:\n\n{state['text']}")
        return {"french": response.content.strip()}
    
    def _translate_spanish(self, state: State) -> dict:
        print("Translating to spanich...")
        response = self.llm.invoke(f"Translate the following text to Spanish:\n\n{state['text']}")
        return {"spanish": response.content.strip()}
    
    def _translate_japanese(self, state: State) -> dict:
        print("Translating to japanese...")
        response = self.llm.invoke(f"Translate the following text to Japanese:\n\n{state['text']}")
        return {"japanese": response.content.strip()}
    
    def _aggregator(self, state: State) -> dict:
        combined = f"Original Text: {state['text']}\n\n"
        combined += f"French: {state['french']}\n\n"
        combined += f"Spanish: {state['spanish']}\n\n"
        combined += f"Japanese: {state['japanese']}\n"
        return {"combined_output": combined}
    
    def invoke(self, input_text):
        graph = StateGraph(State)

        graph.add_node("translate_french", self._translate_french)
        graph.add_node("translate_spanish", self._translate_spanish)
        graph.add_node("translate_japanese", self._translate_japanese)
        graph.add_node("aggregator", self._aggregator)

        # Connect parallel nodes from START
        graph.add_edge(START, "translate_french")
        graph.add_edge(START, "translate_spanish")
        graph.add_edge(START, "translate_japanese")

        # Connect all translation nodes to the aggregator
        graph.add_edge("translate_french", "aggregator")
        graph.add_edge("translate_spanish", "aggregator")
        graph.add_edge("translate_japanese", "aggregator")

        # Final node
        graph.add_edge("aggregator", END)

        # Compile the graph
        app = graph.compile()

        result = app.invoke(input_text)

        return result