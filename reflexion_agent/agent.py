import os
import json
from dotenv import load_dotenv
from typing import List, Annotated, TypedDict
import operator
from pydantic import BaseModel, Field

from langchain_cohere.chat_models import ChatCohere
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, ToolMessage, BaseMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import END, StateGraph

load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]

class Reflection(BaseModel):
	missing: str = Field(description="What information is missing")
	superfluous: str = Field(description="What information is unnecessary")

class AnswerQuestion(BaseModel):
	answer: str = Field(description="Main response to the question")
	reflection: Reflection = Field(description="Self-critique of the answer")
	search_queries: List[str] = Field(description="Queries for additional research")
     
class ReviseAnswer(AnswerQuestion):
    """Revise your original answer to your question."""
    references: List[str] = Field(description="Citations motivating your updated answer.")
     
class ReflexionAgent:
    def __init__(self):
        self.llm = ChatCohere(
            cohere_api_key=os.environ.get("CO_API_KEY"), 
            model="command-a-03-2025"
        )

        self.prompt_template = ChatPromptTemplate.from_messages([
            (
                "system",
                """You are Dr. Paul Saladino, "Carnivore MD," advocating for animal-based nutrition and challenging plant-centric dietary dogma. Focus on the superior bioavailability of animal foods and the potential toxicity of plant compounds such as oxalates, lectins, and phytates.

                Your response must follow these steps:
                1. {first_instruction}
                2. Present the evolutionary and biochemical rationale for animal-based nutrition, emphasizing nutrient density, bioavailability, and the absence of plant antinutrients.
                3. Challenge conventional "plants are healthy" narratives with mechanistic evidence about plant defense compounds and their impact on human physiology.
                4. Reflect and critique your answer. Be rigorous about evolutionary consistency and nutrient science.
                5. After the reflection, **list 1-3 search queries separately** for researching antinutrients, bioavailability studies, or evolutionary nutrition. Do not include them inside the reflection.

                Focus on: organ meats as superfoods, the carnivore elimination protocol, and why "fiber is unnecessary" based on human evolutionary biology.
                """
            ),
            MessagesPlaceholder(variable_name="messages"),
            (
                "human", 
                "Answer the user's question above using the required format, emphasizing the superiority of animal-based nutrition."
            ),
        ])

        revise_instructions = """Revise your previous answer using the new information, applying the rigor and evidence-based approach of Dr. David Attia.
        - Incorporate the previous critique to add clinically relevant information, focusing on mechanistic understanding and individual variability.
        - You MUST include numerical citations referencing peer-reviewed research, randomized controlled trials, or meta-analyses to ensure medical accuracy.
        - Distinguish between correlation and causation, and acknowledge limitations in current research.
        - Address potential biomarker considerations (lipid panels, inflammatory markers, and so on) when relevant.
        - Add a "References" section to the bottom of your answer (which does not count towards the word limit) in the form of:
        - [1] https://example.com
        - [2] https://example.com
        - Use the previous critique to remove speculation and ensure claims are supported by high-quality evidence. Keep response under 250 words with precision over volume.
        - When discussing nutritional interventions, consider metabolic flexibility, insulin sensitivity, and individual response variability.
        """
        self.revisor_prompt = self.prompt_template.partial(first_instruction=revise_instructions)
        
        self.loop_count = 0
        self.tavily_tool = TavilySearchResults(max_results=3)
        self.graph = StateGraph(state_schema=AgentState)
    
    def _execute_tools(self, state: AgentState):
        last_ai_message = state["messages"][-1]

        with open("reflexion_agent/result.txt", "a", encoding="utf-8") as file:
            file.write("\n\n--------EXECUTE TOOLS--------\n\n")
            file.write(f"last message: {last_ai_message}")

        tool_messages = []
        for tool_call in last_ai_message.tool_calls:
            if tool_call["name"] in ["AnswerQuestion", "ReviseAnswer"]:
                call_id = tool_call["id"]
                search_queries = tool_call["args"].get("search_queries", [])
                query_results = {}
                for query in search_queries:
                    result = self.tavily_tool.invoke(query)

                    with open("reflexion_agent/result.txt", "a", encoding="utf-8") as file:
                        file.write(f"\nsearch result: {result}\n\n")

                    query_results[query] = result
                tool_messages.append(ToolMessage(
                    content=json.dumps(query_results),
                    tool_call_id=call_id)
                )

        return {"messages": tool_messages}

    def _revisor(self, state: AgentState):
        revisor_chain = self.revisor_prompt | self.llm.bind_tools(tools=[ReviseAnswer])
        response = revisor_chain.invoke({"messages": state["messages"]})

        with open("reflexion_agent/result.txt", "a", encoding="utf-8") as file:
            file.write("\n\n-----REVISOR-----\n\n")
            file.write(f"response: {response}")
        
        return {"messages": [response]}

    def _responder(self, state: AgentState):
        messages = state["messages"]

        first_responder_prompt = self.prompt_template.partial(first_instruction="Provide a detailed ~250 word answer")
        chain = first_responder_prompt | self.llm.bind_tools(tools=[AnswerQuestion])

        response = chain.invoke({"messages": messages})

        with open("reflexion_agent/result.txt", "a", encoding="utf-8") as file:
            file.write("\n-----RESPONDER-----\n")
            file.write(f"response: {response}")

        return {"messages": [response]}

    def _event_loop(self, state: AgentState) -> str:
        with open("reflexion_agent/result.txt", "a", encoding="utf-8") as file:
            file.write(f"\n\nevent loope count: {self.loop_count}\n\n")

        self.loop_count += 1

        if self.loop_count >= 4:
            return END
        
        return "execute_tools"

    
    def invoke(self, query):
        self.graph.add_node("respond", self._responder)
        self.graph.add_node("execute_tools", self._execute_tools)
        self.graph.add_node("revisor", self._revisor)

        self.graph.add_edge("respond", "execute_tools")
        self.graph.add_edge("execute_tools", "revisor")

        self.graph.add_conditional_edges("revisor", self._event_loop)
        self.graph.set_entry_point("respond")  
        
        app = self.graph.compile()

        initial_state = {"messages": [HumanMessage(content=query)]}
        responses = app.invoke(initial_state)

        with open("reflexion_agent/result.txt", "a", encoding="utf-8") as file:
            file.write(f"\n\n--------FINAL RESPONSE---------\n\n")

        for response in responses["messages"]:
            with open("reflexion_agent/result.txt", "a", encoding="utf-8") as file:
                file.write(f"\n\nresponse: {response}\n\n")

        return responses["messages"][-1]