import os
import json
from dotenv import load_dotenv
from typing import Annotated, Sequence, TypedDict

from tools.clothing_recommendation_tool import recommend_clothing
from tools.search_tool import search_tool
from langchain_cohere.chat_models import ChatCohere
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage, ToolMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END

load_dotenv()

class AgentState(TypedDict):
    """The state of the agent."""

    # add_messages is a reducer
    # See https://langchain-ai.github.io/langgraph/concepts/low_level/#reducers
    messages: Annotated[Sequence[BaseMessage], add_messages]
    
class ClothingRecommendationAgent():
    def __init__(self):
        self.llm = ChatCohere(
            cohere_api_key=os.environ.get("CO_API_KEY"), 
            model="command-a-03-2025"
        )

        tools = [search_tool, recommend_clothing]
        self.tools_by_name = { tool.name:tool for tool in tools}

        self.chat_prompt = ChatPromptTemplate.from_messages([
            ("system", """
            You are a helpful AI assistant that thinks step-by-step and uses tools when needed.

            When responding to queries:
            1. First, think about what information you need
            2. Use available tools if you need current data or specific capabilities  
            3. Provide clear, helpful responses based on your reasoning and any tool results

            Always explain your thinking process to help users understand your approach.
            """),
                MessagesPlaceholder(variable_name="scratch_pad")
            ])
        
        self.model_react = self.chat_prompt | self.llm.bind_tools(tools)
    
    def _tool_node(self, state: AgentState):
        """Execute all tool calls from the last message in the state."""
        outputs = []
        for tool_call in state["messages"][-1].tool_calls:
            tool_result = self.tools_by_name[tool_call["name"]].invoke(tool_call["args"])
            print(f"tool result: {tool_result}\n\n")
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": outputs}
    
    def _should_continue(self, state: AgentState):
        """Determine whether to continue with tool use or end the conversation."""
        messages = state["messages"]
        last_message = messages[-1]
        # If there is no function call, then we finish
        if not last_message.tool_calls:
            return "end"
        # Otherwise if there is, we continue
        else:
            return "continue"
        
    def _call_model(self, state: AgentState):
        """Invoke the model with the current conversation state."""
        response = self.model_react.invoke({"scratch_pad": state["messages"]})
        return {"messages": [response]}
    
    def invoke(self, input):
        # Define a new graph
        workflow = StateGraph(AgentState)

        # Define the two nodes we will cycle between
        workflow.add_node("agent", self._call_model)
        workflow.add_node("tools", self._tool_node)

        # Add edges between nodes
        workflow.add_edge("tools", "agent")  # After tools, always go back to agent

        # Add conditional logic
        workflow.add_conditional_edges(
            "agent",
            self._should_continue,
            {
                "continue": "tools",  # If tools needed, go to tools node
                "end": END,          # If done, end the conversation
            },
        )

        # Set entry point
        workflow.set_entry_point("agent")

        # Compile the graph
        graph = workflow.compile()

        responses = graph.invoke(input=input)

        return responses["messages"][-1].content