import os
from dotenv import load_dotenv
from typing import List, Annotated, TypedDict
import operator

from langchain_cohere.chat_models import ChatCohere
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END, StateGraph
from langchain.schema import HumanMessage, AIMessage, BaseMessage

load_dotenv()

class AgentState(TypedDict):
    """
    Annotated[..., operator.add] sugere que o grafo pode “somar/concatenar” listas ao combinar estados (dependendo de como StateGraph usa o schema). Na prática, isso indica que quando um nó retorna {"messages": [...]}, esses itens são concatenados à lista existente no estado.
    """
    messages: Annotated[List[BaseMessage], operator.add]

class ReflectionAgent:
    def __init__(self):
        llm = ChatCohere(
            cohere_api_key=os.environ.get("CO_API_KEY"), 
            model="command-a-03-2025"
        )

        generation_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a professional LinkedIn content assistant tasked with crafting engaging, insightful, and well-structured LinkedIn posts."
                    " Generate the best LinkedIn post possible for the user's request."
                    " If the user provides feedback or critique, respond with a refined version of your previous attempts, improving clarity, tone, or engagement as needed.",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        self.generate_chain = generation_prompt | llm

        reflection_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """You are a professional LinkedIn content strategist and thought leadership expert. Your task is to critically evaluate the given LinkedIn post and provide a comprehensive critique. Follow these guidelines:

                1. Assess the post’s overall quality, professionalism, and alignment with LinkedIn best practices.
                2. Evaluate the structure, tone, clarity, and readability of the post.
                3. Analyze the post’s potential for engagement (likes, comments, shares) and its effectiveness in building professional credibility.
                4. Consider the post’s relevance to the author’s industry, audience, or current trends.
                5. Examine the use of formatting (e.g., line breaks, bullet points), hashtags, mentions, and media (if any).
                6. Evaluate the effectiveness of any call-to-action or takeaway.

                Provide a detailed critique that includes:
                - A brief explanation of the post’s strengths and weaknesses.
                - Specific areas that could be improved.
                - Actionable suggestions for enhancing clarity, engagement, and professionalism.

                Your critique will be used to improve the post in the next revision step, so ensure your feedback is thoughtful, constructive, and practical.
                """
            ),
            ("human", "Here's the LinkedIn post draft: {post_content}"),
        ])

        self.reflect_chain = reflection_prompt | llm

        # Initialize a predefined MessageGraph
        self.graph = StateGraph(state_schema=AgentState)

    def _generation_node(self, state: AgentState) -> dict:
        messages = state["messages"]

        # Passar a lista de mensagens diretamente, não um dicionário
        generated_post = self.generate_chain.invoke({"messages": messages})
        return {"messages": [AIMessage(content=generated_post.content)]}
    
    def _reflection_node(self, state: AgentState) -> dict:
        messages = state["messages"]

        # Passar a lista de mensagens diretamente, não um dicionário
        last_ai_message = next((m for m in reversed(messages) if isinstance(m, AIMessage)), None)

        if not last_ai_message:
            raise ValueError("No AIMessage found to reflect on.")

        res = self.reflect_chain.invoke({"post_content": last_ai_message.content})

        return {"messages": [HumanMessage(content=res.content)]}
    
    def _should_continue(self, state: AgentState):
        messages = state["messages"]
        if len(messages) >= 6:  # 3 ciclos completos (cada ciclo adiciona 2 mensagens)
            return END
        return "reflect"

    def call(self, inputs):
        # Configurar o grafo
        self.graph.add_node("generate", self._generation_node)
        self.graph.add_node("reflect", self._reflection_node)
        
        # Definir as transições
        self.graph.set_entry_point("generate")
        self.graph.add_edge("reflect", "generate") # após finalizar o reflect, agora meu grafo sabe que deve voltar para generate
        self.graph.add_conditional_edges( # aqui eu defino que o meu grafo vai de generate para should continue e daí para reflect
            "generate", 
            self._should_continue,
            {
                "reflect": "reflect",
                END: END
            }
        )
        
        # Compilar e executar
        workflow = self.graph.compile()
        response = workflow.invoke(inputs)
        return response
