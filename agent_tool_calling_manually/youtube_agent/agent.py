import os
from dotenv import load_dotenv

from langchain_cohere.chat_models import ChatCohere
from langchain_core.messages import HumanMessage, ToolMessage

from tools.extract_video_id import extract_video_id
from tools.fetch_transcript import fetch_transcript
from tools.get_full_metadata import get_full_metadata
from tools.get_thumbnails import get_thumbnails
from tools.get_trending_videos import get_trending_videos
from tools.search_youtube import search_youtube

load_dotenv()

class Agent:
    def __init__(self):
        llm = ChatCohere(
            cohere_api_key=os.environ.get("CO_API_KEY"), 
            model="command-a-03-2025"
        )

        self.agent = llm.bind_tools(tools=[
            extract_video_id, fetch_transcript, get_full_metadata, get_thumbnails, get_trending_videos, search_youtube
        ])

        self.tool_map = {
            "extract_video_id": extract_video_id, 
            "fetch_transcript": fetch_transcript,
            "get_full_metadata": get_full_metadata,
            "get_thumbnails": get_thumbnails,
            "get_trending_videos": get_trending_videos,
            "search_youtube": search_youtube
        }

    def invoke(self, query: str) -> str:
        # Step 1: Initial user message
        chat_history = [HumanMessage(content=query)]
        print(f"chat history: {chat_history}\n\n")

        while True:
            # Step 2: LLM chooses tool
            response = self.agent.invoke(chat_history)
            print(f"llm response: {response}")

            if not response.tool_calls:
                return response.content # Direct response, no tool needed
            
            # Step 3: Handle tool call
            tool_name = response.tool_calls[0]["name"]
            tool_args = response.tool_calls[0]["args"]
            tool_call_id = response.tool_calls[0]["id"]

            # Step 4: Call tool manually
            tool_result = self.tool_map[tool_name].invoke(tool_args)

            # Step 5: Send result back to LLM
            tool_message = ToolMessage(content=str(tool_result), tool_call_id=tool_call_id)
            chat_history.extend([response, tool_message])
