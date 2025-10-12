from langchain.tools import tool
import yt_dlp
import logging
from typing import List, Dict

yt_dpl_logger = logging.getLogger("yt_dlp")
yt_dpl_logger.setLevel(logging.ERROR)

@tool
def get_thumbnails(url: str) -> List[Dict]:
    """
    Get available thumbnails for a YouTube video using its URL.
    
    Args:
        url (str): YouTube video URL (any format)
        
    Returns:
        List of dictionaries with thumbnail URLs and resolutions in YouTube"s native order
    """
    
    try:
        with yt_dlp.YoutubeDL({"quiet": True, "logger": yt_dpl_logger}) as ydl:
            info = ydl.extract_info(url, download=False)
            
            thumbnails = []
            for t in info.get("thumbnails", []):
                if "url" in t:
                    thumbnails.append({
                        "url": t["url"],
                        "width": t.get("width"),
                        "height": t.get("height"),
                        "resolution": f"{t.get('width', '')}x{t.get('height', '')}".strip("x")
                    })
            
            return thumbnails

    except Exception as e:
        return [{"error": f"Failed to get thumbnails: {str(e)}"}]