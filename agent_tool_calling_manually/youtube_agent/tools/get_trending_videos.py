from langchain.tools import tool
from typing import List, Dict
import yt_dlp
import logging

yt_dpl_logger = logging.getLogger("yt_dlp")
yt_dpl_logger.setLevel(logging.ERROR)

@tool
def get_trending_videos(region_code: str) -> List[Dict]:
    """
    Fetches currently trending YouTube videos for a specific region.
    
    Args:
        region_code (str): 2-letter country code (e.g., "US", "IN", "GB")
        
    Returns:
        List of dictionaries with video details: title, video_id, channel, view_count, duration
    """
    ydl_opts = {
        "geo_bypass_country": region_code.upper(),
        "extract_flat": True,
        "quiet": True,
        "force_generic_extractor": True,
        "logger": yt_dpl_logger
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(
                "https://www.youtube.com/feed/trending",
                download=False
            )
            
            trending_videos = []
            for entry in info["entries"]:
                video_data = {
                    "title": entry.get("title", "N/A"),
                    "video_id": entry.get("id", "N/A"),
                    "url": entry.get("url", "N/A"),
                    "channel": entry.get("uploader", "N/A"),
                    "duration": entry.get("duration", 0),
                    "view_count": entry.get("view_count", 0)
                }
                trending_videos.append(video_data)
                
            return trending_videos[:25]  # Return top 25 trending videos
            
    except Exception as e:
        return [{"error": f"Failed to fetch trending videos: {str(e)}"}]