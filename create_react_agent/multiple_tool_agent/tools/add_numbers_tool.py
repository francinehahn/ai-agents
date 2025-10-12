from typing import Dict, Union
import re
from langchain.tools import tool

@tool
def add_numbers(inputs: str) -> Dict:
    """
    Extracts numbers so we can sum all of them
    """
    print("add numbers tool being used...")
    numbers = [int(num) for num in re.findall(r"\d+", inputs)]
    print(f"numbers extracted: {numbers}")
    
    if not numbers:
        return {"result": 0}
    
    return {"result": sum(numbers)}
