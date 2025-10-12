from typing import Dict
import re
from langchain.tools import tool

@tool
def multiply_numbers(inputs: str) -> Dict:
    """
    Extracts numbers so we can multiply all of them
    """
    print("multiply numbers tool being used...")

    numbers = [int(num) for num in re.findall(r"\d+", inputs)]
    print(f"numbers extracted: {numbers}")

    if not numbers:
        return {"result": 1}
    
    result = 1
    for n in numbers:
        result *= n

    print(f"result: {result}")
    return {"result": result}