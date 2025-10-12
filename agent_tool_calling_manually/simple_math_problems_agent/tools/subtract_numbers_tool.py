from typing import Dict
import re
from langchain.tools import tool

@tool
def subtract_numbers(inputs: str) -> Dict:
    """
    Extracts numbers so we can subtract them
    """
    print("subtract numbers tool being used...")

    numbers = [int(num) for num in re.findall(r"\d+", inputs)]
    print(f"numbers extracted: {numbers}")

    if not numbers:
        return {"result": 0}
    
    result = numbers[0]
    for i in range(1, len(numbers)):
        result -= numbers[i]

    print(f"result: {result}")
    return {"result": result}