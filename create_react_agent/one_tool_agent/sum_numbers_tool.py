from typing import Dict, Union
import re
from langchain.tools import tool

@tool
def sum_numbers_with_complex_output(inputs: str) -> Dict[str, Union[float, str]]:
    """
    Extracts and sums all integers and decimals from the input
    """
    matches = re.findall(r"-?\d+(?:\.\d+)?", inputs)
    if not matches:
        return {"result": "No numbers found in input."}
    try:
        numbers = [float(num) for num in matches]
        total = sum(numbers)
        return {"result": total}
    except Exception as e:
        return {"result": f"Error during summation: {str(e)}"}