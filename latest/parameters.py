from llm_sdk import Small_LLM_Model
import json
import sys
import re
import string

def get_allowed_tokens(param_type: str, model: Small_LLM_Model) -> set[int]:
    if param_type == "number":
        allowed_char = "0123456789-"
    elif param_type == "string":
        allowed_char = string.ascii_letters + string.digits + string.punctuation + " "
    else:
        allowed_char = ""

    allowed_tokens = set()
    for char in allowed_char:
        token_id = model.encode(char)[0].tolist()[0]
        allowed_tokens.add(token_id)
    return allowed_tokens

def build_param_context(prompt: str, param_name: str, param_type: str, index: int) -> str:
    ordinals = ["first", "second", "third", "fourth", "fifth"]
    ordinal = ordinals[index]
    lines = [f"Prompt: {prompt}", 
             f"Extract the {ordinal} {param_type}",
             "Value:"]
    return "\n".join(lines)

def extract_all_numbers(prompt: str) -> list[str]:
    """Extract all integers and floats from prompt using regex."""
    numbers = []
    pattern = r"(?<!\w)[+-]?(?:\d*\.\d+|\d+\.\d*|\d+)(?!\w)"
    for match in re.finditer(pattern, prompt):
        numbers.append(match.group(0))
    return numbers

def extract_string_value(prompt: str, param_name: str) -> str:
    """Extract a string parameter using prompt-specific heuristics."""
    prompt_lower = prompt.lower()
    quoted_strings = re.findall(r"['\"]([^'\"]+)['\"]", prompt)

    if param_name == "source_string":
        if quoted_strings:
            return quoted_strings[0].strip()
        return ""

    if param_name == "regex":
        if "vowel" in prompt_lower:
            return "[aeiouAEIOU]"
        if "digit" in prompt_lower:
            return r"\d"
        if "letter" in prompt_lower:
            return r"[A-Za-z]"
        return ""

    if param_name == "replacement":
        replacement_match = re.search(r"\bwith\s+(.+)$", prompt, re.IGNORECASE)
        if replacement_match:
            value = replacement_match.group(1).strip().strip(".?!")
            value = value.strip("'\"")
            if "asterisk" in value.lower():
                return "*"
            return value
        return ""

    if quoted_strings:
        return quoted_strings[0].strip()

    return ""

def extract_params(prompt: str, model: Small_LLM_Model, params: dict) -> dict:
    """Extract all parameters from a prompt."""
    numbers = extract_all_numbers(prompt)
    results = {}
    number_index = 0
    for index, (param_name, param_info) in enumerate(params.items()):
        param_type = param_info["type"]
        if param_type == "number":
            if number_index < len(numbers):
                results[param_name] = numbers[number_index]
                number_index += 1
            else:
                results[param_name] = ""
        elif param_type == "string":
            results[param_name] = extract_string_value(prompt, param_name)
        else:
            results[param_name] = ""
    return results

def get_parameters(func: str, func_src: str):
    """Return a dict with the parameters and their type"""
    try:
        with open(func_src, "r") as f:
            func_def = json.load(f)
    except FileNotFoundError:
        print("File not Found")
        sys.exit()
    for function in func_def:
        if function.get('name') == func:
            return(function.get('parameters'))