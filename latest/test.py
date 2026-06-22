from llm_sdk import Small_LLM_Model
import json
import sys
import re
from parameters import extract_params, get_parameters

def get_functions(func_src: str) -> dict[str, str]:
    """Return {function_name: function_description} from JSON file."""
    func_dict = {}
    try:
        with open(func_src, "r") as f:
            func_def = json.load(f)
    except FileNotFoundError:
        print("File not Found")
        sys.exit()
    for function in func_def:
        name = function.get('name')
        desc = function.get('description')
        func_dict.update({name: desc})
    return func_dict

def build_context(prompt: str, functions: dict[str, str]) -> str:
    """Build the context to let the model compare functions"""
    lines = [f"User request:{prompt}", "Available functions:"]
    for name, desc in functions.items():
        lines.append(f"- {name}: {desc}")
    lines.append("Choose the best function name.")
    return "\n".join(lines)

def score_function(model: Small_LLM_Model, context_ids: list[int], function_name: str) -> float:
    function_name_ids = model.encode(function_name)[0].tolist()

    if len(function_name_ids) == 0:
        return float("-inf")
    
    current_ids = list(context_ids)
    total_score = 0.0

    for expected_token_id in function_name_ids:
        logits = model.get_logits_from_input_ids(current_ids)

        if expected_token_id >= len(logits):
            return float("-inf")
        
        expected_token_score = float(logits[expected_token_id])
        total_score += expected_token_score

        current_ids.append(expected_token_id)
    average_score = total_score / len(function_name_ids)
    return average_score

def select_function(model: Small_LLM_Model, prompt: str, functions: dict[str, str]) -> str:
    """
    Select the best function using constrained decoding.

    Build the function name token-by-token, allowing only tokens that
    keep at least one valid function name possible at each step.
    """
    context_text = build_context(prompt, functions) + "\nFunction name:"
    context_ids = model.encode(context_text)[0].tolist()

    name_to_ids: dict[str, list[int]] = {}
    for name in functions:
        token_ids = model.encode(" " + name)[0].tolist()
        name_to_ids[name] = token_ids

    generated: list[int] = []
    candidates = dict(name_to_ids)

    while True:
        logits = model.get_logits_from_input_ids(context_ids + generated)

        allowed: set[int] = set()
        for name, token_ids in candidates.items():
            if len(token_ids) > len(generated):
                next_token_for_candidate = token_ids[len(generated)]
                allowed.add(next_token_for_candidate)

        if not allowed:
            if candidates:
                return max(candidates)
            return ""

        next_token = max(allowed, key=lambda tid: logits[tid])
        generated.append(next_token)

        new_candidates = {}
        for name, token_ids in candidates.items():
            is_long_enough = len(token_ids) >= len(generated)
            matches_current_token = token_ids[len(generated) - 1] == next_token

            if is_long_enough and matches_current_token:
                new_candidates[name] = token_ids

        candidates = new_candidates

        fully_matched = []
        for name, token_ids in candidates.items():
            if len(token_ids) == len(generated):
                fully_matched.append(name)

        if len(fully_matched) == 1:
            return fully_matched[0]

        if not candidates:
            return ""


def main():
    func_src = "data/input/functions_definition.json"
    model = Small_LLM_Model()
    prompt = "Replace all numbers in \"Hello 34 I'm 233 years old\" with NUMBERS"
    functions = get_functions(func_src)
    selected = select_function(model, prompt, functions)
    print(f"Selected function: {selected}")
    print(functions[selected])
    params = get_parameters(selected, func_src)
    print(extract_params(prompt, model, params))

main()