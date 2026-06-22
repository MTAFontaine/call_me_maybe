from llm_sdk.llm_sdk import Small_LLM_Model
import sys
import json
import re
import os


def get_functions(func_src):
    try:
        with open(func_src, "r") as f:
            func_def = json.load(f)
    except FileNotFoundError:
        print("File not Found")
        sys.exit()
    return func_def

def select_func(model, input_ids, funcs_name):
    func_name_to_tokens = {name: model.encode(name)[0].tolist() for name in funcs_name}
    valid_func_tokens = set(token for tokens in func_name_to_tokens.values() for token in tokens)

    ids_list = input_ids[0].tolist() 
    logits = model.get_logits_from_input_ids(ids_list)
    for token_id in range(len(logits)):
        if token_id not in valid_func_tokens:
            logits[token_id] = float('-inf')
    return (normalize_logits_by_token_count(func_name_to_tokens, logits))

def normalize_logits_by_token_count(func_name_to_tokens, logits):
    normalized_func_logits = {}
    for func_name, tokens in func_name_to_tokens.items():
        logit_sum = sum(logits[token] for token in tokens)
        normalized_logit = logit_sum / len(tokens)
        normalized_func_logits[func_name] = normalized_logit
    return (max(normalized_func_logits, key=normalized_func_logits.get))

def main():
    model = Small_LLM_Model()
    user_prompt = "What is the sum of 265 and 345?"

    func_src = "data/input/functions_definition.json"
    func_def = get_functions(func_src)
    funcs_name = [func["name"] for func in func_def]
    vocab = {}
    token_to_str = {}
    vocab_path = None
    for getter in ("get_path_to_vocab_file", "get_path_to_tokenizer_file", "get_path_to_merges_file"):
        if hasattr(model, getter):
            try:
                vocab_path = getattr(model, getter)()
                break
            except Exception:
                vocab_path = None
    if vocab_path:
        try:
            with open(vocab_path, "r") as vf:
                vocab = json.load(vf)
        except Exception:
            vocab = {}
    if isinstance(vocab, dict) and vocab:
        if all(isinstance(k, str) and isinstance(v, int) for k, v in vocab.items()):
            token_to_str = {v: k for k, v in vocab.items()}
        else:
            try:
                token_to_str = {int(k): v for k, v in vocab.items()}
            except Exception:
                token_to_str = {}
    else:
        token_to_str = {}
    func_name_tokens = {name: model.encode(name)[0].tolist() for name in funcs_name}
    prefix_text = '{"prompt": ' + json.dumps(user_prompt) + ', "name": "'
    prefix_ids = model.encode(prefix_text)[0].tolist()
    candidates = dict(func_name_tokens)
    generated_name_ids = []
    selected_name = None
    while True:
        logits = model.get_logits_from_input_ids(prefix_ids + generated_name_ids)

        allowed = set()
        for name, toks in candidates.items():
            if len(toks) > len(generated_name_ids):
                allowed.add(toks[len(generated_name_ids)])

        for i in range(len(logits)):
            if i not in allowed:
                logits[i] = float("-inf")

        next_token = max(range(len(logits)), key=lambda i: logits[i])
        generated_name_ids.append(next_token)

        candidates = {n: t for n, t in candidates.items() if len(t) >= len(generated_name_ids) and t[len(generated_name_ids) - 1] == next_token}

        done = [n for n, t in candidates.items() if len(t) == len(generated_name_ids)]
        if done:
            selected_name = done[0]
            break
    suffix_after_name = '", "parameters": {'
    suffix_ids = model.encode(suffix_after_name)[0].tolist()
    context_ids = prefix_ids + generated_name_ids + suffix_ids
    selected_function = next(func for func in func_def if func["name"] == selected_name)
    required_parameters = selected_function["parameters"]
    print("Selected function name:", selected_name)
    print(f"Required params : {required_parameters}")

    parameters = {}
    param_items = list(required_parameters.items())
    for idx, (pname, pinfo) in enumerate(param_items):

        key_text = json.dumps(pname) + ": "
        key_ids = model.encode(key_text)[0].tolist()
        context_ids += key_ids

        ptype = pinfo.get("type")
        if ptype == "number":
            num_re_full = re.compile(r"^[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?$")
            num_char_re = re.compile(r"^[0-9+\-\.eE]+$")
            value_ids = []
            for _ in range(10):
                logits = model.get_logits_from_input_ids(context_ids + value_ids)
        
                allowed = [tid for tid, tok in token_to_str.items() if num_char_re.match(tok.strip())]
                if not allowed:
                    break
                for i in range(len(logits)):
                    if i not in allowed:
                        logits[i] = float("-inf")
                next_token = max(range(len(logits)), key=lambda i: logits[i])
                value_ids.append(next_token)
        
                try:
                    cur = model.decode(value_ids).strip()
                except Exception:
                    cur = ''.join(token_to_str.get(t, '') for t in value_ids).strip()
        
                if num_re_full.match(cur):
                    break
    
            context_ids += value_ids
            try:
                value_str = model.decode(value_ids).strip()
            except Exception:
                value_str = ''.join(token_to_str.get(t, '') for t in value_ids).strip()
            try:
                parameters[pname] = float(value_str)
            except Exception:
                parameters[pname] = 0.0
        elif ptype == "string":
    
            logits = model.get_logits_from_input_ids(context_ids)
    
            next_token = max(range(len(logits)), key=lambda i: logits[i]) if logits else None
            if next_token is not None:
                context_ids.append(next_token)
                value_str = token_to_str.get(next_token, model.decode([next_token]) if hasattr(model, 'decode') else '')
            else:
                value_str = ""
            parameters[pname] = value_str.strip() if value_str is not None else ""
        elif ptype == "boolean":
            logits = model.get_logits_from_input_ids(context_ids)
    
            chosen = None
            for tid, tok in token_to_str.items():
                if tok.strip().lower() in ("true", "false"):
                    chosen = tid
                    break
            if chosen is None and logits:
                chosen = max(range(len(logits)), key=lambda i: logits[i])
            if chosen is not None:
                context_ids.append(chosen)
                v = token_to_str.get(chosen, model.decode([chosen]) if hasattr(model, 'decode') else '')
                parameters[pname] = True if str(v).strip().lower() == 'true' else False
            else:
                parameters[pname] = False
        else:
    
            logits = model.get_logits_from_input_ids(context_ids)
            if logits:
                tid = max(range(len(logits)), key=lambda i: logits[i])
                context_ids.append(tid)
                parameters[pname] = token_to_str.get(tid, model.decode([tid]) if hasattr(model, 'decode') else '')
            else:
                parameters[pname] = None


        if idx != len(param_items) - 1:
            comma_ids = model.encode(", ")[0].tolist()
            context_ids += comma_ids

    closing_ids = model.encode("}}")[0].tolist()
    context_ids += closing_ids

    result = {
        "prompt": user_prompt,
        "name": selected_name,
        "parameters": parameters
    }

    out_dir = os.path.join("data", "output")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "function_calling_results.json")
    try:
        with open(out_path, "w") as outf:
            json.dump([result], outf, indent=4)
        print(f"Wrote output to {out_path}")
    except Exception as e:
        print(f"Failed writing output: {e}")
    print("Result:", json.dumps(result, indent=4))
if __name__ == "__main__":
    main()