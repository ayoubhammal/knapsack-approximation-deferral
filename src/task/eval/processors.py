import re

def get_last_number(output: str) -> str:

    output = re.sub(r"(\d),(\d)", r"\1\2", output)
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", output)
    if numbers:
        return numbers[-1]
    else:
        return "NaN"


def remove_boxed(s: str) -> str:
    left = "\\boxed{"
    try:
        assert s[: len(left)] == left
        assert s[-1] == "}"
        answer = s[len(left): -1]
        if "=" in answer:
            answer = answer.split("=")[-1].lstrip(" ")
        return answer
    except Exception:
        return "NaN"


def last_boxed_only_string(output: str) -> str:
    idx = output.rfind("\\boxed")
    if idx < 0:
        idx = output.rfind("\\fbox")
        if idx < 0:
            return "NaN"
    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(output):
        if output[i] == "{":
            num_left_braces_open += 1
        if output[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = "NaN"
    else:
        retval = remove_boxed(output[idx: right_brace_idx + 1])

    return retval

def get_last_option(text: str) -> str:
    pattern = r"\b[A-J]\b(?!.*\b[A-J]\b)"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(0)
    else:
        return "NaN"
