import copy
import numpy as np

DEFAULT_USER_PROMPT = "Question: {content}\n"
DEFAULT_AI_PROMPT = "Answer: {content}\n"
DEFAULT_TEMPLATE = "{prompt}"
DEFAULT_SYSTEM_PROMPT = "Answer the question by walking through the reasoning step by step."

from .fewshots import (
    GSM8K_EXEMPLARS,
    MATH_EXAMPLARS,
)

def chat_template_processor(entry: dict, tokenizer) -> dict:
    try:
        entry_ = copy.deepcopy(entry)
        chat_template = [
            {
                "role": "system",
                "content": DEFAULT_SYSTEM_PROMPT,
            }
        ] + entry_.pop("chat_template")
        return {
            **entry_,
            "chat_template": chat_template,
            "chat_template_prompt": tokenizer.apply_chat_template(
                chat_template[:-1],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            ),
            "answer": chat_template[-1]["content"]
        }
    except:
        entry_ = copy.deepcopy(entry)
        chat_template = entry_.pop("chat_template")
        return {
            **entry_,
            "chat_template": chat_template,
            "chat_template_prompt": tokenizer.apply_chat_template(
                chat_template[:-1],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            ),
            "answer": chat_template[-1]["content"]
        }

def process_gsm_datum(
    entry,
    prompt_template=DEFAULT_TEMPLATE,
    use_few_shot=False,
    **kwargs
) -> dict:

    if use_few_shot:
        gsm_messages = [
            [
                {
                    "role": "user",
                    "content": prompt_template.format(prompt=sample["question"]),
                },
                {"role": "assistant", "content": sample["cot_answer"]},
            ]
            for sample in GSM8K_EXEMPLARS
        ]
        # flatten
        gsm_messages = [item for sublist in gsm_messages for item in sublist]
    else:
        gsm_messages = []

    chat_template = gsm_messages + [
        {
            "role": "user",
            "content": prompt_template.format(prompt=entry["question"]),
        },
        {
            "role": "assistant",
            "content": entry["answer"],
        },
    ]

    return {
        "chat_template": chat_template,
    }


def process_math_datum(
    entry,
    prompt_template=DEFAULT_TEMPLATE,
    use_few_shot=False,
    **kwargs
) -> dict:

    if use_few_shot:
        math_messages = [
            [
                {
                    "role": "user",
                    "content": prompt_template.format(prompt=sample["question"]),
                },
                {"role": "assistant", "content": sample["cot_answer"]},
            ]
            for sample in MATH_EXAMPLARS
        ]
        # flatten
        math_messages = [item for sublist in math_messages for item in sublist]
    else:
        math_messages = []

    chat_template = math_messages + [
        {
            "role": "user",
            "content": prompt_template.format(prompt=entry["problem"]),
        },
        {
            "role": "assistant",
            "content": entry["solution"],
        },
    ]

    return {
        "chat_template": chat_template,
    }


def process_arc_datum(
    entry,
    prompt_template=DEFAULT_TEMPLATE,
    use_few_shot=False,
    **kwargs
):
    assert not use_few_shot

    instruction = f"Choose the correct answer to the following multiple-choice question.\n\n"

    prompt = entry["question"]
    choices_text = entry["choices"]["text"]
    choices_label = entry["choices"]["label"]
    answer_key = entry["answerKey"]

    instruction += "Question: {}\n\n".format(prompt)

    for j in range(len(choices_text)):
        instruction += "{}). {}\n".format(choices_label[j], choices_text[j])

    instruction += "\nProvide your reasoning about the answer and finish your answer with the letter corresponding to the correct option (e.g., A, B, C, or D).\n\n"

    prompt_ans = "\nAnswer:"

    # if include_answer:
    prompt_ans += " {}\n\n".format(answer_key)

    entry = {"prompt": instruction, "answer": prompt_ans}

    chat_template = [
        {
            "role": "user",
            "content": prompt_template.format(prompt=instruction),
        },
        {
            "role": "assistant",
            "content": prompt_ans,
        },
    ]

    return {
        "chat_template": chat_template,
    }

def process_tqa_datum(
    entry,
    prompt_template=DEFAULT_TEMPLATE,
    use_few_shot=False,
    **kwargs
):
    choices_key = "mc1_targets"

    choice_options = [chr(i) for i in range(ord("A"), ord("Z") + 1)]

    instruction = (
        "Choose the correct answer to the following multiple-choice question.\n\n"
    )

    prompt = entry["question"]
    choices = entry[choices_key]["choices"]
    answer = np.argmax(entry[choices_key]["labels"])

    # answer = int(entry["answer"])

    instruction += "Question: {}\n\n".format(prompt)

    for j in range(len(choices)):
        instruction += "{}). {}\n".format(choice_options[j], choices[j])

    instruction += "\nProvide your reasoning about the answer and finish your answer with the letter corresponding to the correct option (e.g., A, B, C, or D).\n\n"

    prompt_ans = "\nAnswer:"

    # if include_answer:
    prompt_ans += " {}\n\n".format(choice_options[answer])

    chat_template = [
        {
            "role": "user",
            "content": prompt_template.format(prompt=instruction),
        },
        {
            "role": "assistant",
            "content": prompt_ans,
        },
    ]
    return {
        "chat_template": chat_template,
        "labels": [label for flag, label in zip(entry[choices_key]["labels"], choice_options) if flag == 1]
    }

def process_svamp_datum(
    entry,
    prompt_template=DEFAULT_TEMPLATE,
    use_few_shot=False,
    **kwargs
):
    return {
        "chat_template": [
            {
                "role": "user",
                "content": prompt_template.format(prompt=entry["question_concat"]),
            },
            {
                "role": "assistant",
                "content": entry["Equation"] + " = " + entry["Answer"]
            },
        ]
    }

