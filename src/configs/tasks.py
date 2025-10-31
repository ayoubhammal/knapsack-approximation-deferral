import enum
import json

from ..task.processing import (
    process_gsm_datum,
    process_math_datum,
    process_svamp_datum,
    process_tqa_datum,
    process_arc_datum,
)
from ..task.eval import (
    ExactLastNumberEvaluator,
    ExactBoxedAnswerEvaluator,
    ExactLastOptionEvaluator,
    Throughput,
    AcceptationRate,
    AcceptanceLength,
    AcceptanceLengthProportions,
)

class GenerationMode(enum.StrEnum):
    AR = "ar"
    SPEC_Q = "spec_q"
    SPEC_QS = "spec_q*"
    CASCADE_SPEC_NUDGING = "cascade_spec_nudging"
    NUDGING = "nudging"


DATASET_CONFIGS: dict = {
    "openai/gsm8k": {
        "path": "openai/gsm8k",
        "split": "test",
        "kwargs": {
            "name": "socratic",
        },
        "proc_fn": process_gsm_datum,
    },
    "HuggingFaceH4/MATH-500": {
        "path": "HuggingFaceH4/MATH-500",
        "split": "test",
        "kwargs": {},
        "proc_fn": process_math_datum,
    },
    "truthfulqa/truthful_qa": {
        "path": "truthfulqa/truthful_qa",
        "split": "validation",
        "kwargs": {
            "name": "multiple_choice"
        },
        "proc_fn": process_tqa_datum,
    },
    "ChilleD/SVAMP": {
        "path": "ChilleD/SVAMP",
        "split": "test",
        "kwargs": {},
        "proc_fn": process_svamp_datum
    },
    "allenai/ai2_arc": {
        "path": "allenai/ai2_arc",
        "split": "test",
        "kwargs": {
            "name": "ARC-Challenge",
        },
        "proc_fn": process_arc_datum
    },
    "tau/commonsense_qa": {
        "path": "tau/commonsense_qa",
        "split": "validation",
        "kwargs": {},
        "proc_fn": process_arc_datum
    }
}

TASK_CONFIGS_WO_MODE: dict = {
    "gsm8k::zs::cot": {
        **DATASET_CONFIGS["openai/gsm8k"],
        "use_fewshots": False,
        "metrics": [
            {
                "name": "accuracy",
                "evaluator": ExactLastNumberEvaluator,
                "evaluator2result_keys": {
                    "labels": "answer",
                    "predictions": "generated_answer",
                },
                "unit": "",

            },
            {
                "name": "throughput",
                "evaluator": Throughput,
                "evaluator2result_keys": {
                    "labels": "n_tokens",
                    "predictions": "time",
                },
                "unit": "tok/s",
            },
            {
                "name": "acceptation_rate",
                "evaluator": AcceptationRate,
                "evaluator2result_keys": {
                    "labels": "window_size",
                    "predictions": "n_accept",
                },
                "unit": "",
            },
            {
                "name": "acceptance_lengths_rates",
                "evaluator": AcceptanceLengthProportions,
                "evaluator2result_keys": {
                    "labels": "window_size",
                    "predictions": "n_accept",
                },
                "unit": "",
            },
            {
                "name": "acceptance_length",
                "evaluator": AcceptanceLength,
                "evaluator2result_keys": {
                    "labels": "window_size",
                    "predictions": "n_accept",
                },
                "unit": "tok",
            },
        ],
    },
    "math500::zs::cot": {
        **DATASET_CONFIGS["HuggingFaceH4/MATH-500"],
        "use_fewshots": False,
        "prompt_template": "{prompt}\n\nPresent the answer in LaTex format: \\boxed{{Your answer}}",
        "metrics": [
            {
                "name": "accuracy",
                "evaluator": ExactBoxedAnswerEvaluator,
                "evaluator2result_keys": {
                    "labels": "answer",
                    "predictions": "generated_answer",
                },
                "unit": "",

            },
            {
                "name": "throughput",
                "evaluator": Throughput,
                "evaluator2result_keys": {
                    "labels": "n_tokens",
                    "predictions": "time",
                },
                "unit": "tok/s",
            },
            {
                "name": "acceptation_rate",
                "evaluator": AcceptationRate,
                "evaluator2result_keys": {
                    "labels": "window_size",
                    "predictions": "n_accept",
                },
                "unit": "",
            },
            {
                "name": "acceptance_lengths_rates",
                "evaluator": AcceptanceLengthProportions,
                "evaluator2result_keys": {
                    "labels": "window_size",
                    "predictions": "n_accept",
                },
                "unit": "",
            },
            {
                "name": "acceptance_length",
                "evaluator": AcceptanceLength,
                "evaluator2result_keys": {
                    "labels": "window_size",
                    "predictions": "n_accept",
                },
                "unit": "tok",
            },
        ],
    },
    "tqa::zs::cot": {
        **DATASET_CONFIGS["truthfulqa/truthful_qa"],
        "use_fewshots": False,
        "metrics": [
            {
                "name": "mc1",
                "evaluator": ExactLastOptionEvaluator,
                "evaluator2result_keys": {
                    "labels": "answer",
                    "predictions": "generated_answer",
                },
                "unit": "",

            },
            {
                "name": "throughput",
                "evaluator": Throughput,
                "evaluator2result_keys": {
                    "labels": "n_tokens",
                    "predictions": "time",
                },
                "unit": "tok/s",
            },
            {
                "name": "acceptation_rate",
                "evaluator": AcceptationRate,
                "evaluator2result_keys": {
                    "labels": "window_size",
                    "predictions": "n_accept",
                },
                "unit": "",
            },
            {
                "name": "acceptance_lengths_rates",
                "evaluator": AcceptanceLengthProportions,
                "evaluator2result_keys": {
                    "labels": "window_size",
                    "predictions": "n_accept",
                },
                "unit": "",
            },
            {
                "name": "acceptance_length",
                "evaluator": AcceptanceLength,
                "evaluator2result_keys": {
                    "labels": "window_size",
                    "predictions": "n_accept",
                },
                "unit": "tok",
            },
        ],
    },
    "arc::zs::cot": {
        **DATASET_CONFIGS["allenai/ai2_arc"],
        "use_fewshots": False,
        "metrics": [
            {
                "name": "accuracy",
                "evaluator": ExactLastOptionEvaluator,
                "evaluator2result_keys": {
                    "labels": "answer",
                    "predictions": "generated_answer",
                },
                "unit": "",

            },
            {
                "name": "throughput",
                "evaluator": Throughput,
                "evaluator2result_keys": {
                    "labels": "n_tokens",
                    "predictions": "time",
                },
                "unit": "tok/s",
            },
            {
                "name": "acceptation_rate",
                "evaluator": AcceptationRate,
                "evaluator2result_keys": {
                    "labels": "window_size",
                    "predictions": "n_accept",
                },
                "unit": "",
            },
            {
                "name": "acceptance_lengths_rates",
                "evaluator": AcceptanceLengthProportions,
                "evaluator2result_keys": {
                    "labels": "window_size",
                    "predictions": "n_accept",
                },
                "unit": "",
            },
            {
                "name": "acceptance_length",
                "evaluator": AcceptanceLength,
                "evaluator2result_keys": {
                    "labels": "window_size",
                    "predictions": "n_accept",
                },
                "unit": "tok",
            },
        ],
    },
    "csqa::zs::cot": {
        **DATASET_CONFIGS["tau/commonsense_qa"],
        "use_fewshots": False,
        "metrics": [
            {
                "name": "accuracy",
                "evaluator": ExactLastOptionEvaluator,
                "evaluator2result_keys": {
                    "labels": "answer",
                    "predictions": "generated_answer",
                },
                "unit": "",

            },
            {
                "name": "throughput",
                "evaluator": Throughput,
                "evaluator2result_keys": {
                    "labels": "n_tokens",
                    "predictions": "time",
                },
                "unit": "tok/s",
            },
            {
                "name": "acceptation_rate",
                "evaluator": AcceptationRate,
                "evaluator2result_keys": {
                    "labels": "window_size",
                    "predictions": "n_accept",
                },
                "unit": "",
            },
            {
                "name": "acceptance_lengths_rates",
                "evaluator": AcceptanceLengthProportions,
                "evaluator2result_keys": {
                    "labels": "window_size",
                    "predictions": "n_accept",
                },
                "unit": "",
            },
            {
                "name": "acceptance_length",
                "evaluator": AcceptanceLength,
                "evaluator2result_keys": {
                    "labels": "window_size",
                    "predictions": "n_accept",
                },
                "unit": "tok",
            },
        ],
    },
    "svamp::zs::cot": {
        **DATASET_CONFIGS["ChilleD/SVAMP"],
        "use_fewshots": False,
        "metrics": [
            {
                "name": "accuracy",
                "evaluator": ExactLastNumberEvaluator,
                "evaluator2result_keys": {
                    "labels": "answer",
                    "predictions": "generated_answer",
                },
                "unit": "",

            },
            {
                "name": "throughput",
                "evaluator": Throughput,
                "evaluator2result_keys": {
                    "labels": "n_tokens",
                    "predictions": "time",
                },
                "unit": "tok/s",
            },
            {
                "name": "acceptation_rate",
                "evaluator": AcceptationRate,
                "evaluator2result_keys": {
                    "labels": "window_size",
                    "predictions": "n_accept",
                },
                "unit": "",
            },
            {
                "name": "acceptance_lengths_rates",
                "evaluator": AcceptanceLengthProportions,
                "evaluator2result_keys": {
                    "labels": "window_size",
                    "predictions": "n_accept",
                },
                "unit": "",
            },
            {
                "name": "acceptance_length",
                "evaluator": AcceptanceLength,
                "evaluator2result_keys": {
                    "labels": "window_size",
                    "predictions": "n_accept",
                },
                "unit": "tok",
            },
        ],
    },
}

TASK_CONFIGS: dict = {
    f"{k}::{mode}": {
        **v,
        "mode": mode,
        "alpha": 0.4,
        "window_size": 8, # 3, 5, 7
        "temperature": 1.0,
        "max_new_tokens": 1024,
        "deferral_rule": "max",
    }
    for k, v in TASK_CONFIGS_WO_MODE.items()
    for mode in GenerationMode
}

if __name__ == "__main__":
    print(json.dumps(TASK_CONFIGS, indent=2))
