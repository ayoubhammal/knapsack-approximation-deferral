import os
import json
import argparse
import pathlib
import copy
import logging

import torch
torch.backends.cuda.matmul.allow_tf32 = True
import torch.multiprocessing as mp

from .configs.models import MODEL_CONFIGS
from .configs.tasks import TASK_CONFIGS
from .evaluation import evaluation
from .utils import initialize_seed

logger = logging.getLogger("evaluation")
logger.setLevel(logging.DEBUG)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

logger.addHandler(stream_handler)


parser = argparse.ArgumentParser()

parser.add_argument("--task", nargs="+", default=None)
parser.add_argument("--model", type=str, default=None)
parser.add_argument("--output-dir", type=pathlib.Path, default=None)

parser.add_argument("--list-tasks", action="store_true", default=False)

parser.add_argument("--temperature", type=float, default=None)
parser.add_argument("--deferral-rule", type=str, default=None)
parser.add_argument("--window-size", type=int, default=None)
parser.add_argument("--alpha", type=float, default=None)
parser.add_argument("--n-instances", type=int, default=None)
parser.add_argument("--n-samples", type=int, default=1)
parser.add_argument("--n-workers", type=int, default=1)
parser.add_argument("--dtype", type=str, default="bfloat16")
parser.add_argument("--seed", type=int, default=0)


def launch_evaluation(args_dict: dict) -> None:
    # model configs
    model_alias = args_dict["model"]
    if model_alias not in MODEL_CONFIGS:
        raise ValueError(f"Model {model_alias} not found.")
    args_dict["model_config"] = copy.deepcopy(MODEL_CONFIGS[model_alias])
    if "metadata" not in args_dict["model_config"]:
        args_dict["model_config"]["metadata"] = {}
    args_dict["model_config"]["metadata"]["alias"] = model_alias
    args_dict["model_config"]["dtype"] = args_dict["dtype"]

    # task configs
    tasks: list[str] = args_dict["task"]
    args_dict["task_configs"] = []
    for task in tasks:
        if task not in TASK_CONFIGS:
            raise ValueError(f"Task {task} not found.")

        task_config = copy.deepcopy(TASK_CONFIGS[task])
        if "metadata" not in task_config:
            task_config["metadata"] = {}
        task_config["metadata"]["alias"] = task

        for k in ("n_instances", "temperature", "window_size", "alpha", "deferral_rule"):
            if args_dict.get(k) is not None:
                task_config[k] = args_dict[k]

        task_config["n_samples"] = args_dict["n_samples"]

        args_dict["task_configs"].append(task_config)

    if not args_dict['output_dir'].exists():
        args_dict['output_dir'].mkdir()

    # launching the evaluation
    logger.info("Running evaluation on")
    logger.info(f"  * model: {args_dict['model']}")
    logger.info(f"  * tasks: {args_dict['task']}")

    logger.info("Model config:\n{}".format(json.dumps(args_dict["model_config"], indent=2, default=str)))
    logger.info("Tasks config:\n{}".format(json.dumps(args_dict["task_configs"], indent=2, default=str)))

    logger.info(f"Outputs stored under {args_dict['output_dir']}")
    logger.info(f"Using {args_dict['n_workers']} worker(s)")

    if args_dict["seed"] > 0:
        logger.info(f"Setting seed {args_dict["seed"]}.")
        initialize_seed(args_dict["seed"])
    else:
        logger.info("No seed is set.")

    evaluation(args_dict)

def main():
    args = parser.parse_args()
    args_dict = vars(args)
    if args_dict["list_tasks"]:
        output = "\n---- TASKS ----\n"
        for task in TASK_CONFIGS:
            output += f"  * {task}\n"
        logger.info(output)
        return
    else:
        assert (
            args_dict.get("task", None) is not None and
            args_dict.get("model", None) is not None and
            args_dict.get("output_dir", None) is not None
        )

    assert torch.cuda.is_available() and torch.cuda.device_count() > 0, "Requires 1 or more cuda gpus."
    mp.set_start_method("spawn")
    # environment variables
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    launch_evaluation(args_dict)

if __name__ == "__main__":
    main()
