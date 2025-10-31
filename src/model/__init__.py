import os
import copy
import logging
import traceback
import gc
import time

import torch
import torch.multiprocessing as mp
import transformers

from .base import ComposedModel
from ..utils import initialize_seed
from .speculative_generation import standard_generate

logger = logging.getLogger("evaluation")

def load_model_mp(model_config: dict, worker_id:int, gpu_ids: list[int], request_queue: mp.Queue, response_queue: mp.Queue, seed: int) -> None:
    # initialization
    try:
        logger.info(f"WORKER {worker_id}: using gpu {gpu_ids}.")
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
        is_hf = model_config.get("is_hf", False)
        if is_hf:
            model = transformers.AutoModelForCausalLM.from_pretrained(
                model_config["model"],
                device_map="balanced" if torch.cuda.is_available() else None,
                local_files_only=True,
                torch_dtype=getattr(torch, model_config["dtype"]),
            )
            model.tokenizer = transformers.AutoTokenizer.from_pretrained(
                model_config["tokenizer"],
                local_files_only=True,
            )
            model.eos_stopping_criteria = transformers.generation.stopping_criteria.EosTokenCriteria(model.generation_config.eos_token_id)
        else:
            model = ComposedModel(model_config)
        model.eval()
        if seed > 0:
            logger.info(f"WORKER {worker_id}: Setting seed {seed}.")
            initialize_seed(seed)
        else:
            logger.info(f"WORKER {worker_id}: No seed is set.")
    except Exception as e:
        logger.exception(f"WORKER {worker_id}: Failed to initialize model on GPU {gpu_ids}: {e}")
        logger.exception(traceback.format_exc())
        response_queue.put(("Init Failure", gpu_ids))

        del model # type: ignore
        model = None
        gc.collect()
        torch.cuda.empty_cache()
        return

    logger.info(f"WORKER {worker_id}: ready.")
    response_queue.put(("OK", worker_id))

    # waiting for evaluation requests
    try:
        while True:
            request = request_queue.get()
            logger.info(f"WORKER {worker_id}: processing request.")
            if request == "TERMINATE":
                logger.info(f"WORKER {worker_id}: terminating.")
                break

            # parse the request
            request_id, instances, task_config = request
            if is_hf:
                results = hf_generate(model, task_config, instances, worker_id)
            else:
                results = model.evaluate(instances, task_config, worker_id)
            response_queue.put((results, request_id))
            logger.info(f"WORKER {worker_id}: sending response.")
    except Exception as e:
        logger.exception(f"WORKER {worker_id}: Failed to evalute on GPU {gpu_ids}: {e}")
        response_queue.put(("Eval Failure", gpu_ids))

        del model # type: ignore
        model = None
        gc.collect()
        torch.cuda.empty_cache()
        return

    del model # type: ignore
    model = None
    gc.collect()
    torch.cuda.empty_cache()


def hf_generate(model, task_config, instances, worker_id):
    results = []
    task_max_new_tokens = task_config["max_new_tokens"]
    temperature = task_config["temperature"]

    LOG_EVERY = 1
    for i_instance, instance in enumerate(instances):
        if i_instance % LOG_EVERY == 0:
            logger.info(f"WORKER {worker_id}: instance {i_instance + 1} / {len(instances)}")

        result_instance = copy.deepcopy(instance)
        inputs = {k: v.to("cuda") for k, v in model.tokenizer(instance["chat_template_prompt"], return_tensors="pt").items()}
        max_new_tokens = task_max_new_tokens
        generation_start_time = time.time()
        output = standard_generate(
            target_model=model,
            inputs=inputs,
            eos_stopping_criteria=model.eos_stopping_criteria,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )[0]["generated_ids"]
        generation_time = time.time() - generation_start_time

        generated_ids = output[0]
        generated_ids = generated_ids[inputs["input_ids"].shape[1]:].detach().cpu().tolist()
        generation = model.tokenizer.decode(generated_ids, skip_special_tokens=True)
        generation_debug = model.tokenizer.decode(generated_ids, skip_special_tokens=False)
        stats = {
            "time": generation_time,
            "n_accept": [1],
            "window_size": 1,
        }
        results.append(
            {
                **result_instance,
                "generated_answer": generation,
                "generated_answer_debug": generation_debug,
                "generated_ids": generated_ids,
                "n_tokens": len(generated_ids),
                **stats,
            }
        )
    return results
