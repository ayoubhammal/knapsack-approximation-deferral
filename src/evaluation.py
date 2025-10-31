import os
import copy
import json
import logging
import time

import torch
import torch.multiprocessing as mp
import transformers

from .utils import (
    get_gpu_memory,
)
from .task import load_task
from .model import load_model_mp

logger = logging.getLogger("evaluation")

def evaluation(args_dict: dict) -> None:
    task_configs = args_dict["task_configs"]
    model_config = args_dict["model_config"]
    n_workers = args_dict["n_workers"]
    output_dir = args_dict["output_dir"]
    seed = args_dict["seed"]

    # loading tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_config["tokenizer"],
        local_files_only=True,
    ) # type: ignore

    # loading model
    n_gpus = torch.cuda.device_count()
    cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    assert n_gpus % n_workers == 0, f"GPUs should be evenly distributed to processes, got {n_workers} workers and {n_gpus} GPUs."
    n_gpus_per_worker = n_gpus // n_workers

    request_queues: list[mp.Queue] = []
    response_queue: mp.Queue = mp.Queue()
    worker_processes: list[mp.Process] = []

    for i_worker in range(n_workers):
        worker_request_queue: mp.Queue = mp.Queue()
        request_queues.append(worker_request_queue)

        worker_gpu_ids = list(
            range(
                i_worker * n_gpus_per_worker,
                (i_worker + 1) * n_gpus_per_worker
            )
        )

        p = mp.Process(
            target=load_model_mp,
            args=(
                model_config,
                i_worker,
                worker_gpu_ids,
                worker_request_queue,
                response_queue,
                seed,
            ),
            # daemon=True
        )
        worker_processes.append(p)
        p.start()

    for _ in range(n_workers):
        checkup, worker_id = response_queue.get()
        if checkup == "OK":
            continue
        if checkup == "Init Failure" or checkup == "Eval Failure":
            logger.error(f"Exiting due to {checkup} on worker {worker_id}.")

            for i_process, p in enumerate(worker_processes):
                if p.is_alive():
                    p.terminate()
                logger.error(f"Terminated worker {i_process}.")

            os._exit(1)

    logger.info(f"Model loaded.")
    logger.info(f"GPU memory:\n{get_gpu_memory()}")

    task_metric_scores = {}
    for task_config in task_configs:
        task_start_time = time.time()
        # loading task data
        logger.info(f"--- Task: {task_config['metadata']['alias']} ---")
        instances = load_task(task_config, tokenizer)
        logger.info(f"Loaded {len(instances)} evaluation example(s)")
        logger.info(f"  First instance:\n{json.dumps(instances[0], indent=2)}")

        # we query workers with evaluation request and gather their responses
        instance_batches = [[] for _ in range(n_workers)]
        for i_instance, instance in enumerate(instances):
            instance_batches[i_instance % n_workers].extend([instance] * task_config["n_samples"])

        for i_worker in range(n_workers):
            request_queues[i_worker].put(
                (
                    i_worker,
                    instance_batches[i_worker],
                    copy.deepcopy(task_config)
                )
            )
        logger.info("Task distributed.")

        logger.info("Collecting results.")
        results_for_requests = []
        for _ in range(n_workers):
            result, id = response_queue.get()
            if result == "Init Failure" or result == "Eval Failure":
                logger.error(f"Exiting due to {result} on worker {id}.")

                for i_process, p in enumerate(worker_processes):
                    if p.is_alive():
                        p.terminate()
                    logger.error(f"Terminated worker {i_process}.")

                os._exit(1)
            results_for_requests.extend(result)

        # calculating metrics
        metric_scores = {}
        for metric in task_config["metrics"]:
            metric_name = metric["name"]
            evaluator = metric["evaluator"]()
            e2r_keys = metric["evaluator2result_keys"]
            metric_unit = metric["unit"]

            evaluator_meal = {
                k: []
                for k in e2r_keys
            }
            for result in results_for_requests:
                for k, v in e2r_keys.items():
                    evaluator_meal[k].append(result[v])

            evaluator.process_pairs(**evaluator_meal)

            metric_scores[metric_name] = (
                evaluator.aggregate_score(),
                metric_unit
            )

        metric_scores["total_time"] = (time.time() - task_start_time, "s")
        task_metric_scores[task_config["metadata"]["alias"]] = metric_scores
        logger.info("--- Metrics report ---")
        for k, (v, u) in metric_scores.items():
            logger.info(f"  * {k}: {v} {u}")

        # caching up predictions and metrics
        cached_data = {
            "metrics": metric_scores,
            **{
                k: [r[k] for r in results_for_requests]
                for k in ("chat_template_prompt", "answer", "generated_answer", "generated_answer_debug", "labels")
                if k in results_for_requests[0]
            }
        }
        with open(output_dir / f"{model_config['metadata']['alias']}-{task_config['metadata']['alias']}-results.json", "w") as cache_file:
            json.dump(cached_data, cache_file)

    # Finish all tasks, close mp
    for queue in request_queues:
        queue.put("TERMINATE")
    for p in worker_processes:
        p.join()

    output_str = ""
    for task_name, metric_scores in task_metric_scores.items():
        output_str += f"task: {task_name}\n"
        for k, (v, u) in metric_scores.items():
            output_str += f"  * {k}: {v} {u}\n"
    logger.info(f"Score report\n{output_str}")

    # Reset for future allocation
    if cuda_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices
    else:
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
