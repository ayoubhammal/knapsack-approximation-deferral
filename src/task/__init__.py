from typing import Callable
import random
import functools

import datasets

from .processing import chat_template_processor, DEFAULT_TEMPLATE


def load_task(task_config: dict, tokenizer) -> list[dict]:
    path: str = task_config["path"]
    split: str = task_config["split"]
    kwargs: dict = task_config["kwargs"]
    use_fewshots: int = task_config["use_fewshots"]
    n_instances: int = task_config.get("n_instances", 0)
    proc_fn: Callable = task_config["proc_fn"]
    prompt_template: str = task_config.get("prompt_template", DEFAULT_TEMPLATE)

    subjects: list[str] | None = task_config.get("subjects", None)

    if subjects is None:
        ds: list[dict] = [
            {
                k: v
                for k, v in datum.items()
            }
            for datum in datasets.load_dataset(path, split=split, **kwargs)
        ]
    else:
        ds: list[dict] = []
        for subject in subjects:
            ds.extend(
                [
                    {
                        "subject": subject,
                        **{
                            k: v
                            for k, v in datum.items()
                        }
                    }
                    for datum in datasets.load_dataset(path, subject, split=split, **kwargs)
                ]
            )

    if n_instances > 0:
        n_instances = min(n_instances, len(ds))
        ds = random.sample(ds, n_instances)

    ds_chat_template: list[dict] = list(
        map(
            functools.partial(
                proc_fn,
                prompt_template=prompt_template,
                use_fewshots=use_fewshots,
            ),
            ds
        )
    )

    ds_processed: list[dict] = list(
        map(
            functools.partial(
                chat_template_processor,
                tokenizer=tokenizer,
            ),
            ds_chat_template
        )
    )

    return ds_processed
