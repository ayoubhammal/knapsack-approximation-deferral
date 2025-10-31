import transformers
from transformers.cache_utils import HybridCache

import src.model.base as model_base

def init_cache(model, kv_cache, inputs, max_new_tokens):
    if kv_cache is not None:
        return kv_cache
    # if isinstance(model, model_base.ComposedModel):
    #     return {
    #         "big_reference": init_cache(model.big_reference_model, kv_cache if kv_cache is not None else kv_cache["big_reference"], inputs, max_new_tokens),
    #         "small_reference": init_cache(model.small_reference_model, kv_cache if kv_cache is not None else kv_cache["small_reference"], inputs, max_new_tokens),
    #         "small_aligned": init_cache(model.small_aligned_model, kv_cache if kv_cache is not None else kv_cache["small_aligned"], inputs, max_new_tokens),
    #     }
    # if hasattr(model.config, "cache_implementation") and model.config.cache_implementation == "hybrid" and kv_cache is None:
    #     return HybridCache(config=model.config, max_batch_size=1, max_cache_len=inputs["input_ids"].shape[-1] + max_new_tokens + 1)
    return None

def get_cache_length(kv_cache):
    if kv_cache is None:
        return 0
    elif isinstance(kv_cache, transformers.cache_utils.Cache):
        return kv_cache.get_seq_length()
    elif isinstance(kv_cache, tuple):
        return kv_cache[0][0].shape[2]
    elif isinstance(kv_cache, dict):
        return get_cache_length(next(iter(kv_cache.values())))
    else:
        raise Exception()

def crop_cache(kv_cache, length):
    if isinstance(kv_cache, transformers.cache_utils.Cache):
        kv_cache.crop(length)
        return kv_cache
    elif isinstance(kv_cache, tuple):
        dynamic_kv_cache = transformers.DynamicCache.from_legacy_cache(kv_cache)
        dynamic_kv_cache.crop(length)
        return dynamic_kv_cache.to_legacy_cache()
    elif isinstance(kv_cache, dict):
        return {
            k: crop_cache(v, length)
            for k, v in kv_cache.items()
        }
    else:
        raise Exception()
    
