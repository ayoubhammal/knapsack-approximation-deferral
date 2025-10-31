MODEL_CONFIGS: dict = {
    ##########" OLMo-2 ##############
    "olmo-2-1b": {
        "tokenizer": "allenai/OLMo-2-0425-1B-Instruct",
        "model": "allenai/OLMo-2-0425-1B",
        "metadata": {
            "alias": "olmo-2-1b",
        },
        "is_hf": True,
    },
    "olmo-2-1b-instruct": {
        "tokenizer": "allenai/OLMo-2-0425-1B-Instruct",
        "model": "allenai/OLMo-2-0425-1B-Instruct",
        "metadata": {
            "alias": "olmo-2-1b-instruct",
        },
        "is_hf": True,
    },
    "olmo-2-13b": {
        "tokenizer": "allenai/OLMo-2-1124-13B-Instruct",
        "model": "allenai/OLMo-2-1124-13B",
        "metadata": {
            "alias": "olmo-2-13b",
        },
        "is_hf": True,
    },
    "olmo-2-13b-instruct": {
        "tokenizer": "allenai/OLMo-2-1124-13B-Instruct",
        "model": "allenai/OLMo-2-1124-13B-Instruct",
        "metadata": {
            "alias": "olmo-2-13b-instruct",
        },
        "is_hf": True,
    },
    "olmo-2-1b-instruct-13b": {
        "tokenizer": "allenai/OLMo-2-0425-1B-Instruct",
        "small_reference": {
            "model": "allenai/OLMo-2-0425-1B",
            "metadata": {
                "alias": "olmo-2-1b",
            },
        },
        "small_aligned": {
            "model": "allenai/OLMo-2-0425-1B-Instruct",
            "metadata": {
                "alias": "olmo-2-1b-instruct",
            },
        },
        "big_reference": {
            "model": "allenai/OLMo-2-1124-13B",
            "metadata": {
                "alias": "olmo-2-13b",
            },
        },
        "big_aligned": {
            "model": "allenai/OLMo-2-1124-13B-Instruct",
            "metadata": {
                "alias": "olmo-2-13b-instruct",
            },
        },
    },

    ##########" Qwen3 ##############
    "qwen-3-1.7b-base": {
        "tokenizer": "Qwen/Qwen3-1.7B",
        "model": "Qwen/Qwen3-1.7B-Base",
        "metadata": {
            "alias": "qwen-3-1.7b-base",
        },
        "is_hf": True,
    },
    "qwen-3-1.7b-it": {
        "tokenizer": "Qwen/Qwen3-1.7B",
        "model": "Qwen/Qwen3-1.7B",
        "metadata": {
            "alias": "qwen-3-1.7b-it",
        },
        "is_hf": True,
    },
    "qwen-3-14b-base": {
        "tokenizer": "Qwen/Qwen3-14B",
        "model": "Qwen/Qwen3-14B-Base",
        "metadata": {
            "alias": "qwen-3-14b-base",
        },
        "is_hf": True,
    },
    "qwen-3-14b-it": {
        "tokenizer": "Qwen/Qwen3-14B",
        "model": "Qwen/Qwen3-14B",
        "metadata": {
            "alias": "qwen-3-14b-it",
        },
        "is_hf": True,
    },
    "qwen-3-1.7b-base-14b-it": {
        "tokenizer": "Qwen/Qwen3-1.7B",
        "small_reference": {
            "model": "Qwen/Qwen3-1.7B-Base",
            "metadata": {
                "alias": "qwen-3-1.7b-base",
            },
        },
        "small_aligned": {
            "model": "Qwen/Qwen3-1.7B",
            "metadata": {
                "alias": "qwen-3-1.7b-it",
            },
        },
        "big_reference": {
            "model": "Qwen/Qwen3-14B-Base",
            "metadata": {
                "alias": "qwen-3-14b-base",
            },
        },
        "big_aligned": {
            "model": "Qwen/Qwen3-14B",
            "metadata": {
                "alias": "qwen-3-14b-it",
            },
        },
    },
}
