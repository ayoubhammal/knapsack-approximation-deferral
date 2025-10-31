import copy
import time
from typing import Union, Tuple
import logging

import numpy as np
import torch
import transformers

from .speculative_generation import (
    standard_generate,
    speculative_aligned_generate_from_q,
    speculative_aligned_generate_from_qs,
)
from .cascade_generation import (
    cascade_spec_nudging_generate,
    nudging_generate,
)
from ..configs.tasks import (
    GenerationMode
)

logger = logging.getLogger("evaluation")

class ComposedModel(torch.nn.Module):

    def __init__(self, model_config):
        super().__init__()

        self.model_config = copy.deepcopy(model_config)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_config["tokenizer"],
            local_files_only=True
        ) # type: ignore
        self.generation_config = transformers.GenerationConfig.from_pretrained(model_config["tokenizer"])
        self.eos_stopping_criteria = transformers.generation.stopping_criteria.EosTokenCriteria(self.generation_config.eos_token_id)

        self.small_reference_model = None
        self.small_aligned_model = None
        self.big_reference_model = None

    def init_small_reference_model(self):
        self.small_reference_model = transformers.AutoModelForCausalLM.from_pretrained(
            self.model_config["small_reference"]["model"],
            device_map="balanced_low_0" if torch.cuda.is_available() else None,
            local_files_only=True,
            torch_dtype=getattr(torch, self.model_config["dtype"]),
        ) # type: ignore

    def init_small_aligned_model(self):
        self.small_aligned_model = transformers.AutoModelForCausalLM.from_pretrained(
            self.model_config["small_aligned"]["model"],
            device_map="balanced_low_0" if torch.cuda.is_available() else None,
            local_files_only=True,
            torch_dtype=getattr(torch, self.model_config["dtype"]),
        ) # type: ignore

    def init_big_reference_model(self):
        self.big_reference_model = transformers.AutoModelForCausalLM.from_pretrained(
            self.model_config["big_reference"]["model"],
            device_map="balanced_low_0" if torch.cuda.is_available() else None,
            local_files_only=True,
            torch_dtype=getattr(torch, self.model_config["dtype"]),
        ) # type: ignore

    # def init_big_aligned_model(self):
    #     self.big_alignd_model = transformers.AutoModelForCausalLM.from_pretrained(
    #         model_config["big_aligned"]["model"],
    #         device_map="balanced_low_0" if torch.cuda.is_available() else None,
    #         local_files_only=True,
    #         torch_dtype=getattr(torch, model_config["dtype"]),
    #     ) # type: ignore


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        past_key_values=None,
        return_dict=None,
        labels=None,
        **kwargs,
    ) -> Union[Tuple, transformers.modeling_outputs.CausalLMOutputWithPast]:
        if return_dict is None:
            return_dict = True

        big_reference_output = self.big_reference_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values["big_reference"] if past_key_values is not None else None,
            return_dict=return_dict,
            **kwargs,
        )
        small_reference_output = self.small_reference_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values["small_reference"] if past_key_values is not None else None,
            return_dict=return_dict,
            **kwargs,
        )
        small_aligned_output = self.small_aligned_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values["small_aligned"] if past_key_values is not None else None,
            return_dict=return_dict,
            **kwargs,
        )

        big_reference_probs = torch.softmax(big_reference_output.logits, dim=-1).to("cuda")
        small_reference_probs = torch.softmax(small_reference_output.logits, dim=-1).to("cuda")
        small_aligned_probs = torch.softmax(small_aligned_output.logits, dim=-1).to("cuda")

        R = (big_reference_probs * small_aligned_probs / small_reference_probs).sum(dim=-1, keepdim=True)

        logits = torch.log(big_reference_probs * small_aligned_probs / (small_reference_probs * R))

        past_key_values = {
            "big_reference": big_reference_output.past_key_values,
            "small_reference": small_reference_output.past_key_values,
            "small_aligned": small_aligned_output.past_key_values,
        }
        hidden_states = {
            "big_reference": big_reference_output.hidden_states,
            "small_reference": small_reference_output.hidden_states,
            "small_aligned": small_aligned_output.hidden_states,
        }
        attentions = {
            "big_reference": big_reference_output.attentions,
            "small_reference": small_reference_output.attentions,
            "small_aligned": small_aligned_output.attentions,
        }
    
        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            loss = self.big_reference_model.loss_function(
                logits,
                labels,
                vocab_size=self.big_reference_model.config.vocab_size,
                **kwargs,
            )

        if not return_dict:
            output = (logits, past_key_values, hidden_states, attentions)
            return (loss,) + output if loss is not None else output

        return transformers.modeling_outputs.CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values,
            hidden_states=hidden_states,
            attentions=attentions,
        )

    @torch.no_grad()
    def generate(
        self,
        inputs,
        mode,
        **kwargs
    ):
        # TODO: set better defaults
        if mode == GenerationMode.AR:
            output = standard_generate(
                target_model=self,
                inputs=inputs,
                eos_stopping_criteria=self.eos_stopping_criteria,
                max_new_tokens=kwargs["max_new_tokens"],
                temperature=kwargs["temperature"],
            )
        elif mode == GenerationMode.SPEC_Q:
            output = speculative_aligned_generate_from_q(
                target_reference_model=self.big_reference_model,
                draft_reference_model=self.small_reference_model,
                draft_aligned_model=self.small_aligned_model,
                inputs=inputs,
                eos_stopping_criteria=self.eos_stopping_criteria,
                max_new_tokens=kwargs["max_new_tokens"],
                window_size=kwargs["window_size"],
                temperature=kwargs["temperature"],
            )
        elif mode == GenerationMode.SPEC_QS:
            output = speculative_aligned_generate_from_qs(
                target_reference_model=self.big_reference_model,
                draft_reference_model=self.small_reference_model,
                draft_aligned_model=self.small_aligned_model,
                inputs=inputs,
                eos_stopping_criteria=self.eos_stopping_criteria,
                max_new_tokens=kwargs["max_new_tokens"],
                window_size=kwargs["window_size"],
                temperature=kwargs["temperature"],
            )
        elif mode == GenerationMode.CASCADE_SPEC_NUDGING:
            output = cascade_spec_nudging_generate(
                target_reference_model=self.big_reference_model,
                draft_aligned_model=self.small_aligned_model,
                inputs=inputs,
                eos_stopping_criteria=self.eos_stopping_criteria,
                max_new_tokens=kwargs["max_new_tokens"],
                window_size=kwargs["window_size"],
                temperature=kwargs["temperature"],
                alpha=kwargs["alpha"],
                rule=kwargs["deferral_rule"],
            )
        elif mode == GenerationMode.NUDGING:
            output = nudging_generate(
                target_reference_model=self.big_reference_model,
                draft_aligned_model=self.small_aligned_model,
                inputs=inputs,
                eos_stopping_criteria=self.eos_stopping_criteria,
                max_new_tokens=kwargs["max_new_tokens"],
                window_size=kwargs["window_size"],
                temperature=kwargs["temperature"],
                alpha=kwargs["alpha"],
                rule=kwargs["deferral_rule"],
            )
        else:
            raise ValueError(f"Unrecognized generation mode `{mode}`")

        return output

    def load_models(self, mode):
        if mode == GenerationMode.AR:
            self.init_small_reference_model()
            self.init_small_aligned_model()
            self.init_big_reference_model()
        elif mode == GenerationMode.SPEC_Q:
            self.init_small_reference_model()
            self.init_small_aligned_model()
            self.init_big_reference_model()
        elif mode == GenerationMode.SPEC_QS:
            self.init_small_reference_model()
            self.init_small_aligned_model()
            self.init_big_reference_model()
        elif mode == GenerationMode.CASCADE_SPEC_NUDGING:
            self.init_small_aligned_model()
            self.init_big_reference_model()
        elif mode == GenerationMode.NUDGING:
            self.init_small_aligned_model()
            self.init_big_reference_model()
        elif mode == GenerationMode.CASCADE_SPEC_QS:
            self.init_small_aligned_model()
            self.init_big_reference_model()
        elif mode == GenerationMode.RSPEC_QS:
            self.init_small_reference_model()
            self.init_small_aligned_model()
            self.init_big_reference_model()
        else:
            raise ValueError(f"Unrecognized generation mode `{mode}`")

    def unload_models(self, mode):
        if mode == GenerationMode.AR:
            del self.small_reference_model, self.small_aligned_model, self.big_reference_model
        elif mode == GenerationMode.SPEC_Q:
            del self.small_reference_model, self.small_aligned_model, self.big_reference_model
        elif mode == GenerationMode.SPEC_QS:
            del self.small_reference_model, self.small_aligned_model, self.big_reference_model
        elif mode == GenerationMode.CASCADE_SPEC_NUDGING:
            del self.small_aligned_model, self.big_reference_model
        elif mode == GenerationMode.NUDGING:
            del self.small_aligned_model, self.big_reference_model
        elif mode == GenerationMode.CASCADE_SPEC_QS:
            del self.small_aligned_model, self.big_reference_model
        elif mode == GenerationMode.RSPEC_QS:
            del self.small_reference_model, self.small_aligned_model, self.big_reference_model
        else:
            raise ValueError(f"Unrecognized generation mode `{mode}`")

    def evaluate(
        self,
        instances,
        task_config,
        worker_id,
    ):
        mode = task_config["mode"]
        task_max_new_tokens = task_config["max_new_tokens"]
        temperature = task_config["temperature"]
        window_size = task_config["window_size"]
        alpha = task_config["alpha"]
        deferral_rule = task_config["deferral_rule"]

        LOG_EVERY = 1

        self.load_models(mode)

        outputs = []
        for i_instance, instance in enumerate(instances):
            if i_instance % LOG_EVERY == 0:
                logger.info(f"WORKER {worker_id}: instance {i_instance + 1} / {len(instances)}")
            result_instance = copy.deepcopy(instance)

            inputs = self.tokenizer(instance["chat_template_prompt"], return_tensors="pt")
            max_new_tokens = task_max_new_tokens
            generation_start_time = time.time()
            output = self.generate(
                inputs=inputs,
                mode=mode,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                window_size=window_size,
                alpha=alpha,
                deferral_rule=deferral_rule,
            )
            generation_time = time.time() - generation_start_time

            generated_ids = output[0]["generated_ids"][0]
            generated_ids = generated_ids[inputs["input_ids"].shape[1]:].detach().cpu().tolist()
            generation = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            generation_debug = self.tokenizer.decode(generated_ids, skip_special_tokens=False)
            probs = output[0]["probs"].detach().cpu().tolist() if output[0]["probs"] is not None else None
            stats = {
                "time": generation_time,
                **output[1],
            }
            outputs.append(
                {
                    **result_instance,
                    "generated_answer": generation,
                    "generated_answer_debug": generation_debug,
                    "generated_ids": generated_ids,
                    "n_tokens": len(generated_ids),
                    **stats,
                }
            )
        self.unload_models(mode)

        return outputs
