import collections

import torch
import transformers
from transformers.cache_utils import HybridCache

from .utils import (
    init_cache,
    crop_cache,
    get_cache_length,
)

DEBUG = False

@torch.no_grad()
def standard_generate(target_model, inputs, eos_stopping_criteria, max_new_tokens, temperature, past_key_values=None):
    past_key_values = None if past_key_values is None else past_key_values
    
    past_key_values = init_cache(target_model, past_key_values, inputs, max_new_tokens)

    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    generated_ids = inputs["input_ids"]

    cache_length = get_cache_length(past_key_values)
    inputs["input_ids"] = inputs["input_ids"][:, cache_length:]
    
    probs = None
    
    for _ in range(max_new_tokens):
        if DEBUG:
            print("inptut size:", inputs["input_ids"].size())
            print("attention_mask size:", inputs["attention_mask"].size())
        output = target_model(
            **inputs,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )
        if DEBUG:
            print("logits size:", output.logits.size())
        past_key_values = output.past_key_values
        logits = output.logits
        probs = torch.softmax(logits / (temperature + 1e-7), dim=-1, dtype=torch.float32) if probs is None else torch.cat((probs, torch.softmax(logits[:, -1:, :] / (temperature + 1e-7), dim=-1, dtype=torch.float32)), dim=1)
        
        next_token_probs = probs[:, -1]
        next_token_ids = torch.distributions.categorical.Categorical(probs=next_token_probs, validate_args=False).sample().reshape(1, 1)
        
        generated_ids = torch.cat(
            (generated_ids, next_token_ids),
            dim=-1
        )

        if eos_stopping_criteria(next_token_ids, None).item():
            break
        
        attention_mask = inputs["attention_mask"]
        attention_mask = torch.cat([attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1)
        inputs = {"input_ids": next_token_ids, "attention_mask": attention_mask}

    assert probs is not None

    return {"generated_ids": generated_ids.to("cuda"), "probs": probs.to("cuda"), "past_key_values": past_key_values}, {"n_accept": [1], "window_size": 1}

@torch.no_grad()
def speculative_aligned_generate_from_qs(target_reference_model, draft_reference_model, draft_aligned_model, inputs, eos_stopping_criteria, max_new_tokens, window_size, temperature):
    draft_reference_past_key_values = None
    draft_aligned_past_key_values = None
    target_reference_past_key_values = None

    target_reference_past_key_values = init_cache(target_reference_model, target_reference_past_key_values, inputs, max_new_tokens)
    draft_reference_past_key_values = init_cache(draft_reference_model, draft_reference_past_key_values, inputs, max_new_tokens)
    draft_aligned_past_key_values = init_cache(draft_aligned_model, draft_aligned_past_key_values, inputs, max_new_tokens)

    stats = {
        "n_accept": [],
        "window_size": window_size,
    }
    
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    generated_ids = inputs["input_ids"]

    generation_cpt = 0
    while generation_cpt < max_new_tokens:
        # draft aligned generation
        prefix_length = inputs["input_ids"].size(1) - get_cache_length(draft_aligned_past_key_values) - 1
        draft_aligned_outputs, _ = standard_generate(
            draft_aligned_model,
            inputs,
            eos_stopping_criteria,
            max_new_tokens=window_size,
            temperature=temperature,
            past_key_values=draft_aligned_past_key_values,
        )
        draft_aligned_input_ids, draft_aligned_probs, draft_aligned_past_key_values = draft_aligned_outputs["generated_ids"].to("cuda"), draft_aligned_outputs["probs"][:, prefix_length:].to("cuda"), draft_aligned_outputs["past_key_values"]
        draft_length = draft_aligned_input_ids.size(1) - inputs["input_ids"].size(1)

        if DEBUG:
            print("inputs:", inputs["input_ids"].shape)
            print("draft_aligned_input_ids:", draft_aligned_input_ids.shape)
            print("prefix_length:", prefix_length)
            print("draft_length:", draft_length)
            print("draft_aligned_probs:", draft_aligned_probs.shape)

        # draft reference conditional probs
        draft_reference_input_ids = draft_aligned_input_ids[:, get_cache_length(draft_reference_past_key_values):-1].to("cuda")
        draft_reference_attention_mask = torch.ones_like(draft_aligned_input_ids[:, :-1]).to("cuda")
        draft_reference_output = draft_reference_model(
            input_ids=draft_reference_input_ids,
            attention_mask=draft_reference_attention_mask,
            past_key_values=draft_reference_past_key_values,
            use_cache=True,
            return_dict=True,
        )
        draft_reference_past_key_values = draft_reference_output.past_key_values
        draft_reference_probs = torch.softmax(draft_reference_output.logits[:, - draft_length:] / (temperature + 1e-7), dim=-1, dtype=torch.float32).to("cuda")

        if DEBUG:
            print("draft_reference_probs:", draft_reference_probs.shape)

        # conditional probs
        target_reference_input_ids = draft_aligned_input_ids[:, get_cache_length(target_reference_past_key_values):-1].to("cuda")
        target_reference_attention_mask = torch.ones_like(draft_aligned_input_ids[:, :-1]).to("cuda")
        target_reference_output = target_reference_model(
            input_ids=target_reference_input_ids,
            attention_mask=target_reference_attention_mask,
            past_key_values=target_reference_past_key_values,
            use_cache=True,
            return_dict=True,
        )
        target_reference_past_key_values = target_reference_output.past_key_values
        target_reference_probs = torch.softmax(target_reference_output.logits[:, - draft_length:] / (temperature + 1e-7), dim=-1, dtype=torch.float32).to("cuda")

        if DEBUG:
            print("target_reference_probs:", target_reference_probs.shape)

        # draft validation
        R = (
            target_reference_probs
            * draft_aligned_probs
            / draft_reference_probs
        )
        R_denominator = R.sum(dim=-1, keepdim=True)
        R_div_mask = R_denominator > 0
        target_aligned_probs = torch.where(R_div_mask, R / R_denominator, torch.full_like(R, 1 / R.shape[-1]))
        acceptation_probs = (
            target_aligned_probs / draft_aligned_probs
        ).gather(dim=-1, index=draft_aligned_input_ids[:, - draft_length:].unsqueeze(2)).squeeze(2).clamp(max=1)
        r = torch.rand_like(acceptation_probs)
        
         # no rejections
        if (r <= acceptation_probs).all():
            accepted_draft_ids = draft_aligned_input_ids[:, - draft_length:]
            generated_ids = torch.cat(
                (generated_ids.to("cuda"), accepted_draft_ids),
                dim=-1
            )
            generation_cpt += draft_length
            n_accepted = draft_length
            stats["n_accept"].append(n_accepted)
        else:
            first_reject_offset = (r <= acceptation_probs).float().argmin(dim=-1).item()
            if DEBUG:
                print("first_reject_offset:", first_reject_offset)
            accepted_draft_ids = draft_aligned_input_ids[:, - draft_length: - draft_length + first_reject_offset]
            
            draft_aligned_residual_probs = draft_aligned_probs[:, first_reject_offset]

            target_aligned_residual_probs = target_aligned_probs[:, first_reject_offset]

            residual_probs = (target_aligned_residual_probs - draft_aligned_residual_probs).clamp(min=0)
            residual_probs_denominator = residual_probs.sum(dim=-1, keepdim=True)
            residual_probs_div_mask = residual_probs_denominator > 0
            residual_probs = torch.where(residual_probs_div_mask, residual_probs / residual_probs_denominator, torch.full_like(residual_probs, 1 / residual_probs.shape[-1]))

            next_token_ids = torch.distributions.categorical.Categorical(probs=residual_probs, validate_args=False).sample().reshape(1, 1)

            generated_ids = torch.cat(
                (generated_ids.to("cuda"), accepted_draft_ids, next_token_ids),
                dim=-1
            )
            
            generation_cpt += first_reject_offset + 1

            # cache resizing
            if DEBUG:
                print("target_reference_past_key_values:", get_cache_length(target_reference_past_key_values))
                print("draft_reference_past_key_values:", get_cache_length(draft_reference_past_key_values))
                print("draft_aligned_past_key_values:", get_cache_length(draft_aligned_past_key_values))
            target_reference_past_key_values = crop_cache(target_reference_past_key_values, -(draft_length - 1 - first_reject_offset))
            draft_reference_past_key_values = crop_cache(draft_reference_past_key_values, -(draft_length - 1 - first_reject_offset))
            draft_aligned_past_key_values = crop_cache(draft_aligned_past_key_values, -(draft_length - 1 - first_reject_offset))
            if DEBUG:
                print("after target_reference_past_key_values:", get_cache_length(target_reference_past_key_values))
                print("after draft_reference_past_key_values:", get_cache_length(draft_reference_past_key_values))
                print("after draft_aligned_past_key_values:", get_cache_length(draft_aligned_past_key_values))
            n_accepted = first_reject_offset
            stats["n_accept"].append(n_accepted)
            
        if eos_stopping_criteria(generated_ids, None).item():
            break

        inputs = {
            "input_ids": generated_ids.to("cuda"),
            "attention_mask": torch.ones_like(generated_ids).to("cuda"),
        }
        
    return {"generated_ids": generated_ids.to("cuda"), "probs": None}, stats

@torch.no_grad()
def speculative_aligned_generate_from_q(target_reference_model, draft_reference_model, draft_aligned_model, inputs, eos_stopping_criteria, max_new_tokens, window_size, temperature):
    draft_reference_past_key_values = None
    draft_aligned_past_key_values = None
    target_reference_past_key_values = None

    target_reference_past_key_values = init_cache(target_reference_model, target_reference_past_key_values, inputs, max_new_tokens)
    draft_reference_past_key_values = init_cache(draft_reference_model, draft_reference_past_key_values, inputs, max_new_tokens)
    draft_aligned_past_key_values = init_cache(draft_aligned_model, draft_aligned_past_key_values, inputs, max_new_tokens)

    stats = {
        "n_accept": [],
        "window_size": window_size,
    }
    
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    generated_ids = inputs["input_ids"]

    generation_cpt = 0
    while generation_cpt < max_new_tokens:
        # draft aligned generation
        prefix_length = inputs["input_ids"].size(1) - get_cache_length(draft_reference_past_key_values) - 1
        draft_reference_outputs, _ = standard_generate(
            draft_reference_model,
            inputs,
            eos_stopping_criteria,
            temperature=temperature,
            max_new_tokens=window_size,
            past_key_values=draft_reference_past_key_values,
        )
        draft_reference_input_ids, draft_reference_probs, draft_reference_past_key_values = draft_reference_outputs["generated_ids"].to("cuda"), draft_reference_outputs["probs"][:, prefix_length:].to("cuda"), draft_reference_outputs["past_key_values"]
        draft_length = draft_reference_input_ids.size(1) - inputs["input_ids"].size(1)

        if DEBUG:
            print("inputs:", inputs["input_ids"].shape)
            print("draft_reference_input_ids:", draft_reference_input_ids.shape)
            print("prefix_length:", prefix_length)
            print("draft_length:", draft_length)
            print("draft_reference_probs:", draft_reference_probs.shape)

        # draft reference conditional probs
        draft_aligned_input_ids = draft_reference_input_ids[:, get_cache_length(draft_aligned_past_key_values):-1].to("cuda")
        draft_aligned_attention_mask = torch.ones_like(draft_reference_input_ids[:, :-1]).to("cuda")
        draft_aligned_output = draft_aligned_model(
            input_ids=draft_aligned_input_ids,
            attention_mask=draft_aligned_attention_mask,
            past_key_values=draft_aligned_past_key_values,
            use_cache=True,
            return_dict=True,
        )
        draft_aligned_past_key_values = draft_aligned_output.past_key_values
        draft_aligned_probs = torch.softmax(draft_aligned_output.logits[:, - draft_length:] / (temperature + 1e-7), dim=-1, dtype=torch.float32).to("cuda")

        if DEBUG:
            print("draft_aligned_probs:", draft_aligned_probs.shape)

        # conditional probs
        target_reference_input_ids = draft_reference_input_ids[:, get_cache_length(target_reference_past_key_values):-1].to("cuda")
        target_reference_attention_mask = torch.ones_like(draft_reference_input_ids[:, :-1]).to("cuda")
        target_reference_output = target_reference_model(
            input_ids=target_reference_input_ids,
            attention_mask=target_reference_attention_mask,
            past_key_values=target_reference_past_key_values,
            use_cache=True,
            return_dict=True,
        )
        target_reference_past_key_values = target_reference_output.past_key_values
        target_reference_probs = torch.softmax(target_reference_output.logits[:, - draft_length:] / (temperature + 1e-7), dim=-1, dtype=torch.float32).to("cuda")

        if DEBUG:
            print("target_reference_probs:", target_reference_probs.shape)

        # draft validation
        R = (
            target_reference_probs
            * draft_aligned_probs
            / draft_reference_probs
        )
        R_denominator = R.sum(dim=-1, keepdim=True)
        R_div_mask = R_denominator > 0
        target_aligned_probs = (
            torch.where(R_div_mask, R, 0) /
            torch.where(R_div_mask, R_denominator, 1)
        )
        acceptation_probs = (
            target_aligned_probs / draft_reference_probs
        ).gather(dim=-1, index=draft_reference_input_ids[:, - draft_length:].unsqueeze(2)).squeeze(2).clamp(max=1)
        r = torch.rand_like(acceptation_probs)
        
         # no rejections
        if (r <= acceptation_probs).all():
            accepted_draft_ids = draft_reference_input_ids[:, - draft_length:]
            generated_ids = torch.cat(
                (generated_ids.to("cuda"), accepted_draft_ids),
                dim=-1
            )
            generation_cpt += draft_length
            n_accepted = draft_length
            stats["n_accept"].append(n_accepted)
        else:
            first_reject_offset = (r <= acceptation_probs).float().argmin(dim=-1).item()
            if DEBUG:
                print("first_reject_offset:", first_reject_offset)
            accepted_draft_ids = draft_reference_input_ids[:, - draft_length: - draft_length + first_reject_offset]
            
            draft_reference_residual_probs = draft_reference_probs[:, first_reject_offset]

            target_aligned_residual_probs = target_aligned_probs[:, first_reject_offset]

            residual_probs = (target_aligned_residual_probs - draft_reference_residual_probs).clamp(min=0)
            residual_probs_denominator = residual_probs.sum(dim=-1, keepdim=True)
            residual_probs_div_mask = residual_probs_denominator > 0
            residual_probs = (
                torch.where(residual_probs_div_mask, residual_probs, 0) /
                torch.where(residual_probs_div_mask, residual_probs_denominator, 1)
            )

            next_token_ids = torch.distributions.categorical.Categorical(probs=residual_probs, validate_args=False).sample().reshape(1, 1)

            generated_ids = torch.cat(
                (generated_ids.to("cuda"), accepted_draft_ids, next_token_ids),
                dim=-1
            )
            
            generation_cpt += first_reject_offset + 1

            # cache resizing
            if DEBUG:
                print("target_reference_past_key_values:", get_cache_length(target_reference_past_key_values))
                print("draft_reference_past_key_values:", get_cache_length(draft_reference_past_key_values))
                print("draft_aligned_past_key_values:", get_cache_length(draft_aligned_past_key_values))
            target_reference_past_key_values = crop_cache(target_reference_past_key_values, -(draft_length - 1 - first_reject_offset))
            draft_reference_past_key_values = crop_cache(draft_reference_past_key_values, -(draft_length - 1 - first_reject_offset))
            draft_aligned_past_key_values = crop_cache(draft_aligned_past_key_values, -(draft_length - 1 - first_reject_offset))
            if DEBUG:
                print("after target_reference_past_key_values:", get_cache_length(target_reference_past_key_values))
                print("after draft_reference_past_key_values:", get_cache_length(draft_reference_past_key_values))
                print("after draft_aligned_past_key_values:", get_cache_length(draft_aligned_past_key_values))
            n_accepted = first_reject_offset
            stats["n_accept"].append(n_accepted)
            
        if eos_stopping_criteria(generated_ids, None).item():
            break

        inputs = {
            "input_ids": generated_ids.to("cuda"),
            "attention_mask": torch.ones_like(generated_ids).to("cuda"),
        }
        
    return {"generated_ids": generated_ids.to("cuda"), "probs": None}, stats

@torch.no_grad()
def speculative_generate(target_model, draft_model, inputs, eos_stopping_criteria, max_new_tokens, window_size, temperature):
    draft_past_key_values = None
    target_past_key_values = None

    target_past_key_values = init_cache(target_model, target_past_key_values, inputs, max_new_tokens)
    draft_past_key_values = init_cache(draft_model, draft_past_key_values, inputs, max_new_tokens)

    stats = {
        "n_accept": [],
        "window_size": window_size,
    }
    
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    generated_ids = inputs["input_ids"]
    probs = None

    generation_cpt = 0
    while generation_cpt < max_new_tokens:
        if DEBUG:
            print("input_ids size:", inputs["input_ids"].size())
        # draft generation
        prefix_length = inputs["input_ids"].size(1) - get_cache_length(draft_past_key_values) - 1
        draft_outputs, _ = standard_generate(
            draft_model,
            inputs,
            eos_stopping_criteria,
            max_new_tokens=window_size,
            temperature=temperature,
            past_key_values=draft_past_key_values,
        )
        draft_input_ids, draft_probs, draft_past_key_values = draft_outputs["generated_ids"].to("cuda"), draft_outputs["probs"][:, prefix_length:].to("cuda"), draft_outputs["past_key_values"]
        draft_length = draft_input_ids.size(1) - inputs["input_ids"].size(1)
        if DEBUG:
            print("draft_input_ids size:", draft_input_ids.size())
            print("draft_probs size:", draft_probs.size())
            print("draft_length:", draft_length)

        # conditional probs
        target_input_ids = draft_input_ids[:, get_cache_length(target_past_key_values):]
        target_attention_mask = torch.ones_like(draft_input_ids)
        if DEBUG:
            print("target_input_ids size:", target_input_ids.size())
        target_output = target_model(
            input_ids=target_input_ids,
            attention_mask=target_attention_mask,
            past_key_values=target_past_key_values,
            use_cache=True,
            return_dict=True,
        )
        target_past_key_values = target_output.past_key_values
        target_probs = torch.softmax(target_output.logits[:, - draft_length - 1:] / (temperature + 1e-7), dim=-1, dtype=torch.float32).to("cuda")
        if DEBUG:
            print("target_probs size:", target_probs.size())

        # draft validation
        acceptation_probs = (target_probs[:, - draft_length - 1: - 1] / draft_probs[:, -draft_length:]).gather(dim=-1, index=draft_input_ids[:, - draft_length:].unsqueeze(2)).squeeze(2).clamp(max=1)
        r = torch.rand_like(acceptation_probs)
        if DEBUG:
            print("acceptation_probs size:", acceptation_probs.size())
        
         # no rejections
        if (r <= acceptation_probs).all():
            if DEBUG:
                print("all draft accepted")

            accepted_draft_ids = draft_input_ids[:, - draft_length:]
            if eos_stopping_criteria(accepted_draft_ids, None).item():
                # eos generated already, the generation stops here
                generated_ids = torch.cat(
                    (generated_ids.to("cuda"), accepted_draft_ids),
                    dim=-1
                )
                generation_cpt += draft_length
                stats["n_accept"].append(draft_length)
                break
            else:
                next_token_probs = target_probs[:, - 1, :]
                next_token_ids = torch.distributions.categorical.Categorical(probs=next_token_probs, validate_args=False).sample().reshape(1, 1)
                generated_ids = torch.cat(
                    (generated_ids.to("cuda"), accepted_draft_ids, next_token_ids),
                    dim=-1
                )
                generation_cpt += draft_length + 1
                n_accepted = draft_length
                stats["n_accept"].append(n_accepted)
        else:
            first_reject_offset = (r <= acceptation_probs).float().argmin(dim=-1).item()
            accepted_draft_ids = draft_input_ids[:, - draft_length: - draft_length + first_reject_offset]
            
            if DEBUG:
                print("first_reject_offset:", first_reject_offset)
                print("accepted_draft_ids size:", accepted_draft_ids.size())
            
            target_residual_probs = target_probs[:, first_reject_offset]
            draft_residual_probs = draft_probs[:, first_reject_offset]

            residual_probs = (target_residual_probs - draft_residual_probs).clamp(min=0)
            residual_probs_denominator = residual_probs.sum(dim=-1, keepdim=True)
            residual_probs_div_mask = residual_probs_denominator > 0
            residual_probs = (
                torch.where(residual_probs_div_mask, residual_probs, 0) /
                torch.where(residual_probs_div_mask, residual_probs_denominator, 1)
            )

            next_token_ids = torch.distributions.categorical.Categorical(probs=residual_probs, validate_args=False).sample().reshape(1, 1)

            generated_ids = torch.cat(
                (generated_ids.to("cuda"), accepted_draft_ids, next_token_ids),
                dim=-1
            )
            
            generation_cpt += first_reject_offset + 1

            # cache resizing
            if DEBUG:
                print("target cache length:", get_cache_length(target_past_key_values))
                print("draft cache length:", get_cache_length(draft_past_key_values))
            target_past_key_values = crop_cache(target_past_key_values, -(draft_length - first_reject_offset))
            if draft_length - 1 - first_reject_offset > 0:
                draft_past_key_values = crop_cache(draft_past_key_values, -(draft_length - 1 - first_reject_offset))
            if DEBUG:
                print("target cache length:", get_cache_length(target_past_key_values))
                print("draft cache length:", get_cache_length(draft_past_key_values))
                print()
            n_accepted = first_reject_offset
            stats["n_accept"].append(n_accepted)
            
            if eos_stopping_criteria(generated_ids, None).item():
                break

        inputs = {
            "input_ids": generated_ids.to("cuda"),
            "attention_mask": torch.ones_like(generated_ids).to("cuda"),
        }
        
    return {"generated_ids": generated_ids.to("cuda"), "probs": None}, stats

