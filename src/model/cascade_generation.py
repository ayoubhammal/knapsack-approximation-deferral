from fractions import Fraction
import enum
import numpy as np
import torch

from .speculative_generation import standard_generate, DEBUG, get_cache_length, crop_cache, init_cache

class DeferralRule(enum.StrEnum):
    TOKEN_WISE = "token_wise"
    MAX = "max"
    MAXQMINUS = "max_q_minus"
    MAXQPLUS = "max_q_plus"
    MAXPPLUS = "max_p_plus"
    MINUSMAXPPLUS = "minus_max_p_plus"
    TOKENV2 = "token_v2"
    TOKEN_MAX = "token_max"
    BUDGET_LR = "budget_lr"

def lcm(values):
    denoms = [Fraction(x).denominator for x in values]
    return np.lcm.reduce(denoms)

def nudging_deferral_logic(p, q, alpha, rule):
    if rule == DeferralRule.MAX:
        return (p.max(dim=-1, keepdim=True).values < alpha).float()
    if rule == DeferralRule.TOKEN_WISE:
        return (p < alpha).float()
    if rule == DeferralRule.MAXQMINUS:
        if q is None:
            return None
        return (p < q.max(dim=-1, keepdim=True).values - alpha).float()
    if rule == DeferralRule.MAXQPLUS:
        if q is None:
            return None
        return (p < q.max(dim=-1, keepdim=True).values + alpha).float()
    if rule == DeferralRule.MAXPPLUS:
        # q(y) > max(p(y)) + alpha
        if q is None:
            return None
        return (q < p.max(dim=-1, keepdim=True).values + alpha).float()
    if rule == DeferralRule.MINUSMAXPPLUS:
        # q(y) > max(p(y)) + alpha
        if q is None:
            return None
        return (q > p.max(dim=-1, keepdim=True).values + alpha).float()
    if rule == DeferralRule.TOKENV2:
        return (p < p.max(dim=-1, keepdim=True).values - alpha).float()
    if rule == DeferralRule.TOKEN_MAX:
        return (p < p.max(dim=-1, keepdim=True).values).float()
    if rule == DeferralRule.BUDGET_LR:
        weights = p.view(-1) # plugin estimator de P
        densities = - torch.log(weights) # - P log(p) / P = - log(p) 
        sorted_ids = torch.argsort(densities, descending=True)
        sorted_weights = weights[sorted_ids]

        sorted_weights_cumsum = torch.cumsum(sorted_weights, dim=-1)
        test = sorted_weights_cumsum <= alpha
        if test.all():
            return torch.ones_like(p).to(p.device)
        critical_sorted_id = test.float().argmin()
        d = torch.zeros_like(sorted_ids)
        d[sorted_ids[:critical_sorted_id]] = 1
        return d.view_as(p).to(p.device)
    else:
        raise Exception()

@torch.no_grad()
def nudging_generate(target_reference_model, draft_aligned_model, inputs, eos_stopping_criteria, max_new_tokens, window_size, temperature, alpha, rule):
    target_reference_past_key_values = None
    draft_aligned_past_key_values = None

    target_reference_past_key_values = init_cache(target_reference_model, target_reference_past_key_values, inputs, max_new_tokens)
    draft_aligned_past_key_values = init_cache(draft_aligned_model, draft_aligned_past_key_values, inputs, max_new_tokens)

    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    generated_ids = inputs["input_ids"]

    cache_length = get_cache_length(target_reference_past_key_values)
    inputs["input_ids"] = inputs["input_ids"][:, cache_length:]
    
    for _ in range(max_new_tokens):
        if DEBUG:
            print("inptut size:", inputs["input_ids"].size())
            print("attention_mask size:", inputs["attention_mask"].size())
        target_reference_output = target_reference_model(
            **inputs,
            past_key_values=target_reference_past_key_values,
            use_cache=True,
            return_dict=True,
        )
        if DEBUG:
            print("logits size:", target_reference_output.logits.size())
        target_reference_past_key_values = target_reference_output.past_key_values
        target_reference_probs = torch.softmax(target_reference_output.logits / (temperature + 1e-7), dim=-1, dtype=torch.float32)
        
        next_token_probs = target_reference_probs[:, -1]

        deferral = nudging_deferral_logic(next_token_probs, None, alpha, rule)
        if DEBUG:
            print("deferral size:", deferral.size())
            print("next_token_probs size:", next_token_probs.size())
        if deferral is None or (deferral == 1).any().item():
            draft_aligned_inputs = {
                "input_ids": generated_ids[:, get_cache_length(draft_aligned_past_key_values):],
                "attention_mask": inputs["attention_mask"]
            }
            draft_aligned_output = draft_aligned_model(
                **draft_aligned_inputs,
                past_key_values=draft_aligned_past_key_values,
                use_cache=True,
                return_dict=True,
            )
            if DEBUG:
                print("logits size:", draft_aligned_output.logits.size())
            draft_aligned_past_key_values = draft_aligned_output.past_key_values
            draft_aligned_probs = torch.softmax(draft_aligned_output.logits / (temperature + 1e-7), dim=-1, dtype=torch.float32)

            if deferral is None:
                deferral = nudging_deferral_logic(next_token_probs, draft_aligned_probs[:, -1], alpha, rule)
                assert deferral is not None

            next_token_probs = (1 - deferral) * next_token_probs + draft_aligned_probs[:, -1] * (deferral * next_token_probs).sum(dim=-1, keepdims=True)

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


    return {"generated_ids": generated_ids.to("cuda"), "probs": None}, {"n_accept": [1], "window_size": 1}


@torch.no_grad()
def cascade_spec_nudging_generate(target_reference_model, draft_aligned_model, inputs, eos_stopping_criteria, max_new_tokens, window_size, temperature, alpha, rule):
    target_reference_past_key_values = None
    draft_aligned_past_key_values = None

    target_reference_past_key_values = init_cache(target_reference_model, target_reference_past_key_values, inputs, max_new_tokens)
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

        deferrals = nudging_deferral_logic(target_reference_probs, draft_aligned_probs, alpha, rule)
        target_probs = (1 - deferrals) * target_reference_probs + draft_aligned_probs * (deferrals * target_reference_probs).sum(dim=-1, keepdims=True)

        acceptation_probs = (
            target_probs / draft_aligned_probs
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

            target_residual_probs = target_probs[:, first_reject_offset]

            residual_probs = (target_residual_probs - draft_aligned_residual_probs).clamp(min=0)
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
                print("draft_aligned_past_key_values:", get_cache_length(draft_aligned_past_key_values))
            target_reference_past_key_values = crop_cache(target_reference_past_key_values, -(draft_length - 1 - first_reject_offset))
            draft_aligned_past_key_values = crop_cache(draft_aligned_past_key_values, -(draft_length - 1 - first_reject_offset))
            if DEBUG:
                print("after target_reference_past_key_values:", get_cache_length(target_reference_past_key_values))
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
