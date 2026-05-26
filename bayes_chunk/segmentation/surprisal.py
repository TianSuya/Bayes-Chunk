from typing import List, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def compute_surprisal(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    target: str,
    sep: str = " ",
) -> Tuple[np.ndarray, List[str], List[int]]:
    """Compute per-target-token surprisal in bits."""
    device = next(model.parameters()).device
    model.eval()

    enc_prompt = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
    enc_full = tokenizer(prompt + sep + target, add_special_tokens=False, return_tensors="pt")
    input_ids_full = enc_full["input_ids"][0]
    input_ids_prompt = enc_prompt["input_ids"][0]

    prompt_len = len(input_ids_prompt)
    if prompt_len <= len(input_ids_full) and torch.equal(input_ids_full[:prompt_len], input_ids_prompt):
        target_start = prompt_len
    else:
        enc_target = tokenizer(target, add_special_tokens=False, return_tensors="pt")
        target_ids_alone = enc_target["input_ids"][0]
        target_len = len(target_ids_alone)
        if target_len and target_len <= len(input_ids_full) and torch.equal(
            input_ids_full[-target_len:], target_ids_alone
        ):
            target_start = len(input_ids_full) - target_len
        else:
            target_start = prompt_len

    target_positions = list(range(target_start, len(input_ids_full)))
    if not target_positions:
        raise ValueError("Target has no tokens. Check prompt/target tokenization.")
    if target_positions[0] == 0:
        raise ValueError("Target starts at token 0; surprisal requires left context.")

    input_ids = input_ids_full.to(device)
    target_ids = input_ids[target_positions].detach().cpu().tolist()

    with torch.no_grad():
        logits = model(input_ids.unsqueeze(0)).logits[0]
        log_probs = torch.log_softmax(logits, dim=-1)

    surprisal = []
    for pos, token_id in zip(target_positions, target_ids):
        log_prob = log_probs[pos - 1, token_id].item()
        surprisal.append(-log_prob / np.log(2.0))

    tokens = tokenizer.convert_ids_to_tokens(target_ids)
    return np.asarray(surprisal), tokens, target_ids
