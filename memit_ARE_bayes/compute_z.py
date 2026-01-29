from typing import Dict, List, Tuple
import numpy as np
import torch
import math
from transformers import AutoModelForCausalLM, AutoTokenizer
from .memit_ARE_hparams import MEMITAREHyperParams
from util import nethook


def compute_surprisal(model: AutoModelForCausalLM,
                      tokenizer: AutoTokenizer,
                      question: str,
                      answer: str,
                      sep: str = " ") -> Tuple[np.ndarray, List[str], List[int]]:
    device = next(model.parameters()).device
    model.eval()

    enc_q = tokenizer(question, add_special_tokens=False, return_tensors="pt")
    enc_full = tokenizer(question + sep + answer, add_special_tokens=False, return_tensors="pt")
    
    input_ids_full = enc_full["input_ids"][0]
    input_ids_q = enc_q["input_ids"][0]
    
    num_q_tokens = len(input_ids_q)
    
    if num_q_tokens <= len(input_ids_full) and torch.equal(input_ids_full[:num_q_tokens], input_ids_q):
        answer_start = num_q_tokens
    else:
        enc_ans = tokenizer(answer, add_special_tokens=False, return_tensors="pt")
        answer_ids_alone = enc_ans["input_ids"][0]
        
        expected_answer_len = len(answer_ids_alone)
        if expected_answer_len > 0 and expected_answer_len <= len(input_ids_full):
            if torch.equal(input_ids_full[-expected_answer_len:], answer_ids_alone):
                answer_start = len(input_ids_full) - expected_answer_len
            else:
                answer_start = num_q_tokens
        else:
            answer_start = num_q_tokens
    
    answer_positions = list(range(answer_start, len(input_ids_full)))
    
    if not answer_positions:
        raise ValueError("Answer has no tokens. Check input.")
    
    input_ids = input_ids_full.to(device)
    a_ids = input_ids[answer_positions].detach().cpu().tolist()
    
    with torch.no_grad():
        logits = model(input_ids.unsqueeze(0)).logits[0]
        log_probs = torch.log_softmax(logits, dim=-1)
    
    surps = []
    for pos_i, token_id in zip(answer_positions, a_ids):
        lp = log_probs[pos_i - 1, token_id].item()
        surps.append(-lp / np.log(2.0))
    
    answer_tokens = tokenizer.convert_ids_to_tokens(a_ids)
    
    assert len(surps) == len(answer_tokens) == len(a_ids), \
        f"Length mismatch: surps={len(surps)}, tokens={len(answer_tokens)}, ids={len(a_ids)}"
    
    return np.array(surps), answer_tokens, a_ids


def find_boundaries_by_top_k(
    surprisal_values: np.ndarray,
) -> List[int]:
    if len(surprisal_values) == 0:
        return []
    
    k = math.ceil(len(surprisal_values) / 40)
    
    if k <= 0:
        return []
    
    if k >= len(surprisal_values):
        return list(range(len(surprisal_values)))
    
    sorted_indices = np.argsort(surprisal_values)
    top_k_indices = sorted_indices[-k:].tolist()
    
    top_k_indices.sort()
    
    if 0 in top_k_indices:
        return top_k_indices
    else:
        if len(top_k_indices) > 0:
            return top_k_indices[1:]
        else:
            return []


def segment_by_surprisal_head(
    target_ids: torch.Tensor,
    boundaries: List[int],
) -> List[torch.Tensor]:
    L = len(target_ids)
    
    boundary_positions = sorted({t for t in boundaries if 0 <= t <= L})
    
    segments = []
    
    if not boundary_positions:
        segments.append(target_ids)
        return segments
    
    if boundary_positions[0] != 0:
        boundary_positions = [0] + boundary_positions
    
    for i in range(len(boundary_positions)):
        start = boundary_positions[i]
        if i + 1 < len(boundary_positions):
            end = boundary_positions[i + 1]
        else:
            end = L
        
        if start < end:
            seg_tokens = target_ids[start:end]
            segments.append(seg_tokens)
    
    return segments


def compute_z(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    data: Dict,
    layer: int,
    hparams: MEMITAREHyperParams,
) -> Tuple[torch.Tensor, torch.Tensor]:
    lm_w, ln_f = (
        nethook.get_parameter(model, f"{hparams.lm_head_module}.weight").T,
        nethook.get_module(model, hparams.ln_f_module),
    )
    try:
        lm_b = nethook.get_parameter(model, f"{hparams.lm_head_module}.bias")
    except LookupError as _:
        lm_b = next(model.parameters()).new_zeros(model.config.vocab_size)

    target_ids = tok(data["answer"], return_tensors="pt", add_special_tokens=False).to("cuda")[
        "input_ids"
    ][0]
    
    input_tok = tok(
        [data["question"]],  
        return_tensors="pt",
    ).to("cuda")

    cur_input_ids = input_tok['input_ids'] 
    all_delta = []
    all_target = []
    all_idxs = []

    answer_surprisal, _, _ = compute_surprisal(
            model=model,
            tokenizer=tok,
            question=data["question"],
            answer=data["answer"]
        )

    boundaries = find_boundaries_by_top_k(
                surprisal_values=answer_surprisal
            )
    
    last_token_idx = len(target_ids) - 1
    if last_token_idx not in boundaries:
        boundaries.append(last_token_idx)
        boundaries.sort()
        print(f"  ⚡ Force-added boundary {last_token_idx} (</s> position) so end token is optimized separately")

    print(f"\n{'='*80}")
    print(f"📍 Found {len(boundaries)} boundary position(s):")
    print(f"{'='*80}")
    target_ids_list = target_ids.cpu().tolist()
    target_tokens = tok.convert_ids_to_tokens(target_ids_list)
    for i, boundary_idx in enumerate(boundaries):
        if boundary_idx < len(target_tokens):
            boundary_token = target_tokens[boundary_idx]
            start_ctx = max(0, boundary_idx - 2)
            end_ctx = min(len(target_tokens), boundary_idx + 3)
            context_tokens = target_tokens[start_ctx:end_ctx]
            context_str = tok.convert_tokens_to_string(context_tokens)
            
            print(f"  Boundary {i+1}: position {boundary_idx:3d} | token: '{boundary_token}' | context: ...{context_str}...")
        else:
            print(f"  Boundary {i+1}: position {boundary_idx:3d} (end of sequence)")
    
    segments = segment_by_surprisal_head(target_ids, boundaries)

    print(f"\n{'='*80}")
    print(f"✂️  Segmentation result: {len(segments)} segment(s)")
    print(f"{'='*80}")
    for seg_idx, segment in enumerate(segments):
        seg_tokens = tok.convert_ids_to_tokens(segment.cpu().tolist())
        seg_text = tok.convert_tokens_to_string(seg_tokens)
        print(f"\n  Segment {seg_idx + 1}: length {len(segment)} tokens")
        print(f"    Text: {seg_text[:100]}{'...' if len(seg_text) > 100 else ''}")
        print(f"    First/last token: '{seg_tokens[0]}' ... '{seg_tokens[-1]}'")
    print(f"{'='*80}\n")
    for seg_idx, current_target_ids in enumerate(segments):
        print(f"\n{'─'*80}")
        print(f"🔧 Processing segment {seg_idx + 1}/{len(segments)}")
        print(f"{'─'*80}")

        input_ids = torch.cat([cur_input_ids, current_target_ids[:-1].unsqueeze(0)], dim=1)
        cur_input_ids = torch.cat([cur_input_ids, current_target_ids.unsqueeze(0)], dim=1)

        rewriting_targets = torch.tensor(-100, device="cuda").repeat(
            1, len(input_ids[0])
        )
   
        ex_len = len(input_ids[0])
        rewriting_targets[0, ex_len - len(current_target_ids) : ex_len] = current_target_ids

        lookup_idxs = [ex_len - len(current_target_ids)]
        
        edit_token = tok.convert_ids_to_tokens([input_ids[0][lookup_idxs[0]].cpu().item()])[0]
        boundary_token = tok.convert_ids_to_tokens([current_target_ids[0].cpu().item()])[0]
        print(f"  📌 Edit position: {lookup_idxs[0]} | edit token: '{edit_token}' → predict boundary token: '{boundary_token}'")

        loss_layer = max(hparams.v_loss_layer, layer)
        if hasattr(model.config, 'n_embd'):
            delta = torch.zeros((model.config.n_embd,), requires_grad=True, device="cuda")
        elif hasattr(model.config, 'hidden_size'):
            delta = torch.zeros((model.config.hidden_size,), requires_grad=True, device="cuda")
        else:
            raise NotImplementedError
        target_init = None

        def edit_output_fn(cur_out, cur_layer):
            nonlocal target_init

            if cur_layer == hparams.layer_module_tmp.format(layer):
                if target_init is None:
                    target_init = cur_out[0][0, lookup_idxs[0]].detach().clone()

                for idxs_pre, delta_pre in all_delta:
                    for i, idx in enumerate(idxs_pre):
                        if len(idxs_pre)!=len(cur_out[0]):
                            cur_out[0][idx, i, :] += delta_pre
                        else:
                            cur_out[0][i, idx, :] += delta_pre

                for i, idx in enumerate(lookup_idxs):
                    if len(lookup_idxs)!=len(cur_out[0]):
                        cur_out[0][idx, i, :] += delta
                    else:
                        cur_out[0][i, idx, :] += delta

            return cur_out

        opt = torch.optim.Adam([delta], lr=hparams.v_lr)
        nethook.set_requires_grad(False, model)

        for it in range(hparams.v_num_grad_steps):
            opt.zero_grad()

            with nethook.TraceDict(
                module=model,
                layers=[
                    hparams.layer_module_tmp.format(loss_layer),
                    hparams.layer_module_tmp.format(layer),
                ],
                retain_input=False,
                retain_output=True,
                edit_output=edit_output_fn,
            ) as tr:
                logits = model(input_ids).logits

            output=tr[hparams.layer_module_tmp.format(loss_layer)].output[0]  
            if output.shape[1]!=rewriting_targets.shape[1]:
                output=torch.transpose(output, 0, 1)
            full_repr =  output

            log_probs = torch.log_softmax(ln_f(full_repr) @ lm_w.to(full_repr.device) + lm_b.to(full_repr.device), dim=2)
            loss = torch.gather(
                log_probs,
                2,
                torch.where(rewriting_targets != -100, rewriting_targets, 0).unsqueeze(2).to(log_probs.device),
            ).squeeze(2)
            mask = (rewriting_targets != -100).float()

            nll_loss_each = -(loss * mask.to(loss.device)).sum(1) / current_target_ids.size(0)
            nll_loss = nll_loss_each.mean()
            
            weight_decay = hparams.v_weight_decay * (
                torch.norm(delta) / torch.norm(target_init) ** 2
            )
            loss = nll_loss + weight_decay.to(nll_loss.device)
            if it % 5 == 0 or it == hparams.v_num_grad_steps - 1:
                print(f"    iter {it:3d}: loss={loss.item():.4f} (nll={nll_loss.item():.4f} + wd={weight_decay.item():.4f}) prob={torch.exp(-nll_loss_each).mean().item():.4f}")
            if loss < 1e-2:
               break

            if it == hparams.v_num_grad_steps - 1:
                break

            loss.backward()
            opt.step()

            max_norm = hparams.clamp_norm_factor * target_init.norm()
            if delta.norm() > max_norm:
                with torch.no_grad():
                    delta[...] = delta * max_norm / delta.norm()
        
        target = target_init + delta  
        all_delta.append((lookup_idxs, delta.clone()))
        all_target.append(target)
        all_idxs.append(lookup_idxs[0])
        print(f"\n  ✅ Segment {seg_idx + 1} optimization complete:")
        print(f"     Init norm: {target_init.norm():.4f} | Delta norm: {delta.norm():.4f} | Target norm: {target.norm():.4f}")
        
        try:
            del logits, output, full_repr, log_probs, loss, mask
            del nll_loss_each, nll_loss, weight_decay, tr
        except NameError:
            pass

        try:
            del input_ids, rewriting_targets, current_target_ids
        except NameError:
            pass

        try:
            del delta, target_init, opt
        except NameError:
            pass

        torch.cuda.empty_cache()

    return all_idxs, all_target
