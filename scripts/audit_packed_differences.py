"""Replay packed mismatches, inspect the first differing decision and perturb neighbors."""
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from benchmark_batch_generation import make_prompts, solo_reference
from inference_lab.prefill import generate_batch


def top_two(logits):
    top = logits.float().topk(2)
    return dict(ids=top.indices.tolist(), scores=top.values.tolist(),
                margin=float(top.values[0] - top.values[1]))


@torch.inference_mode()
def inspect_case(model, lengths, seed, row, steps):
    prompts = make_prompts(model, lengths, seed)
    prompt = prompts[row]
    _, reference = solo_reference(model, [prompt], steps)
    scores = []
    handle = model.lm_head.register_forward_hook(
        lambda module, args, out: scores.append(out.reshape(len(prompts), -1, out.shape[-1])[row, -1].clone())
    )
    try:
        packed = generate_batch(model, prompts, "packed", steps)
    finally:
        handle.remove()
    tokens = packed.token_ids[row]
    changed = [p if i == row else (p + 1) % model.config.vocab_size for i, p in enumerate(prompts)]
    replay = generate_batch(model, changed, "packed", steps)
    result = dict(lengths=lengths, seed=seed, row=row, reference=reference[0].tolist(),
                  packed=tokens.tolist(), neighbor_change_tokens_equal=torch.equal(tokens, replay.token_ids[row]),
                  neighbor_change_first_logits_equal=torch.equal(packed.first_logits[row], replay.first_logits[row]))
    differences = (tokens != reference[0]).nonzero().flatten()
    if len(differences):
        step = int(differences[0])
        prefix = torch.cat([prompt, tokens[:step]])[None]
        solo = model(prefix, use_cache=False, logits_to_keep=1).logits[0, -1]
        result.update(first_different_step=step + 1, solo_top_two=top_two(solo),
                      packed_top_two=top_two(scores[step]))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a fresh output path")
    bench = json.loads(args.input.read_text(encoding="utf8"))
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=getattr(torch, bench["dtype"]),
                                                 attn_implementation="sdpa").cuda().eval()
    cases = []
    for shape in bench["shapes"]:
        for check in shape["checks"]:
            if check["method"] == "packed":
                for i, equal in enumerate(check["sequences_equal"]):
                    if not equal:
                        cases.append(inspect_case(model, shape["lengths"], check["seed"], i, bench["new_tokens"]))
    with args.output.open("x", encoding="utf8") as f:
        json.dump(cases, f, indent=2)


if __name__ == "__main__":
    main()
