"""Sanity-check the 4-bit base model: does ProLLaMA_Stage_1 generate protein-like output?

No LoRA, no projector. The prompt is just the instruction the base model was pre-trained on,
so this is "random" generation in the sense of unconditional sampling.

    uv run python inference_stage1.py
    uv run python inference_stage1.py --num-samples 10 --temperature 0.2 --repetition-penalty 1.2

Defaults follow the ProLLaMA reference generation config: temperature 0.2, top_k 40,
top_p 0.9, repetition_penalty 1.2, do_sample with num_beams 1.

If the 4-bit quantization is sound, the output should be amino-acid-like (mostly standard
residues, plausible composition) rather than punctuation or word salad.
"""
import argparse
import collections
import json
from pathlib import Path

import numpy as np

from train import (
    INSTRUCTION,
    MAX_SEQ_LENGTH,
    MODEL_NAME,
    STANDARD_AMINO_ACIDS,
    split_terminator,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type = int, default = 10)
    parser.add_argument("--max-new-tokens", type = int, default = 256)
    parser.add_argument("--temperature", type = float, default = 0.2)
    parser.add_argument("--top-k", type = int, default = 40)
    parser.add_argument("--top-p", type = float, default = 0.9)
    parser.add_argument("--repetition-penalty", type = float, default = 1.2)
    parser.add_argument("--seed", type = int, default = 0)
    parser.add_argument("--greedy", action = "store_true")
    parser.add_argument("--jsonl", default = None)
    return parser.parse_args()


def clean_sequence(text):
    return "".join(text.split())


def report(index, text, residues):
    # "Seq=<sequence>" — the trailing ">" closes the bracket, so it is not an invalid residue.
    sequence, closed = split_terminator(text)
    length = len(sequence)
    invalid = sum(residue not in residues for residue in sequence)
    composition = collections.Counter(residue for residue in sequence if residue in residues)
    top = ", ".join(f"{residue} {count / max(length, 1):.2f}" for residue, count in composition.most_common(5))
    print(f"\n[{index}] {length} aa | closed {closed} | invalid {invalid}"
          f" | unique {len(set(sequence))} | top: {top}")
    print(f"  {sequence}")
    return {"index": index, "length": length, "closed": closed, "invalid": invalid,
            "unique": len(set(sequence)), "sequence": sequence}


def main():
    args = parse_args()

    import unsloth
    from unsloth import FastLanguageModel

    print(f"loading {MODEL_NAME} in 4-bit ...")
    model, processor = FastLanguageModel.from_pretrained(
        model_name = MODEL_NAME,
        max_seq_length = min(MAX_SEQ_LENGTH, args.max_new_tokens + 32),
        dtype = None,
        load_in_4bit = True,
    )
    tokenizer = getattr(processor, "tokenizer", processor)
    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token
    FastLanguageModel.for_inference(model)
    model.eval()

    instruction_ids = tokenizer(INSTRUCTION, add_special_tokens = False)["input_ids"]
    prompt_ids = [tokenizer.bos_token_id] + instruction_ids
    print(f"prompt: {prompt_ids} -> {tokenizer.decode(prompt_ids)!r}")

    import torch
    device = next((p.device for p in model.parameters() if p.device.type == "cuda"),
                  next(model.parameters()).device)

    kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.eos_token_id,
        "attention_mask": torch.ones((1, len(prompt_ids)), dtype = torch.long, device = device),
        "input_ids": torch.tensor(prompt_ids, dtype = torch.long, device = device).unsqueeze(0),
    }
    if args.greedy:
        kwargs["do_sample"] = False
    else:
        kwargs.update(
            do_sample = True,
            temperature = args.temperature,
            top_k = args.top_k,
            top_p = args.top_p,
            repetition_penalty = args.repetition_penalty,
        )

    results = []
    for index in range(1, args.num_samples + 1):
        torch.manual_seed(args.seed + index)
        with torch.no_grad():
            generated = model.generate(**kwargs)
        ids = generated[0].tolist()[len(prompt_ids):]
        text = clean_sequence(tokenizer.decode(ids, skip_special_tokens = True))
        results.append(report(index, text, STANDARD_AMINO_ACIDS))

    lengths = [result["length"] for result in results]
    invalid = sum(result["invalid"] for result in results)
    print("\n=== summary ===")
    print(f"samples          : {len(results)}")
    print(f"bracket closed   : {sum(result['closed'] for result in results)} / {len(results)}")
    print(f"mean length      : {np.mean(lengths):.1f} (min {min(lengths)}, max {max(lengths)})")
    print(f"invalid residues : {invalid} / {sum(lengths)}"
          f" ({invalid / max(sum(lengths), 1):.2%} of emitted characters)")
    for result in results:
        print(f"TEST output_len={result['length']} closed={result['closed']}"
              f" invalid={result['invalid']} sample={result['sequence'][:20]}")

    if args.jsonl:
        output_path = Path(args.jsonl)
        output_path.parent.mkdir(parents = True, exist_ok = True)
        with output_path.open("w", encoding = "utf-8") as handle:
            for result in results:
                handle.write(json.dumps(result, ensure_ascii = False) + "\n")
        print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
