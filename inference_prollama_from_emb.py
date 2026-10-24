"""Sample from train_prollama_from_emb.py: ProTrek vector -> K soft slots -> "Seq=<...>".

Rebuilds the projector from protein_projector.pt saved next to the adapter, so it must match the
training run's --num-tokens and hidden size. The projector class is the one
train_prollama_from_emb.py trained and saved, imported from there so there is a single definition.

    uv run python inference_prollama_from_emb.py
    uv run python inference_prollama_from_emb.py --start 100 --num-samples 20
"""
import argparse
import collections
import json
from pathlib import Path

import unsloth  # noqa: F401  -- before transformers (unsloth patches them)
import numpy as np
import torch
from datasets import load_dataset

from train_prollama_from_emb import ANSWER_TERMINATOR, ProteinProjector, STANDARD_AMINO_ACIDS

MODEL_NAME = "GreatCaptainNemo/ProLLaMA_Stage_1"
DATASET_REPO = "yosshstd/protrek650m-swissprot50-reps"
DATASET_SPLIT = "train"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default = MODEL_NAME)
    parser.add_argument("--adapter", default = "outputs/prollama_stage1_emb")
    parser.add_argument("--dataset", default = DATASET_REPO)
    parser.add_argument("--split", default = DATASET_SPLIT)
    parser.add_argument("--start", type = int, default = 0, help = "first dataset row to use")
    parser.add_argument("--num-samples", type = int, default = 10)
    parser.add_argument("--max-new-tokens", type = int, default = 256)
    parser.add_argument("--temperature", type = float, default = 0.5)
    parser.add_argument("--top-k", type = int, default = 40)
    parser.add_argument("--top-p", type = float, default = 0.9)
    parser.add_argument("--repetition-penalty", type = float, default = 1.2)
    parser.add_argument("--seed", type = int, default = 0)
    parser.add_argument("--greedy", action = "store_true")
    parser.add_argument("--jsonl", default = None)
    return parser.parse_args()


def report(index, text, residues):
    sequence, closed = split_terminator(text)
    length = len(sequence)
    invalid = sum(residue not in residues for residue in sequence)
    composition = collections.Counter(residue for residue in sequence if residue in residues)
    top = ", ".join(f"{residue} {count / max(length, 1):.2f}"
                    for residue, count in composition.most_common(5))
    print(f"\n[{index}] {length} aa | closed {closed} | invalid {invalid}"
          f" | unique {len(set(sequence))} | top: {top}")
    print(f"  {sequence}")
    return {"index": index, "length": length, "closed": closed, "invalid": invalid,
            "unique": len(set(sequence)), "sequence": sequence}


def split_terminator(text, terminator = ANSWER_TERMINATOR):
    index = text.find(terminator)
    return (text, False) if index < 0 else (text[:index], True)


def main():
    args = parse_args()
    adapter = Path(args.adapter)
    projector_path = adapter / "protein_projector.pt"
    if not projector_path.exists():
        raise SystemExit(f"no projector at {projector_path}; run train_prollama_from_emb.py first")

    saved = torch.load(projector_path, map_location = "cpu")
    num_tokens = saved["num_tokens"]
    hidden_size = saved["hidden_size"]
    prefix_ids = saved["prefix_ids"]
    positions = torch.tensor(saved["placeholder_positions"], dtype = torch.long)

    rows = list(range(args.start, args.start + args.num_samples))
    dataset = load_dataset(args.dataset, split = args.split, columns = ["sequence", "embedding"])
    sequences = [dataset["sequence"][row] for row in rows]
    embeddings = np.asarray([dataset["embedding"][row] for row in rows], dtype = np.float32)

    from unsloth import FastLanguageModel

    print(f"loading {args.model} in 4-bit + adapter {adapter} ...")
    model, processor = FastLanguageModel.from_pretrained(
        model_name = args.model,
        max_seq_length = len(prefix_ids) + args.max_new_tokens,
        dtype = None,
        load_in_4bit = True,
    )
    model = FastLanguageModel.get_peft_model(model, r = 32, lora_alpha = 32,
                                             use_gradient_checkpointing = False)
    model.load_adapter(str(adapter), adapter_name = "trained")
    model.set_adapter("trained")

    tokenizer = getattr(processor, "tokenizer", processor)
    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token
    FastLanguageModel.for_inference(model)
    model.eval()

    projector = ProteinProjector(saved["protrek_dim"], hidden_size, num_tokens)
    # load_state_dict restores the mean buffer fitted during training.
    projector.load_state_dict(saved["state_dict"])
    device = next((p.device for p in model.parameters() if p.device.type == "cuda"),
                  next(model.parameters()).device)
    projector.to(device).eval()
    embed_tokens = model.get_input_embeddings()

    def inputs_embeds(protein):
        kernel = torch.tensor(prefix_ids, dtype = torch.long, device = device).unsqueeze(0)
        embedded = embed_tokens(kernel).clone()
        embedded.scatter_(1, positions.to(device).unsqueeze(0).unsqueeze(-1)
                          .expand(1, -1, embedded.shape[-1]), projector(protein))
        return embedded

    kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.eos_token_id,
        "attention_mask": torch.ones((1, len(prefix_ids)), dtype = torch.long, device = device),
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
        protein = torch.from_numpy(embeddings[index - 1]).unsqueeze(0).to(device)
        torch.manual_seed(args.seed + index)
        with torch.no_grad():
            generated = model.generate(inputs_embeds = inputs_embeds(protein), **kwargs)
        results.append(report(index, tokenizer.decode(generated[0], skip_special_tokens = True),
                              STANDARD_AMINO_ACIDS))
        print(f"  reference: {sequences[index - 1]}{ANSWER_TERMINATOR}")

    lengths = [result["length"] for result in results]
    invalid = sum(result["invalid"] for result in results)
    print("\n=== summary ===")
    print(f"samples          : {len(results)}")
    print(f"bracket closed   : {sum(result['closed'] for result in results)} / {len(results)}")
    print(f"mean length      : {np.mean(lengths):.1f} (min {min(lengths)}, max {max(lengths)})")
    print(f"invalid residues : {invalid} / {sum(lengths)}"
          f" ({invalid / max(sum(lengths), 1):.2%} of emitted characters)")

    if args.jsonl:
        output_path = Path(args.jsonl)
        output_path.parent.mkdir(parents = True, exist_ok = True)
        with output_path.open("w", encoding = "utf-8") as handle:
            for result in results:
                handle.write(json.dumps(result, ensure_ascii = False) + "\n")
        print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
