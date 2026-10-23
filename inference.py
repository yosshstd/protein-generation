"""Sample inference with a trained LoRA adapter + protein projector.

    uv run python inference.py
    uv run python inference.py --num-samples 20 --temperature 0.7 --jsonl outputs/inference.jsonl

The checkpoint directory holds the LoRA adapter only, so the 4-bit base model is loaded
from MODEL_NAME and the adapter is attached on top. The projector is a separate module
and has to come from the same step: train.py writes it to <output_dir>/checkpoints/protein_projector.pt
on every save, overwriting the previous one.

Test samples are drawn from the same precomputed ProTrek embeddings that training used,
so this is a reconstruction check, not a held-out evaluation.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from train import (
    MAX_SEQ_LENGTH,
    MODEL_NAME,
    STANDARD_AMINO_ACIDS,
    ProteinProjector,
    build_prefix,
    filter_candidates,
    load_dataset,
    split_terminator,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default = "outputs/prollama_stage1_protrek650m/checkpoints/checkpoint-4000")
    parser.add_argument("--projector", default = "outputs/prollama_stage1_protrek650m/checkpoints/protein_projector.pt")
    parser.add_argument("--data-dir", default = "data")
    parser.add_argument("--embeddings-dir", default = "outputs/embeddings")
    parser.add_argument("--num-samples", type = int, default = 5)
    parser.add_argument("--seed", type = int, default = 0)
    parser.add_argument("--max-new-tokens", type = int, default = None,
                        help = "default: MAX_SEQ_LENGTH minus the prefix length")
    parser.add_argument("--temperature", type = float, default = 0.2)
    parser.add_argument("--top-k", type = int, default = 40)
    parser.add_argument("--top-p", type = float, default = 0.9)
    parser.add_argument("--repetition-penalty", type = float, default = 1.2)
    parser.add_argument("--greedy", action = "store_true", help = "disable sampling and use greedy decoding")
    parser.add_argument("--jsonl", default = None, help = "optional path to write one JSON object per sample")
    return parser.parse_args()


def clean_sequence(text):
    return "".join(text.split())


def alignment_identity(reference, generated):
    """Needleman-Wunsch identity: matching columns / alignment columns."""
    if not reference or not generated:
        return 0.0
    match, mismatch, gap = 2, -1, -2
    rows, columns = len(reference), len(generated)
    trace = [bytearray(columns + 1) for _ in range(rows + 1)]  # 0=diag, 1=up, 2=left
    previous = [gap * j for j in range(columns + 1)]
    for i in range(1, rows + 1):
        current = [gap * i] + [0] * columns
        residue = reference[i - 1]
        moves = trace[i]
        for j in range(1, columns + 1):
            best = previous[j - 1] + (match if residue == generated[j - 1] else mismatch)
            move = 0
            up = previous[j] + gap
            if up > best:
                best, move = up, 1
            left = current[j - 1] + gap
            if left > best:
                best, move = left, 2
            current[j] = best
            moves[j] = move
        previous = current

    i, j, matches, aligned = rows, columns, 0, 0
    while i > 0 or j > 0:
        move = trace[i][j] if i > 0 and j > 0 else (1 if i > 0 else 2)
        if move == 0:
            matches += reference[i - 1] == generated[j - 1]
            aligned += 1
            i, j = i - 1, j - 1
        elif move == 1:
            aligned += 1
            i -= 1
        else:
            aligned += 1
            j -= 1
    return matches / aligned


def load_projector(path, in_features, out_features, device):
    """Rebuild the projector from a train.py checkpoint.

    The checkpoint records how many soft-prompt slots the projector was trained with, so the
    prefix and the injection have to follow that value rather than the current default.
    """
    saved = torch.load(path, map_location = "cpu", weights_only = True)
    state = saved["state_dict"] if isinstance(saved, dict) and "state_dict" in saved else saved
    num_tokens = int(saved["num_tokens"]) if isinstance(saved, dict) and "num_tokens" in saved else 1
    projector = ProteinProjector(in_features, out_features, num_tokens)
    projector.load_state_dict(state)
    return projector.to(device = device).eval(), num_tokens


def projector_slot_count(path):
    """Slot count recorded in a projector checkpoint (1 for the old single-slot format)."""
    saved = torch.load(path, map_location = "cpu", weights_only = True)
    return int(saved["num_tokens"]) if isinstance(saved, dict) and "num_tokens" in saved else 1


def main():
    args = parse_args()
    checkpoint, projector_path = Path(args.checkpoint), Path(args.projector)
    assert checkpoint.is_dir(), f"missing checkpoint: {checkpoint}"
    assert projector_path.exists(), f"missing projector: {projector_path}"

    embeddings_path = Path(args.embeddings_dir) / "protrek650m_protein_swissprot_50pct.npy"
    meta_path = Path(args.embeddings_dir) / "protrek650m_protein_swissprot_50pct_meta.parquet"
    reps_prefix = Path(args.data_dir) / "protrek650m_swissprot_50pct_reps"
    for path in (embeddings_path, meta_path, Path(f"{reps_prefix}.parquet")):
        assert path.exists(), f"missing: {path} (run build_dataset.py first)"

    embeddings, meta, reps = load_dataset(embeddings_path, meta_path, reps_prefix)
    rows, accessions = filter_candidates(meta, reps)
    count = min(args.num_samples, len(rows))
    picked = np.random.default_rng(args.seed).choice(len(rows), size = count, replace = False)
    print(f"checkpoint: {checkpoint}")
    print(f"projector : {projector_path}")
    print(f"samples   : {count} / {len(rows)} candidates (seed {args.seed})")

    from transformers import AutoTokenizer

    # The prefix depends on how many soft-prompt slots the projector was trained with, so read
    # that from the checkpoint before sizing the model. build_prefix only needs the token ids,
    # so the plain tokenizer is enough here.
    num_tokens = projector_slot_count(projector_path)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    prefix_ids, placeholder_positions = build_prefix(tokenizer, num_tokens)
    del tokenizer

    import unsloth
    from peft import PeftModel
    from unsloth import FastLanguageModel

    max_new_tokens = args.max_new_tokens if args.max_new_tokens is not None else MAX_SEQ_LENGTH - len(prefix_ids)
    max_seq_length = len(prefix_ids) + max_new_tokens
    print(f"loading {MODEL_NAME} in 4-bit (max_seq_length={max_seq_length}) ...")
    model, processor = FastLanguageModel.from_pretrained(
        model_name = MODEL_NAME,
        max_seq_length = max_seq_length,
        dtype = None,
        load_in_4bit = True,
    )
    model = PeftModel.from_pretrained(model, str(checkpoint))
    FastLanguageModel.for_inference(model)
    model.eval()
    model.config.use_cache = True

    tokenizer = getattr(processor, "tokenizer", processor)
    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token
    # The checkpoint prefix was built with the tokenizer train.py used; a different one would
    # silently inject at the wrong slots.
    assert build_prefix(tokenizer, num_tokens) == (prefix_ids, placeholder_positions), \
        "checkpoint prefix does not match this tokenizer"
    eos_id = tokenizer.eos_token_id

    # Unsloth can offload the embedding table to RAM, so pick a CUDA device from any parameter.
    device = next((p.device for p in model.parameters() if p.device.type == "cuda"), None)
    assert device is not None, "no CUDA parameter found"
    embed_tokens = model.get_input_embeddings()
    projector, num_tokens = load_projector(
        projector_path, embeddings.shape[1], model.config.hidden_size, device)
    print(f"device: {device} | prefix: {prefix_ids} | slots: {num_tokens}"
          f" | placeholder positions: {placeholder_positions}")

    def build_inputs_embeds(input_ids, protein, positions):
        embedded = embed_tokens(input_ids)                       # (B, L, H)
        projected = projector(protein).to(embedded.dtype)        # (B, K, H)
        if positions.dim() == 1:
            positions = positions.unsqueeze(0)
        index = positions.expand(input_ids.shape[0], -1).unsqueeze(-1)
        return embedded.scatter(1, index.expand(-1, -1, embedded.shape[-1]), projected)

    results = []
    for index, sample in enumerate(picked, start = 1):
        row = int(rows[sample])
        accession = accessions[sample]
        reference = reps.at[accession, "sequence"]
        input_ids = torch.tensor(prefix_ids, dtype = torch.long, device = device).unsqueeze(0)
        protein = torch.from_numpy(np.array(embeddings[row], dtype = np.float32)).unsqueeze(0).to(device)
        positions = torch.tensor(placeholder_positions, dtype = torch.long, device = device)
        inputs_embeds = build_inputs_embeds(input_ids, protein, positions)
        if index == 1:
            print(f"projector output norm: {inputs_embeds[0, placeholder_positions].norm(dim = -1).mean().item():.4f}")

        kwargs = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": torch.ones_like(input_ids),
            "max_new_tokens": max_new_tokens,
            "eos_token_id": eos_id,
            "pad_token_id": eos_id,
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

        torch.manual_seed(args.seed + index)  # reproducible sampling for each sample
        with torch.no_grad():
            generated = model.generate(**kwargs)

        ids = generated[0].tolist()
        # The model writes "Seq=<sequence>": ">" closes the bracket, so it is not a residue
        # and not an invalid character.
        text, closed = split_terminator(clean_sequence(tokenizer.decode(ids, skip_special_tokens = True)))
        sequence = text
        result = {
            "accession": accession,
            "row": row,
            "reference_length": len(reference),
            "generated_length": len(sequence),
            "closed": closed,
            "eos": eos_id in ids,
            "exact_match": sequence == reference,
            "identity": alignment_identity(reference, sequence),
            "invalid_residues": sum(residue not in STANDARD_AMINO_ACIDS for residue in sequence),
            "reference": reference,
            "generated": sequence,
        }
        results.append(result)
        print(f"\n[{index}/{count}] {accession} | ref {len(reference)} aa | gen {len(sequence)} aa"
              f" | closed {result['closed']} | eos {result['eos']}"
              f" | identity {result['identity']:.3f}"
              f" | exact {result['exact_match']} | invalid {result['invalid_residues']}")
        print(f"  ref: {reference}")
        print(f"  gen: {sequence}")

    if results:
        print("\n=== summary ===")
        print(f"exact match      : {sum(r['exact_match'] for r in results)} / {len(results)}")
        print(f"mean identity    : {np.mean([r['identity'] for r in results]):.3f}")
        print(f"mean length      : ref {np.mean([r['reference_length'] for r in results]):.1f}"
              f" / gen {np.mean([r['generated_length'] for r in results]):.1f}")
        print(f"eos reached      : {sum(r['eos'] for r in results)} / {len(results)}")
        print(f"bracket closed   : {sum(r['closed'] for r in results)} / {len(results)}")
        print(f"invalid residues : {sum(r['invalid_residues'] for r in results)}")

    if args.jsonl:
        output_path = Path(args.jsonl)
        output_path.parent.mkdir(parents = True, exist_ok = True)
        with output_path.open("w", encoding = "utf-8") as handle:
            for result in results:
                handle.write(json.dumps(result, ensure_ascii = False) + "\n")
        print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
