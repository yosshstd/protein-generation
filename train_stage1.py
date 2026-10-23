"""LoRA-tune ProLLaMA_Stage_1 on plain "Seq=<sequence>" text, with no embeddings.

The stage-1 baseline for train.py: same base model, same prompt, same answer masking, but the
conditioning is the bare instruction instead of K soft-prompt slots projected from a ProTrek
vector. Train this to separate "can the model write amino acids at all" from "can the projector
steer it"; train.py should then only have to learn the steering part.

Settings come from config.yaml (see that file for the knobs and their defaults) and can be
overridden on the command line with `key=value` pairs, which are merged into the YAML:

    python train_stage1.py training.learning_rate=1e-4 training.num_train_epochs=2

Run build_dataset.py first (only the representative sequences are used, not the embeddings).

Usage: python train_stage1.py [--config PATH] [--max-steps N] [--resume] [key=value ...]
"""
import argparse
from dataclasses import fields
from pathlib import Path

import unsloth  # noqa: F401  -- must be imported before transformers/trl (unsloth patches them)
import numpy as np
import pyarrow.parquet as pq
import torch
from datasets import Dataset
from omegaconf import OmegaConf
from tqdm.auto import tqdm

from train import (
    MAX_RESIDUE_LENGTH,
    MODEL_NAME,
    STANDARD_AMINO_ACIDS,
    build_prefix,
    encode_answer,
    split_terminator,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default = "config.yaml")
    parser.add_argument("--max-steps", type = int, default = None,
                        help = "override training.max_steps (useful for a smoke run)")
    parser.add_argument("--resume", action = "store_true")
    parser.add_argument("overrides", nargs = "*", metavar = "key=value",
                        help = "dotted config overrides, e.g. training.learning_rate=1e-4")
    return parser.parse_args()


def load_config(args):
    """config.yaml merged with the command-line overrides."""
    for item in args.overrides:
        assert "=" in item, f"override must be key=value, got: {item!r}"
    config = OmegaConf.merge(OmegaConf.load(args.config),
                             OmegaConf.from_dotlist(args.overrides))
    if args.max_steps is not None:
        config.training.max_steps = args.max_steps
    return config


def load_sequences(reps_path, limit = None):
    table = pq.read_table(reps_path)
    sequences = table.column("sequence").to_pylist()
    return sequences[:limit] if limit is not None else sequences


def filter_sequences(sequences):
    """Drop sequences the model cannot be trained on, like train.filter_candidates.

    - longer than MAX_RESIDUE_LENGTH: the same 1024-residue cutoff train.py uses. Nothing here
      depends on ProTrek, but keeping the two runs on the identical subset is the point of this
      baseline: a run over longer sequences would not be comparable to train.py.
    - containing residues outside the 20 standard amino acids (X, U, B, Z, O): those are
      ambiguous or non-canonical, and the model should not be trained to emit them.

    The token-length cutoff stays in build_dataset, since only the tokeniser knows how many
    residues collapse into one token.
    """
    kept, dropped_alphabet, dropped_length = [], [], []
    for sequence in sequences:
        if len(sequence) > MAX_RESIDUE_LENGTH:
            dropped_length.append(sequence)
        elif not STANDARD_AMINO_ACIDS.issuperset(sequence):
            dropped_alphabet.append(sequence)
        else:
            kept.append(sequence)

    print(f"candidates: {len(sequences)} | kept: {len(kept)}"
          f" | dropped (>{MAX_RESIDUE_LENGTH} residues): {len(dropped_length)}"
          f" | dropped (non-standard residues): {len(dropped_alphabet)}")
    return kept


def tokenize_sequences(tokenizer, prefix_ids, sequences, max_seq_length):
    """Tokenise every answer and keep the samples that fit in max_seq_length."""
    answers, dropped_tokens = [], 0
    for sequence in tqdm(sequences, total = len(sequences), desc = "tokenize"):
        answer_ids = encode_answer(tokenizer, sequence)
        if len(prefix_ids) + len(answer_ids) + 1 > max_seq_length:
            dropped_tokens += 1
            continue
        answers.append(answer_ids)

    print("samples:", len(answers), "| dropped (too many tokens):", dropped_tokens)
    assert answers, "no sample survived the filters"
    return answers


def build_example(answer_ids, prefix_ids, eos_id):
    """One training row: the instruction as context, the answer as target.

    input_ids = <s> <unk>*K ▁Seq = < + answer + EOS
    labels    = -100 * len(prefix) + answer + EOS

    The instruction is context, not a target, so it carries no loss. Same masking as train.py,
    which is what keeps the two runs comparable.
    """
    ids = prefix_ids + answer_ids + [eos_id]
    return {
        "input_ids": ids,
        "labels": [-100] * len(prefix_ids) + answer_ids + [eos_id],
    }


def to_dataset(tokenizer, prefix_ids, answers):
    """Tokeniser-ready dataset: SFTTrainer packs it but does not have to re-tokenise it.

    Having an "input_ids" column marks the dataset as already processed, so trl skips its text
    pipeline (formatting, EOS handling, label building) and only runs packing over it.
    """
    dataset = Dataset.from_dict({"answer_ids": answers})
    dataset = dataset.map(
        lambda example: build_example(example["answer_ids"], prefix_ids, tokenizer.eos_token_id),
        remove_columns = ["answer_ids"],
    )
    lengths = [len(item) for item in dataset["input_ids"]]
    print(f"token length: mean {np.mean(lengths):.1f} | max {max(lengths)}")
    return dataset


def training_kwargs(config):
    """config.training as a plain dict, checked against SFTConfig's fields."""
    from trl import SFTConfig

    kwargs = OmegaConf.to_container(config.training, resolve = True)
    known = {field.name for field in fields(SFTConfig)}
    unknown = sorted(set(kwargs) - known)
    assert not unknown, f"unknown key(s) in training: {unknown}; SFTConfig has no such field"
    if kwargs.get("packing") and kwargs.get("packing_strategy") not in {"bfd", "bfd_split", "wrapped"}:
        raise ValueError(f"invalid packing_strategy: {kwargs.get('packing_strategy')}")
    return kwargs


def report_packing_attention(model, kwargs):
    """Report how packed samples are kept apart.

    trl warns that packing needs Flash Attention, but that is only true for kernels that get
    their document boundaries from cu_seqlens. Here the padding-free collator restarts
    position_ids at 0 for every packed sample, and transformers (>= 5.x) sees that reset via
    find_packed_sequence_indices and adds a document mask for sdpa as well. Verified on this
    stack: logits of the second sample in a packed row match the same sample run alone exactly
    (max |diff| 0.0), so packing is document-isolated without Flash Attention.
    """
    if not kwargs.get("packing"):
        return
    implementation = model.config._attn_implementation.split("@")[0].split(":")[0]
    print("attention implementation:", implementation,
          "| packing: samples are isolated by the position_ids-driven document mask")


def main():
    args = parse_args()
    config = load_config(args)
    print(OmegaConf.to_yaml(config))

    output_dir = Path(config.output_dir)
    data_dir = Path(config.dataset.data_dir)
    reps_path = data_dir / config.dataset.reps_file
    assert reps_path.exists(), f"missing: {reps_path} (run build_dataset.py first)"

    limit = config.dataset.limit
    sequences = filter_sequences(load_sequences(reps_path, int(limit) if limit is not None else None))

    from unsloth import FastLanguageModel

    model_kwargs = OmegaConf.to_container(config.model, resolve = True)
    model_kwargs.setdefault("model_name", MODEL_NAME)
    # The model is loaded before tokenising on purpose: the dataset has to be tokenised with
    # the tokenizer the model itself will use, and Unsloth does not hand it back until now.
    model, processor = FastLanguageModel.from_pretrained(**model_kwargs)
    tokenizer = getattr(processor, "tokenizer", processor)
    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token
    # Zero slots: the prefix is just <s> ▁Seq = <, the exact form the base model was
    # pre-trained on, so this baseline trains the answer and nothing else.
    prefix_ids, _ = build_prefix(tokenizer, 0)
    print("prefix:", prefix_ids)

    max_seq_length = model_kwargs["max_seq_length"]
    answers = tokenize_sequences(tokenizer, prefix_ids, sequences, max_seq_length)
    hf_dataset = to_dataset(tokenizer, prefix_ids, answers)

    model = FastLanguageModel.get_peft_model(model, **config.lora)
    FastLanguageModel.for_training(model)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs = {"use_reentrant": True})

    device = next((p.device for p in model.parameters() if p.device.type == "cuda"),
                  next(model.parameters()).device)
    print("device:", device, "| trainable:", f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    kwargs = training_kwargs(config)
    # Packing fills sequences up to SFTConfig.max_length (default 1024), which is longer than
    # the model was loaded for. Default it to max_seq_length and refuse anything longer.
    kwargs.setdefault("max_length", max_seq_length)
    assert kwargs["max_length"] <= max_seq_length, (
        f"training.max_length ({kwargs['max_length']}) exceeds model.max_seq_length "
        f"({max_seq_length}); packing would build sequences longer than the model's context")
    report_packing_attention(model, kwargs)

    from trl import SFTConfig, SFTTrainer

    # No data_collator: with packing, trl builds its padding-free collator itself and rejects a
    # custom one. Labels are already in the dataset (built by build_example), so the collator
    # only has to pad/flatten them.
    trainer = SFTTrainer(
        model = model,
        processing_class = tokenizer,
        train_dataset = hf_dataset,
        args = SFTConfig(output_dir = str(output_dir / "checkpoints"), **kwargs),
    )

    resume_from = None
    if args.resume:
        checkpoints = sorted(
            (path for path in (output_dir / "checkpoints").glob("checkpoint-*") if path.is_dir()),
            key = lambda path: int(path.name.split("-")[-1]),
        )
        if checkpoints:
            resume_from = checkpoints[-1]
            print("resuming from:", resume_from)

    trainer.train(resume_from_checkpoint = resume_from)

    output_dir.mkdir(parents = True, exist_ok = True)
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print("saved:", sorted(path.name for path in output_dir.iterdir()))

    # Same sanity check as train.py: generate from the instruction alone.
    FastLanguageModel.for_inference(model)
    model.eval()
    eos_id = tokenizer.eos_token_id
    prompt = torch.tensor(prefix_ids, dtype = torch.long, device = device).unsqueeze(0)
    with torch.no_grad():
        generated = model.generate(
            input_ids = prompt,
            attention_mask = torch.ones_like(prompt),
            max_new_tokens = max_seq_length - len(prefix_ids),
            do_sample = False,
            eos_token_id = eos_id,
        )
    # input_ids generation returns the prompt too, unlike the inputs_embeds path in train.py.
    text, closed = split_terminator(
        tokenizer.decode(generated[0][len(prefix_ids):], skip_special_tokens = True))
    reference = tokenizer.decode(answers[0], skip_special_tokens = True)
    print("reference:", reference)
    print("generated:", text, f"| closed: {closed}")


if __name__ == "__main__":
    main()
