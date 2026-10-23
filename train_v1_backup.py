"""LoRA-tune ProLLaMA_Stage_1 to decode amino acid sequences from ProTrek_650M embeddings.

Run build_dataset.py first.

Usage: python train.py [--max-steps N] [--resume]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from datasets import Dataset
from tqdm.auto import tqdm

MODEL_NAME = "GreatCaptainNemo/ProLLaMA_Stage_1"
PROMPT_PREFIX = "<unk>Seq=<"   # -> <s> <unk> ▁Seq = <
PLACEHOLDER = "<unk>"          # its embedding slot receives the ProTrek vector
MAX_SEQ_LENGTH = 704
MAX_RESIDUE_LENGTH = 1024     # ProTrek only embeds the first 1024 residues
STANDARD_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")
SEED = 0

BATCH_SIZE = 1
GRAD_ACCUM = 16
NUM_EPOCHS = 1
LR_LORA = 1e-4
LR_PROJECTOR = 1e-4
PROJECTOR_WEIGHT_DECAY = 0.01
PROJECTED_NORM = 1.0   # ProLLaMA token embeddings have norm ~1.08


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default = "data")
    parser.add_argument("--embeddings-dir", default = "outputs/embeddings")
    parser.add_argument("--output-dir", default = "outputs/prollama_stage1_protrek650m")
    parser.add_argument("--max-seq-length", type = int, default = MAX_SEQ_LENGTH)
    parser.add_argument("--max-steps", type = int, default = None)
    parser.add_argument("--resume", action = "store_true")
    return parser.parse_args()


def build_prefix(tokenizer):
    prefix_ids = tokenizer(PROMPT_PREFIX, add_special_tokens = True)["input_ids"]
    placeholder_id = tokenizer.convert_tokens_to_ids(PLACEHOLDER)
    assert prefix_ids.count(placeholder_id) == 1, "placeholder must be exactly one token"
    return prefix_ids, placeholder_id, prefix_ids.index(placeholder_id)


def load_dataset(embeddings_path, meta_path, reps_prefix):
    embeddings = np.load(embeddings_path, mmap_mode = "r")
    meta = pd.read_parquet(meta_path)
    reps = pq.read_table(f"{reps_prefix}.parquet").to_pandas().set_index("primaryAccession")
    assert set(meta["primaryAccession"]) <= set(reps.index), "meta and reps table are out of sync"
    return embeddings, meta, reps


def filter_candidates(meta, reps):
    """Drop sequences the precomputed embeddings cannot represent.

    - longer than MAX_RESIDUE_LENGTH: ProTrek only embeds the first 1024 residues, so the
      target sequence and the embedding would describe different proteins.
    - containing residues outside the 20 standard amino acids (X, U, B, Z, O): those are
      ambiguous or non-canonical, and the model should not be trained to emit them.
    """
    keep_rows, keep_accessions = [], []
    dropped_length, dropped_alphabet = [], []
    for row, accession in zip(meta["row"].to_numpy(), meta["primaryAccession"].tolist()):
        sequence = reps.at[accession, "sequence"]
        if len(sequence) > MAX_RESIDUE_LENGTH:
            dropped_length.append(row)
        elif not STANDARD_AMINO_ACIDS.issuperset(sequence):
            dropped_alphabet.append(row)
        else:
            keep_rows.append(row)
            keep_accessions.append(accession)

    print(f"candidates: {len(meta)} | kept: {len(keep_rows)}"
          f" | dropped (>{MAX_RESIDUE_LENGTH} residues): {len(dropped_length)}"
          f" | dropped (non-standard residues): {len(dropped_alphabet)}")
    return np.asarray(keep_rows, dtype = np.int64), keep_accessions


def tokenize_samples(tokenizer, rows, accessions, reps, max_seq_length):
    prefix_ids = tokenizer(PROMPT_PREFIX, add_special_tokens = True)["input_ids"]
    samples, dropped_tokens = [], 0
    for row, accession in tqdm(zip(rows.tolist(), accessions),
                               total = len(rows), desc = "tokenize"):
        sequence = reps.at[accession, "sequence"]
        # Tokenise together: the first residue merges into the preceding "<".
        full_ids = tokenizer(PROMPT_PREFIX + sequence, add_special_tokens = True)["input_ids"]
        assert full_ids[:len(prefix_ids)] == prefix_ids, f"prefix changed for {accession}"
        answer_ids = full_ids[len(prefix_ids):]
        if len(prefix_ids) + len(answer_ids) + 1 > max_seq_length:
            dropped_tokens += 1
            continue
        samples.append({"row": int(row), "answer_ids": np.asarray(answer_ids, dtype = np.int32)})

    print("samples:", len(samples), "| dropped (too many tokens):", dropped_tokens)
    return samples


def to_dataset(samples, prefix_length):
    return Dataset.from_dict({
        "row": np.array([sample["row"] for sample in samples], dtype = np.int32),
        "answer_ids": [sample["answer_ids"].tolist() for sample in samples],
        "length": [prefix_length + len(sample["answer_ids"]) + 1 for sample in samples],
    })


class ProteinProjector(torch.nn.Module):
    """ProTrek protein vector -> one ProLLaMA input-embedding slot."""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.norm = torch.nn.LayerNorm(in_features)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_features, out_features),
            torch.nn.GELU(),
            torch.nn.Linear(out_features, out_features),
        )

    def forward(self, protein):
        out = self.mlp(self.norm(protein))
        # Keep the injected vector at the scale of the model's token embeddings:
        # an unconstrained output grows without bound and saturates the LM logits.
        return torch.nn.functional.normalize(out, dim = -1) * PROJECTED_NORM


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    embeddings_path = Path(args.embeddings_dir) / "protrek650m_protein_swissprot_50pct.npy"
    meta_path = Path(args.embeddings_dir) / "protrek650m_protein_swissprot_50pct_meta.parquet"
    reps_prefix = Path(args.data_dir) / "protrek650m_swissprot_50pct_reps"
    for path in (embeddings_path, meta_path, Path(f"{reps_prefix}.parquet")):
        assert path.exists(), f"missing: {path} (run build_dataset.py first)"

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token
    prefix_ids, placeholder_id, placeholder_pos = build_prefix(tokenizer)
    print("prefix:", prefix_ids, "| placeholder pos:", placeholder_pos)

    max_seq_length = args.max_seq_length
    embeddings, meta, reps = load_dataset(embeddings_path, meta_path, reps_prefix)

    # Sequence-level filtering happens before any tokenisation or model load.
    rows, accessions = filter_candidates(meta, reps)
    samples = tokenize_samples(tokenizer, rows, accessions, reps, max_seq_length)
    hf_dataset = to_dataset(samples, len(prefix_ids))

    import unsloth
    from unsloth import FastLanguageModel

    model, processor = FastLanguageModel.from_pretrained(
        model_name = MODEL_NAME,
        max_seq_length = max_seq_length,
        dtype = None,
        load_in_4bit = True,   # NF4 QLoRA: 7B base fits in 8GB
    )
    tokenizer = getattr(processor, "tokenizer", processor)
    model = FastLanguageModel.get_peft_model(
        model, r = 64, lora_alpha = 128, lora_dropout = 0.0, bias = "none",
        use_rslora = False, use_gradient_checkpointing = True, random_state = SEED,
    )
    FastLanguageModel.for_training(model)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs = {"use_reentrant": True})
    model.enable_input_require_grads()  # lets inputs_embeds carry grad under checkpointing

    # Unsloth can offload the embedding table to RAM, so the first parameter may be on
    # CPU. Pick a CUDA device from any parameter instead.
    device = next((p.device for p in model.parameters() if p.device.type == "cuda"),
                  next(model.parameters()).device)
    print("device:", device, "| trainable:", f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Unsloth may rewrite pad/eos: re-derive prefix and re-filter.
    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token
    prefix_ids, placeholder_id, placeholder_pos = build_prefix(tokenizer)
    kept = [s for s in samples if len(prefix_ids) + len(s["answer_ids"]) + 1 <= max_seq_length]
    if len(kept) != len(samples):
        print(f"re-filtered for the new prefix: {len(kept)} / {len(samples)}")
        samples = kept
        hf_dataset = to_dataset(samples, len(prefix_ids))

    embed_tokens = model.get_input_embeddings()
    if embed_tokens is None or not hasattr(embed_tokens, "weight"):
        embed_tokens = next(m for n, m in model.named_modules() if n.endswith("embed_tokens"))
    hidden_size = model.config.hidden_size
    protrek_dim = embeddings.shape[1]
    eos_id = tokenizer.eos_token_id
    projector = ProteinProjector(protrek_dim, hidden_size).to(device = device)

    projector_stats = {}   # latest mean output norm, surfaced in the training log

    def build_inputs_embeds(input_ids, protein, positions):
        embedded = embed_tokens(input_ids)
        projected = projector(protein)
        projector_stats["norm"] = projected.detach().norm(dim = -1).mean().item()
        projected = projected.to(embedded.dtype).unsqueeze(1)
        replace = torch.zeros(input_ids.shape, dtype = torch.bool, device = input_ids.device)
        replace.scatter_(1, positions.unsqueeze(1), True)
        return torch.where(replace.unsqueeze(-1), projected, embedded)

    def collate_fn(features):
        max_length = max(len(prefix_ids) + len(item["answer_ids"]) + 1 for item in features)
        size = len(features)
        input_ids = torch.full((size, max_length), tokenizer.pad_token_id, dtype = torch.long)
        attention_mask = torch.zeros((size, max_length), dtype = torch.long)
        labels = torch.full((size, max_length), -100, dtype = torch.long)
        proteins = torch.empty((size, protrek_dim), dtype = torch.float32)
        for index, item in enumerate(features):
            answer_ids = item["answer_ids"]
            ids = prefix_ids + answer_ids + [eos_id]
            input_ids[index, :len(ids)] = torch.tensor(ids, dtype = torch.long)
            attention_mask[index, :len(ids)] = 1
            labels[index, len(prefix_ids):len(ids)] = torch.tensor(answer_ids + [eos_id], dtype = torch.long)
            proteins[index] = torch.from_numpy(np.array(embeddings[item["row"]], dtype = np.float32))
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "protein": proteins,
            "positions": torch.full((size,), placeholder_pos, dtype = torch.long),
        }

    from trl import SFTConfig, SFTTrainer

    class ProteinToSequenceTrainer(SFTTrainer):
        def __init__(self, *args, projector, projector_stats, **kwargs):
            super().__init__(*args, **kwargs)
            self.projector = projector
            self.projector_stats = projector_stats

        def create_optimizer(self):
            # Projector is outside the PEFT model: own param group.
            optimizer = super().create_optimizer()
            if not getattr(self, "_projector_group_added", False):
                optimizer.add_param_group({
                    "params": [p for p in self.projector.parameters() if p.requires_grad],
                    "lr": LR_PROJECTOR,
                    "weight_decay": PROJECTOR_WEIGHT_DECAY,
                })
                self._projector_group_added = True
            return optimizer

        def log(self, logs, *args, **kwargs):
            if "norm" in self.projector_stats:
                logs = {**logs, "projector_norm": self.projector_stats["norm"]}
            return super().log(logs, *args, **kwargs)

        def compute_loss(self, model, inputs, return_outputs = False, num_items_in_batch = None):
            inputs = dict(inputs)
            input_ids = inputs.pop("input_ids")
            protein = inputs.pop("protein")
            positions = inputs.pop("positions")
            outputs = model(
                inputs_embeds = build_inputs_embeds(input_ids, protein, positions),
                attention_mask = inputs.get("attention_mask"),
                labels = inputs.get("labels"),
                # The fused CE returns sum(CE) / num_items_in_batch when it is given, and
                # Trainer leaves the loss unscaled in that case (it assumes the model did the
                # normalisation). Without this the logged loss and the gradient are each
                # GRAD_ACCUM times too large.
                num_items_in_batch = num_items_in_batch,
                use_cache = False,
                fused_lm_head = True,
            )
            return (outputs.loss, outputs) if return_outputs else outputs.loss

        def _save_checkpoint(self, model, trial):
            super()._save_checkpoint(model, trial)
            torch.save(self.projector.state_dict(),
                       Path(self._get_output_dir(trial)) / "protein_projector.pt")

    trainer = ProteinToSequenceTrainer(
        model = model,
        processing_class = tokenizer,
        data_collator = collate_fn,
        projector = projector,
        projector_stats = projector_stats,
        train_dataset = hf_dataset,
        args = SFTConfig(
            output_dir = str(output_dir / "checkpoints"),
            per_device_train_batch_size = BATCH_SIZE,
            gradient_accumulation_steps = GRAD_ACCUM,
            num_train_epochs = NUM_EPOCHS,
            max_steps = args.max_steps if args.max_steps is not None else -1,
            learning_rate = LR_LORA,
            warmup_steps = 0.03,  # trl 1.15: float warmup_steps is a ratio
            lr_scheduler_type = "cosine",
            optim = "adamw_8bit",  # falls back to "adamw_torch" if bnb has no CUDA
            weight_decay = 0.0,
            bf16 = True,
            logging_steps = 10,
            save_strategy = "steps",
            save_steps = 500,
            save_total_limit = 2,
            seed = SEED,
            report_to = "none",
            dataloader_num_workers = 2,
            dataloader_pin_memory = True,
            remove_unused_columns = False,
            dataset_text_field = "",
            dataset_kwargs = {"skip_prepare_dataset": True},
            max_length = max_seq_length,
            packing = False,
            train_sampling_strategy = "group_by_length",
            gradient_checkpointing = False,
        ),
    )

    resume_from = None
    if args.resume:
        checkpoints = sorted(
            (path for path in (output_dir / "checkpoints").glob("checkpoint-*") if path.is_dir()),
            key = lambda path: int(path.name.split("-")[-1]),
        )
        if checkpoints:
            resume_from = checkpoints[-1]
            projector_path = resume_from / "protein_projector.pt"
            if not projector_path.exists():
                projector_path = output_dir / "checkpoints" / "protein_projector.pt"
            saved = torch.load(projector_path, map_location = device)
            projector.load_state_dict(saved["state_dict"] if "state_dict" in saved else saved)
            print("resuming from:", resume_from)

    trainer.train(resume_from_checkpoint = resume_from)

    output_dir.mkdir(parents = True, exist_ok = True)
    model.save_pretrained(str(output_dir))
    torch.save(
        {
            "state_dict": projector.state_dict(),
            "protrek_dim": protrek_dim,
            "hidden_size": hidden_size,
            "prompt_prefix": PROMPT_PREFIX,
            "prefix_ids": prefix_ids,
            "placeholder": PLACEHOLDER,
            "placeholder_pos": placeholder_pos,
            "embedding_source": "ProTrek_650M_UniRef50",
        },
        output_dir / "protein_projector.pt",
    )
    tokenizer.save_pretrained(str(output_dir))
    print("saved:", sorted(path.name for path in output_dir.iterdir()))

    FastLanguageModel.for_inference(model)
    model.eval()
    projector.eval()
    row = int(samples[0]["row"])
    prefix = torch.tensor(prefix_ids, dtype = torch.long, device = device).unsqueeze(0)
    protein = torch.from_numpy(np.array(embeddings[row], dtype = np.float32)).unsqueeze(0).to(device)
    with torch.no_grad():
        generated = model.generate(
            inputs_embeds = build_inputs_embeds(
                prefix, protein, torch.tensor([placeholder_pos], device = device)),
            attention_mask = torch.ones_like(prefix),
            max_new_tokens = max_seq_length,
            do_sample = False,
            eos_token_id = eos_id,
        )
    print("reference:", tokenizer.decode(samples[0]["answer_ids"], skip_special_tokens = True))
    print("generated:", tokenizer.decode(generated[0], skip_special_tokens = True))


if __name__ == "__main__":
    main()
