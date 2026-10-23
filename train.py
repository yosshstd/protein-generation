"""LoRA-tune ProLLaMA_Stage_1 to generate amino acid sequences from ProTrek_650M embeddings.

Run build_dataset.py first.

The ProTrek vector is injected as K soft-prompt slots (default 10) placed before the
"Seq=<" instruction, so the instruction keeps the exact form the base model was trained on.

Usage: python train.py [--max-steps N] [--resume] [--num-tokens 10]
"""
import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from datasets import Dataset
from tqdm.auto import tqdm

MODEL_NAME = "GreatCaptainNemo/ProLLaMA_Stage_1"
SOFT_PROMPT_TOKENS = 10        # ProTrek-derived slots, placed before the instruction
INSTRUCTION = "Seq=<"          # ProLLaMA prompt: <s> ▁Seq = <
ANSWER_TERMINATOR = ">"        # the prompt format is Seq=<sequence>; the model closes the bracket
PLACEHOLDER = "<unk>"          # slots whose embeddings receive the projected vectors
MAX_SEQ_LENGTH = 720
MAX_RESIDUE_LENGTH = 1024     # ProTrek only embeds the first 1024 residues
STANDARD_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")
SEED = 0

BATCH_SIZE = 1
GRAD_ACCUM = 2
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
    parser.add_argument("--num-tokens", type = int, default = SOFT_PROMPT_TOKENS)
    parser.add_argument("--max-steps", type = int, default = None)
    parser.add_argument("--resume", action = "store_true")
    return parser.parse_args()


def build_prefix(tokenizer, num_tokens = SOFT_PROMPT_TOKENS):
    """Return (prefix_ids, placeholder_positions).

    Layout: <s> <unk>*K ▁Seq = <
    The placeholder slots sit *before* the instruction so the instruction keeps the exact
    tokenisation the base model was pre-trained on.

    The ids are built explicitly rather than by tokenising "<unk>"*K + "Seq=<": the fast
    tokenizer turns the placeholder run into a different leading-space form, so "Seq" in
    "<unk><unk>Seq=<" becomes a bare "Seq" (23718) instead of the pre-training "▁Seq" (25981).
    """
    instruction_ids = tokenizer(INSTRUCTION, add_special_tokens = False)["input_ids"]
    placeholder_id = tokenizer.convert_tokens_to_ids(PLACEHOLDER)
    assert tokenizer.bos_token_id is not None, "tokenizer has no BOS token"
    assert tokenizer.convert_ids_to_tokens(placeholder_id) == PLACEHOLDER, \
        f"{PLACEHOLDER} does not round-trip through the tokenizer"
    prefix_ids = [tokenizer.bos_token_id] + [placeholder_id] * num_tokens + instruction_ids
    positions = list(range(1, num_tokens + 1))
    assert prefix_ids[len(prefix_ids) - len(instruction_ids):] == instruction_ids, \
        "the instruction tokens must be unchanged by the soft prompt"
    return prefix_ids, positions


def encode_answer(tokenizer, sequence, terminator = ANSWER_TERMINATOR):
    """Token ids of `sequence` as it appears after the instruction, closing bracket included.

    Tokenising "Seq=<" + sequence (rather than the sequence alone) is what makes the first
    residue come out the way the model will produce it. The closing ">" is part of the answer:
    the pre-training format is "Seq=<sequence>", so the model emits ">" to close the bracket,
    and training on sequences without it taught the model to stop at a token it never saw.
    """
    instruction_ids = tokenizer(INSTRUCTION, add_special_tokens = False)["input_ids"]
    full_ids = tokenizer(INSTRUCTION + sequence + terminator, add_special_tokens = False)["input_ids"]
    assert full_ids[:len(instruction_ids)] == instruction_ids, "instruction changed by the sequence"
    answer_ids = full_ids[len(instruction_ids):]
    assert answer_ids, f"empty answer for a {len(sequence)}-residue sequence"
    return answer_ids


def split_terminator(text, terminator = ANSWER_TERMINATOR):
    """Split generated text into (sequence, closed).

    The model writes "Seq=<sequence>", so a trailing ">" is the bracket closing, not a residue.
    Anything after it is not part of the answer.
    """
    index = text.find(terminator)
    if index < 0:
        return text, False
    return text[:index], True


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


def tokenize_samples(tokenizer, prefix_ids, rows, accessions, reps, max_seq_length):
    """Tokenise the answer sequence and keep samples that fit in max_seq_length."""
    samples, dropped_tokens = [], 0
    for row, accession in tqdm(zip(rows.tolist(), accessions),
                               total = len(rows), desc = "tokenize"):
        sequence = reps.at[accession, "sequence"]
        answer_ids = encode_answer(tokenizer, sequence)
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


def embedding_mean(embeddings, rows, chunk = 20000):
    """Mean ProTrek vector.

    Every vector is L2-normalised, but they share a large common component (the mean vector
    has norm ~0.48, against 1.0 for each vector). Projecting the raw vector lets that common
    direction dominate, so the projector subtracts the mean first.
    """
    total = np.zeros(embeddings.shape[1], dtype = np.float64)
    for start in range(0, len(rows), chunk):
        total += np.asarray(embeddings[rows[start:start + chunk]], dtype = np.float64).sum(axis = 0)
    return (total / len(rows)).astype(np.float32)


class ProteinProjector(torch.nn.Module):
    """ProTrek protein vector -> K ProLLaMA input-embedding slots.

    One output head per slot, so the K vectors can specialise instead of collapsing onto a
    single direction.

    The output is LayerNorm-normalised rather than L2-normalised on purpose. The Jacobian of
    F.normalize scales as 1/||out||, so once the raw output drifts to a large norm its
    gradient vanishes and it can never recover (measured on the previous run: the projector
    gradient was 1.4e-3 of the LoRA gradient, against 8.6e-2 with LayerNorm).
    """

    def __init__(self, in_features, out_features, num_tokens, mean = None, width = None):
        super().__init__()
        self.num_tokens = num_tokens
        # Trunk width defaults to half the output width (2048 for a 4096-d model).
        width = out_features // 2 if width is None else width
        self.width = width
        self.register_buffer("mean", torch.zeros(in_features) if mean is None else torch.as_tensor(mean))
        self.input_norm = torch.nn.LayerNorm(in_features)
        self.trunk = torch.nn.Sequential(
            torch.nn.Linear(in_features, width),
            torch.nn.GELU(),
            torch.nn.Linear(width, width),
            torch.nn.GELU(),
        )
        self.heads = torch.nn.ModuleList(
            torch.nn.Linear(width, out_features) for _ in range(num_tokens))
        self.output_norm = torch.nn.LayerNorm(out_features)
        # LayerNorm leaves per-dimension std ~1; rescale to the token-embedding scale.
        self.register_buffer("scale", torch.tensor(PROJECTED_NORM / math.sqrt(out_features)))

    def forward(self, protein):
        hidden = self.trunk(self.input_norm(protein - self.mean))
        slots = torch.stack([head(hidden) for head in self.heads], dim = 1)   # (B, K, H)
        return self.output_norm(slots) * self.scale


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    embeddings_path = Path(args.embeddings_dir) / "protrek650m_protein_swissprot_50pct.npy"
    meta_path = Path(args.embeddings_dir) / "protrek650m_protein_swissprot_50pct_meta.parquet"
    reps_prefix = Path(args.data_dir) / "protrek650m_swissprot_50pct_reps"
    for path in (embeddings_path, meta_path, Path(f"{reps_prefix}.parquet")):
        assert path.exists(), f"missing: {path} (run build_dataset.py first)"

    num_tokens = args.num_tokens
    max_seq_length = args.max_seq_length
    embeddings, meta, reps = load_dataset(embeddings_path, meta_path, reps_prefix)

    # Sequence-level filtering happens before any tokenisation or model load.
    rows, accessions = filter_candidates(meta, reps)

    import unsloth
    from unsloth import FastLanguageModel

    # The model is loaded before tokenising on purpose: the dataset has to be tokenised with
    # the tokenizer the model itself will use, and Unsloth does not hand it back until now.
    model, processor = FastLanguageModel.from_pretrained(
        model_name = MODEL_NAME,
        max_seq_length = max_seq_length,
        dtype = None,
        load_in_4bit = True,   # NF4 QLoRA: 7B base fits in 8GB
    )
    tokenizer = getattr(processor, "tokenizer", processor)
    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token
    prefix_ids, placeholder_positions = build_prefix(tokenizer, num_tokens)
    print("prefix:", prefix_ids, "| placeholder positions:", placeholder_positions)

    samples = tokenize_samples(tokenizer, prefix_ids, rows, accessions, reps, max_seq_length)
    hf_dataset = to_dataset(samples, len(prefix_ids))

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

    embed_tokens = model.get_input_embeddings()
    if embed_tokens is None or not hasattr(embed_tokens, "weight"):
        embed_tokens = next(m for n, m in model.named_modules() if n.endswith("embed_tokens"))
    hidden_size = model.config.hidden_size
    protrek_dim = embeddings.shape[1]
    eos_id = tokenizer.eos_token_id
    mean = embedding_mean(embeddings, rows)
    projector = ProteinProjector(protrek_dim, hidden_size, num_tokens, mean = mean).to(device = device)
    print("projector params:", f"{sum(p.numel() for p in projector.parameters()):,}")

    projector_stats = {}   # latest slot diagnostics, surfaced in the training log

    def projector_checkpoint():
        """Everything needed to rebuild this projector; inference.py reads the metadata."""
        return {
            "state_dict": projector.state_dict(),
            "protrek_dim": protrek_dim,
            "hidden_size": hidden_size,
            "num_tokens": num_tokens,
            "instruction": INSTRUCTION,
            "prefix_ids": prefix_ids,
            "placeholder": PLACEHOLDER,
            "placeholder_positions": placeholder_positions,
            "embedding_mean": projector.mean.detach().cpu(),
            "projected_norm": PROJECTED_NORM,
            "embedding_source": "ProTrek_650M_UniRef50",
        }

    def build_inputs_embeds(input_ids, protein, positions):
        embedded = embed_tokens(input_ids)                       # (B, L, H)
        projected = projector(protein).to(embedded.dtype)        # (B, K, H)
        if positions.dim() == 1:
            positions = positions.unsqueeze(0)
        with torch.no_grad():
            # Mean cosine between the K slot vectors: near 1.0 means the slots are redundant
            # and the soft prompt carries no more than a single slot would.
            flat = torch.nn.functional.normalize(projected[0].float(), dim = -1)
            cosine = flat @ flat.T
            off_diagonal = ~torch.eye(len(cosine), dtype = bool, device = cosine.device)
            projector_stats["slot_cos"] = cosine[off_diagonal].mean().item()
            projector_stats["norm"] = projected[0].float().norm(dim = -1).mean().item()
        # Scatter rather than torch.where: projected is (B, K, H), which does not broadcast
        # against (B, L, H). clone() keeps the gradient path through embed_tokens intact.
        merged = embedded.clone()
        index = positions.expand(input_ids.shape[0], -1).unsqueeze(-1)
        merged.scatter_(1, index.expand(-1, -1, merged.shape[-1]), projected)
        return merged

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
            "positions": torch.tensor(placeholder_positions, dtype = torch.long),
        }

    from transformers import TrainerCallback

    class ProjectorGradReset(TrainerCallback):
        """Zero the projector's grads after every optimizer step.

        Trainer clears gradients with model.zero_grad(), which only reaches parameters inside
        the PEFT model. The projector is a separate module, so without this its gradient
        accumulates across steps and the effective lr grows without bound.
        """

        def __init__(self, projector):
            self.projector = projector

        def on_optimizer_step(self, args, state, control, **kwargs):
            self.projector.zero_grad(set_to_none = True)

    from trl import SFTConfig, SFTTrainer

    class ProteinToSequenceTrainer(SFTTrainer):
        def __init__(self, *args, projector, projector_stats, **kwargs):
            super().__init__(*args, **kwargs)
            self.projector = projector
            self.projector_stats = projector_stats
            # compute_loss below hands num_items_in_batch to the fused CE head, which returns
            # sum(CE) / num_items_in_batch. Trainer must therefore *not* divide the loss by
            # GRAD_ACCUM again, and it only skips that division while this flag is True.
            # Trainer/Unsloth/PEFT decide the flag by inspecting the model's forward
            # signature, so it can silently come out False. Should that happen *and* the count
            # still arrive, Trainer divides the already-normalised loss by GRAD_ACCUM a second
            # time (reproduced on a toy Trainer: 0.02 became 0.01 at GRAD_ACCUM=2). On this
            # exact stack the flag resolves True and the logged loss matched the token-weighted
            # reference to 0.3%; forcing it False made unsloth's get_batch_samples blank the
            # count instead, so the double division did not actually occur here. Pinning the
            # flag keeps the pairing explicit rather than depending on that guard in a library
            # we do not control.
            resolved = self.model_accepts_loss_kwargs
            self.model_accepts_loss_kwargs = True
            print(f"model_accepts_loss_kwargs: {resolved} -> True "
                  "(compute_loss passes the token count, so Trainer must not rescale by GRAD_ACCUM)")
            self._normalisation_reported = False

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

        def _clip_grad_norm(self, model):
            # Trainer clips and zeroes model.parameters() only, and the projector is outside the
            # model, so without this its gradient would never be clipped and never be cleared:
            # it would accumulate over every optimizer step (measured on a toy replica: the grad
            # norm grew 1x, 2x, 3x, 4x across four identical steps). Zeroing happens in
            # ProjectorGradReset below, since Trainer calls model.zero_grad() directly.
            projector_parameters = [p for p in self.projector.parameters() if p.requires_grad]
            # Logged before the clip so the value is the raw projector gradient.
            self.projector_stats["grad_norm"] = float(
                torch.nn.utils.get_total_norm([p.grad for p in projector_parameters if p.grad is not None]))
            # Clip the two groups separately. The model path keeps exactly the clip it had
            # before; the projector, whose gradient is several times larger, gets the same
            # max_grad_norm of its own.
            model_norm = float(super()._clip_grad_norm(model))
            projector_norm = float(torch.nn.utils.clip_grad_norm_(
                projector_parameters, self.args.max_grad_norm))
            # Disjoint parameters: the reported total is the quadrature sum.
            return (model_norm ** 2 + projector_norm ** 2) ** 0.5

        def log(self, logs, *args, **kwargs):
            stats = {f"projector_{key}": value for key, value in self.projector_stats.items()}
            return super().log({**logs, **stats}, *args, **kwargs)

        def compute_loss(self, model, inputs, return_outputs = False, num_items_in_batch = None):
            inputs = dict(inputs)
            input_ids = inputs.pop("input_ids")
            protein = inputs.pop("protein")
            positions = inputs.pop("positions")
            # The contract with Trainer has two halves, and they only balance as a pair:
            #  - count given: the fused CE returns sum(CE) / count and Trainer leaves the loss
            #    alone (it only does so when model_accepts_loss_kwargs is True, which __init__
            #    pins);
            #  - count missing: the fused CE falls back to a micro-batch mean and Trainer divides
            #    by the accumulation steps itself.
            # Scoring the batch with the count while Trainer also divides by GRAD_ACCUM is the
            # bug this guards against: the loss and every gradient come out 1/GRAD_ACCUM too
            # small, and the run looks like it is learning more slowly than it is.
            if not self._normalisation_reported:
                self._normalisation_reported = True
                print("loss normalisation:", "sum(CE)/num_items_in_batch, Trainer does not rescale"
                      if num_items_in_batch is not None else
                      "micro-batch mean, Trainer divides by the accumulation steps")
            # With the flag pinned True, Trainer never divides by the accumulation steps, so the
            # count *must* arrive: without it the fused CE returns a micro-batch mean and Trainer
            # leaves it alone, making the loss and gradients GRAD_ACCUM times too large. (The
            # opposite mismatch — count given, flag False — would make them too small; pinning
            # the flag True rules it out.) Eval calls this without a count, hence the guard.
            assert (num_items_in_batch is not None
                    or not self.model.training
                    or self.args.gradient_accumulation_steps == 1), (
                "num_items_in_batch is missing under gradient accumulation while "
                "model_accepts_loss_kwargs is True: the fused CE would return a micro-batch "
                "mean that Trainer does not rescale, scaling loss and gradients by GRAD_ACCUM")
            outputs = model(
                inputs_embeds = build_inputs_embeds(input_ids, protein, positions),
                attention_mask = inputs.get("attention_mask"),
                labels = inputs.get("labels"),
                num_items_in_batch = num_items_in_batch,
                use_cache = False,
                fused_lm_head = True,
            )
            return (outputs.loss, outputs) if return_outputs else outputs.loss

        def _save_checkpoint(self, model, trial):
            super()._save_checkpoint(model, trial)
            # The metadata travels with every checkpoint: inference.py rebuilds the projector
            # (slot count included) from it, so a bare state_dict is not enough.
            # _get_output_dir() is the *run* directory, not the checkpoint directory — Trainer
            # builds the "checkpoint-N" part locally — so reconstruct it here. Writing to the
            # run directory would leave every checkpoint sharing one projector file, and a
            # resume would then load a projector from the wrong step.
            checkpoint_dir = Path(self._get_output_dir(trial)) / f"checkpoint-{self.state.global_step}"
            torch.save(projector_checkpoint(), checkpoint_dir / "protein_projector.pt")

    trainer = ProteinToSequenceTrainer(
        model = model,
        processing_class = tokenizer,
        data_collator = collate_fn,
        callbacks = [ProjectorGradReset(projector)],
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
            weight_decay = 0.01,
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
            state = saved["state_dict"] if "state_dict" in saved else saved
            saved_tokens = int(saved["num_tokens"]) if isinstance(saved, dict) and "num_tokens" in saved else 1
            assert saved_tokens == num_tokens, (
                f"{projector_path} was trained with {saved_tokens} slots, "
                f"but --num-tokens is {num_tokens}; pass --num-tokens {saved_tokens} to resume it")
            projector.load_state_dict(state)
            print("resuming from:", resume_from)

    trainer.train(resume_from_checkpoint = resume_from)

    output_dir.mkdir(parents = True, exist_ok = True)
    model.save_pretrained(str(output_dir))
    torch.save(projector_checkpoint(), output_dir / "protein_projector.pt")
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
                prefix, protein,
                torch.tensor(placeholder_positions, dtype = torch.long, device = device).unsqueeze(0)),
            attention_mask = torch.ones_like(prefix),
            max_new_tokens = max_seq_length - len(prefix_ids),
            do_sample = False,
            eos_token_id = eos_id,
        )
    reference = tokenizer.decode(samples[0]["answer_ids"], skip_special_tokens = True)
    text, closed = split_terminator(tokenizer.decode(generated[0], skip_special_tokens = True))
    print("reference:", reference)
    print("generated:", text, f"| closed: {closed}")


if __name__ == "__main__":
    main()
