"""LoRA-tune ProLLaMA_Stage_1 with ProTrek soft prompts, reading both columns from the Hub.

The prompt carries K projected embedding slots before "Seq=<":
    <s> <unk>*K  ▁Seq = <   sequence >
The projector maps a 1024-d ProTrek vector to K slot embeddings; it is trained alongside the LoRA
and saved next to the adapter. The projector, prefix, answer encoding and the projector's
optimizer/clipping are defined in this file and imported by inference_prollama_from_emb.py, so
training and sampling share one definition (the old local-.npy trainer that used to own them is
legacy/train.py).

Samples are packed into 720-token rows by default (--no-packing to disable): the collator restarts
position_ids at each sample, which is what keeps packed samples from attending to each other.

    uv run python train_prollama_from_emb.py
    uv run python train_prollama_from_emb.py --limit 200 --max-steps 3
"""
import argparse
import math
from pathlib import Path

import unsloth  # noqa: F401  -- must be imported before transformers/trl (unsloth patches them)
import numpy as np
import torch
from datasets import Dataset, load_dataset

# --- shared with inference_prollama_from_emb.py --------------------------------------
# These used to live in train.py (now legacy/train.py, next to the old local-.npy pipeline).
# Keeping one copy here means the projector that training saves and the one inference
# rebuilds cannot drift apart.

INSTRUCTION = "Seq=<"          # ProLLaMA prompt: <s> ▁Seq = <
ANSWER_TERMINATOR = ">"        # the prompt format is Seq=<sequence>; the model closes the bracket
PLACEHOLDER = "<unk>"          # slots whose embeddings receive the projected vectors
SOFT_PROMPT_TOKENS = 10        # ProTrek-derived slots, placed before the instruction
MAX_SEQ_LENGTH = 720
MAX_RESIDUE_LENGTH = 1024      # ProTrek only embeds the first 1024 residues
STANDARD_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")
PROJECTED_NORM = 1.0           # ProLLaMA token embeddings have norm ~1.08
PROJECTOR_WEIGHT_DECAY = 0.01


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
    assert tokenizer.convert_ids_to_tokens(placeholder_id) == PLACEHOLDER, (
        f"{PLACEHOLDER} does not round-trip through the tokenizer")
    prefix_ids = [tokenizer.bos_token_id] + [placeholder_id] * num_tokens + instruction_ids
    positions = list(range(1, num_tokens + 1))
    assert prefix_ids[len(prefix_ids) - len(instruction_ids):] == instruction_ids, (
        "the instruction tokens must be unchanged by the soft prompt")
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


class ProjectorTrainerMixin:
    """Optimizer, clipping and zeroing for a projector that lives outside the PEFT model.

    Trainer only builds param groups, clips and clears gradients over model.parameters(), and the
    projector is not part of the model, so all three have to be extended by hand. Subclasses set
    `projector` and `projector_lr`.
    """

    def create_optimizer(self):
        optimizer = super().create_optimizer()
        if not getattr(self, "_projector_group_added", False):
            optimizer.add_param_group({
                "params": [p for p in self.projector.parameters() if p.requires_grad],
                "lr": self.projector_lr,
                "weight_decay": PROJECTOR_WEIGHT_DECAY,
            })
            self._projector_group_added = True
        return optimizer

    def _clip_grad_norm(self, model):
        # The projector's gradient would otherwise be neither clipped nor reach the log.
        parameters = [p for p in self.projector.parameters() if p.requires_grad]
        model_norm = float(super()._clip_grad_norm(model))
        # Clip the two groups separately. The model path keeps exactly the clip it had before;
        # the projector, whose gradient is several times larger, gets the same max_grad_norm of
        # its own. Disjoint parameters: the reported total is the quadrature sum.
        projector_norm = float(torch.nn.utils.clip_grad_norm_(parameters, self.args.max_grad_norm))
        return (model_norm ** 2 + projector_norm ** 2) ** 0.5

MODEL_NAME = "GreatCaptainNemo/ProLLaMA_Stage_1"
DATASET_REPO = "yosshstd/protrek650m-swissprot50-reps"
DATASET_SPLIT = "train"

SEED = 3407


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default = MODEL_NAME)
    parser.add_argument("--dataset", default = DATASET_REPO)
    parser.add_argument("--split", default = DATASET_SPLIT, choices = ["train", "test"])
    parser.add_argument("--limit", type = int, default = None)
    parser.add_argument("--max-seq-length", type = int, default = MAX_SEQ_LENGTH)
    parser.add_argument("--num-tokens", type = int, default = SOFT_PROMPT_TOKENS)
    parser.add_argument("--output-dir", default = "outputs/prollama_stage1_emb")
    parser.add_argument("--max-steps", type = int, default = None)
    parser.add_argument("--resume", action = "store_true")
    parser.add_argument("--lora-r", type = int, default = 128)
    parser.add_argument("--lora-alpha", type = int, default = 256)
    parser.add_argument("--learning-rate", type = float, default = 2e-4)
    parser.add_argument("--projector-lr", type = float, default = 2e-4)
    parser.add_argument("--batch-size", type = int, default = 1)
    parser.add_argument("--grad-accum", type = int, default = 2)
    parser.add_argument("--epochs", type = float, default = 1.0)
    parser.add_argument("--gradient-checkpointing", default = "unsloth",
                        choices = ["unsloth", "standard", "off"])
    parser.add_argument("--no-4bit", dest = "load_in_4bit", action = "store_false",
                        help = "load the base model in fp16/bf16 instead of 4-bit NF4 "
                               "(much more VRAM; 8 GB cards should keep the default)")
    parser.add_argument("--no-packing", dest = "packing", action = "store_false",
                        help = "one sample per row, padded to the longest in the batch")
    return parser.parse_args()


def pack_groups(lengths, max_length):
    """Group sample indices so each group's total length fits in max_length.

    Longest first into the first bin with room (first-fit decreasing). TRL's own BFD produces
    the same bins, but it packs arrays in the dataset and would drop the sample index the
    collator needs to look the protein vector up, so the binning is done here.
    """
    groups = []   # [remaining space, [indices]]
    for index in sorted(range(len(lengths)), key = lambda index: -lengths[index]):
        length = lengths[index]
        for group in groups:
            if group[0] >= length:
                group[0] -= length
                group[1].append(index)
                break
        else:
            groups.append([max_length - length, [index]])
    return [sorted(indices) for _, indices in groups]


def load_rows(repo_id, split, limit = None):
    """Sequence and 1024-d ProTrek vector of every row, filtered by the rules build_dataset.py applies."""
    dataset = load_dataset(repo_id, split = split, columns = ["sequence", "embedding"])
    if limit is not None:
        dataset = dataset.select(range(limit))
    all_sequences = dataset["sequence"]
    sequences, embed_rows = [], []
    for index, sequence in enumerate(all_sequences):
        if len(sequence) > MAX_RESIDUE_LENGTH:
            continue
        if not STANDARD_AMINO_ACIDS.issuperset(sequence):
            continue
        sequences.append(sequence)
        embed_rows.append(index)
    embeddings = np.asarray(dataset["embedding"], dtype = np.float32)
    print(f"candidates: {len(dataset):,} | kept: {len(sequences):,}")
    return sequences, embeddings, np.asarray(embed_rows, dtype = np.int64)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    num_tokens = args.num_tokens
    # Read here: inside the Trainer subclass below, `args` is shadowed by *args.
    projector_lr = args.projector_lr

    sequences, embeddings, embed_rows = load_rows(args.dataset, args.split, args.limit)
    from unsloth import FastLanguageModel

    model, processor = FastLanguageModel.from_pretrained(
        model_name = args.model,
        max_seq_length = args.max_seq_length,
        dtype = None,
        load_in_4bit = args.load_in_4bit,
    )
    tokenizer = getattr(processor, "tokenizer", processor)
    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token

    prefix_ids, placeholder_positions = build_prefix(tokenizer, num_tokens)
    # The collator places the slots at a fixed offset from the sample start, so they must be the
    # contiguous run right after BOS that build_prefix produces.
    assert placeholder_positions == list(range(1, 1 + num_tokens)), placeholder_positions
    print("prefix:", prefix_ids, "| slots:", placeholder_positions)

    samples, dropped = [], 0
    for sequence, embed_row in zip(sequences, embed_rows.tolist()):
        answer_ids = encode_answer(tokenizer, sequence)
        if len(prefix_ids) + len(answer_ids) + 1 > args.max_seq_length:
            dropped += 1
            continue
        samples.append({"row": embed_row, "answer_ids": answer_ids})
    print(f"samples: {len(samples):,} | dropped (too many tokens): {dropped:,}")
    assert samples, "no sample survived the filters; raise --max-seq-length"

    lengths = [len(prefix_ids) + len(sample["answer_ids"]) + 1 for sample in samples]
    groups = pack_groups(lengths, args.max_seq_length) if args.packing else \
             [[index] for index in range(len(samples))]
    print(f"rows: {len(groups):,} for {len(samples):,} samples "
          f"({len(samples) / len(groups):.2f} samples/row)")
    # One dataset row per group, so the shuffle and the step count operate on groups. The
    # collator keeps `rows` intact; it is the sample indices the projector is looked up by.
    hf_dataset = Dataset.from_list([{"rows": group} for group in groups])

    model = FastLanguageModel.get_peft_model(
        model,
        r = args.lora_r,
        lora_alpha = args.lora_alpha,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
        lora_dropout = 0.0,
        bias = "none",
        use_gradient_checkpointing = {"unsloth": "unsloth",
                                      "standard": True,
                                      "off": False}[args.gradient_checkpointing],
        use_rslora = False,
        random_state = SEED,
    )
    FastLanguageModel.for_training(model)
    if args.gradient_checkpointing != "off":
        model.enable_input_require_grads()   # inputs_embeds carries grad through checkpointing

    device = next((p.device for p in model.parameters() if p.device.type == "cuda"),
                  next(model.parameters()).device)
    embed_tokens = model.get_input_embeddings()
    hidden_size = model.config.hidden_size
    eos_id = tokenizer.eos_token_id
    protrek_dim = embeddings.shape[1]

    mean = embedding_mean(embeddings, embed_rows)
    projector = ProteinProjector(protrek_dim, hidden_size, num_tokens, mean = mean).to(device = device)
    print("device:", device, "| projector params:",
          f"{sum(p.numel() for p in projector.parameters()):,}")

    def build_inputs_embeds(input_ids, protein, rows, offsets):
        """Scatter each sample's K projected slots into its own place in the (packed) row.

        `rows` and `offsets` index the samples packed into the batch: sample i sits in row
        `rows[i]` at token `offsets[i]`, and its slots take the K tokens after that offset.
        A single unpadded sample is the ordinary case (row 0, offset 1).
        """
        embedded = embed_tokens(input_ids)                          # (B, L, H)
        projected = projector(protein).to(embedded.dtype)           # (N, K, H)
        slot_index = (rows.view(-1, 1) * input_ids.shape[1]
                      + offsets.view(-1, 1)
                      + torch.arange(num_tokens, device = input_ids.device))
        # Out-of-place index_put: the untouched positions keep their gradient path through
        # embed_tokens, exactly like the clone() + scatter_ it replaces.
        merged = embedded.reshape(-1, embedded.shape[-1])
        merged = merged.index_put((slot_index.reshape(-1),), projected.reshape(-1, merged.shape[-1]))
        return merged.view(embedded.shape)

    def collate_fn(features):
        """One batch row per packed group; position_ids restart at every sample in the row.

        The restart is what marks a sample boundary. With attention_mask=None and
        use_cache=False, transformers turns position_ids into a block-diagonal causal mask, so
        a packed row really is several independent sequences rather than one long one; a 2D
        attention_mask would switch that detection off and leak across samples, so none is
        returned. Verified numerically: a packed row's loss equals the token-weighted loss of
        its samples fed one at a time.
        """
        row_lengths = [sum(lengths[index] for index in feature["rows"]) for feature in features]
        length = max(row_lengths)
        size = len(features)
        input_ids = torch.full((size, length), tokenizer.pad_token_id, dtype = torch.long)
        # A monotonic default, so a pad tail reads as a continuation of the last sample rather
        # than one tiny segment per pad token. Its labels are -100 and causality keeps the real
        # tokens from ever attending to it, so the tail cannot affect the loss.
        position_ids = torch.arange(length).repeat(size, 1)
        labels = torch.full((size, length), -100, dtype = torch.long)
        slot_rows, slot_offsets = [], []
        for row, feature in enumerate(features):
            offset = 0
            for index in feature["rows"]:
                sample = samples[index]
                ids = prefix_ids + sample["answer_ids"] + [eos_id]
                span = slice(offset, offset + lengths[index])
                input_ids[row, span] = torch.tensor(ids, dtype = torch.long)
                position_ids[row, span] = torch.arange(lengths[index])
                # The prefix (BOS and slots included) stays -100, so the shift at a sample
                # boundary never predicts a token across two proteins.
                labels[row, offset + len(prefix_ids):offset + lengths[index]] = torch.tensor(
                    sample["answer_ids"] + [eos_id], dtype = torch.long)
                # Slots sit right after each sample's BOS, wherever that sample starts.
                slot_rows.append(row)
                slot_offsets.append(offset + 1)
                offset += lengths[index]
        return {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "labels": labels,
            "protein": torch.from_numpy(embeddings[
                [samples[index]["row"] for feature in features for index in feature["rows"]]]),
            "slot_rows": torch.tensor(slot_rows, dtype = torch.long),
            "slot_offsets": torch.tensor(slot_offsets, dtype = torch.long),
        }

    from transformers import TrainerCallback

    class ProjectorGradReset(TrainerCallback):
        """Trainer's zero_grad never reaches the projector, which sits outside the PEFT model."""

        def on_optimizer_step(self, args, state, control, **kwargs):
            projector.zero_grad(set_to_none = True)

    from trl import SFTConfig, SFTTrainer

    class ProteinToSequenceTrainer(ProjectorTrainerMixin, SFTTrainer):
        def __init__(self, *trainer_args, **trainer_kwargs):
            super().__init__(*trainer_args, **trainer_kwargs)
            self.projector = projector
            self.projector_lr = projector_lr

        def compute_loss(self, model, inputs, return_outputs = False, num_items_in_batch = None):
            input_ids = inputs["input_ids"]
            outputs = model(
                inputs_embeds = build_inputs_embeds(
                    input_ids, inputs["protein"], inputs["slot_rows"], inputs["slot_offsets"]),
                # Restarted per sample: this, not an attention mask, is what isolates the packed
                # samples (see collate_fn).
                position_ids = inputs["position_ids"],
                labels = inputs["labels"],
                # The count keeps the fused CE on sum(CE)/count; without it Trainer does not
                # rescale by GRAD_ACCUM and the gradients come out GRAD_ACCUM times too small.
                num_items_in_batch = num_items_in_batch,
                use_cache = False,
            )
            return (outputs.loss, outputs) if return_outputs else outputs.loss

        def _save_checkpoint(self, model, trial):
            super()._save_checkpoint(model, trial)
            checkpoint_dir = Path(self._get_output_dir(trial)) / f"checkpoint-{self.state.global_step}"
            torch.save(projector_state(), checkpoint_dir / "protein_projector.pt")

    def projector_state():
        """Everything needed to rebuild this projector; inference_prollama_from_emb.py reads it."""
        return {"state_dict": projector.state_dict(), "protrek_dim": protrek_dim,
                "hidden_size": hidden_size, "num_tokens": num_tokens,
                "instruction": INSTRUCTION, "placeholder": PLACEHOLDER,
                "prefix_ids": prefix_ids, "placeholder_positions": placeholder_positions,
                "embedding_mean": projector.mean.detach().cpu(),
                "embedding_source": "ProTrek_650M_UniRef50"}

    trainer = ProteinToSequenceTrainer(
        model = model,
        processing_class = tokenizer,
        data_collator = collate_fn,
        callbacks = [ProjectorGradReset()],
        train_dataset = hf_dataset,
        args = SFTConfig(
            output_dir = str(output_dir / "checkpoints"),
            max_length = args.max_seq_length,
            # The collator packs; TRL's own packing would rewrite the dataset and drop the row
            # index the projector needs, so it stays off here.
            packing = False,
            per_device_train_batch_size = args.batch_size,
            gradient_accumulation_steps = args.grad_accum,
            num_train_epochs = args.epochs,
            max_steps = args.max_steps if args.max_steps is not None else -1,
            learning_rate = args.learning_rate,
            lr_scheduler_type = "cosine",
            warmup_steps = 0.03,  # trl 1.15: float warmup_steps is a ratio
            weight_decay = 0.001,
            optim = "adamw_8bit",
            bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
            fp16 = not (torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
            seed = SEED,
            logging_steps = 10,
            save_strategy = "steps",
            save_steps = 500,
            save_total_limit = 1,
            report_to = "none",
            dataloader_num_workers = 2,
            dataloader_pin_memory = True,
            remove_unused_columns = False,
            dataset_kwargs = {"skip_prepare_dataset": True},
            gradient_checkpointing = False,   # model side enabled by get_peft_model
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
            saved = torch.load(resume_from / "protein_projector.pt", map_location = device)
            assert saved["num_tokens"] == num_tokens, (
                f"checkpoint used {saved['num_tokens']} slots, not {num_tokens}")
            projector.load_state_dict(saved["state_dict"])
            print("resuming from:", resume_from)

    trainer.train(resume_from_checkpoint = str(resume_from) if resume_from else None)

    output_dir.mkdir(parents = True, exist_ok = True)
    model.save_pretrained(str(output_dir))
    torch.save(projector_state(), output_dir / "protein_projector.pt")
    tokenizer.save_pretrained(str(output_dir))
    print("saved:", sorted(path.name for path in output_dir.iterdir()))


if __name__ == "__main__":
    main()