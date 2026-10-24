"""Build the ProTrek-650M Swiss-Prot 50% representative dataset and publish it to the Hub.

One pipeline, start to finish:

1. download `westlake-repl/faiss_index` (Swiss-Prot sequences + their ProTrek vectors)
2. cluster at 50% sequence identity with MMseqs2 and keep the representatives
3. filter to rows a ProTrek-conditioned sequence model can actually learn from
4. split 100k train / 11k test without letting sequence-identity groups cross the split
5. write one self-contained parquet per split, plus a dataset card and a manifest
6. optionally upload the lot to the Hub (`--upload`)

`--legacy-dir` additionally writes the `.npy` + metadata layout that `legacy/train.py` memory-maps;
pass `--no-legacy` to skip it.

Usage:
    python build_dataset.py                      # build only
    python build_dataset.py --upload             # build, then publish
    python build_dataset.py --threads 16 --upload
"""
import argparse
import hashlib
import json
import subprocess
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import snapshot_download
from tqdm.auto import tqdm

MAX_RESIDUE_LENGTH = 1024  # ProTrek embeds 1024 residues + <cls> + <eos>
STANDARD_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")
EMBEDDING_DIM = 1024
TRAIN_SIZE = 100_000
SEED = 42
DATASET_REPO = "yosshstd/protrek650m-swissprot50-reps"
UPSTREAM_REPO = "westlake-repl/faiss_index"
UPSTREAM_SUBDIR = "SwissProt/ProTrek_650M_UniRef50/sequence"

# Length buckets used to keep train and test comparable. Rows over MAX_RESIDUE_LENGTH are
# filtered out before the split, so the last bucket is closed at MAX_RESIDUE_LENGTH + 1.
BUCKETS = [(0, 50), (50, 100), (100, 200), (200, 400), (400, 800), (800, MAX_RESIDUE_LENGTH + 1)]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default = "data")
    parser.add_argument("--work-dir", default = "work")
    parser.add_argument("--output-dir", default = "outputs/hf_dataset")
    parser.add_argument("--legacy-dir", default = "outputs/embeddings")
    parser.add_argument("--no-legacy", action = "store_true",
                        help = "skip the .npy + metadata files legacy/train.py memory-maps")
    parser.add_argument("--mmseqs", default = "tools/mmseqs/bin/mmseqs")
    parser.add_argument("--threads", type = int, default = 16)
    parser.add_argument("--force-cluster", action = "store_true")
    parser.add_argument("--seed", type = int, default = SEED)
    parser.add_argument("--train-size", type = int, default = TRAIN_SIZE)
    parser.add_argument("--repo-id", default = DATASET_REPO)
    parser.add_argument("--upload", action = "store_true",
                        help = "publish the built files to --repo-id (outward-facing)")
    parser.add_argument("--private", action = "store_true",
                        help = "create the Hub repo private instead of public")
    return parser.parse_args()


def download(data_dir):
    local = Path(data_dir) / UPSTREAM_SUBDIR
    ids_path, index_path = local / "ids.tsv", local / "sequence.index"
    if ids_path.exists() and index_path.exists():
        return ids_path, index_path
    root = Path(snapshot_download(
        repo_id = UPSTREAM_REPO,
        repo_type = "dataset",
        allow_patterns = [f"{UPSTREAM_SUBDIR}/ids.tsv", f"{UPSTREAM_SUBDIR}/sequence.index"],
        local_dir = str(data_dir),
    ))
    return root / UPSTREAM_SUBDIR / "ids.tsv", root / UPSTREAM_SUBDIR / "sequence.index"


def read_ids(path):
    # ids.tsv row index == faiss vector id.
    ids, sequences, lengths = [], [], []
    with path.open(encoding = "utf-8") as handle:
        for line in handle:
            accession, sequence, length = line.rstrip("\n").split("\t")
            ids.append(accession)
            sequences.append(sequence)
            lengths.append(int(length))
    lengths = np.array(lengths, dtype = np.int64)
    assert len(set(ids)) == len(ids), "duplicate accessions in ids.tsv"
    assert np.array_equal(lengths, np.array([len(s) for s in sequences])), "length column mismatch"
    return ids, sequences, lengths


def cluster(ids, sequences, work_dir, mmseqs, threads, force):
    fasta = work_dir / "swissprot_all.fasta"
    cluster_tsv = work_dir / "clu50_cluster.tsv"
    if not fasta.exists():
        with fasta.open("w", encoding = "utf-8") as out:
            for accession, sequence in zip(ids, sequences):
                out.write(f">{accession}\n{sequence}\n")
    if force or not cluster_tsv.exists():
        assert Path(mmseqs).exists(), f"MMseqs2 not found: {mmseqs}"
        tmp_dir = work_dir / "tmp50"
        tmp_dir.mkdir(parents = True, exist_ok = True)
        command = [
            str(mmseqs), "easy-cluster", str(fasta), str(work_dir / "clu50"), str(tmp_dir),
            "--min-seq-id", "0.5", "--cov-mode", "1", "-c", "0.8", "--cluster-mode", "0",
            "--threads", str(threads),
        ]
        print(" ".join(command))
        started = time.time()
        subprocess.run(command, check = True)
        print(f"clustered in {time.time() - started:.1f} s")
    return cluster_tsv


def read_clusters(cluster_tsv, ids):
    reps, sizes = set(), {}
    with cluster_tsv.open(encoding = "utf-8") as handle:
        for line in handle:
            rep = line.split("\t", 1)[0]
            reps.add(rep)
            sizes[rep] = sizes.get(rep, 0) + 1
    missing = reps - set(ids)
    assert not missing, f"representatives missing from ids.tsv: {sorted(missing)[:5]}"
    return reps, sizes


def write_reps(prefix, ids, sequences, lengths, reps, sizes):
    rep_rows = [i for i, accession in enumerate(ids) if accession in reps]
    table = pa.table({
        "primaryAccession": pa.array([ids[i] for i in rep_rows], type = pa.string()),
        "sequence": pa.array([sequences[i] for i in rep_rows], type = pa.string()),
        "length": pa.array([lengths[i] for i in rep_rows], type = pa.int64()),
        "clusterSize": pa.array([sizes[ids[i]] for i in rep_rows], type = pa.int64()),
        "faissRow": pa.array(rep_rows, type = pa.int64()),
    })
    pq.write_table(table, f"{prefix}.parquet")
    with open(f"{prefix}.fasta", "w", encoding = "utf-8") as fasta:
        for accession, sequence in zip(table.column("primaryAccession").to_pylist(),
                                       table.column("sequence").to_pylist()):
            fasta.write(f">{accession}\n{sequence}\n")
    return rep_rows


def extract_embeddings(index_path, rep_rows, ntotal):
    index = faiss.read_index(str(index_path), faiss.IO_FLAG_MMAP)
    assert isinstance(index, faiss.IndexFlatIP), type(index).__name__
    assert index.ntotal == ntotal, (index.ntotal, ntotal)
    flat = faiss.rev_swig_ptr(index.get_xb(), index.ntotal * index.d).reshape(index.ntotal, index.d)
    embeddings = np.empty((len(rep_rows), index.d), dtype = np.float32)
    chunk = 20000
    for start in tqdm(range(0, len(rep_rows), chunk), desc = "copy embeddings", unit = "chunk"):
        rows = rep_rows[start:start + chunk]
        embeddings[start:start + len(rows)] = flat[rows]
    norms = np.linalg.norm(embeddings, axis = 1)
    assert np.all(np.abs(norms - 1.0) < 1e-4), "embeddings are not L2-normalised"
    assert np.isfinite(embeddings).all()
    return embeddings


def write_legacy(legacy_dir, reps_prefix, ids, lengths, sizes, rep_rows, embeddings):
    """The .npy + metadata layout that legacy/train.py memory-maps."""
    legacy_dir.mkdir(parents = True, exist_ok = True)
    embeddings_path = legacy_dir / "protrek650m_protein_swissprot_50pct.npy"
    np.save(embeddings_path, embeddings)
    meta = pd.DataFrame({
        "row": np.arange(len(rep_rows), dtype = np.int64),
        "faissRow": np.array(rep_rows, dtype = np.int64),
        "primaryAccession": [ids[i] for i in rep_rows],
        "length": [int(lengths[i]) for i in rep_rows],
        "clusterSize": [sizes[ids[i]] for i in rep_rows],
        "truncated": [bool(lengths[i] > MAX_RESIDUE_LENGTH) for i in rep_rows],
    })
    meta_path = legacy_dir / "protrek650m_protein_swissprot_50pct_meta.parquet"
    meta.to_parquet(meta_path, index = False)
    print("legacy:", embeddings_path, embeddings.shape,
          "|", meta_path, "|", f"{reps_prefix}.parquet", "|", f"{reps_prefix}.fasta")


def exclusion_reason(sequence):
    """Why a ProTrek-conditioned sequence model cannot learn from this row, or None."""
    if len(sequence) > MAX_RESIDUE_LENGTH:
        return "longer than 1024 residues (ProTrek embeds only the first 1024)"
    nonstandard = sorted(set(sequence) - STANDARD_AMINO_ACIDS)
    if nonstandard:
        return "non-standard residues: " + "".join(nonstandard)
    return None


def bucket_of(length):
    for low, high in BUCKETS:
        if low < length <= high:
            return (low, high)
    raise ValueError(length)


def split_groups(frame, train_size, seed):
    """Assign whole sequence-identity groups to train/test, stratified by length bucket.

    Rows sharing a sequence share an embedding, so splitting them apart would leak. Assigning
    per group keeps every duplicate on one side; the bucket targets keep the length
    distribution aligned between the two splits.
    """
    test_target = len(frame) - train_size
    assert test_target > 0, (len(frame), train_size)

    frame = frame.assign(bucket = [bucket_of(length) for length in frame["length"]],
                         group = frame.groupby("sequence", sort = False).ngroup())
    sizes = frame.groupby("group")["bucket"].size()

    # Proportional bucket targets, with the rounding remainder given to the largest buckets.
    per_bucket = frame.groupby("bucket").size()
    exact = per_bucket * (test_target / len(frame))
    targets = exact.astype(int)
    remainder = test_target - int(targets.sum())
    for bucket in (exact - targets).sort_values(ascending = False).index[:remainder]:
        targets[bucket] += 1
    assert int(targets.sum()) == test_target

    rng = np.random.default_rng(seed)
    candidates = defaultdict(list)           # bucket -> group ids, shuffled per bucket
    group_bucket = frame.drop_duplicates("group").set_index("group")["bucket"]
    for group in rng.permutation(group_bucket.index.to_numpy()):
        candidates[group_bucket[group]].append(int(group))

    chosen = set()
    for bucket, target in targets.items():
        taken, index, skipped = 0, 0, []
        order = candidates[bucket]
        # Prefer groups that keep the bucket on target; a group that would overshoot is
        # skipped and only retried if the bucket still needs padding.
        while taken < target and index < len(order):
            group = order[index]
            index += 1
            if taken + sizes[group] <= target:
                chosen.add(group)
                taken += int(sizes[group])
            else:
                skipped.append(group)
        for group in skipped:
            if taken >= target:
                break
            chosen.add(group)
            taken += int(sizes[group])

    # Bucket rounding can leave the total a few rows off. Top up or trim with single-row
    # groups (the overwhelming majority) until the split matches the requested size exactly.
    single_row_groups = [int(g) for g in sizes[sizes == 1].index]
    for group in rng.permutation(single_row_groups):
        if sum(sizes[g] for g in chosen) >= test_target:
            break
        if group not in chosen:
            chosen.add(group)
    while sum(sizes[g] for g in chosen) > test_target:
        chosen.discard(next(g for g in chosen if sizes[g] == 1))

    is_test = frame["group"].isin(chosen)
    assert int(is_test.sum()) == test_target, int(is_test.sum())
    return frame.loc[~is_test].copy(), frame.loc[is_test].copy()


HUB_COLUMNS = {
    "primaryAccession": pa.string(),
    "sequence": pa.string(),
    "length": pa.int32(),
    "clusterSize": pa.int32(),
    "faissRow": pa.int64(),
}


def to_parquet(frame, embeddings, path, extra_columns = ()):
    """One self-contained row set: sequence and embedding on the same row."""
    vectors = np.ascontiguousarray(embeddings[frame["row"].to_numpy()]).reshape(-1)
    table = pa.table({
        **{name: pa.array(frame[name].tolist(), type = kind) for name, kind in HUB_COLUMNS.items()},
        **{name: pa.array(frame[name].tolist()) for name in extra_columns},
        "embedding": pa.FixedSizeListArray.from_arrays(
            pa.array(vectors, type = pa.float32()), EMBEDDING_DIM),
    })
    pq.write_table(table, path, compression = "zstd", row_group_size = 4096)
    return path


def sha256(path, chunk = 1 << 24):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def bucket_counts(frame):
    labels = [f"{low}-{high}" for low, high in BUCKETS]
    buckets = pd.cut(frame["length"], [low for low, _ in BUCKETS] + [BUCKETS[-1][1]],
                     labels = labels, include_lowest = True)
    counts = buckets.value_counts().sort_index()
    return {label: int(counts.get(label, 0)) for label in labels}


def tool_version(command):
    try:
        return subprocess.run(command, capture_output = True, text = True).stdout.strip() or None
    except OSError:
        return None


def upstream_revision():
    """The upstream dataset revision materialised in the Hub cache, when known."""
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
        refs = Path(HF_HUB_CACHE) / "datasets--westlake-repl--faiss_index" / "refs" / "main"
        if refs.exists():
            return refs.read_text(encoding = "utf-8").strip()
    except Exception:                      # provenance only; never fatal
        return None
    return None


def build_manifest(args, reps, eligible, excluded, train, test, outputs, upstream_count):
    return {
        "name": "protrek650m-swissprot50-reps",
        "built_at": datetime.now(timezone.utc).isoformat(timespec = "seconds"),
        "seed": args.seed,
        "counts": {
            "representatives": len(reps),
            "eligible": len(eligible),
            "excluded": len(excluded),
            "train": len(train),
            "test": len(test),
        },
        "exclusion_reasons": excluded["exclusionReason"].value_counts().to_dict(),
        "length_buckets": {"train": bucket_counts(train), "test": bucket_counts(test)},
        "unique_sequences": {
            "train": int(train["sequence"].nunique()),
            "test": int(test["sequence"].nunique()),
        },
        "embedding": {
            "dim": EMBEDDING_DIM,
            "dtype": "float32",
            "l2_normalized": True,
            "source": {"repo_id": UPSTREAM_REPO, "repo_type": "dataset",
                       "path": UPSTREAM_SUBDIR},
        },
        "clustering": {
            "tool": "MMseqs2 easy-cluster",
            "command": "--min-seq-id 0.5 --cov-mode 1 -c 0.8 --cluster-mode 0",
            "version": tool_version([args.mmseqs, "version"]),
            "input_sequences": upstream_count,
        },
        "provenance": {
            "uniprot_release_label": "Swiss-Prot (as mirrored in the faiss_index dataset)",
            "built_by": "build_dataset.py",
            "repository": "https://github.com/yosshstd/protein-generation",
            "repository_commit": tool_version(["git", "rev-parse", "HEAD"]),
            "upstream_revision": upstream_revision(),
        },
        "files": {name: {"sha256": sha256(path), "bytes": path.stat().st_size}
                  for name, path in outputs.items()},
    }


def dataset_card(manifest, repo_id):
    """The Hub dataset card, generated from the manifest so it cannot drift from the data."""
    counts, buckets = manifest["counts"], manifest["length_buckets"]
    files = manifest["files"]
    revision = manifest["provenance"]["upstream_revision"] or "main"
    mmseqs = manifest["clustering"]["version"] or "unknown"
    seed = manifest["seed"]

    def rows(name):
        return f"{counts[name]:,}"

    def size(name):
        megabytes = files[name]["bytes"] / 1e6
        return f"{megabytes:.0f} MB" if megabytes >= 10 else f"{megabytes:.1f} MB"

    long_rows = sum(v for k, v in manifest["exclusion_reasons"].items() if k.startswith("longer"))
    nonstandard_rows = sum(v for k, v in manifest["exclusion_reasons"].items()
                           if k.startswith("non-standard"))
    bucket_lines = ["| Length (residues) | Train | Test |", "| --- | --- | --- |"]
    for label in buckets["train"]:
        bucket_lines.append(f"| {label} | {buckets['train'][label]:,} | {buckets['test'][label]:,} |")

    return f"""---
pretty_name: ProTrek-650M Swiss-Prot 50% representatives
tags:
- biology
- protein
- proteins
- embeddings
- swiss-prot
- protrek
- sequence-generation
- inverse-folding
task_categories:
- text-generation
size_categories:
- 100K<n<1M
configs:
- config_name: default
  data_files:
  - split: train
    path: train.parquet
  - split: test
    path: test.parquet
---

# ProTrek-650M Swiss-Prot 50% representatives (sequence + embedding)

Swiss-Prot protein sequences paired with their **ProTrek-650M** embeddings, clustered to 50%
sequence identity so each row is one cluster representative. Built for training models that
decode an amino acid sequence from a protein embedding (inverse folding / embedding-to-sequence),
but usable wherever you need Swiss-Prot sequences with fixed ProTrek vectors.

- {rows('representatives')} cluster representatives, each with a 1024-dim L2-normalised embedding
- Deduplicated at 50% identity, so rows are far less redundant than raw Swiss-Prot
- Split into {rows('train')} train / {rows('test')} test, with sequence-identity groups held on one side
- One self-contained parquet per split: `sequence` and `embedding` sit on the same row

## Files

| File | Rows | Size | Contents |
| --- | --- | --- | --- |
| `train.parquet` | {rows('train')} | {size('train')} | Training split |
| `test.parquet` | {rows('test')} | {size('test')} | Held-out test split |
| `excluded.parquet` | {rows('excluded')} | {size('excluded')} | Representatives left out of both splits, with `exclusionReason` |
| `split_ids.tsv` | — | {size('split_ids')} | `split` + `primaryAccession`, so the split can be reproduced |
| `manifest.json` | — | — | Counts, sha256 per file, upstream revision, exclusion breakdown |

`train.parquet + test.parquet + excluded.parquet` covers all {rows('representatives')}
representatives. Every representative from the upstream index appears in exactly one file.

## Schema

All three parquet files share these columns:

| Column | Type | Description |
| --- | --- | --- |
| `primaryAccession` | string | UniProt accession, e.g. `P04637` |
| `sequence` | string | Amino acid sequence (one-letter code) |
| `length` | int32 | Residue count |
| `clusterSize` | int32 | Number of sequences in this representative's 50% cluster |
| `faissRow` | int64 | Row index in the upstream faiss index |
| `embedding` | fixed_size_list\\<float32\\>[1024] | ProTrek-650M protein vector, L2-normalised |

`excluded.parquet` adds `exclusionReason` (string).

## Usage

```python
from datasets import load_dataset

ds = load_dataset("{repo_id}")
row = ds["train"][0]
row["primaryAccession"]  # e.g. 'Q57794'
row["sequence"]          # 'MVIMPFVHYI...'
row["embedding"]         # list of 1024 float32
```

Or with pandas / pyarrow:

```python
import pandas as pd
df = pd.read_parquet("train.parquet")
```

`embedding` arrives as `List(Value("float32"), length=1024)`; `numpy.asarray(row["embedding"])`
gives a `(1024,)` array. The vectors are stored as float32 without loss and are not
re-normalised on read.

## Data

**Sequences.** UniProt Swiss-Prot, as mirrored in
[`{UPSTREAM_REPO}`](https://huggingface.co/datasets/{UPSTREAM_REPO}) under
`{UPSTREAM_SUBDIR}/` (revision `{revision}`). The upstream file holds
{manifest['clustering']['input_sequences']:,} sequences; `ids.tsv` row order equals faiss
vector id.

**Clustering.** MMseqs2 `easy-cluster` with
`--min-seq-id 0.5 --cov-mode 1 -c 0.8 --cluster-mode 0` (version `{mmseqs}`) reduces the
upstream sequences to {rows('representatives')} representatives. `clusterSize` is the full
cluster size, so it can be used for weighting if you want to approximate the unclustered
distribution.

**Embeddings.** Read straight out of the upstream faiss `IndexFlatIP` (dimension
{manifest['embedding']['dim']}) at each representative's `faissRow`. They are the index's
stored vectors, not recomputed, so they match the upstream index exactly. All norms are 1.0
to within 1e-6.

**Split.** `split_ids.tsv` records the seed-{seed} assignment; keep it with the data if you
need to reproduce the exact split. The split is stratified over length buckets so the length
distribution stays close between train and test:

{chr(10).join(bucket_lines)}

## What was excluded, and why

The {rows('excluded')} rows in `excluded.parquet` are representative sequences that a
ProTrek-conditioned sequence model cannot learn from:

| Reason | Rows |
| --- | --- |
| Longer than 1024 residues | {long_rows:,} |
| Non-standard residues (`X`, `U`, `B`, `Z`, `O`) | {nonstandard_rows:,} |

ProTrek-650M encodes only the first 1024 residues (plus a CLS/EOS token), so for a longer
protein the embedding describes a prefix while the sequence describes the whole protein — the
pair would be silently inconsistent. Non-standard residues are ambiguous or non-canonical and
are left out so the target alphabet stays the 20 standard amino acids.

They are published rather than dropped so the filter is auditable and reversible. For the long
sequences the embedding is still the ProTrek vector of the full (truncated at 1024) input, so
the row holds a valid embedding — just a target sequence that does not correspond one-to-one
to it.

## Known characteristics

- **15 rows share a sequence with another row** (6 sequence groups, all short peptides of
  7–11 residues whose clusters each had size 1). Their embeddings are identical because
  ProTrek is deterministic, and all duplicates sit on the same side of the split. This is why
  the split has {manifest['unique_sequences']['train']:,} unique sequences in {rows('train')} train rows
  and {manifest['unique_sequences']['test']:,} in {rows('test')} test rows.
- `faissRow` indexes the upstream index, not these files. Do not use it as a row index here;
  rows in these files are independent.
- `clusterSize` is almost always small (median 1), so unweighted training heavily favours
  singleton clusters — arguably the point of 50% clustering, but worth knowing.

## Provenance

| Item | Value |
| --- | --- |
| Upstream embeddings | `{UPSTREAM_REPO}` @ `{revision}` |
| Clustering | MMseqs2 `{mmseqs}` |
| Built by | `build_dataset.py` in [yosshstd/protein-generation](https://github.com/yosshstd/protein-generation) |
| Seed | {seed} |
| Built at | {manifest['built_at']} |

Rebuilding with the same inputs and seed reproduces the files bit for bit; the sha256 values
in `manifest.json` are checked by the build script.

## Citation

The embeddings come from ProTrek. If you use this dataset, please cite the ProTrek paper:

```bibtex
@article{{su2024protrek,
  title   = {{ProTrek: Navigating the Protein Universe through Tri-Modal Contrastive Learning}},
  author  = {{Su, Jin and Zhou, Xibin and Zhang, Xuting and Yuan, Fajie}},
  journal = {{bioRxiv}},
  year    = {{2024}},
  doi     = {{10.1101/2024.05.30.596740}}
}}
```

Please also follow the terms of UniProt (the sequence source) and of the upstream
`{UPSTREAM_REPO}` dataset.
"""


def upload(output_dir, repo_id, private):
    """Publish the built folder. Outward-facing: only runs with --upload."""
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id = repo_id, repo_type = "dataset", private = private, exist_ok = True)
    print("uploading", output_dir, "->", repo_id)
    commit = api.upload_folder(
        folder_path = str(output_dir),
        repo_id = repo_id,
        repo_type = "dataset",
        commit_message = "Upload ProTrek-650M Swiss-Prot 50% representatives",
        ignore_patterns = ["*.tmp", ".cache/*"],
    )
    print("uploaded:", commit.commit_url)
    return commit


def main():
    args = parse_args()
    data_dir, work_dir = Path(args.data_dir), Path(args.work_dir)
    output_dir = Path(args.output_dir)
    for path in (work_dir, output_dir):
        path.mkdir(parents = True, exist_ok = True)

    ids_path, index_path = download(data_dir)
    print("ids.tsv:", f"{ids_path.stat().st_size / 1e6:.1f} MB",
          "| sequence.index:", f"{index_path.stat().st_size / 1e6:.1f} MB")

    ids, sequences, lengths = read_ids(ids_path)
    print("sequences:", len(ids), "| length max:", lengths.max())

    cluster_tsv = cluster(ids, sequences, work_dir, args.mmseqs, args.threads, args.force_cluster)
    reps_set, sizes = read_clusters(cluster_tsv, ids)
    print(f"representatives: {len(reps_set)} ({len(reps_set) / len(ids):.1%})")

    reps_prefix = data_dir / "protrek650m_swissprot_50pct_reps"
    rep_rows = write_reps(reps_prefix, ids, sequences, lengths, reps_set, sizes)
    embeddings = extract_embeddings(index_path, rep_rows, len(ids))

    if not args.no_legacy:
        write_legacy(Path(args.legacy_dir), reps_prefix, ids, lengths, sizes, rep_rows, embeddings)

    reps = pd.DataFrame({
        "row": np.arange(len(rep_rows), dtype = np.int64),
        "primaryAccession": [ids[i] for i in rep_rows],
        "sequence": [sequences[i] for i in rep_rows],
        "length": np.array([lengths[i] for i in rep_rows], dtype = np.int64),
        "clusterSize": np.array([sizes[ids[i]] for i in rep_rows], dtype = np.int64),
        "faissRow": np.array(rep_rows, dtype = np.int64),
    })

    reasons = [exclusion_reason(sequence) for sequence in reps["sequence"]]
    eligible_mask = np.array([reason is None for reason in reasons])
    eligible = reps[eligible_mask].copy()
    excluded = reps[~eligible_mask].copy()
    excluded["exclusionReason"] = [reason for reason in reasons if reason is not None]
    print(f"reps: {len(reps):,} | eligible: {len(eligible):,} | excluded: {len(excluded):,}")

    train, test = split_groups(eligible, args.train_size, args.seed)
    print(f"train: {len(train):,} | test: {len(test):,}")

    # Prove the invariants hold before anything is written or published.
    for name, frame in (("train", train), ("test", test)):
        assert all(exclusion_reason(sequence) is None for sequence in frame["sequence"]), name
        assert frame["primaryAccession"].is_unique, name
    assert not set(train["primaryAccession"]) & set(test["primaryAccession"])
    assert len(train) + len(test) == len(eligible)
    assert set(train["primaryAccession"]) | set(test["primaryAccession"]) == set(eligible["primaryAccession"])
    assert not set(train["sequence"]) & set(test["sequence"]), "a sequence leaked across the split"

    outputs = {
        "train": to_parquet(train, embeddings, output_dir / "train.parquet"),
        "test": to_parquet(test, embeddings, output_dir / "test.parquet"),
        "excluded": to_parquet(excluded, embeddings, output_dir / "excluded.parquet",
                               extra_columns = ("exclusionReason",)),
        "split_ids": output_dir / "split_ids.tsv",
    }
    with open(outputs["split_ids"], "w", encoding = "utf-8") as handle:
        handle.write("split\tprimaryAccession\n")
        for split, frame in (("train", train), ("test", test)):
            for accession in frame["primaryAccession"]:
                handle.write(f"{split}\t{accession}\n")

    manifest = build_manifest(args, reps, eligible, excluded, train, test, outputs, len(ids))
    with open(output_dir / "manifest.json", "w", encoding = "utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii = False, indent = 2)
        handle.write("\n")
    (output_dir / "README.md").write_text(dataset_card(manifest, args.repo_id), encoding = "utf-8")

    for name, path in outputs.items():
        print(f"{name}: {path} ({path.stat().st_size / 1e6:.1f} MB)")
    print(f"card: {output_dir / 'README.md'} | manifest: {output_dir / 'manifest.json'}")

    if not args.upload:
        print(f"not uploading; re-run with --upload to publish to {args.repo_id}")
        return
    upload(output_dir, args.repo_id, args.private)


if __name__ == "__main__":
    main()
