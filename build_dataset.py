"""Build the ProTrek_650M Swiss-Prot 50% representative dataset.

Download westlake-repl/faiss_index, cluster at 50% identity with MMseqs2, keep representatives,
export their embeddings from the faiss index.

Usage: python build_dataset.py [--threads 16]
"""
import argparse
import subprocess
import time
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import snapshot_download
from tqdm.auto import tqdm

MAX_RESIDUE_LENGTH = 1024  # ProTrek embeds 1024 residues + <cls> + <eos>


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default = "data")
    parser.add_argument("--work-dir", default = "work")
    parser.add_argument("--embeddings-dir", default = "outputs/embeddings")
    parser.add_argument("--mmseqs", default = "tools/mmseqs/bin/mmseqs")
    parser.add_argument("--threads", type = int, default = 16)
    parser.add_argument("--force-cluster", action = "store_true")
    return parser.parse_args()


def download(data_dir):
    subdir = "SwissProt/ProTrek_650M_UniRef50/sequence"
    local = Path(data_dir) / subdir
    ids_path, index_path = local / "ids.tsv", local / "sequence.index"
    if ids_path.exists() and index_path.exists():
        return ids_path, index_path
    root = Path(snapshot_download(
        repo_id = "westlake-repl/faiss_index",
        repo_type = "dataset",
        allow_patterns = [f"{subdir}/ids.tsv", f"{subdir}/sequence.index"],
        local_dir = str(data_dir),
    ))
    return root / subdir / "ids.tsv", root / subdir / "sequence.index"


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
    assert not np.isnan(embeddings).any()
    return embeddings


def main():
    args = parse_args()
    data_dir, work_dir, embeddings_dir = Path(args.data_dir), Path(args.work_dir), Path(args.embeddings_dir)
    for path in (work_dir, embeddings_dir):
        path.mkdir(parents = True, exist_ok = True)

    ids_path, index_path = download(data_dir)
    print("ids.tsv:", f"{ids_path.stat().st_size / 1e6:.1f} MB",
          "| sequence.index:", f"{index_path.stat().st_size / 1e6:.1f} MB")

    ids, sequences, lengths = read_ids(ids_path)
    print("sequences:", len(ids), "| length max:", lengths.max())

    cluster_tsv = cluster(ids, sequences, work_dir, args.mmseqs, args.threads, args.force_cluster)
    reps, sizes = read_clusters(cluster_tsv, ids)
    print(f"representatives: {len(reps)} ({len(reps) / len(ids):.1%})")

    reps_prefix = data_dir / "protrek650m_swissprot_50pct_reps"
    rep_rows = write_reps(reps_prefix, ids, sequences, lengths, reps, sizes)

    embeddings = extract_embeddings(index_path, rep_rows, len(ids))
    embeddings_path = embeddings_dir / "protrek650m_protein_swissprot_50pct.npy"
    np.save(embeddings_path, embeddings)

    meta = pd.DataFrame({
        "row": np.arange(len(rep_rows), dtype = np.int64),
        "faissRow": np.array(rep_rows, dtype = np.int64),
        "primaryAccession": [ids[i] for i in rep_rows],
        "length": [int(lengths[i]) for i in rep_rows],
        "clusterSize": [sizes[ids[i]] for i in rep_rows],
        "truncated": [bool(lengths[i] > MAX_RESIDUE_LENGTH) for i in rep_rows],
    })
    meta_path = embeddings_dir / "protrek650m_protein_swissprot_50pct_meta.parquet"
    meta.to_parquet(meta_path, index = False)

    print("wrote", embeddings_path, embeddings.shape, f"{embeddings_path.stat().st_size / 1e6:.1f} MB")
    print("wrote", meta_path, "|", f"{reps_prefix}.parquet", "|", f"{reps_prefix}.fasta")


if __name__ == "__main__":
    main()
