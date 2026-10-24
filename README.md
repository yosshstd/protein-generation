# protein-generation

ProTrek-650M のタンパク質埋め込みを入力に、ProLLaMA_Stage_1 でアミノ酸配列を生成するモデルを
LoRA + projector で学習するリポジトリです。

配列を ProTrek で 1024 次元ベクトルにし、`ProteinProjector` で K 個（既定 10）の入力埋め込み
スロットに射影して、ProLLaMA 本来のプロンプト `Seq=<` の直前に差し込みます。指示部分
（`▁Seq = <`）はベースモデルが事前学習した形のまま残すので、モデルは「指示 + ベクトル」から
続きの配列を生成します。

```
アミノ酸配列 ──ProTrek_650M──▶ 1024 次元ベクトル（L2 正規化済み）
                                     │
                        平均ベクトルを引く → ProteinProjector（K ヘッド）
                                     │
                                     ▼
<s> <unk>×K ▁Seq = < [K 個のベクトル] ──ProLLaMA_Stage_1 + LoRA──▶ 配列 >
```

## パイプライン

3 本で完結します。

| # | コマンド | やること |
| --- | --- | --- |
| 1 | `uv run python build_dataset.py` | データ構築（50% クラスタリング → フィルタ → split → parquet、`--upload` で Hub 公開） |
| 2 | `uv run python train_prollama_from_emb.py` | ベクトル → K スロットの soft prompt で LoRA + projector を学習 |
| 3 | `uv run python inference_prollama_from_emb.py` | 学習済みアダプタで生成し、長さ・括弧閉じ・不正残基を集計 |

学習と推論は既定で Hub のデータセット（`yosshstd/protrek650m-swissprot50-reps`）を読みます。
Hub に上げずに試す場合は、`build_dataset.py` の出力ディレクトリを `--dataset` に渡してください。

## ファイル構成

```
build_dataset.py                  # 1. データ構築（HF 取得 → クラスタリング → parquet → 任意で公開）
train_prollama_from_emb.py        # 2. 学習 + 共有パーツ（projector / prefix / 定数）
inference_prollama_from_emb.py    # 3. 生成・評価
pyproject.toml / uv.lock          # 依存（uv で管理）
tools/get_mmseqs.sh               # MMseqs2（静的ビルド）の取得
legacy/                           # 旧スクリプト置き場（.gitignore 済み・ローカルのみ）
```

| ファイル | 役割 |
| --- | --- |
| `build_dataset.py` | Swiss-Prot 配列 + ProTrek ベクトルを 50% 同一性でクラスタリングし、学習用 parquet を作る |
| `train_prollama_from_emb.py` | ProLLaMA_Stage_1 に LoRA を付け、projector と一緒に学習する。定数・prefix / answer の組み立て・`ProteinProjector`・`ProjectorTrainerMixin` もここにあり、3 がこれを import する |
| `inference_prollama_from_emb.py` | 保存したアダプタ + projector でサンプルを生成し、スコアを出す |
| `pyproject.toml` / `uv.lock` | 依存パッケージとバージョン固定 |
| `tools/get_mmseqs.sh` | `build_dataset.py` が使う MMseqs2 を取得 |

`data/` `work/` `outputs/` は生成物で、すべて `.gitignore` 済みです。

## セットアップ

Python 3.13 を前提としています。依存は `pyproject.toml` にまとまっているので 1 コマンドです。

```bash
uv sync
```

`uv.lock` に固定されたバージョンが `.venv` に入ります。unsloth は PyPI ではなく git から取得します
（`[tool.uv.sources]` で指定済み）。git の `unsloth` 本体には CLI しか入っておらず、学習用の
依存（`unsloth_zoo` / `transformers` / `trl` など）は `base` extra にあるので `unsloth[base]` を
指定しています。

torch / torchvision / triton / xformers は cu129 系を `pyproject.toml` の `[[tool.uv.index]]` で
指定しています。WSL のドライバが CUDA 12.6 のため、cu130 の wheel だと
`torch.cuda.is_available()` が False になります。バージョンを上げるときはこの index 指定を
消さないようにしてください。

起動時に次の警告が出ることがありますが、学習には影響しません。cu129 index の `xformers` が
torch 2.13.0 向けにビルドされていないためで、unsloth は PyTorch の attention に自動で
切り替えます。不要なら `uv remove xformers` で消せます。

```
xformers is installed but cannot load with torch 2.13.0+cu129 ...
```

MMseqs2（約 32 MB）を取得します。

```bash
bash tools/get_mmseqs.sh
```

確認:

```bash
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# -> 2.13.0+cu129 True
uv run python -c "from unsloth import FastLanguageModel; from trl import SFTTrainer"
tools/mmseqs/bin/mmseqs version
```

パッケージを追加するときは `uv add <package>` です。

## 1. データセットを作る

```bash
uv run python build_dataset.py
```

やっていること:

1. `westlake-repl/faiss_index` の `SwissProt/ProTrek_650M_UniRef50/sequence/` を取得
   （`ids.tsv` 約 203 MB + `sequence.index` 約 2.2 GB。`--data-dir` にキャッシュ）
2. MMseqs2 `easy-cluster` で 50% 同一性・カバレッジ 0.8 にクラスタリング
   （569,792 配列 → 代表 118,576 配列）
3. ProTrek が表現できない行を `excluded.parquet` に退避
   （1024 残基超 6,723 件 + 非標準残基 851 件 = 7,574 件）
4. 残り 111,002 件を、配列が同一の行を跨がせない形で 100,000 train / 11,002 test に分割
   （seed 42、長さバケットで層化）
5. `sequence` と `embedding` を同じ行に入れた自己完結 parquet を書き出す
6. dataset card と `manifest.json`（件数・sha256・上流リビジョン・ツール版）を生成
7. `--upload` を付けたときだけ Hub へ公開する

出力:

```
data/protrek650m_swissprot_50pct_reps.{parquet,fasta}   # クラスタ代表
outputs/hf_dataset/
  train.parquet / test.parquet / excluded.parquet
  split_ids.tsv / manifest.json / README.md             # manifest は件数・sha256・除外内訳
outputs/embeddings/                                      # --legacy-dir（legacy/train.py 用）
work/                                                    # MMseqs2 の中間ファイル
```

ダウンロードとクラスタリングは済んでいれば再利用されるので、中断しても再実行は安く済みます。
同じ入力からは同じ sha256 のファイルが出ます（`manifest.json` で確認できます）。

主なオプション:

| オプション | 既定値 | 説明 |
| --- | --- | --- |
| `--data-dir` | `data` | HF データと代表配列の置き場所 |
| `--work-dir` | `work` | MMseqs2 の中間ファイル |
| `--output-dir` | `outputs/hf_dataset` | parquet の出力先 |
| `--train-size` | `100000` | train に置く件数（残りが test） |
| `--seed` | `42` | split の乱数 |
| `--threads` | `16` | クラスタリングのスレッド数 |
| `--mmseqs` | `tools/mmseqs/bin/mmseqs` | MMseqs2 のパス |
| `--force-cluster` | なし | クラスタリングをやり直す |
| `--legacy-dir` | `outputs/embeddings` | `.npy` + meta の出力先（`legacy/train.py` 用） |
| `--no-legacy` | なし | `.npy` + meta を書かない |
| `--upload` | なし | `--repo-id` へ公開する |
| `--private` | なし | 公開リポジトリを private で作る |
| `--repo-id` | `yosshstd/protrek650m-swissprot50-reps` | 公開先 |

`--upload` は外向きの操作なので、付けたときだけ実行されます。付けない場合は
`not uploading; re-run with --upload ...` と表示して終わります。Hub に上げずに学習したいときは、
`train_prollama_from_emb.py --dataset outputs/hf_dataset` のようにローカルディレクトリを渡します。

公開されたデータセットは `datasets` からそのまま読めます。

```python
from datasets import load_dataset

dataset = load_dataset("yosshstd/protrek650m-swissprot50-reps")
dataset["train"][0]["sequence"], len(dataset["train"][0]["embedding"])
# -> ('MKTAYIAKQR...', 1024)
```

## 2. 学習する

```bash
uv run python train_prollama_from_emb.py
```

流れ:

1. `--dataset` から `sequence` と `embedding` を読み、残基数とアルファベットでフィルタ
2. トークン長が `--max-seq-length` を超えるサンプルを落とす
3. サンプルを `--max-seq-length` の行に詰める（packing。`--no-packing` で 1 サンプル 1 行）
4. ベースモデルを 4bit（`--no-4bit` で fp16/bf16）で読み、LoRA と projector を載せる
5. 学習 → `--output-dir` にアダプタ・projector・tokenizer を保存 → サンプル 1 件で生成確認

保存物:

```
outputs/prollama_stage1_emb/
  adapter_config.json / adapter_model.safetensors   # LoRA
  protein_projector.pt                              # projector + 復元用メタデータ
  tokenizer 一式
  checkpoints/checkpoint-N/                         # 500 step ごと、最新 2 個を保持
      trainer_state.json など + protein_projector.pt
```

`protein_projector.pt` には `state_dict` に加えて `num_tokens` / `hidden_size` / `prefix_ids` /
`placeholder_positions` / `embedding_mean` などが入っていて、推論側はこれだけで同じ構成の
projector を組み直せます。チェックポイントには必ず一緒に保存されるので、`--resume` しても
別 step の projector を読むことはありません（run 直下に 1 つ書くと全チェックポイントが共有して
しまうため、`checkpoint-N/` を自前で組み立てています）。

主なオプション:

| オプション | 既定値 | 説明 |
| --- | --- | --- |
| `--model` | `GreatCaptainNemo/ProLLaMA_Stage_1` | ベースモデル |
| `--dataset` | `yosshstd/protrek650m-swissprot50-reps` | Hub の repo id かローカルディレクトリ |
| `--split` | `train` | `train` / `test` |
| `--limit` | なし | 先頭 N 行だけ使う（フィルタはその後なので実際は N 件未満） |
| `--max-seq-length` | `720` | 1 行のトークン長上限 |
| `--num-tokens` | `10` | ベクトルを何スロットに射影するか |
| `--output-dir` | `outputs/prollama_stage1_emb` | 保存先 |
| `--max-steps` | なし（1 epoch） | 動作確認は `--max-steps 3` |
| `--resume` | なし | `checkpoints/` の最新から再開（`--num-tokens` を合わせる） |
| `--lora-r` / `--lora-alpha` | `128` / `256` | LoRA の rank / alpha |
| `--learning-rate` | `2e-4` | LoRA 側の学習率 |
| `--projector-lr` | `2e-4` | projector 側の学習率（別パラメータグループ） |
| `--batch-size` / `--grad-accum` | `1` / `2` | 8 GB 前提の設定 |
| `--epochs` | `1.0` | epoch 数 |
| `--gradient-checkpointing` | `unsloth` | `unsloth` / `standard` / `off` |
| `--no-4bit` | なし（4bit で読み込む） | 4bit NF4 をやめて fp16/bf16 で読み込む（VRAM を大幅に使う） |
| `--no-packing` | なし | packing をやめ、バッチ内最長まで pad する |

学習ログには `candidates` / `kept` / `samples` / `dropped (too many tokens)` / `rows`
（詰めた後の行数と samples/row）が出ます。`--limit 200 --max-steps 3` で一通り動くか確認できます。

## 3. 生成する

```bash
uv run python inference_prollama_from_emb.py
uv run python inference_prollama_from_emb.py --start 100 --num-samples 20 --temperature 0.7
```

`--adapter` のディレクトリに `protein_projector.pt` が必要です（無い場合は学習を促して終了）。
データセットの `[--start, --start + --num-samples)` 行を順に使い、各行の正解配列も一緒に表示します。

```text
[1] 120 aa | closed True | invalid 0 | unique 18 | top: A 0.12, L 0.10, ...
  MKTAYIAKQRQISFVKSHFSRQ...
  reference: MKTAYIAKQRQISFVKSHFSRQ...>

=== summary ===
samples          : 10
bracket closed   : 10 / 10
mean length      : 118.4 (min 42, max 201)
invalid residues : 0 / 1184 (0.00% of emitted characters)
```

| オプション | 既定値 | 説明 |
| --- | --- | --- |
| `--model` | `GreatCaptainNemo/ProLLaMA_Stage_1` | ベースモデル |
| `--adapter` | `outputs/prollama_stage1_emb` | 学習で保存したディレクトリ |
| `--dataset` / `--split` | Hub の repo / `train` | 参照配列とベクトルの取得元 |
| `--start` / `--num-samples` | `0` / `10` | 使う行の範囲 |
| `--max-new-tokens` | `256` | 生成長の上限 |
| `--temperature` / `--top-k` / `--top-p` | `0.5` / `40` / `0.9` | サンプリング |
| `--repetition-penalty` | `1.2` | 同じ残基の繰り返し抑制 |
| `--seed` | `0` | 行ごとに `seed + index` で固定 |
| `--greedy` | なし | サンプリングせず貪欲法 |
| `--jsonl` | なし | 結果を JSONL に書き出す |

`--jsonl` の各行は `{"index", "length", "closed", "invalid", "unique", "sequence"}` です。
`closed` は末尾の `>` が出たか、`invalid` は標準 20 アミノ酸以外の文字数です。

## データセット仕様

`train.parquet` / `test.parquet` / `excluded.parquet` は同じ列を持ちます。

| 列 | 型 | 内容 |
| --- | --- | --- |
| `primaryAccession` | string | UniProt accession |
| `sequence` | string | アミノ酸配列（1 文字表記） |
| `length` | int32 | 残基数 |
| `clusterSize` | int32 | 50% クラスタのサイズ |
| `faissRow` | int64 | 上流 faiss インデックスの行番号（この parquet の行番号ではない） |
| `embedding` | fixed_size_list\<float32\>[1024] | ProTrek-650M のベクトル（L2 正規化済み） |

`excluded.parquet` だけ `exclusionReason` 列が増えます。

**除外する行**（`build_dataset.py` の `exclusion_reason`。[1] と [2] は
`train_prollama_from_emb.py` のフィルタと同じ判定）:

| 条件 | 件数 | 理由 |
| --- | --- | --- |
| 残基数が 1024 を超える | 6,723 | ProTrek は先頭 1024 残基しか見ていないので、ベクトルと正解配列が別物になる |
| 標準 20 アミノ酸以外を含む（`X` / `U` / `B` / `Z` / `O`） | 851 | 曖昧・非 canonical 残基を生成させない |
| トークン長が `--max-seq-length` を超える | 0（720 のとき） | 学習時に落とす（データセットには残る） |

除外行は捨てずに `excluded.parquet` に残すので、フィルタは監査・巻き戻しできます。

**split**: `--train-size`（既定 100,000）を train に、残りを test に置きます。配列が同一の行は
必ず同じ側に入れ、長さバケット（0–50 / 50–100 / 100–200 / 200–400 / 400–800 / 800–1025）で
層化するので、train と test の長さ分布が揃います。`split_ids.tsv` に割り当てが残ります。

**埋め込み**: 上流 faiss の `IndexFlatIP`（次元 1024）から代表行のベクトルをそのまま取り出して
います（再計算しないので上流と一致）。ノルムは 1.0 です。クラスタリング条件は SeqStudio の
50% クラスタリングに合わせています（`--min-seq-id 0.5 --cov-mode 1 -c 0.8 --cluster-mode 0`）。

### フィルタ後のトークン長（実測）

111,002 件を `"<unk>×10 Seq=<"` + 配列 + EOS でトークナイズした結果です（prefix 14 トークンを含む）。

| 平均 | p50 | p90 | p99 | p99.9 | 最大 |
| --- | --- | --- | --- | --- | --- |
| 234.6 | 207 | 440 | 638 | 681 | 711 |

既定の `--max-seq-length 720` を超えるサンプルは 0 件です。`576` まで下げると 3,329 件
（3.0%）が学習から外れ、`704` なら 5 件です。

## 仕組み

### soft prompt の組み立て（`train_prollama_from_emb.py`）

- prefix は `<s> <unk>×K ▁Seq = <`。`<unk>` の K トークンだけを projector の出力で置き換えます
- prefix は文字列をトークナイズせず**トークン ID を直接組み立てます**。fast tokenizer は
  `<unk>` の連続の直後で `Seq` を先頭スペースなしの形（id 23718）にしてしまい、事前学習で
  使われた `▁Seq`（id 25981）にならないためです
- 正解は `"Seq=<" + 配列 + ">"` をトークナイズし、指示部分の ID を差し引いて作ります。末尾の
  `>` も学習対象です（事前学習の形式が `Seq=<配列>` なので、モデルは括弧を閉じる `>` を出します）
- labels は配列 + EOS のみで、prefix は `-100`（プロンプト部分は損失に含めない）

### packing と collator（`train_prollama_from_emb.py`）

- サンプルを長い順に first-fit で詰め、合計が `--max-seq-length` に収まる組を作ります。
  データセットの 1 行 = 1 組なので、shuffle も step 数も「組」単位です
- collator は **サンプルごとに `position_ids` を 0 から振り直します**。これがサンプル境界の
  目印で、`attention_mask` を渡さないことで transformers がブロック対角の因果マスクを組みます
  （2D の `attention_mask` を渡すとこの判定が切れて、詰めたサンプル同士が互いを見てしまいます）
- スロットは各サンプルの BOS 直後（offset 1）に置きます。組の中のどこに始まっても同じです
- pad 部分は `labels = -100` で、因果マスクにより損失に影響しません

### projector

```
LayerNorm(1024) → Linear(1024→2048) → GELU → Linear(2048→2048) → GELU
   → [Linear(2048→4096)] × K（スロットごとに別ヘッド）→ LayerNorm → × 1/√4096
```

- 入力は**平均ベクトルを引いてから**通します。L2 正規化済みでも平均ベクトル（ノルム約 0.48）の
  方向が支配的で、そのままだと全タンパク質が似た入力になるためです
- 出力は `LayerNorm × 固定ゲイン`（`1/√4096`、ノルム約 1.0）。トークン埋め込みのノルム
  （約 1.08）に合わせるためです。`F.normalize` は微分のヤコビアンが `1/‖out‖` で減衰し、
  出力ノルムが大きくなると gradient が消えて戻らなくなるため使っていません
- スロットごとにヘッドを分けているのは、1 本のベクトルでは帯域が足りず、モデルが
  「配列らしい組成」だけを覚えてしまうためです

### optimizer と loss

- projector は PEFT の外側にあるので、optimizer に専用パラメータグループ（`--projector-lr`、
  weight decay 0.01）として追加し、勾配クリップも本体とは別に掛けます
- `Trainer` が勾配をゼロにするのは `model.zero_grad()` だけで projector には届きません。
  そのままだと勾配が step をまたいで足し込まれ、実効学習率が 1 倍、2 倍、3 倍…と増えるため、
  `on_optimizer_step` で projector の勾配を明示的にゼロにするコールバックを入れています
- `num_items_in_batch` を渡して fused CE の正規化（sum(CE)/トークン数）と gradient
  accumulation のスケールを合わせています
- 量子化は 4bit（NF4）QLoRA、optimizer は `adamw_8bit`、スケジューラは cosine（warmup は
  総 step の 3%）、bf16 が使えない環境では fp16 に自動で切り替わります

## メモリの目安

8 GB の GPU（RTX 3060 Ti）で、4bit 量子化 + LoRA + gradient checkpointing の構成での実測です。
packing は 1 行を最大 `--max-seq-length` まで詰めるので、この表がそのまま上限の目安になります。

| max_seq_length | ピーク VRAM |
| --- | --- |
| 256 | 5.3 GB |
| 384 | 5.8 GB |
| 512 | 6.2 GB |
| 576 | 6.5 GB |
| 640 | OOM |
| 720（既定） | 未計測（640 で OOM のため 8 GB では足りない見込み） |

既定の `720` は 8 GB には収まらない見込みです。VRAM が足りない場合は `--max-seq-length` を
下げてください。`--gradient-checkpointing off` は速い代わりにメモリを食い、`--no-packing` は
バッチ内最長まで pad するため packing より不利です。`--no-4bit` にすると 7B を fp16/bf16 で
載せることになり、8 GB ではまず収まりません。

## うまくいかないとき

- **`torch.cuda.is_available()` が False**
  torch が cu130 になっていませんか。`pyproject.toml` の cu129 index 指定を確認して
  `uv sync --reinstall-package torch` を実行してください。
- **`MMseqs2 not found`**
  `bash tools/get_mmseqs.sh` を実行するか、`--mmseqs` でパスを指定してください。
- **CUDA OOM**
  `--max-seq-length` を下げてください（上表参照）。
- **`no sample survived the filters; raise --max-seq-length`**
  フィルタ後に 1 件も残っていません。`--max-seq-length` を上げるか `--limit` を外してください。
- **`no projector at ...; run train_prollama_from_emb.py first`**
  `--adapter` のディレクトリに `protein_projector.pt` がありません。学習を先に回すか、
  `--adapter` を学習時の `--output-dir` に合わせてください。
- **`--resume` でスロット数の assert に落ちる**
  チェックポイントの `num_tokens` と `--num-tokens` が違います。学習時と同じ値を渡してください。
- **`--upload` が 401 / 403 で失敗する**
  書き込み権限のあるトークンで `hf auth login` してから `--upload` を付けてください。
  リポジトリ名を変えるなら `--repo-id`、非公開にしたければ `--private` を指定します。

## legacy/

`legacy/` には、この 3 本に関係しない旧スクリプトが入っています。`.gitignore` 済みで、
パイプラインには不要です。

- `train.py` — ローカル `.npy`（`build_dataset.py --legacy-dir` の出力）から学習する旧本体
- `train_stage1.py` / `inference_stage1.py` / `config.yaml` — ProTrek を使わない stage1 ベースライン
- `train_prollama.py` / `inference_prollama.py` — ProTrek なしの ProLLaMA テキスト版
- `inference.py` — 旧 `train.py` のローカル `.npy` 経路用の推論
- `train_v1_backup.py` — 旧 `train.py` のバックアップ
- `stage1_colab.ipynb` — stage1 の Colab ノートブック

projector・prefix・answer encoding などの共有パーツは `train_prollama_from_emb.py` に移したので、
メイン 3 本はこのフォルダに依存しません。legacy 側は自己完結していて、`uv run python
legacy/train_stage1.py` のように直接実行すれば import も通ります（`legacy/train.py` も同じ
フォルダにあります）。
