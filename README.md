# protein-generation

ProTrek-650M のタンパク質埋め込みを入力に、LLM（ProLLaMA_Stage_1）でアミノ酸配列を生成するモデルを LoRA で学習するためのリポジトリです。

ProTrek で配列をベクトルにし、そのベクトルを ProLLaMA の 10 トークン分の入力（soft prompt）に変換して差し込んで、続きの配列を生成させます。

```
アミノ酸配列 ──ProTrek_650M──▶ 1024次元ベクトル
                                   │
                    (平均ベクトルを引き、Linear で 10×4096 次元へ射影)
                                   │
"<s> <unk>×10 ▁Seq = <" [10個のベクトル] ──ProLLaMA_Stage_1 + LoRA──▶ アミノ酸配列
```

## ファイル構成

| ファイル | 役割 |
| --- | --- |
| `build_dataset.py` | Hugging Face からデータ取得 → 50% クラスタリング → 代表配列の埋め込みを書き出し |
| `train.py` | ProLLaMA_Stage_1 を LoRA 学習 → 保存 → 生成チェック（ProTrek ベクトルを入力） |
| `train_stage1.py` | ProTrek ベクトルを使わない stage1 ベースライン（指示のみ） |
| `config.yaml` | `train_stage1.py` の設定 |
| `pyproject.toml` | 依存パッケージ（uv で管理） |
| `uv.lock` | 解決済みのバージョン固定 |
| `tools/get_mmseqs.sh` | MMseqs2（静的ビルド）の取得スクリプト |

生成されるもの（すべて `.gitignore` 済み）:

```
data/
  SwissProt/ProTrek_650M_UniRef50/sequence/{ids.tsv, sequence.index}   HF から取得
  protrek650m_swissprot_50pct_reps.parquet / .fasta                    クラスタ代表
work/          MMseqs2 の作業ディレクトリ
outputs/
  embeddings/*.npy, *_meta.parquet                                    学習入力の埋め込み
  prollama_stage1_protrek650m/                                        学習済み LoRA + projector
```

## セットアップ

Python 3.13 を前提としています。依存はすべて `pyproject.toml` で管理しているので、環境構築は 1 コマンドです。

```bash
uv sync
```

`uv.lock` に固定されたバージョンが `.venv` に入ります。unsloth は PyPI ではなく git から取得します
（`[tool.uv.sources]` で指定済み）。git の `unsloth` 本体には CLI しか入っておらず、学習用の
依存（`unsloth_zoo` / `transformers` / `trl` など）は `base` extra にあるので
`unsloth[base]` を指定しています。

torch / torchvision / triton / xformers は cu129 系を `pyproject.toml` の `[[tool.uv.index]]`
で指定しています。WSL のドライバが CUDA 12.6 のため、cu130 の wheel だと
`torch.cuda.is_available()` が False になります。バージョンを上げるときはこの index 指定を
消さないようにしてください。

起動時に次の警告が出ることがありますが、学習には影響しません。cu129 index の `xformers` が
torch 2.13.0 向けにビルドされていないためで、unsloth は PyTorch の attention に自動で
切り替えます。不要なら `uv remove xformers` で消せます。

```
xformers is installed but cannot load with torch 2.13.0+cu129 ...
```

パッケージを追加するとき:

```bash
uv add <package>
```

MMseqs2 を取得します（約 32 MB）:

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

## 使い方

### 1. データセットを作る

```bash
uv run python build_dataset.py
```

やっていること:

1. `westlake-repl/faiss_index` の `SwissProt/ProTrek_650M_UniRef50/sequence/` を取得（`ids.tsv` 203 MB + `sequence.index` 2.2 GB）
2. MMseqs2 `easy-cluster` で 50% 同一性・カバレッジ 0.8 でクラスタリング（569,792 配列 → 代表 118,576 配列）
3. 代表配列だけを `parquet` と `fasta` に書き出し
4. faiss インデックスから代表配列の埋め込みを取り出して `outputs/embeddings/` に保存

主なオプション:

| オプション | 既定値 | 説明 |
| --- | --- | --- |
| `--data-dir` | `data` | HF データと代表配列の置き場所 |
| `--work-dir` | `work` | MMseqs2 の中間ファイル |
| `--embeddings-dir` | `outputs/embeddings` | 埋め込みの出力先 |
| `--mmseqs` | `tools/mmseqs/bin/mmseqs` | MMseqs2 のパス |
| `--threads` | `16` | クラスタリングのスレッド数 |
| `--force-cluster` | なし | クラスタリングをやり直す |

### 2. 学習する

```bash
uv run python train.py
```

学習後に `outputs/prollama_stage1_protrek650m/` へ LoRA アダプタ・`protein_projector.pt`・tokenizer を保存し、サンプル 1 件で生成を確認します。

| オプション | 既定値 | 説明 |
| --- | --- | --- |
| `--max-steps` | なし（1 epoch） | 動作確認だけしたいときは `--max-steps 3` |
| `--resume` | なし | 最新のチェックポイントから再開 |
| `--max-seq-length` | `720` | トークン長の上限（フィルタ後の最長は 711 トークン） |
| `--num-tokens` | `10` | ProTrek ベクトルを何個の soft prompt トークンに変換するか |

### 3. stage1 ベースラインを学習する

`train_stage1.py` は ProTrek ベクトルを一切使わない比較用のベースラインです。入力は `<s> ▁Seq = <`
だけで、配列を生成する LoRA を学習します。`train.py` との差分を「ベクトルで制御できるか」だけに
絞るのが目的で、モデル・プロンプト・loss のマスクは `train.py` と共通です。

```bash
uv run python train_stage1.py
```

設定は `config.yaml` にまとまっていて、コマンドラインから `key=value` で上書きできます。

```bash
uv run python train_stage1.py training.learning_rate=1e-4
uv run python train_stage1.py training.packing=false dataset.limit=200 training.max_steps=3
```

| オプション | 既定値 | 説明 |
| --- | --- | --- |
| `--config` | `config.yaml` | 設定ファイルのパス |
| `--max-steps` | なし（1 epoch） | 動作確認だけしたいときは `--max-steps 3` |
| `--resume` | なし | 最新のチェックポイントから再開 |
| `key=value ...` | なし | 設定の上書き（OmegaConf の dotlist）。`dataset.limit` で先頭 N 件だけ使えます |

`train.py` と違う点:

- 独自 collator をやめ、trl 標準の packing（`packing: true` / `packing_strategy: bfd`）と
  padding-free collator に任せています。サンプル境界は `position_ids` のリセットで表現され、
  transformers 5.x がそこから文書マスクを組むため、Flash Attention が無くても
  詰めたサンプル同士は混ざりません（実測: packed 系列の 2 サンプル目の logits は単体実行と完全一致）
- バッチは `per_device_train_batch_size: 1` × `gradient_accumulation_steps: 16`
- wandb は使いません（`report_to: "none"`）
- 7B を 8GB に載せるため `load_in_4bit: true`（参考実装の fp16 設定はそのままでは OOM）

## 学習前のフィルタリング

トークン化の前に、マージされた配列を次の 2 条件で絞ります（`train.py` の `filter_candidates`）。

| 条件 | 除外件数 |
| --- | --- |
| 残基数が 1024 を超える | 6,723 |
| 標準 20 アミノ酸以外を含む（`X` / `U` / `B` / `Z` / `O`） | 851 |

118,576 件 → 111,002 件（93.6%）が残ります。件数はログにも出力されます。

- **1024 残基**: ProTrek のエンコーダは先頭 1024 残基しか見ていないため、それより長い配列は
  埋め込みと正解配列が別のタンパク質を指してしまう
- **標準アミノ酸のみ**: 曖昧残基や非 canonical 残基を生成対象にしないため

この後、トークン長が `--max-seq-length` を超えるサンプルがさらに除外されます。

### フィルタ後のトークン長（実測）

111,002 件を `"<unk>×10 Seq=<"` + 配列 + EOS でトークナイズした結果です（prefix 14 トークンを含む）。

| 平均 | p50 | p90 | p99 | p99.9 | 最大 |
| --- | --- | --- | --- | --- | --- |
| 234.6 | 207 | 440 | 638 | 681 | 711 |

最長は 1,013 残基で 711 トークンでした。アミノ酸は 1〜2 残基が 1 トークンにまとまるため、
残基数よりトークン数は少なくなります。既定の `--max-seq-length 720` を超えるサンプルは
0 件で、トークン長による除外は発生しません（704 なら 5 件）。

## メモリの目安

8 GB の GPU（RTX 3060 Ti）で、4bit 量子化 + LoRA + gradient checkpointing の構成での実測値です。

| max_seq_length | ピーク VRAM |
| --- | --- |
| 256 | 5.3 GB |
| 384 | 5.8 GB |
| 512 | 6.2 GB |
| 576 | 6.5 GB |
| 640 | OOM |
| 720（既定） | 未計測（640 で OOM のため 8 GB では足りない見込み） |

既定の `720` は 8 GB の GPU では収まらない見込みです。VRAM が足りない場合は
`--max-seq-length` を下げてください。ただし 576 にすると、トークン長が上限を超える
サンプル（3,329 件 / 3.0%）が学習から外れます。

上表は長いバッチをまとめて通した場合の実測です。実際の学習は `batch=1`・
`gradient_accumulation_steps=16`・`group_by_length`（似た長さのサンプルをまとめる）なので、
ピークはもっと低くなります。1 スロット版は `--max-seq-length 704` で 4,000 step
完走しています（`checkpoint-4000/training_args.bin` の `max_length = 704`）。

## データについて

- 埋め込み: `westlake-repl/faiss_index` の `SwissProt/ProTrek_650M_UniRef50/sequence`
  - faiss `IndexFlatIP`、次元 1024、L2 正規化済み
  - `ids.tsv` の行番号 = faiss のベクトル番号
  - `ids.tsv` の 2 列目が配列、3 列目が長さ
- クラスタリング条件は SeqStudio の 50% クラスタリングに合わせています
  （`--min-seq-id 0.5 --cov-mode 1 -c 0.8 --cluster-mode 0`）
- ProTrek のエンコーダは先頭 1024 残基しか見ていないため、それより長い配列は学習から除外します

## 学習の仕組み（要点）

- 入力は `<s> <unk>×10 ▁Seq = <` + 配列。`<unk>` の 10 トークン分の埋め込みだけを ProTrek ベクトルで置き換えます
  （指示部分 `▁Seq = <` は元モデルが事前学習した形のまま残す）
- prefix は文字列 `"<unk>"*10 + "Seq=<"` をトークナイズするのではなく、**トークン ID を直接組み立てます**。
  fast tokenizer は `<unk>` の連続の直後では `Seq` を先頭スペースなしの形（id 23718）にしてしまい、
  事前学習で使われた `▁Seq`（id 25981）にならないためです
- projector は `LayerNorm → Linear(512) → GELU → Linear(512) → GELU` の共通 trunk と、
  スロットごとの `Linear(4096)` ヘッド 10 本で構成します。1 本のベクトルでは情報量が
  足りず（下記）、モデルが「配列らしい composition」に逃げてしまうためです
- 入力は ProTrek ベクトルの**平均ベクトルを引いてから**通します。L2 正規化済みのベクトルでも
  平均ベクトル（ノルム約 0.48）の方向が支配的で、そのままだと全タンパク質が似た入力になります
- 出力は `LayerNorm × 固定ゲイン`（`1/√4096`、ノルム約 1.0）。トークン埋め込みのノルム
  （約 1.08）に合わせるためです。以前は `F.normalize`（L2）を使っていましたが、
  この微分のヤコビアンが `1/‖out‖` で減衰するため、出力ノルムが大きくなると projector の
  勾配が消えて二度と戻らなくなります（実測で LoRA 勾配の 1.4e-3 まで低下）
- projector は PEFT の外側にあるので、optimizer に専用のパラメータグループ
  （lr `1e-4`、weight decay `0.01`）として追加しています。ただし `Trainer` が勾配を
  ゼロにするのは `model.zero_grad()` だけで、これはモデル内のパラメータしか消しません。
  projector の勾配は消えないまま次の step に足し込まれ、実効学習率が step ごとに
  1 倍、2 倍、3 倍…と増えていきます（同一勾配を 4 step 流した検証で 1.03 → 2.06 → 3.09 → 4.12）。
  そのため `on_optimizer_step` で projector の勾配を明示的にゼロにするコールバックと、
  projector 側の勾配クリップ（`_clip_grad_norm`）を入れています
- labels は配列部分と EOS のみ（プロンプト部分は `-100`）
- チェックポイント保存のたびに `checkpoint-N/protein_projector.pt` も一緒に保存します。
  スロット数・prefix・平均ベクトルなどのメタデータも一緒に入れてあり、`inference.py` は
  これを読んで同じ構成の projector を組み直します
  （`Trainer._get_output_dir()` は run ディレクトリを返すので、`checkpoint-N` は自前で付けて
  います。run 直下に書くと全チェックポイントが 1 つのファイルを共有してしまい、`--resume` が
  別 step の projector を読むことになります）
- 学習ログの `projector_slot_cos` は 10 スロット間の平均コサインです。1.0 に近づくなら
  スロットが縮退していて、soft prompt が実質 1 本分の情報しか運んでいないことになります

### なぜ 1 トークンから 10 トークンに変えたか

以前の 1 スロット版では loss が 4.1〜4.2 から下がりませんでした。計測すると:

| 計測 | 値 |
| --- | --- |
| 1 スロット版: 正解配列の CE / シャッフルした配列の CE | 4.1899 / 4.2240 |
| 1 スロット版: 学習後の projector 出力の平均コサイン | 0.954（初期値は 0.275） |
| 1 スロット版: 学習後の projector 出力の実効ランク | 1.45（初期値 159.7） |
| 注入なしで同じ LoRA を 250 step 学習したときの heldout CE | 4.2186（学習 loss 4.2227） |
| 10 スロット版で同じ 250 step 学習したときの heldout CE | 4.2236（学習 loss 4.2071） |
| 10 スロット版のスロット間コサイン（学習後） | 0.030 |
| ユニグラムのエントロピー | 4.7725 |
| 組成だけを当てる oracle の CE | 4.33 |

つまり**タンパク質ベクトルはほとんど使われておらず**、モデルはアミノ酸組成だけを覚えて
いました。原因は (1) 1 トークンでは帯域が足りない、(2) `F.normalize` で勾配が消える、
(3) 平均ベクトルで入力が縮退する、の 3 点です。上の設計はこの 3 点に対応しています。

なお 10 スロット版は 250 step 時点の heldout CE が注入なしとほぼ同じ（4.2236 対 4.2186）です。
これは「改善した」とも「まだ効いていない」とも言えない、判定不能な状態です
（1 スロット版も同じく注入なしと同じ値で、そちらは実際に壊れていました）。
250 step の絶対値では区別できないので、判定には同じモデルで
**正解ベクトルとシャッフルしたベクトルの CE を比べる** 必要があります。
差がなければモデルがスロットを無視している証拠になり、絶対値では区別できません。

## うまくいかないとき

- **`torch.cuda.is_available()` が False**
  torch が cu130 になっていませんか。`uv lock` で `pyproject.toml` の cu129 index 指定が
  残っていることを確認して `uv sync --reinstall-package torch` を実行してください。
- **`MMseqs2 not found`**
  `bash tools/get_mmseqs.sh` を実行するか、`--mmseqs` でパスを指定してください。
- **CUDA OOM**
  `--max-seq-length` を下げてください（上表参照）。
- **`missing: ... (run build_dataset.py first)`**
  先に `build_dataset.py` を実行してください。
- **`Missing key(s) in state_dict: "trunk.0.weight" ...`**
  1 スロット版（旧 `protein_projector.pt`）のチェックポイントです。projector の構成が
  変わったので、学習をやり直すか、`--output-dir` を分けて新しい学習を回してください
  （旧チェックポイントと LoRA アダプタの互換性はありません）。
