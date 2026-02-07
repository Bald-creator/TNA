# TNA

Hierarchical graph Transformer for brain network analysis (e.g. psychiatric disorder classification).

**License:** This project is licensed under the MIT License — see the [LICENSE](LICENSE) file.

## Requirements

See `requirements.txt`. Key dependencies are pinned for reproducibility.

```bash
pip install -r requirements.txt
```

## Data and checkpoints

- **Data:** This repository does not include datasets. Place your data under a base directory (see below).
- **Model weights:** Trained checkpoints are not included. To share weights for reproduction, use GitHub Releases or Zenodo and document the download path here or in the paper.

## Data directory layout

Use a single base directory (e.g. `/path/to/base_dir`) with this structure:

```
<base_dir>/
  data/
    raw/             # raw data
    processed/       # (optional) preprocessed .pt
    atlas_metadata/
  logs/
    output/          # model checkpoints and results
    tensorboard/
```

## How to run

### Step 1: Clone and install

```bash
git clone <your-repo-url>
cd TNA
pip install -r requirements.txt
```

### Step 2: Prepare data

- Put your data under `<base_dir>/data/raw/` (and optionally preprocessed under `data/processed/`).
- Put atlas metadata under `<base_dir>/data/atlas_metadata/` as required by the data loaders.

### Step 3: Train

From the **TNA project root** (the directory that contains `tna/` and `scripts/`):

```bash
python scripts/train.py --base_dir /path/to/base_dir
```

- `--base_dir` is **required**: the path to the base directory that contains `data/` and `logs/`.
- Other arguments are optional (e.g. `--dataset`, `--atlas`, `--epochs`, `--batch_size`, `--lr`, `--kfold`, `--gpu`, `--seed`). Omitted options use defaults from `tna.configs.model_config.TNAConfig`.

**Example with options:**

```bash
python scripts/train.py --base_dir /path/to/base_dir --dataset REST-MDD --atlas cc200 --epochs 70 --batch_size 64 --kfold 10 --gpu 0
```

**Dual-atlas mode:**

```bash
python scripts/train.py --base_dir /path/to/base_dir --dual_atlas
```

### Step 4: Evaluate

After training, checkpoints are saved under `<base_dir>/logs/output/<run_name>/`. To evaluate:

```bash
python scripts/evaluate.py --base_dir /path/to/base_dir --checkpoint-dir /path/to/base_dir/logs/output/<run_name>
```

Example:

```bash
python scripts/evaluate.py --base_dir /path/to/base_dir --checkpoint-dir /path/to/base_dir/logs/output/REST-MDD_cc200_20250101_120000
```

Optional arguments: `--dataset`, `--atlas`, `--kfold`, `--batch_size`, `--gpu` (same as training).

## Optional: save attention weights

To dump attention weights to a file for analysis:

```bash
export TNA_ATTN_LOG_DIR=/path/to/log/dir
python scripts/train.py --base_dir /path/to/base_dir
```

Weights are appended to `attention_weights.txt` under that directory.

## Citation

If you use this code, please cite our paper (add your paper reference here).
