# Art Similarity Retrieval for NGA Task 2

This repository contains the final version of my artwork similarity project for NGA Task 2. The goal is to learn image embeddings that retrieve visually related works while still respecting artist identity as much as possible.

The final workflow in this repository has three main stages:

1. Build one unified metadata file from several local portrait image sources.
2. Train a retrieval model with a fine-tuned ViT backbone and retrieval-oriented losses.
3. Evaluate on the NGA Task-2 benchmark using model fusion, database-side augmentation, and optional face-aware reranking.

The main scripts used in the final pipeline are:

- `build_metadata_2100.py`
- `train_improved.py`
- `run_task2_fair_benchmark.py`

## What is included in this repository

- Source code for metadata building, training, and evaluation
- Shared model and path configuration code in `src/`
- A processed metadata CSV at `data/processed/all_images_metadata.csv`
- Final benchmark outputs in `results/task2_fair_benchmark/`
- A fine-tuned checkpoint in `models/portrait_vit_improved_l40.pth`

Two important notes:

- The training image folders are large. If you clone a cleaned version of this repository later, you may choose not to keep all raw images on GitHub.
- The `Task-2/` benchmark folder is not present in this working copy. The benchmark script expects that folder to be added locally before evaluation.

## Project summary

The training set combines portraits from four sources:

- WikiArt
- National Gallery of Art
- Metropolitan Museum of Art
- A small manually collected auxiliary set

Current metadata statistics from `data/processed/all_images_metadata.csv`:

- Total rows: `2169`
- Labelled images: `2167`
- Labelled artists: `421`
- Source breakdown: `1982` WikiArt, `104` NGA, `77` Met, `6` Alternative

The two training stages used for the final model were:

- Stage 1 with `min_per_artist=2`: `2006` images across `260` artists
- Stage 2 with `min_per_artist=3`: `1758` images across `136` artists

The final benchmark report in `results/task2_fair_benchmark/report.json` uses:

- `300` indexed benchmark images
- `204` evaluated queries
- artist identity as the relevance label

## Final results

The kept benchmark setting uses:

- ResNet50 weight: `0.38`
- EfficientNet-B0 weight: `0.26`
- Fine-tuned ViT weight: `0.36`
- DBA: `top_k=2`, `alpha=3.0`
- Face rerank: `mode=hybrid`, `face_weight=0.10`, `top_n=20`, `padding=0.20`

Reported metrics:

| Metric | Score |
| --- | ---: |
| Precision@1 | `0.348039` |
| Precision@5 | `0.220588` |
| Precision@10 | `0.152451` |
| Recall@5 | `0.317420` |
| Recall@10 | `0.424275` |
| MAP@10 | `0.296091` |
| MRR | `0.420647` |

Compared with the baseline stored in the report, this corresponds to:

- `1.35x` improvement in Precision@10
- `1.28x` improvement in Recall@10
- `1.21x` improvement in MAP@10
- `1.12x` improvement in MRR

## Repository structure

```text
Art_similarity_nga-main/
├── build_metadata_2100.py
├── train_improved.py
├── run_task2_fair_benchmark.py
├── README.md
├── IMPROVEMENTS.md
├── requirements.txt
├── src/
│   ├── config.py
│   └── models/
│       ├── __init__.py
│       └── embedding_model.py
├── data/
│   ├── processed/
│   │   └── all_images_metadata.csv
│   └── raw/
│       ├── alternative/
│       ├── images/
│       ├── met/
│       ├── nga/
│       └── wikiart/
├── models/
│   └── portrait_vit_improved_l40.pth
└── results/
    └── task2_fair_benchmark/
```

## Method and architecture

### 1. Metadata building

`build_metadata_2100.py` merges all local training sources into one CSV.

What it does:

- Reads the NGA portrait images from `data/raw/images/`
- Reads WikiArt images from `data/raw/wikiart/`
- Reads Met images from `data/raw/met/`
- Reads an optional extra set from `data/raw/alternative/`
- Cleans artist names
- Uses CSV metadata when available
- Falls back to filename-based artist inference for WikiArt and Met images when necessary
- Verifies that image files actually exist on disk
- Writes the final merged CSV to `data/processed/all_images_metadata.csv`

The output CSV contains the filename, image directory, artist, title, source, and date for each image.

### 2. Embedding model

`src/models/embedding_model.py` defines a small wrapper around pretrained torchvision backbones.

The model structure is:

- A pretrained backbone (`vit_b_16`, `resnet50`, or `resnet18`)
- Removal of the original classification head
- A projection head with two linear layers and a ReLU
- L2 normalization on the final embedding

For the final training path, the main model is `vit_b_16` with a `512`-dimensional embedding space.

### 3. Training pipeline

`train_improved.py` is the training script used for the final checkpoint.

Important training details:

- The dataset reads the unified metadata CSV and loads images from their source folders
- The sampler builds batches in `P x K` form, meaning multiple artists per batch and multiple images per artist
- Training uses three losses together:
  - supervised contrastive loss
  - batch-hard triplet loss
  - cross-entropy loss on artist labels
- The ViT backbone is only partially unfrozen, so the last few transformer blocks are trained more aggressively than the earlier blocks
- Checkpoints are saved using the lowest training loss seen so far

This setup is meant to learn embeddings that work for retrieval rather than plain classification.

### 4. Benchmark pipeline

`run_task2_fair_benchmark.py` performs evaluation on the NGA Task-2 benchmark.

The evaluation pipeline is:

1. Load benchmark metadata and images.
2. Extract ResNet50 embeddings.
3. Extract EfficientNet-B0 embeddings using the helper code inside the Task-2 package.
4. Extract embeddings from the fine-tuned ViT checkpoint.
5. Concatenate the three embeddings with fixed weights and normalize the result.
6. Apply database-side augmentation (DBA).
7. Optionally detect faces with OpenCV Haar cascades and rerank the shortlist using face-crop embeddings.
8. Compute retrieval metrics and write reports.

The benchmark script reports:

- Precision@1
- Precision@5
- Precision@10
- Recall@5
- Recall@10
- MAP@10
- MRR

One important clarification: the final benchmark script in this repository does not build a FAISS index. Retrieval is done with normalized embedding similarity and reranking. Older notes and figures in the repository mention FAISS from earlier experimentation, but the final submission path here is the fusion-plus-rerank pipeline described above.

## Data layout and how to prepare the data

If you already have a full local copy of the project data, you can skip this section. If you are rebuilding the project from scratch, the code expects the following directory layout.

### Training data layout

```text
data/
├── processed/
│   └── all_images_metadata.csv
└── raw/
    ├── images/
    │   └── painting_<OBJECTID>.jpg
    ├── nga/
    │   ├── objects.csv
    │   └── published_images.csv
    ├── wikiart/
    │   ├── wikiart_portraits.csv
    │   └── *.jpg
    ├── met/
    │   ├── met_portraits.csv
    │   └── *.jpg
    └── alternative/
        ├── alternative_portraits.csv
        └── *.jpg / *.png
```

### What each folder is used for

- `data/raw/nga/objects.csv` and `data/raw/nga/published_images.csv`
  - NGA metadata files kept with the project.
  - The current metadata builder reads `objects.csv` directly for title, artist, and date information.
- `data/raw/images/`
  - Local NGA training images.
  - The code expects filenames like `painting_12345.jpg`.
- `data/raw/wikiart/`
  - Local WikiArt portrait images.
  - `wikiart_portraits.csv` is used when available, but the script can also infer artists from filenames.
- `data/raw/met/`
  - Local Met portrait images.
  - `met_portraits.csv` helps map filenames to artists.
- `data/raw/alternative/`
  - Small optional extra portrait set with its own CSV.

### How to get the data

This repository does not contain a single script that bulk-downloads the full training set from all sources. The training code assumes that the images are already available locally in the folders above.

Practical setup:

- Obtain the NGA metadata CSVs and place them under `data/raw/nga/`.
- Download or collect the NGA training images you want to use and store them as `painting_<OBJECTID>.jpg` in `data/raw/images/`.
- Place WikiArt portrait images in `data/raw/wikiart/` and keep `wikiart_portraits.csv` there if you have it.
- Place Met portrait images in `data/raw/met/` and keep `met_portraits.csv` there if you have it.
- Place any extra images in `data/raw/alternative/`.

### Benchmark data layout

The benchmark script expects a separate `Task-2/` folder at the repository root:

```text
Task-2/
├── data/
│   ├── images/
│   │   └── <OBJECTID>.jpg
│   └── processed/
│       └── nga_similarity_metadata.csv
└── task2/
    └── nga_similarity.py
```

This repository does not currently include that folder, so you need to add it locally from the benchmark package provided for the task.

If you already have `Task-2/data/processed/nga_similarity_metadata.csv` but not the images, the benchmark script can download missing benchmark images automatically:

```bash
python run_task2_fair_benchmark.py --download-missing
```

That option only handles benchmark image download. It does not download the full training corpus.

## Installation

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

The main scripts depend on PyTorch, torchvision, pandas, OpenCV, requests, and tqdm. Some packages in `requirements.txt` were kept from earlier experiments, but the final code path is centered on the three scripts described in this README.

## How to run the project

### Step 1: Build the unified metadata CSV

```bash
python build_metadata_2100.py
```

Expected output:

- `data/processed/all_images_metadata.csv`

### Step 2: Train the first-stage model

This stage trains on artists with at least two images.

```bash
python train_improved.py \
  --model vit \
  --metadata_csv data/processed/all_images_metadata.csv \
  --checkpoint_path models/portrait_vit_improved_l40.pth \
  --epochs 24 \
  --batch 48 \
  --samples_per_class 2 \
  --workers 4 \
  --min_per_artist 2 \
  --unfreeze_blocks 8 \
  --warmup_epochs 4
```

### Step 3: Run the second-stage refinement

This stage starts from the previous checkpoint and keeps only artists with at least three images.

```bash
python train_improved.py \
  --model vit \
  --metadata_csv data/processed/all_images_metadata.csv \
  --init_checkpoint models/portrait_vit_improved_l40.pth \
  --checkpoint_path models/portrait_vit_improved_l40.pth \
  --epochs 20 \
  --batch 48 \
  --samples_per_class 3 \
  --workers 4 \
  --min_per_artist 3 \
  --unfreeze_blocks 12 \
  --warmup_epochs 2 \
  --lr_head 1.5e-4 \
  --lr_backbone 1e-5 \
  --label_smoothing 0.05
```

Expected output:

- Updated checkpoint at `models/portrait_vit_improved_l40.pth`

### Step 4: Run benchmark evaluation

After adding the `Task-2/` benchmark folder, run:

```bash
python run_task2_fair_benchmark.py \
  --metadata-csv Task-2/data/processed/nga_similarity_metadata.csv \
  --image-dir Task-2/data/images \
  --vit-checkpoint models/portrait_vit_improved_l40.pth \
  --resnet-weight 0.38 \
  --efficientnet-weight 0.26 \
  --vit-weight 0.36 \
  --dba-k 2 \
  --dba-alpha 3.0 \
  --face-detector-mode hybrid \
  --face-weight 0.10 \
  --face-rerank-topn 20 \
  --face-padding 0.20
```

If the benchmark metadata is present but the image files are missing, you can add `--download-missing`.

Expected outputs in `results/task2_fair_benchmark/`:

- `report.txt`
- `report.json`
- `selection_summary.txt`
- `qualitative_examples.md`
- `qualitative_examples.json`
- `index_metadata.csv`
- `embeddings.pt`

## Useful output files

- `data/processed/all_images_metadata.csv`
  - Final merged training metadata.
- `models/portrait_vit_improved_l40.pth`
  - Final fine-tuned checkpoint used in evaluation.
- `results/task2_fair_benchmark/report.txt`
  - Human-readable benchmark summary.
- `results/task2_fair_benchmark/report.json`
  - Full machine-readable benchmark report.
- `results/task2_fair_benchmark/qualitative_examples.md`
  - A few example retrieval cases.
- `results/task2_fair_benchmark/images/`
  - Figures created during analysis and write-up.

## Current limitations

- The full training data is large, so cloning and sharing the entire raw dataset through GitHub is not ideal.
- The benchmark package under `Task-2/` must be added separately.
- Training is GPU-friendly but can be slow on CPU.
- Artist-based retrieval still struggles on difficult cases where many paintings share similar composition, iconography, or style.

## Short project explanation

In simple terms, this project first collects portrait paintings from multiple sources, then trains a model so that paintings by the same artist end up closer together in embedding space. During evaluation, it does not rely on only one model. Instead, it combines general-purpose visual features from ResNet50 and EfficientNet-B0 with a fine-tuned ViT model that was trained specifically for this task. That fused representation is then improved with database-side augmentation, and for portraits with detectable faces, the ranking can be refined using face-crop embeddings.

This combination gave a better balance than using a single backbone alone, especially on the 300-image NGA benchmark.
# Art Similarity Retrieval for NGA Task 2

This repository contains the final version of my artwork similarity project for NGA Task 2. The goal is to learn image embeddings that retrieve visually related works while still respecting artist identity as much as possible.

The final workflow in this repository has three main stages:

1. Build one unified metadata file from several local portrait image sources.
2. Train a retrieval model with a fine-tuned ViT backbone and retrieval-oriented losses.
3. Evaluate on the NGA Task-2 benchmark using model fusion, database-side augmentation, and optional face-aware reranking.

The main scripts used in the final pipeline are:

- `build_metadata_2100.py`
- `train_improved.py`
- `run_task2_fair_benchmark.py`

## What is included in this repository

- Source code for metadata building, training, and evaluation
- Shared model and path configuration code in `src/`
- A processed metadata CSV at `data/processed/all_images_metadata.csv`
- Final benchmark outputs in `results/task2_fair_benchmark/`
- A fine-tuned checkpoint in `models/portrait_vit_improved_l40.pth`

Two important notes:

- The training image folders are large. If you clone a cleaned version of this repository later, you may choose not to keep all raw images on GitHub.
- The `Task-2/` benchmark folder is not present in this working copy. The benchmark script expects that folder to be added locally before evaluation.

## Project summary

The training set combines portraits from four sources:

- WikiArt
- National Gallery of Art
- Metropolitan Museum of Art
- A small manually collected auxiliary set

Current metadata statistics from `data/processed/all_images_metadata.csv`:

- Total rows: `2169`
- Labelled images: `2167`
- Labelled artists: `421`
- Source breakdown: `1982` WikiArt, `104` NGA, `77` Met, `6` Alternative

The two training stages used for the final model were:

- Stage 1 with `min_per_artist=2`: `2006` images across `260` artists
- Stage 2 with `min_per_artist=3`: `1758` images across `136` artists

The final benchmark report in `results/task2_fair_benchmark/report.json` uses:

- `300` indexed benchmark images
- `204` evaluated queries
- artist identity as the relevance label

## Final results

The kept benchmark setting uses:

- ResNet50 weight: `0.38`
- EfficientNet-B0 weight: `0.26`
- Fine-tuned ViT weight: `0.36`
- DBA: `top_k=2`, `alpha=3.0`
- Face rerank: `mode=hybrid`, `face_weight=0.10`, `top_n=20`, `padding=0.20`

Reported metrics:

| Metric | Score |
| --- | ---: |
| Precision@1 | `0.348039` |
| Precision@5 | `0.220588` |
| Precision@10 | `0.152451` |
| Recall@5 | `0.317420` |
| Recall@10 | `0.424275` |
| MAP@10 | `0.296091` |
| MRR | `0.420647` |

Compared with the baseline stored in the report, this corresponds to:

- `1.35x` improvement in Precision@10
- `1.28x` improvement in Recall@10
- `1.21x` improvement in MAP@10
- `1.12x` improvement in MRR

## Repository structure

```text
Art_similarity_nga-main/
├── build_metadata_2100.py
├── train_improved.py
├── run_task2_fair_benchmark.py
├── README.md
├── IMPROVEMENTS.md
├── requirements.txt
├── src/
│   ├── config.py
│   └── models/
│       ├── __init__.py
│       └── embedding_model.py
├── data/
│   ├── processed/
│   │   └── all_images_metadata.csv
│   └── raw/
│       ├── alternative/
│       ├── images/
│       ├── met/
│       ├── nga/
│       └── wikiart/
├── models/
│   └── portrait_vit_improved_l40.pth
└── results/
    └── task2_fair_benchmark/
```

## Method and architecture

### 1. Metadata building

`build_metadata_2100.py` merges all local training sources into one CSV.

What it does:

- Reads the NGA portrait images from `data/raw/images/`
- Reads WikiArt images from `data/raw/wikiart/`
- Reads Met images from `data/raw/met/`
- Reads an optional extra set from `data/raw/alternative/`
- Cleans artist names
- Uses CSV metadata when available
- Falls back to filename-based artist inference for WikiArt and Met images when necessary
- Verifies that image files actually exist on disk
- Writes the final merged CSV to `data/processed/all_images_metadata.csv`

The output CSV contains the filename, image directory, artist, title, source, and date for each image.

### 2. Embedding model

`src/models/embedding_model.py` defines a small wrapper around pretrained torchvision backbones.

The model structure is:

- A pretrained backbone (`vit_b_16`, `resnet50`, or `resnet18`)
- Removal of the original classification head
- A projection head with two linear layers and a ReLU
- L2 normalization on the final embedding

For the final training path, the main model is `vit_b_16` with a `512`-dimensional embedding space.

### 3. Training pipeline

`train_improved.py` is the training script used for the final checkpoint.

Important training details:

- The dataset reads the unified metadata CSV and loads images from their source folders
- The sampler builds batches in `P x K` form, meaning multiple artists per batch and multiple images per artist
- Training uses three losses together:
  - supervised contrastive loss
  - batch-hard triplet loss
  - cross-entropy loss on artist labels
- The ViT backbone is only partially unfrozen, so the last few transformer blocks are trained more aggressively than the earlier blocks
- Checkpoints are saved using the lowest training loss seen so far

This setup is meant to learn embeddings that work for retrieval rather than plain classification.

### 4. Benchmark pipeline

`run_task2_fair_benchmark.py` performs evaluation on the NGA Task-2 benchmark.

The evaluation pipeline is:

1. Load benchmark metadata and images.
2. Extract ResNet50 embeddings.
3. Extract EfficientNet-B0 embeddings using the helper code inside the Task-2 package.
4. Extract embeddings from the fine-tuned ViT checkpoint.
5. Concatenate the three embeddings with fixed weights and normalize the result.
6. Apply database-side augmentation (DBA).
7. Optionally detect faces with OpenCV Haar cascades and rerank the shortlist using face-crop embeddings.
8. Compute retrieval metrics and write reports.

The benchmark script reports:

- Precision@1
- Precision@5
- Precision@10
- Recall@5
- Recall@10
- MAP@10
- MRR

One important clarification: the final benchmark script in this repository does not build a FAISS index. Retrieval is done with normalized embedding similarity and reranking. Older notes and figures in the repository mention FAISS from earlier experimentation, but the final submission path here is the fusion-plus-rerank pipeline described above.

## Data layout and how to prepare the data

If you already have a full local copy of the project data, you can skip this section. If you are rebuilding the project from scratch, the code expects the following directory layout.

### Training data layout

```text
data/
├── processed/
│   └── all_images_metadata.csv
└── raw/
    ├── images/
    │   └── painting_<OBJECTID>.jpg
    ├── nga/
    │   ├── objects.csv
    │   └── published_images.csv
    ├── wikiart/
    │   ├── wikiart_portraits.csv
    │   └── *.jpg
    ├── met/
    │   ├── met_portraits.csv
    │   └── *.jpg
    └── alternative/
        ├── alternative_portraits.csv
        └── *.jpg / *.png
```

### What each folder is used for

- `data/raw/nga/objects.csv` and `data/raw/nga/published_images.csv`
  - NGA metadata files kept with the project.
  - The current metadata builder reads `objects.csv` directly for title, artist, and date information.
- `data/raw/images/`
  - Local NGA training images.
  - The code expects filenames like `painting_12345.jpg`.
- `data/raw/wikiart/`
  - Local WikiArt portrait images.
  - `wikiart_portraits.csv` is used when available, but the script can also infer artists from filenames.
- `data/raw/met/`
  - Local Met portrait images.
  - `met_portraits.csv` helps map filenames to artists.
- `data/raw/alternative/`
  - Small optional extra portrait set with its own CSV.

### How to get the data

This repository does not contain a single script that bulk-downloads the full training set from all sources. The training code assumes that the images are already available locally in the folders above.

Practical setup:

- Obtain the NGA metadata CSVs and place them under `data/raw/nga/`.
- Download or collect the NGA training images you want to use and store them as `painting_<OBJECTID>.jpg` in `data/raw/images/`.
- Place WikiArt portrait images in `data/raw/wikiart/` and keep `wikiart_portraits.csv` there if you have it.
- Place Met portrait images in `data/raw/met/` and keep `met_portraits.csv` there if you have it.
- Place any extra images in `data/raw/alternative/`.

### Benchmark data layout

The benchmark script expects a separate `Task-2/` folder at the repository root:

```text
Task-2/
├── data/
│   ├── images/
│   │   └── <OBJECTID>.jpg
│   └── processed/
│       └── nga_similarity_metadata.csv
└── task2/
    └── nga_similarity.py
```

This repository does not currently include that folder, so you need to add it locally from the benchmark package provided for the task.

If you already have `Task-2/data/processed/nga_similarity_metadata.csv` but not the images, the benchmark script can download missing benchmark images automatically:

```bash
python run_task2_fair_benchmark.py --download-missing
```

That option only handles benchmark image download. It does not download the full training corpus.

## Installation

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

The main scripts depend on PyTorch, torchvision, pandas, OpenCV, requests, and tqdm. Some packages in `requirements.txt` were kept from earlier experiments, but the final code path is centered on the three scripts described in this README.

## How to run the project

### Step 1: Build the unified metadata CSV

```bash
python build_metadata_2100.py
```

Expected output:

- `data/processed/all_images_metadata.csv`

### Step 2: Train the first-stage model

This stage trains on artists with at least two images.

```bash
python train_improved.py \
  --model vit \
  --metadata_csv data/processed/all_images_metadata.csv \
  --checkpoint_path models/portrait_vit_improved_l40.pth \
  --epochs 24 \
  --batch 48 \
  --samples_per_class 2 \
  --workers 4 \
  --min_per_artist 2 \
  --unfreeze_blocks 8 \
  --warmup_epochs 4
```

### Step 3: Run the second-stage refinement

This stage starts from the previous checkpoint and keeps only artists with at least three images.

```bash
python train_improved.py \
  --model vit \
  --metadata_csv data/processed/all_images_metadata.csv \
  --init_checkpoint models/portrait_vit_improved_l40.pth \
  --checkpoint_path models/portrait_vit_improved_l40.pth \
  --epochs 20 \
  --batch 48 \
  --samples_per_class 3 \
  --workers 4 \
  --min_per_artist 3 \
  --unfreeze_blocks 12 \
  --warmup_epochs 2 \
  --lr_head 1.5e-4 \
  --lr_backbone 1e-5 \
  --label_smoothing 0.05
```

Expected output:

- Updated checkpoint at `models/portrait_vit_improved_l40.pth`

### Step 4: Run benchmark evaluation

After adding the `Task-2/` benchmark folder, run:

```bash
python run_task2_fair_benchmark.py \
  --metadata-csv Task-2/data/processed/nga_similarity_metadata.csv \
  --image-dir Task-2/data/images \
  --vit-checkpoint models/portrait_vit_improved_l40.pth \
  --resnet-weight 0.38 \
  --efficientnet-weight 0.26 \
  --vit-weight 0.36 \
  --dba-k 2 \
  --dba-alpha 3.0 \
  --face-detector-mode hybrid \
  --face-weight 0.10 \
  --face-rerank-topn 20 \
  --face-padding 0.20
```

If the benchmark metadata is present but the image files are missing, you can add `--download-missing`.

Expected outputs in `results/task2_fair_benchmark/`:

- `report.txt`
- `report.json`
- `selection_summary.txt`
- `qualitative_examples.md`
- `qualitative_examples.json`
- `index_metadata.csv`
- `embeddings.pt`

## Useful output files

- `data/processed/all_images_metadata.csv`
  - Final merged training metadata.
- `models/portrait_vit_improved_l40.pth`
  - Final fine-tuned checkpoint used in evaluation.
- `results/task2_fair_benchmark/report.txt`
  - Human-readable benchmark summary.
- `results/task2_fair_benchmark/report.json`
  - Full machine-readable benchmark report.
- `results/task2_fair_benchmark/qualitative_examples.md`
  - A few example retrieval cases.
- `results/task2_fair_benchmark/images/`
  - Figures created during analysis and write-up.

## Current limitations

- The full training data is large, so cloning and sharing the entire raw dataset through GitHub is not ideal.
- The benchmark package under `Task-2/` must be added separately.
- Training is GPU-friendly but can be slow on CPU.
- Artist-based retrieval still struggles on difficult cases where many paintings share similar composition, iconography, or style.

## Short project explanation

In simple terms, this project first collects portrait paintings from multiple sources, then trains a model so that paintings by the same artist end up closer together in embedding space. During evaluation, it does not rely on only one model. Instead, it combines general-purpose visual features from ResNet50 and EfficientNet-B0 with a fine-tuned ViT model that was trained specifically for this task. That fused representation is then improved with database-side augmentation, and for portraits with detectable faces, the ranking can be refined using face-crop embeddings.

This combination gave a better balance than using a single backbone alone, especially on the 300-image NGA benchmark.
