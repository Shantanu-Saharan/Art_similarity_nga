# Portrait Similarity Retrieval using Weakly Supervised Embeddings

## Overview

This project addresses Task 2: Similarity from the ArtExtract (HumanAI) GSoC evaluation.

The goal is to build a system that can retrieve visually similar paintings, with a focus on portrait similarity (face, pose, and style) using the National Gallery of Art (NGA) open dataset.

I implement an end-to-end retrieval pipeline that learns meaningful visual embeddings using weak supervision and performs efficient similarity search using FAISS.

---

## Key Idea

Since the NGA dataset does not provide explicit annotations for face similarity or pose, I adopt a weakly supervised learning approach:

- Use artist metadata to construct triplets (anchor, positive, negative)
- Train a model to learn a semantic embedding space
- Enhance retrieval using a combined feature mode:
  - Full-image representation
  - Face-aware cues (via MediaPipe)

This allows the model to capture both global visual style and localized facial structure.

---

## Pipeline

The pipeline runs in order: `run_prepare.py` downloads and filters NGA portraits by title keyword. `run_train.py` fine-tunes a ViT backbone with triplet loss using artist labels as weak supervision, and `run_embed.py` extracts embeddings in `full` or `combined` mode. `run_build_index.py` wraps the embeddings in a FAISS index, while `run_search.py` and `run_evaluate.py` handle retrieval and metrics.

---

## Results

### Quantitative Evaluation (Combined Mode)

- Precision@1: 0.4474  
- Precision@5: 0.2842  
- Precision@10: 0.1763  
- mAP: 0.5423  

mAP ~0.54 is decent given the label noise — Precision@1 dropping to ~0.45 is expected since artist attribution isn't a strict similarity signal.

The model performs well on stylistic and artist-level similarity, but struggles on cross-domain queries (e.g., photographic vs painted portraits).

---

## Qualitative Examples

Some retrievals match on artist (Glenn Ligon's portraits cluster well), others on period (17th-century works group by palette and composition). Pose similarity works in certain queries but is inconsistent — mediapipe fails on more stylized faces.

See `outputs/` for sample retrieval visualizations.

---

## Design Choices

I initially tried using cosine similarity on pretrained features, but results were inconsistent across different artists and periods. Triplet loss with artist-based weak supervision proved much more stable. The artist-as-proxy label is quite noisy — many paintings are attributed or from a workshop rather than the actual artist, but it's the best signal available at scale.

---

## Limitations

- No explicit ground-truth labels for face or pose similarity  
- Weak supervision introduces noise  
- Performance is constrained by dataset size (~200 portraits)  
- Artist attribution can be unreliable (workshop vs individual artist)  

---

## Project Structure

```
├── run_prepare.py
├── run_train.py
├── run_embed.py
├── run_build_index.py
├── run_search.py
├── run_evaluate.py
├── models/
├── outputs/
├── notebooks/
└── src/
```

---

## How to Run

Prepare data:
```bash
python run_prepare.py --limit 400
```

Train:
```bash
python run_train.py --model vit --epochs 5
```

Generate embeddings:
```bash
python run_embed.py --model vit --feature_mode combined
```

Build index:
```bash
python run_build_index.py --model vit --feature_mode combined --pca_dim 128
```

Run search:
```bash
python run_search.py --query_filename painting_100870.jpg --model vit --feature_mode combined --pca_dim 128
```

Evaluate:
```bash
python run_evaluate.py --model vit --feature_mode combined --n_queries 50
```

---

## Notes

- Developed as part of the GSoC ArtExtract evaluation (Task 2)  
- Focused on building a clean, reproducible retrieval pipeline  
- Model checkpoint (`.pth`) is not included due to size constraints and can be regenerated using the training script  
