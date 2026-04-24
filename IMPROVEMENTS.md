# Improvements & Evolution

This document explains what I changed between my first submission and this updated version, why I made those changes, and how much better the results got.

---

## Overview of the Pipeline

![Pipeline Overview](results/task2_fair_benchmark/images/pipeline_overview.png)

The biggest shift was moving from a simple single-model retrieval system to a proper multi-stage pipeline with model fusion and reranking. The image above shows how the new pipeline is structured end to end.

---

## What I Started With

My first submission trained only on NGA portrait images — roughly 200 images filtered by title keywords like "portrait" and "self-portrait". The model was a single fine-tuned ViT with triplet loss, and I was using FAISS with PCA compression to 128 dimensions for retrieval. The evaluation was on 38 self-designed queries.

The results looked decent on paper (mAP ~0.54) but I was essentially testing on the same small dataset I trained on, which wasn't a fair evaluation at all.

---

## What Changed and Why

### 1. Dataset — I added way more training data

The biggest problem with the first version was the tiny training set. Triplet/contrastive learning just doesn't work well when most artists only have 1-2 images — you can't form good positive pairs. So I pulled in data from four sources:

| Source | Images |
|---|---:|
| WikiArt | 1,982 |
| NGA | 104 |
| Metropolitan Museum of Art | 77 |
| Auxiliary (manually collected) | 6 |
| **Total** | **2,169** |

I wrote `build_metadata_2100.py` to merge everything into one unified CSV, clean up artist names, and handle cases where only filename-based artist inference was possible (which happens a lot with WikiArt). Going from ~200 to 2,169 images made a noticeable difference in training stability.

I also split training into two stages — first with `min_per_artist=2` to use as much data as possible, then a second stage with `min_per_artist=3` to refine on higher-quality clusters.

---

### 2. Architecture — Moved from single model to ensemble

![Architecture Evolution](results/task2_fair_benchmark/images/architecture_evolution.png)

In the first version I was only using a fine-tuned ViT, with an optional face-aware mode that fused full-image features with MediaPipe face landmarks. This turned out to be a bad idea for historical paintings — MediaPipe just can't detect faces reliably in stylised, non-photorealistic artwork from the 16th or 17th century.

For this version I switched to a three-model ensemble at inference:

| Model | Weight |
|---|---:|
| ResNet50 (pretrained) | 0.38 |
| EfficientNet-B0 (pretrained) | 0.26 |
| Fine-tuned ViT-B/16 | 0.36 |

The idea is that no single model captures everything. ResNet50 is really good at texture and local patterns, EfficientNet adds complementary features with a different scaling strategy, and the fine-tuned ViT handles the portrait-specific semantic similarity. The three embeddings get concatenated with the weights above and L2-normalised.

The face-aware MediaPipe channel from the old version is still there, but now it's an optional reranking step that runs *after* initial retrieval rather than being baked into the embedding itself. This way it only activates when a face is actually detected, and doesn't hurt performance when it can't find one.

---

### 3. Embedding Design

![Embedding Comparison](results/task2_fair_benchmark/images/embedding_comparison.png)

A few specific things I changed about the embedding:

- **512-dimensional space** for the fine-tuned ViT. Not too large, not too small — large enough to capture fine-grained differences but compact enough to compute similarity quickly.
- **L2 normalisation** on everything before fusion. Without this, embedding magnitude dominates the similarity score and you get weird retrieval behaviour.
- **Projection head** between backbone and embedding (two linear layers + ReLU). This decouples the backbone's pretrained features from the retrieval embedding, so fine-tuning doesn't aggressively distort what the backbone already learned from ImageNet.

---

### 4. Training — Better losses and sampling

The old version used offline triplet mining — I pre-mined triplets before training started. The problem is that as the model gets better, those triplets become too easy and stop giving useful gradients. This version uses three losses together:

1. **Supervised Contrastive Loss** — pulls together all embeddings from the same artist in each batch, not just pairs. More stable gradient signal.
2. **Batch-Hard Triplet Loss** — online mining that picks the hardest positive and hardest negative in each batch. Keeps training challenging as the model improves.
3. **Cross-Entropy Loss** — an auxiliary classification head to prevent the embedding space from collapsing, especially early in training.

The batch sampler builds `P × K` batches (P artists, K images per artist) to guarantee genuine positive pairs in every batch. The ViT backbone is also only partially unfrozen — the last few transformer blocks train more aggressively, the earlier blocks stay close to pretrained. This worked better than freezing everything or unfreezing everything.

---

### 5. Retrieval — Dropped FAISS, added DBA and face reranking

![FAISS Workflow](results/task2_fair_benchmark/images/faiss_workflow.png)

I was using FAISS with PCA compression in the first version to keep things fast. But on a 300-image benchmark, exact similarity search is fast enough without approximation, and PCA was actually hurting precision on fine-grained distinctions. So I dropped it.

Instead I added two post-processing steps:

**Database-Side Augmentation (DBA):** Each database embedding gets replaced by a weighted average of itself and its top-2 neighbours. This smooths the embedding space and makes retrieval more robust to small query variations — works surprisingly well for a simple idea.

**Face-Aware Reranking:** After getting the initial top-20 results, I run an OpenCV Haar cascade face detector on the query and results. If faces are detected, face-crop similarity gets blended into the final score with a weight of 0.10. It's a soft boost rather than a hard rerank, so it doesn't break things when no face is found.

---

## Before and After

![Before After](results/task2_fair_benchmark/images/before_vs_after.png)

The qualitative difference is visible — retrieval results are more visually and stylistically coherent, especially for artists with a distinctive style like Frans Hals.

---

## Final Results

![Top-K Results](results/task2_fair_benchmark/images/topk_results.png)

Evaluated on the standardised NGA Task-2 benchmark: 300 indexed images, 204 queries.

| Metric | Our Results | Competitor Baseline | Improvement |
|---|---:|---:|---:|
| Precision@10 | 0.1525 | 0.1130 | **1.35×** |
| Recall@10 | 0.4243 | 0.3325 | **1.28×** |
| MAP@10 | 0.2961 | 0.2449 | **1.21×** |
| MRR | 0.4206 | 0.3772 | **1.12×** |

Best case: *Portrait of a Member of the Haarlem Civic Guard* (Frans Hals) — rank-1 score 0.999, Precision@5 = 1.0, all top-5 results from the same artist.

Worst case: *The Madonna of Humility* (Fra Angelico) — no relevant results in top 10. When an artist has only one work in the benchmark, exact artist-identity retrieval isn't really possible. The model retrieves compositionally similar works from the same period, which is visually reasonable but scores as zero by the artist-identity metric.

---

## Things I Tried That Didn't Work

- **Pretrained ViT without fine-tuning** — cosine similarity on off-the-shelf ViT features clusters by ImageNet object category, not by artistic style. Results were all over the place.
- **MediaPipe as a primary feature channel** — fails too often on historical portraits to be reliable as a main embedding component.
- **PCA to 128 dims** — made things faster but hurt precision noticeably. Not worth it at this corpus size.
- **ResNet18** — faster but clearly weaker than ResNet50 on this task.
- **Triplet loss alone** — loss would occasionally diverge, and performance was less stable than the three-loss combination.

---

## Key Takeaways

The two changes that made the biggest difference were (1) expanding the training dataset from ~200 to 2,169 images across multiple sources, and (2) switching to a multi-model ensemble at inference. The DBA postprocessing was a nice bonus that required almost no extra work to implement.

The evaluation also became fairer — 204 queries on a standardised benchmark is much more meaningful than 38 self-designed queries on the same dataset I trained on.