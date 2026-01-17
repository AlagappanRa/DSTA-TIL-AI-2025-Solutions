# DSTA TIL-AI Brainhacks 2025: Advanced Computer Vision & ASR Solutions

[![Award](https://img.shields.io/badge/Result-National%20Winner-gold)]()
[![Task](https://img.shields.io/badge/Task-Object%20Detection%20%7C%20ASR-blue)]()
[![Framework](https://img.shields.io/badge/Tech-PyTorch%20%7C%20NVIDIA%20NeMo%20%7C%20DINO-red)]()

> **Context:** Solutions for the **DSTA Brainhacks TIL-AI 2025 Challenge**, a high-stakes AI competition focusing on defense-related computer vision and speech recognition tasks under strict compute constraints.

## Executive Summary
This repository contains the source code and optimization strategies used to achieve state-of-the-art performance in the competition.
*   **Computer Vision:** Achieved **94.8% mAP** on the Advanced Track using a DINO-DETR architecture with a Swin-Large backbone.
*   **ASR:** Achieved **98.2% Accuracy** on noisy speech transcription using a fine-tuned NVIDIA Parakeet-TDT Transducer model.

---

## Computer Vision: DINO-DETR Optimization

We tackled the 18-class object detection task (including small targets like drones and missiles) by deploying **DINO (DETR with Improved DeNoising Anchor Boxes)**.

### Key Technical Implementations
*   **Backbone:** Utilized `swin_L_384_22k` (Swin Transformer Large), a hierarchical Vision Transformer, to capture global context for dense scenes.
*   **Custom Operators:** Compiled custom **CUDA kernels** for Multi-Scale Deformable Attention (MSDeformAttn) to run efficiently within the contest's environment.
*   **Multi-Scale Training:** Increased `num_feature_levels` to 5 and implemented aggressive data augmentation (scales 480-800) to resolve the "small object" detection bottleneck.

### Performance Metrics (Test Set)
The model demonstrated exceptional performance on small objects, traditionally the hardest category in detection tasks.

| Metric | IoU Threshold | Area | Value |
| :--- | :--- | :--- | :--- |
| **mAP (Primary)** | **0.50:0.95** | **All** | **0.941** |
| AP | 0.50 | All | 0.993 |
| AP | 0.75 | All | 0.972 |
| **AP (Small)** | **0.50:0.95** | **Small** | **0.742** |
| AP (Medium) | 0.50:0.95 | Medium | 0.923 |
| AP (Large) | 0.50:0.95 | Large | 0.954 |

> *Note: Achieving 74.2% AP on Small objects is significantly higher than standard YOLOv8 baselines.*

---

## Automatic Speech Recognition (ASR): Parakeet-TDT

For the ASR challenge, we focused on transcribing noisy, accented English speech. We moved beyond standard CTC models to utilize **RNN-Transducer (RNN-T)** architectures.

### Optimization Strategy
*   **Model Selection:** Utilized **Parakeet-TDT 0.6b** (Transducer with Fast Conformer encoder) for its robustness to noise.
*   **Catastrophic Forgetting Mitigation:** Implemented a strategic **Layer Freezing** protocol. We froze the pre-encoder and the first 8 layers of the Conformer encoder, training only the top layers to adapt to the specific domain accents without losing general phonetic understanding.
*   **Text Normalization:** Integrated custom JiWER transformations (Regex substitution, punctuation removal) directly into the validation loop to align optimization objectives with competition scoring.

### Results
*   **Word Error Rate (WER):** < 1.8%
*   **Final Accuracy:** **98.2%**

---

## Engineering & Infrastructure

Deploying these heavy models required significant systems engineering to handle dependency management and GPU throughput.

*   **Mixed Precision Training:** Enabled FP16 (AMP) for DINO and 32-bit mixed capability for ASR to maximize throughput on A100 GPUs.
*   **Dependency Management:** Resolved low-level library conflicts by patching `pycocotools` and manually aligning PyTorch/CUDA versions to support custom operator compilation.
*   **Checkpointing:** Implemented robust checkpoint management to handle preemptible training environments.

---

## Repository Structure

```text
.
├── ASR
│   ├── FinetunedASR.ipynb             # Full training pipeline with NeMo & Parakeet-TDT
│   └── FinetunedASR_no_outputs.ipynb  # Clean version for code review
└── DINO
    ├── Finetuned_DINO.ipynb           # DINO-DETR implementation with Swin-L
    └── Finetuned_DINO_no_outputs.ipynb # Clean version