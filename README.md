## Vision Transformer (ViT) from Scratch: Oxford-IIIT Pet Classification

This project implements the Vision Transformer (ViT) architecture from the ground up using PyTorch and einops, specifically designed for fine-grained image classification on the Oxford-IIIT Pet Dataset.

## 🚀 Overview

The goal of this project was to move away from traditional Convolutional Neural Networks (CNNs) and implement the "An Image is Worth 16x16 Words" methodology. By treating image patches as tokens in a sequence, we leverage the Self-Attention Mechanism to capture global spatial relationships.

Key Features:

Custom Patch Embedding: Efficiently "shreds" images into patches using einops Rearrange.

Multi-Head Self-Attention: Implemented a modular Attention class to allow the model to focus on different visual features (texture, shape, color) in parallel.

Residual Connections & Pre-Norm: Utilized modern Transformer stabilization techniques to ensure smooth gradient flow.

CLS Token & Positional Encoding: Included a learnable classification token and positional embeddings to maintain spatial context.

## 🏗️ Architecture Detail

Input Size: 144x144 pixels (RGB)

Patch Size: 8x8

Embedding Dimension: 128

Encoder Depth: 6 Layers

Attention Heads: 8

Dropout: 0.1 (Regularization)

## 📊 Dataset

The Oxford-IIIT Pet Dataset consists of ~3,700 images of 37 different breeds of cats and dogs.

Training/Validation Split: 80% / 20%

Augmentations: Random Horizontal Flips and Random Rotations (10°) to prevent overfitting.


## 🛠️ Installation & Usage

Clone the repository


Install Dependencies

Bash

pip install torch torchvision einops matplotlib scikit-learn

Run Training:

python transformers.py


## 🧠 Lessons Learned

Transformer Sensitivity: Unlike CNNs, Transformers are highly sensitive to hyperparameters such as the learning rate (3e-4 "Karpathy Constant") and initialization.

Data Hunger: Training from scratch on a small dataset (~3.6k images) is a significant challenge for Transformers, often requiring thousands of epochs or pre-training to achieve high accuracy.

Modular Design: Building the model with Residual and PreNorm wrappers makes the architecture significantly more readable and easier to debug.

