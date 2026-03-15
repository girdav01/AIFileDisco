# Training Guide

## Prerequisites
- Python 3.9+
- CUDA 11.8+ (for GPU training)

## Configuration
Edit `configs/training_bert.yaml` to adjust hyperparameters.

## Distributed Training
Use `torchrun --nproc_per_node=4 src/train.py` for multi-GPU training.

## Monitoring
Training metrics are logged to Weights & Biases.
