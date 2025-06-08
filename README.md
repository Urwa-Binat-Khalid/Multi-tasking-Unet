# Multi-tasking-Unet
Multi-task U-Net for end-to-end image classification and segmentation, ideal for medical imaging and other vision applications.

# Multi-task U-Net Project

This repository contains the implementation of a Multi-task U-Net deep learning model designed to simultaneously perform semantic segmentation and binary classification on input images. The model leverages shared convolutional features to improve overall performance by jointly optimizing for multiple related tasks.

# Project Overview

Multi-task learning can enhance model generalization and efficiency by sharing knowledge across related tasks. This project implements a Multi-task U-Net architecture that addresses two tasks concurrently:

- **Segmentation:** Pixel-wise classification of images into multiple classes (3 classes in this project), using a U-Net inspired encoder-decoder architecture.
- **Classification:** Binary classification of the entire image to detect the presence or absence of a specific attribute.

The model utilizes custom metrics and loss weighting to balance task performance effectively. This approach is particularly beneficial in medical imaging, remote sensing, or other domains where segmentation and classification are complementary.

---

## Features

- **Multi-task Architecture:** A single model with shared layers and task-specific output heads.
- **Custom Metrics:** Includes Dice Coefficient, Mean Intersection over Union (IoU), Precision, Recall, F1-Score, and Matthews Correlation Coefficient (MCC) to thoroughly evaluate model performance.
- **Flexible Input Shape:** Currently configured for 256x256 RGB images but can be adapted.
- **Loss Weighting:** Adjustable loss weights to prioritize segmentation or classification during training.
- **Optimized with Adam optimizer** with a custom learning rate.

---

## Repository Structure

| File                     | Description                                          |
|--------------------------|------------------------------------------------------|
| `multi_task_unet_model.py` | Defines the multi-task U-Net model and compiles it with losses, metrics, and optimizers. |
| `custom_metrics.py`       | Contains implementations of custom metrics used in model evaluation. |

---
