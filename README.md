# ASD Face Alignment Diagnosis - PyTorch Replication

## Overview
This repository contains a PyTorch implementation replicating the methodology detailed in the paper: **"POWER OF ALIGNMENT: EXPLORING THE EFFECT OF FACE ALIGNMENT ON ASD DIAGNOSIS USING FACIAL IMAGES"** by Mohammad Shafiul Alam et al. (IIUM Engineering Journal, 2024).

The core objective of this project is to classify facial images of children to aid in the early diagnosis of Autism Spectrum Disorder (ASD). The methodology demonstrates that applying facial alignment using MTCNN (Multi-task Cascaded Convolutional Networks) prior to training significantly improves the prediction accuracy of deep learning architectures like ResNet50.

## Dataset Preparation
The project utilizes the publicly available "children facial image ASD dataset" from Kaggle. 
The dataset consists of 3014 samples (1:1 ratio of autistic to non-autistic children) and must be structured as follows:

* **Train**: 2654 samples
* **Validation**: 80 samples
* **Test**: 280 samples

Download the dataset and place the images in the `dataset/` directory following the structure outlined below.

## Project Structure
```text
asd-alignment-project/
|
|-- dataset/                    # The raw dataset extracted from Kaggle
|   |-- train/                  
|   |   |-- Autistic/
|   |   |-- Non_Autistic/
|   |-- valid/                  
|   |   |-- Autistic/
|   |   |-- Non_Autistic/
|   |-- test/                   
|   |       |-- Autistic/
|       |-- Non_Autistic/
|
|-- saved_models/               # Destination for saving model weights
|   |-- .gitkeep
|
|-- alignment_test.ipynb        # Jupyter notebook for visual verification of MTCNN alignment
|-- main.py                     # Main training and evaluation script
|-- pyproject.toml              # Project metadata and dependencies
|-- uv.lock                     # Locked dependency tree for deterministic builds
```


## Installation
This project uses uv for lightning-fast, deterministic dependency management.
1. Ensure you have Python 3.8+ installed.
2. Install uv if you haven't already (e.g., curl -LsSf https://astral.sh/uv/install.sh | sh).
3. Clone this repository and navigate into the directory.
4. Sync the environment using the locked dependencies:
```
uv sync
```
This command reads the uv.lock file and perfectly recreates the required environment, including torch, torchvision, facenet-pytorch, scikit-learn, matplotlib, and jupyter.

Usage
You do not need to manually activate a virtual environment. Prefix your execution commands with uv run.

1. **Visual Verification (Exploratory Phase)**  
Before initiating a full training run, it is highly recommended to verify the MTCNN alignment logic.
Launch the Jupyter Notebook to visualize the landmark detection and rotation mechanism on a few sample images.
```
uv run jupyter notebook alignment_test.ipynb
```
2. **Model Training and Evaluation**  
Run the primary script to begin the training pipeline.
```
uv run main.py
```
The script performs the following operations:

1. Loads the PyTorch ImageFolder datasets.
2. Applies MTCNN face alignment on-the-fly during data loading (calculates eye displacement angle, rotates anti-clockwise, and crops the face).
3. Initializes a pretrained ResNet50 architecture (ImageNet weights) and modifies the top layer for binary classification.
4. Trains the model using Adagrad (lr=0.001) and Binary Cross Entropy with Logits loss.
5. Evaluates the model on the test set, outputting Accuracy, AUC, Precision, and Recall metrics.

__Note: For faster training iterations, you may modify the script to preprocess and save the aligned images to disk rather than performing the MTCNN operation on-the-fly during the dataloader sequence.__

## Target Benchmarks
Based on the original paper, the ResNet50 model with aligned training samples is expected to achieve the following approximate benchmarks on the test set:

- Accuracy: ~93.9%
- AUC: ~96.3%

## References  
Alam, M. S., Rashid, M. M., Faizabadi, A. R., & Zaki, H. F. M. (2024). Power of Alignment: Exploring the Effect of Face Alignment on ASD Diagnosis Using Facial Images. IIUM Engineering Journal, 25(1), 317-327.