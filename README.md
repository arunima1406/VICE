# VICE
Vehicle Insurance Cost Estimator

## Overview
This project automates the process of estimating insurance costs for vehicles by analyzing car images for damages using a Vision Transformer (ViT) deep learning model. The ViT model is fine-tuned on a dataset of labeled car damage images. The 'vision' is to reduce turnaround time for insurance inspections of damaged cars and estimate costs with a fair and unbiased model.

## Project Structure

### 1. Libraries Imported

- PIL for image processing
- 'sklearn/' for train-test split and evaluation metrics
- 'torch/' and torchvision for deep learning pipeline
- 'transformers/' for Vision Transformer model and processor
- 'tqdm/' for progress bars during training


### 2. Dataset

- Dataset downloaded from [Kaggle](https://www.kaggle.com/datasets/sudhanshu2198/ripik-hackfest?resource=download)

- Labels: 
   - 0: crack
   - 1: scratch
   - 2: tire flat
   - 3: dent
   - 4: glass shatter
   - 5: lamp broken

### 3. Feature Extraction & Image Processing

- Using the ViTImageProcessor from Hugging Face pretrained on ImageNet
- Images normalized and resized to 224x224

### 4. DataLoaders

- Created PyTorch DataLoaders for training and validation sets with a batch size of 16

### 5. Model Definition

- Classification head modified for 6 classes corresponding to damage types.
- Model moved to GPU
- Defined loss (CrossEntropyLoss) and optimizer (AdamW)

## Steps to Execute

1. Download the entire project zip file or clone the repository.
   ```bash
   git clone https://github.com/arunima1406/VICE
   ```
2. Download the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Jupyter Notebook file to initialize and test the ViT model:
   ```bash
   jupyter notebook notebook.ipynb
   ```
4. After completing the Jupyter Notebook execution, run the Python script for real-time testing:
   ```bash
   python main.py
   ```
5. Test the model using sample images from the web.

## Requirements
### Ensure you have the following dependencies installed:
- Python 3.9
- PyTorch
- TorchVision
- ##Vision transformer Dependencies
- ##NumPy
- Matplotlib

You can install them using:
```bash
pip install -r requirements.txt
```

## Future Scope

- Add a user friendly interface to provide accessibility to users.
- Expand the dataset to include other types of vehicles and their damages.
- Include other cost estimation techniques required by car insurance companies into an algorithm.
