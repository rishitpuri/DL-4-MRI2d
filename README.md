# DL-4-MRI2d

# Multi-Modal Deep Learning Pipeline for MRI and Genetic Data

## Overview
This project implements a deep learning pipeline for processing and analyzing 2D MRI images and genetic (SNP) data for classification tasks. It supports both uni-modal models (processing either MRI images or genetic data) and multi-modal models (combining both modalities). The pipeline leverages PyTorch for model development, training, and evaluation, and utilizes SLURM for job scheduling on GPU clusters.

The primary components include:
- **Data Loading**: Custom PyTorch datasets for loading MRI images (`MRIDataset.py`, `ImageDataset.py`), genetic data (`GeneDataset.py`, `NewGeneDataset.py`), and combined modalities (`SMDataset.py`).
- **Models**: Pre-trained CNNs (DenseNet, ResNet) for image feature extraction, transformer-based architectures for genetic data processing (`GeneTransformer`), and multi-modal fusion (`SIGNet`).
- **Training**: A flexible training script (`train.py`) supporting different optimizers, loss functions, and training modes (checkpoint resuming, pre-trained weights, or evaluation).
- **SLURM Integration**: A SLURM job script (`job.sh`) for running training jobs on GPU clusters.

## Project Structure
```
project/
├── logs/                       # Output and error logs from SLURM jobs
├── runs/                       # Directory for saving trained models and checkpoints
├── GeneDataset.py             # Dataset for loading genetic data
├── ImageDataset.py            # Dataset for loading MRI images
├── MRIDataset.py              # Dataset for loading MRI images from directory structure
├── NewGeneDataset.py          # Dataset for loading genetic data from VCF files
├── SMDataset.py               # Dataset for loading combined MRI and genetic data
├── genetest.py                # Script for testing VCF file loading
├── job.sh                     # SLURM script for submitting training jobs
├── model.py                   # CNN-based feature extraction models (DenseNet, ResNet, TestNet)
├── multi_modal.py             # Multi-modal model (SIGNet) combining image and genetic data
├── train.py                   # Training and evaluation script
├── transformer.py             # Transformer architecture components (MultiHeadAttention, EncoderLayer)
├── uni_modal.py               # Uni-modal models (ImagingNet, GeneTransformer)
└── README.md                  # This file
```

## Dataset
The dataset consists of:
- **2D MRI Images**: Grayscale images stored in a directory structure (used by `MRIDataset.py`) or referenced in a CSV file (used by `ImageDataset.py` and `SMDataset.py`).
- **Genetic Data**: SNP data stored in a VCF file (`VCF_useThis.vcf`) or CSV files with phenotype labels and genetic features (e.g., `phenotypes_final.csv`).
- **CSV Format**: For `ImageDataset.py`, `GeneDataset.py`, and `SMDataset.py`, the CSV files are expected to have columns like:
  - `IID`: Sample identifier
  - `Image Path`: Path to the MRI image
  - `Label`: Classification label (e.g., 0, 1, 2 for three classes)
  - `SNP1`, `SNP2`, ..., or `PC1`, `PC2`, ...: Genetic features or principal components

The `NewGeneDataset.py` loads SNP data directly from a VCF file and phenotype labels from a CSV file.

## Dependencies
- Python 3.8+
- PyTorch
- Torchvision
- Torchmetrics
- Pandas
- NumPy
- Scikit-learn
- Scikit-allel (for VCF file processing)
- Pillow (PIL)
- SLURM (for cluster job submission)

Install dependencies using:
```bash
pip install torch torchvision torchmetrics pandas numpy scikit-learn scikit-allel pillow
```

## Setup
1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd project
   ```

2. **Prepare the Dataset**:
   - Place MRI images in a directory structure compatible with `MRIDataset.py` (e.g., subdirectories for each class) or prepare a CSV file with image paths and labels for `ImageDataset.py` or `SMDataset.py`.
   - Ensure the VCF file (`VCF_useThis.vcf`) and phenotype CSV (`phenotypes_final.csv`) are accessible at the paths specified in `NewGeneDataset.py` or other dataset scripts.
   - Update file paths in `job.sh` and dataset scripts if necessary.

3. **Load Modules** (for cluster environments):
   On a SLURM cluster, load the required Python and GPU modules:
   ```bash
   module load python/gpu
   ```

## Usage
### Training a Model
To train a model, use the `job.sh` script to submit a job to the SLURM cluster. Modify the script parameters as needed:
```bash
sbatch job.sh
```

The `job.sh` script runs `train.py` with arguments specified in the script, such as:
- `--data`: Path to the phenotype CSV file (e.g., `/N/project/SingleCell_Image/Nischal/Dilip/FINAL-FILES/phenotypes_final.csv`)
- `--batch_size`: Batch size (e.g., 128)
- `--epochs`: Number of training epochs (e.g., 50)
- `--ttv_split`: Train-test-validation split (e.g., 0.85 0.10 0.05)
- `--optim`: Optimizer (`adam` or `sgd`)
- `--optim_params`: Optimizer parameters (e.g., 0.001 0.0001 for Adam)
- `--model`: Model class (e.g., `uni_modal.GeneTransformer`)
- `--workers`: Number of data loader workers (e.g., 1)

Example command for running `train.py` directly:
```bash
python train.py --data /path/to/phenotypes_final.csv --batch_size 128 --epochs 50 --ttv_split 0.85 0.10 0.05 --optim adam --optim_params 0.001 0.0001 --model uni_modal.GeneTransformer --workers 1
```

### Modes
The `train.py` script supports three modes:
- **CH** (Checkpoint): Resume training from a checkpoint.
- **PT** (Pre-Trained): Load pre-trained weights and continue training.
- **E** (Evaluation): Evaluate a trained model on the test set.

Specify the mode using the `--mode` argument and provide a checkpoint file with `--checkpoint` if needed.

### Outputs
- **Logs**: Training and error logs are saved in the `logs/` directory (`output_%j.txt` and `error_%j.err`).
- **Models**: Trained models and checkpoints are saved in the `runs/` directory (e.g., `trained/model_<random_id>.pth`).

## Models
- **Uni-Modal Models** (`uni_modal.py`):
  - `ImagingNet`: Processes MRI images using a DenseNet-based architecture.
  - `GeneTransformer`: Processes genetic data using a transformer-based architecture with two encoder layers.
- **Multi-Modal Model** (`multi_modal.py`):
  - `SIGNet`: Combines MRI image features (from DenseNet) and genetic data (processed by a transformer) using element-wise multiplication.
- **Feature Extraction** (`model.py`):
  - `DenseNetFeatures`, `DenseNetFeaturesOnly`, `ResNetFeatures`: CNN-based feature extractors for images.
  - `TestNet`: A lightweight CNN for testing purposes.

## Notes
- The `NewGeneDataset.py` loads SNP data from a VCF file and organizes it into tuples of 64 SNPs for transformer processing.
- The `train.py` script currently uses the `NewGeneDataset` for genetic data, but commented-out code supports `ImageDataset`, `MRIDataset`, and `SMDataset` for image and multi-modal data.
- The `genetest.py` script is a utility for testing VCF file loading and conversion to a numeric format.
- Ensure the input dimensions (e.g., image size: 192x192, SNP vector size) match the model expectations.
- The pipeline assumes three-class classification (labels 0, 1, 2). Adjust the model output layers and loss function if the number of classes differs.

## Future Improvements
- Add support for weighted loss to handle class imbalance (uncomment relevant code in `train.py`).
- Optimize data loading for large datasets by increasing the number of workers (`--workers`).
- Extend the pipeline to support additional modalities or model architectures.
- Implement data augmentation in the transform pipeline for improved generalization.

## Contact
- www.rishitpuri.com 
