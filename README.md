### Code used for BMED 6517 project: Deep Learning Driven Epithelial segmentation in Oral Cavity Squamous Cell Carcinoma Whole Slide Images


## Project Structure
```
.
├── extract_patches.py   # WSI patch extraction script
├── unet.py              # unet architecture python file
├── train_unet.py        # Main training script
└── README.md
```

## Requirements

```
torch
pytorch-lightning  
albumentations
wandb
histoprep
monai
torchmetrics
pillow
numpy
scikit-learn
tqdm
```

## Training

The training script implements:
- Three loss functions: Cross Entropy, Dice Loss, and Focal Loss
- Ablation study with reduced encoder/decoder blocks
- Data augmentation with horizontal/vertical flips, rotations, and brightness adjustments
- Wandb logging and model checkpointing

To train the models:
```bash
python train_unet.py
```

This will train different models:
- baseline models (one with each loss function)
- ablation models with reduced depth

## Model Architecture

Base UNet architecture parameters:
- Input size: 2000x2000 pixels
- Initial features: 64
- Depth: 4 (baseline) / 3 (ablation)
- Batch normalization enabled
- ReLU activation functions

