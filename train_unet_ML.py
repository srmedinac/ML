from torch import nn
import torch
import numpy as np
import os
import UNet
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from albumentations import *
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import torchmetrics
from monai.losses import DiceLoss, FocalLoss

# Configuration
patch_size = 2000  
batch_size = 16  
num_workers = 8
n_classes = 2
in_channels = 3
num_epochs = 90
lr = 1e-4

# Paths for OCSCC dataset
wsi_dir = "home/OCSCC/patches"
mask_dir = "home/OCSCC/masks"
wsis = os.listdir(wsi_dir)

# Split dataset
train_wsis, val_wsis = train_test_split(wsis, test_size=0.2, random_state=42)

class OCSCCDataset(Dataset):
    def __init__(self, wsi_dir, wsis, mask_dir, transform=None):
        self.wsis = wsis
        self.wsi_dir = wsi_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_names = []
        self.mask_names = []

        for wsi in self.wsis:
            wsi_image_dir = os.path.join(self.wsi_dir, wsi)
            for img_file in os.listdir(wsi_image_dir):
                if img_file.endswith('.png'):
                    image_path = os.path.join(wsi_image_dir, img_file)
                    mask_path = os.path.join(self.mask_dir, wsi, img_file)
                    if os.path.exists(mask_path):
                        self.image_names.append(image_path)
                        self.mask_names.append(mask_path)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image = np.array(Image.open(self.image_names[idx]))
        mask = np.array(Image.open(self.mask_names[idx]))
        
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        image = np.transpose(image, (2, 0, 1)) / 255.0
        mask = (mask > 0).astype(np.int64)
        
        return torch.FloatTensor(image), torch.LongTensor(mask)

# Data augmentation
transforms = Compose([
    HorizontalFlip(p=0.5),
    VerticalFlip(p=0.5),
    RandomRotate90(p=0.5),
    RandomBrightnessContrast(p=0.2),
])

class OCSCCUNet(pl.LightningModule):
    def __init__(self, n_classes, loss_type='ce', ablation=False):
        super().__init__()
        depth = 3 if ablation else 4
        self.model = UNet(
            n_classes=n_classes,
            in_channels=3,
            depth=depth,
            wf=6,
            padding=True,
            batch_norm=True
        )
        
        self.loss_type = loss_type
        if loss_type == 'ce':
            self.criterion = nn.CrossEntropyLoss()
        elif loss_type == 'dice':
            self.criterion = DiceLoss(sigmoid=True)
        elif loss_type == 'focal':
            self.criterion = FocalLoss(gamma=2.0)
            
        self.dice_metric = torchmetrics.Dice(num_classes=n_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        
        loss = self.criterion(y_hat, y)
        pred = torch.argmax(y_hat, dim=1)
        dice_score = self.dice_metric(pred, y)
        
        self.log(f'train_loss_{self.loss_type}', loss)
        self.log('train_dice', dice_score)
        
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        pred = torch.argmax(y_hat, dim=1)
        dice_score = self.dice_metric(pred, y)
        self.log('val_dice', dice_score)
        
        if batch_idx == 0:
            self.logger.experiment.log({
                "examples": [wandb.Image(x[0], caption="Input"),
                           wandb.Image(y[0], caption="Ground Truth"),
                           wandb.Image(pred[0], caption="Prediction")]
            })

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [scheduler]

def train_model_with_loss(loss_type, ablation=False):
    train_dataset = OCSCCDataset(wsi_dir, train_wsis, mask_dir, transform=transforms)
    val_dataset = OCSCCDataset(wsi_dir, val_wsis, mask_dir, transform=None)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    model = OCSCCUNet(n_classes=n_classes, loss_type=loss_type, ablation=ablation)
    
    exp_name = f"ablation_{loss_type}" if ablation else f"baseline_{loss_type}"
    wandb_logger = WandbLogger(project=f"ocscc_epithelial_segmentation_{exp_name}")
    
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        accelerator="gpu",
        logger=wandb_logger,
        callbacks=[
            EarlyStopping(monitor="val_dice", patience=10, mode="max"),
            ModelCheckpoint(
                dirpath=f"checkpoints_{exp_name}",
                filename=f"ocscc_{exp_name}_best",
                monitor="val_dice",
                mode="max"
            )
        ]
    )
    
    trainer.fit(model, train_loader, val_loader)
    return trainer.callback_metrics["val_dice"].item()

def main():
    loss_types = ['ce', 'dice', 'focal']
    results = {}
    
    # Train baseline models
    for loss_type in loss_types:
        print(f"\nTraining baseline with {loss_type} loss...")
        val_dice = train_model_with_loss(loss_type, ablation=False)
        results[f"baseline_{loss_type}"] = val_dice
    
    # Train ablation models (reduced depth)
    for loss_type in loss_types:
        print(f"\nTraining ablation study (reduced depth) with {loss_type} loss...")
        val_dice = train_model_with_loss(loss_type, ablation=True)
        results[f"ablation_{loss_type}"] = val_dice
    
    print("\nFinal Results:")
    print("\nBaseline Models:")
    for loss_type in loss_types:
        score = results[f"baseline_{loss_type}"]
        print(f"{loss_type.upper()} Loss - Validation Dice Score: {score:.4f}")
    
    print("\nAblation Study (Reduced Depth):")
    for loss_type in loss_types:
        score = results[f"ablation_{loss_type}"]
        print(f"{loss_type.upper()} Loss - Validation Dice Score: {score:.4f}")

if __name__ == "__main__":
    main()