from pathlib import Path
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchvision.models as models
from pytorch_lightning.callbacks import ModelCheckpoint
import wandb
from pytorch_lightning.loggers import WandbLogger
from PIL import Image

import os
import pyprojroot
from pyprojroot import here
root = pyprojroot.find_root(pyprojroot.has_dir(".git"))

import sys
sys.path.append(str(root))

from src.dataset.bps_datamodule import BPSDataModule
from src.dataset.bps_dataset import BPSMouseDataset
from src.dataset.augmentation import ToTensor

import boto3
from botocore import UNSIGNED
from botocore.config import Config

from matplotlib import pyplot as plt
from torchvision.transforms import transforms
from dataclasses import dataclass

@dataclass
class BPSConfig:
    """
    Configuration options for BPS Microscopy dataset.
    """
    data_dir:           str = root / 'data' / 'processed'
    train_dir:          str = data_dir / 'train-hi_hr_4'
    validation_dir:     str = data_dir / 'validation-hi_hr_4'
    train_csv_file:     str = 'meta_dose_hi_hr_4_post_exposure_train.csv'
    validation_csv_file:str = 'meta_dose_hi_hr_4_post_exposure_valid.csv'

    bucket_name:        str = "nasa-bps-training-data"
    s3_path:            str = "Microscopy/train"
    s3_client:          str = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    s3_meta_fname:      str = "meta.csv"

    save_dir:           str = root / 'src' / 'model'/ 'resnet101'
    save_models_dir:    str = root / 'models' / 'baselines'
    checkpoints_dir:    str = root / 'src' / 'model' / 'checkpoints'

    resize_dims:        tuple = (224, 224)
    batch_size:         int = 64
    max_epochs:         int = 3
    accelerator:        str = 'auto'
    acc_devices:        int = 1
    device:             str = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_workers:        int = 4

class ResNetModel(pl.LightningModule):
    def __init__(self, num_classes, label_mapping, learning_rate=1e-3):
        super(ResNetModel, self).__init__()
        self.resnet = models.resnet101(pretrained=True)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)

        self.loss_fn = nn.CrossEntropyLoss()
        self.label_mapping = label_mapping
        self.learning_rate  = learning_rate

    def forward(self, x):
        return self.resnet(x)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        targets = torch.tensor([self.label_mapping[label] for label in targets])

        inputs = inputs.to(self.device)
        targets = targets.to(self.device)

        outputs = self(inputs)
        loss = self.loss_fn(outputs, targets)

        # self.log('train_loss', loss)
        wandb.log({'train_loss': loss})
        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        targets = torch.tensor([self.label_mapping[label] for label in targets])

        inputs = inputs.to(self.device)
        targets = targets.to(self.device)

        outputs = self(inputs)
        loss = self.loss_fn(outputs, targets)
    
        # Calculate accuracy
        _, predicted_labels = torch.max(outputs, dim=1)
        accuracy = torch.sum(predicted_labels == targets).item() / targets.size(0)

        # Log validation loss and accuracy with global step
        # self.log("val_loss", loss, on_step=True, on_epoch=True)
        # self.log("val_accuracy", accuracy, on_step=True, on_epoch=True)
        wandb.log({'val_loss': loss})
        wandb.log({'val_accuracy': accuracy})

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
    
    def load_from_checkpoint(self, checkpoint_path):
        self.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.eval()

    def predict(self, image_path):
        image = Image.open(image_path).convert("RGB")
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        input_tensor = transform(image).unsqueeze(0).to(self.device)
        output = self(input_tensor)
        _, predicted_idx = torch.max(output, 1)
        predicted_label = [k for k, v in self.label_mapping.items() if v == predicted_idx.item()][0]
        return predicted_label
    
def main():
    # Initialize a BPSConfig object
    config = BPSConfig()

    # Initialize wanb project and wanb logger
    wandb.init(
        # set the wandb project where this run will be logged
        project="resnet101",
        dir=config.save_dir
    )
    # wandb_logger = WandbLogger(wandb.config.project, log_model=True)

    # Create an instance of the ResNetModel
    num_classes = 2  # Number of classes in bps mouse dataset
    label_mapping = {'Fe': 0, 'X-ray': 1}
    model = ResNetModel(num_classes, label_mapping, learning_rate=wandb.config.lr)

    # Define the BPSDataModule for your dataset
    data_module = BPSDataModule(
        train_csv_file = config.train_csv_file, 
        train_dir = config.train_dir, 
        val_csv_file = config.validation_csv_file, 
        val_dir = config.validation_dir, 
        resize_dims = config.resize_dims,
        meta_csv_file = config.s3_meta_fname,
        meta_root_dir = config.s3_path,
        s3_client = config.s3_client,
        bucket_name = config.bucket_name,
        s3_path = config.s3_path,
        data_dir = config.data_dir,
        num_workers = config.num_workers,
        batch_size=wandb.config.batch_size
    )

    # data_module.prepare_data()
    data_module.setup(stage='train')
    data_module.setup(stage='validate')

    # Define the PyTorch Lightning Trainer and train the model
    pl.seed_everything(42)
    trainer = pl.Trainer(max_epochs=wandb.config.epochs,
                        #  logger=wandb_logger,
                         default_root_dir=config.save_dir,
                         accelerator=config.accelerator,
                         devices=config.acc_devices)
    trainer.fit(model=model,
            train_dataloaders=data_module.train_dataloader(),
            val_dataloaders=data_module.val_dataloader())

    # Save trained model to checkpoint
    checkpoint_path = 'checkpoints/resnet101_model-hi_hr_4-sweep_best_params.pth'
    torch.save(model.state_dict(), checkpoint_path)

    wandb.finish()

def sweep():
    sweep_config = {
            'method': 'random',
            'name': 'sweep',
            'metric': {
                'goal': 'minimize', 
                'name': 'loss'
                },
            'parameters': {
                'batch_size': {'values': [32]}, #{'values': [16, 32, 64]},
                'epochs': {'values': [15]}, #{'values': [5, 10, 15]},
                'lr': {'values': [0.007957998759594619
    ]}, #{'min': 0.0003, 'max': 0.01}
            }
        }

    sweep_id = wandb.sweep(
            sweep=sweep_config,
            project="resnet101"
    )

    wandb.agent(sweep_id=sweep_id, function=main, count=1)

def predict():
    num_classes = 2
    label_mapping = {'Fe': 0, 'X-ray': 1}
    model = ResNetModel(num_classes, label_mapping)

    checkpoint_path = "checkpoints/resnet101_model-hi_hr_4-sweep_best_params.pth"
    model.load_from_checkpoint(checkpoint_path)

    model.eval()

    print("Predicted Labels:")
    for i in range(1, 6):
        image_path = f"synthetic{i}.png"
        predicted_label = model.predict(image_path)
        print(image_path, predicted_label)

    return 1

if __name__ == '__main__':
    # sweep()

    predict()
    