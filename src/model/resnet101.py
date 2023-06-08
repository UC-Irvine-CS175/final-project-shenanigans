import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchvision.models as models
from pytorch_lightning.callbacks import ModelCheckpoint
import wandb
from pytorch_lightning.loggers import WandbLogger

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

from matplotlib import pyplot as plt, transforms

class ResNetModel(pl.LightningModule):
    def __init__(self, num_classes, label_mapping):
        super(ResNetModel, self).__init__()
        self.resnet = models.resnet101(pretrained=True)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)

        self.loss_fn = nn.CrossEntropyLoss()
        self.label_mapping = label_mapping

    def forward(self, x):
        return self.resnet(x)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        targets = torch.tensor([self.label_mapping[label] for label in targets])

        inputs = inputs.to(self.device)
        targets = targets.to(self.device)

        outputs = self(inputs)
        loss = self.loss_fn(outputs, targets)

        self.log("train_loss", loss, on_step=True, on_epoch=True)  # Log training loss with global step
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
        self.log("val_loss", loss, on_step=True, on_epoch=True)
        self.log("val_accuracy", accuracy, on_step=True, on_epoch=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer
    
def main():
    wandb_logger = WandbLogger(project='resnet101', log_model=True)

    # Create an instance of the ResNetModel
    num_classes = 2  # Number of classes in bps mouse dataset
    label_mapping = {'Fe': 0, 'X-ray': 1}
    model = ResNetModel(num_classes, label_mapping)

    # Define the BPSDataModule for your dataset
    data_dir = os.path.join(root, 'data', 'processed')
    train_dir = os.path.join(data_dir, 'train-hi_hr_4')
    validation_dir = os.path.join(data_dir, 'validation-hi_hr_4')

    train_csv_file = 'meta_dose_hi_hr_4_post_exposure_train.csv'
    validation_csv_file = 'meta_dose_hi_hr_4_post_exposure_valid.csv'

    bucket_name = "nasa-bps-training-data"
    s3_path = "Microscopy/train"
    s3_client = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    s3_meta_fname = "meta.csv"


    data_module = BPSDataModule(
        train_csv_file = train_csv_file, 
        train_dir = train_dir, 
        val_csv_file = validation_csv_file, 
        val_dir = validation_dir, 
        resize_dims=(224, 224),
        meta_csv_file = s3_meta_fname,
        meta_root_dir=s3_path,
        s3_client= s3_client,
        bucket_name=bucket_name,
        s3_path=s3_path,
        data_dir=data_dir,
        num_workers=4
    )

    # data_module.prepare_data()
    data_module.setup(stage='train')
    data_module.setup(stage='validate')

    # Define the PyTorch Lightning Trainer and train the model
    pl.seed_everything(42)
    trainer = pl.Trainer(max_epochs=3, logger=wandb_logger)
    trainer.fit(model=model,
            train_dataloaders=data_module.train_dataloader(),
            val_dataloaders=data_module.val_dataloader())

    # Save trained model to checkpoint
    checkpoint_path = 'checkpoints/resnet101_model-hi_hr_4-epoch_3.pth'
    torch.save(model.state_dict(), checkpoint_path)

if __name__ == '__main__':
    main()