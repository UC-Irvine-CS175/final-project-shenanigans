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
        self.resnet = models.resnet50(pretrained=True)
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
        self.log('train_loss', loss)
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

        # Log loss and accuracy
        wandb.log({"val_loss": loss, "val_accuracy": accuracy})

        return loss
    
    # def configure_callbacks(self):
    #     checkpoint_dir = os.path.join(os.getcwd(), 'src', 'model', 'checkpoints')
    #     os.makedirs(checkpoint_dir, exist_ok=True)
    #     checkpoint_callback = ModelCheckpoint(
    #         dirpath=checkpoint_dir,
    #         filename='resnet50-{epoch:02d}-{val_loss:.2f}',
    #         save_top_k=1,
    #         verbose=True,
    #         monitor='val_loss',
    #         mode='min'
    #     )
    #     return [checkpoint_callback]


    def configure_optimizers(self):
        # optimizer = torch.optim.SGD(self.parameters(), lr=0.01, momentum=0.9)
        optimizer = torch.optim.Adam(self.parameters())
        return optimizer
    
def main():
    wandb_logger = WandbLogger(project='resnet50', log_model=True)

    # Create an instance of the ResNetModel
    num_classes = 2  # Number of classes in bps mouse dataset
    label_mapping = {'Fe': 0, 'X-ray': 1}
    model = ResNetModel(num_classes, label_mapping)

    # Define the BPSDataModule for your dataset
    data_dir = os.path.join(root, 'data', 'processed')
    train_dir = os.path.join(data_dir, 'train')
    validation_dir = os.path.join(data_dir, 'validation')
    # train_dir = os.path.join(data_dir, 'test_train')
    # validation_dir = os.path.join(data_dir, 'test_validation')

    train_csv_file = 'meta_dose_hi_hr_4_post_exposure_train.csv'
    validation_csv_file = 'meta_dose_hi_hr_4_post_exposure_valid.csv'
    # train_csv_file = 'test_train.csv'
    # validation_csv_file = 'test_valid.csv'


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
    trainer = pl.Trainer(max_epochs=10, logger=wandb_logger)
    trainer.fit(model=model,
            train_dataloaders=data_module.train_dataloader(),
            val_dataloaders=data_module.val_dataloader())

    # Save trained model to checkpoint
    checkpoint_path = 'checkpoints/resnet50_model-hi_hr_4.pth'
    torch.save(model.state_dict(), checkpoint_path)

if __name__ == '__main__':
    main()


















# # from transformers import AutoImageProcessor, ResNetForImageClassification
# # from datasets import load_dataset
# import torch
# import torchvision.models as models
# from src.dataset.bps_datamodule import BPSDataModule
# from src.dataset.bps_dataset import BPSMouseDataset

# import pyprojroot
# from pyprojroot import here
# root = pyprojroot.find_root(pyprojroot.has_dir(".git"))

# data_dir = root / 'data'

# train_csv_file = 'meta_dose_hi_hr_4_post_exposure_train.csv'
# train_dir = data_dir / 'processed'
# validation_csv_file = 'meta_dose_hi_hr_4_post_exposure_valid.csv'
# validation_dir = data_dir / 'processed'

# # model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
# # model.eval()

# # Load the pretrained ResNet model
# model = models.resnet50(pretrained=True)

# # Get the number of input features for the last layer
# num_features = model.fc.in_features

# # Replace the last fully connected layer
# num_classes = 2  # Number of classes in bps mouse dataset (Fe, gamma)
# model.fc = torch.nn.Linear(num_features, num_classes)







# # Instantiate BPSDataModule 
# bps_datamodule = BPSMouseDataset(train_csv_file=train_csv_file,
#                                 train_dir=train_dir,
#                                 val_csv_file=validation_csv_file,
#                                 val_dir=validation_dir,
#                                 resize_dims=(64, 64),
#                                 batch_size=4,
#                                 num_workers=2)

# # Setup BPSDataModule which will instantiate the BPSMouseDataset objects
# # to be used for training and validation depending on the stage ('train' or 'val')
# bps_datamodule.setup(stage='train')

# dataset = bps_datamodule.train_dataloader()
# image = dataset[0]

# # processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
# # model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")

# # inputs = processor(image, return_tensors="pt")

# with torch.no_grad():
#     logits = model(**inputs).logits

# # model predicts one of the 1000 ImageNet classes
# predicted_label = logits.argmax(-1).item()
# print(model.config.id2label[predicted_label])