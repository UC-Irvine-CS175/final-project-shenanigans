import os
import random
import numpy as np
import pyprojroot
root = pyprojroot.find_root(pyprojroot.has_dir(".git"))
import sys
sys.path.append(str(root))

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import pandas as pd
from matplotlib import pyplot as plt

from src.dataset.bps_dataset import BPSMouseDataset 
from src.dataset.augmentation import(
    NormalizeBPS,
    ResizeBPS,
    ToTensor
)
from src.dataset.bps_datamodule import BPSDataModule

import numpy as np

from torchvision import datasets
from torch.autograd import Variable 

import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass
from tqdm import tqdm


@dataclass
class BPSConfig:
    """ Configuration options for BPS Microscopy dataset.

    Args:
        data_dir: Path to the directory containing the image dataset. Defaults
            to the `data/processed` directory from the project root.

        train_meta_fname: Name of the training CSV file.
            Defaults to 'meta_dose_hi_hr_4_post_exposure_train.csv'

        val_meta_fname: Name of the validation CSV file.
            Defaults to 'meta_dose_hi_hr_4_post_exposure_test.csv'
        
        save_dir: Path to the directory where the model will be saved. Defaults
            to the `models/SAP_model` directory from the project root.

        batch_size: Number of images per batch. Defaults to 4.

        max_epochs: Maximum number of epochs to train the model. Defaults to 3.

        accelerator: Type of accelerator to use for training.
            Can be 'cpu', 'gpu', 'tpu', 'ipu', 'auto', or None. Defaults to 'auto'
            Pytorch Lightning will automatically select the best accelerator if
            'auto' is selected.

        acc_devices: Number of devices to use for training. Defaults to 1.
        
        device: Type of device used for training, checks for 'cuda', otherwise defaults to 'cpu'

        num_workers: Number of cpu cores dedicated to processing the data in the dataloader

        dm_stage: Set the partition of data depending to either 'train', 'val', or 'test'
                    However, our test images are not yet available.

        ckpt_dir: Directory and filename of the checkpoint to be loaded.

        save_ckpt_dir: Directory where checkpoint will be saved.

        vis_title: Title of png files to be generated.
    """
    data_dir:           str = root / 'data' / 'processed'
    train_meta_fname:   str = 'meta_dose_hi_hr_4_post_exposure_train.csv'
    val_meta_fname:     str = 'meta_dose_hi_hr_4_post_exposure_test.csv'
    save_vis_dir:       str = root / 'models' / 'dummy_vis'
    save_models_dir:    str = root / 'models' / 'baselines'
    batch_size:         int = 64
    max_epochs:         int = 3
    accelerator:        str = 'auto'
    acc_devices:        int = 1
    device:             str = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_workers:        int = 4
    dm_stage:           str = 'train'
    ckpt_dir:           str = 'C:/Users/Nitro5/cs175/final-project-shenanigans/src/model/checkpoints/testgan4_model.pth'
    save_ckpt_dir:      str = 'src/model/checkpoints/testgan5_model.pth'
    vis_title:          str = 'gan3'


channels = 1 # suggested default : 1, number of image channels (gray scale)
img_size = 200 # suggested default : 28, size of each image dimension
img_shape = (channels, img_size, img_size) # (Channels, Image Size(H), Image Size(W))

latent_dim = 500 # suggested default. dimensionality of the latent space

cuda = True if torch.cuda.is_available() else False # GPU Setting

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        
        def block(input_features, output_features, normalize=True):
            layers = [nn.Linear(input_features, output_features)]
            if normalize: # Default
                layers.append(nn.BatchNorm1d(output_features, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True)) # inplace=True : modify the input directly. It can slightly decrease the memory usage.
            return layers # return list of layers
        
        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False), # Asterisk('*') in front of block means unpacking list of layers - leave only values(layers) in list
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))), # np.prod(1, 28, 28) == 1*28*28
            nn.Tanh() # result : from -1 to 1
        )

    def forward(self, z): # z == latent vector(random input vector)
        img = self.model(z) # (64, 100) --(model)--> (64, 784)
        img = img.view(img.size(0), *img_shape) # img.size(0) == N(Batch Size), (N, C, H, W) == default --> (64, 1, 28, 28)
        return img
    

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512), # (28*28, 512)
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid() # result : from 0 to 1
        )
    
    def forward(self, img):
        img_flat = img.view(img.size(0), -1) #flatten -> from (64, 1, 28, 28) to (64, 1*28*28)
        validity = self.model(img_flat) # Discriminate -> Real? or Fake? (64, 784) -> (64, 1)
        return validity
    

adversarial_loss = torch.nn.BCELoss()

generator = Generator()
discriminator = Discriminator()

# state_dict = torch.load('C:/Users/Nitro5/.cache/torch/hub/checkpoints/CGAN_MNIST-5fda105b1f24ad665b105873e9b8dcfc838bd892bce9373ac3035d109c61ed6e.pth', map_location=torch.device('cpu'))

# modified_state_dict = {}

# # Modify the keys in the state dictionary
# for key, value in state_dict.items():
#     if key.startswith("main"):
#         new_key = key.replace("main", "model", 1)
#         modified_state_dict[new_key] = value

# generator.load_state_dict(modified_state_dict)



if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()



def main():
    # Initialize a BPSConfig object
    config = BPSConfig()
    
    # Fix random seed for reproducibility
    seed = 42
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Instantiate BPSDataModule
    bps_datamodule = BPSDataModule(train_csv_file=config.train_meta_fname,
                                   train_dir=config.data_dir,
                                   val_csv_file=config.val_meta_fname,
                                   val_dir=config.data_dir,
                                   resize_dims=(200, 200),
                                   batch_size=config.batch_size,
                                   num_workers=config.num_workers)
    
    # Using BPSDataModule's setup, define the stage name ('train' or 'val')
    bps_datamodule.setup(stage=config.dm_stage)

    # for batch_idx, (image, target) in tqdm(enumerate(bps_datamodule.train_dataloader()), desc="Running model inference"):
    #     print(image.shape)
    #     break

    # suggested default - beta parameters (decay of first order momentum of gradients)
    b1 = 0.5
    b2 = 0.999

    # suggested default - learning rate
    lr = 0.0002 

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1,b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1,b2))

    if config.ckpt_dir != '':
        ckpt = torch.load(config.ckpt_dir)
        # print(ckpt.keys())
        generator.load_state_dict(ckpt)

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    n_epochs = 200 # suggested default = 200
    for epoch in range(n_epochs):
        count = 0
        nrow=1
        ncols=5
        fig, axes = plt.subplots(nrows=nrow,ncols=ncols, figsize=(8,2))
        for i, (imgs, _) in enumerate(tqdm(bps_datamodule.train_dataloader())): # This code(enumerate) is dealt with once more in the *TEST_CODE below.
                                                        # Used 'tqdm' for showing progress 
            
            # Adversarial ground truths (For more detail, refer *Read_More below)
            valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False) # imgs.size(0) == batch_size(1 batch) == 64, *TEST_CODE
            fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False) # And Variable is for caclulate gradient. In fact, you can use it, but you don't have to. 
                                                                                    # requires_grad=False is default in tensor type. *Read_More
            
            # Configure input
            real_imgs = imgs.type(Tensor) # As mentioned, it is no longer necessary to wrap the tensor in a Variable.
        # real_imgs = Variable(imgs.type(Tensor)) # requires_grad=False, Default! It's same.
        
    # ------------
    # Train Generator
    # ------------
            optimizer_G.zero_grad()
            
            # sample noise 'z' as generator input
            z = Tensor(np.random.normal(0, 1, (imgs.shape[0],latent_dim))) # Random sampling Tensor(batch_size, latent_dim) of Gaussian distribution
            # z.shape == torch.Size([64, 100])
            
            # Generate a batch of images
            gen_imgs = generator(z)
            # gen_imgs.shape == torch.Size([64, 1, 28, 28])
            
            # Loss measures generator's ability to fool the discriminator
            g_loss = adversarial_loss(discriminator(gen_imgs), valid) # torch.nn.BCELoss() compare result(64x1) and valid(64x1, filled with 1)
            
            g_loss.backward()
            optimizer_G.step()
            
    # ------------
    # Train Discriminator
    # ------------
            optimizer_D.zero_grad()
            
            # Measure discriminator's ability to classify real from generated samples
            real_loss = adversarial_loss(discriminator(real_imgs), valid) # torch.nn.BCELoss() compare result(64x1) and valid(64x1, filled with 1)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake) # We are learning the discriminator now. So have to use detach() 
                                                                                
            d_loss = (real_loss + fake_loss) / 2
            
            d_loss.backward()# If didn't use detach() for gen_imgs, all weights of the generator will be calculated with backward(). 
            optimizer_D.step()
            
        

    # ------------
    # Real Time Visualization (While Training)
    # ------------
            
            sample_z_in_train = Tensor(np.random.normal(0, 1, (imgs.shape[0],latent_dim)))
            # z.shape == torch.Size([64, 100])
            sample_gen_imgs_in_train = generator(sample_z_in_train).detach().cpu()
            # gen_imgs.shape == torch.Size([64, 1, 28, 28])
            
            if i in [22,44,66,88, 110]: # show while batch - 200/657, 400/657, 600/657
                            plt.suptitle('EPOCH : {}'.format(epoch+1))
                            axes[count].imshow(sample_gen_imgs_in_train.permute(0,2,3,1)[count], cmap='gray')
                            axes[count].axis('off')
                            count += 1

        print(
            "[Epoch: %d/%d] [Batch: %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch+1, n_epochs, i+1, len(bps_datamodule.train_dataloader()), d_loss.item(), g_loss.item())
        )
        # print("GENERATING IMAGE...")
        # nrow=1
        # ncols=5
        # fig, axes = plt.subplots(nrows=nrow,ncols=ncols, figsize=(8,2))
        # plt.suptitle('EPOCH : {}'.format(epoch+1))
        # for ncol in range(ncols):
        #     axes[ncol].imshow(sample_gen_imgs_in_train.permute(0,2,3,1)[ncol], cmap='gray')
        #     axes[ncol].axis('off')

        plt.savefig(f"{config.vis_title}_epoch_{epoch+1}.png")
    
    torch.save(generator.state_dict(), config.save_ckpt_dir)
    print('Saved checkpoint at: ' + config.save_ckpt_dir)

if __name__ == '__main__':
    main()