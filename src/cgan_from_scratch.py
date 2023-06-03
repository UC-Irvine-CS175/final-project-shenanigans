from tensorflow import keras
from tensorflow.keras import layers

from tensorflow_docs.vis import embed
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import imageio

import os
import torch
import pyprojroot
import sys
from torchvision import transforms, utils

sys.path.append(str(pyprojroot.here()))

from src.dataset.bps_dataset import BPSMouseDataset

BATCH_SIZE = 64 # Batch size for training
NUM_CHANNELS = 1 # 1: Gray scale
NUM_CLASSES = 2 # 0: X-ray, 1: Iron
IMAGE_SIZE = 200 # 200x200
LATENT_DIM = 128 # Noise vector size

generator_in_channels = latent_dim + num_classes # 128 + 2 = 130
discriminator_in_channels = num_channels + num_classes # 1 + 2 = 3

# Create the discriminator.
discriminator = keras.Sequential(
    [
        keras.layers.InputLayer((200, 200, discriminator_in_channels)),
        layers.Conv2D(64, (3, 3), strides=(2, 2), padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2D(128, (3, 3), strides=(2, 2), padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.GlobalMaxPooling2D(),
        layers.Dense(1),
    ],
    name="discriminator",
)

# Create the generator.
generator = keras.Sequential(
    [
        keras.layers.InputLayer((generator_in_channels,)),
        # We want to generate 128 + num_classes coefficients to reshape into a
        # 7x7x(128 + num_classes) map.
        layers.Dense(7 * 7 * generator_in_channels),
        layers.LeakyReLU(alpha=0.2),
        layers.Reshape((7, 7, generator_in_channels)),
        layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2D(1, (7, 7), padding="same", activation="sigmoid"),
    ],
    name="generator",
)

