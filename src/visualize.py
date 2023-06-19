import os
import random
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

def visualize_images(title, image_dir, label_file):
    count = 0
    nrow = 1
    ncols = 5
    fig, axes = plt.subplots(nrows=nrow, ncols=ncols, figsize=(12, 4))
    image_files = os.listdir(image_dir)
    random.shuffle(image_files)
    
    label_df = pd.read_csv(label_file)

    for i in range(ncols):
        image_path = os.path.join(image_dir, image_files[i])
        image = Image.open(image_path)
        
        filename = os.path.basename(image_path)
        label_row = label_df[label_df['filename'] == filename]
        
        particle_type = label_row['particle_type'].values[0]
        
        axes[count].imshow(image)
        axes[count].axis('off')
        axes[count].text(0.5, -0.15, particle_type, transform=axes[count].transAxes, horizontalalignment='center', fontsize=10)
        count += 1

    plt.tight_layout()
    plt.savefig(f"{title}.png")
    plt.show()

image_dir = '../data/processed/train-hi_hr_4'
label_file = '../data/processed/meta_dose_hi_hr_4_post_exposure_train.csv'
title = 'High Dose Post 4 Hour Data Visualization'

visualize_images(title, image_dir, label_file)
