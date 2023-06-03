import tensorflow as tf
from tensorflow.keras.models import load_model

# import torch
# import torch.nn as nn
# from torchvision import transforms, utils

import os
import sys
import pyprojroot
sys.path.append(str(pyprojroot.here()))



# Load the SavedModel
saved_model_path = 'hugging_faces'  # Replace with the path to your SavedModel directory
model = tf.keras.models.load_model(saved_model_path)



# Perform inference using the loaded model
# inputs = ...  # Replace with your input data
# outputs = model(inputs)




# from src.dataset.bps_dataset import BPSMouseDataset
# from src.dataset.augmentation import NormalizeBPS, ResizeBPS, VFlipBPS, HFlipBPS, RotateBPS, RandomCropBPS, ToTensor

# model = tf.keras.models.load_model('saved_model.pb')


# dataset = BPSMouseDataset(
#     'data.csv',
#     os.path.join(pyprojroot.here(),'tests/test_dir'),
#     transform=transforms.Compose([
#         NormalizeBPS(),
#         ResizeBPS(224, 224),
#         VFlipBPS(),
#         HFlipBPS(),
#         RotateBPS(90),
#         RandomCropBPS(200, 200),
#         ToTensor()
#     ]),
#     file_on_prem=True
#     )

# img_tensor, particle_type = dataset[0]
# print(img_tensor == torch.Tensor)
# print(img_tensor.shape, particle_type)

# # Load the .pb model
# tf.compat.v1.disable_eager_execution()
# graph_def = tf.compat.v1.GraphDef()
# with tf.io.gfile.GFile('src/hugging_faces', 'rb') as f:
#     graph_def.ParseFromString(f.read())

# # Import the graph definition into a TensorFlow session
# tf.compat.v1.import_graph_def(graph_def, name='')

# # Get the input and output nodes of the TensorFlow model
# input_node = 'input_node_name'
# output_node = 'output_node_name'

# # Define a PyTorch model with the same architecture as the TensorFlow model
# class PyTorchModel(nn.Module):
#     def __init__(self):
#         super(PyTorchModel, self).__init__()
#         # Define your model layers


#     def forward(self, x):
#         # Define the forward pass


# # Create an instance of the PyTorch model
# model = PyTorchModel()

# # Load the weights from the TensorFlow model to the PyTorch model
# with torch.no_grad():
#     for name, param in model.named_parameters():
#         param.copy_(torch.from_numpy(sess.run(name)))

# # Save the PyTorch model in the .pth format
# torch.save(model.state_dict(), 'src/hugging_faces')