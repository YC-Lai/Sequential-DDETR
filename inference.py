import os
import time

import imageio
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data as data
import torchvision.transforms as transforms


def create_label_image(output, color_palette):
	"""Create a label image, given a network output (each pixel contains class index) and a color palette.
	Args:
	- output (``np.array``, dtype = np.uint8): Output image. Height x Width. Each pixel contains an integer, 
	corresponding to the class label of that pixel.
	- color_palette (``OrderedDict``): Contains (R, G, B) colors (uint8) for each class.
	"""
	
	label_image = np.zeros((output.shape[0], output.shape[1], 3), dtype=np.uint8)
	for idx, color in enumerate(color_palette):
		label_image[output==idx] = color_palette[color]
	return label_image
