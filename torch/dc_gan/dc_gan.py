from __future__ import print_function
#%matplotlib inline
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from config import *
from IPython.display import HTML

# Set random seed for reproducibility
manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

def plot_image_set():
	# Create the dataset
	dataset = dset.ImageFolder(root=dataroot,
				   transform=transforms.Compose([
				       transforms.Resize(image_size),
				       transforms.CenterCrop(image_size),
				       transforms.ToTensor(),
				       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
				   ]))
	# Create the dataloader
	dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
						 shuffle=True, num_workers=workers)

	# Decide which device we want to run on
	device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

	# Plot some training images
	real_batch = next(iter(dataloader))
	plt.figure(figsize=(8,8))
	plt.axis("off")
	plt.title("Training Images")
	plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))


if __name__== "__main__":
	# freeze_support()
	plot_image_set()
