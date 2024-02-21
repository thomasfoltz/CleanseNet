import os, torch, torchvision
import torch.nn as nn

from adv import AdvEMNIST
from noisy import NoisyEMNIST, gaussianNoise
from model import CondUNET, CondDiscriminator

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms

num_epochs, batch_size = 10, 32

generator, discriminator = CondUNET, CondDiscriminator

transform = transforms.Compose([transforms.ToTensor(),])
trainDataEMNIST = torchvision.datasets.EMNIST(root='./data', split='balanced', train=True, download=True, transform=transform)
noisyTrainDataEMNIST = NoisyEMNIST(trainDataEMNIST, gaussianNoise)
noisyAdvDataEMNIST = AdvEMNIST(trainDataEMNIST, generator)