import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim

transform = transforms.ToTensor() # Definindo a conversão de imagem para tensor

trainset = datasets.MNIST('./MNIST_data/', download=True, train=True, transform=transform) # Carrega parte de treino do dataset
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True) # Cria um buffer para pegar os dados por partes

valset = datasets.MNIST('./MNIST_data/', download=True, train=False, transform=transform) # Carrega a parte da validação 
valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)# Cria um buffer para pegar os dados

dataiter = iter(trainloader)
imagens, etiquetas = dataiter._next_data()
plt.imshow(imagens[0].numpy().squeeze(), cmap='gray_r')
plt.show()