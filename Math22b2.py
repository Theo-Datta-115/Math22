#Borrowed dataloaders from https://www.kaggle.com/code/conradtsmith/symbolset-pavel-s-model

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import keras

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

import time

# at beginning of the script
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

batch_size = 256
epochCount = 25
learningRate = 0.0005

data_path = "/Users/theodatta/Downloads/mathsymbols_data/"

dataset = torchvision.datasets.ImageFolder(root=data_path, transform=transforms.ToTensor())

trainset, valset = torch.utils.data.random_split(dataset, [300000, 75974])
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
valloader = torch.utils.data.DataLoader(valset, batch_size=int(batch_size / 5), shuffle=True)

classes = ['}','{',']','[','z','y','X','w','v','u','times','theta','tan','T','sum','sqrt','sin','sigma',
               'S','rightarrow','R','q','prime','pm','pi','phi','p','o','neq','N','mu','M','lt','log','lim','leq',
               'ldots','lambda','l','k','j','int','infty','in','i','H','gt','geq','gamma','G','forward_slash','forall','f',
               'f','exists','e','div','Delta','d','cos','C','beta','b','ascii_124','alpha','A','=','9','8','7','6',
               '5','4','3','2','1','0','-',',','+','(',')','!']

model = model = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(6075, 100), 
                    nn.Linear(100, 100), 
                    nn.Linear(100,83))


optimizer = torch.optim.SGD(model.parameters(), lr=0.0005, momentum=0.9)
loss_function = nn.MSELoss()
losses = []
accuracies = []

for epoch in range(25):
    for i,(data,targets) in enumerate(trainloader):
      print(tf.shape(data))
      print(tf.shape(targets))
      predictions = model(data)
      loss = loss_function(predictions, targets)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      if i % 100 == 0:
        losses.append(loss.detach().numpy())
        accuracies.append((predictions.detach().numpy().T.round() == targets.T.numpy()).mean())

plt.subplot(211)
plt.title('Loss')
plt.plot(losses, label='train')

plt.subplot(212)
plt.title('Accuracy')
plt.plot(accuracies, label='train')
plt.show()


