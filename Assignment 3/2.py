# -*- coding: utf-8 -*-
"""question_2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1MXv1Ok-d8UovaEYtygYlykMpO3J_3SsG
"""

from torchvision import transforms
from torchvision import datasets
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
import torchvision
import numpy as np
import torch

# Commented out IPython magic to ensure Python compatibility.
#from __future__ import absolute_import, division, print_function, unicode_literals
#try:
#   %tensorflow_version 2.x
#except Exception:
#  pass
  
#import tensorflow_datasets as tfds
#import tensorflow as tf
#tf.test.gpu_device_name()

"""# CNN Class"""

class CNN(nn.Module):
  
  def __init__(self,classes):
    self.classes=classes
    super(CNN,self).__init__()
    
    self.hidden_layer=nn.Sequential(
      nn.Conv2d(1,20,kernel_size=(3,3),stride=1),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=(3,3),stride=2))
  
    self.fc_layer=nn.Linear(12*12*20,self.classes)

  def forward(self,X):
    out1=self.hidden_layer(X)
    out2=out1.reshape(out1.size(0),-1)
    out3=self.fc_layer(out2)
    return(out3)

"""# Loading the Data"""

tr=transforms.Compose([transforms.ToTensor()])
batch_size=100
epochs=10
features=728 #(28,28)
classes=10

training=datasets.FashionMNIST('~/.pytorch/F_MNIST_data/',download=True,train=True,transform=tr)
train=torch.utils.data.DataLoader(training,batch_size=batch_size,shuffle=True)

testing=datasets.FashionMNIST('~/datasets/F_MNIST/',download=True,train=False,transform=tr)
test=torch.utils.data.DataLoader(testing,batch_size=batch_size,shuffle=True)

"""# Main"""

model=CNN(classes)
criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters(),lr=0.001)

train_losses=[]
test_losses=[]

for e in range(epochs):
  
  train_loss=0
  test_loss=0
  train_acc=0
  test_acc=0
  
  # For Test Set
  for images,labels in train:
      optimizer.zero_grad()
      
      log_ps=model(images)
      prob=torch.exp(log_ps)
      top_probs,top_classes=prob.topk(1,dim=1)
      equals=labels==top_classes.view(labels.shape)
      train_acc=train_acc+equals.type(torch.FloatTensor).mean()
      
      loss=criterion(model(images),labels)
      train_loss=train_loss+loss.item()
      loss.backward()
      optimizer.step()
  else:
    # For Train Set
    with torch.no_grad():
          model.eval()
          for images,labels in test:
              log_ps=model(images)
              prob=torch.exp(log_ps)
              top_probs,top_classes=prob.topk(1,dim=1)
              equals=labels==top_classes.view(labels.shape)
              test_acc=test_acc+equals.type(torch.FloatTensor).mean()
              test_loss=(test_loss+criterion(log_ps,labels)).item()
    model.train()
        
  print("Accuracy on Train Set:",((train_acc*100)/len(train)).item())
  train_loss=train_loss/len(train)
  train_losses.append(train_loss)
  
  print("Accuracy on Test Set:",((test_acc*100)/len(test)).item())
  test_loss=test_loss/len(test)
  test_losses.append(test_loss)
  print()

"""# Plot"""

plt.plot(train_losses,label="Train losses")
plt.title("loss-per-epoch for Train Set")

plt.plot(test_losses,label="Test losses")
plt.title("loss-per-epoch for Test Set")
