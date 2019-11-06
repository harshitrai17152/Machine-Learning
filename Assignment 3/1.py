# -*- coding: utf-8 -*-
"""question_1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1heIvRhxUJ0AzMeB0oS2Ohux3iEPaaTxy
"""

from mlxtend.data import loadlocal_mnist
import matplotlib.pyplot as plt
import numpy as np
import random

### Commented out IPython magic to ensure Python compatibility.
##from __future__ import absolute_import, division, print_function, unicode_literals
##try:
###   %tensorflow_version 2.x
##except Exception:
##  pass
##  
##import tensorflow_datasets as tfds
##import tensorflow as tf
##tf.test.gpu_device_name()
##
##from google.colab import drive
##drive.mount('/content/drive')

"""# Nueral Network Class"""

class Neural_Net:

  X_train=np.array([])
  Y_train=np.array([])
    
  batches=0
  epochs=0
  m=0
  W=[]
  bias=[]

  del_w=[]
  del_bias=[]

  y_i=[]
  y_j=[]
    
  def __init__(self,N,node,act_fn,alpha): 
    self.layers=N
    self.node=node
    self.activation=act_fn
    self.learning_rate=alpha
    
    for i in range(self.layers-1):
      b=np.zeros((1,self.node[i+1]))
      w=(0.01)*np.random.normal(0,1,(self.node[i],self.node[i+1]))
      self.bias.append(b)
      self.W.append(w)
      
  def relu(self,X):
    return(np.maximum(X,0))
  
  def relu_gradient(self,X,k,i):
    X[k[i]<=0]=0
  
    return(X)

  def sigmoid(self,X):
    return(1/(1+np.exp(-X)))

  def sigmoid_gradient(self,X):
    return(np.multiply(self.sigmoid(X),(1-self.sigmoid(X))))
 
  def linear(self,X):
    return(X)
  
  def linear_gradient(self,X):
    return(np.ones(X.shape))
  
  def tanh(self,X):
    nume=np.exp(X)-np.exp(-X)
    deno=np.exp(X)+np.exp(-X)
    q=nume/deno
    return(q)

  def tanh_gradient(self,X): 
    q=self.tanh(X)
    return(1-q**2)

  def fit(self,data,labels,batch,epochs):
    self.X_train=data
    self.Y_train=labels
    self.m=X_train.shape[0]
    self.batches=batch
    self.epochs=epochs
    self.fb_phase()
        
  # Feed Forward and feed Backward Phase
  def fb_phase(self): 
    gap=self.m//self.batches

    fin=[]
    for i in range(self.epochs):
      for fold in range(gap):
        temp1=self.X_train
        temp2=self.Y_train
        ind=fold*self.batches
        training_data_X=self.X_train[ind:ind+self.batches,:]
        training_data_Y=self.Y_train[ind:ind+self.batches]

        out=self.forward(training_data_X) # Forward phase
        
        if(fold==0):
          error=self.cost(out,training_data_Y)
          fin.append(error)
          
        self.backward(training_data_X,training_data_Y,out) # Backward Phase

        self.weight_update() #Updating weights
        self.bias_update() #Updating bias
   
   
    self.plot(fin)

  def plot(self,fin):
    itr=self.epochs
    lin=[i+1 for i in range(itr)]
    plt.plot(lin,fin,'b')
    plt.xlabel('Number of iterations')
    plt.ylabel('Cross Entropy Error')
    plt.title('Cross Entropy Error vs Iterations')
    plt.show()

  def weight_update(self):
    t=self.layers-1
    for i in range(t):
      self.W[i]=self.W[i]-(self.learning_rate*self.del_w[i])

  def bias_update(self):
    t=self.layers-1
    for i in range(t):
      self.bias[i]=self.bias[i]-(self.learning_rate*self.del_bias[i])

  def forward(self,X):
    vj=self.find(X)
    return(self.softmax(vj))

  def backward(self,X,Y,out):
    t=self.layers-1 
    self.del_w=[0]*(t)           
    self.del_bias=[0]*(t)
    yk=self.cost_gradient(out,Y)
        
    for i in range(t-1,0,-1):
      self.del_bias[i]=np.sum(yk,axis=0)
      self.del_w[i]=np.dot(self.y_j[i-1].T,yk)    
      yk=np.dot(yk,self.W[i].T)
            
      if(self.activation=='ReLU'):
        yk=self.relu_gradient(yk,self.y_j,i-1) 
      elif(self.activation=='Sigmoid'):
        yk=np.multiply(yk,self.sigmoid_gradient(self.y_i[i-1]))
      elif(self.activation=='linear'):
        yk=np.multiply(yk,self.linear_gradient(self.y_i[i-1]))
      elif(self.activation=='tanh'):
        yk=np.multiply(yk,self.tanh_gradient(self.y_i[i-1]))
        
    self.del_bias[i-1]=np.sum(yk,axis=0)
    self.del_w[i-1]=np.dot(X.T,yk)
         
  def find(self,X): 
    t=self.layers-1
    self.y_i=[0]*(t-1) # Before activation
    self.y_j=[0]*(t-1) # After activation
        
    for i in range(t-1):
      vj=np.dot(X,self.W[i])+self.bias[i]
      self.y_i[i]=vj # Before activation
      self.y_j[i]=self.activation_function(self.y_i[i]) # After activation
      X=self.y_j[i]
            
    vj=np.dot(X,self.W[i+1])+self.bias[i+1]
    return(vj)

  def activation_function(self,vj):
    if(self.activation=="ReLU"):
      phi_j=self.relu(vj)
    elif(self.activation=="Sigmoid"):
      phi_j=self.sigmoid(vj)
    elif(self.activation=="linear"):
      phi_j=self.linear(vj)
    elif(self.activation=="tanh"):
      phi_j=self.tanh(vj)

    return(phi_j)
  
  def predict(self,X):
    return(self.forward(X))

  def score(self,data,labels):
    predicted_Y=self.predict(data)
    p=0
    t=len(predicted_Y)
    for i in range(t):
      max_prob=0
      most_probable_class=0
      for j in range(10):
        if(predicted_Y[i,j]>max_prob):
          max_prob=predicted_Y[i,j]
          most_probable_class=j
           
      if(most_probable_class==labels[i]):
        p=p+1

    return((p*100)/t)
  
  def softmax(self,X):
    predict_proba=[]
    for i in range(X.shape[0]):
      data=X[i,:]
      numerator=np.exp(data)
      denominator=sum(numerator)
      prob=numerator/denominator
      predict_proba.append(prob)
    return(np.array(predict_proba))

  def cost(self,prob,Y):
    N=Y.shape[0]
    rag=[i for i in range(N)]
    nume=np.sum(np.log(prob[rag,Y]))
    loss=(-nume)/(N)
    return(loss)

  def cost_gradient(self,prob,Y):
    N=Y.shape[0]
    rag=[i for i in range(N)]
    prob[rag,Y]=prob[rag,Y]-1
    return(prob/N)

  
"""# Reading tha data"""

# Loading the Training Dataset
X_train,Y_train=loadlocal_mnist(
    images_path='C:/Users/hp/Documents/Study/Semester 5/ML/Assignment 3/code/train-images-idx3-ubyte',
    labels_path='C:/Users/hp/Documents/Study/Semester 5/ML/Assignment 3/code/train-labels-idx1-ubyte')

# Loading the Testing Dataset
X_test,Y_test=loadlocal_mnist(
    images_path='C:/Users/hp/Documents/Study/Semester 5/ML/Assignment 3/code/t10k-images-idx3-ubyte',
    labels_path='C:/Users/hp/Documents/Study/Semester 5/ML/Assignment 3/code/t10k-labels-idx1-ubyte')

"""# Main"""

def mean_normalization(X):
    mu=X.mean()
    sigma=X.std()
    X=(X-mu)/sigma
    
    return(X)

X_train=mean_normalization(X_train)
X_test=mean_normalization(X_test)

m=X_train.shape[0]
features=X_train.shape[1]
classes=len(np.unique(Y_train))
batch_size=100
#epochs=100
epochs=5

N=5
node=[features,256,128,64,classes]
act_fn=["ReLU","Sigmoid","linear","tanh"]
alpha=0.1

nn=Neural_Net(N,node,act_fn[2],alpha)   
nn.fit(X_train,Y_train,batch_size,epochs)

print(nn.score(X_test,Y_test))
