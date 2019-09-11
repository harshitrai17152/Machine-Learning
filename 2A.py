from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge,Lasso,RidgeCV,MultiTaskLassoCV
from sklearn import linear_model
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math

def mean_normalization(X):
    mu=X.mean()
    sigma=X.std()
    X=(X-mu)/sigma
    
    return(X)

def sigmoid(z):
    return(1/(1+np.exp(-z)))

def Cost(X,Y,theta):
    m=X.shape[0]
    
    hypotheses=sigmoid(X*theta)
    a=np.matmul(Y.T,(np.log(hypotheses)))
    b=np.matmul((1-Y).T,(np.log(1-hypotheses)))
    
    J=(-1/m)*(a+b)
    gradient=(1/m)*(np.matmul((hypotheses-Y).T,X)).T
    
    return(J,gradient)

def Cost_ridge(X,Y,theta,lmbda):
    m=X.shape[0]
    
    hypotheses=sigmoid(X*theta)
    a=np.matmul(Y.T,(np.log(hypotheses)))
    b=np.matmul((1-Y).T,(np.log(1-hypotheses)))

    reg_term=(lmbda/(2*m))*np.sum( np.power(theta[1:,:],2) )
    J=((-1/m)*(a+b))+reg_term
    ree=(lmbda/m)*np.sum(theta[1:,:])
    gradient=(1/(2*m))*(np.matmul((hypotheses-Y).T,X)).T+ree
    
    return(J,gradient)

def GradientDescent_ridge(X,Y,theta,alpha,itrs,lmbda):
    m=X.shape[0]
    cost_=np.matrix(np.zeros((itrs,1)))
     
    for i in range(itrs):
        cost,gradient=Cost_ridge(X,Y,theta,lmbda)
        theta=theta-(gradient*alpha)
        cost_[i]=cost
        
    return(theta,cost_)

def Cost_lasso(X,Y,theta,lmbda):
    m=X.shape[0]
    
    hypotheses=sigmoid(X*theta)
    a=np.matmul(Y.T,(np.log(hypotheses)))
    b=np.matmul((1-Y).T,(np.log(1-hypotheses)))

    reg_term=(lmbda/m)*np.sum(theta[1:,:])
    J=((-1/m)*(a+b))+reg_term
    ree=(lmbda/m)
    gradient=(1/(2*m))*(np.matmul((hypotheses-Y).T,X)).T+ree
    
    return(J,gradient)

def GradientDescent(X,Y,theta,alpha,itrs):
    m=X.shape[0]
    cost_=np.matrix(np.zeros((itrs,1)))
    
    for i in range(itrs):
        cost,gradient=Cost(X,Y,theta)
        theta=theta-(gradient*alpha)
        cost_[i]=cost
        
    return(theta,cost_)

def GradientDescent_lasso(X,Y,theta,alpha,itrs,lmbda):
    m=X.shape[0]
    cost_=np.matrix(np.zeros((itrs,1)))
    
    for i in range(itrs):
        cost,gradient=Cost_lasso(X,Y,theta,lmbda)
        theta=theta-(gradient*alpha)
        cost_[i]=cost
        
    return(theta,cost_)

def predict(X,theta,Y):
    z=sigmoid(X*theta)
    p=0
    total=X.shape[0]
    for i in range(total):
        if(z[i]>=0.5 and Y[i]==1):
            p=p+1
        elif(z[i]<0.5 and Y[i]==0):
            p=p+1
    
    return((100*p)/total)

    
data=pd.read_csv("train.csv")
data=np.matrix(data)
X=[]
Y=[]

data2=pd.read_csv("test.csv")
data2=np.matrix(data2)
X_test=[]
Y_test=[]

for i in range(data.shape[0]):
    if(data[i,14]==" <=50K"):
        data[i,14]=0
    else:
        data[i,14]=1

for i in range(data2.shape[0]):
    if(data2[i,14]==" <=50K."):
        data2[i,14]=0
    else:
        data2[i,14]=1

X_test=np.matrix(X_test)
Y_test=np.matrix(Y_test) 
X_test=data2[:,:14]
Y_test=data2[:,-1]

X=np.matrix(X)
Y=np.matrix(Y) 
X=data[:,:14]
Y=data[:,-1]

# Training set
##we are removing 3 columns from the given dataset
X=np.delete(X,[2],axis=1)
X=np.delete(X,[9],axis=1)
X=np.delete(X,[11],axis=1)

p=np.array(X[:,8])
dum=pd.get_dummies(p.flatten())
dum=np.matrix(dum)
X=np.concatenate((X,dum),axis=1)
X=np.delete(X,[8],axis=1)

p=np.array(X[:,1])
dum=pd.get_dummies(p.flatten())
dum=np.matrix(dum)
X=np.concatenate((X,dum),axis=1)
X=np.delete(X,[1],axis=1)

p=np.array(X[:,1])
dum=pd.get_dummies(p.flatten())
dum=np.matrix(dum)
X=np.concatenate((X,dum),axis=1)
X=np.delete(X,[1],axis=1)

p=np.array(X[:,2])
dum=pd.get_dummies(p.flatten())
dum=np.matrix(dum)
X=np.concatenate((X,dum),axis=1)
X=np.delete(X,[2],axis=1)

p=np.array(X[:,2])
dum=pd.get_dummies(p.flatten())
dum=np.matrix(dum)
X=np.concatenate((X,dum),axis=1)
X=np.delete(X,[2],axis=1)

p=np.array(X[:,2])
dum=pd.get_dummies(p.flatten())
dum=np.matrix(dum)
X=np.concatenate((X,dum),axis=1)
X=np.delete(X,[2],axis=1)

p=np.array(X[:,2])
dum=pd.get_dummies(p.flatten())
dum=np.matrix(dum)
X=np.concatenate((X,dum),axis=1)  
X=np.delete(X,[2],axis=1)

X=X.astype('int32')
Y=Y.astype('int32')

features=X.shape[1]+1
X=mean_normalization(X)

m=X.shape[0] 
one=np.ones((m,1))
X=np.concatenate((one,X),axis=1)

# Test Set
##we are removing 3 columns from the given dataset
X_test=np.delete(X_test,[2],axis=1)
X_test=np.delete(X_test,[9],axis=1)
X_test=np.delete(X_test,[11],axis=1)

p=np.array(X_test[:,8])
dum=pd.get_dummies(p.flatten())
dum=np.matrix(dum)
X_test=np.concatenate((X_test,dum),axis=1)
X_test=np.delete(X_test,[8],axis=1)

p=np.array(X_test[:,1])
dum=pd.get_dummies(p.flatten())
dum=np.matrix(dum)
X_test=np.concatenate((X_test,dum),axis=1)
X_test=np.delete(X_test,[1],axis=1)

p=np.array(X_test[:,1])
dum=pd.get_dummies(p.flatten())
dum=np.matrix(dum)
X_test=np.concatenate((X_test,dum),axis=1)
X_test=np.delete(X_test,[1],axis=1)

p=np.array(X_test[:,2])
dum=pd.get_dummies(p.flatten())
dum=np.matrix(dum)
X_test=np.concatenate((X_test,dum),axis=1)
X_test=np.delete(X_test,[2],axis=1)

p=np.array(X_test[:,2])
dum=pd.get_dummies(p.flatten())
dum=np.matrix(dum)
X_test=np.concatenate((X_test,dum),axis=1)
X_test=np.delete(X_test,[2],axis=1)

p=np.array(X_test[:,2])
dum=pd.get_dummies(p.flatten())
dum=np.matrix(dum)
X_test=np.concatenate((X_test,dum),axis=1)
X_test=np.delete(X_test,[2],axis=1)

p=np.array(X_test[:,2])
dum=pd.get_dummies(p.flatten())
dum=np.matrix(dum)
X_test=np.concatenate((X_test,dum),axis=1)  
X_test=np.delete(X_test,[2],axis=1)

X_test=X_test.astype('int32')
Y_test=Y_test.astype('int32')

features=X_test.shape[1]+1
X_test=mean_normalization(X_test)

m=X_test.shape[0] 
one=np.ones((m,1))
X_test=np.concatenate((one,X_test),axis=1)

# Validation set

temp1=X
temp2=Y
gap=int(X.shape[0]*0.2)
training_data_X=np.delete(temp1,[i for i in range(gap) ],axis=0)
training_data_Y=np.delete(temp2,[i for i in range(gap) ],axis=0)
validation_data_X=X[:gap,:]
validation_data_Y=Y[:gap,:]

X=training_data_X
Y=training_data_Y

# Main Logic

iters=500
alpha=0.1

# For L1

ok=MultiTaskLassoCV(cv=5)
lcv=ok.fit(X,Y)
lasso_lmbda=ok.alpha_
print("Hyperparameter for Lasso Regularisation: "+str(lasso_lmbda))

       # Train
theta2=np.matrix(np.zeros((features,1)))
theta2,cost_las=GradientDescent_lasso(X,Y,theta2,alpha,iters,lasso_lmbda)
lin=[i+1 for i in range(iters)]
plt.plot(lin,cost_las)
plt.xlabel('Number of iterations')
plt.ylabel('Error')
plt.title('Error vs Iterations for Lasso regression L1 on Training set')
plt.show()
print("Accuracy with Lasso Regularisation on Tain set: "+str(predict(X,theta2,Y)))

       # Validation
#theta=theta2
theta=np.matrix(np.zeros((features,1)))
theta,cost_las_val=GradientDescent_lasso(validation_data_X,validation_data_Y,theta,alpha,iters,lasso_lmbda)
lin=[i+1 for i in range(iters)]
plt.plot(lin,cost_las_val)
plt.xlabel('Number of iterations')
plt.ylabel('Error')
plt.title('Error vs Iterations for Lasso regression L1 on Validation set')
plt.show()
print("Accuracy with Lasso Regularisation on Validation set: "+str(predict(validation_data_X,theta,validation_data_Y)))

        # Testing
#theta=theta2
theta=np.matrix(np.zeros((features,1)))
theta,cost_las_test=GradientDescent_lasso(X_test,Y_test,theta,alpha,iters,lasso_lmbda)
lin=[i+1 for i in range(iters)]
plt.plot(lin,cost_las_test)
plt.xlabel('Number of iterations')
plt.ylabel('Error')
plt.title('Error vs Iterations for Lasso regression L1 on Testing set')
plt.show()
print("Accuracy with Lasso Regularisation on Test set: "+str(predict(X_test,theta,Y_test)))




# For L2

ok=RidgeCV(cv=5)
lcv=ok.fit(X,Y)
ridge_lmbda=ok.alpha_
print("Hyperparameter for Ridge Regularisation: "+str(ridge_lmbda))

       # Train
theta2=np.matrix(np.zeros((features,1)))
accu=np.matrix(np.zeros((features,1)))
theta2,cost_rid=GradientDescent_ridge(X,Y,theta2,alpha,iters,ridge_lmbda)
lin=[i+1 for i in range(iters)]
plt.plot(lin,cost_rid)
plt.xlabel('Number of iterations')
plt.ylabel('Error')
plt.title('Error vs Iterations for Ridge regression L2 on Training set')
plt.show()
print("Accuracy with Ridge Regularisation on Tain set: "+str(predict(X,theta2,Y)))

       # Validation
#theta=theta2
theta=np.matrix(np.zeros((features,1)))
theta,cost_rid_val=GradientDescent_ridge(validation_data_X,validation_data_Y,theta,alpha,iters,ridge_lmbda)
lin=[i+1 for i in range(iters)]
plt.plot(lin,cost_rid_val)
plt.xlabel('Number of iterations')
plt.ylabel('Error')
plt.title('Error vs Iterations for Ridge regression L2 on Validation set')
plt.show()
print("Accuracy with Ridge Regularisation on Validation set: "+str(predict(validation_data_X,theta,validation_data_Y)))

        # Testing
#theta=theta2
theta=np.matrix(np.zeros((features,1)))
theta,cost_rid_test=GradientDescent_ridge(X_test,Y_test,theta,alpha,iters,lasso_lmbda)
lin=[i+1 for i in range(iters)]
plt.plot(lin,cost_rid_test)
plt.xlabel('Number of iterations')
plt.ylabel('Error')
plt.title('Error vs Iterations for Lasso regression L1 on Testing set')
plt.show()
print("Accuracy with Lasso Regularisation on Test set: "+str(predict(X_test,theta,Y_test)))
print("Accuracy without Regularisation: "+str(predict(X,theta,Y)))
