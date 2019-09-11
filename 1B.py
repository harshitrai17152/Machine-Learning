from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge,Lasso
import matplotlib.pyplot as plt
import numpy as np


def mean_normalization(X):
    mu=X.mean()
    sigma=X.std()
    X=(X-mu)/sigma
    
    return(X)

def GradientDescent_ridge(X,Y,theta,alpha,iters,lmbda):
    rmse=np.matrix(np.zeros((iters,1)))
    m=X.shape[0]
    
    for i in range(iters):
        temp1=(np.matmul(X,theta)-Y).T
        temp2=np.matmul(temp1,X).T
        theta= (theta*(1-((alpha*lmbda)/m))) - ((alpha/m)*temp2)  
        rmse[i]=Cost_ridge(X,Y,theta,lmbda)
        
    return(theta,rmse)

def GradientDescent_lasso(X,Y,theta,alpha,iters,lmbda):
    rmse=np.matrix(np.zeros((iters,1)))
    m=X.shape[0]
    for i in range(iters):
        temp1=(np.matmul(X,theta)-Y).T
        temp2=np.matmul(temp1,X).T
        theta= theta - ((lmbda*alpha)/m) - ((alpha/m)*temp2)
        rmse[i]=Cost_lasso(X,Y,theta,lmbda)
        
    return(theta,rmse)

def Cost_ridge(X,Y,theta,lmbda):
    m=X.shape[0]
    a=np.power( (np.matmul(X,theta)-Y).T ,2 )
    b=((lmbda/100))*np.sum( np.power(theta[1:,:],2) )
    temp=a+b
    J=(1/(2*m))*np.sum(temp)
    
    return(J)

def Cost_lasso(X,Y,theta,lmbda):
    m=X.shape[0]
    a=np.power( (np.matmul(X,theta)-Y).T ,2 )
    b=lmbda*np.sum( np.absolute(theta[1:,:]) )
    temp=a+b
    J=(1/(2*m))*np.sum(temp)
    
    return(J)

def plot_rmse_ridge(rmse_testset_ridge,itr):
    lin=[i+1 for i in range(itr)]
    plt.plot(lin,rmse_testset_ridge)
    plt.xlabel('Number of iterations');
    plt.ylabel('RMSE');
    plt.title('RMSE vs Iterations (Ridge Regression)');
    plt.show()

def plot_rmse_lasso(rmse_testset_lasso,itr):
    lin=[i+1 for i in range(itr)]
    plt.plot(lin,rmse_testset_lasso)
    plt.xlabel('Number of iterations');
    plt.ylabel('RMSE');
    plt.title('RMSE vs Iterations (Lasso Regression)');
    plt.show()

    
d=open("Dataset.data","r")

f1=d.readlines()
data=[]
X=[]
Y=[]

for i in f1:
    line=i.strip("\n").split(" ")

    if(line[0]=="M"):
        line[0]=-0.5
    elif(line[0]=="F"):
        line[0]=0
    else:
        line[0]=0.5

    for i in range(len(line)):
        line[i]=float(line[i])
    
    X.append(line[:-1])
    Y.append(line[-1])

m=len(X)
X=np.matrix(X)
Y=np.matrix(Y)
Y=Y.T

X=mean_normalization(X)
one=np.ones((m,1))
X=np.concatenate((one,X),axis=1)

itr=200
alpha=0.009
k=5
gap=X.shape[0]//k

lowest_rmse_fold=2 # Seen in Part A
ind=gap*(lowest_rmse_fold-1)
remaining_dataset_X=np.delete(X,[i for i in range(ind,ind+gap) ],axis=0)
remaining_dataset_Y=np.delete(Y,[i for i in range(ind,ind+gap) ],axis=0)
# Test Set
remaining_testset_X=X[ind:ind+gap,:]
remaining_testset_Y=Y[ind:ind+gap,:]

# Finding the appropriate Hyperparameter for Ridge regression
values=[0.02,0.001,6,1,4,20,0.5,50,100,2,1e-4,8,0.09]
parameters={'alpha':values}	
ridge_reg=Ridge(max_iter=1000)
temp=GridSearchCV(ridge_reg,parameters,cv=k,iid=True,scoring='neg_mean_squared_error')
# Searching in train+validation set the appropriate regularisation parameter
# on the basis of lowest RMSE value.

temp.fit(remaining_dataset_X,remaining_dataset_Y)
lmbda_rid=temp.best_params_['alpha']
print('Regularisation Parameter for Ridge Regression: '+str(lmbda_rid))

# Finding the appropriate Hyperparameter for Lasso regression
lsso_reg=Lasso(max_iter=1000)
temp1=GridSearchCV(lsso_reg,parameters,cv=5,iid=True,scoring='neg_mean_squared_error')
# Searching in train+validation set the appropriate regularisation parameter
# on the basis of lowest RMSE value.

temp1.fit(remaining_dataset_X,remaining_dataset_Y)
lmbda_las=temp1.best_estimator_.alpha
print('Regularisation Parameter for Lasso Regression: '+str(lmbda_las))

# For Ridge
theta_testset_ridge=np.matrix(np.zeros((9,1)))
theta_testset_ridge,rmse_testset_ridge=GradientDescent_ridge(remaining_testset_X,remaining_testset_Y,theta_testset_ridge,alpha,itr,lmbda_rid)
print("RMSE on Test set for Ridge Regression= "+str(Cost_ridge(remaining_testset_X,remaining_testset_Y,theta_testset_ridge,lmbda_rid)))
plot_rmse_ridge(rmse_testset_ridge,itr)

# For Lasso
theta_testset_lasso=np.matrix(np.zeros((9,1)))
theta_testset_lasso,rmse_testset_lasso=GradientDescent_lasso(remaining_testset_X,remaining_testset_Y,theta_testset_lasso,alpha,itr,lmbda_las)
print("RMSE on Test set for Lasso Regression= "+str(Cost_lasso(remaining_testset_X,remaining_testset_Y,theta_testset_lasso,lmbda_las)))
plot_rmse_lasso(rmse_testset_lasso,itr)

