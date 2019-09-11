from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge,Lasso
import matplotlib.pyplot as plt
import numpy as np

def mean_normalization(X):
    mu=X.mean()
    sigma=X.std()
    X=(X-mu)/sigma
    
    return(X)

def GradientDescent(X,Y,theta,alpha,iters,m):
    rmse=np.matrix(np.zeros((iters,1)))
    
    for i in range(iters):
        temp1=(np.matmul(X,theta)-Y).T
        temp2=np.matmul(temp1,X).T
        theta=theta - (alpha/m)*temp2  
        rmse[i]=Cost(X,Y,theta,m)
        
    return(theta,rmse)

def Cost(X,Y,theta,m):
    temp=np.power( (np.matmul(X,theta)-Y).T ,2 )
    J=( (1/(2*m))*np.sum(temp) )

    return(J)

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
    b=(lmbda/100)*np.sum( np.power(theta[1:,:],2) )
    temp=a+b
    J=(1/(2*m))*np.sum(temp)
    
    return(J)

def Cost_lasso(X,Y,theta,lmbda):
    m=X.shape[0]
    a=np.power( (np.matmul(X,theta)-Y).T ,2 )
    b=(lmbda*50)*np.sum( np.absolute(theta[1:,:]) )
    temp=a+b
    J=(1/(2*m))*np.sum(temp)
    
    return(J)

def normal_equation(X,Y):
    theta=np.matrix(np.zeros((9,1)))
    b=np.matmul(X.T,Y)
    a=np.matmul(X.T,X)
    a=np.linalg.inv(a) 
    theta=np.matmul(a,b)

    return(theta)


d=open("data.csv","r")
f1=d.readlines()
data=[]
X=[]
Y=[]

for i in range(1,len(f1)):
    line=f1[i].strip("\n").split(",")

    for i in range(len(line)):
        line[i]=float(line[i])

    X.append(line[0])
    Y.append(line[1])

m=len(X)
X=np.matrix(X)
Y=np.matrix(Y)
X=mean_normalization(X)
Y=mean_normalization(Y)
X=np.matrix(X).T
Y=np.matrix(Y).T

ex=X.tolist()
ey=Y.tolist()
plt.scatter(ex,ey)

one=np.ones((m,1))
X=np.concatenate((one,X),axis=1)

iters=1000
alpha=0.01

# Part A
theta=np.matrix(np.zeros((2,1)))
theta,rmse=GradientDescent(X,Y,theta,alpha,iters,m)
hypotheses=np.matmul(X,theta)

plt.plot(X[:,1:],hypotheses,'k',linewidth=1.5)
plt.xlabel('Brain Weight')
plt.ylabel('Body Weight')
plt.title('Best fit line using Linear Regression (without regularisation)')
plt.show()

# Part B
lmbda_reg=4
plt.scatter(ex,ey)

theta_ridge=np.matrix(np.zeros((2,1)))
theta_ridge,rmse_ridge=GradientDescent_ridge(X,Y,theta_ridge,alpha,iters,lmbda_reg)
hypotheses_ridge=np.matmul(X,theta_ridge)

plt.plot(X[:,1:],hypotheses_ridge,'k',linewidth=1.5)
plt.xlabel('Brain Weight')
plt.ylabel('Body Weight')
plt.title('Best fit line using Linear Regression with Ridge regularisation)')
plt.show()

# Part C
lmbda_lasso=0.001
plt.scatter(ex,ey)

theta_lasso=np.matrix(np.zeros((2,1)))
theta_lasso,rmse_lasso=GradientDescent_lasso(X,Y,theta_lasso,alpha,iters,lmbda_lasso)
hypotheses_lasso=np.matmul(X,theta_lasso)

plt.plot(X[:,1:],hypotheses_lasso,'k',linewidth=1.5)
plt.xlabel('Brain Weight')
plt.ylabel('Body Weight')
plt.title('Best fit line using Linear Regression with Lasso regularisation)')
plt.show()

