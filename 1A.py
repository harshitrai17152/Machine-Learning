import matplotlib.pyplot as plt
import numpy as np

def mean_normalization(X):
    mu=X.mean()
    sigma=X.std()
    X=(X-mu)/sigma
    
    return(X)

def scale(X):
    nom=(X-X.min(axis=0))*2
    denom=X.max(axis=0)-X.min(axis=0)
    denom[denom==0]=1
    
    return((nom/denom)-1)

def GradientDescent(X,Y,theta,alpha,iters,m):
    mse=np.matrix(np.zeros((iters,1)))
    rmse=np.matrix(np.zeros((iters,1)))

    for i in range(iters):
        temp1=(np.matmul(X,theta)-Y).T
        temp2=np.matmul(temp1,X).T
        theta=theta - (alpha/m)*temp2  
        rmse[i],mse[i]=Cost(X,Y,theta,m)
        
    return(theta,rmse,mse)

def Cost(X,Y,theta,m):
    temp=np.power( (np.matmul(X,theta)-Y).T ,2 )
    J1=( (1/(2*m))*np.sum(temp) )**0.5
    J2=(1/(2*m))*np.sum(temp)
    
    return(J1,J2)

def plot_rmse_train(rmse_5_folds_train,itr):
    lin=[i+1 for i in range(itr)]
    plt.plot(lin,rmse_5_folds_train[0],'r')
    plt.plot(lin,rmse_5_folds_train[1],'b')
    plt.plot(lin,rmse_5_folds_train[2],'g')
    plt.plot(lin,rmse_5_folds_train[3],'c')
    plt.plot(lin,rmse_5_folds_train[4],'y')
    plt.gca().legend(('Fold 1','Fold 2','Fold 3','Fold 4','Fold 5'))
    plt.xlabel('Number of iterations')
    plt.ylabel('RMSE')
    plt.title('RMSE vs Iterations (Training Set)')
    plt.show()

def plot_rmse_train_mean(fin,itr):
    lin=[i+1 for i in range(itr)]
    plt.plot(lin,fin,'r')
    plt.xlabel('Number of iterations')
    plt.ylabel('RMSE')
    plt.title('RMSE vs Iterations (Training Set)')
    plt.show()

def plot_rmse_test_mean(fin,itr):
    lin=[i+1 for i in range(itr)]
    plt.plot(lin,fin,'b')
    plt.xlabel('Number of iterations')
    plt.ylabel('RMSE')
    plt.title('RMSE vs Iterations (Testing Set)')
    plt.show()


def plot_rmse_test(rmse_5_folds_test,itr):
    lin=[i+1 for i in range(itr)]
    plt.plot(lin,rmse_5_folds_test[0],'r')
    plt.plot(lin,rmse_5_folds_test[1],'b')
    plt.plot(lin,rmse_5_folds_test[2],'g')
    plt.plot(lin,rmse_5_folds_test[3],'c')
    plt.plot(lin,rmse_5_folds_test[4],'y')
    plt.gca().legend(('Fold 1','Fold 2','Fold 3','Fold 4','Fold 5'))
    plt.xlabel('Number of iterations')
    plt.ylabel('RMSE')
    plt.title('RMSE vs Iterations (Validation Set)')
    plt.show()

def normal_equation(X,Y):
    theta=np.matrix(np.zeros((9,1)))
    b=np.matmul(X.T,Y)
    a=np.matmul(X.T,X)
    a=np.linalg.inv(a) 
    theta=np.matmul(a,b)

    return(theta)

    
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

itr=1000
alpha=0.01
k=5
gap=X.shape[0]//k

theta_5_folds_train=[]
rmse_5_folds_train=[]
mse_5_folds_train=[]

theta_5_folds_test=[]
rmse_5_folds_test=[]
mse_5_folds_test=[]

optimal_theta=normal_equation(X,Y) # Optimal parameters using normal equation
rmse_5_folds_test_optimal=[]
rmse_5_folds_train_optimal=[]

for fold in range(k):
    temp1=X
    temp2=Y
    ind=fold*gap
    training_data_X=np.delete(temp1,[i for i in range(ind,ind+gap) ],axis=0)
    training_data_Y=np.delete(temp2,[i for i in range(ind,ind+gap) ],axis=0)
    testing_data_X=X[ind:ind+gap,:]
    testing_data_Y=Y[ind:ind+gap,:]

    # For Training data set
    theta=np.matrix(np.zeros((9,1)))
    theta,rmse_train,mse_train=GradientDescent(training_data_X,training_data_Y,theta,alpha,itr,gap)

    theta_5_folds_train.append(theta)
    rmse_5_folds_train.append(rmse_train)
    mse_5_folds_train.append(mse_train)

    # For Testing data set
    theta=np.matrix(np.zeros((9,1)))
    theta,rmse_test,mse_test=GradientDescent(testing_data_X,testing_data_Y,theta,alpha,itr,gap)
    
    theta_5_folds_test.append(theta)
    rmse_5_folds_test.append(rmse_test)
    mse_5_folds_test.append(mse_test)
    
    # Calculating RMSE for optimal theta

    optimal_rmse_train=Cost(training_data_X,training_data_Y,optimal_theta,gap)[0]
    optimal_rmse_test=Cost(testing_data_X,testing_data_Y,optimal_theta,gap)[0]

    rmse_5_folds_test_optimal.append(optimal_rmse_test)
    rmse_5_folds_train_optimal.append(optimal_rmse_train)

#Part A
    
rmse_5_folds_train=np.array(rmse_5_folds_train)
rmse_5_folds_train=np.matrix(rmse_5_folds_train)
fin=rmse_5_folds_train.mean(axis=0).T
plot_rmse_train_mean(fin,itr) # RMSE vs Training set

rmse_5_folds_test=np.array(rmse_5_folds_test)
rmse_5_folds_test=np.matrix(rmse_5_folds_test)
fin=rmse_5_folds_test.mean(axis=0).T
plot_rmse_test_mean(fin,itr) # RMSE vs Validation set

#Part B
print("RMSE after getting the optimal parameters for training set:");
for i in range(k):
    print("Fold "+str(i+1)+": "+str(rmse_5_folds_train_optimal[i]))
print()
print("RMSE after getting the optimal parameters for validation set:");
for i in range(k):
    print("Fold "+str(i+1)+": "+str(rmse_5_folds_test_optimal[i]))
print()

#Part C
print("RMSE of training set from Part A: "+str(np.mean(rmse_5_folds_train)))
print("RMSE of training set from Part B: "+str(np.mean(rmse_5_folds_train_optimal)))
print()
print("RMSE of validation set from Part A: "+str(np.mean(rmse_5_folds_test)))
print("RMSE of validation set from Part B: "+str(np.mean(rmse_5_folds_test_optimal)))


# We see that after getting the optimal theta, lowest RMSE 80% set is the
# 2nd Fold of testing set
lowest_rmse_set=rmse_5_folds_test_optimal.index(min(rmse_5_folds_test_optimal))

