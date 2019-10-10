import matplotlib.pyplot as plt
from sklearn import svm
import numpy as np
import h5py
import math

def linear_kernel(xi,xj):
    return(np.dot(xi,xj.T))

def rbf_kernel(xi,xj,gamma):
    pro=xi-xj.T
    z=np.dot(pro,pro.T)
    return(math.exp(-1*gamma*z))

def signum(z):
    if(z>=0):
        return(1)
    else:
        return(0)

def predict_linear(X,W,b):
    p=[]
    for i in range(X.shape[0]):
        z=( (W[0][0]*X[i,0]) + (W[0][1]*X[i,1]) )+b
        p.append(signum(z))

    return(p)

def predict_rbf(X,alpha,sv,b,m,gamma):
    p=[]
    for i in range(X.shape[0]):
        z=0
        for j in range(m):
            z=z+( alpha[0][j]*rbf_kernel(X[i],sv[j],gamma) )
        p.append(signum(z+b))
        
    return(p)

def accuracy(X,Y):
    p=0
    t=len(X)
    for i in range(t):
        if(X[i]==Y[i]):
            p=p+1
            
    return((p*100)/t)


for i in range(4,6):
    filename="data_"+str(i)+".h5"
    f=h5py.File(filename,'r')
    l=list(f.keys())
    attri=np.array(f[l[0]])
    label=np.array(f[l[1]])

    m=label.shape[0]
    train_size=(m*80)//100
    test_size=(m*20)//100

    training_data_X=np.delete(attri,[i for i in range(test_size) ],axis=0)
    training_data_Y=np.delete(label,[i for i in range(test_size) ],axis=0)
    testing_data_X=attri[0:test_size,:]
    testing_data_Y=label[0:test_size]

    clf=svm.SVC(kernel='linear',C=1)
    clf.fit(training_data_X,training_data_Y)
    W=clf.coef_
    b=clf.intercept_
    
    training_predict=predict_linear(training_data_X,W,b)
    testing_predict=predict_linear(testing_data_X,W,b)

    print("Data Set "+str(i))
    print("Accuracy on training Data with implemented predict function (Linear SVM): "+str(accuracy(training_predict,training_data_Y)))    
    print("Accuracy on training Data with Sklearn predict function (Linear SVM): "+str(clf.score(training_data_X,training_data_Y)*100))
    print()
    print("Accuracy on testing Data with implemented predict function (Linear SVM): "+str(accuracy(testing_predict,testing_data_Y)))    
    print("Accuracy on testing Data with Sklearn predict function (Linear SVM): "+str(clf.score(testing_data_X,testing_data_Y)*100))
    print()


    g=1/(2*np.var(training_data_X))
    clf=svm.SVC(kernel='rbf',gamma='scale',C=1)
    clf.fit(training_data_X,training_data_Y)
    alpha=clf.dual_coef_
    support_vectors=clf.support_vectors_
    b=clf.intercept_
    m=sum(clf.n_support_)
    
    training_predict_rbf=predict_rbf(training_data_X,alpha,support_vectors,b,m,g)
    testing_predict_rbf=predict_rbf(testing_data_X,alpha,support_vectors,b,m,g)

    print("Data Set "+str(i))
    print("Accuracy on training Data with implemented predict function (RBF Kernel): "+str(accuracy(training_predict_rbf,training_data_Y)))    
    print("Accuracy on training Data with Sklearn predict function (RBF Kernel): "+str(clf.score(training_data_X,training_data_Y)*100))
    print()
    print("Accuracy on testing Data with implemented predict function (RBF Kernel): "+str(accuracy(testing_predict_rbf,testing_data_Y)))    
    print("Accuracy on testing Data with Sklearn predict function (RBF Kernel): "+str(clf.score(testing_data_X,testing_data_Y)*100))
    print()
