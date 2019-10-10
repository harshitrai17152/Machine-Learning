import matplotlib.pyplot as plt
from sklearn.svm import SVC
import numpy as np
import h5py
from mlxtend.plotting import plot_decision_regions

def linear_kernel(xi,xj):
    return(np.dot(xi,xj.T))

def poly_kernel_2(xi,xj):
    return((np.dot(xi,xj.T)+1)**2)

def poly_kernel_3(xi,xj):
    return((np.dot(xi,xj.T)+1)**3)


for i in range(1,6):
    filename="data_"+str(i)+".h5"
    f=h5py.File(filename,'r')
    l=list(f.keys())
    attri=np.array(f[l[0]])
    label=np.array(f[l[1]])

    if(i==1 or i==4):
        svclassifier=SVC(kernel=poly_kernel_2)
    elif(i==2 or i==5):
        svclassifier=SVC(kernel=poly_kernel_3)
    elif(i==3):
        svclassifier=SVC(kernel=linear_kernel)

    svclassifier.fit(attri,label)
    plt.title("Dataset "+str(i)+" with decision boundary")
    plot_decision_regions(attri,label,svclassifier)
    plt.show()



    
