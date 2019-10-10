import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import h5py

''' We are finding outliers on the basis of classes '''

for i in range(1,6):

    filename="data_"+str(i)+".h5"
    f=h5py.File(filename,'r')
    l=list(f.keys())
    attri=f[l[0]]
    label=f[l[1]]

    l1=[]
    l2=[]
    threshold=2
    
    if(i==3):
        l3=[]
        for j in range(label.shape[0]):
            if(label[j]==0):
                l1.append([attri[j,0],attri[j,1]])
            elif(label[j]==1):
                l2.append([attri[j,0],attri[j,1]])
            elif(label[j]==2):
                l3.append([attri[j,0],attri[j,1]])
                
        outliers1=np.abs(stats.zscore(l1))>threshold
        non_outliers1=np.abs(stats.zscore(l1))<threshold
        outliers2=np.abs(stats.zscore(l2))>threshold
        non_outliers2=np.abs(stats.zscore(l2))<threshold
        outliers3=np.abs(stats.zscore(l3))>threshold
        non_outliers3=np.abs(stats.zscore(l3))<threshold

        print("For Dataset "+str(i))
        print("Outliers in each Class:",np.sum(outliers1),np.sum(outliers2),np.sum(outliers3))

        l1=np.array(l1)
        l1=l1[(np.abs(stats.zscore(l1))<threshold).all(axis=1)]
        l2=np.array(l2)
        l2=l2[(np.abs(stats.zscore(l2))<threshold).all(axis=1)]
        l3=np.array(l3)
        l3=l3[(np.abs(stats.zscore(l3))<threshold).all(axis=1)]

        c1=plt.scatter(l1[:,0],l1[:,1],c='m')
        c2=plt.scatter(l2[:,0],l2[:,1],c='r')
        c3=plt.scatter(l3[:,0],l3[:,1],c='b')
        plt.legend((c1,c2,c3),("Class 0","Class 1","Class 2"))
        
    else:
        for j in range(label.shape[0]):
            if(label[j]==0):
                l1.append([attri[j,0],attri[j,1]])
            elif(label[j]==1):
                l2.append([attri[j,0],attri[j,1]])
            
        outliers1=np.abs(stats.zscore(l1))>threshold
        non_outliers1=np.abs(stats.zscore(l1))<threshold
        outliers2=np.abs(stats.zscore(l2))>threshold
        non_outliers2=np.abs(stats.zscore(l2))<threshold

        print("For Dataset "+str(i))
        print("Outliers in each Class:",np.sum(outliers1),np.sum(outliers2))

        l1=np.array(l1)
        l1=l1[(np.abs(stats.zscore(l1))<threshold).all(axis=1)]
        l2=np.array(l2)
        l2=l2[(np.abs(stats.zscore(l2))<threshold).all(axis=1)]
    
        c1=plt.scatter(l1[:,0],l1[:,1],c='m')
        c2=plt.scatter(l2[:,0],l2[:,1],c='r')
        plt.legend((c1,c2),("Class 0","Class 1"))  

    plt.title("Scatter Plot for Dataset "+str(i)+" Without Outliers")
    plt.show()
    print()
