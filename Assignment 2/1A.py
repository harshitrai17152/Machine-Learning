import matplotlib.pyplot as plt
import h5py

for i in range(1,6):
    filename="data_"+str(i)+".h5"
    f=h5py.File(filename,'r')
    l=list(f.keys())
    attri=f[l[0]]
    label=f[l[1]]

    for j in range(label.shape[0]):
        if(label[j]==0):
            c1=plt.scatter(attri[j,0],attri[j,1],c='m')
        elif(label[j]==1):
            c2=plt.scatter(attri[j,0],attri[j,1],c='r')
        elif(label[j]==2):
            c3=plt.scatter(attri[j,0],attri[j,1],c='b')

    plt.title("Scatter Plot for Dataset "+str(i))
    if(i==3):
        plt.legend((c1,c2,c3),("Class 0","Class 1","Class 2"))
    else:
        plt.legend((c1,c2),("Class 0","Class 1"))  
    plt.show()

    
             
        

