from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import scikitplot as skplt
from sklearn import svm
import numpy as np
import pickle
import random
#!pip install scikit-plot

#import tensorflow as tf 
#tf.test.gpu_device_name()

#from google.colab import drive
#drive.mount('/content/drive')

"""# Combining 5 training batches"""

total_training_X=[]
total_training_Y=[]

for i in range(1,6):
    #file="/content/drive/My Drive/Jupyter/data_batch_"+str(i)
    file="data_batch_"+str(i)
    with open(file,'rb') as fo:
        dict=pickle.load(fo,encoding="bytes")

    keys=list(dict.keys())
    Y=np.array(dict[keys[1]])
    X=dict[keys[2]]

    total_training_Y.append(Y)
    total_training_X.append(X.ravel())

total_training_Y=np.array(total_training_Y)
total_training_Y=total_training_Y.ravel()

total_training_X=np.array(total_training_X)
total_training_X=total_training_X.ravel()
total_training_X=total_training_X.reshape(50000,3072)

"""# 500 images of each of the 10 classes for the training set"""

limit=500
l=[[],[],[],[],[],[],[],[],[],[]]
m=[[],[],[],[],[],[],[],[],[],[]]

for num in range(10):
    for i in range(total_training_X.shape[0]):
        if(total_training_Y[i]==num and len(l[num])<limit):
            l[num].append(total_training_X[i])
            m[num].append(total_training_Y[i])
    l[num]=np.array(l[num])
    m[num]=np.array(m[num])

X=np.array(l)
X=X.ravel()
X=X.reshape(5000,3072)

Y=np.array(m)
Y=Y.ravel()

"""# Shuffling the 5k training set"""

temp=list(zip(X,Y)) 
random.shuffle(temp) 
X,Y=zip(*temp)

X=np.array(X)
Y=np.array(Y)

"""# 100 images of each of the 10 classes for the testning set"""

#file="/content/drive/My Drive/Jupyter/test_batch"
file="test_batch"
with open(file,'rb') as fo:
    dict=pickle.load(fo,encoding="bytes")

keys=list(dict.keys())
total_testing_Y=np.array(dict[keys[1]]).T
total_testing_X=np.matrix(dict[keys[2]])

limit=100
l=[[],[],[],[],[],[],[],[],[],[]]
m=[[],[],[],[],[],[],[],[],[],[]]

for num in range(10):
    for i in range(total_testing_X.shape[0]):
        if(total_testing_Y[i]==num and len(l[num])<limit):
            l[num].append(total_testing_X[i])
            m[num].append(total_testing_Y[i])
    l[num]=np.array(l[num])
    m[num]=np.array(m[num])

test_X=np.array(l)
test_X=test_X.ravel()
test_X=test_X.reshape(1000,3072)

test_Y=np.array(m)
test_Y=test_Y.ravel()

"""# Shuffling the 1k testning set"""

temp=list(zip(test_X,test_Y)) 
random.shuffle(temp) 
test_X,test_Y=zip(*temp)

test_X=np.array(test_X)
test_Y=np.array(test_Y)

"""# 30 trained SVM models"""

cm11,cm12,cm21,cm22,cm31,cm32=np.zeros((10,10)),np.zeros((10,10)),np.zeros((10,10)),np.zeros((10,10)),np.zeros((10,10)),np.zeros((10,10))
f11,f12,f21,f22,f31,f32=0,0,0,0,0,0

k=5
gap=X.shape[0]//k

for fold in range(k):
  temp1=X
  temp2=Y
  ind=fold*gap
  training_data_X=np.delete(temp1,[i for i in range(ind,ind+gap) ],axis=0)
  training_data_Y=np.delete(temp2,[i for i in range(ind,ind+gap) ],axis=0)
  #testing_data_X=X[ind:ind+gap,:]
  #testing_data_Y=Y[ind:ind+gap]

  clf11=svm.SVC(kernel='linear',decision_function_shape='ovo',probability=True)
  clf12=svm.SVC(kernel='linear',decision_function_shape='ovr',probability=True)

  clf21=svm.SVC(kernel='rbf',decision_function_shape='ovo',gamma='scale',probability=True)
  clf22=svm.SVC(kernel='rbf',decision_function_shape='ovr',gamma='scale',probability=True)
    
  clf31=svm.SVC(kernel='poly',degree=2,decision_function_shape='ovo',gamma='scale',probability=True)
  clf32=svm.SVC(kernel='poly',degree=2,decision_function_shape='ovr',gamma='scale',probability=True)

  print("FOLD: "+str(fold+1))
  clf11.fit(training_data_X,training_data_Y)
  print("Accuracy of Linear kernel for one-vs-one: "+str(clf11.score(test_X,test_Y)*100))
  clf12.fit(training_data_X,training_data_Y)
  print("Accuracy of Linear kernel for one-vs-all: "+str(clf12.score(test_X,test_Y)*100))

  clf21.fit(training_data_X,training_data_Y)
  print("Accuracy of RBF kernel for one-vs-one: "+str(clf21.score(test_X,test_Y)*100))
  clf22.fit(training_data_X,training_data_Y)
  print("Accuracy of RBF kernel for one-vs-all: "+str(clf22.score(test_X,test_Y)*100))

  clf31.fit(training_data_X,training_data_Y)
  print("Accuracy of Polynomial kernel for one-vs-one: "+str(clf31.score(test_X,test_Y)*100))
  clf32.fit(training_data_X,training_data_Y)
  print("Accuracy of Polynomial kernel for one-vs-all: "+str(clf32.score(test_X,test_Y)*100))


  p11,p12,p21,p22,p31,p32=clf11.predict(test_X),clf12.predict(test_X),clf21.predict(test_X),clf22.predict(test_X),clf31.predict(test_X),clf32.predict(test_X)

  #For Confusion Matrix
  cm11=cm11+confusion_matrix(test_Y,p11)
  cm12=cm12+confusion_matrix(test_Y,p12)
  cm21=cm21+confusion_matrix(test_Y,p21)
  cm22=cm22+confusion_matrix(test_Y,p22)
  cm31=cm31+confusion_matrix(test_Y,p31)
  cm32=cm32+confusion_matrix(test_Y,p32)

  #For F Score
  f11=f11+f1_score(test_Y,p11,average="macro")
  f12=f12+f1_score(test_Y,p12,average="macro")
  f21=f21+f1_score(test_Y,p21,average="macro")
  f22=f22+f1_score(test_Y,p22,average="macro")
  f31=f31+f1_score(test_Y,p31,average="macro")
  f32=f32+f1_score(test_Y,p32,average="macro")

"""# Confusion Matrix"""

print("Mean Confusion Matrix over 5 folds of Linear kernel:")
print(cm11/5)
print("Mean Confusion Matrix over 5 folds of RBF kernel: ")
print(cm21/5)
print("Mean Confusion Matrix over 5 folds of the quadratic polynomial kernel:")
print(cm31/5)

"""# F Score"""

print("Mean F-Score for linear kernel:")
print((f11/5)*100)
print("Mean F-Score for RBF kernel:")
print((f21/5)*100)
print("Mean F-Score for the quadratic polynomial kernel:")
print((f31/5)*100)

"""#ROC Curve"""

prob11=clf11.predict_proba(test_X)
prob12=clf12.predict_proba(test_X)
prob21=clf21.predict_proba(test_X)
prob22=clf22.predict_proba(test_X)
prob31=clf31.predict_proba(test_X)
prob32=clf32.predict_proba(test_X)

skplt.metrics.plot_roc(test_Y,prob11,title="ROC plot for linear kernel, OVO")
plt.show()
skplt.metrics.plot_roc(test_Y,prob12,title="ROC plot for linear kernel, OVR")
plt.show()

skplt.metrics.plot_roc(test_Y,prob21,title="ROC plot for RBF kernel, OVO")
plt.show()
skplt.metrics.plot_roc(test_Y,prob22,title="ROC plot for RBF kernel, OVR")
plt.show()

skplt.metrics.plot_roc(test_Y,prob31,title="ROC plot for the quadratic polynomial kernel, OVO")
plt.show()
skplt.metrics.plot_roc(test_Y,prob32,title="ROC plot for the quadratic polynomial kernel, OVR")
plt.show()

