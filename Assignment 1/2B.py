from sklearn.linear_model import LogisticRegression
from mlxtend.data import loadlocal_mnist
import numpy as np

# Loading the Training Dataset
X_train,Y_train=loadlocal_mnist(
    images_path='C:/Users/hp/Documents/Study/Semester 5/ML/Assignment 1/code/train-images-idx3-ubyte',
    labels_path='C:/Users/hp/Documents/Study/Semester 5/ML/Assignment 1/code/train-labels-idx1-ubyte')

# Loading the Testing Dataset
X_test,Y_test=loadlocal_mnist(
    images_path='C:/Users/hp/Documents/Study/Semester 5/ML/Assignment 1/code/t10k-images-idx3-ubyte',
    labels_path='C:/Users/hp/Documents/Study/Semester 5/ML/Assignment 1/code/t10k-labels-idx1-ubyte')

X_train=np.matrix(X_train)
X_test=np.matrix(X_test)

# For L1 Regularisation 
# Multinomial,L1= saga
log_l1=LogisticRegression(C=50/X_train.shape[0],multi_class='ovr',
                       penalty='l1',solver='saga',tol=0.1,dual=False)
log_l1.fit(X_train,Y_train.T)
accuracy_l1_train=log_l1.score(X_train,Y_train)*100
accuracy_l1_test=log_l1.score(X_test,Y_test)*100
print("Accuracy of Train Set in case of L1 Regularisation: "+str(accuracy_l1_train))
print("Accuracy of Test Set in case of L1 Regularisation: "+str(accuracy_l1_test))
print()

# For L2 Regularisation 
# Multinomial,L2= lbfgs,newton-cg,sag,saga
log_l2=LogisticRegression(C=50/X_train.shape[0],multi_class='ovr',
                         penalty='l2',solver='sag',tol=0.1,dual=False)
log_l2.fit(X_train,Y_train.T)
accuracy_l2_train=log_l2.score(X_train,Y_train)*100
accuracy_l2_test=log_l2.score(X_test,Y_test)*100
print("Accuracy of Train Set in case of L2 Regularisation: "+str(accuracy_l2_train))
print("Accuracy of Test Set in case of L2 Regularisation: "+str(accuracy_l2_test))
print()
print()

# One vs All for L1 Regularisation

zero,one,two,three,four,five,six,seven,eight,nine=0,0,0,0,0,0,0,0,0,0 
tzero,tone,ttwo,tthree,tfour,tfive,tsix,tseven,teight,tnine=0,0,0,0,0,0,0,0,0,0
nzero,none,ntwo,nthree,nfour,nfive,nsix,nseven,neight,nnine=0,0,0,0,0,0,0,0,0,0
ttzero,ttone,tttwo,ttthree,ttfour,ttfive,ttsix,ttseven,tteight,ttnine=0,0,0,0,0,0,0,0,0,0
pred_l1_train=log_l1.predict(X_train)
pred_l1_test=log_l1.predict(X_test)

for i in range(X_train.shape[0]):
    if(Y_train[i]==0):
        tzero=tzero+1
    elif(Y_train[i]==1):
        tone=tone+1
    elif(Y_train[i]==2):
        ttwo=ttwo+1
    elif(Y_train[i]==3):
        tthree=tthree+1
    elif(Y_train[i]==4):
        tfour=tfour+1
    elif(Y_train[i]==5):
        tfive=tfive+1
    elif(Y_train[i]==6):
        tsix=tsix+1
    elif(Y_train[i]==7):
        tseven=tseven+1
    elif(Y_train[i]==8):
        teight=teight+1
    elif(Y_train[i]==9):
        tnine=tnine+1
    
    if(pred_l1_train[i]==Y_train[i]):
        if(pred_l1_train[i]==0):
            zero=zero+1
        elif(pred_l1_train[i]==1):
            one=one+1
        elif(pred_l1_train[i]==2):
            two=two+1
        elif(pred_l1_train[i]==3):
            three=three+1
        elif(pred_l1_train[i]==4):
            four=four+1
        elif(pred_l1_train[i]==5):
            five=five+1
        elif(pred_l1_train[i]==6):
            six=six+1
        elif(pred_l1_train[i]==7):
            seven=seven+1
        elif(pred_l1_train[i]==8):
            eight=eight+1
        elif(pred_l1_train[i]==9):
            nine=nine+1

for i in range(X_test.shape[0]):
    if(Y_test[i]==0):
        ttzero=ttzero+1
    elif(Y_test[i]==1):
        ttone=ttone+1
    elif(Y_test[i]==2):
        tttwo=tttwo+1
    elif(Y_test[i]==3):
        ttthree=ttthree+1
    elif(Y_test[i]==4):
        ttfour=ttfour+1
    elif(Y_test[i]==5):
        ttfive=ttfive+1
    elif(Y_test[i]==6):
        ttsix=ttsix+1
    elif(Y_test[i]==7):
        ttseven=ttseven+1
    elif(Y_test[i]==8):
        tteight=tteight+1
    elif(Y_test[i]==9):
        ttnine=ttnine+1        

    if(pred_l1_test[i]==Y_test[i]):
        if(pred_l1_test[i]==0):
            nzero=nzero+1
        elif(pred_l1_test[i]==1):
            none=none+1
        elif(pred_l1_test[i]==2):
            ntwo=ntwo+1
        elif(pred_l1_test[i]==3):
            nthree=nthree+1
        elif(pred_l1_test[i]==4):
            nfour=nfour+1
        elif(pred_l1_test[i]==5):
            nfive=nfive+1
        elif(pred_l1_test[i]==6):
            nsix=nsix+1
        elif(pred_l1_test[i]==7):
            nseven=nseven+1
        elif(pred_l1_test[i]==8):
            neight=neight+1
        elif(pred_l1_test[i]==9):
            nnine=nnine+1

print("Accuracy of 0 in Training set for L1: "+str((100*zero)/tzero))
print("Accuracy of 0 in Testing set for L1: "+str((100*nzero)/ttzero))
print()
print("Accuracy of 1 in Training set for L1: "+str((100*one)/tone))
print("Accuracy of 1 in Testing set for L1: "+str((100*none)/ttone))
print()
print("Accuracy of 2 in Training set for L1: "+str((100*two)/ttwo))
print("Accuracy of 2 in Testing set for L1: "+str((100*ntwo)/tttwo))
print()
print("Accuracy of 3 in Training set for L1: "+str((100*three)/tthree))
print("Accuracy of 3 in Testing set for L1: "+str((100*nthree)/ttthree))
print()
print("Accuracy of 4 in Training set for L1: "+str((100*four)/tfour))
print("Accuracy of 4 in Testing set for L1: "+str((100*nfour)/ttfour))
print()
print("Accuracy of 5 in Training set for L1: "+str((100*five)/tfive))
print("Accuracy of 5 in Testing set for L1: "+str((100*nfive)/ttfive))
print()
print("Accuracy of 6 in Training set for L1: "+str((100*six)/tsix))
print("Accuracy of 6 in Testing set for L1: "+str((100*nsix)/ttsix))
print()
print("Accuracy of 7 in Training set for L1: "+str((100*seven)/tseven))
print("Accuracy of 7 in Testing set for L1: "+str((100*nseven)/ttseven))
print()
print("Accuracy of 8 in Training set for L1: "+str((100*eight)/teight))
print("Accuracy of 8 in Testing set for L1: "+str((100*neight)/tteight))
print()
print("Accuracy of 9 in Training set for L1: "+str((100*nine)/tnine))
print("Accuracy of 9 in Testing set for L1: "+str((100*nnine)/ttnine))
print()
print()

# One vs All for L2 Regularisation

zero,one,two,three,four,five,six,seven,eight,nine=0,0,0,0,0,0,0,0,0,0 
tzero,tone,ttwo,tthree,tfour,tfive,tsix,tseven,teight,tnine=0,0,0,0,0,0,0,0,0,0 #AC

nzero,none,ntwo,nthree,nfour,nfive,nsix,nseven,neight,nnine=0,0,0,0,0,0,0,0,0,0
ttzero,ttone,tttwo,ttthree,ttfour,ttfive,ttsix,ttseven,tteight,ttnine=0,0,0,0,0,0,0,0,0,0

pred_l1_train=log_l2.predict(X_train)
pred_l1_test=log_l2.predict(X_test)

for i in range(X_train.shape[0]):
    if(Y_train[i]==0):
        tzero=tzero+1
    elif(Y_train[i]==1):
        tone=tone+1
    elif(Y_train[i]==2):
        ttwo=ttwo+1
    elif(Y_train[i]==3):
        tthree=tthree+1
    elif(Y_train[i]==4):
        tfour=tfour+1
    elif(Y_train[i]==5):
        tfive=tfive+1
    elif(Y_train[i]==6):
        tsix=tsix+1
    elif(Y_train[i]==7):
        tseven=tseven+1
    elif(Y_train[i]==8):
        teight=teight+1
    elif(Y_train[i]==9):
        tnine=tnine+1
    
    if(pred_l1_train[i]==Y_train[i]):
        if(pred_l1_train[i]==0):
            zero=zero+1
        elif(pred_l1_train[i]==1):
            one=one+1
        elif(pred_l1_train[i]==2):
            two=two+1
        elif(pred_l1_train[i]==3):
            three=three+1
        elif(pred_l1_train[i]==4):
            four=four+1
        elif(pred_l1_train[i]==5):
            five=five+1
        elif(pred_l1_train[i]==6):
            six=six+1
        elif(pred_l1_train[i]==7):
            seven=seven+1
        elif(pred_l1_train[i]==8):
            eight=eight+1
        elif(pred_l1_train[i]==9):
            nine=nine+1

for i in range(X_test.shape[0]):
    if(Y_test[i]==0):
        ttzero=ttzero+1
    elif(Y_test[i]==1):
        ttone=ttone+1
    elif(Y_test[i]==2):
        tttwo=tttwo+1
    elif(Y_test[i]==3):
        ttthree=ttthree+1
    elif(Y_test[i]==4):
        ttfour=ttfour+1
    elif(Y_test[i]==5):
        ttfive=ttfive+1
    elif(Y_test[i]==6):
        ttsix=ttsix+1
    elif(Y_test[i]==7):
        ttseven=ttseven+1
    elif(Y_test[i]==8):
        tteight=tteight+1
    elif(Y_test[i]==9):
        ttnine=ttnine+1        

    if(pred_l1_test[i]==Y_test[i]):
        if(pred_l1_test[i]==0):
            nzero=nzero+1
        elif(pred_l1_test[i]==1):
            none=none+1
        elif(pred_l1_test[i]==2):
            ntwo=ntwo+1
        elif(pred_l1_test[i]==3):
            nthree=nthree+1
        elif(pred_l1_test[i]==4):
            nfour=nfour+1
        elif(pred_l1_test[i]==5):
            nfive=nfive+1
        elif(pred_l1_test[i]==6):
            nsix=nsix+1
        elif(pred_l1_test[i]==7):
            nseven=nseven+1
        elif(pred_l1_test[i]==8):
            neight=neight+1
        elif(pred_l1_test[i]==9):
            nnine=nnine+1

print("Accuracy of 0 in Training set for L2: "+str((100*zero)/tzero))
print("Accuracy of 0 in Testing set for L2: "+str((100*nzero)/ttzero))
print()
print("Accuracy of 1 in Training set for L2: "+str((100*one)/tone))
print("Accuracy of 1 in Testing set for L2: "+str((100*none)/ttone))
print()
print("Accuracy of 2 in Training set for L2: "+str((100*two)/ttwo))
print("Accuracy of 2 in Testing set for L2: "+str((100*ntwo)/tttwo))
print()
print("Accuracy of 3 in Training set for L2: "+str((100*three)/tthree))
print("Accuracy of 3 in Testing set for L2: "+str((100*nthree)/ttthree))
print()
print("Accuracy of 4 in Training set for L2: "+str((100*four)/tfour))
print("Accuracy of 4 in Testing set for L2: "+str((100*nfour)/ttfour))
print()
print("Accuracy of 5 in Training set for L2: "+str((100*five)/tfive))
print("Accuracy of 5 in Testing set for L2: "+str((100*nfive)/ttfive)) 
print()
print("Accuracy of 6 in Training set for L2: "+str((100*six)/tsix))
print("Accuracy of 6 in Testing set for L2: "+str((100*nsix)/ttsix))
print()
print("Accuracy of 7 in Training set for L2: "+str((100*seven)/tseven))
print("Accuracy of 7 in Testing set for L2: "+str((100*nseven)/ttseven))
print()
print("Accuracy of 8 in Training set for L2: "+str((100*eight)/teight))
print("Accuracy of 8 in Testing set for L2: "+str((100*neight)/tteight))
print()
print("Accuracy of 9 in Training set for L2: "+str((100*nine)/tnine))
print("Accuracy of 9 in Testing set for L2: "+str((100*nnine)/ttnine))
print()
print()
