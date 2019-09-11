from sklearn.linear_model import LogisticRegression
from mlxtend.data import loadlocal_mnist
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as metrics

def find(Y_test,p,num):
    TP,FP,TN,FN=0,0,0,0
    false_positive_rate=[0]*Y_test.shape[0]
    true_positive_rate=[0]*Y_test.shape[0]

    for i in range(Y_test.shape[0]): 
        if(Y_test[i]==p[i]==num):
           TP=TP+1
        if(Y_test[i]==p[i] and p[i]!=num):
           TN=TN+1
        if(p[i]!=num and Y_test[i]!=p[i]):
           FP=FP+1
        if(p[i]==num and Y_test[i]!=p[i]):
           FN=FN+1
           
        if((TN+FP)!=0):
            false_positive_rate[i]=(FP/(TN+FP))
        if((TP+FN)!=0):
            true_positive_rate[i]=(TP/(TP+FN))

    return(false_positive_rate,true_positive_rate)

    
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

# For L2 Regularisation 
# Multinomial,L2= lbfgs,newton-cg,sag,saga
log_l2=LogisticRegression(penalty='l2',solver='sag',multi_class='ovr',tol=0.1,dual=False)
log_l2.fit(X_train,Y_train.T)
accuracy_l2_train=log_l2.score(X_train,Y_train)*100
accuracy_l2_test=log_l2.score(X_test,Y_test)*100



zero,one,two,three,four,five,six,seven,eight,nine=0,0,0,0,0,0,0,0,0,0
tzero,tone,ttwo,tthree,tfour,tfive,tsix,tseven,teight,tnine=0,0,0,0,0,0,0,0,0,0
nzero,none,ntwo,nthree,nfour,nfive,nsix,nseven,neight,nnine=0,0,0,0,0,0,0,0,0,0
ttzero,ttone,tttwo,ttthree,ttfour,ttfive,ttsix,ttseven,tteight,ttnine=0,0,0,0,0,0,0,0,0,0

pred_l2_train=log_l2.predict(X_train)
pred_l2_test=log_l2.predict(X_test)

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
    
    if(pred_l2_train[i]==Y_train[i]):
        if(pred_l2_train[i]==0):
            zero=zero+1
        elif(pred_l2_train[i]==1):
            one=one+1
        elif(pred_l2_train[i]==2):
            two=two+1
        elif(pred_l2_train[i]==3):
            three=three+1
        elif(pred_l2_train[i]==4):
            four=four+1
        elif(pred_l2_train[i]==5):
            five=five+1
        elif(pred_l2_train[i]==6):
            six=six+1
        elif(pred_l2_train[i]==7):
            seven=seven+1
        elif(pred_l2_train[i]==8):
            eight=eight+1
        elif(pred_l2_train[i]==9):
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

    if(pred_l2_test[i]==Y_test[i]):
        if(pred_l2_test[i]==0):
            nzero=nzero+1
        elif(pred_l2_test[i]==1):
            none=none+1
        elif(pred_l2_test[i]==2):
            ntwo=ntwo+1
        elif(pred_l2_test[i]==3):
            nthree=nthree+1
        elif(pred_l2_test[i]==4):
            nfour=nfour+1
        elif(pred_l2_test[i]==5):
            nfive=nfive+1
        elif(pred_l2_test[i]==6):
            nsix=nsix+1
        elif(pred_l2_test[i]==7):
            nseven=nseven+1
        elif(pred_l2_test[i]==8):
            neight=neight+1
        elif(pred_l2_test[i]==9):
            nnine=nnine+1

#false_positive_rate,true_positive_rate=find(Y_test,pred_l2_test,0)
probs=log_l2.predict_proba(X_test)
lin=[i for i in range(2)]

# For 0
preds=probs[:,1]
ffpr,tpr,thresholds=metrics.roc_curve(Y_test,preds,pos_label=0)
plt.plot(tpr,ffpr,'r')
plt.plot(lin,lin,'g',linestyle='--')
plt.gca().legend(('ROC','Threshold'))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic curve for 0')
plt.show()

# For 1
preds=probs[:,2]
ffpr,tpr,thresholds=metrics.roc_curve(Y_test,preds,pos_label=2)
plt.plot(ffpr,tpr,'r')
plt.plot(lin,lin,'g',linestyle='--')
plt.gca().legend(('ROC','Threshold'))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic curve for 1')
plt.show()

# For 2
preds=probs[:,3]
ffpr,tpr,thresholds=metrics.roc_curve(Y_test,preds,pos_label=3)
plt.plot(ffpr,tpr,'r')
plt.plot(lin,lin,'g',linestyle='--')
plt.gca().legend(('ROC','Threshold'))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic curve for 2')
plt.show()

# For 3
preds=probs[:,4]
ffpr,tpr,thresholds=metrics.roc_curve(Y_test,preds,pos_label=4)
plt.plot(ffpr,tpr,'r')
plt.plot(lin,lin,'g',linestyle='--')
plt.gca().legend(('ROC','Threshold'))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic curve for 3')
plt.show()

# For 4
preds=probs[:,5]
ffpr,tpr,thresholds=metrics.roc_curve(Y_test,preds,pos_label=5)
plt.plot(ffpr,tpr,'r')
plt.plot(lin,lin,'g',linestyle='--')
plt.gca().legend(('ROC','Threshold'))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic curve for 4')
plt.show()

# For 5
preds=probs[:,6]
ffpr,tpr,thresholds=metrics.roc_curve(Y_test,preds,pos_label=7)
plt.plot(tpr,ffpr,'r')
plt.plot(lin,lin,'g',linestyle='--')
plt.gca().legend(('ROC','Threshold'))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic curve for 5')
plt.show()

# For 6
preds=probs[:,7]
ffpr,tpr,thresholds=metrics.roc_curve(Y_test,preds,pos_label=7)
plt.plot(ffpr,tpr,'r')
plt.plot(lin,lin,'g',linestyle='--')
plt.gca().legend(('ROC','Threshold'))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic curve for 6')
plt.show()

# For 7
preds=probs[:,8]
ffpr,tpr,thresholds=metrics.roc_curve(Y_test,preds,pos_label=8)
plt.plot(ffpr,tpr,'r')
plt.plot(lin,lin,'g',linestyle='--')
plt.gca().legend(('ROC','Threshold'))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic curve for 7')
plt.show()

# For 8
preds=probs[:,9]
ffpr,tpr,thresholds=metrics.roc_curve(Y_test,preds,pos_label=9)
plt.plot(ffpr,tpr,'r')
plt.plot(lin,lin,'g',linestyle='--')
plt.gca().legend(('ROC','Threshold'))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic curve for 8')
plt.show()

# For 9
preds=probs[:,9]
ffpr,tpr,thresholds=metrics.roc_curve(Y_test,preds,pos_label=2)
plt.plot(tpr,ffpr,'r')
plt.plot(lin,lin,'g',linestyle='--')
plt.gca().legend(('ROC','Threshold'))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic curve for 9')
plt.show()
