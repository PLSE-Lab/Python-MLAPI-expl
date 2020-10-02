#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import random
from sklearn.preprocessing import scale


# In[ ]:


def other_scores(Y_test,prediction):
    n=prediction.shape[0]
    precision_score=0
    recall_score=0
    F1_score=0
    for i in range(n):
        actual_l=set(Y_test.rows[i])
        predict_l=set(prediction.rows[i])
        if len(predict_l)!=0:
            precision_score+=len(actual_l.intersection(predict_l))/len(predict_l)
        recall_score+=len(actual_l.intersection(predict_l))/len(actual_l)
        F1_score+=2*len(actual_l.intersection(predict_l))/(len(actual_l)+len(predict_l))
    
    return precision_score/n,recall_score/n,F1_score/n
    
        
    


# In[ ]:


def label_score(Y_test,pred):
    label_count=[0]*5
    true_pred_count=[0]*5
    for j  in range(5):
        for i in range(len(pred)):
            if Y_test[i][j]==1:
                label_count[j]+=1
            if Y_test[i][j]==1 and pred[i][j]==1:
                true_pred_count[j]+=1
    pred_sum=0
    label_sum=0
    macro_accuracy=[]
    for i in range(5):
        if label_count[i]!=0:
            macro_accuracy.append(true_pred_count[i]/label_count[i])
        else:
            macro_accuracy.append('None')
        pred_sum=pred_sum+true_pred_count[i]
        label_sum=label_sum+label_count[i]
    micro_accuracy=pred_sum/label_sum
    return macro_accuracy,micro_accuracy


# In[ ]:



X=np.load('../input/benchmark_pssm_feat_multi.npy')
len(X),len(X[1])
labels=np.load("../input/benchmark_pssm_label_multi.npy")
len(labels), len(labels[1])
X=list(X)
X=list(scale(X)) # scaling, mean at zero and unit variance
labels=list(labels)


# In[ ]:



rand_y=[]
rand_x=[]
X_dash=X.copy()
labels_dash=labels.copy()
while len(X_dash)!=0:
    Random=random.randint(0,len(X_dash)-1)
    rand_x.append(X_dash.pop(Random))
    rand_y.append(labels_dash.pop(Random))
it=len(rand_x)//5
rand_x=np.array(rand_x)
#print(rand_x.shape)
rand_y=np.array(rand_y)


# In[ ]:


from sklearn.model_selection import GridSearchCV
import numpy as np

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score,jaccard_similarity_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.svm import SVC
import random
from sklearn.svm import SVC

parameters = [
    
    {
        'classifier': [SVC(gamma='auto')],
        'classifier__kernel': ['rbf'],
        'classifier__C':[1,5,10,15,20,25,30,35,40,45,50],
        'classifier__gamma':[0.0020,0.0023,0.0026,0.0029,0.0032,0.0035,0.0038,0.0041]
        
        
    },
]


clf = GridSearchCV(BinaryRelevance(), parameters, cv=5,
                       scoring='accuracy')
clf.fit(rand_x, rand_y)
print("Best parameters set found on development set:")
print()
print(clf.best_params_)
clf


# In[ ]:


#SVM
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score,jaccard_similarity_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.svm import SVC
import random
from sklearn.svm import SVC


classifier = BinaryRelevance(
    classifier =SVC(kernel='rbf', C=25,gamma='auto'),
    require_dense = [False, True])

import numpy as np
from sklearn.metrics import accuracy_score



#print(rand_x.shape)

classifier.fit(rand_x,rand_y)
from joblib import dump, load
dump(classifier,"SVMC")


# In[ ]:


#SVM
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score,jaccard_score,jaccard_similarity_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.svm import SVC
import random
from sklearn.svm import SVC
from scipy.sparse import lil_matrix
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

classifier = BinaryRelevance(
    classifier =SVC(kernel='rbf', C=25,gamma=0.0029),
    require_dense = [False, True]
)

import numpy as np
from sklearn.metrics import accuracy_score


acc_svm = []
jac=[]
precision=[]
recall=[]
F1_score=[]
acc_svm_training=[]
macro_accuracy=[]
micro_accuracy=[]
macro_accuracy1=[]
micro_accuracy1=[]

#print(rand_y.shape)
for fold_10 in range(0,5):
    X_train = np.concatenate((rand_x[:fold_10*it],rand_x[((fold_10+1)*it):]),axis=0)
    Y_train=np.concatenate((rand_y[:fold_10*it],rand_y[((fold_10+1)*it):]),axis=0)
    X_test=rand_x[fold_10*it:(fold_10+1)*it]
    #print(X_test.shape)
    Y_test=rand_y[fold_10*it:(fold_10+1)*it]
    
    classifier.fit(X_train, Y_train)
    pred = classifier.predict(X_test)
    #print(pred.toarray())
    mac,mic=label_score(Y_test,pred.toarray())
    macro_accuracy.append(mac)
    micro_accuracy.append(mic)
    #print(pred.toarray()[0],"hi",Y_test[0])
    Y_test=lil_matrix(Y_test)
    pred=lil_matrix(pred)
    prec,rec,F1 =other_scores(Y_test,pred)
    precision.append(prec)
    recall.append(rec)
    F1_score.append(F1)
    jac.append(jaccard_score(Y_test,pred,average="samples"))
    acc_svm.append(accuracy_score( Y_test,pred))
    
    #print(classification_report(Y_test,pred))
average_macro=[0]*5
macro_accuracy=np.array(macro_accuracy)
#macro_accuracy1=np.array(macro_accuracy1)
average_macro1=[0]*5
for i in range(5):
    average_macro[i]=np.mean(macro_accuracy[:,i]),np.std(macro_accuracy[:,i])  
    


# In[ ]:


print ('\n OAA: ', np.mean(acc_svm),np.std(acc_svm))
print('\n  Acc',np.mean(jac),np.std(jac))
print("\n  Precision",np.mean(precision),np.std(precision))
print("\n  Recall",np.mean(recall),np.std(recall))
print("\n F1 score ",np.mean(F1_score),np.std(F1_score))
print("\n Average micro accuracy",np.mean(micro_accuracy),np.mean(micro_accuracy1),np.std(micro_accuracy))
print("\n Average macro accuracy",average_macro)


# In[ ]:


from sklearn.model_selection import LeaveOneOut
#SVM
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score,jaccard_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.svm import SVC
import random
from sklearn.svm import SVC
from scipy.sparse import lil_matrix

loo=LeaveOneOut()
classifier = BinaryRelevance(
    classifier =SVC(kernel='rbf', C=25,gamma=0.0029),
    require_dense = [False, True]
)

import numpy as np
from sklearn.metrics import accuracy_score
acc_svm = []
jac=[]
rand_y=[]
rand_x=[]
X_dash=X.copy()
labels_dash=labels.copy()
    while len(X_dash)!=0:
        Random=random.randint(0,len(X_dash)-1)
        rand_x.append(X_dash.pop(Random))
        rand_y.append(labels_dash.pop(Random))
it=len(rand_x)//5
rand_x=np.array(rand_x)
#print(rand_x.shape)
rand_y=np.array(rand_y)
acc_svm_training=[]
#print(rand_y.shape)
precision=[]
recall=[]
F1_score=[]
label1_acc=0
label1=0
label2=0
label3=0
label2_acc=0
label3_acc=0
label_jac=0
true_pred_count=[0]*5
label_count=[0]*5
label_5=0
label_4=0
label_2=0
label_1=0
label_3=0
for train, test in loo.split(rand_x, rand_y):
    X_train, X_test = rand_x[train], rand_x[test]
    Y_train, Y_test = rand_y[train], rand_y[test]
    
    classifier.fit(X_train, Y_train)
    pred = classifier.predict(X_test)
    #print(pred.toarray())
    pred1=pred.toarray()
#     print(Y_test,pred1)
    for j  in range(5):
        if Y_test[0][j]==1:
            label_count[j]+=1
        if Y_test[0][j]==1 and pred1[0][j]==1:
            true_pred_count[j]+=1



    pred=lil_matrix(pred.toarray())
    Y_test=lil_matrix(Y_test)
    prec,rec,F1=other_scores(Y_test,pred)
    precision.append(prec)
    recall.append(rec)
    F1_score.append(F1)
    a=set(Y_test.rows[0])
    b=set(pred.rows[0])
    if len(a)==1:
        label1+=1
        if a.intersection(b)==a.union(b):
            label1_acc+=1
    if len(a)==2:
        label2+=1
        label_jac+=len(a.intersection(b))/len(a.union(b))
        if a.intersection(b)==a.union(b):
            label2_acc+=1
            
    if len(a)==3:
        label3+=1
        if a.intersection(b)==a.union(b):
            label3_acc+=1
            
    
    jac.append(jaccard_score(Y_test,pred,average="samples"))
    
    acc_svm.append(accuracy_score( Y_test,pred))
    #print(classification_report(Y_test,pred))
pred_sum=0
label_sum=0
macro_accuracy=[]
micro_accuracy=0
for i in range(5):
    if label_count[i]!=0:
        macro_accuracy.append(true_pred_count[i]/label_count[i])
    else:
        macro_accuracy.append('None')
    pred_sum=pred_sum+true_pred_count[i]
    label_sum=label_sum+label_count[i]
micro_accuracy=pred_sum/label_sum
    


# In[ ]:



print ('\n OAA: ', np.mean(acc_svm),np.std(acc_svm))
print('\n  Acc',np.mean(jac),np.std(jac))
print("\n  Precision",np.mean(precision),np.std(precision))
print("\n  Recall",np.mean(recall),np.std(recall))
print("\n F1 score ",np.mean(F1_score),np.std(F1_score))
print("\n Average micro accuracy",np.mean(micro_accuracy),np.std(micro_accuracy))
print("\n Average macro accuracy",macro_accuracy)
print("grand mean", 1/5*(np.mean(acc_svm)+np.mean(jac)+np.mean(precision)+np.mean(recall)+np.mean(F1_score)))


# In[ ]:


print("Label -1 accuracy",label1_acc/label1)
print("Label -2 accuracy",label2_acc/label2,label_jac/label2)
print("Label -3 accuracy",label3_acc/label3)
label1,label2,label3
label1_acc,label2_acc,label3_acc


# In[ ]:


X_test=np.load('../input/novel_pssm_feat_multi.npy')
Y_test=np.load('../input/novel_pssm_label_multi.npy')
X_test=scale(X_test)
b=np.zeros(122*5).reshape(122,5)
b[:,:-1]=Y_test
Y_test=b
Y_test.shape


# In[ ]:


from joblib import dump, load
classifier=load('SVMC')
pred_novel=classifier.predict(X_test)
macro_accuracy,micro_accuracy=label_score(Y_test,pred_novel.toarray())
OAA=accuracy_score(Y_test,pred_novel)
Y_test=lil_matrix(Y_test)
pred_novel=lil_matrix(pred_novel.toarray())
precision,recall,F1=other_scores(Y_test,pred_novel)

print("OAA ",OAA)
print("Accuracy",jaccard_score(Y_test,pred_novel,average="samples"))  
print("precision",precision)
print("recall",recall)
print("F1 score",F1)
print("Micro accuracy",micro_accuracy)
print("Macro accuarcy",macro_accuracy)


# In[ ]:


label1_acc=0
label1=0
label2=0
label3=0
label2_acc=0
label3_acc=0
label_jac=0
for i in range(122):
    a=set(Y_test.rows[i])
    b=set(pred_novel.rows[i])
    if len(a)==1:
        label1+=1
        if a.intersection(b)==a.union(b):
            label1_acc+=1
    if len(a)==2:
        label2+=1
        label_jac+=len(a.intersection(b))/len(a.union(b))
        if a.intersection(b)==a.union(b):
            label2_acc+=1
            


# In[ ]:


print("Label -1 accuracy",label1_acc/label1)
print("Label -2 accuracy",label2_acc/label2,label_jac/label2)

print(label1,label2)
print(label1_acc,label2_acc)

