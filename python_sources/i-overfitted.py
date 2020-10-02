#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[ ]:


loading=pd.read_csv("../input/train.csv").drop('id', axis=1)
loading2=pd.read_csv("../input/test.csv").drop('id',axis=1)


# In[ ]:


y=loading['target']
X=loading.drop('target', axis=1)


# In[ ]:


#Test -train splittting
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.20,random_state = 0)


# In[ ]:


#from sklearn.svm import SVC
#classifier=SVC(kernel='linear',class_weight='balanced',gamma='auto',probability=True)#This one overfit BEAUTIFULLY
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(class_weight='balanced',penalty='l1',C=0.1,random_state=0,solver='liblinear')#L1='Lasso',l2="Ridge"
classifier.fit(X_train,y_train)
#print(classifier.score(X_train,y_train))


# In[ ]:


def auc_curve():
    from sklearn import metrics
    y_pred_proba=classifier.predict_proba(X_test)[::,1]
    fpr,tpr,_=metrics.roc_curve(y_test,y_pred_proba)
    auc=metrics.roc_auc_score(y_test,y_pred_proba)
    plt.plot(fpr,tpr,label="auc="+str(auc))
    plt.legend(loc=4)
    plt.show()
    return y_pred_proba
y_pred_proba=auc_curve()


# In[ ]:


y_pred=classifier.predict(X_test)


# In[ ]:


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
cm


# """State of the model fitting on Training Set"""
# from matplotlib.colors import ListedColormap
# X_set, y_set = X_train, y_train
# X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
#                      np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
# plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
#              alpha = 0.75, cmap = ListedColormap(('red', 'green')))
# plt.xlim(X1.min(), X1.max())
# plt.ylim(X2.min(), X2.max())
# for i, j in enumerate(np.unique(y_set)):
#     plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
#                 c = ListedColormap(('yellow', 'green'))(i), label = j)
# plt.title('Model (Training Set)')
# plt.xlabel('X1')
# plt.ylabel('X2')
# plt.legend()
# plt.show()

# """State of the model fitting on Test Set"""
# from matplotlib.colors import ListedColormap
# X_set, y_set = X_test, y_test
# X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
#                      np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
# plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
#              alpha = 0.75, cmap = ListedColormap(('red', 'green')))
# plt.xlim(X1.min(), X1.max())
# plt.ylim(X2.min(), X2.max())
# for i, j in enumerate(np.unique(y_set)):
#     plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
#                 c = ListedColormap(('yellow', 'green'))(i), label = j)
# plt.title('Model (Training Set)')
# plt.xlabel('X1')
# plt.ylabel('X2')
# plt.legend()
# plt.show()

# In[ ]:


from sklearn.model_selection import cross_val_score
accuracies=cross_val_score(classifier,X_test,y_test,cv=10)
print(accuracies.mean())


# In[ ]:


sub=pd.read_csv("../input/sample_submission.csv")


# In[ ]:


sub['target']=classifier.predict_proba(loading2)[::,1]
sub.to_csv('submission.csv', index=False)

