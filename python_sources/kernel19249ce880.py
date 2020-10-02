#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np


# In[ ]:


DF = pd.read_csv('../input/predicting-a-pulsar-star/pulsar_stars.csv')
DF.head()


# In[ ]:


DF.info()


# In[ ]:


DF.describe()


# In[ ]:


np.unique(DF['target_class'])


# *I'm not going do any visualizations here, and there is not much preprocessing *
# <p> F1 score is a good metric when dealing with unbalanced labels</p>
# *Micro, and macro metrics are also good but i'm going to consider only f1 score *
# 
# <p> This is my first kaggle kernel so please don't expect a detailed explanation </p>

# In[ ]:


Features = DF.columns[:-1]
Label = DF.columns[-1]


# In[ ]:


X = DF[Features].values
y = DF[Label].values


# In[ ]:


X.shape


# In[ ]:


y.shape


# In[ ]:





# In[ ]:


y.shape


# In[ ]:


from sklearn.model_selection import train_test_split as tts
X_train,X_test,y_train,y_test = tts(X,y,stratify=y,random_state =1,test_size=0.2)


# In[ ]:


from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
def My_pipe(estimator,Compression = True):
    if Compression:
        return make_pipeline(StandardScaler(),
                        PCA(n_jobs = 2),
                            clone(estimator))
    return make_pipeline(StandardScaler(),
                            clone(estimator))


# In[ ]:


from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
Lr = LogisticRegression(random_state=1,max_iter=10000,solver='liblinear',C=10)
Lr_pipe = My_pipe(estimator=Lr,Compression=False)


# In[ ]:


scores = cross_val_score(estimator=Lr_pipe,X=X_train,y=y_train,cv=10,scoring='f1')


# In[ ]:


print(scores.mean(),' +/- ',scores.std())


# In[ ]:


from sklearn.svm import SVC
sv  =SVC(random_state=1,kernel='linear')
svc_pipe = My_pipe(estimator=sv,Compression=False)


# In[ ]:


scores = cross_val_score(estimator=svc_pipe,X=X_train,y=y_train,cv=10,scoring='f1')


# In[ ]:


print(scores.mean(),scores.std())


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(random_state=1,n_estimators=100)
forest_pipe = My_pipe(estimator=rfc,Compression=False)


# In[ ]:


scores_forest = cross_val_score(estimator=forest_pipe,X=X_train,y=y_train,cv=10,scoring='f1',n_jobs=-1)


# In[ ]:


print(scores_forest.mean(),' +/- ',scores_forest.std())


# In[ ]:


forest_pipe.fit(X_train,y_train)


# In[ ]:


y_train_pred = forest_pipe.predict(X_train)
y_pred_test = forest_pipe.predict(X_test)


# In[ ]:


from sklearn.metrics import f1_score
f1_train_1 = f1_score(y_pred=y_train_pred,y_true=y_train,pos_label=1)
f1_test_1 = f1_score(y_pred=y_pred_test,y_true=y_test,pos_label=1)


# In[ ]:


print('train : ' ,f1_train_1)
print('test : ',f1_test_1)


# In[ ]:


f1_train_0 = f1_score(y_pred=y_train_pred,y_true=y_train,pos_label=0)
f1_test_0 = f1_score(y_pred=y_pred_test,y_true=y_test,pos_label=0)


# In[ ]:


print('train : ' ,f1_train_0)
print('test : ',f1_test_0)


# In[ ]:


N_train_1 = X_train[y_train==1].shape[0]
n_test_1 = X_test[y_test==1].shape[0]
f1_avg_train = (N_train_1*(f1_train_1) +(X_train.shape[0]-N_train_1)*f1_train_0)/X_train.shape[0]
f1_avg_test = (n_test_1*(f1_test_1) +(X_test.shape[0]-n_test_1)*f1_test_0)/X_test.shape[0]


# In[ ]:


print('f1 avg train :',(f1_avg_train))
print('f1 avg test :',(f1_avg_test))


# So random_forest with 100 trees is the good estimator
# <p> The reason to choose Random forest is cz of its performance after doing 10 fold cross validation</p> 

# In[ ]:





# 
