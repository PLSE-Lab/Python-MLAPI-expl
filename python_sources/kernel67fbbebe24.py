#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd


# In[ ]:


df_train=pd.read_csv('../input/digit-recognizer/train.csv')
df_train.head()


# In[ ]:


x_train=df_train.drop('label',axis=1)
x_train.head()


# In[ ]:


y_train=df_train['label']
y_train.head()


# In[ ]:


from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA


# ## 1 logistic regression

# In[ ]:


from sklearn.linear_model import LogisticRegression
lgr=LogisticRegression()
pipe=make_pipeline(PCA(n_components=0.90),StandardScaler(),lgr)
score=np.mean(cross_val_score(pipe,x_train,y_train,cv=2,scoring='accuracy'))
print("lgr")
print(score)


# ## 2 SGD

# In[ ]:


from sklearn.linear_model import SGDClassifier
sgd=SGDClassifier()
pipe=make_pipeline(PCA(n_components=0.90),StandardScaler(),sgd)
score=np.mean(cross_val_score(pipe,x_train,y_train,cv=2,scoring='accuracy'))
print("sgd")
print(score)


# ## 3 LINEAR svc

# In[ ]:


from sklearn.svm import LinearSVC
lsvc=LinearSVC(C=5)
pipe=make_pipeline(PCA(n_components=0.80),StandardScaler(),lsvc)
score=np.mean(cross_val_score(pipe,x_train,y_train,cv=2,scoring='accuracy',n_jobs=-1))
print("lsvc")
print(score)


# ## 4 SVC

# In[ ]:


from sklearn.svm import SVC
svc=SVC()
pipe=make_pipeline(PCA(n_components=0.90),StandardScaler(),svc)
score=np.mean(cross_val_score(pipe,x_train,y_train,cv=2,scoring='accuracy'))
print("svc")
print(score)


# ## 5 DT

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()
pipe=make_pipeline(PCA(n_components=0.90),StandardScaler(),dt)
score=np.mean(cross_val_score(pipe,x_train,y_train,cv=2,scoring='accuracy'))
print("dt")
print(score)


# ## 6 KNN(grid_search)

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
#knn=KNeighborsClassifier(n_neighbors=5)
for i in range(10,31):
    pipe=make_pipeline(PCA(n_components=0.90),StandardScaler(),KNeighborsClassifier(n_neighbors=i))
    score=np.mean(cross_val_score(pipe,x_train,y_train,cv=2,scoring='accuracy'))
    print(i)
    print(score)
#param={'kneighborsclassifier__n_neighbors':}
#grid=GridSearchCV(pipe,param_grid=param,cv=2,scoring='accuracy')
#grid.fit(x_train,y_train)
#print(grid_best_params_)
#print(grid_best_score_)


# ## 7 RF

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()
pipe=make_pipeline(PCA(n_components=0.90),StandardScaler(),rf)
score=np.mean(cross_val_score(pipe,x_train,y_train,cv=2))
print("rf")
print(score)


# ## 8  ada boost

# In[ ]:


from sklearn.ensemble import AdaBoostClassifier
adb=AdaBoostClassifier()
pipe=make_pipeline(PCA(n_components=0.90),StandardScaler(),adb)
score=np.mean(cross_val_score(pipe,x_train,y_train,cv=2))
print("adb")
print(score)


# ## 9 bagging

# In[ ]:


from sklearn.ensemble import BaggingClassifier
bc=BaggingClassifier()
pipe=make_pipeline(PCA(n_components=0.90),StandardScaler(),bc)
score=np.mean(cross_val_score(pipe,x_train,y_train,cv=2))
print("bc")
print(score)


# ## 10 gradient boosting

# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier
gb=GradientBoostingClassifier()
pipe=make_pipeline(PCA(n_components=0.90),StandardScaler(),gb)
score=np.mean(cross_val_score(pipe,x_train,y_train,cv=2))
print("gb")
print(score)


# # parameter tuning

# #### among all SVC(support vector classifier) has highest score so let's tune it's parameter

# ## with MinMaxScaler

# In[ ]:


from sklearn.preprocessing import MinMaxScaler


# In[ ]:


pipe=make_pipeline(PCA(n_components=0.90),MinMaxScaler(),svc)
score=np.mean(cross_val_score(pipe,x_train,y_train,cv=2,scoring='accuracy'))
print("svc")
print(score)


# ## without MinMasScaler

# In[ ]:


pipe=make_pipeline(PCA(n_components=0.90),svc)
score=np.mean(cross_val_score(pipe,x_train,y_train,cv=2,scoring='accuracy'))
print("svc")
print(score)


# ## with 98% explained varience(PCA)

# In[ ]:


from sklearn.svm import SVC
svc=SVC()
pipe=make_pipeline(PCA(n_components=0.98),StandardScaler(),svc)
score=np.mean(cross_val_score(pipe,x_train,y_train,cv=2,scoring='accuracy'))
print("svc")
print(score)


# ## with 95% explained varience(PCA)

# In[ ]:


pipe=make_pipeline(PCA(n_components=0.95),StandardScaler(),svc)
score=np.mean(cross_val_score(pipe,x_train,y_train,cv=2,n_jobs=-1,scoring='accuracy'))
print("svc")
print(score)


# ## with 85% explained varience(PCA)

# In[ ]:


pipe=make_pipeline(PCA(n_components=0.85),StandardScaler(),svc)
score=np.mean(cross_val_score(pipe,x_train,y_train,cv=2,n_jobs=-1,scoring='accuracy'))
print("svc")
print(score)


# ## with 80% explained varience(PCA)

# In[ ]:


pipe=make_pipeline(PCA(n_components=0.80),StandardScaler(),svc)
score=np.mean(cross_val_score(pipe,x_train,y_train,cv=2,n_jobs=-1,scoring='accuracy'))
print("svc")
print(score)


# In[ ]:


pipe=make_pipeline(PCA(n_components=0.75),StandardScaler(),svc)
score=np.mean(cross_val_score(pipe,x_train,y_train,cv=2,n_jobs=-1,scoring='accuracy'))
print("svc")
print(score)


# In[ ]:


pipe=make_pipeline(PCA(n_components=0.70),StandardScaler(),svc)
score=np.mean(cross_val_score(pipe,x_train,y_train,cv=2,n_jobs=-1,scoring='accuracy'))
print("svc")
print(score)


# In[ ]:


pipe=make_pipeline(PCA(n_components=0.65),StandardScaler(),svc)
score=np.mean(cross_val_score(pipe,x_train,y_train,cv=2,n_jobs=-1,scoring='accuracy'))
print("svc")
print(score)


# In[ ]:


pipe=make_pipeline(PCA(n_components=0.60),StandardScaler(),svc)
score=np.mean(cross_val_score(pipe,x_train,y_train,cv=2,n_jobs=-1,scoring='accuracy'))
print("svc")
print(score)


# In[ ]:


pipe=make_pipeline(PCA(n_components=0.50),StandardScaler(),svc)
score=np.mean(cross_val_score(pipe,x_train,y_train,cv=2,n_jobs=-1,scoring='accuracy'))
print("svc")
print(score)


# In[ ]:


pipe=make_pipeline(PCA(n_components=0.10),StandardScaler(),svc)
score=np.mean(cross_val_score(pipe,x_train,y_train,cv=2,n_jobs=-1,scoring='accuracy'))
print("svc")
print(score)


# In[ ]:


pipe=make_pipeline(PCA(n_components=0.01),StandardScaler(),svc)
score=np.mean(cross_val_score(pipe,x_train,y_train,cv=2,n_jobs=-1,scoring='accuracy'))
print("svc")
print(score)


# In[ ]:


pipe=make_pipeline(PCA(n_components=2),StandardScaler(),svc)
score=np.mean(cross_val_score(pipe,x_train,y_train,cv=2,n_jobs=-1,scoring='accuracy'))
print("svc")
print(score)


# ## best score is at 80% explaind variance(PCA)

# In[ ]:


pipe=make_pipeline(PCA(n_components=0.80),StandardScaler(),svc)
param={ 'svc__C':[5,6,7,8,9,10]}
grid=GridSearchCV(pipe,param_grid=param,cv=2,scoring='accuracy',n_jobs=-1)
grid.fit(x_train,y_train)
#grid.best_estimator_
#grid.best_params_
grid.best_score_
#score=np.mean(cross_val_score(pipe,x_train,y_train,cv=2,n_jobs=-1,scoring='accuracy'))
#print("svc")
#print(score)


# In[ ]:


grid.best_params_


# In[ ]:


df_test=pd.read_csv('../input/digit-recognizer/test.csv
                    ')
df_test.head()


# In[ ]:


df_test.shape


# In[ ]:


pipe=make_pipeline(PCA(n_components=0.80),StandardScaler(),SVC(C=5,degree=1))


# In[ ]:


pipe.fit(x_train,y_train)


# In[ ]:


final=pipe.predict(df_test)


# In[ ]:


final


# In[ ]:


final_df=pd.DataFrame({'ImageId':np.arange(1,28001),'Label':final})


# In[ ]:


final_df.to_csv('sub1.csv',index=False)

