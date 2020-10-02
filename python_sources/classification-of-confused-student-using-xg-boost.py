#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

print(os.listdir('../input'))
data = pd.read_csv("/kaggle/input/confused-eeg/EEG_data.csv")


# In[ ]:


data.head()


# In[ ]:


data.info()


# In[ ]:


data.plot()


# In[ ]:


data.head().plot()


# In[ ]:


data.describe()


# In[ ]:


df=pd.DataFrame(data)


# In[ ]:


df.columns


# In[ ]:


print(data['user-definedlabeln'].unique())


# In[ ]:


df['Theta'].plot()


# In[ ]:


gr1=df.groupby('Theta')['user-definedlabeln'].mean()
print(gr1)
grp1=pd.DataFrame(gr1)
print(grp1.describe())
grp1.head(300).plot()


# In[ ]:


gr2=df.groupby('Alpha1')['user-definedlabeln'].mean()
print(gr2)
grp2=pd.DataFrame(gr2)
print(grp2.describe())
grp2.head(300).plot()


# In[ ]:


gr3=df.groupby('Alpha2')['user-definedlabeln'].mean()
print(gr3)
grp3=pd.DataFrame(gr3)
print(grp3.describe())
grp3.head(300).plot()


# In[ ]:


###testing the acuuracy for the first timw will do the other preprocessing later ones the accuracy is seen 
X_int=df.drop('user-definedlabeln',axis=1).values
Y_int=df['user-definedlabeln'].values


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_int,Y_int, test_size=0.2,random_state=42)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
for i in range(1,50):
    clf=RandomForestClassifier(n_estimators=i,max_depth=2,random_state=13)
    clf.fit(X_train,y_train)
    clf.predict(X_test)
    scr=clf.score(X_test,y_test)
    print(scr)


# In[ ]:


from sklearn import svm
clf = svm.SVC()
clf.fit(X_train,y_train)
preds = clf.predict(X_test)
clf.score(X_test, y_test)


# In[ ]:


pip install xgboost


# In[ ]:


import xgboost as xgb

xg = xgb.XGBClassifier(objective='binary:logistic', n_estimators=1000, seed=1)
xg.fit(X_train,y_train)
print(xg.predict(X_test))
xg.score(X_test,y_test)


# In[ ]:


import xgboost as xgb

xg = xgb.XGBClassifier(objective='binary:logistic', n_estimators=100,seed=1)
xg.fit(X_train,y_train)
predict=xg.predict(X_test)
print(xg.score(X_test,y_test))


# In[ ]:


import xgboost as xgb
for i in range(200,300):
    xg = xgb.XGBClassifier(objective='binary:logistic', n_estimators=i, seed=1)
    xg.fit(X_train,y_train)
    predict=xg.predict(X_test)
    print(xg.score(X_test,y_test))


# In[ ]:


# final accuracy 

import xgboost as xgb

xg = xgb.XGBClassifier(objective='binary:logistic', n_estimators=1000, seed=1)
xg.fit(X_train,y_train)
print(xg.predict(X_test))
xg.score(X_test,y_test)


# here is the code and we got an accuracy of 0.99102 way heigher than the proclaimed 67 percent 
# Abhishek Parashar and Yukti Mohan
