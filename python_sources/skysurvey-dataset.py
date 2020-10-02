#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import pandas as pd
df = pd.read_csv('../input/sloan-digital-sky-survey-dr16/Skyserver_12_30_2019 4_49_58 PM.csv')
df.head()


# In[ ]:


df.info()


# In[ ]:


df['class'].unique()


# In[ ]:


df.shape


# In[ ]:


df.isnull().sum()


# In[ ]:


df.var()


# In[ ]:


df.describe()


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt

correlation = df.corr()
sns.heatmap(correlation,cmap='magma',linecolor='white',linewidths=.1)


# In[ ]:


correlation = df.corr(method='pearson')
correlation.head()


# In[ ]:


sns.barplot(x='class',y='objid',data=df,palette='Purples')


# In[ ]:


sns.barplot(x='class',y='ra',data=df,palette='RdPu')


# In[ ]:


sns.barplot(x='class',y='ra',data=df,palette='YlOrRd')


# In[ ]:


sns.barplot(x='class',y='specobjid',data=df,palette='Purples')


# In[ ]:


sns.barplot(x='class',y='run',data=df,palette='Reds')


# In[ ]:



fig, (ax1, ax2, ax3) = plt.subplots(ncols = 3, figsize = (24, 6))
sns.distplot(df[df['class'] == 'STAR'].redshift, ax = ax1, bins = 30, color = 'g')
sns.distplot(df[df['class'] == 'GALAXY'].redshift, ax = ax2, bins = 30, color = 'r')
sns.distplot(df[df['class'] == 'QSO'].redshift, ax = ax3, bins = 30, color = 'b')
plt.show()


# In[ ]:


df.head()


# In[ ]:


sns.stripplot(x="class", y="ra", data=df, palette="BuPu")


# In[ ]:


sns.stripplot(x="class", y="plate", data=df, palette="BuPu")


# In[ ]:


X = df.drop("class",axis  = 1)
y = df["class"]


# In[ ]:


X.head()


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.33, random_state=42)


# In[ ]:


X_train.head()


# In[ ]:


import numpy as np
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(X_train)
X_train.head()


# In[ ]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
scaler.transform(X_train)


# In[ ]:


from sklearn import preprocessing
import numpy as np

X_scaled = preprocessing.scale(X_train)
X_scaled.mean(axis=0)
X_scaled.std(axis=0)


# In[ ]:


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(random_state = 101,max_iter=200,multi_class="multinomial",solver='sag')
lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)
print("Test score: " , lr.score(X_test,y_test))
print("Train score: " , lr.score(X_train,y_train))
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# In[ ]:


from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(random_state=42)
dtc.fit(X_train,y_train)
y_pred = dtc.predict(X_test)
print("Test accuracy",dtc.score(X_test,y_test))
print("Train accuracy",dtc.score(X_train,y_train))
print("Classification report :")
print(classification_report(y_test,y_pred))


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
rfc = RandomForestClassifier(max_depth=2, random_state=42,n_estimators=10)
rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)
print("Test accuracy",rfc.score(X_test,y_test))
print("Train accuracy",rfc.score(X_train,y_train))
print("Classification report :")
print(classification_report(y_test,y_pred))


# In[ ]:


from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
svc = make_pipeline(StandardScaler(), SVC(gamma='auto',random_state=42,kernel='sigmoid'))
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)
print("Test accuracy",svc.score(X_test,y_test))
print("Train accuracy",svc.score(X_train,y_train))
print("Classification report :")
print(classification_report(y_pred,y_test))


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
error_rate = []
for i in range(1,40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))

plt.figure(figsize = (10,8))
plt.plot(range(1,40), error_rate, color='blue', linestyle='dashed', marker='v', markerfacecolor='red', markersize=10)


# In[ ]:


from sklearn.metrics import classification_report, confusion_matrix
knn = KNeighborsClassifier(n_neighbors= 5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print('Classification Report: \n', classification_report(y_test, y_pred))
print('Confusion Matrix: \n', confusion_matrix(y_test, y_pred))
knn_train_acc = knn.score(X_train, y_train)
print('Training Score: ', knn_train_acc)
knn_test_acc = knn.score(X_test, y_test)
print('Testing Score: ', knn_test_acc)


# In[ ]:




