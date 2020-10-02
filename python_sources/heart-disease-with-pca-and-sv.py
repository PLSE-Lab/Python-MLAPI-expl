#!/usr/bin/env python
# coding: utf-8

# In[1280]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import chi2,SelectKBest
from sklearn.metrics import accuracy_score
import os
from scipy import stats
from sklearn.preprocessing import scale
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[1281]:


df=pd.read_csv('../input/heart.csv')


# In[1282]:


df.head()


# In[1283]:


df.info() #No Null values are present


# In[1284]:


plt.figure(figsize=(12,8))
plt.tight_layout()
df.hist()
plt.show()
# Histogram as chol has some skewness applying log function to chol


# In[1285]:


#chl and oldpeak having skewness
df['chol']=np.log(df['chol'])


# In[1286]:


plt.figure(figsize=(12,8))
plt.tight_layout()
df.hist()
plt.show()


# In[1287]:


plt.figure(figsize=(12,8))
plt.tight_layout()
df.boxplot()
plt.show()

#Finding the Outliers


# In[1288]:


z=np.abs(stats.zscore(df))


# In[1289]:


#Removing the outliers
df = df[(z < 3).all(axis=1)]


# In[1290]:


plt.figure(figsize=(12,8))
plt.tight_layout()
df.boxplot()
plt.show()


# In[ ]:





# In[1291]:


#corelation map
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(),cmap='winter',annot=True)


# In[1292]:


sns.pairplot(df)
plt.figure(figsize=(10,8))
plt.show()


# In[1293]:


#change the values to Bins
bins=[35,45,55,65,75]
labels=['>35','>45','>55','>65']
df['Age_cat']=pd.cut(df['age'],bins=bins,labels=labels)


# In[1294]:


dummy_df=pd.get_dummies(df,columns=['cp','fbs','restecg','exang','slope','ca','thal','Age_cat'])


# In[1295]:


dummy_df=dummy_df.drop(columns=['age'])


# In[1296]:


Feature=dummy_df.drop(columns='target')
Target=dummy_df['target']


# In[1297]:


kbest=SelectKBest(chi2)
best_fit=kbest.fit(Feature,Target)


# In[1298]:


#Select columns which has more 15 score during chi2 square test
columns=list()
for i,j in zip(Feature.columns.tolist(),best_fit.scores_.tolist()):
    if j>=15:
        print(i,'=',j)
        columns.append(i)


# In[1299]:


columns=pd.Series(columns)


# In[1300]:


Feature=dummy_df.loc[:,columns]


# In[1301]:


Feature.columns


# In[1302]:


Feature=scale(Feature)


# In[1303]:


XR,XT,YR,YT= train_test_split(Feature,Target,test_size=0.20,random_state=1)


# In[1304]:


model=LogisticRegression().fit(XR,YR)


# In[1305]:


ypred=model.predict(XT)


# In[1306]:


accuracy_score(YT,ypred)


# In[1307]:


from sklearn.tree import DecisionTreeClassifier
model2=DecisionTreeClassifier(max_depth=4,).fit(XR,YR)
ypred=model2.predict(XT)
accuracy_score(YT,ypred)


# In[1308]:


from sklearn.ensemble import RandomForestClassifier
model3=RandomForestClassifier(max_depth=15).fit(XR,YR)
ypred=model3.predict(XT)
accuracy_score(YT,ypred)


# In[1309]:


from sklearn.decomposition import PCA
data_transoform=PCA(n_components=10).fit(XR)


# In[1310]:


pca_values=data_transoform.fit_transform(XR)


# In[1311]:


model.fit(pca_values,YR)


# In[1312]:


model.score(pca_values,YR)


# In[1313]:


data_transoform=PCA(n_components=10).fit(XT)
pca_values=data_transoform.fit_transform(XT)


# In[1314]:


model.score(pca_values,YT)


# In[1315]:


pca_values1=PCA().fit_transform(XR)
model2.fit(pca_values1,YR)
print("train score",model2.score(pca_values1,YR))
pca_values2=PCA().fit_transform(XT)
print("test score",model2.score(pca_values2,YT))


# In[1316]:


pca_values1=PCA().fit_transform(XR)
model3.fit(pca_values1,YR)
print("train score",model3.score(pca_values1,YR))
pca_values2=PCA().fit_transform(XT)
print("test score",model3.score(pca_values2,YT))


# In[1317]:


from xgboost import XGBClassifier
model4=XGBClassifier()


# In[1318]:


pca_values1=PCA().fit_transform(XR)
model4.fit(pca_values1,YR)
print("train score",model4.score(pca_values1,YR))
pca_values2=PCA().fit_transform(XT)
print("test score",model4.score(pca_values2,YT))


# In[1319]:


from sklearn.svm import SVC
model5=SVC(C=1,kernel='rbf')


# In[1320]:


pca_values1=PCA().fit_transform(XR)
model5.fit(pca_values1,YR)
print("train score",model5.score(pca_values1,YR))
pca_values2=PCA().fit_transform(XT)
print("test score",model5.score(pca_values2,YT))


# In[ ]:




