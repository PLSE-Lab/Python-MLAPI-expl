#!/usr/bin/env python
# coding: utf-8

# Prediction using Randon Forest for Forest Cover Type

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
df = pd.read_csv("../input/covtype.csv")
df.head()


# In[ ]:


df.shape


# In[ ]:


df.isnull().sum()


# In[ ]:


df.describe()


# In[ ]:


df.Cover_Type.unique()


# In[ ]:


df.corr()["Cover_Type"]


# In[ ]:


plt.hist(df["Cover_Type"])


# In[ ]:


df.corr()["Cover_Type"].plot(kind="bar")


# In[ ]:


df1=df.iloc[:,0:14]
df2=df['Cover_Type']
df1=df1.join(df2)
df1.head()


# In[ ]:


df1.corr()


# In[ ]:


fig = plt.subplots(figsize=(10,10))
sns.heatmap(df1.corr(),vmax=0.5,square=True,annot=True,cmap='Blues')
plt.xticks(rotation=90)
plt.yticks(rotation=0)


# In[ ]:


fig,axs=plt.subplots(ncols=3)
sns.boxplot(x='Cover_Type',y='Elevation',data=df,ax=axs[0])#highest in 1 & 7 lowest in 4
sns.boxplot(x='Cover_Type',y='Aspect',data=df,ax=axs[1])
sns.boxplot(x='Cover_Type',y='Slope',data=df,ax=axs[2])


# In[ ]:


X=df.drop(["Cover_Type"],axis=1)
y=df["Cover_Type"]
X.shape
y.shape
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,random_state=42)
rf=RandomForestClassifier(n_estimators=7,class_weight='balanced',n_jobs=2,random_state=42)
rf.fit(X_train,y_train)
y_predict=rf.predict(X_test)
score=rf.score(X_test,y_test)
print(score)

