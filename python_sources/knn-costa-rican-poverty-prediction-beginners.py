#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


df_train = pd.read_csv("../input/train.csv")


# In[ ]:


df_train.head()


# In[ ]:


df_test = pd.read_csv("../input/test.csv")
df_origin_test = df_test.copy()


# In[ ]:


df_train.head()


# In[ ]:


plt.figure(figsize = (30,10))
sns.heatmap(df_train.isna(),yticklabels=False,cbar=False,cmap='viridis')


# In[ ]:


pd.DataFrame(df_train.isna().sum(),columns=["Count"])[pd.DataFrame(df_train.isna().sum(),columns=["Count"])["Count"]>0]


# In[ ]:


df_train.fillna((df_train.mean()), inplace=True,axis=0)


# In[ ]:


df_test.fillna((df_test.mean()), inplace=True,axis=0)


# In[ ]:


plt.figure(figsize = (30,10))
sns.heatmap(df_train.isna(),yticklabels=False,cbar=False,cmap='viridis')


# In[ ]:


sns.set_style('whitegrid')
sns.countplot(x='parentesco1',data=df_train,palette='RdBu_r')


# In[ ]:


df_train["parentesco1"].value_counts()


# In[ ]:


df_categorial_train = df_train.select_dtypes(include=['object']).head()


# In[ ]:


df_categorial_train.head()


# In[ ]:


df_train.dependency.replace(to_replace=dict(yes=1, no=0), inplace=True)


# In[ ]:


df_test.dependency.replace(to_replace=dict(yes=1, no=0), inplace=True)


# In[ ]:


df_train.edjefe.replace(to_replace=dict(yes=1, no=0), inplace=True)


# In[ ]:


df_test.edjefe.replace(to_replace=dict(yes=1, no=0), inplace=True)


# In[ ]:


df_train.edjefa.replace(to_replace=dict(yes=1, no=0), inplace=True)


# In[ ]:


df_test.edjefa.replace(to_replace=dict(yes=1, no=0), inplace=True)


# In[ ]:


df_train.pop('Id')
df_train.pop('idhogar')


# In[ ]:


df_test.pop('Id')
df_test.pop('idhogar')


# In[ ]:


df_train.dependency.value_counts()


# In[ ]:


df_train.dependency=df_train.dependency.astype('float64')
df_train.dependency.dtype


# In[ ]:


df_test.dependency=df_test.dependency.astype('float64')


# In[ ]:


df_train.edjefe=df_train.edjefe.astype('int')
df_train.edjefa=df_train.edjefa.astype('int')


# In[ ]:


df_test.edjefe=df_test.edjefe.astype('int')
df_test.edjefa=df_test.edjefa.astype('int')


# In[ ]:


df_train.select_dtypes(include=['object']).count()


# In[ ]:


sns.set_style('whitegrid')
sns.countplot(x='Target',data=df_train,palette='RdBu_r')


# In[ ]:


y_train = df_train.pop("Target")


# In[ ]:


X_train = df_train


# In[ ]:


df_test.head()


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
arr=[2,4,8,16,32,64,128,168,200,220]
train_score_randomfor=[]
test_score_randomfor=[]
for i in arr:
    rfreg = RandomForestClassifier(n_estimators = i)
    rfreg = rfreg.fit(X_train, y_train)
    train_score_randomfor.append(rfreg.score(X_train , y_train))


# In[ ]:


from matplotlib.legend_handler import HandlerLine2D
line1, = plt.plot(arr, np.array(train_score_randomfor)*100, 'b', label="Train AUC")
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('AUC score')
plt.xlabel('n_estimators')
plt.show()


# In[ ]:


rfreg = RandomForestClassifier(n_estimators = 30)
rfreg = rfreg.fit(X_train, y_train)
rfreg.score(X_train , y_train)


# In[ ]:


predict = rfreg.predict(df_test)


# In[ ]:


pd.Series(predict).value_counts()


# In[ ]:


predict


# In[ ]:


df_submission = pd.DataFrame({"Id":df_origin_test.Id,"Target":predict})


# In[ ]:


df_submission.to_csv("Submission_Finale.csv",index=False)

