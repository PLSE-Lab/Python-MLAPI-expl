#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df = pd.read_csv("/kaggle/input/adult-pmr3508/train_data.csv",
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")


# In[ ]:


df.describe()


# In[ ]:


df.columns


# In[ ]:


df.tail(10)


# In[ ]:


df.dropna(inplace=True)
df.drop(axis=1, columns="Id", inplace=True)


# In[ ]:


df.head()


# In[ ]:


sns.pairplot(df, hue='income')


# In[ ]:


df['income'].value_counts().plot(kind='bar')


# In[ ]:


df['sex'].value_counts().plot(kind='bar')


# In[ ]:


df['hours.per.week'].plot(kind='hist')


# In[ ]:


sns.scatterplot(df['education.num'], df['hours.per.week'], hue=df['income'])


# In[ ]:


sns.countplot(y=df['occupation'], hue=df['income'])


# In[ ]:


sns.countplot(y=df['race'], hue=df['income'])


# In[ ]:


sns.countplot(x=df['education.num'], hue=df['race'])


# 

# In[ ]:


testAdult = pd.read_csv("/kaggle/input/adult-pmr3508/test_data.csv",
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")


# In[ ]:


testAdult.head(3)


# In[ ]:


testAdult.dropna(inplace=True)
testAdult.drop(axis='1', columns=['Id','education'], inplace=True)


# In[ ]:


adult = df.drop(axis='1', columns='education')


# In[ ]:


from sklearn import preprocessing


# In[ ]:


df.columns


# In[ ]:


qualiVars = ['workclass', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'native.country']


# In[ ]:


adult[qualiVars] = adult[qualiVars].apply(preprocessing.LabelEncoder().fit_transform)


# In[ ]:


testAdult[qualiVars] = testAdult[qualiVars].apply(preprocessing.LabelEncoder().fit_transform)


# 

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


from sklearn.model_selection import cross_val_score


# In[ ]:


Xtrain = adult.iloc[:,0:-1]
Ytrain = adult.income


# In[ ]:


scores = []
for i in range(1,50):
    knn = KNeighborsClassifier(n_neighbors=i)
    scores.append(np.mean(cross_val_score(knn, Xtrain, Ytrain, cv=10)))


# In[ ]:


scores


# In[ ]:


plt.figure()
plt.plot(range(1,50), scores, color='blue',linestyle='dashed', marker='o', markerfacecolor='red')
plt.xlabel("K")
plt.ylabel("Cross_val score")


# In[ ]:


plt.figure(figsize=(10,6))
plt.plot(range(10,30), scores[10:30], color='blue',linestyle='dashed', marker='o', markerfacecolor='red')
plt.xlabel("K")
plt.ylabel("Cross_val score")


# In[ ]:


K = 18


# In[ ]:


knn = KNeighborsClassifier(n_neighbors=K)


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(Xtrain,Ytrain,test_size=0.20)


# In[ ]:


knn.fit(X_train,y_train)


# In[ ]:


pred = knn.predict(X_test)


# In[ ]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# In[ ]:


print("K=18\n")
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))
print('Accuracy: ',accuracy_score(y_test, pred))


# In[ ]:


knn = KNeighborsClassifier(n_neighbors=K)
knn.fit(Xtrain,Ytrain)
YTestAdult = knn.predict(testAdult)


# In[ ]:


YTestAdult


# In[ ]:


resultSubmission = pd.DataFrame(YTestAdult)


# In[ ]:


resultSubmission.index.name = 'Id'


# In[ ]:


resultSubmission.rename({0:'income'}, axis=1, inplace=True)


# In[ ]:


resultSubmission


# In[ ]:


#resultSubmission.to_csv("resultSubmission.csv")

