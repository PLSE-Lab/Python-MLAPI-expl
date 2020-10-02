#!/usr/bin/env python
# coding: utf-8

# **Wisconsin Breast Cancer Detection with KNN, SVM & Lazy Predict(All-rounder Python Library)**
# 
# UCI data repository
# 
# Kaggle data repository
# 
# Attribute Information:
# 
# Sample code number: id number Clump Thickness: 1 - 10 Uniformity of Cell Size: 1 - 10 Uniformity of Cell Shape: 1 - 10 Marginal Adhesion: 1 - 10 Single Epithelial Cell Size: 1 - 10 Bare Nuclei: 1 - 10 Bland Chromatin: 1 - 10 Normal Nucleoli: 1 - 10 Mitoses: 1 - 10 Class: (2 for benign, 4 for malignant) Malignant==> Cancerous
# 
# Benign==> Not Cancerous (Healthy)
# 
# Background
# 
# All of our bodies are composed of cells. The human body has about 100 trillion cells within it. And usually those cells behave in a certain way. However, occasionally, one of these 100 trillion cells, behave in a different way and keeps dividing and pushes the other cells around it out of the way. That cell stops observing the rules of the tissue within which it is located and begins to move out of its normal position and starts invading into the tissues around it and sometimes entering the bloodstream and becoming is called a metastasis.
# 
# In summary, as we grow older,throughout a lifetime, we go through this knid of situation where a particular kind of gene is mutated where the protein that it makes is abnormal and drives the cell to behave in a different way that we call cancer.
# 
# This is what Dr. WIlliam H. Wolberg was observing and put together this dataset.
# 
# Can we predict whether a cell is Malignant or Benign?
# 
# **Let's start!!!****

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


my_filepath = "../input/breast-cancer-csv/breastCancer.csv"


# In[ ]:


my_data=pd.read_csv("../input/breast-cancer-csv/breastCancer.csv")


# In[ ]:


my_data.head()


# **Data Pre-processing**

# In[ ]:


my_data['class'].value_counts()


# In[ ]:


my_data.shape


# In[ ]:


my_data.dtypes


# In[ ]:


my_data['bare_nucleoli']


# In[ ]:


my_data[my_data['bare_nucleoli']=='?']


# In[ ]:


my_data[my_data['bare_nucleoli']=='?'].sum()


# In[ ]:


digits_in_bare_nucleoli=my_data.bare_nucleoli.str.isdigit()


# In[ ]:


digits_in_bare_nucleoli


# In[ ]:


my_df=my_data.replace('?',np.nan)


# In[ ]:


my_df.bare_nucleoli


# In[ ]:


my_df.median()


# In[ ]:


my_df.describe


# In[ ]:


my_df=my_df.fillna(my_df.median())


# In[ ]:


my_df.bare_nucleoli


# In[ ]:


my_df.dtypes


# In[ ]:


my_df['bare_nucleoli']=my_df['bare_nucleoli'].astype('int64')


# In[ ]:


my_df.dtypes


# **Exploratory Data Analysis**

# In[ ]:


my_df.head()


# In[ ]:


my_df.drop('id',axis=1,inplace=True)


# In[ ]:


my_df.head()


# In[ ]:


my_df.describe().T


# In[ ]:


import seaborn as sns


# In[ ]:


sns.distplot(my_df['class'])


# In[ ]:


my_df.hist(bins=20, figsize=(40,40),layout=(6,3));


# In[ ]:


plt.figure(figsize=(25,20))
sns.boxplot(data=my_df, orient='h')


# In[ ]:


my_df.corr()


# In[ ]:


plt.figure(figsize=(40,20))

sns.heatmap(my_df.corr(), vmax=1, square=True, annot=True,cmap='viridis')
plt.title('Correlation Between Different Attributes')
plt.show()


# In[ ]:


try:
    sns.distplot(my_df)
except RuntimeError as re:
    if str(re).startswith("Selected KDE bandwidth is 0. Cannot estimate density."):
        sns.distplot(my_df, kde_kws={'bw': 0.1})
    else:
        raise re


# In[ ]:


try:
    sns.pairplot(my_df)
except RuntimeError as re:
    if str(re).startswith("Selected KDE bandwidth is 0. Cannot estimate density."):
        sns.pair(my_df, kde_kws={'bw': 0.1})
    else:
        raise re


# **Building Our Model**

# In[ ]:


my_df.head()


# In[ ]:


X=my_df.drop('class',axis=1)
y=my_df['class']


# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=1)


# **KNeighborsClassifier**

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


KNN=KNeighborsClassifier(n_neighbors=5, weights='distance')


# In[ ]:


KNN.fit(X_train,y_train)


# In[ ]:


predicted_1=KNN.predict(X_test)
predicted_1


# In[ ]:


from scipy.stats import zscore


# In[ ]:


print("KNeighborsClassifier Algorithm has predicted {0:2g}%".format(KNN.score(X_test,y_test)*100))


# **Support Vector Machine**

# In[ ]:


from sklearn.svm import SVC


# In[ ]:


svc=SVC(gamma=0.025, C=3)
svc.fit(X_train,y_train)


# In[ ]:


prediction_2=svc.predict(X_test)
prediction_2


# In[ ]:


print("Support Vector Machine Algorithm has predicted {0:2g}%".format(svc.score(X_test,y_test)*100))


# In[ ]:


Knn_Predictions=pd.DataFrame(predicted_1)
Svc_Predictions=pd.DataFrame(prediction_2)


# In[ ]:


df_new=pd.concat([Knn_Predictions,Svc_Predictions],axis=1)


# In[ ]:


df_new.columns=[['Knn_Predictions','Svc_Predictions']]
df_new


# In[ ]:


from sklearn.metrics import classification_report


# In[ ]:


print('KNN Classification Report')

print('>>>'*10)

print(classification_report(y_test,predicted_1))


# In[ ]:


print('SVC Classification Report')

print('>>>'*10)

print(classification_report(y_test,prediction_2))


# **Let's try Lazy Predict The All-rounder!**

# In[ ]:


get_ipython().system('pip install lazypredict')


# In[ ]:


import lazypredict
from lazypredict.Supervised import LazyClassifier


# In[ ]:


data = my_df
X=my_df.drop('class',axis=1)
y=my_df['class']
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=.3,random_state =1)
clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
models,predictions = clf.fit(X_train, X_test, y_train, y_test)
models


# In[ ]:


print('Top 10 Performing Models')
print('>>>'*10)
models.head(10)


# In[ ]:


from sklearn import metrics

print('KNN Confusion Matrix')

cm=metrics.confusion_matrix(y_test,predicted_1, labels=[2,4])

df_cm=pd.DataFrame(cm, index=[i for i in [2,4]],columns=[i for i in ['Predict M','predict B']])

plt.figure(figsize=(10,8))
sns.heatmap(df_cm, annot=True)


# In[ ]:


from sklearn import metrics

print('SVC Confusion Matrix')

cm=metrics.confusion_matrix(y_test,prediction_2, labels=[2,4])

df_cm=pd.DataFrame(cm, index=[i for i in [2,4]],columns=[i for i in ['Predict M','predict B']])

plt.figure(figsize=(10,8))
sns.heatmap(df_cm, annot=True)

