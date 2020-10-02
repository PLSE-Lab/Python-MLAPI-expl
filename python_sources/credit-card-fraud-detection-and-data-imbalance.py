#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np


# ###### loading the data

# In[ ]:


train=pd.read_csv('../input/creditcardfraud/creditcard.csv')


# In[ ]:


train.head()


# ##### getting basic information about the dataset

# In[ ]:


train.info()


# In[ ]:


train['Class'].unique()


# #### Time - Number of seconds elapsed between this transaction and the first transaction in the dataset
# 
# ##### Amount - Transaction amount
# 
# ##### Class 1 for fraudulent transactions , 0 otherwise

# In[ ]:


train.describe()


# ##### cheching for null values in the dataset

# In[ ]:


train.isnull().sum()


# ###### no null values in the dataset
# ###### that's really good

# In[ ]:


train.tail()


# In[ ]:


train.shape


# ###### So there are 284807 rows in the dataset and 31 columns .
# ###### Let's check the count of different target variables
# 

# In[ ]:


train.Amount[train.Class == 1].describe()


# In[ ]:


train.Amount[train.Class == 0].describe()


# ###### it's clearly visible that the dataset is highly imbalanced
# ###### the number of rows for both the classes in the target variable differs with a very large margin,which is not good
# ###### so we don't have enough examples of one class(ie the fraud class in this case) to give it to our model for training.
# ###### we will take some steps for this imbalanced dataset but let's try first with this dataset.

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# In[ ]:


y=train['Class']
X=train.drop(['Class'],axis=1)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10)


# In[ ]:


model=LogisticRegression(solver='lbfgs',max_iter=1000)


# In[ ]:


model.fit(X_train,y_train)


# In[ ]:


model.predict(X_test)


# In[ ]:


model.score(X_test,y_test)


# ###### the accuracy with a logistic regression model is really good somewhere around 99.8%

# ###### let's do some data visualisation to get some idea about the behaviour of the features

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec


# In[ ]:


v_features = train.ix[:,1:29].columns


# In[ ]:


plt.figure(figsize=(12,28*4))
gs = gridspec.GridSpec(28, 1)
for i, cn in enumerate(train[v_features]):
    ax = plt.subplot(gs[i])
    sns.distplot(train[cn][train.Class == 1], bins=50)
    sns.distplot(train[cn][train.Class == 0], bins=50)
    ax.set_xlabel('')
    ax.set_title('histogram of feature: ' + str(cn))
plt.show();


# In[ ]:


correlation_matrix = train.corr()
fig = plt.figure(figsize=(12,9))
sns.heatmap(correlation_matrix,vmax=0.8,square = True)
plt.show()


# In[ ]:


df=train


# In[ ]:


df.head()


# Dropping some of the columns from the dataset that were similar with the other columns

# In[ ]:


df = df.drop(['V28','V27','V26','V25','V24','V23','V22','V20','V15','V13','V8'], axis =1)


# as we dont know exactly what are these features dropping columns is not a good idea but to check how the model performs in this data we are dropping these columns.

# In[ ]:


df.head()


# In[ ]:


y_1=df['Class']
X_1=df.drop(['Class'],axis=1)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X_1, y_1, test_size=0.20)


# In[ ]:


model.fit(X_train,y_train)


# In[ ]:


model.score(X_test,y_test)


# ###### The result is almost the same in both the cases 
# ###### Let's do something about the data imbalance

# In[ ]:


sns.countplot(train['Class'])


# ###### Clearly the data is highly imbalaced

# In[ ]:


sns.boxplot(x="Class", y="Amount", hue="Class",data=train, palette="PRGn",showfliers=False)


# In[ ]:


train.head()


# Data imbalance can be treated with resampling the data.
# data resampling can be of two types
# 
# 1.under sampling
# 
# under sampling-Undersampling aims to balance class distribution by randomly eliminating majority class examples.  This is done until the majority and minority class instances are balanced out.
# 
# 2.over sampling
# 
# Over-Sampling increases the number of instances in the minority class by randomly replicating them in order to present a higher representation of the minority class in the sample.
# 
# for more information about resampling
# refer analytics vidhya's blog https://www.analyticsvidhya.com/blog/2017/03/imbalanced-classification-problem/
# 

# #  UNDER-SAMPLING

# ###### Collecting all the rows of fraud class.

# In[ ]:


df_fraud=train[train.Class == 1]


# In[ ]:


df_fraud.head()


# In[ ]:


df_fraud.info()


# ###### collecting all the rows of genuine transaction.

# In[ ]:


df_genuine=train[train.Class == 0]


# In[ ]:


df_genuine.info()


# In[ ]:


# Randomly selecting 4000 rows from the genuine dataset

df_new_genuine=df_genuine.iloc[58457:60457]

#df_new_gen=df_gen.sample(4000)


# In[ ]:


# combining the both dataset the dataset with genuine transaction details and fraud transaction details
train_new = pd.concat([df_new_genuine, df_fraud],ignore_index=True, sort =False)


# In[ ]:


train_new.head()


# In[ ]:


train_new.info()


# In[ ]:


y_2=train_new['Class']
X_2=train_new.drop(['Class'],axis=1)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X_2, y_2, test_size=0.20)


# In[ ]:


model.fit(X_train,y_train)


# In[ ]:


model.score(X_test,y_test)


# # OVER-SAMPLING

# In[ ]:


train.head()


# In[ ]:


for _ in range(5):
    df_fraud = pd.concat([df_fraud, df_fraud],ignore_index=True, sort =False)


# ###### running the above cell will replicate the df_fraud dataframe and thus it will create a dataframe with more rows with the same data.

# In[ ]:


df_fraud_new=df_fraud


# In[ ]:


df_fraud_new.info()


# ###### Now the dataframe has 15744 rows for the fraud class.
# ###### we can proceed with this over-sampled data for training and prediction.

# In[ ]:


df_fraud_new.head()


# ###### Concatenating the two dataframes df_genuine and df_fraud_new to create a training dataset for further use. 

# In[ ]:


train_new_1 = pd.concat([df_genuine, df_fraud_new],ignore_index=True, sort =False)


# ###### Shuffling the dataset 

# In[ ]:


train_new_1 .iloc[np.random.permutation(len(train_new_1 ))]


# In[ ]:


train_new_1.info()


# In[ ]:


y_3=train_new_1['Class']
X_3=train_new_1.drop(['Class'],axis=1)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X_3, y_3, test_size=0.20)


# In[ ]:


model.fit(X_train,y_train)


# In[ ]:


model.score(X_test,y_test)


# first we used logistic regression on the imbalaced data then we did some sampling and again performed logistic regression on the data.
# 
# we did both under-sampling and over-sampling and checked the performance of our model.
# 
# it's always good to do sampling on the dataset if it is imbalanced to give the model appropriate number of examples of both the target for better training.

# # If you like my work give it a thumps UP.

# In[ ]:




