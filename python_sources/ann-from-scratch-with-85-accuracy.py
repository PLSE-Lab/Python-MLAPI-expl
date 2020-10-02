#!/usr/bin/env python
# coding: utf-8

# # ANN FROM SCRATCH WITH 85% ACCURACY.

# <img src='https://drive.google.com/uc?id=14nbP-PHEEtkaALbx0Yihr44qipu0Sz_l' width=1000 >

# In[478]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from plotly.offline import download_plotlyjs, init_notebook_mode, iplot
import cufflinks as cf
init_notebook_mode(connected=True)
cf.go_offline()

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[479]:


# Let us Import the Dataset.
df_train = pd.read_csv("../input/train.csv")
df_test = pd.read_csv("../input/test.csv")


# In[480]:


# Top 5 Entries in Training Dataset
df_train.head()


# In[481]:


# This is to check the correlation between the Passenger Class and Survived ones.
df_train.groupby(['Pclass', 'Survived']).count()


# * Upper class person Survived more then the other class person.

# # BASIC EDA

# In[482]:


# Let Us See the Basic Information about training dataset
df_train.info()


# ### 1). Let us get detail about Null Values.

# In[483]:


null = df_train.isnull().sum()
per = null/len(df_train)*100
null = pd.DataFrame(data={'Number of Null Values':null, 'Percentage Of Null Values':per})
null


# * As 77% of values in Cabin Column are Null values then we will drop the Cabin column.
# * As 20% values in 'Age' column are Null values, here we will replace them by mean of the column.

# ### 2). Drop the Unwanted Columns.

# In[484]:


df_train.drop(labels='Cabin', axis=1, inplace=True)
df_test.drop(labels='Cabin', axis=1, inplace=True)


# ### 3). Dealing With Missing Values.

# * a). Deal with 'Age' column.

# In[485]:


# Let's replace Age column with mean of the column.
print(df_train['Age'].mean())
df_train['Age'].fillna(value=df_train['Age'].mean(), inplace=True)
df_test['Age'].fillna(value=df_test["Age"].mean(), inplace=True)


# * b). Deal with 'Embarked' column.

# In[486]:


# Let us visualise the Embarked column.
df_train['Embarked'].value_counts().iplot(kind='bar', color='deepskyblue')


# In[487]:


# Replace Embarked column with mode of the column which is 'S'.

df_train['Embarked'].fillna(value='S',inplace=True)
df_test['Embarked'].fillna(value='S', inplace=True)
df_train['Embarked'].unique()


# In[488]:


df_train.head()


# # FEATURE ENGINEERING

# ### 1). Dealing With "Name" Column.
# * Our Target is to replace the values of Name Column with Values 0,1,2,....
# 

# In[489]:


combine = [df_train, df_test]
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(df_train['Title'], df_train['Sex'])


# In[490]:



for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col', 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    
df_train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()


# In[491]:


# Mapping with 1,2,3,4,5
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

df_train.head()


# In[492]:


# Let us Drop the Unwanted Columns
df_train.drop(labels=['PassengerId','Name','Ticket'], inplace=True, axis=1)
df_test.drop(labels=['PassengerId','Name','Ticket'], inplace=True, axis=1)


# ### 2). Feature Engineering in "Age" Column.
# * Here first we will see the band of Age and then we will replace values of Age in column which fall in particular band with 1 or 2 or 3 or 4.

# In[493]:


df_train['AgeBand'] = pd.cut(df_train['Age'], 5)
df_train[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)


# In[494]:


combine = [df_train, df_test]
for dataset in combine:    
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age'] = 4

df_train = df_train.drop(['AgeBand'], axis=1)
combine = [df_train, df_test]
df_train.head()    


# In[ ]:





# ### 3). Dealing With "SibSp" and "Parch" columns.

# In[495]:


combine = [df_train, df_test]
for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

df_train[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[496]:


# Making New Column "IsAlone" which will have values 1 or 0 for whether she/he is alone or not respectively.
for dataset in combine:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

df_train[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()


# In[497]:


# Droping Parch, SibSp and FamilySize columns.
df_train = df_train.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
df_test = df_test.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
combine = [df_train, df_test]

df_train.head()


# ### 4). Dealing With "FareBand" Column.

# In[498]:


df_train['FareBand'] = pd.qcut(df_train['Fare'], 4)
df_train[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)


# In[499]:


# Filling Null values in Fare column in test dataset.
df_test['Fare'].fillna(value=df_test['Fare'].mean(), inplace=True)


# In[500]:


# Replacing values of Fare column with 0,1,2,3.

combine = [df_train, df_test]
for dataset in combine:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

df_train = df_train.drop(['FareBand'], axis=1)
combine = [df_train, df_test]
    
df_train.head(10)


# In[ ]:





# ## =============================================================

# ## PREDICTION WITH ANN
# * 1). Here we will deal with MCSS.
#     * M = dealing with Missing data, which we have already done.
#     * C = dealing with Categorical values
#     * S = Spliting of Dataset which we do not required as data is already splited.
#     * S = Scaling of features on same scale.
# * 2). Then we will create our model and then we will Train out model and then we will make prediction on our test dataset.

# In[501]:


# Converting dataset into array.
X_train = df_train.iloc[:, 1:].values
y_train = df_train.iloc[:, 0].values
X_test = df_test.iloc[:,:].values


# In[502]:


# Dealing with categorical features. 

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_x=LabelEncoder()
X_train[:, 1]=labelencoder_x.fit_transform(X_train[:,1])  
X_train[:, 4]=labelencoder_x.fit_transform(X_train[:, 4])

X_test[:, 1]=labelencoder_x.fit_transform(X_test[:,1])  
X_test[:, 4]=labelencoder_x.fit_transform(X_test[:, 4])


onehotencoder_x=OneHotEncoder(categorical_features=[0,2,3,4,5]) 
X_train=onehotencoder_x.fit_transform(X_train).toarray()
X_test=onehotencoder_x.fit_transform(X_test).toarray()


# In[503]:


len(X_train[0]), len(X_test[0])


# In[504]:


# Scaling the features on the same scale.
# Scaling of training set and test set must be done with the same sc_x object.
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
X_train = sc_x.fit_transform(X_train)
X_test = sc_x.fit_transform(X_test)


# In[505]:


# Importing tensorflow library
import tensorflow as tf
from tensorflow.keras.layers import Dropout


# In[506]:


model = tf.keras.Sequential()


# In[507]:


# Creating Input layer and 1st Hidden layer.
# Here Dropout is our regularization paramter(penalty parameter), which will deal with problem of overfitting .

model.add(tf.keras.layers.Dense(units=12, input_dim=22, activation='relu', kernel_initializer='uniform'))
model.add(Dropout(rate=0.1))


# In[508]:


# Creating 4 Hidden layers.

model.add(tf.keras.layers.Dense(units=12,  activation='relu', kernel_initializer='uniform'))
model.add(Dropout(rate=0.1))
model.add(tf.keras.layers.Dense(units=12,  activation='relu', kernel_initializer='uniform'))
model.add(Dropout(rate=0.1))
model.add(tf.keras.layers.Dense(units=12,  activation='relu', kernel_initializer='uniform'))
model.add(Dropout(rate=0.1))
model.add(tf.keras.layers.Dense(units=12,  activation='relu', kernel_initializer='uniform'))
model.add(Dropout(rate=0.1))


# In[509]:


# Adding Output Layer.

model.add(tf.keras.layers.Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))


# In[510]:


# Compiling the Model.
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# In[511]:


# Trian our Model.
history = model.fit(X_train, y_train, batch_size=50, epochs=500, validation_split=0.2)


# In[512]:


print(f"Accuracy of training dataste\t {np.mean(history.history['acc'])}")
print(f"Accuracy of Validating dataset\t {np.mean(history.history['val_acc'])}")


# * Here we are getting the accuracy of 85%, which is good in my point of view.

# In[ ]:





# In[513]:


# Prediction on Test dataset.
y_pred = model.predict(X_test)


# In[514]:


# Converting the values of y_pred into 0 and 1.
# 1 means person will Survived and 0 means person will not Survived.

new_y_pred = []
for var in y_pred:
    if var>=0.7:
        new_y_pred.append(1)
    else:
        new_y_pred.append(0)


# In[515]:


# Creating Submission file.

df_test = pd.read_csv("../input/test.csv")
df_result = pd.DataFrame(data={'Passengerid':df_test['PassengerId'], 'Survived':new_y_pred})
df_result.to_csv("submission.csv", index=False)


# # IF THIS KERNEL IS HELPFUL THEN PLEASE UPVOTE.
# <img src='https://drive.google.com/uc?id=1qihsaxx33SiVo5dIw-djeIa5SrU_oSML' width=500 >

# In[ ]:




