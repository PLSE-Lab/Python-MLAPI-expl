#!/usr/bin/env python
# coding: utf-8

# This is python notebook code written in kernel directly. Objective of this problem is to identify the features that affect the survival of passengers.
# This we will achieve through EDA and then applying models to verify our assumptions.
# 

# In[1]:


# import the required packages
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # visualization
import seaborn as sns # visualization
from sklearn.linear_model import LogisticRegression, RidgeClassifier, LinearRegression, Ridge
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import scale, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# ## Read the data and understand the features first

# In[2]:


# read the train dataset
df1 = pd.read_csv("../input/train.csv")
print(df1.shape)
df1.head()


# In[3]:


# create a new dataset that we will use for further computations and recreate the same when required from df1.
titanic = df1.copy()


# In[4]:


# check datatype of features
titanic.describe().T


#     We have 10 independent variables (excluding Passenger Id), and we have a task to identify on which features is our target variable dependent most.
#     we can see that data has minimum age of 0.42, which we can consider it to be a baby on onboard.
#     
#     We can generate a hypothesis before we move based on features we can see.
#     1. If the passenger is from 1st class he/she would be given first preference to be saved on lifeboat, then second, then third.
#     2. Females will be given preference over males, so higher number of females will be saved.
#     3. Based on age, children and old age passengers will be saved before young and middle aged generation.
#     4. If a person has siblings his/her chances are higher to survive, since we can assume that they can work together.
#     5. If a passenger is accompanied by parents or children then he/she might try to save the dependents first.
#     We cannot say right now based on information provided, about the survival of passengers based on port from which passenger is embarked.

# In[5]:


# check the datatype of each feature
titanic.info()


#     We can see that there are total 891 records, and Age, Cabin and Embarked features have missing values.
#     Check the count of missing values in each feature

# In[6]:


print(titanic.isna().sum())
print(f"percentage of missing values:\nAge: {titanic['Age'].isna().sum()/len(titanic) :.2f}, \nCabin: {titanic['Cabin'].isna().sum()/len(titanic) :.2f}, \nEmbarked: {titanic['Embarked'].isna().sum()/len(titanic) :.2f}")


#     We can impute the Embarked missing values with mode straight away since we have only 2 missing values. 
#     Cabin has 77% missing values, hence it is logical to drop the column
#     Age has 20% missing values, we can neither drop these rows nor column. So we can impute the same at later stage before generating model.

# In[7]:


titanic.Embarked.fillna(titanic.Embarked.mode()[0], inplace=True)


# In[8]:


# Convert all features into required datatype
# titanic.Survived = titanic.Survived.astype('object')
# titanic.Pclass = titanic.Pclass.astype('object')
# titanic.SibSp = titanic.SibSp.astype('object')
# titanic.Parch = titanic.Parch.astype('object')


# In[9]:


titanic.drop('Cabin', axis=1, inplace=True)


# ## Exploratory Data Analysis
#     Now we test our hypothesis and try to get more information

# In[10]:


survived_count = round(titanic.Survived.value_counts(normalize=True)*100, 2)
survived_count.plot.bar(title='Proportion of Survived and non-Survived passengers in dataset')
plt.xticks(rotation=0, fontsize=12)
plt.yticks(fontsize=12)
for x,y in zip([0,1],survived_count):
    plt.text(x,y,y,fontsize=12)
plt.show()


#     The data is not sparse in target variable

# ### Univariate analysis

# In[11]:


plt.subplots(figsize=(15,15))

# We check the number of passengers survived in each class.
ind = sorted(titanic.Pclass.unique())
sur_0 = titanic.Pclass[titanic['Survived'] == 0].value_counts().sort_index()
sur_1 = titanic.Pclass[titanic['Survived'] == 1].value_counts().sort_index()
total = sur_0.values+sur_1.values
sur_0_prop = np.true_divide(sur_0, total)*100
sur_1_prop = np.true_divide(sur_1, total)*100
plt.subplot(321)
plt.bar(ind, sur_1_prop.values, bottom=sur_0_prop.values, label='1')
plt.bar(ind, sur_0_prop.values, label='0')
plt.title("Number of Passengers survived in each class", fontsize=15)
for x,y,z in zip(ind,[100]*3,sur_1):
    plt.text(x,y,z,fontsize=12)
for x,y,z in zip(ind,sur_0_prop,sur_0):
    plt.text(x,y,z,fontsize=12)
plt.xticks(ind)


# plot survival proportion in Age, for it we create age bins
bins = [0,10,20,30,40,50,60,70,80]
names = ['0-10','10-20','20-30','30-40','40-50','50-60','60-70','70-80']
df_temp = titanic.dropna()
df_temp['Age_bins'] = pd.cut(x=titanic.Age, bins=bins, labels=names, right=False)

ind = sorted(df_temp.Age_bins.unique()[:8])
age_0 = df_temp.Age_bins[df_temp['Survived'] == 0].value_counts().sort_index()
age_1 = df_temp.Age_bins[df_temp['Survived'] == 1].value_counts().sort_index()
total = age_0.values+age_1.values
age_0_prop = np.true_divide(age_0, total)*100
age_1_prop = np.true_divide(age_1, total)*100
plt.subplot(322)
plt.bar(ind, age_1_prop.values, bottom=age_0_prop.values, label='1')
plt.bar(ind, age_0_prop.values, label='0')
plt.title("Number of Passengers survived in each age group", fontsize=15)
for x,y,z in zip(ind,[100]*8,age_1):
    plt.text(x,y,z,fontsize=12)
for x,y,z in zip(ind,age_0_prop,age_0):
    plt.text(x,y,z,fontsize=12)

plt.legend(loc='upper right')
    
# check the proportion of passengers survived as per gender
ind = sorted(titanic.Sex.unique())
sex_0 = titanic.Sex[titanic['Survived'] == 0].value_counts().sort_index()
sex_1 = titanic.Sex[titanic['Survived'] == 1].value_counts().sort_index()
total = sex_0.values+sex_1.values
sex_0_prop = np.true_divide(sex_0, total)*100
sex_1_prop = np.true_divide(sex_1, total)*100
plt.subplot(323)
plt.bar(ind, sex_1_prop.values, bottom=sex_0_prop.values, label='1')
plt.bar(ind, sex_0_prop.values, label='0')
plt.title("Number of Passengers survived genderwise", fontsize=15)
for x,y,z in zip(ind,[100]*2,sex_1):
    plt.text(x,y,z,fontsize=12)
for x,y,z in zip(ind,sex_0_prop,sex_0):
    plt.text(x,y,z,fontsize=12)


# check the proportion of passengers survived from port embarked
ind = sorted(titanic.Embarked.unique())
emb_0 = titanic.Embarked[titanic['Survived'] == 0].value_counts().sort_index()
emb_1 = titanic.Embarked[titanic['Survived'] == 1].value_counts().sort_index()
total = emb_0.values+emb_1.values
emb_0_prop = np.true_divide(emb_0, total)*100
emb_1_prop = np.true_divide(emb_1, total)*100
plt.subplot(324)
plt.bar(ind, emb_1_prop.values, bottom=emb_0_prop.values, label='1')
plt.bar(ind, emb_0_prop.values, label='0')
plt.title("Number of Passengers survived from port embarked", fontsize=15)
for x,y,z in zip(ind,[100]*3,emb_1):
    plt.text(x,y,z,fontsize=12)
for x,y,z in zip(ind,emb_0_prop,emb_0):
    plt.text(x,y,z,fontsize=12)


# check the proportion of passengers survived with Siblings and Spouse
ind = sorted(titanic.SibSp.unique())
sib_0 = titanic.SibSp[titanic['Survived'] == 0].value_counts().sort_index()
sib_1 = titanic.SibSp[titanic['Survived'] == 1].value_counts().sort_index()
sib_1 = titanic.SibSp[titanic['Survived'] == 1].value_counts().sort_index()
for i in sib_0.index:
    if i not in sib_1.index:
        sib_1[i]=0
total = sib_0.values+sib_1.values
sib_0_prop = np.true_divide(sib_0, total)*100
sib_1_prop = np.true_divide(sib_1, total)*100
plt.subplot(325)
plt.bar(ind, sib_1_prop.values, bottom=sib_0_prop.values, label='1')
plt.bar(ind, sib_0_prop.values, label='0')
plt.title("Number of Passengers survived with Siblings and Spouse onboard", fontsize=15, loc='center')
for x,y,z in zip(ind,[100]*9,sib_1):
    plt.text(x,y,z,fontsize=12)
for x,y,z in zip(ind,sib_0_prop,sib_0):
    plt.text(x,y,z,fontsize=12)
plt.xticks(ind)


ind = sorted(titanic.Parch.unique())
par_0 = titanic.Parch[titanic['Survived'] == 0].value_counts().sort_index()
par_1 = titanic.Parch[titanic['Survived'] == 1].value_counts().sort_index()
for i in par_0.index:
    if i not in par_1.index:
        par_1[i]=0
total = par_0.values+par_1.values
par_0_prop = np.true_divide(par_0, total)*100
par_1_prop = np.true_divide(par_1, total)*100
plt.subplot(326)
plt.bar(ind, par_1_prop.values, bottom=par_0_prop.values, label='1')
plt.bar(ind, par_0_prop.values, label='0')
plt.title("Number of Passengers survived with Parents and Children onboard", fontsize=15, loc='left')
for x,y,z in zip(ind,[100]*7,par_1):
    plt.text(x,y,z,fontsize=12)
for x,y,z in zip(ind,par_0_prop,par_0):
    plt.text(x,y,z,fontsize=12)
plt.xticks(ind)

plt.show()


#     Analysis of our hypothesis:
#     1. Proportion of passengers survived is higher in 1st class and reducing in 2nd and 3rd. So our first hypothesis is true and this becomes an important variable for survival prediction.
#     2. Going by age wise, children have higher survival rate, whereas old ones above the age of 70 have not survived at all. We can use this feature to predict survival rate on test data.
#     3. Females have higher survival rate than males, even though there are more number of male passengers.
#     4. Passengers embarked at Cherbourg have higher survival chances than others. We don't know the reason behind this since the existing data does not reveal much about this.
#     5. As we said about siblings and spouse, these passengers have higher survival chances.
#     6. Also passengers with lesser dependents have higher survival chances.
#     
#     so our all the hypothesis are true.

# In[12]:


# check the distribution of Fare in data. For better prediction results we need the continous data to be nearly closely normally distributed
plt.subplots(figsize=(15,6))
plt.subplot(121)
sns.distplot(titanic.Fare)
# We can see the data is positively skewed. We remove the skewness by taking natlog of the Fare column values
# we have 0s in Fare column which we replace with 0.0001 so as to have the record in data but not change it's meaning
titanic.Fare[titanic['Fare'] == 0] = 0.0001
Fare_ln = np.log(titanic.Fare)
plt.subplot(122)
sns.distplot(Fare_ln)
plt.show()


# ### Multivariate analysis
# 
#     We can analyse multiple features at a time to understand interaction between the feature. I will skip this and move towards missing values imputation and model building.

# ## Impute missing value in Age column using Regression technique

# In[13]:


# separate the missing value records from titanic data for imputation
age_X = titanic[titanic['Age'].notna()].drop(['Age','PassengerId','Name','Ticket','Fare','Survived'], axis=1)
age_y = titanic.Age[titanic['Age'].notna()]
age_test = titanic[titanic['Age'].isna()].drop(['Age','PassengerId','Name','Ticket','Fare','Survived'], axis=1)


# In[14]:


age_X[['SibSp','Parch']] = MinMaxScaler().fit_transform(age_X[['SibSp','Parch']])
age_test[['SibSp','Parch']] = MinMaxScaler().fit_transform(age_test[['SibSp','Parch']])

age_X = pd.get_dummies(age_X)
age_test = pd.get_dummies(age_test)


# In[15]:


linreg = LinearRegression()
linreg.fit(age_X, age_y)
age_pred = linreg.predict(age_test)


# In[16]:


titanic.Age[titanic['Age'].isna()] = age_pred


# In[17]:


titanic.isna().sum()


# ## Build model

# In[18]:


X_train = titanic.drop(['Survived','PassengerId','Name','Ticket','Fare'], axis=1)
col = X_train.columns
y_train = titanic.Survived
y_train = y_train.astype('int')


# In[19]:


X_train[['SibSp','Parch','Age']] = MinMaxScaler().fit_transform(X_train[['SibSp','Parch','Age']])
X_train = pd.get_dummies(X_train)


# In[20]:


logreg = LogisticRegression().fit(X_train,y_train)


# ## Predict resutls on test data

# In[21]:


# Read the test data
test = pd.read_csv('../input/test.csv')
test = test[col]
test.isna().sum()


# In[22]:


# separate the missing value records from test for imputation
age_test = test[test['Age'].isna()].drop('Age', axis=1)

age_test[['SibSp','Parch']] = MinMaxScaler().fit_transform(age_test[['SibSp','Parch']])

age_test = pd.get_dummies(age_test)

test.Age[test['Age'].isna()] = linreg.predict(age_test)
test.isna().sum()


# In[23]:


test[['SibSp','Parch','Age']] = MinMaxScaler().fit_transform(test[['SibSp','Parch','Age']])
test = pd.get_dummies(test)
y_pred = logreg.predict(test)


# ### Create submission file

# In[24]:


submit = pd.read_csv('../input/gender_submission.csv')
submit.head()


# In[25]:


submit['Survived'] = y_pred
# submit.to_csv('gender_submission.csv', index=False)


# ## Comparing with Neural Network

# In[26]:


import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.callbacks import EarlyStopping
earl = EarlyStopping(patience=3) # early stopping


# In[30]:


#Setting up the model
model1 = Sequential()
model2 = Sequential()
#Add first layer
model1.add(Dense(50,activation='relu',input_shape=(titanic.shape[1],)))
model2.add(Dense(100,activation='relu',input_shape=(titanic.shape[1],)))
#Add second layer
model1.add(Dense(32,activation='relu'))
model2.add(Dense(50,activation='relu'))
#Add output layer
model1.add(Dense(2,activation='sigmoid'))
model2.add(Dense(2,activation='sigmoid'))
#Compile the model
model1.compile(optimizer='sgd',loss='binary_crossentropy',metrics=['accuracy'])
model2.compile(optimizer='sgd',loss='binary_crossentropy',metrics=['accuracy'])


# In[35]:


X_train = titanic.drop(['Survived','PassengerId','Name','Ticket','Fare'], axis=1).values
y_train = titanic.Survived.values
y_train = y_train.astype('int')


# In[ ]:





# In[ ]:




