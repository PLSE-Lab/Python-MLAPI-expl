#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[2]:


# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# machine learning
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

#Evaluation
from sklearn.metrics import accuracy_score, classification_report


# In[3]:


df_train = pd.read_csv('../input/train.csv')
df_test  = pd.read_csv('../input/test.csv')
df_sub   = pd.read_csv('../input/gender_submission.csv')


# In[4]:


df_train.head()


# In[5]:


df_all = [df_train, df_test] #New list with both Train and Test data as it's elements
#train -> df_all[0]
#test  -> df_all[1] 


# In[6]:


df_all[0][:5]


# In[7]:


df_all[1][:5] #or .head()


# In[8]:


print(df_train.columns.values)  # or use tolist() to convert ir into a list from Index Array


# In[9]:


def Nan_data(df):
    key = df.isnull().sum().index.values
    value = df.isnull().sum().values
    return dict(zip(key,value))


# In[10]:


Nan_dict_train = Nan_data(df_train)
Nan_dict_test  = Nan_data(df_test)


# In[11]:


print('Training data Nan summary: \n {} \n'.format(Nan_dict_train))
print('Test data Nan summary: \n {}'.format(Nan_dict_test))


# In[12]:


#print(tuple(zip(df_train.dtypes.index,df_train.dtypes.values)))
df_train.dtypes


# In[13]:


Survived = df_train[df_train.Survived == 1].Survived.count()
print('No. of people survived in the Train dataset: {}'.format(Survived))


# In[14]:


Died = df_train[df_train.Survived == 0].Survived.count()
print('No. of people died in the Train dataset: {}'.format(Died))


# In[15]:


total = Survived+Died
survival_rate = Survived/total
print('Survived: {} \nSurvival_rate: {:.2f}%'.format(Survived, survival_rate*100))


# In[16]:


df_train.describe(percentiles=[0.25,0.50,0.61,0.62,0.75])


# Change in the value of Survived from 61% to 62% percentile gives an idea about the percentage of people survived is between 38%-39% of the total number of passengers.

# In[17]:


print(tuple(zip(df_train.dtypes.index,df_train.dtypes.values)))


# In[18]:


Cat_col = ([x[0] for x in tuple(zip(df_train.dtypes.index,df_train.dtypes.values)) 
           if x[1] == np.dtype('O') and x[0] not in ['Name','Cabin','Ticket']])
#df_train.Sex.dtype = dtype('O')
Cat_col


# Summary of Categorical Features:

# In[19]:


#df_train[Cat_col].describe() 
df_train.describe(include=['O']) 


# In[20]:


df_train.corr()


# In[21]:


def Feat_en(df):
    df['Family_Size'] = df.Parch + df.SibSp + 1
    df['SoloTravel'] = (df['Family_Size'] <= 1)
    df.loc[df['SoloTravel']==False,'SoloTravel'] = 0
    df.loc[df['SoloTravel']==True,'SoloTravel'] = 1
    # df[df['Family_Size'] <= 1, 'SoloTravel'] = 1
    # df[df['Family_Size'] > 1, 'SoloTravel'] = 0
    return df


# In[22]:


for df in df_all:
    df['Sex'] = df['Sex'].map({'female': 1, 'male': 0}).astype(int)


# In[23]:


# fillinf the Nan values for Embarked and Fare columns in Train and Test dataset respectively.
df_train['Embarked'].fillna(df_train['Embarked'].dropna().mode()[0], inplace=True)
df_test['Fare'].fillna(df_train['Fare'].dropna().mean(), inplace=True)


# In[24]:


for df in df_all:
    df['Embarked'] = df['Embarked'].map({'C': 0, 'S': 1, 'Q': 2}).astype(int)


# In[25]:


df_all[0].head()


# In[26]:


guess_ages = np.zeros((2,3))

for df in df_all:
    for i in range(0, 2):
        for j in range(1, 4):
            df_age = df[(df['Sex'] == i) & (df['Pclass'] == j)]['Age'].dropna()
            age_guess = df_age.median()
            # Convert random age float to nearest .5 age
            guess_ages[i,j-1] = int( age_guess/0.5 + 0.5 ) * 0.5
            
    for i in range(0, 2):
        for j in range(1, 4):
            df.loc[(df.Age.isnull() & (df.Sex==i) & (df.Pclass==j)), 'Age'] = guess_ages[i,j-1]

    df['Age'] = df['Age'].astype(int)


# In[27]:


df_test.isnull().sum()


# In[28]:


df_train.isnull().sum()


# In[29]:


# Take copy of original train/test dataframe before aplpying Feat_engineering
df_train_0 = df_train.copy() 
df_test_0 = df_test.copy()
#Apply Feat_en
train_df = Feat_en(df_train)
test_df = Feat_en(df_test)


# In[30]:


#check if there are nay nan/Null values in the dataframe
print(train_df.isnull().sum())
print(test_df.isnull().sum())


# In[31]:


train_df.head()


# In[32]:


def drop_col(df_todrop):
    drop_col = ['PassengerId', 'SibSp', 'Parch', 'Family_Size', 'Name', 'Cabin','Ticket']
    rem_col = [col for col in drop_col if col in df_todrop.columns.tolist()]
    return df_todrop.drop(columns=rem_col)


# In[33]:


train_df = drop_col(train_df)
test_df = drop_col(test_df)


# **Now we have the final training and test data prepared with all features in numerical forms.**

# In[34]:


train_df.head(3)


# In[35]:


test_df.head(3)


# > **Analysis of Correlation between various features and Survival**

# **1. Survival rate w.r.t. the Passenger's Class:**

# In[36]:


train_df[['Pclass','Survived']].groupby(['Pclass'],as_index=False).mean().sort_values(by='Survived',ascending=False)


# > Approx 63% of 1st Class Passengers Survived the accident.
# > Not so good numbers for 2nd Class Passengers and even worse Survival rate for 3rd Class Passengers.
# > This shows a bit of bias in the Survival vs Pclass. This will lead to a model which favours Survival of 1st Class Passengers as compared to 2nd and 3rd Class Passengers.

# **2. Survival rate w.r.t. Passenger's Gender:**

# In[37]:


train_df[['Sex','Survived']].groupby(['Sex'],as_index=False).mean().sort_values(by='Survived',ascending=False)


# > Female Survival rate(74.2%) is way higher than the male Survival rate(18.89%).
# So females had a better chance of surviving the accident. So, this model will have this behaviour shown to the test set as well.

# **3. Survival rate w.r.t. Traveller type(Solo vs Family):**

# In[38]:


train_df[['SoloTravel','Survived']].groupby(['SoloTravel'],as_index=False).mean().sort_values(by='Survived',ascending=False)


# > SoloTravellers(1) had a less survival rate as compared to the Family travellers(0).

# **4. Survival rate w.r.t. Embarked location:**

# In[39]:


train_df[['Embarked','Survived']].groupby(['Embarked'],as_index=False).mean().sort_values(by='Survived',ascending=False)


# Passengers embarking from 0, i.e., C had a better survival rate than the other two embarking locations.

# **Visualisation:**

# **1. Histogram of Count of Pclass W.r.t. Survival**

# In[40]:


g=sns.FacetGrid(train_df,col='Survived')
g.map(plt.hist, 'Pclass', bins=15)


# > We can observe that huge number of 3rd class passengers didn't survive as compared to 1st & 2nd class passengers.

# **2. Histogram of Count of people of various ages w.r.t. Survival**

# In[41]:


g=sns.FacetGrid(train_df,col='Survived')
g.map(plt.hist, 'Age', bins=15)


# > We see that passengers between 20-40 are the ones who were in huge numbers who died and survived. This could be simply because the number of passengers between 20-40 years of agegroup was the most.

# In[42]:


train_df[(train_df['Age'] >= 20) & (train_df['Age'] <= 40)].count()[0]


# > There are 579 passengers out of 891, who are between 20-40Yrs of age. 

# In[43]:


print('Percentage of passengers in 20-40 age group: {:.2f}%'.format((579/891)*100))


# In[44]:


#Count of people who Died in 20-40 Age group
train_df[(train_df['Age'] >= 20) & (train_df['Age'] <= 40) & (train_df['Survived'] == 0)].count()[0]


# In[45]:


#Count of people who Survived in 20-40 Age group
train_df[(train_df['Age'] >= 20) & (train_df['Age'] <= 40) & (train_df['Survived'] == 1)].count()[0]


# In[46]:


print('Percentage of passengers in 20-40 age group who survived: {:.2f}%'.format((208/579)*100))


# **3. Histogram of Count of people travelling Solo Vs Family w.r.t. Survival**

# In[47]:


g=sns.FacetGrid(train_df,col='Survived')
g.map(plt.hist, 'SoloTravel', bins=15)


# > People travelling solo died in more number as compared to people with Family

# **4. Histogram of Ticket Fare w.r.t. Survival**

# In[48]:


g=sns.FacetGrid(train_df,col='Survived')
g.map(plt.hist, 'Fare', bins=15)


# > Passengers who bought low cost tickets, didn't survive in big numbers. whereas, Passengers who bought a bit costly tickets had a better survival rate. All Passengers with $500 ticket survived.

# **5. Histogram of Count of people Emarking from various locations w.r.t. Survival**

# In[49]:


g=sns.FacetGrid(train_df,col='Survived')
g.map(plt.hist, 'Embarked', bins=15)


# > Passengers Embarking from 1 (S) died in most numbers w.r.t. other 2 embarking locations (S, Q).

# **Survival rate w.r.t Pclass and Gender**

# In[50]:


sns.catplot(x='Pclass',y='Survived',kind='bar',hue='Sex', data=train_df) 
# hue distributes the plot as per provided col. Here, Sex


# > Females survival rate is more than twice to thrice that of male survival rate.

# In[51]:


sns.catplot(x='Survived',y='Fare',kind='bar',hue='Sex',data=train_df)


# In[52]:


sns.catplot(x='Pclass',hue='Sex',kind='count',data=train_df)


# In[53]:


train_df.head()


# In[54]:


def get_OHE(df_toOHE):
    cat_cols = ['Pclass','Sex','Embarked','SoloTravel']
    df_dummies = pd.DataFrame()
    for col in cat_cols:
        df_dummies = pd.concat([df_dummies,pd.get_dummies(df_toOHE[col], prefix=col)],axis=1)
    return df_dummies
    
train_dummies = get_OHE(train_df)
test_dummies = get_OHE(test_df)


# In[55]:


train_dummies.head(2)


# In[56]:


def Merge_OHE(df_toMOHE, df_OHE):
    drop_col = ['Pclass','Sex','Embarked','SoloTravel']
    return df_toMOHE.merge(df_OHE, left_index=True, right_index=True).drop(columns=drop_col)


# In[57]:


df_train_1 = Merge_OHE(train_df, train_dummies)
df_test_1 = Merge_OHE(test_df, test_dummies)


# In[58]:


df_train_1.iloc[:,1:].head()


# In[59]:


X = df_train_1.iloc[:,1:]
y = df_train_1.iloc[:,0]

scaler = MinMaxScaler()  
#MinMAxScaling doesn't work on Categorical features as they are alreary scaled between 0 & 1.
#X[['Age','Fare']] = scaler.fit_transform(X[['Age','Fare']])
#X_df = pd.DataFrame(scaler.fit_transform(X), columns=df_train_1.columns[1:])
#df_test_1[['Age','Fare']] = scaler.transform(df_test_1[['Age','Fare']])
#X_test_df = pd.DataFrame(scaler.transform(df_test_1), columns=df_test_1.columns)

X = scaler.fit_transform(X)
X_sub = scaler.transform(df_test_1)


# In[60]:


X_train, X_test, y_train, y_test = train_test_split(X, y,random_state = 56, test_size = 0.25)


# In[61]:


model_0 = DecisionTreeClassifier(random_state=56)
model_0.fit(X_train,y_train)
print('Train Set Accuracy: {:.2f}%'.format(model_0.score(X_train,y_train)*100))
print('Test Set Accuracy: {:.2f}%'.format(model_0.score(X_test,y_test)*100))


# In[62]:


model_1 = RandomForestClassifier(n_estimators=220, random_state=56)
model_1.fit(X_train,y_train)
print('Train Set Accuracy: {:.2f}%'.format(model_1.score(X_train,y_train)*100))
print('Test Set Accuracy: {:.2f}%'.format(model_1.score(X_test,y_test)*100))


# In[63]:


model_2 = LogisticRegression(random_state=56)
model_2.fit(X_train,y_train)
print('Train Set Accuracy: {:.2f}%'.format(model_2.score(X_train,y_train)*100))
print('Test Set Accuracy: {:.2f}%'.format(model_2.score(X_test,y_test)*100))


# In[64]:


model_3 = SVC(kernel = 'linear',gamma='auto',random_state=56)
model_3.fit(X_train,y_train)
print('Train Set Accuracy: {:.2f}%'.format(model_3.score(X_train,y_train)*100))
print('Test Set Accuracy: {:.2f}%'.format(model_3.score(X_test,y_test)*100))


# In[65]:


model_4 = SVC(kernel = 'rbf', gamma='auto')
model_4.fit(X_train,y_train)
print('Train Set Accuracy: {:.2f}%'.format(model_4.score(X_train,y_train)*100))
print('Test Set Accuracy: {:.2f}%'.format(model_4.score(X_test,y_test)*100))


# In[66]:


for k in range(4,15):
    model_5 = KNeighborsClassifier(n_neighbors=k)
    model_5.fit(X_train,y_train)
    print('k={} Train Set Accuracy: {:.2f}%'.format(k,model_5.score(X_train,y_train)*100))
    print('k={} Test Set Accuracy: {:.2f}%\n'.format(k,model_5.score(X_test,y_test)*100))


# k=12 gives the best accuracy of 82.51%

# In[67]:


k=12
model_5 = KNeighborsClassifier(n_neighbors=k)
model_5.fit(X_train,y_train)
print('k={} Train Set Accuracy: {:.2f}%'.format(k,model_5.score(X_train,y_train)*100))
print('k={} Test Set Accuracy: {:.2f}%'.format(k,model_5.score(X_test,y_test)*100))


# In[68]:


Y_pred = model_3.predict(X_sub)


# In[69]:


df_sub.Survived = Y_pred
df_sub.head()


# In[70]:


df_sub.to_csv('Sub_1.csv',index=False)


# In[71]:


print(os.listdir("../working"))


# Used the following Kernel as Reference:
# https://www.kaggle.com/startupsci/titanic-data-science-solutions
