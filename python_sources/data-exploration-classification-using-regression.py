#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

#import numpy as np # linear algebra
#import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# **Importing the packages**

# In[113]:


import pandas as pd                    # For Data Exploration
import numpy as np                     # For mathematical calculations 
import seaborn as sns                  # For data visualization 
import matplotlib.pyplot as plt        # For plotting graphs 
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings                        # To ignore any warnings 
warnings.filterwarnings("ignore")


# **Reading the train dataset and displaying its columns**

# In[114]:


train_df = pd.read_csv('../input/train.csv')
train_df.columns


# **Displaying top 5 records of train dataset**

# In[115]:


train_df.head(5)


# **Reading the test dataset and displaying its columns**

# In[116]:


test_df = pd.read_csv('../input/test.csv')
test_df.columns


# **Displaying top 5 records of test dataset**

# In[117]:


test_df.head(5)


# **Making copy of the dataset**

# In[118]:


train_original = train_df.copy()
test_original = test_df.copy()


# **Checking of datatypes of columns**

# In[119]:


train_df.dtypes


# **To check shape of the train and test dataset**

# In[120]:


train_df.shape, test_df.shape


# **Finding Missing Values in the train dataset**

# In[121]:


train_df.isnull().sum()


# In[122]:


train_df['Age'].fillna(train_df['Age'].mean(),inplace = True)


# In[123]:


train_df['Cabin'].fillna(train_df['Cabin'].mode()[0], inplace = True)


# In[124]:


train_df['Embarked'].fillna(train_df['Embarked'].mode()[0], inplace = True)


# **Finding Missing Values in the test dataset**

# In[125]:


test_df.isnull().sum()


# In[126]:


test_df['Age'].fillna(test_df['Age'].mean(),inplace = True)
test_df['Cabin'].fillna(test_df['Cabin'].mode()[0], inplace = True)


# **Hypothesis for survival**
# 1. Passengers belonging to Upper class had higher chances of survival
# 2. Women, Children Passengers had higher chances of survival
# 3. Passengers who boarded as a Family had higher chances of survival

# **Univariate Analysis of each features**

# **Frequency Table of Dependent/Target Variable**

# In[127]:


train_df['Survived'].value_counts(normalize = True)


# 38% of Passengers in Train dataset survived

# **Bar Plot of Dependent/Target Variable**

# In[128]:


train_df['Survived'].value_counts(normalize = True).plot.bar(title = "Survival %")


# **Frequency table for Independent Categorical Features**

# In[129]:


train_df['Pclass'].value_counts(normalize =  True)


# 1. 24% passengers belonged Upper or 1st class
# 2. 20% passengers belonged Middle or 2nd class
# 3. 55% passengers belonged Lower or 3rd class

# In[130]:


train_df['Sex'].value_counts(normalize= True)


# 1. 65% passengers were male
# 2. 35% passengers were female
# 

# In[131]:


train_df['Embarked'].value_counts()


# 1. 644 passengers embarked on Titanic from Southampton
# 2. 168 passengers embarked on Titanic from Cherbourg
# 3. 77 passengers embarked on Titanic from Queenstown

# In[132]:


train_df['SibSp'].value_counts()


# 608 passengers were travelling without any sibling, spouse

# In[133]:


train_df['Parch'].value_counts()


# 678 passengers were travelling without parent/children

# **Plot of Independent Numerical variable**

# In[134]:


plt.figure(1)
plt.subplot(121) 
sns.distplot(train_df['PassengerId']); 
plt.subplot(122) 
train_df['PassengerId'].plot.box(figsize=(16,5)) 
plt.show()


# In[135]:


plt.figure(2)
plt.subplot(221)
df = train_df.dropna()
sns.distplot(df['Age']);
plt.subplot(222)
df['Age'].plot.box(figsize = (16,5))
plt.show()


# In[136]:


plt.figure(3)
plt.subplot(321)
sns.distplot(train_df['Fare']);
plt.subplot(322)
train_df['Fare'].plot.box(figsize=(16,5))
plt.show()


# **Bi-Variate Analysis between each feature and target variable**

# In[137]:


Ticket_Class=pd.crosstab(train_df['Pclass'],train_df['Survived'])
Sex=pd.crosstab(train_df['Sex'],train_df['Survived'])
Siblings=pd.crosstab(train_df['SibSp'],train_df['Survived'])
Parents=pd.crosstab(train_df['Parch'],train_df['Survived'])
Embarked=pd.crosstab(train_df['Embarked'],train_df['Survived'])


# In[138]:


Ticket_Class.div(Ticket_Class.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))
plt.show() 
Sex.div(Sex.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))
plt.show() 
Siblings.div(Siblings.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))
plt.show() 
Parents.div(Parents.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))
plt.show() 
Embarked.div(Embarked.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))
plt.show() 


# 1. Survival rate was high for Passengers travelling in 1st and 2nd Class 
# 2. Female passengers had a higher survival rate
# 3. Passengers travelling with 1 or 2 siblings/spouse had a higher survival rate
# 4. Passengers travelling with 3 parents/children had a higher survival rate
# 5. Passengers who boarded from Cherbourg survived more
# 
# From point 3 and 4, it can be inferred that, passengers who were travelling as family with parents/children/spouse/sibilings had a higher survival rate compared to that of lone passengers.
# 
# 

# In[139]:


train_df.groupby('Survived')['PassengerId'].mean().plot.bar()
plt.show()


# Passenger Id doesnot seem to have much impact on survival

# In[140]:


bins=[0,12,20,60,80] 
group=['Children','Teenage','Adult', 'Senior Citizen'] 
train_df['Age_bin']=pd.cut(df['Age'],bins,labels=group)
Age_bin=pd.crosstab(train_df['Age_bin'],train_df['Survived']) 
Age_bin.div(Age_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True) 
plt.xlabel('Age') 
P = plt.ylabel('Percentage')
plt.show()


# Passengers from Age 0-60 had a higher survial rate. Mostly childrens and teenage

# **Heat Map to find Co-relation with numerial features and target variable**

# In[141]:


matrix = train_df.corr() 
fax = plt.subplots(figsize=(9, 6)) 
sns.heatmap(matrix, vmax=.8, square=True, cmap="BuPu");


# 1. SibSp and Parch are highly co-related as both indicates family members and its count
# 2. Fare and Survived are also highly co-related

# **Building the Model**

# **Dropping the columns which has no impact **

# In[142]:


train = train_df.drop(['PassengerId','Name','Ticket','Fare','Cabin','Age_bin'], axis =1)
test = test_df.drop(['PassengerId','Name','Ticket','Fare','Cabin'], axis =1)


# **Dropping Target variable from Train dataset and putting it into another dataset**

# In[143]:


X = train.drop('Survived',1)
y = train.Survived


# **Creating Dummy variables for Categorical Features**

# In[144]:


X=pd.get_dummies(X) 
train=pd.get_dummies(train) 
test=pd.get_dummies(test)


# **Splitting Train dataset into Train and Validation set**

# In[145]:


from sklearn.model_selection import train_test_split
x_train, x_cv, y_train, y_cv = train_test_split(X,y, test_size =0.3)


# **Fitting into Logistic Regression**

# In[146]:


from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score
model = LogisticRegression() 
model.fit(x_train, y_train)


# **Prediciting the Loan Status for Validation set**

# In[147]:


pred_cv = model.predict(x_cv)


# **Calculate Accuracy**

# In[148]:


accuracy_score(y_cv,pred_cv)


# **Prediciting Loan status for Test data set**

# In[149]:


pred_test = model.predict(test)


# **Taking PassengerId into a numpy array from test dataset**

# In[150]:


Passenger_Id = test_df['PassengerId'].values


# **Combining 2 numpy array into Panda dataframe**

# In[151]:


gender_submission = pd.DataFrame({'PassengerId': Passenger_Id, 'Survived': pred_test}, columns=['PassengerId', 'Survived'])


# **Saving the output**

# In[152]:


gender_submission.to_csv('gender_submission.csv', index = False)

