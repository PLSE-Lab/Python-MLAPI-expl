#!/usr/bin/env python
# coding: utf-8

# ## **Titanic Dataset solution**
# 
# **Question-Problem Statement**
# 
# The sinking of the RMS Titanic is one of the most infamous shipwrecks in history.  On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. This sensational tragedy shocked the international community and led to better safety regulations for ships.
# 
# One of the reasons that the shipwreck led to such loss of life was that there were not enough lifeboats for the passengers and crew. Although there was some element of luck involved in surviving the sinking, some groups of people were more likely to survive than others, such as women, children, and the upper-class.
# 
# In this solution we will predict whether a person survives or not based on various factors like social class, gender and age.

# In[1]:


# Import all the necessary packages
import numpy as np 
import pandas as pd
import statsmodels.api as sms
from sklearn.linear_model import LogisticRegression
import os
print(os.listdir("../input"))
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from random import randint
# suppress warnings from final output
import warnings
warnings.simplefilter("ignore")


# ## **Wrangle Data**
# 
# ### **Gather Data**
# 
# We acquire data from various sources as needed to solve our problem statement.
# 
# **Data has already been provided by Kaggle in form of two datasets train and test**

# In[2]:


# Load test dataset
test_df=pd.read_csv('../input/test.csv')
test_df.head()


# In[3]:


# Load train dataset
train_df=pd.read_csv('../input/train.csv')
train_df.head()


# ### **Assess Data**
# 
# Assess the data visually and programmatically to find dirty or untidy data.
# 1. Dirty data means missing or inorrect data
# 2. Untidy data mean that the data has structural problems
# 
# After assessment note the problems found at the end of this segment.

# In[4]:


train_df.info()


# In[5]:


test_df.info()


# In[6]:


train_df.sample(5)


# In[7]:


train_df.Name.value_counts()


# ### **Documentation**
# 
# The data is dirty but it does not have structural problems. 

# * name has special characters with nicknamesin and maiden names for married women is specified
# * pclass,sex,embarked and survived should be categorical datatypes
# * missing values in age, cabin and embarked(train) and fare(test)

# ### **Clean**
# 
# Data cleaning can be done manually and programmatically.
# Cleaning tha dats does not make sure that your model is correct, it only makes sure that your model will work.
# Clean the data using python function with iterations to save time.

# **Issue-1**
# name column has special characters with nicknames and maiden names of married women have been specified
# 
# **Define**
# Remove special characters and maiden names of married women

# In[8]:


# Combine datasets to reduce repetition of code
single=[train_df,test_df]


# In[9]:


# Remove the maiden names specified for married women
for data in single:
    data['Name']=data['Name'].apply(lambda x: list(x.split('('))[0])


# In[10]:


# Remove the special characters with nicknames
for data in single:
    data['Name']=data['Name'].apply(lambda x: list(x.split('"'))[0])


# In[11]:


train_df['Name'].value_counts()


# **Issue-2**
# 
# Missing data in age and embarked in train dataset
# Fare data missing in test dataset
# 
# **Define**
# Fill the missing data in various columns
# 1. Age column - Take the average in each class and generate random numbers between highest and lowest mean.
# 2. Embarked- Only two values are missing. Fill the missing data with most frequent data.
# 3. Fare - Only one value is missing. Fill the missing data with the median.

# In[12]:


# Calculate mean of age in each class for train dataset
train_df.groupby(['Pclass'])['Age'].describe()


# In[13]:


# Generate random numbers between the lowest and highest mean
train_df['Age']=train_df['Age'].fillna(randint(24,37))


# In[14]:


# Calculate mean of age in each class for test dataset
test_df.groupby(['Pclass'])['Age'].describe()


# In[15]:


# Generate random numbers between the lowest and highest mean
test_df['Age']=test_df['Age'].fillna(randint(24,42))


# In[16]:


# Find the most frequent 'Embarked' value
freq_embark=train_df.Embarked.mode()[0]
freq_embark


# In[17]:


# Fill the missing values
train_df['Embarked']=train_df['Embarked'].fillna(freq_embark)


# In[18]:


test_df['Fare']=test_df['Fare'].fillna(test_df['Fare'].median())


# In[19]:


#Test whether the above code worked
train_df.info()


# In[20]:


#Test whether the above code worked
test_df.info()


# **Issue-3**
# 
# Pclass,Sex,Embarked and Survived should be categorical data types
# 
# **Define**
# In order to perform mathematical operations on the dataset, the columns need to be int or float.
# Assign numerical values intead of characters in categorical columns
# 1. Sex: 1-female, 0-male
# 2. Embarked: 1- Southampton, 2-Cherbourg, 3-Queenstown

# In[21]:


# Map columns to numerical values
for data in single:
    data['Sex']=data['Sex'].map({'female':1,'male':0}).astype(int)
    data['Embarked']=data['Embarked'].map({'S':1,'C':2,'Q':3}).astype(int)


# In[22]:


train_df.head()


# ## **Analyze using visuals**
# 
# We only want columns which statistically significant in helping solve our problem statement. To determine these columns we will use Logistic Regression. 
# 
# **Logistic Regression**
# 
# This machine learning techinque is used to predict binar outcomes. For example whether a transaction is fraud or not.
# In the summary of the model we can see **p-value** which help us in determining whether a column is statistically significant in predicting our model. **p-value** should be less than **alpha(0.05)**

# In[23]:


train_df.info()


# In[24]:


train_df['intercept']=1
model=sms.Logit(train_df['Survived'],train_df[['intercept','Pclass','Sex','Age','SibSp','Parch','Embarked']])
result=model.fit()
result.summary()


# **Observation**
# 
# As we can see from the above model of Logistic Regression that **p-value** for Parch and Embarked (subject to change as we are dealing with random numbers) column is more than  the **alpha-value** of 0.05.
# 
# This suggests that these two columns are not statistically significant for our prediction model.

# ## Visualizations

# In[25]:


g=sns.FacetGrid(data=train_df,col='Survived',row='Pclass');
g.map(plt.hist,'Age');


# The above plot depicts the division of the survived column depending upon age and class of the passengers.
# Many people btween the age of 20-50 did not survive and the count is especially high among 3rd class passengers.
# 
# As the age increases or decreases from 20-50 range the chances of survival increases

# In[26]:


train_df.groupby(['Pclass','Sex','Survived'])['Survived'].count()


# In[27]:


g=sns.FacetGrid(data=train_df,col='Survived',row='Pclass');
g.map(sns.countplot,'Sex');


# The above plot depicts the division of the survived column depending upon gender and class of the passengers.
# It is very clear that most of the females on board survived.

# ## **Feature Engineering**
# 
# Drop columns that are not needed.
# 

# In[28]:


train_df.drop(columns=['PassengerId','Name','Parch','Ticket','Fare','Cabin','Embarked','intercept'],inplace=True)


# In[29]:


train_df.info()


# In[30]:


test_df.drop(columns=['Name','Parch','Ticket','Fare','Cabin','Embarked'],inplace=True)


# In[31]:


test_df.info()


# ## **Model and predict**
# 
# Using Logistic Regression.

# In[32]:


# Diivide the data into test and train
X_train=train_df.drop('Survived',axis=1)
Y_train=train_df['Survived']
X_test=test_df.drop('PassengerId',axis=1)


# In[33]:


# Fit the model
model = LogisticRegression()
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)


# In[34]:


# Calucalte the accuracy
accuracy = round(model.score(X_train, Y_train), 4)
accuracy


# In[35]:


# Gather the solution into a new dataframe
final_df = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_pred
    })


# In[36]:


final_df.head()


# In[37]:


final_df.shape


# In[38]:


# Save the dataframe to a csv file
final_df.to_csv('submission.csv',index=False)


# **References**
# 
# https://www.kaggle.com/startupsci/titanic-data-science-solutions

# In[ ]:




