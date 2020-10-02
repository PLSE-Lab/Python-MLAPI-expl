#!/usr/bin/env python
# coding: utf-8

# # Random Forest Approach
# Applied Random Forest Classifier (Supervised learning model) on Titanic: Machine Learning from Disaster Data-set from Kaggle
# 
# ## Overview 
# The following jupyter notebook contains steps I used for drawing out predictions on the survived column. Since there is a target column/ variable present, this problem gets classified as a supervised learning problem.
# Steps:
# 1. Importing necessary libraries. (numpy, pandas, matplotlib, seaborn).
# 2. Reading in data.
# 2. Data Exploration and Visualization.
# 3. Feature Engineering. 
# 4. Construction of a predictive model( Any machine learning model in Surpervised Learning)
# 5. Fitting featurized data into the predictive model.
# 6. Drawing out predictions with test data.
# 
# ## Dependencies:
# * conda install numpy
# * conda install pandas
# * conda install scikit-learn
# * conda install matplotlib

# 
# ### Importing Libraries

# In[101]:


# Standard libraries
import pandas as pd
import numpy as np


# In[102]:


# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# #### Reading Data

# In[103]:


train = pd.read_csv('../input/train.csv')


# In[104]:


train.head()


# ### Data exploration and visualization

# In[105]:


train.info()


# In[106]:


train.describe()


# #### Checking for null values

# In[107]:


train.isnull().head()


# Plotting the null values on a heatmap for better visualization

# In[108]:


sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='plasma')
# every yellow line means missing data 


# 
# * Age and Cabin columns have missing data.
# *  Age column has relatively less missing data when compared to Cabin

# In[109]:


# First of all lets plot who survived and who didn't
sns.countplot(x = 'Survived', data = train)
# So about 580 didn't survive 
# And 320 survived


# * *Dataset has a greater number rows with survived label as 0*
# * *About 580+ classify as 0*
# * *And 320 as 1*

# In[110]:


# Lets look at survived with a hue of gender
sns.countplot(x = 'Survived', data = train, hue = 'Sex', palette='RdBu_r')


# In[111]:


# Lets look at survived with a hue of Pasenger class
sns.countplot(x = 'Survived', data = train, hue = 'Pclass')


# In[112]:


# Lets get an idea about the age of people in the data set
sns.distplot(train['Age'].dropna(), kde= False, bins = 30)


# In[113]:


sns.countplot(x = 'SibSp', data = train)
# By looking at this plot, most people on board neither had  siblings / spouses aboard


# In[114]:


# Another column which we haven't explored yet is the fare column
train['Fare'].mean()


# In[115]:


train['Fare'].hist( bins = 40, figsize = (10,4))
# most of the distribution is between 0 and 100 


# In[116]:


plt.figure(figsize = (10,7))
sns.boxplot(x = 'Pclass', y = 'Age', data = train)
# The figure shows that the Passengers in class 1 have older people 
# And younger people in lower Pclass


# In[117]:


# Filling in null age values
def substitution(columns):
    Age = columns[0]
    Pclass = columns[1]
    if pd.isnull(Age):
        if Pclass == 1:
            return 36         # approx mean value from blue box
        elif Pclass == 2:
            return 29        # approx mean value from orange box
        else:
            return 23         # approx mean value from green box  
    else:
        return Age           # is not null


# In[118]:


train['Age'] = train[['Age', 'Pclass']].apply(substitution, axis = 1)


# In[119]:


sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='plasma')
# no more missing values in Age 


# In[120]:


sns.heatmap(train.corr(), annot= True)
# Checking for correlation between columns


# In[121]:


train.drop('Cabin',axis=1,inplace=True)
# there are so many missing columns in cabin
# that it seems right to drop it


# In[122]:


train.head()


# In[123]:


sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='plasma')
# Final check for null values


# In[124]:


pd.get_dummies(train['Sex']).head()
# we need to convert the sex column
# otherwise the machine learning alogorithm won't be able process the data


# In[125]:


pd.get_dummies(train['Sex'], drop_first= True).head()
# now you can not feed both these columns as male and female are opposite
# and it will mess up the machine learning algorthim


# In[126]:


sex = pd.get_dummies(train['Sex'], drop_first= True)


# In[127]:


embark = pd.get_dummies(train['Embarked'],drop_first=True)
# same process with Embarked column


# In[128]:


embark.head()


# In[129]:


# Since Pclass is also a categorical column
pclass = pd.get_dummies(train['Pclass'],drop_first=True)


# In[130]:


train = pd.concat([train, sex, embark, pclass], axis = 1)


# In[131]:


train.head()
# now, we don't need sex, embarked, plcass column because we have encoded them.


# In[132]:


train.drop(['Sex','Embarked','Name','Ticket', 'Pclass'],axis=1,inplace=True)
# dropping columns which we are not going to use


# In[133]:


train.head()
# looks perfect for our machine learning algorithm
# all data is numeric


# In[134]:


# Features
X = train.drop('Survived', axis = 1)

# Target variable
y = train['Survived']


# In[135]:


from sklearn.ensemble import RandomForestClassifier
# Supervised learning 


# In[136]:


model = RandomForestClassifier(n_estimators=2756, max_depth= 5)
model.fit(X,y)


# In[137]:


test = pd.read_csv('../input/test.csv')


# In[138]:


test.columns


# In[139]:


test.info()


# In[140]:


test['Age'] = test[['Age', 'Pclass']].apply(substitution, axis = 1)


# In[141]:


# Preparing test data according to the model
sex = pd.get_dummies(test['Sex'], drop_first= True)
embark = pd.get_dummies(test['Embarked'],drop_first=True)
pclass = pd.get_dummies(test['Pclass'],drop_first=True)

test = pd.concat([test, sex, embark, pclass], axis = 1)

test.drop(['Sex','Embarked','Name','Ticket', 'Pclass', 'Cabin'],axis=1,inplace=True)


# In[142]:


test.columns


# In[143]:


# Checking for null values in test
sns.heatmap(test.isnull(),yticklabels=False,cbar=False,cmap='plasma')


# In[144]:


test.info()


# In[145]:


test.fillna(value=test['Fare'].mean(), inplace= True)


# In[146]:


predictions = model.predict(test)


# In[147]:


d = {'PassengerId': test['PassengerId'], 'Survived': predictions}
result = pd.DataFrame(d)


# In[148]:


result.to_csv('submission.csv', index= False)


# In[149]:


result.head()


# In[ ]:




