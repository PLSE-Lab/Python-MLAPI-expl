#!/usr/bin/env python
# coding: utf-8

# ## Let's first important data

# In[1]:


import numpy as np 
import pandas as pd 

import os
print(os.listdir("../input"))


# In[2]:


dataset = pd.read_csv('../input/train.csv')


# ## Let's check the data

# In[3]:


dataset.head()


# In[4]:


#So we need to first study of our data

#So first of all
# 1 - PassengerId - Column does not depend on survive   
# 2 - Pclacss - it's needed for prediction      [--Select--]
# 4 - Name - it's not useful for our prediction
# 5 - Sex - Male/ Female depends on survived because female always get a first chance   [--Select--]
# 6 - Age - Age depends on survivde  [--Select--]
# 7 - SibSp - Having siblings/spouse depends on survived  [--Select--]
# 8 - Parch - Number of childs depends on survived  [--Select--]
# 9 - Ticket - Ticket not create impact on survived
#10 - Fare - Fare create impact om survived because who have a costly tickets ,that person have more chance to get first in lifeboat  [--Select--]
#11 - Cabin - Cabin have more null values and its not create any impact on survived
#12 - Embarked - it's create impact on survived  [--Select--]


# ## Now visualize the data

# In[9]:


##Let's create funcition for barplot
def bar_chart(feature):
    survived = dataset[dataset['Survived']==1][feature].value_counts()
    dead = dataset[dataset['Survived']==0][feature].value_counts()
    df = pd.DataFrame([survived,dead])
    df.index = ['Survived','Dead']
    df.plot(kind='bar',stacked=True, figsize=(15,7))
    
bar_chart('Sex')


# * **According to Barchart females survived more than male**

# In[10]:


##Now let's see a barplot of Pclass
bar_chart('Pclass')


# **The Chart confirms 1st class people more survivied than other classes**
# 
# **The Chart confirms 3rd class people more dead than other classes**

# In[11]:


##Let's see for SibSp
bar_chart('SibSp')


# **The Chart confirms a person aboarded with more than 2 siblings or spouse are survived**
# 
# **The Chart confirms a person aboarded without siblings or spouse are dead**

# In[12]:


##Let's see for Parch
bar_chart('Parch')


# **The Chart confirms a person aboarded with more than 2 parents or children are survived**
# 
# **The Chart confirms a person aboarded alone are dead**

# In[13]:


##Let's plot the Embarked
bar_chart('Embarked')


# **The Chart confirms a person aboarded from C slightly more likely survived**
# 
# **The Chart confirms a person aboarded from Q more likely dead**
# 
# **The Chart confirms a person aboarded from S more likely dead**

# ## Let's devide data into X & Y

# In[14]:


##Now let's make a list of our features matrix list
features= [ 'Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']
##Let's devide in X and Y
x = dataset[features]
y = dataset['Survived']


# In[15]:


x.head()


# ## Now check the null values & fill it

# In[16]:


x.isnull().sum()


# In[17]:


##Now fill the null values
x['Age'] = x['Age'].fillna(x['Age'].median())
x['Embarked']= x['Embarked'].fillna(x['Embarked'].value_counts().index[0])


# In[18]:


x.isnull().sum()


# ## Let's encode the categorical values

# In[19]:


###Now let's enocde categorical values 
from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
x['Sex'] = LE.fit_transform(x['Sex'])
x['Embarked'] = LE.fit_transform(x['Embarked'])


# ## Check the x

# In[20]:


##let's see x
print(x)


# # Now the check NULL values in Y

# In[21]:


##Now let's check null values in Y
y.isnull().sum()


# # Now we have to split the data into training & testing

# In[22]:


##Now everything is ok 
##Now let's Split the Data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.1,random_state =0)


# # Now we have to create a Machine Learning Model 

# In[23]:


##Now we fit our model
from xgboost import XGBClassifier
classifier = XGBClassifier(colsample_bylevel= 0.9,
                    colsample_bytree = 0.8, 
                    gamma=0.99,
                    max_depth= 5,
                    min_child_weight= 1,
                    n_estimators= 10,
                    nthread= 4,
                    random_state= 2,
                    silent= True)
classifier.fit(x_train,y_train)
classifier.score(x_test,y_test)


# **That's good we got a 0.85 accuracy**

# ## Now we have to get a Test data in Dataframe

# In[24]:


##Now take the test data for prediction
test_data = pd.read_csv('../input/test.csv')
test_x = test_data[features]


# In[25]:


##test_x is our testing data which we will give to our model for prediction
test_x.head()


# # Let's check NULL values in Test dataset

# In[26]:


test_x.isnull().sum()


# # Let's fill NULL values

# In[27]:


##Let's fill values
test_x['Age'] = test_x['Age'].fillna(test_x['Age'].median())
test_x['Fare'] = test_x['Fare'].fillna(test_x['Fare'].median())


# In[28]:


test_x.isnull().sum()


# # Let's encode a categorical value in Test Data

# In[29]:


##Let's enocde categorical values
test_x['Sex'] = LE.fit_transform(test_x['Sex'])
test_x['Embarked'] = LE.fit_transform(test_x['Embarked'])


# In[30]:


test_x.head()


# # Let's predict a Testing data with our XGB Model

# In[31]:


##Now we predict the values
prediction = classifier.predict(test_x)


# # Let' convert our prediction int Submission.csv

# In[33]:


##Now according to rules we have to store a prediction in csv file
output = pd.DataFrame({'PassengerId': test_data.PassengerId,'Survived': prediction})
output.to_csv('submission.csv', index=False)
output.head()
##Submission.csv is a file which we have to submit in a competition

