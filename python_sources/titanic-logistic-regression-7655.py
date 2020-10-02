#!/usr/bin/env python
# coding: utf-8

# **Overview: **This Ultimate goal of this kernal is designed to be a guide for analytical method selection; rational; and include graphic,  programming and explanation of each approach. <br/>
# <b>Using the Notebook</b> simply edit hashtags to activate and deactivate visual functions.
# 
# **Analytical Libraries:** Pandas, Numpy, Matplotlib, and Seaborn <br/>
# **Research Sources:** [Introduction to Statistical Learning](http://www-bcf.usc.edu/~gareth/ISL/ISLR%20First%20Printing.pdf)<br/>
#  UDemy.com:  [Python for Data Science and Machine Learning BootCamp](https://www.udemy.com/python-for-data-science-and-machine-learning-bootcamp/)<br/>
#  Seaborn: [Color Palettes](http://seaborn.pydata.org/tutorial/color_palettes.html)
#  
#  
#  
# 
# <b>Exploring the data:</b><br>
# (1) heatmap null values.<br/>
# Findings: Cabin has the largest number of omitted values and should be removed as a feature
# 
# (2) Countplot<br/>
# Findings: 3rd class had highest incident of deaths of the classes; Males had the highest incident of death among gender. 
# 
# Methods:<br/>
# **Logistic Regression:** - Use: Classification Problem<br/>
# Logistic regression (not to be confused with Linear Regression) could be used when the output is binary (e.g. Pass/Fail, Win/Lose, Healthy/Sick, Survived/Died); Said another way, the output is a Binary Dependent Variable and can have several input variables (or Independent variables). Alternatively, if the output had more than 2 results, you could use multi-nominal Logistic Regression (e.g. outcomes being poor, fair, good, great). In the case of Titanic, Logistic Regression could be well suited as survival is a binary output based on several input variables. <br/>
# 
# **Additional Notes**<br/>
# High Level; Logistic Regression is a Linear Regression that has been re-shaped into a Logistic (sigmoidal function) whereby the outputs will always reside between 0 and 1. Additionally, it would be incorrect to use a Linear Regression when the outputs are noncontinuous / discrete. 
# 
# Titanic Test results:
# Test 001 - Simple excel model whereby gender was used as the baseline; results .7655
# Test 002 - Logistic Regression using Gender and Age; results .7655. 
# Logistic Regression model used: 
# <br/> (i) Survival (Dependent Variable)<br/>(ii) Gender and (iiI) Age as (Independent Variables); 3 Coefficients (often referred to as b0, b1 and b2) that I arbitrarily set to .10 (solver adjusted); Eulers Number (2.718281828459045); Used to solve MLL (Maximum Log Likelyhood).
# 
# **Linear Regression**<br/>
# (Continuous output)
# 
# Search
# Dummy variable trap
# Co-Linearity
# 
# 

# In[24]:


#import the libraries & file(s)
import numpy as np
import pandas as pd
import seaborn as sms
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

#Import the file
data = pd.read_csv('../input/train.csv')


# In[25]:


#reading data contents.
#data.head()
#data.tail()

#function List
#data.isnull()
#data.info()
#data.drop('Cabin',axis=1,inplace=True)
#data.dropna(inplace=True) 
#pd.get_dummies()
#pd.get_dummies(data['Sex'],drop_first=True)
#pd.get_dummies(data['Embarked'], drop_first=True)
#pd.concat([data,sex,embark], axis=1)

plt.figure(figsize=(10,7))

##################################################################
#Exploring the data; single line commands
##################################################################

#sms.distplot(data['Age'].dropna(),kde=False,bins=25)

#Countplots
#sms.countplot(x='Survived', hue='Sex', data=data)
#sms.countplot(x='Survived', hue='Pclass', data=data)
#sms.countplot(x = "Sibsp", hue="sex", data=data)

#Jointplots
#sms.jointplot(x='Age', y='Fare', data=data, kind='reg')
#sms.jointplot(x='Age', y='Fare', data=data, kind='kde')

#sms.pairplot(data,)
#sns.heatmap(data.isnull(),yticklabels=False,cbar=False,cmap='plasma')

#BoxPlots

sms.boxplot(x='Pclass', y='Age',data=data)


# In[26]:


##################################################################
#Data Cleaning, Purifying, and Imputing
##################################################################
#Here we're going to impute the average age for each class using the boxplot above to determine the values for a function we will build below. 
#this script is credited to Jose Portilla and his training book, which I highly recommend Python for Datascience and Machine Learning BootCamp (linked above)

def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]

    if pd.isnull(Age):
        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 29
        else:
            return 24
    else:
        return Age


# In[29]:


#using the function above the ages, 37, 29 and 24 were imputed (i.e. replaced the null values in the titanic dataset)
data['Age'] = data[['Age','Pclass']].apply(impute_age,axis=1)


# In[31]:


#test that the function worked correctly. 
#drop rows that are missing values
#drop column Cabin
data.dropna(inplace=True)
data.drop('Cabin', axis=1, inplace=True)
sms.heatmap(data.isnull(),yticklabels=False,cbar=False,cmap='plasma')
#data.info()


# In[32]:


#################################################################
#converting data to usable format; Essentially we need to convert Categorical values Male & Female to (1 or 0) for computational purposes.
#################################################################
sex = pd.get_dummies(data['Sex'],drop_first=True)
embark = pd.get_dummies(data['Embarked'], drop_first=True)


# In[33]:


##adding the new binary results for sex and adding the departure points QS from the newly created variables above into the data dataframe.
data = pd.concat([data,sex,embark], axis=1)


# In[34]:


data.head()


# In[36]:


data.drop(['Sex','Embarked','Name','Ticket','PassengerId'], axis=1, inplace=True)
data.head()


# In[37]:


X = data.drop('Survived', axis=1)
y = data['Survived']


# In[38]:


from sklearn.cross_validation import train_test_split


# In[100]:


X_train, X_test, y_train, y_test, = train_test_split(X,y, test_size=0.30)


# In[101]:


from sklearn.linear_model import LogisticRegression


# In[102]:


logisticmodel = LogisticRegression()


# In[103]:


logisticmodel.fit(X_train,y_train)


# In[104]:


predictions = logisticmodel.predict(X_test)


# In[105]:


from sklearn.metrics import classification_report


# In[106]:


print(classification_report(y_test,predictions))


# In[110]:


X_train

