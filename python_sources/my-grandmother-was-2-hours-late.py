#!/usr/bin/env python
# coding: utf-8

# **My Grandmother Was 2 Hours Late to the Titanic**
# 
# 
# In 1912, my grandmother traveled from a small town in Hungary to South Hampton, England, in hopes of beginning a new life in America. With her mother and sister in toe, they made the trek to be united with their father in The States. 
# 
# Upon arrival, they were dissapointed learn that their vessel of freedom had left without them, a mere two hours before. 
# I can only image their relief when they learned of their good fortune to nearly miss the most famous ship distaster in history.
# 
# I had always heard this story by family members, but much to my suprise, elements of this story actually exist in her obitutary. [Check it out!](https://www.findagrave.com/memorial/64694097/mary-barilich)
# 
# 
# 
# **Introduction:**
# 
# After learning a bit about python, I've decided to tip my toes in machine learning. I beginning with a very simple dataset to grasp the fundamentals of Data Science and attempt to answer a very interesting and person question...
# 
# **Had my grandmother sailed on the Titanic, would she have survived? Would I have ever existed!??**
# 
# Let see what we can do. 
# 
# *(Guidance on this mini-project comes from Data Science and Machine Learning with Python by Jose Portilla)*
# 
# 

# **First, lets load our data set and necessary package for analysis**

# In[ ]:



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df_test = pd.read_csv("../input/titanic/test.csv")
df_train = pd.read_csv('../input/titanic/train.csv')


# **Exploratory Data Analysis and Cleaning:**

# In[ ]:


sns.countplot(x='Survived',hue='Sex',data=df_train)


# From the graph, we can see that the majority of females surived and disporportionally to their gender. 

# In[ ]:


sns.countplot(x='Survived',hue='Pclass',data=df_train)


# You better be a wealthy, classy lady though...

# In[ ]:


sns.distplot(df_train['Age'].dropna(),bins=30,kde=False)


# .info() gives us some a great overview of the data types we have in our columns and shows were data may be missing. 

# In[ ]:


df_train.info()


# Recall, that the SibSP field denotes siblings. 

# In[ ]:


df_train['Fare'].hist(bins=100,figsize=(10,10))


# And some paid over 500 dollars for a ticket! Did they survive?! 

# In[ ]:


df_train[df_train['Fare']>500]


# Indeed. 

# In[ ]:


#adjusting for inflation, one dollar back then is equal to $25.89/
#lets ajust for inflation


# In[ ]:


df_train[df_train['Fare']>500]['Fare']*25.89


# BALLIN!

# So, we noticed from .info() that we were missing a good chunk of our age data. The course that I followed suggested taking the average of age by class and imputing (filling in) those values to remove the nulls. And, there's a good portion of the cabin info missing, so we'll drop that column all together. 

# In[ ]:


age_means = pd.pivot_table(df_train,values = 'Age',index= 'Pclass',aggfunc='mean')
age_means


# In[ ]:


def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):
        
        if Pclass ==1:
            return 38
        elif Pclass ==2:
            return 30
        else:
            return 25
    else:
        return Age


# In[ ]:


df_train['Age'] = df_train[['Age','Pclass']].apply(impute_age,axis=1)


# In[ ]:


df_train.drop('Cabin',axis=1,inplace=True)


# To run logistic regression, we'll need to turn Male and Female into 1s and 0s, along with the departure location, because there are three categories. 2 columns are used to fix this! 

# In[ ]:


sex = pd.get_dummies(df_train['Sex'],drop_first=True)


# In[ ]:


embark = pd.get_dummies(df_train['Embarked'],drop_first=True)


# In[ ]:


embark.head()


# In[ ]:



train = pd.concat([df_train,sex,embark],axis=1)


# In[ ]:


train.head()


# In[ ]:


train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)


# In[ ]:


train.drop(['PassengerId'],axis=1,inplace=True)


# **Logistic Regression:**
# 
# In this section, we'll take all of our explanatory variables and our predicted values and fit them to the logistic model. 

# In[ ]:


X = train.drop('Survived',axis=1)
y = train['Survived']


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=50)


# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


logmodel = LogisticRegression()


# In[ ]:


logmodel.fit(X_train,y_train)


# In[ ]:


predictions = logmodel.predict(X_test)


# Here are the predictions for the test sample of our model. 

# In[ ]:


predictions


# And here is the acrruacy of the logistic model to the actual variable. 

# In[ ]:


from sklearn.metrics import classification_report


# In[ ]:


print(classification_report(y_test,predictions))


# 80% isn't bad. 

# In[ ]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score


# In[ ]:


confusion_matrix(y_test,predictions)


# In[ ]:


precision_score(y_test,predictions)


# Let's round and call it 80%!
# 
# I was interested to find what the coefficents were for each of the X variables and the intercept. 

# In[ ]:


pd.DataFrame(index= list(logmodel.coef_),data = list(X_train.columns))


# In[ ]:


logmodel.intercept_


# **Would my grandmother have survived?**
# 
# Lets see based on the predictions of the model. 

# In[ ]:


gma  = pd.read_csv("../input/grandmas-attributes/grandma.csv")


# From what I know from her obituary and other information from my data family, I make the following inputs:
# 
# * **Class = 3** (They were quite poor)
# * **Age = 13**
# * **Siblings = One sister**
# * **Parents or Children = Her mother**
# * **Fare = 3**  This is an estimate from information obtained online. 
# * **Male = 0**
# * **Departure = South Hampton**

# In[ ]:


gma.head()


# 
# 

# In[ ]:


gma_surv = logmodel.predict(gma.drop('Adj. Fare',axis=1))


# Does she survive!?

# In[ ]:


print(gma_surv)


# **Nice!**

# In[ ]:


logmodel.predict_proba(gma.drop('Adj. Fare',axis=1))


# But the probabilites are 59-60%. I don't know about you, but with a 40% chance of rain, I bring an umbrella. 
# 
# I'm not sure if the data are realistic, but this was cool first experiment to see just how likely my ancestor would be around had she been on that big boat. 
# 
# 
