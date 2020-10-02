#!/usr/bin/env python
# coding: utf-8

# In[36]:


import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score,precision_score
from sklearn.linear_model import LogisticRegression


# In[37]:


s=pd.read_csv('../input/train.csv')


# In[38]:


s.head()


# In[39]:


s.info()
#More null values are in Cabin and some are in Age


# In[40]:


s['Survived'].value_counts()
# Total 342 Members are survived


# In[41]:


s.isnull().sum()
# There are total  687, 177 null values are in Age, Cabin respectively


# In[42]:



s['Pclass'].value_counts()


# In[43]:


#just see which class are more survived. wheather money matter when titanic is shrinking
s[(s['Pclass']==3) & (s['Survived']==1)]['Pclass'].count()


# In[44]:


s[(s['Pclass']==2) & (s['Survived']==1)]['Pclass'].count()


# In[45]:


#It seems Yes
s[(s['Pclass']==1) & (s['Survived']==1)]['Pclass'].count()


# In[46]:


# which Embarked people are more survived
s['Embarked'].value_counts()


# In[47]:


s[(s['Embarked']=='S') & (s['Survived']==1)]['Pclass'].count()


# In[48]:


s[(s['Embarked']=='C') & (s['Survived']==1)]['Pclass'].count()


# In[49]:


s[(s['Embarked']=='Q') & (s['Survived']==1)]['Pclass'].count()


# In[50]:


#handling na values
s['Age'].isnull().sum() #Total 177 values are null


# In[51]:


s['Age'].hist().plot()


# In[52]:


from scipy import stats
stats.mode(s['Age'])
#Just replacing null values with mode i.e 24


# In[ ]:





# I am writing a function to handle the data so that i can easily apply the same to test.csv also

# In[53]:


def Datacleaning(s):
    s["Age"]=s['Age'].fillna(24)
    # filling with 'S' as there are more no of peoples are survivied 
    s['Embarked']=s['Embarked'].fillna('S')
    #PID,Name,'Ticket' attributes could not help while titanic is shrinking
    # As we are keeping Pclass removing Cabin, Fare
    s=s.drop(['PassengerId','Name','Ticket','Cabin','Fare'],axis=1)
    s=pd.get_dummies(s)
    return s
    
    


# In[54]:


Feature=s.drop('Survived',axis=1)
Feature=Datacleaning(Feature)
Target=s['Survived']


# In[ ]:





# Applying logit algoritham

# In[55]:


l=LogisticRegression().fit(Feature,Target)


# In[56]:


#Predict with train data validate with test data
TEST=pd.read_csv('../input/test.csv')
passengerID=TEST['PassengerId']


# In[57]:


TEST.head()


# In[58]:


TEST=Datacleaning(TEST)


# In[59]:


#Precdiction
ypred=l.predict(TEST)


# In[60]:


submission_df=pd.DataFrame()


# In[61]:


submission_df['PassengerID']=pd.Series(passengerID)
submission_df['Survived']=pd.Series(ypred)


# In[62]:


submission_df.head()


# In[63]:


logit_data=submission_df.to_csv('LogitSurvive.csv')


# In[ ]:




