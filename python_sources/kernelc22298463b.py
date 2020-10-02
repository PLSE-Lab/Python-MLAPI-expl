#!/usr/bin/env python
# coding: utf-8

# # Logistic Regression on Dependent(Survived) Titanic Dataset

# In[ ]:


import numpy as np
import pandas as pd

mydata=pd.read_csv("../input/titanic_train.csv")

# Any results you write to the current directory are saved as output.


# In[ ]:


#top 5 will be reflects and use .tail to lower data and .loc with index numbers for consideration.(basics)
mydata.head()


# In[ ]:


#info gives the missing values in our data
mydata.info()


# In[ ]:


#if required to know the True values
mydata.isnull()


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sb

#null values present in our data using data visuals
sb.heatmap(mydata.isnull())


# In[ ]:


#percentage of persons survived in disaster
sb.countplot(x='Survived',data=mydata)


# In[ ]:


#defining in sex
sb.countplot(x='Survived',hue='Sex',data=mydata)


# In[ ]:


#classifying in Pclass(L,M,H)
sb.countplot(x='Survived',hue='Pclass',data=mydata)


# In[ ]:


#defining pclass on reference with age for suvival.
plt.figure(figsize=(12,7))
sb.boxplot(x='Pclass',y='Age',data=mydata,palette='autumn')


# In[ ]:


#imputing age on null values
def impute_age(cols):
    Age=cols[0]
    Pclass=cols[1]
    
    if pd.isnull(Age):
        if Pclass==1:
            return 37
        elif Pclass==2:
            return 29
        else:
            return 24
    else:
        return Age


# In[ ]:


mydata['Age']=mydata[['Age','Pclass']].apply(impute_age,axis=1)


# In[ ]:


#age null values filled using decision making
sb.heatmap(mydata.isnull())


# In[ ]:


#dropping cabin 
mydata.drop('Cabin',axis=1,inplace=True)


# In[ ]:


mydata.info()


# In[ ]:


#Dropping variables which is not required for study
sex=pd.get_dummies(mydata['Sex'],drop_first=True)
embark=pd.get_dummies(mydata['Embarked'],drop_first=True)


# In[ ]:


mydata.drop(['PassengerId','Sex','Embarked','Name','Ticket'],axis=1,inplace=True)


# In[ ]:


#concantenate to remove the dummies
mydata=pd.concat([mydata,sex,embark],axis=1)


# In[ ]:


mydata.head()


# In[ ]:


#train_test to prediction
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(mydata.drop('Survived',axis=1),mydata['Survived'],test_size=0.20,random_state=101)


# In[ ]:


from sklearn.linear_model import LogisticRegression
logmodel=LogisticRegression(solver='liblinear')
logmodel.fit(x_train,y_train)
predictions=logmodel.predict(x_test)
print(predictions[5])
y_test.head(5)
#package warning


# In[ ]:


logmodel.fit(x_train,y_train)


# In[ ]:


logmodel.score(x_test,y_test)


# In[ ]:


predictions


# # END
