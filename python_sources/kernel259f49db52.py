#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


data=pd.read_csv('../input/titanic-machine-learning-from-disaster/train.csv')


# In[ ]:


data.head()


# In[ ]:


data.isnull()


# In[ ]:


sns.heatmap(data.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[ ]:


sns.set_style('whitegrid')


# In[ ]:


sns.countplot(x='Survived',hue='Sex',data=data)


# In[ ]:


import cufflinks as cf
cf.go_offline()
data['Fare'].iplot(kind='hist',bins=50)


# In[ ]:


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


data['Age']=data[['Age','Pclass']].apply(impute_age,axis=1)


# In[ ]:


sns.heatmap(data.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[ ]:


data.drop('Cabin',axis=1,inplace=True)


# In[ ]:


sns.heatmap(data.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[ ]:


data.drop(['Sex','Embarked','Name','Ticket'],inplace=True,axis=1)


# In[ ]:


data.head()


# In[ ]:


x=data.drop('Survived',axis=1)


# In[ ]:


y=data['Survived']


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.30,random_state=101)


# In[ ]:


from sklearn.tree import DecisionTreeClassifier


# In[ ]:


logmodel=DecisionTreeClassifier()


# In[ ]:


logmodel.fit(X_train,y_train)


# In[ ]:


prediction=logmodel.predict(X_test)


# In[ ]:


from sklearn.metrics import classification_report,confusion_matrix


# In[ ]:


print(confusion_matrix(y_test,prediction))


# In[ ]:


print(classification_report(y_test,prediction))


# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


result=RandomForestClassifier()


# In[ ]:


result.fit(X_train,y_train)


# In[ ]:


result_pred=result.predict(X_test)


# In[ ]:


print(confusion_matrix(y_test,result_pred))


# In[ ]:


print(classification_report(y_test,result_pred))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


submission = pd.DataFrame({'PassengerId':X_test['PassengerId'],'Survived':result_pred})

#Visualize the first 5 rows
submission.head()


# In[ ]:


filename = 'Titanic Predictions 1.csv'

submission.to_csv(filename,index=False)

print('Saved file: ' + filename)


# In[ ]:





# In[ ]:





# 
