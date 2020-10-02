#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Machine Learning to predict titanic survivors
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import sklearn.metrics

#Suppress all warnings (risky business!)
import warnings
warnings.filterwarnings("ignore")

#getting system path
import os
path1 = os.getcwd()
train_path = path1.rsplit('\\',1)[0]+'\\Original_Data\\train.csv'
test_path = path1.rsplit('\\',1)[0]+'\\Original_Data\\test.csv'


# In[ ]:


#reading csv
df_raw = pd.read_csv("../input/titanic/train.csv")
df_test_raw = pd.read_csv("../input/titanic/test.csv")


# In[ ]:


# paramters
test_train_split_size = 0.2


# ## Data cleaning

# In[ ]:


df1 = df_raw.copy()
df1.head()


# Looks like Age & Cabin has missing values!

# In[ ]:


df1.isna().sum()


# Creating functions to help with data cleaning - since we need to clean the train data as well as the test data

# In[ ]:


def get_unique_list(df1,col_name,threshold):
    #function to get list of unique items in a columns that has value count above threshold percentage
    unique_list = df1[col_name].value_counts().index.tolist()
    for index,values in df1[col_name].value_counts().items():
        if values<=(df1.shape[0])*threshold:
            unique_list.remove(index)
            
    return(unique_list)


# In[ ]:


def data_cleaner(df1):
   df1.Age = df1[['Pclass','Sex','Age']].groupby(['Sex','Pclass']).transform(lambda x: x.fillna(x.mean()))
   df1.Fare = df1[['Pclass','Fare']].groupby(['Pclass']).transform(lambda x: x.fillna(x.mean()))
   df1.Embarked.fillna(method='ffill',inplace=True)
   
   df1['Has_cabin'] = np.where(df1.Cabin.notnull(),1,0)
   
   #code to get title from name
   liste = df1.Name.str.split(',').tolist()
   liste = [e[1] for e in liste]
   liste = pd.Series(liste).str.split(".")
   liste = [e[0] for e in liste]
   df1['Prefix'] = liste

   #code to merge similar titles for instance: 'Mlle.' is 'Miss.' in French
   df1['Prefix'] = df1['Prefix'].replace({' Mlle':' Miss',' Mme':' Mrs',' Ms':' Miss'})

   unique_list = get_unique_list(df1,'Prefix',0.03)
   df1['Prefix'][~(df1['Prefix'].isin(unique_list))] = 'Rare'
   
   #adding a new column - 'relatives'
   df1['relatives'] = df1['SibSp'] + df1['Parch']
   #new column - 'Alone'
   df1['Alone'] = np.where(df1.relatives==0,1,0)
   df1.drop(['PassengerId','Name','Ticket','SibSp','Cabin'],axis=1,inplace=True)

   df1 = pd.get_dummies(df1,columns=['Sex','Pclass','Embarked','Prefix'],drop_first=True)
   
   
   return(df1)


# In[ ]:


df1 = df_raw.copy()
df_test = df_test_raw.copy()

df1 = data_cleaner(df1)
df1_test = data_cleaner(df_test)


# In[ ]:


df1.head()


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(df1.drop(['Survived'],axis=1), df1.Survived, test_size=test_train_split_size)


# # Prediction

# #### Logistic Regression

# Works perfect, Has a good CV accuracy score (0.834) and a similar test set score.

# In[ ]:


lr = LogisticRegression()
params = {'penalty':['l1','l2'],'C':[0.001,0.01,0.1,1,10,100,1000]}

gslr = GridSearchCV(lr,params,cv=10,scoring='accuracy')
gslr.fit(X_train,y_train)

print("best parameter = ", gslr.best_params_)
print("best score = ",gslr.best_score_)

print("\naccuracy score on test set:", sklearn.metrics.accuracy_score(y_test,gslr.predict(X_test)))


# In[ ]:


#shows importance of each feature
for i in zip(gslr.best_estimator_.coef_[0].tolist(),X_train.columns.tolist()):
    print(i)


# Submitting the Logistic Regression score!

# In[ ]:


# path1 = os.getcwd()+'\\lr_v10.csv'
pd.DataFrame(data={'PassengerId':df_test_raw.PassengerId,'Survived':gslr.predict(df1_test).astype(int)}).to_csv('titanic_LR',index=False)


# Got a accuracy of 0.789 on Kaggle!

# ![image.png](attachment:image.png)
