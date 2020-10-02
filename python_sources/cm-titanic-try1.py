#!/usr/bin/env python
# coding: utf-8

# **Test with Logistic Regression, 
# part1: data analysis and visualization
# part2: data clean up
# part3: training data splitting and testing
# part4: prediction with testing data**
# >  ***Don't know why the accuracy vs. gender_submission=95%, but in real submission is only 77%

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# **Data Analysis**

# In[ ]:


train = pd.read_csv('../input/train.csv')
train.head()


# In[ ]:


sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[ ]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Sex',data=train,palette='RdBu_r')


# In[ ]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Pclass',data=train,palette='rainbow')


# In[ ]:


sns.distplot(train['Age'].dropna(),kde=False,color='darkred',bins=30)


# In[ ]:


sns.countplot(x='SibSp',data=train)


# In[ ]:


train['Fare'].hist(color='green',bins=40,figsize=(8,4))


# In[ ]:


import cufflinks as cf
cf.go_offline()


# In[ ]:


train['Fare'].iplot(kind='hist',bins=30,color='green')


# **Data Cleaning**

# In[ ]:


plt.figure(figsize=(12, 7))
sns.boxplot(x='Pclass',y='Age',data=train,palette='winter')


# In[ ]:


Pclass1_mean=train[train['Pclass']==1].mean()['Age']
Pclass1_mean


# In[ ]:


Pclass2_mean=train[train['Pclass']==2].mean()['Age']
Pclass2_mean


# In[ ]:


Pclass3_mean=train[train['Pclass']==3].mean()['Age']
Pclass3_mean


# In[ ]:


def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):

        if Pclass == 1:
            return Pclass1_mean

        elif Pclass == 2:
            return Pclass2_mean

        else:
            return Pclass3_mean

    else:
        return Age


# In[ ]:


train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)


# In[ ]:


sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[ ]:


train.drop('Cabin',axis=1,inplace=True)


# In[ ]:


sns.countplot(x='Embarked',data=train,palette='RdBu_r')


# In[ ]:


train['Embarked'].fillna('S', inplace=True)


# In[ ]:


sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# ****Converting categorical features******

# In[ ]:


train.info()


# In[ ]:


sex = pd.get_dummies(train['Sex'],drop_first=True)
embark = pd.get_dummies(train['Embarked'],drop_first=True)


# In[ ]:


train.drop(['PassengerId','Sex','Embarked','Name','Ticket'],axis=1,inplace=True)


# In[ ]:


train = pd.concat([train,sex,embark],axis=1)


# In[ ]:


train.head()


# **LogisticRegression**

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(train.drop('Survived',axis=1), 
                                                    train['Survived'], test_size=0.30, 
                                                    random_state=101)


# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)


# In[ ]:


predictions = logmodel.predict(X_test)


# In[ ]:


from sklearn.metrics import classification_report


# In[ ]:


print (classification_report(y_test, predictions))


# In[ ]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, predictions)


# **Evaluate Test Datset**

# In[ ]:


evaluate = pd.read_csv('../input/test.csv')
evaluate.head()


# In[ ]:


sns.heatmap(evaluate.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[ ]:


evaluate['Age'] = evaluate[['Age','Pclass']].apply(impute_age,axis=1)


# In[ ]:


evaluate.drop('Cabin',axis=1,inplace=True)


# In[ ]:


Pclass1_fmean=evaluate[evaluate['Pclass']==1].mean()['Fare']
Pclass1_fmean


# In[ ]:


Pclass2_fmean=evaluate[evaluate['Pclass']==2].mean()['Fare']
Pclass2_fmean


# In[ ]:


Pclass3_fmean=evaluate[evaluate['Pclass']==3].mean()['Fare']
Pclass3_fmean


# In[ ]:


def impute_fare(cols):
    Fare = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Fare):

        if Pclass == 1:
            return Pclass1_fmean

        elif Pclass == 2:
            return Pclass2_fmean

        else:
            return Pclass3_fmean

    else:
        return Fare


# In[ ]:


evaluate['Fare'] = evaluate[['Fare','Pclass']].apply(impute_fare,axis=1)


# In[ ]:


sns.heatmap(evaluate.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[ ]:


df_result=pd.DataFrame(evaluate['PassengerId'],columns=['PassengerId'])
df_result.head()


# In[ ]:


sex = pd.get_dummies(evaluate['Sex'],drop_first=True)
embark = pd.get_dummies(evaluate['Embarked'],drop_first=True)


# In[ ]:


evaluate.drop(['PassengerId','Sex','Embarked','Name','Ticket'],axis=1,inplace=True)


# In[ ]:


evaluate = pd.concat([evaluate,sex,embark],axis=1)


# In[ ]:


evaluate.head()


# In[ ]:


evaluation_result = logmodel.predict(evaluate)


# In[ ]:


evaluation_result


# In[ ]:


df_result['Survived'] = pd.Series(evaluation_result, index=df_result.index)
df_result.head()


# In[ ]:


gender_submission = pd.read_csv('../input/gender_submission.csv')
gender_submission.head()


# In[ ]:


print (classification_report(gender_submission['Survived'] , df_result['Survived']))


# In[ ]:


# output data for submission in Kaggle
# df_result.to_csv('result.csv',index=False)


# In[ ]:





# In[ ]:





# In[ ]:




