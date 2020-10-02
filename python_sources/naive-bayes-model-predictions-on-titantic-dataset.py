#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


train = pd.read_csv(r"../input/titanic/train.csv")


# In[ ]:


train


# In[ ]:


train.head()


# In[ ]:


sns.heatmap(train.isnull(),yticklabels = False, cbar = False, cmap = 'viridis')


# In[ ]:


sns.set_style('whitegrid')
sns.countplot(x ='Survived', data = train,palette='BuGn_r')


# The data-set contain the missing values mainly in 'Age' and 'Cabin' column and there is also categorical variables that we need to deal because ml model quality rely on that.So we need to do preprocess our dataset. Before that we little bit explore our dataset

# # Data Exploration

# In[ ]:


sns.set_style('whitegrid')
sns.countplot(x = train['Survived'],hue = train['Sex'], palette='BuGn_r')


# In[ ]:


sns.set_style('whitegrid')
sns.countplot(x = train['Pclass'],hue = train['Sex'], palette='BuGn_r')


# # Data Preparation

# ## 1. Handling Missing Values

# In[ ]:


#plt.figure(figsize = 10,5)
sns.violinplot( x= 'Pclass', y = 'Age',data = train)


# In[ ]:


median= train['Age'].median()
train['Age'].fillna(median, inplace=True)


# In[ ]:


median


# In[ ]:


train['Age'].isnull()


# In[ ]:


sns.heatmap(train.isnull(),yticklabels = False, cbar = False, cmap = 'viridis')


# In[ ]:


train.drop('Cabin', axis = 1, inplace = True)


# We will remove the 'Cabin' column as there Cabin column does'nt giving any information related to survival of people in the titanic ship accident 

# In[ ]:


train


# In[ ]:


sns.heatmap(train.isnull(),yticklabels = False, cbar = False, cmap = 'viridis')


# Now,there is no missing values in dataset

# ## 2. Handling categorical values

# In[ ]:


sex = pd.get_dummies(train['Sex'],drop_first=True)
embark = pd.get_dummies(train['Embarked'],drop_first=True)


# We'll need to convert categorical features to dummy variables. Because machine learning algorithm will not be able take directly take in those features as inputs.

# In[ ]:


train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)


# In[ ]:


train = pd.concat([train,sex,embark],axis=1)


# In[ ]:


train.head()


# # Train and Test Split

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(train.drop(['Survived'],axis=1), 
                                                 train['Survived'], test_size=0.25, 
                                                   random_state=101)


# # Training of Naive-Bayes Model

# In[ ]:


from sklearn.naive_bayes import GaussianNB


# In[ ]:


naivemodel = GaussianNB()


# In[ ]:


naivemodel = naivemodel.fit(X_train, y_train)


# In[ ]:


naivemodel


# # Model Evaluation

# In[ ]:


Prediction = naivemodel.predict(X_test)


# In[ ]:


from sklearn import metrics


# In[ ]:


from sklearn.metrics import classification_report,confusion_matrix


# In[ ]:


matrix = (confusion_matrix(y_test,Prediction))


# In[ ]:


matrix


# In[ ]:


print(classification_report(y_test,Prediction))


# In[ ]:


from sklearn import metrics


# In[ ]:


print("Accuracy:",metrics.accuracy_score(y_test,Prediction)*100)
print("Precision:",metrics.precision_score(y_test,Prediction)*100)
print("Recall:",metrics.recall_score(y_test,Prediction)*100)
print("F1 Score:",metrics.f1_score(y_test,Prediction)*100)


# In[ ]:


from sklearn.metrics import roc_curve, roc_auc_score


# In[ ]:


nv_auc= print('roc_auc_score for naive bayes: ', roc_auc_score(y_test, Prediction))


# In[ ]:


Prediction


# In[ ]:


submission = pd.DataFrame({'PassengerId':X_test['PassengerId'],'Survived':Prediction})


# In[ ]:


submission.head()


# In[ ]:


filename = 'Titanic Predictions by naive-bayes model.csv'


# In[ ]:


print('Saved file: ' + filename)


# In[ ]:




