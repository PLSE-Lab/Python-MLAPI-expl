#!/usr/bin/env python
# coding: utf-8

# # Importing relevant libraries

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# In[ ]:


train.head(2)


# In[ ]:


test.head(2)


# ## Exploratory data analysis

# In[ ]:


data = train.copy()


# What are the attributes present in the dataframe ?

# In[ ]:


data.head()


# How many null values are there in each column

# In[ ]:


data.isnull().sum()


# ###### Creating a barplot of Survived vs Sex with Pclass as hue

# In[ ]:


sns.barplot(x='Sex',y='Survived',data=data,hue='Pclass')


# You can see in the above barplot how a persons survival depends on sex and the class in which he/she is travelling

# ##### Now to visualize how having a family member onboard affects the chances of survival
# Creating a new column to get count of family members onboard

# In[ ]:


data['FmemOnboard'] = data['SibSp'] + data['Parch']


# In[ ]:


sns.barplot(x='FmemOnboard',y='Survived',data=data)


# Here you can see passengers travelling alone have less chances of surviving

# Distribution of age among passengers

# In[ ]:


sns.distplot(data[(data['Age'].isnull()==False)]['Age'])


# Number of Males and females onboard

# In[ ]:


sns.countplot(x='Sex',data=data)


# Survived males and females

# In[ ]:


sns.countplot(x='Sex',data=data[data['Survived']==1])


# ##### The chances of survival of females are significantly more than that of males

# Is place embarked significant ?

# In[ ]:


sns.barplot(x=data['Embarked'],y=data['Survived'])


# Here you can see among survived passenger more have embarked at C 

# ## Data preprocessing
# For being able to preprocess the train and test data simultanously, I have merged them info one dataframe named titanic

# In[ ]:


passengerId = test['PassengerId']
titanic = train.append(test,ignore_index=True)
df = titanic.copy()


# In[ ]:


df.info()


# Our dataframe has lot of missing values, we have to remove missing values or fill suitable values in them before feeding the data into Logistic regression model

# storing number of examples of train and test

# In[ ]:


train_count = train.shape[0]
test_count = test.shape[0]


# In[ ]:


df.head()


# Filling two missing embarked values with most frequent embarked value

# In[ ]:


embarked_freq = df['Embarked'].value_counts()


# In[ ]:


most_freq_embarked = embarked_freq[embarked_freq==embarked_freq.max()].index[0]


# In[ ]:


most_freq_embarked


# In[ ]:


df['Embarked'] = df['Embarked'].fillna(most_freq_embarked)


# In[ ]:


df.info()


# Filling missing values of Fare with the respective Pclass averages

# In[ ]:


x = df.groupby(['Pclass'])


# In[ ]:


x.mean()


# In[ ]:


class_of_missing_fare = df[df['Fare'].isnull()]['Pclass'].iloc[0]
class_of_missing_fare


# In[ ]:


x.mean()


# In[ ]:


to_fill_in_missing_fare = x.mean().loc[class_of_missing_fare,'Fare']
to_fill_in_missing_fare


# In[ ]:


df['Fare'].fillna(to_fill_in_missing_fare,inplace=True)


# In[ ]:


df.info()


# We have 891 train examples have have survived value for each of them
# Therefore we don't have to do anything with Survived column

# We will be filling missing age values with average age of that Pclass and Sex

# In[ ]:


age_dict = dict(df.groupby(['Pclass','Sex']).mean()['Age'])


# In[ ]:


age_dict[(1,'female')]


# In[ ]:


age_dict


# In[ ]:


type(list(age_dict.keys())[0])


# In[ ]:


type((df.loc[0]['Pclass'],df.loc[0]['Sex']))


# In[ ]:


def fill_age(row):
    if np.isnan(row['Age']):
        tp = (row['Pclass'],row['Sex'])
        return age_dict[tp]
    else:
        return row['Age']


# In[ ]:


np.isnan(df.iloc[0]['Age'])


# In[ ]:


df['Age'] = df.apply(fill_age,axis=1)


# In[ ]:


df.info()


# In[ ]:


df['Age'] = df['Age'].apply(round)


# In[ ]:


df.head()


# In[ ]:


df['Sex'] = df['Sex'].map({'male':0,'female':1})


# In[ ]:


df.head()


# In[ ]:


df['Cabin'].nunique()


# In[ ]:


type(df['Cabin'][0])


# In[ ]:


df['Cabin'].value_counts()


# Filling missing cabin values with 'Unknown'

# In[ ]:


df['Cabin'].apply(lambda x : 'Unknown' if type(x)==float else x).astype(str)


# In[ ]:


df['Cabin'] = (df['Cabin'].apply(lambda x : 'Unknown' if type(x)==float else x).astype(str)).apply(lambda x : x.split()[0][0])


# In[ ]:


df.head()


# Dropping ticket number column

# In[ ]:


df.drop(['Ticket'],axis=1,inplace=True)


# In[ ]:


df.head()


# In[ ]:


df.info()


# Creating dummies for each categorical feature

# In[ ]:


cabin_dummies = pd.get_dummies(df['Cabin'])


# In[ ]:


cabin_dummies.drop('U',axis=1,inplace=True)


# In[ ]:


embarked_dummies = pd.get_dummies(df['Embarked'])


# In[ ]:


embarked_dummies.drop('S',axis=1,inplace=True)


# In[ ]:


pclass_dummies = pd.get_dummies(df['Pclass'])
pclass_dummies


# In[ ]:


pclass_dummies.drop(3,axis=1,inplace=True)


# In[ ]:


pclass_dummies.head()


# In[ ]:


cabin_dummies.columns = 'Cabin ' + cabin_dummies.columns


# In[ ]:


cabin_dummies.head()


# In[ ]:


embarked_dummies.columns = 'Embark ' + embarked_dummies.columns


# In[ ]:


embarked_dummies.head()


# In[ ]:


(list(pclass_dummies.columns))


# In[ ]:


pclass_dummies.columns = ['Pclass 1','Pclass 2']


# In[ ]:


pclass_dummies.head()


# In[ ]:


df.head()


# Dropping unnecessary columns

# In[ ]:


df.drop(['Cabin','Embarked'],axis=1,inplace=True)


# In[ ]:


df.head()


# In[ ]:


df.drop(['PassengerId','Name','Pclass'],axis=1,inplace=True)


# In[ ]:


df.head()


# Concatening df with dummy values

# In[ ]:


df_dummies = pd.concat([df, pclass_dummies, cabin_dummies, embarked_dummies], axis=1)


# In[ ]:


df_dummies.head()


# Recollecting train and test Dataframes

# In[ ]:


train = df_dummies.iloc[:train_count]
test = df_dummies.iloc[train_count:]


# In[ ]:


train.info()


# In[ ]:


test.info()


# Obtain X and y i.e. input features and label

# In[ ]:


X = train.drop('Survived',axis=1)
y = train['Survived'].astype(int)


# In[ ]:


X_to_predict = test.drop('Survived',axis=1)
X_to_predict.head()


# In[ ]:


X_to_predict.shape[0]


# In[ ]:


X.head()


# In[ ]:


y.head()


# ### Importing the model libraries and training the model

# #### Splitting the data

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)


# #### Training on logistic regression model

# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


logreg = LogisticRegression()


# In[ ]:


logreg.fit(X_train,y_train)


# In[ ]:


from sklearn.metrics import classification_report,confusion_matrix


# In[ ]:


print('For logistic regression model')
print('Confusion Matrix')
print(confusion_matrix(y_test,logreg.predict(X_test)))


# In[ ]:


print('For logistic regression model')
print('Classification Report')
print(classification_report(y_test,logreg.predict(X_test)))


# In[ ]:


logreg_predictions = logreg.predict(X_to_predict)


# #### Training on random forest classifier

# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


clf = RandomForestClassifier(n_estimators=4)


# In[ ]:


clf.fit(X_train,y_train)


# In[ ]:


print('For decision tree classifier')
print('Confusion Matrix')
print(confusion_matrix(y_test,clf.predict(X_test)))


# In[ ]:


print('For decision tree classifier')
print('Classification Report')
print(classification_report(y_test,clf.predict(X_test)))


# In[ ]:


kaggle = pd.DataFrame({'PassengerId':passengerId,'Survived':logreg.predict(X_to_predict)})


# In[ ]:


kaggle.to_csv('submit.csv',index=False)


# In[ ]:




