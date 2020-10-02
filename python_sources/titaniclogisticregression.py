#!/usr/bin/env python
# coding: utf-8

# # Practicing Logistic Regression on the Titanic Dataset.
# # Goal: build a logistic regression model to predict survival on the Titanic.
# # Data Dictionary:
# - survival - Survival (0 = No; 1 = Yes)
# - class - Passenger Class (1 = 1st; 2 = 2nd; 3 = 3rd)
# - name - Name
# - sex - Sex
# - age - Age
# - sibsp - Number of Siblings/Spouses Aboard
# - parch - Number of Parents/Children Aboard
# - ticket - Ticket Number
# - fare - Passenger Fare
# - cabin - Cabin
# - embarked - Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)
# - boat - Lifeboat (if survived)
# - body - Body number (if did not survive and body was recovered)

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import seaborn as sns

pd.options.display.max_columns = 100
pd.options.display.max_rows = 100

import os
print(os.listdir("../input"))


# # Let's read and peek at a sample of the data.

# In[ ]:


df = pd.read_csv('../input/Titanic.csv')

df.head()


# In[ ]:


df.columns


# In[ ]:


df.shape


# In[ ]:


df.isnull().sum()


# # We need to decide what to do with these null values.

# # First let's look at Age.

# In[ ]:


# percent of missing "Age" 
print('Percent of missing "Age" records is %.2f%%' 
      %((df['Age'].isnull().sum()/df.shape[0])*100))


# In[ ]:


ax = df["Age"].hist(bins=15, density=True, stacked=True, color='teal', alpha=0.6)
df["Age"].plot(kind='density', color='teal')
ax.set(xlabel='Age')
plt.xlim(-10,85)
plt.show()


# # Since "Age" is (right) skewed, using the mean might give us biased results by filling in ages that are older than desired. To deal with this, we'll use the median to impute the missing values.

# In[ ]:


# mean age
print('The mean of "Age" is %.2f' %(df["Age"].mean(skipna=True)))
# median age
print('The median of "Age" is %.2f' %(df["Age"].median(skipna=True)))


# # There isn't much difference between mean and median...

# In[ ]:


df['Age'].fillna(df['Age'].median(), inplace=True)
df.isnull().sum()


# # Now let's look at Cabin.

# In[ ]:


# percent of missing "Cabin" 
print('Percent of missing "Cabin" records is %.2f%%' 
      %((df['Cabin'].isnull().sum()/df.shape[0])*100))


# # 77.1% is really high, so we won't use that feature for prediction.

# In[ ]:


df.drop(columns = ['Cabin'], inplace=True)
df.columns


# # Let's check embarked.

# In[ ]:


# percent of missing "Embarked" 
print('Percent of missing "Embarked" records is %.2f%%' 
      %((df['Embarked'].isnull().sum()/df.shape[0])*100))


# In[ ]:


print('Boarded passengers grouped by port of embarkation (C = Cherbourg, Q = Queenstown,S = Southampton):')
print(df['Embarked'].value_counts())
sns.countplot(x='Embarked', data=df, palette='Set2')
plt.show()


# # There are only two records missing, 0.22% of the data, so let's just drop those two rows.

# In[ ]:


df[df.Embarked.isnull()]


# In[ ]:


df.drop([61], inplace=True)
df.drop([829], inplace=True)
df[df.Embarked.isnull()]


# In[ ]:


df.isnull().sum()


# # Looks good! All null values are dropped or imputed. What kind of data do we have?

# In[ ]:


df.dtypes


# In[ ]:


df.columns


# # Describe the numeric data (all except Name, Sex, Ticket, and Embarked).

# In[ ]:


df.describe()


# # Let's break the data into test and train datasets before we explore further.

# In[ ]:


x = df.drop(columns='Survived')
y = df['Survived']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
print('x_train shape:', x_train.shape)
print('y_train shape', y_train.shape)
print('x_test shape:', x_test.shape)
print('y_test shape', y_test.shape)
print('percent in x_train:', x_train.shape[0]/(x_train.shape[0] + x_test.shape[0]))
print('percent in x_test:', x_test.shape[0]/(x_train.shape[0] + x_test.shape[0]))


# In[ ]:


x_train.head()


# In[ ]:


x_train.describe()


# In[ ]:


x_train.dtypes


# # Join the x_train and y_train to explore the data by survival.

# In[ ]:


train = x_train.join(y_train, how='outer')
train.head()


# # How does SEX affect survival?

# In[ ]:


survived_male = train[train['Sex']=='male']['Survived'].value_counts()
survived_female = train[train['Sex']=='female']['Survived'].value_counts()
survived_sex_df = pd.DataFrame([survived_male, survived_female])
# Note: column 0 = died, column 1 = survived
survived_sex_df['total'] = survived_sex_df[0] + survived_sex_df[1]
survived_sex_df.index = ['male','female']
survived_sex_df.rename(index=str,columns={1:'Survived',0:'Died'}, inplace=True)
print (survived_sex_df)
survived_sex_df.plot(kind='bar',label=['Survived','Died'], figsize=(15,8), color = ['r','b','y'])


# # How does AGE impact survival?
# ## Use a histogram in order to bin the ages.

# In[ ]:


total_age = train['Age'].value_counts()
figure = plt.figure(figsize=(15,8))
plt.hist([train[train['Survived']==1]['Age'], 
          train[train['Survived']==0]['Age']], color = ['b','r'],
          bins = 30,label = ['Survived','Dead'])
plt.xlabel('Age')
plt.ylabel('Number of passengers')
plt.legend()


# # Let's decrease the number of bins on Age.

# In[ ]:


figure = plt.figure(figsize=(15,8))
plt.hist([train[train['Survived']==1]['Age'],
          train[train['Survived']==0]['Age']], color = ['b','r'],
          bins = 10,label = ['Survived','Dead'])
plt.xlabel('Age')
plt.ylabel('Number of passengers')
plt.legend()


# # How does FARE impact survival?
# ## Again using a histogram to bin the fares.

# In[ ]:


figure = plt.figure(figsize=(15,8))
plt.hist([train[train['Survived']==1]['Fare'], 
          train[train['Survived']==0]['Fare']], color = ['b','r'],
          bins = 10,label = ['Survived','Dead'])
plt.xlabel('Fare')
plt.ylabel('Number of passengers')
plt.legend()


# # How does CLASS affect survival?

# In[ ]:


survived_1 = train[train['Pclass']==1]['Survived'].value_counts()
survived_2 = train[train['Pclass']==2]['Survived'].value_counts()
survived_3 = train[train['Pclass']==3]['Survived'].value_counts()
survived_df = pd.DataFrame([survived_1,survived_2,survived_3])
# Note: column 0 = died, column 1 = survived
survived_df['total']=survived_df[0] + survived_df[1]
survived_df.index = ['1st class','2nd class','3rd class']
survived_df.rename(index=str,columns={1:'Survived',0:'Died'}, inplace=True)
print (survived_df)
survived_df.plot(kind='bar',label=['Survived','Died'], figsize=(15,8), color = ['r','b','y'])


# # Does their point of EMBARKMENT impact survival?

# In[ ]:


survived_by_embark1 = train[train['Embarked']=='S']['Survived'].value_counts()
survived_by_embark2 = train[train['Embarked']=='C']['Survived'].value_counts()
survived_by_embark3 = train[train['Embarked']=='Q']['Survived'].value_counts()
survived_by_embark_df = pd.DataFrame([survived_by_embark1,survived_by_embark2,survived_by_embark3])
# Note: column 0 = died, column 1 = survived
survived_by_embark_df['total']=survived_by_embark_df[0] + survived_by_embark_df[1]
survived_by_embark_df.index = ['Southampton','Cherbourg','Queenstown']
survived_by_embark_df.rename(index=str,columns={1:'Survived',0:'Died'}, inplace=True)
print (survived_by_embark_df)
survived_by_embark_df.plot(kind='bar',label=['Survived','Died'], figsize=(15,8), color = ['r','b','y'])


# # Does the number of siblings/spouses on board impact their survival?

# In[ ]:


train.SibSp.max()


# In[ ]:


total_sibsp = train['SibSp'].value_counts()
figure = plt.figure(figsize=(15,8))
plt.hist([train[train['Survived']==1]['SibSp'], 
          train[train['Survived']==0]['SibSp']], color = ['b','r'],
          label = ['Survived','Dead'])
plt.xlabel('Number of siblings/spouses on board')
plt.ylabel('Number of passengers')
plt.legend()


# # Does the number of parents/children on board impact their survival?

# In[ ]:


train.Parch.max()


# In[ ]:


total_parch = train['Parch'].value_counts()
figure = plt.figure(figsize=(15,8))
plt.hist([train[train['Survived']==1]['Parch'], 
          train[train['Survived']==0]['Parch']], color = ['b','r'],
          label = ['Survived','Dead'])
plt.xlabel('Number of parents/children on board')
plt.ylabel('Number of passengers')
plt.legend()


# # Let's try a scatterplot on AGE against FARE.

# In[ ]:


plt.figure(figsize=(15,8))
ax = plt.subplot()
ax.scatter(train[train['Survived']==1]['Age'], 
           train[train['Survived']==1]['Fare'],
           c='blue',s=40)
ax.scatter(train[train['Survived']==0]['Age'], 
           train[train['Survived']==0]['Fare'], 
           c='red',s=40)
ax.set_xlabel('Age')
ax.set_ylabel('Fare')
ax.legend(('survived','dead'),scatterpoints=1,loc='upper right',fontsize=15,)


# # Let's encode the data to do logistic regression.

# In[ ]:


train.columns


# In[ ]:


cat_df = train[['Pclass','Sex','Age','Fare','SibSp','Parch']]
one_hot_encoded_training_predictors = pd.get_dummies(cat_df)
one_hot_encoded_training_predictors.head()


# # And split the data into X and y.

# In[ ]:


X2 = one_hot_encoded_training_predictors
y2 = train['Survived']


# In[ ]:


X2.shape


# In[ ]:


y2.shape


# In[ ]:


#dividing the data in training and test data 
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.65, random_state=1)


# # Build the logistic regression model.

# In[ ]:


logreg = LogisticRegression(solver='lbfgs',
                        multi_class='multinomial',
                            max_iter=1000) #logistic regression using python
logreg.fit(X2_train, y2_train), 


# In[ ]:


y_pred = logreg.predict(X2_test) #predicting the values
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X2_test, y2_test)))


# # Now test on real test data.

# In[ ]:


cat_test_df = x_test[['Pclass','Sex','Age','Fare','SibSp','Parch']]
one_hot_encoded_training_predictors = pd.get_dummies(cat_test_df)
one_hot_encoded_training_predictors.head()


# In[ ]:


X1 = one_hot_encoded_training_predictors
y1 = y_test


# In[ ]:


X1.shape


# In[ ]:


y1.shape


# In[ ]:


y_pred = logreg.predict(X1) #predicting the values
print('Accuracy of logistic regression classifier on original test set: {:.2f}'.format(logreg.score(X1, y1)))


# In[ ]:




