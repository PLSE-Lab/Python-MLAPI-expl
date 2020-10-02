#!/usr/bin/env python
# coding: utf-8

# ## **Titanic Dataset solution**
# 
# **Question-Problem Statement**
# 
# The sinking of the RMS Titanic is one of the most infamous shipwrecks in history.  On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. This sensational tragedy shocked the international community and led to better safety regulations for ships.
# 
# One of the reasons that the shipwreck led to such loss of life was that there were not enough lifeboats for the passengers and crew. Although there was some element of luck involved in surviving the sinking, some groups of people were more likely to survive than others, such as women, children, and the upper-class.
# 
# In this solution we will predict whether a person survives or not based on various factors like social class, gender and age.

# In[ ]:


# Import necessary packages
import numpy as np 
import pandas as pd 
import seaborn as sb
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.simplefilter("ignore")
import os
print(os.listdir("../input"))
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
import xgboost
from sklearn.metrics import explained_variance_score
from xgboost import XGBClassifier


# In[ ]:


# Load train dataset
train=pd.read_csv('../input/train.csv')
train.head()


# In[ ]:


# Load test dataset
test=pd.read_csv('../input/test.csv')
test.head()


# ## **Data Wrangling**
# First we will deal with missing and incorrect data.
# First lets drop the columns we will not be needing from both datasets.
# After analysis we can see that Age has missing data in both train and test data.
# In tain dataset Embarked has some missing values.
# 

# In[ ]:


train.info()


# In[ ]:


# Drop uneccessary columns
train.drop(columns=['PassengerId','Name','Cabin','Ticket','Fare'],inplace=True)
test.drop(columns=['Name','Ticket','Cabin','Fare'],inplace=True)


# For filling the missing values of the age dataset lets get the mean of each class furthur divided by sex. Then we will fill these mean values in the coressponding class and sex.

# In[ ]:


# Get index with null values in train dataset
index_list=train[train['Age'].isnull()].index
index_list


# In[ ]:


# Fill those null values with appropiate mean values
for index in index_list:
    if train.loc[index,'Pclass']==1 and train.loc[index,'Sex']=='female':
        train.loc[index,'Age']=np.ceil(train.groupby(['Pclass','Sex'])['Age'].mean()[1][0])
    elif train.loc[index,'Pclass']==1 and train.loc[index,'Sex']=='male':
        train.loc[index,'Age']=np.ceil(train.groupby(['Pclass','Sex'])['Age'].mean()[1][1])
    elif train.loc[index,'Pclass']==2 and train.loc[index,'Sex']=='female':
        train.loc[index,'Age']=np.ceil(train.groupby(['Pclass','Sex'])['Age'].mean()[2][0])
    elif train.loc[index,'Pclass']==2 and train.loc[index,'Sex']=='male':
        train.loc[index,'Age']=np.ceil(train.groupby(['Pclass','Sex'])['Age'].mean()[2][1])
    elif train.loc[index,'Pclass']==3 and train.loc[index,'Sex']=='female':
        train.loc[index,'Age']=np.ceil(train.groupby(['Pclass','Sex'])['Age'].mean()[3][0])
    else:
        train.loc[index,'Age']=np.ceil(train.groupby(['Pclass','Sex'])['Age'].mean()[3][1])


# Since Embarked ghas only two values missing we can fill it with the most common value

# In[ ]:


# Fill Embarked with mode of the column
train['Embarked'].fillna(train['Embarked'][0],inplace=True)


# In[ ]:


# Get index with null values in test dataset
index_list=test[test['Age'].isnull()].index
index_list


# In[ ]:


# Fill those null values with appropiate mean values
for index in index_list:
    if test.loc[index,'Pclass']==1 and test.loc[index,'Sex']=='female':
        test.loc[index,'Age']=np.ceil(test.groupby(['Pclass','Sex'])['Age'].mean()[1][0])
    elif test.loc[index,'Pclass']==1 and test.loc[index,'Sex']=='male':
        test.loc[index,'Age']=np.ceil(test.groupby(['Pclass','Sex'])['Age'].mean()[1][1])
    elif test.loc[index,'Pclass']==2 and test.loc[index,'Sex']=='female':
        test.loc[index,'Age']=np.ceil(test.groupby(['Pclass','Sex'])['Age'].mean()[2][0])
    elif test.loc[index,'Pclass']==2 and test.loc[index,'Sex']=='male':
        test.loc[index,'Age']=np.ceil(test.groupby(['Pclass','Sex'])['Age'].mean()[2][1])
    elif test.loc[index,'Pclass']==3 and test.loc[index,'Sex']=='female':
        test.loc[index,'Age']=np.ceil(test.groupby(['Pclass','Sex'])['Age'].mean()[3][0])
    else:
        test.loc[index,'Age']=np.ceil(test.groupby(['Pclass','Sex'])['Age'].mean()[3][1])


# In[ ]:


# Check if the above operations worked correctly
train.isnull().sum().max(),test.isnull().sum().max()


# ## **Visualizations**

# In[ ]:


base_color=sb.color_palette()[0]


# In[ ]:


# Bivariate plot of Survived vs. Age
sb.distplot(train[train['Survived']==1]['Age'],label='Survived');
sb.distplot(train[train['Survived']==0]['Age'],label='Not Survived');
plt.legend();
plt.title('Survived vs. Age');


# In[ ]:


# Multi-variate plot of Survived vs Age by Gender
sb.pointplot(data=train,x='Survived',y='Age',hue='Sex',linestyles="",dodge=0.3);
xticks=[0,1]
xlabel=['No','Yes']
plt.xticks(xticks,xlabel);
plt.title('Survived vs Age by Gender');


# From the above plot we can see that mostly middle aged women and men survived. The large error bars mean that the number of data for that point is less. Hence more females survived than men. 

# In[ ]:


# Multi-variate plot of Survived vs Age by Class
sb.pointplot(data=train,x='Survived',y='Age',hue='Pclass',linestyles="",dodge=0.3,palette='viridis_r');
xticks=[0,1]
xlabel=['No','Yes']
plt.xticks(xticks,xlabel);
plt.title('Survived vs Age by Class');


# ## Model and Predict

# In[ ]:


'''
single=[train,test]
# Map columns to numerical values
for data in single:
    data['Sex']=data['Sex'].map({'female':1,'male':0}).astype(int)
    data['Embarked']=data['Embarked'].map({'S':1,'C':2,'Q':3}).astype(int)
'''


# In[ ]:


# Merge the two datasets
ntrain = train.shape[0]
ntest = test.shape[0]
all_data = pd.concat((train, test))


# In[ ]:


# Get dummy variables
all_data=pd.get_dummies(all_data)


# In[ ]:


# Seperate the combined dataset into test and train data
test=all_data[all_data['Survived'].isnull()]
train=all_data[all_data['PassengerId'].isnull()]


# In[ ]:


# Check if the new and old sizes are equal
assert train.shape[0]==ntrain
assert test.shape[0]==ntest


# In[ ]:


# Drop extra columns
test.drop(columns='Survived',inplace=True)
train.drop(columns='PassengerId',inplace=True)
test['PassengerId']=test['PassengerId'].astype(int)


# In[ ]:


# Divide the data into test and train
X_train=train.drop('Survived',axis=1)
Y_train=train['Survived']
X_test=test.drop('PassengerId',axis=1)


# In[ ]:


'''
# Fit the model using Random Forest
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
acc_random_forest
'''


# In[ ]:


# Fit the model using XGBClassifier
xgb = xgboost.XGBClassifier(learning_rate= 0.01, max_depth= 4, n_estimators= 300, seed= 0)
xgb.fit(X_train,Y_train)
Y_pred = xgb.predict(X_test)


# In[ ]:


final_df = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": Y_pred
    })


# In[ ]:


final_df['Survived']=final_df['Survived'].astype(int)


# In[ ]:


# Save the dataframe to a csv file
final_df.to_csv('submission.csv',index=False)

