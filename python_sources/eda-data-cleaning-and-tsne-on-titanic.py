#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('.'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # Loading Data

# In[ ]:


train_data = pd.read_csv("../input/titanic/train.csv")
train_data.head()


# In[ ]:


test_data = pd.read_csv('../input/titanic/test.csv')
test_data.head()


# # EDA

# In[ ]:


train_data.shape


# In[ ]:


test_data.shape


# 1. Check the data types of each feature for inconsistencies

# In[ ]:


train_data.dtypes


# In[ ]:


test_data.dtypes


# 2. Check the data for missing values

# In[ ]:


print("Missing data counts in Training Data : ")
print(train_data.isnull().sum())

print("Missing data counts in Test Data : ")
print(test_data.isnull().sum())


# In[ ]:


print("Percentage of data missing Training Data: ")
print(train_data.isnull().sum()/train_data.shape[0])

print("Percentage of data missing Test Data: ")
print(test_data.isnull().sum()/test_data.shape[0])


# ## Observation: 
# 
# #### Train Data:
# * About 77% data is missing from the 'Cabin' column. We can drop that column.
# * Only two data points are missing from 'Embarked'
# * Age has about 20% missing data. 
# 
# #### Test Data:
# * 1 missing data point in Fare 
# * About 20% missing data in Age
# * 78 % missing data in Cabin.
# 
# We can safely remove the column Cabin from both Train and Test Data

# In[ ]:


columns_to_drop = []
columns_to_drop.append('Cabin')


# # Data Cleaning

# 1. Let us check if we can predict the missing Fare Price from the test data

# In[ ]:


test_data[test_data['Fare'].isnull()]


# ## Observations:
# We can see that the **Pclass** is **3** and **Ticket Number** is **3701** and also the **Embarked** is **'S'**.
# Let see if we can use this information to guess a **Fare**.

# In[ ]:


test_data[test_data['Ticket']=='3701']


# In[ ]:


class_3_data = test_data[test_data['Pclass'] == 3]
class_3_S = class_3_data[class_3_data['Embarked'] == 'S']
class_3_S[class_3_S['Age']>40]


# We can see that all the passengers with similar data points have a Fare of either 7 or 15. Lets set the Fare to 14.

# In[ ]:


test_data[test_data["Fare"].isnull()]


# In[ ]:


test_data.iloc[152,-3]= 14


# 2. Lets try to predict the ages for the missing age values

# In[ ]:


train_data[train_data['Age'].isnull()]


# ### Approach :
# We can check the Names of each passenger to understand Title given to each passenger in the data.
# Title corresponds to:
#    * Master
#    * Miss
#    * so on...
#    

# Getting all the titles from the Training data

# In[ ]:


def extract_titles(df):
    pos = df.columns.get_loc('Name')
    titles = set({})
    for row in df.values:
        title = row[pos].split(',')[1].split('.')[0] + '.'.strip()
        titles.add(title)

    return titles


# Adding a new column of titles to the Training Data

# In[ ]:


def add_titles_to_df(df) :
    titles = extract_titles(df)
    pos = df.columns.get_loc('Name')
    title_list = []
    for row in df.values:
        for title in titles:
            if title in row[pos]:
                title_list.append(title)
                break
    df['Title'] = title_list
    return df
    


# In[ ]:


train_data = add_titles_to_df(train_data)


# In[ ]:


train_data.head()


# We can see the Title column added above

# In[ ]:


test_data = add_titles_to_df(test_data)


# In[ ]:


train_data['Title'].value_counts()


# In[ ]:


test_data['Title'].value_counts()


# **Observation:**
# 
# * We can see that there are **Titles** with 1 or 2 entries. 
# * We can also see that there is a lot of inconsitencies in the titles between train and test set.

# Lets combine these titles with the common titles like Mr, Miss, Mrs, Master

# In[ ]:


male_titles = [' Col.',' Major.',' Capt.',' Jonkheer.',' Don.',' Sir.']
female_titles = [' Lady.',' Mme.',' the Countess.',' Dona.',' Mlle.']


# In[ ]:


def replace_uncommon_titles(df,new_title,title_list):
    pos = df.columns.get_loc('Title')
    for title in title_list:
        for i in range(0,df.shape[0]):
            if df.iloc[i,pos] == title:
                print(title)
                df.iloc[i,pos] = new_title
                
    return df
train_data = replace_uncommon_titles(train_data,' Mr.',male_titles)
train_data = replace_uncommon_titles(train_data," Miss.",female_titles)
test_data = replace_uncommon_titles(test_data," Mr.",male_titles)
test_data = replace_uncommon_titles(test_data," Miss.",female_titles)


# Now lets check the Age statistics for each title

# In[ ]:


titles = list(train_data['Title'].value_counts().index)
for title in titles:
    print("Title train:: ",title)
    print(train_data[train_data['Title'] == title].describe()["Age"])
    print("Title test:: ",title)
    print(test_data[test_data['Title'] == title].describe()["Age"])
    


# ## Use Titles to approximate the ages

# ### Approach: 
# * By intuition, the titles do signify some sort of age. For example, Master. is for boys under a certain age. 
# * So the approach is to group the data by titles and assign the mean age to the corresponding entries

# In[ ]:


age_mean = train_data.groupby("Title").mean()['Age']


# Mean ages by group

# In[ ]:


age_mean


# In[ ]:


def fill_age_na(df,age_mean):
    rows_with_age_missing = df[df['Age'].isnull()]
    pos = df.columns.get_loc("Age")
    for title in age_mean.index:
        passengerIds = rows_with_age_missing[rows_with_age_missing['Title'] == title]["PassengerId"]
        for Id in passengerIds:
            df.iloc[df[df['PassengerId'] == Id].index.values,pos] = age_mean[title]
    return df


# In[ ]:


train_data = fill_age_na(train_data,age_mean)
test_data = fill_age_na(test_data,age_mean)

train_data[train_data['Age'].isnull()]


# In[ ]:


test_data[test_data['Age'].isnull()]


# In[ ]:


train_data.isnull().sum()


# In[ ]:


test_data.isnull().sum()


# ### Final touches

# Now that we have filled the age fields, we can move on to the other columns
# 1. In the Embarked Column, we can see that there are only two missing data points. We can fill those with the mode of the data, i.e. 'S'.

# In[ ]:


# Filling Embarked with mode 'S'
train_data['Embarked']= train_data['Embarked'].fillna(value='S',axis=0)


# Adding all the columns to be dropped to the list

# In[ ]:


columns_to_drop.extend(["Ticket"])


# In[ ]:


def drop_columns(df,list_of_columns):
    return df.drop(list_of_columns,axis=1)
    


# In[ ]:


train_data = drop_columns(train_data,columns_to_drop)
test_data = drop_columns(test_data,columns_to_drop)


# # Extracting labels
# 

# In[ ]:


labels = train_data['Survived']
train_data = train_data.drop('Survived',axis=1)


# In[ ]:


cleaned_train_data = train_data
cleaned_test_data = test_data


# # Preprocessing Data

# In[ ]:


categorical_columns = ['Pclass','Sex','Embarked','Title']
numerical_columns = ['Age','Fare','SibSp','Parch']


# In[ ]:


from sklearn.preprocessing import StandardScaler

def preprocess_data(df):
    scaler = StandardScaler()

    # split data into numerical and categorical
    numerical_data = df[numerical_columns]
    categorical_data = df[categorical_columns]

    #scaling the data with StandardScaler
    std_data_numerical = scaler.fit_transform(numerical_data)
    df_numerical = pd.DataFrame(std_data_numerical,columns=numerical_columns,index=df.index)


    #handling the categorical data
    std_data_categorical = pd.get_dummies(categorical_data,columns=categorical_columns)

    # combining the numerical and categorical data into one DataFrame
    final_data = df_numerical.join(std_data_categorical,how='inner')
    
    return final_data


# In[ ]:


preprocessed_train_data = preprocess_data(cleaned_train_data)


# In[ ]:


preprocessed_test_data = preprocess_data(cleaned_test_data)


# In[ ]:


preprocessed_test_data.columns


# # Model Selection

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

train_X, val_X, train_y, val_y = train_test_split(preprocessed_train_data,labels,random_state=1)


# ## RandomForestClassifier

# In[ ]:


from sklearn.ensemble import RandomForestClassifier

estimators = [1, 2, 4, 8, 16, 32, 64, 100, 200]
for num in estimators:
    model = RandomForestClassifier(n_estimators=num)
    model.fit(train_X,train_y)
    preds = model.predict(val_X)
    
    print("Accuracy for {} estimators is {}".format(num,accuracy_score(val_y,preds,normalize=True)))


# ## XGBoostClassifier

# In[ ]:


import xgboost as xgb
n_est = [5,10,20,40,100,200]
for num in n_est:
    xgb_model = xgb.XGBClassifier(n_estimators=num)
    xgb_model.fit(train_X,train_y)
    preds = xgb_model.predict(val_X)

    print("Accuracy lr {} is {}".format(num,accuracy_score(val_y,preds,normalize=True)))


# ## Logistic Regression

# In[ ]:


from sklearn.linear_model import LogisticRegression
iters = [50,100,150,200,300,500]
for num in iters:
    model = LogisticRegression(penalty='l2',max_iter=num,random_state=1,verbose=3)
    model.fit(train_X,train_y)
    preds = model.predict(val_X)
    print("Accuracy for {} iterations  is {}".format(num,accuracy_score(val_y,preds,normalize=True)))


# We get the highest accuracy on the validation with Logistic regression :)

# In[ ]:


model = LogisticRegression(penalty='l2',solver='lbfgs',random_state=1)
model.fit(preprocessed_train_data,labels)
preds = model.predict(preprocessed_test_data)


# In[ ]:





# In[ ]:


output = pd.DataFrame({'PassengerId': test_data.PassengerId,
                      'Survived': preds})
output.to_csv('submission.csv', index=False)

output


# In[ ]:


from sklearn.manifold import TSNE

def plot_TSNE(data,label):
    
    tsne = TSNE(n_components=2,random_state=0,n_iter=5000,verbose=3,perplexity=30,learning_rate=200)
    embeddings = tsne.fit_transform(data)
    return embeddings
    


# In[ ]:


embeddings = plot_TSNE(preprocessed_train_data,labels)


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


plot_data = np.hstack((embeddings,np.array(labels).reshape(len(labels),1)))
    
plot_df = pd.DataFrame(plot_data,columns=['Dim1','Dim2','Survived'])
    
sns.FacetGrid(plot_df,hue="Survived",height=5).map(sns.scatterplot,"Dim1","Dim2").add_legend()


# In[ ]:




