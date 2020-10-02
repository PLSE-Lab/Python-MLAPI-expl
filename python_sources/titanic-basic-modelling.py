#!/usr/bin/env python
# coding: utf-8

# # This notebook was created for a quick experiment on the deployment of machine learning model using the titanic dataset.
# 
# The model performance can greatly be improved on but I will be sticking to basic modelling in other to remian within standard available packages during the deployment without need for the creation of custom packages. 
# ### The focus here is just to get a simple model serialized so as to have it deployed on google app engine. 
# 
# Below is a link to medium article explaining the deployment in stages for further clarification on why this kernel was created:
# https://heartbeat.fritz.ai/deploying-machine-learning-models-on-google-cloud-platform-gcp-7b1ff8140144
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


get_ipython().system('pip install pandas==0.25.3')
get_ipython().system('pip install scikit-learn==0.22')
get_ipython().system('pip install numpy==1.18.0')
get_ipython().system('pip install ppscore')


# In[ ]:


import numpy as np 
import pandas as pd
import sklearn
import matplotlib.pyplot as plt 
import seaborn as sns
import ppscore as pps
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import f1_score
from sklearn.externals import joblib

seed = 2020


# In[ ]:


print('numpy version: '+ np.__version__)
print('pandas version: '+ pd.__version__)
print('sklearn version: '+ sklearn.__version__)


# In[ ]:


pwd


# In[ ]:


path ='/kaggle/input/titanic/'
train = pd.read_csv(os.path.join(path,'train.csv'))
test = pd.read_csv(os.path.join(path,'test.csv'))
sample_sub = pd.read_csv(os.path.join(path,'gender_submission.csv'))


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


train.describe()


# In[ ]:


train.info()


# In[ ]:


# checking for missing values 
fig, ax = plt.subplots(figsize=(10,5))
sns.heatmap(train.isnull(), cbar=False)
plt.show()


# ### There are missing values in age and Cabin features 

# # Short Exploratory Data Analysis  

# In[ ]:


sns.countplot(train.Survived)


# ### Imbalance dataset... resampling techniques will be useful so as to have better performance evaluation

# In[ ]:


sns.barplot(x='Sex', y='Survived', data=train)
plt.ylabel("Rate of Surviving")
plt.title("Plot of Survival as function of Sex", fontsize=16)
plt.show()
train[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# ### There are much more survivals in female than male

# In[ ]:


sns.barplot(x='Pclass', y='Survived', data=train)
plt.ylabel("Survival Rate")
plt.title("Plot of Survival as function of Pclass", fontsize=16)
plt.show()
train[["Pclass", "Survived"]].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# ### Passengers in Pclass 1 and 2 have higher chances of surviving 

# In[ ]:


plt.figure(figsize=(10,5))
sns.heatmap(train.corr(),annot=True)


# Some features are highly correlated with one another, creating new features will be helpful. e.g Pclass and SibSp are highly correlated, a new feature will be created to reduce the Collinearity

# In[ ]:


plt.figure(figsize=(10,5))
ax = train.corr()['Survived'].plot(kind='bar',title='correlation of target variable to features')
ax.set_ylabel('correlation')


# The passengerId feature will be dropped as this is unique across all samples. Pclass and Fare have the highest absolute correletion with the target variable

# In[ ]:


# Predictive Power Score Plot
plt.figure(figsize=(20,10))
sns.heatmap(pps.matrix(train),annot=True)


# * of the whole features, PassengerID and Name have lowest 0 PPS
# * Sex, Pclass and fare have have a good PPS 
# * The Ticket feature can greatly improve model performance with enough feature engineering 
# * Pclass can be predicted using the Fare feature with 90% accuracy
# * The missing values in Cabin can be worked on using the Pclass feature

# In[ ]:


train_copy = train.copy()
train_copy.dropna(inplace = True)
sns.distplot(train_copy.Age)


# Looks like the distribution of ages is slightly skewed right. Because of this, we can fill in the null values with the median for the most accuracy.

# In[ ]:


sns.barplot(x="Pclass", y="Survived", hue="Sex", data=train_copy)
plt.ylabel("Survival Rate")
plt.title("Survival Rates Based on Gender and Class")


# There were more female survivals than male in all the 3 Pclass categories

# In[ ]:


sns.barplot(x="Sex", y="Survived", hue="Pclass", data=train_copy)
plt.ylabel("Survival Rate")
plt.title("Survival Rates Based on Gender and Class")


# From the graphs above,Female survival rate in the all the Pclasses is higher than male survival rate respectively

# In[ ]:


train_null = train.isnull().sum()
test_null = test.isnull().sum()
print(train_null[train_null !=0])
print('-'*40)
print(test_null[test_null !=0])


# # Handling missing values 

# In[ ]:


from sklearn.impute import SimpleImputer
age_imp = SimpleImputer(strategy= 'median')
age_imp.fit(np.array(train.Age).reshape(-1,1))

train.Age = age_imp.transform(np.array(train.Age).reshape(-1,1))
test.Age = age_imp.transform(np.array(test.Age).reshape(-1,1))
train.head()


# In[ ]:


#save age imputer 
with open('age_imputer.joblib', 'wb') as f:
  joblib.dump(age_imp,f)


# In[ ]:


emb_imp = SimpleImputer(strategy= 'most_frequent' )
emb_imp.fit(np.array(train.Embarked).reshape(-1,1))

train.Embarked = emb_imp.transform(np.array(train.Embarked).reshape(-1,1))
test.Embarked = emb_imp.transform(np.array(test.Embarked).reshape(-1,1))
train.head()


# In[ ]:


#save embark imputer 
with open('embark_imputer.joblib', 'wb') as f:
  joblib.dump(emb_imp,f)


# In[ ]:


train.isnull().sum() 
print('-'*40)
test.isnull().sum()


# In[ ]:


drop_cols = ['PassengerId','Ticket','Cabin','Name']
train.drop(columns=drop_cols,axis=1,inplace = True)
test_passenger_id = test.PassengerId
test.drop(columns=drop_cols,axis=1,inplace = True)


# In[ ]:


test.fillna(value = test.mean(),inplace=True)


# In[ ]:


train.isnull().sum().any() , test.isnull().sum().any()


# In[ ]:


train['Number_of_relatives'] = train.Parch + train.SibSp
test['Number_of_relatives'] = test.Parch + test.SibSp

train.drop(columns=['Parch','SibSp'],axis=1,inplace=True)
test.drop(columns=['Parch','SibSp'],axis=1,inplace=True)


# In[ ]:


train.head()


# In[ ]:


gender_dic = {'male':1,'female':0}
train.Sex = train.Sex.map(gender_dic)
test.Sex = test.Sex.map(gender_dic)
train.head()


# In[ ]:


cat_col = ['Embarked', 'Pclass']
One_hot_enc = OneHotEncoder(sparse=False,drop='first',dtype=np.int)


# In[ ]:


encoded_train = pd.DataFrame(data=One_hot_enc.fit_transform(train[cat_col]), columns=['emb_2','emb_3','Pclass_2','Pclass_3'])
encoded_test = pd.DataFrame(data=One_hot_enc.transform(test[cat_col]),columns=['emb_2','emb_3','Pclass_2','Pclass_3'])


# In[ ]:


#save One_hot_enc 
with open('One_hot_enc.joblib', 'wb') as f:
  joblib.dump(One_hot_enc,f)


# In[ ]:


train.drop(columns=cat_col,axis=1,inplace=True)
test.drop(columns=cat_col,axis=1,inplace=True)

train = pd.concat([train,encoded_train],axis=1)
test = pd.concat([test,encoded_test],axis=1)
train.head()


# In[ ]:


features = test.columns
X = train[features]
y = train.Survived


# In[ ]:


scaler = StandardScaler()
X = scaler.fit_transform(X)
test = scaler.transform(test)


# In[ ]:


#save scaler 
with open('scaler.joblib', 'wb') as f:
  joblib.dump(scaler,f)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=seed)


# In[ ]:


logistic_model  = LogisticRegression()
logistic_model.fit(X_train,y_train)


# In[ ]:


print('f1_score on training set: {}'.format(f1_score(logistic_model.predict(X_train),y_train)))
print('f1_score on test set: {}'.format(f1_score(logistic_model.predict(X_test),y_test)))


# In[ ]:


logistic_model.fit(X,y)
#save model 
with open('model-v1.joblib', 'wb') as f:
  joblib.dump(logistic_model,f)


# ### Note that we saved all the objects used during data tranformation and trained model.
# ### All these saved objects can be found and downloaded in the output directory under the Data section at the top right corner of the notebook. these will be needed for test data transformation and inference at the deployment stage.

# In[ ]:




