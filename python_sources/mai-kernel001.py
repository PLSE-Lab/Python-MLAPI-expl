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
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
print (os.listdir('../input/udacity-mlcharity-competition'))
# Any results you write to the current directory are saved as output.


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns


# ## 1.Taking a look to the data

# In[ ]:


df_train=pd.read_csv('../input/udacity-mlcharity-competition/census.csv')
df_test = pd.read_csv("../input/udacity-mlcharity-competition/test_census.csv").drop('Unnamed: 0',axis=1)


# In[ ]:


df_train.info()


# In[ ]:


## Drop the useless columns


# In[ ]:


df_train.drop(columns=['education-num'],inplace=True)


# In[ ]:


df_train.head()


# *1. get the unique values in the data*

# In[ ]:


for column in df_train.columns:
    print(column, df_train[column].unique())


# ### Total number of records
# 
# 
# 

# In[ ]:


n_records = len(df_train)
n_records


# ### Number of records where individual's income is more than $50,000**

# In[ ]:


n_greater_50k = len(df_train[df_train['income'] == '>50K'])
n_greater_50k


# ## Visualizations

# In[ ]:


## Visualization of income level based on sex and education


# In[ ]:


sns.set(style="whitegrid", color_codes=True)
sns.catplot("sex", col='education_level', data=df_train, hue='income', kind="count", col_wrap=4);


# 

# In[ ]:


income=df_train.income.map({'<=50K': 0, '>50K':1})
income.head()


# In[ ]:


#Age distribution


# In[ ]:


plt.hist(df_train.age,bins=20,color='c')
plt.title('Histogram: Age')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show


# **Most age between 30 and 35 **

# **Getting some useful insights** 

# **Create a histogram for people older in age and see thier effect on the income**

# In[ ]:


income.unique()


# In[ ]:


df_train.head(3)


# In[ ]:


hist_above_50=plt.hist(df_train[df_train.income==">50K"].age.values,10,facecolor='c')
plt.title('Age distribution above 50K earners')
plt.xlabel('Age')
plt.ylabel('Frequency')


# **histogram for people less than 50,000$ earners**

# In[ ]:


hist_above_50=plt.hist(df_train[df_train.income=="<=50K"].age.values,10,facecolor='c')
plt.title('Age distribution above 50K earners')
plt.xlabel('Age')
plt.ylabel('Frequency')


# In[ ]:


df_train[df_train.income=='>50K'].groupby('workclass').workclass.count().sort_values().plot(kind='bar')


# As we see most of people earn greater than 50,000$ lies in the category private jobs

# ## 3.Preprocessing the data

# In[ ]:


df_train.head()


# In[ ]:


df_train_with_dummies = pd.get_dummies(df_train,columns=['sex','workclass','education_level','marital-status','occupation','relationship','race','native-country','income'], sparse=True)


# In[ ]:


# Since the data is splitted to numerical fetaures and categorial one 
#so let's normailze 
#the categorial ones


# In[ ]:


features_raw.head()


# In[ ]:


income_raw.head()


# In[ ]:


df_train.head()


# In[ ]:


features_raw.isnull().sum()


# In[ ]:


## good no null values!


# ### Handling categorial data

# ### use get dummies for rest of columns

# In[ ]:


df_train.head()


# In[ ]:


df_train_with_dummies = pd.get_dummies(df_train,columns=['sex','workclass','education_level','marital-status','occupation','relationship','race','native-country','income'], sparse=True)


# In[ ]:


df_train_with_dummies.head()


# ### Normalize the numerical feature for a better model
# 
# 
# 
# 

# In[ ]:


df_train_with_dummies.head()


# In[ ]:


from sklearn.preprocessing import MinMaxScaler


scaler = MinMaxScaler()
numerical = ['age',  'capital-gain', 'capital-loss', 'hours-per-week']
df_train_with_dummies[numerical]=scaler.fit_transform(df_train_with_dummies[numerical])

# Show an example of a record with scaling applied


# ## Now the data is ready for modeling

# ### First we need to split our data to train and test data ..

# In[ ]:


features_raw =df_train_with_dummies.drop(columns=['income_<=50K','income_>50K'],axis=1)


# In[ ]:


def  get_value(x):
    if x=='>50K':
     return 1
    else:
        return 0


# In[ ]:



df_train['new_income'] = df_train['income'].apply(get_value)
new_income=df_train['new_income']


# In[ ]:


# import the modules 
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(features_raw, new_income, test_size = 0.2, random_state = 0)
print(len(x_train))
print(len(x_test))
print(len(features_raw))



# For the diversity of the values i will apply standard scalar

# In[ ]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# take a look how the values was before and after the scalling 

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


# In[ ]:


print("Training set has {} samples.".format(x_train.shape[0]))
print("Testing set has {} samples.".format(x_test.shape[0]))


# ![****](http://)the features_raw and income_raw are the features columns and the goal columns

# ## First:building a simple linear regression model

# In[ ]:


from sklearn.linear_model import LinearRegression

linear_model = LinearRegression(fit_intercept=True).fit(x_train, y_train)


# In[ ]:


print("Training_score : " , linear_model.score(x_train, y_train))


# In[ ]:


y_pred = linear_model.predict(x_test)


# In[ ]:


from sklearn.metrics import r2_score

print("Testing_score : ", r2_score(y_test, y_pred))


# In[ ]:


df_pred_actual = pd.DataFrame({'predicted': y_pred, 'actual': y_test})

df_pred_actual.head(10)


# In[ ]:


## May be the linear regression is not working well 


# In[ ]:


## Try logistic regression as a classification model


# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


model1 = LogisticRegression(random_state=0)


# In[ ]:


model1.fit(x_train,y_train)


# In[ ]:


print ('score for logistic regression  {}'.format(model1.score(x_test, y_test)))


# Good score using logistic regression is high!!!

# ### Second:using conf matrix

# In[ ]:


# peformance metrics
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score


# In[ ]:


# performance metrics
# accuracy
print ('accuracy for logistic regression {}'.format(accuracy_score(y_test, model1.predict(x_test))))
# confusion matrix
print('confusion matrix for logistic regression {}'.format(confusion_matrix(y_test, model1.predict(x_test))))
# precision 
print ('precision for logistic regression {}'.format(precision_score(y_test, model1.predict(x_test))))
# precision 
print ('recall for logistic regression {}'.format(recall_score(y_test, model1.predict(x_test))))


# ## The next phase is building classification model using  k-nearest neighbors and Naive Bayes

# In[ ]:


from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5)
classifier.fit(x_train,y_train)


# In[ ]:



cm = confusion_matrix(y_test, classifier.predict(x_test))
cm


# In[ ]:


print("model accuracy : {:.4f}".format(classifier.score(x_test, y_test)))


# In[ ]:


from sklearn.naive_bayes import GaussianNB
classifier2 = GaussianNB()
classifier2.fit(x_train, y_train)


# In[ ]:


print("model accuracy : {:.4f}".format(classifier2.score(x_test, y_test)))


# In[ ]:




