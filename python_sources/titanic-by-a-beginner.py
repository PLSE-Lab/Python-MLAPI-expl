#!/usr/bin/env python
# coding: utf-8

# Hi guys, this is my first Machine Learning project and I have been excited to share this with you all! 
# This is my first step in my long journey in Machine Learning.
# I would like some tips and advice on how I can improve.

# # Importing Libraries 

# In[ ]:


get_ipython().system('pip install fastai==0.7')
get_ipython().system('pip install numpy')
get_ipython().system('pip install scipy')
get_ipython().system('pip install seaborn')
get_ipython().system('pip install matplotlib')
get_ipython().system('pip install sklearn')
get_ipython().system('pip install pandas')
get_ipython().system('pip install xgboost')

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


import numpy as np
import scipy 
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
import pandas as pd

from xgboost import XGBClassifier

from sklearn import ensemble, preprocessing, linear_model, model_selection, metrics 
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

from pandas_summary import DataFrameSummary

import warnings
warnings.filterwarnings('ignore')


# # Reading and Checking CSV file

# In[5]:


df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')


# In[6]:


df.head()


# In[7]:


df.describe(include='all')


# In[8]:


df.dtypes


# In[9]:


df.shape


# In[10]:


df.isnull().sum()


# Observations made so far:
#     - 891 passengers in this data set
#     - Age is missing 177 values: so I will take the mean of the ages we already have 
#     - Cabin is missing 687 values
#     - Embarked is only missing 3 values
#     - Name, Sex, Ticket, Cabin and Embarked are all objects and not integers

# # Cleaning the Code

# In[11]:


df = df.fillna(df.mean())
test_df = test_df.fillna(df.mean())
df.describe(include='all')


# The missing Age values have been replaced with an averave of the whole column

# # Analysing the Code

# Visualising the data and establishing relationships between Survival rates and other variables

# # Survival

# In[12]:


sns.countplot('Survived', data=df)
#print("Amount of people survived", df["Survived"][df["Survived"] == 1].value_counts(normalize = True)[1]*100)

print("Number of people who survived:", len(df[df['Survived'] > 0]))


# # Passenger ID 

# In[13]:


# Establishing if there was a relationship between PassengerID and Survival

passenger_df = pd.cut(df['PassengerId'],
                      bins=[0, 99, 198, 297, 396, 495, 594, 693, 792, 891], 
                      labels=['0-99', '99-198', '198-297', '297-396', '396-495', '495-594', '594-693', '693-792', '792-891'])

sns.barplot('Survived', passenger_df, data=df)


# Does not seem that there is much relationship between the PassengerId and whether or not someone survived.
# However, it does seem that those with a PassengerId between 297-396 were more likely to survive and those with a PassengerId of 99-198 were less likely to survive.

# # Class

# In[14]:


#Establishing the relationship between Class and Survival

print("The number of people survived in Class 1: ", len(df[df['Pclass'] == 1]))
print("The number of people survived in Class 2: ", len(df[df['Pclass'] == 2]))
print("The number of people survived in Class 3: ", len(df[df['Pclass'] == 3]))
sns.barplot('Pclass', 'Survived', data=df)


# Observation
#     - People who were of a higher class were more likely to survive

# ## Age

# In[15]:


#Establishing the relationship between Survival and Age

a = sns.FacetGrid(df, hue = 'Survived', aspect=4 )
a.map(sns.kdeplot, 'Age', shade= True)
a.set(xlim=(0 , df['Age'].max()))
a.add_legend()


# In[16]:


df['Age'].describe()


# In[17]:


age_df = pd.cut(df['Age'],
    bins=[0, 5, 12, 18, 25, 32, 40, 60, 80],
    labels=['Babies','Children','Teenagers', 'Young Adults', 'Adults', 'Middle-Age Adult', 'Senior', 'Older Senior']
)

plt.subplots(figsize=(10,5))
sns.set(style="ticks", color_codes=True)
sns.barplot(x=age_df, y=df['Survived'])
sns.pointplot(x=age_df, y=df['Survived'], color='black')


# Observation
#     - Babies were most likely to survive than any other age group
#     - Older Seniors were unlikely to survive in comparison to other age groups

# ## SibSp

# In[18]:


##Establishing a relationship between SibSp (Siblings/Spouses) and Survival
sns.barplot(x=df['SibSp'], y=df['Survived'])

print("The amount of people who survived with 0 siblings/spouses: ", 
      df['Survived'][df['SibSp'] == 0].value_counts(normalize=True)[1] * 100)
print("The amount of people who survived with 1 siblings/spouses: ", 
      df['Survived'][df['SibSp'] == 1].value_counts(normalize=True)[1] * 100)
print("The amount of people who survived with 2 siblings/spouses: ", 
      df['Survived'][df['SibSp'] == 2].value_counts(normalize=True)[1] * 100)
print("The amount of people who survived with 3 siblings/spouses: ", 
      df['Survived'][df['SibSp'] == 3].value_counts(normalize=True)[1] * 100)
print("The amount of people who survived with 4 siblings/spouses: ", 
      df['Survived'][df['SibSp'] == 4].value_counts(normalize=True)[1] * 100)


# Observation
#     - Those who had 1-2 siblings or spouses were more likely to survive than those who had more siblings

# ## Parch

# In[19]:


##Establishing a relationship between Parch (Parents/Children) and Survival
sns.barplot(x=df['Survived'], y=df['Parch'])


# ## Fare

# In[20]:


#Establishing a relationship between Fare fee and Survival
fare_df = pd.cut(df['Fare'],
                bins = [0, 25, 50, 75, 100, 200, 300, 400, 512],
                labels = ['0-25', '25-50', '50-75', '75-100', '100-200', '200-300', '300-400', '400+'])

plt.subplots(figsize=(10,5))
sns.barplot(x=fare_df, y=df['Survived'])


# Observation
#     - Those who paid more for their ticket were more likely to survive

# ### Embarked

# In[21]:


#Establishing a relationship between location embarked and Survival

sns.barplot(x=df['Embarked'], y=df['Survived'])


# Observation
#     - Those who embarked at Chersborg, France were more like to survive than those who embarked at Southampton or       Queenstown

# In[22]:


sns.heatmap(df.corr(), vmax=0.8)


# In[23]:


##Pairplot of all the features
allPairPlot =  sns.pairplot(df, hue = 'Survived', palette = 'deep', size=1.2, diag_kind = 'kde', diag_kws=dict(shade=True), plot_kws=dict(s=10) )
allPairPlot.set(xticklabels=[])


# # Processing

# In[24]:


#Dropping columns and making it suitable for predictions 
test_df = test_df.drop(['Name', 'Cabin', 'Ticket'], axis=1)
test_df['Sex'] = pd.Categorical(test_df.Sex)
test_df['Embarked'] = pd.Categorical(test_df.Embarked)

test_df['Sex'] = test_df.Sex.cat.codes
test_df['Embarked'] = test_df.Embarked.cat.codes

test_df.head()
features = ['PassengerId', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']


# In[25]:


#Dropping the columns that are not needed and converting the ones that are needed into integers
dropped = df.drop(['Survived', 'Name', 'Cabin', 'Ticket'], axis=1)
target = df['Survived']

dropped['Sex'] = pd.Categorical(dropped.Sex)
dropped['Embarked'] = pd.Categorical(dropped.Embarked)
#dropped['PassengerId'] = pd.Categorical(dropped.PassengerId)

dropped['Sex'] = dropped.Sex.cat.codes 
#1 = Male, 0 = Female
dropped['Embarked'] = dropped.Embarked.cat.codes
#0 = Southampton, 1 = QueensTown, 2 = Cherborg
#dropped['PassengerId'] = dropped.PassengerId.cat.codes

X_train, X_val, y_train, y_val = train_test_split(dropped, target, test_size=0.25, random_state=0)
X_train.dtypes


# # Testing the Model 
# RFR, KNN, RFC, XGBoost, SVC

# In[26]:


#Creating an list that will handle accuracy scores
accuracies = []


# In[27]:


#Logistic Regression

logistic = linear_model.LogisticRegression()
logistic.fit(X_train, y_train)
logisticscore = logistic.score(X_train, y_train)
y_pred = logistic.predict(X_val)
final_log = round(accuracy_score(y_pred, y_val) * 100, 2)
accuracies.append(final_log)
print(final_log)


# In[28]:


#Gradient Boosting Classifier

gbc = GradientBoostingClassifier()
gbc.fit(X_train, y_train)
gbc_score = gbc.score(X_train, y_train)
y_pred = gbc.predict(X_val)
final_gbc = round(accuracy_score(y_pred, y_val) * 100, 2)
accuracies.append(final_gbc)
print(final_gbc)


# In[29]:


#Random Forest Classifier

rf_classifier = RandomForestClassifier(n_estimators = 20, criterion='entropy')
rf_classifier.fit(X_train, y_train)
rfc_score = rf_classifier.score(X_train, y_train)
y_pred = rf_classifier.predict(X_val)
final_rfc = round(accuracy_score(y_pred, y_val) * 100, 2)
accuracies.append(final_rfc)
print(final_rfc)


# In[30]:


#SVC 

svc = SVC(kernel='linear')
svc.fit(X_train, y_train)
svc_score = svc.score(X_train, y_train)
y_pred = svc.predict(X_val)
final_svc = round(accuracy_score(y_pred, y_val) * 100, 2)
accuracies.append(final_svc)
print(final_svc)


# In[31]:


#KNN

knn = KNeighborsClassifier(p=2, n_neighbors = 10)
knn.fit(X_train, y_train)
knn_score = knn.score(X_train, y_train)
y_pred = knn.predict(X_val)
final_knn = round(accuracy_score(y_pred, y_val) * 100, 2)
accuracies.append(final_knn)
print(final_knn)


# In[32]:


#XGBoost

xgboost = XGBClassifier()
xgboost.fit(X_train, y_train)
xgscores = xgboost.score(X_train, y_train)
y_pred = xgboost.predict(X_val)
final_xgb = round(accuracy_score(y_pred, y_val) * 100, 2)
accuracies.append(final_xgb)
print(final_xgb)


# In[33]:


accuracies


# In[34]:


accuracy_labels = ['Logistic Regression', 'Gradient Boosting Classifier', 'Random Forest Classifier', 'SVC', 'KNN', 'XGBoost']
sns.barplot(x=accuracies, y=accuracy_labels)


# Decided to go with XGBoost as it has the highest accuracy rate.

# # Submitting files to Kaggle

# In[35]:


predictions = xgboost.predict(test_df[features])
predictions


# In[36]:


submission = pd.DataFrame({'PassengerId' : test_df['PassengerId'], 'Survived': predictions})
submission.to_csv('submission.csv', index=False)


# In[37]:


submission.tail()


# 
