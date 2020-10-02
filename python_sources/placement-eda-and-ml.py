#!/usr/bin/env python
# coding: utf-8

# <img src='https://omnifin.in/wp-content/uploads/2017/05/Campus-Placement.jpg'>

# # Table Of Content :

# 1. [EDA](#1)
# 1. [Factors Influencing Placement](#2)
# 1. [Does Marks Matter?](#3)
# 1. [Best Degree Specialization](#4)
# 1. [Machine Learning](#5)
#     1. [Logistic Regression](#5.1)
#     1. [Random Forest](#5.2)

# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# #### Importing Necessary Libraries

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")


# #### Reading the dataframe and renaming column names for easy understanding

# In[ ]:


data = pd.read_csv('/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')


# In[ ]:


data = data.rename(columns={'sl_no':'Id', 'gender':'Gender','ssc_p':'SSC_Per', 'ssc_b':'SSC_Board' ,'hsc_p':'HSC_Per', 'hsc_b':'HSC_Board', 'hsc_s':'HSC_Stream', 
                            'degree_p':'DegreePer', 'degree_t':'DegreeType', 'workex':'WorkExp', 'etest_p':'ETestPer', 'specialisation':'Specialisation',
                            'mba_p':'MBA_Per', 'status':'PlacementStatus', 'salary':'Salary'})


# <a id="1"></a><br>
# # 1. EDA

# In[ ]:


data.head() #Peeking into the DataFrame


# #### Checking for null values and replacing NaN with 0

# In[ ]:


display(data.isnull().sum()) #Checking for NaN
data['Salary'] = data['Salary'].replace(np.nan, 0) #Replace Nan with 0


# ### Basic Stastistical Analysis

# In[ ]:


data.describe()


# ## Salary Insights
# * >Number of Students = 215
# * >Mean Salary of Studens = 198702
# * >Max Salary = 940000
# * >Students with No Placement = 67

# ## Marks Insights
# * > Avg. SSC Percentage = 66.30
# * > Avg. HSC Percentage = 66.33
# * > Avg. Degree Percentage = 66.37
# * > Avg. E-Test Percentage = 72.10
# * > Avg. MBA Percentage = 62.27

# ________________________________________________________________________

# ## Questions
# * > Which factor influenced a candidate in getting placed?
# * > Does percentage matters for one to get placed?
# * > Which degree specialization is much demanded by corporate?
# * > Play with the data conducting all statistical tests.

# #### Changing 'Not Placesd' to 0 and 'Placed' to 1

# In[ ]:


data1 = data
data1['PlacementStatus'].values[data1['PlacementStatus']=='Not Placed'] = 0 
data1['PlacementStatus'].values[data1['PlacementStatus']=='Placed'] = 1
data1.PlacementStatus = data1.PlacementStatus.astype('int')


# <a id="2"></a><br>
# # 2. Which factor influenced a candidate in getting placed?

# ### Pairplot for attributes

# In[ ]:


sns.pairplot(data,kind='reg')


# ### HeatMap of Attributes affeccting Placements and Salary

# In[ ]:


plt.figure(figsize=(14,12))
data2 = data1.loc[:,data1.columns != 'Id']
sns.heatmap(data2.corr(), linewidth=0.2, cmap="YlGnBu", annot=True)


# * > Candidates with __Higher MBA Percentage__ have a higher chance of getting Placed.
# * > Again __Higher SSC Percentage__ is also beneficial for Placement.
# * > __Higher HSC Percentage and Degree Percentage__ also plays a Significant role in Placement.
# * > Surprisingly __E Test Percentage__ does not affect a candidate's Placement Outcome.
# 

# <a id="3"></a><br>
# # 3. Does percentage matters for one to get placed?

# In[ ]:


fig,axes = plt.subplots(3,2, figsize=(20,12))
sns.barplot(x='PlacementStatus', y='SSC_Per', data=data2, ax=axes[0][0])
sns.barplot(x='PlacementStatus', y='HSC_Per', data=data2, ax=axes[0][1])
sns.barplot(x='PlacementStatus', y='DegreePer', data=data2, ax=axes[1][0])
sns.barplot(x='PlacementStatus', y='ETestPer', data=data2, ax=axes[1][1])
sns.barplot(x='PlacementStatus', y='MBA_Per', data=data2, ax=axes[2][0])
fig.delaxes(ax = axes[2][1]) 


# * > Yes, Percentage matter for placement as we have seen from the HeatMap in previous section.
# * > But, Higher Percentage necessarily doesn't guarantee a Placement.

# <a id ="4"></a><br>
# # 4. Which degree specialization is much demanded by corporate?

# In[ ]:


tempdf = data1[data1['PlacementStatus'] == 1]
plt.figure(figsize= (15,7))

ax = sns.countplot(x='Specialisation', data=data1, palette='bright')
ax.set_title(label='Count of Placed Status on Basis of Specialisation\'s Degree', fontsize=18)

ax.set_xlabel(xlabel='Specialisation\'s Stream', fontsize=16)
ax.set_ylabel(ylabel='No of Students', fontsize=16)

plt.show()


# * > Marketing and Finance Specialization is Most Demanded by Corporate.

# -----------------------------------

# <a id="5"></a><br>
# # 5. Machine learning Algorithms

# ### Making data ready for applying ML algorithms

# Converting String values into Binary Values for easy processing

# In[ ]:


data1 = data1[['Gender','SSC_Per','SSC_Board','HSC_Per','HSC_Board','DegreePer','WorkExp','ETestPer','Specialisation','MBA_Per','PlacementStatus']]

data1['Gender'].replace(to_replace='M', value=1, inplace=True)
data1['Gender'].replace(to_replace='F', value=0, inplace=True)

data1['SSC_Board'].replace(to_replace='Others', value=1, inplace=True)
data1['SSC_Board'].replace(to_replace='Central', value=0, inplace=True)

data1['HSC_Board'].replace(to_replace='Others', value=1, inplace=True)
data1['HSC_Board'].replace(to_replace='Central', value=0, inplace=True)

data1['WorkExp'].replace(to_replace='Yes', value=1, inplace=True)
data1['WorkExp'].replace(to_replace='No', value=0, inplace=True)

data1['Specialisation'].replace(to_replace='Mkt&Fin', value=1, inplace=True)
data1['Specialisation'].replace(to_replace='Mkt&HR', value=0, inplace=True)


# Binary converted Dataframe

# In[ ]:


display(data1.head())


# Splitting data into values and target

# In[ ]:


x = data1[['Gender', 'SSC_Per', 'SSC_Board', 'HSC_Per', 'HSC_Board',
      'DegreePer', 'WorkExp', 'ETestPer', 'Specialisation', 'MBA_Per',]]
y = data1['PlacementStatus']


# Dividing data into test and train sub data

# In[ ]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)


# In[ ]:


print('x_train shape : ',x_train.shape)
print('x_test shape : ',x_test.shape)
print('y_train shape : ',y_train.shape)
print('y_test shape : ',y_test.shape)


# <a id="5.1"></a><br>
# # 5.1 Logistic Regression

# In[ ]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(x_train, y_train)
y_predict = model.predict(x_test)

accuracy1 = model.score(x_test,y_test)

print("Accuracy : ",accuracy1)


# In[ ]:


from sklearn.metrics import confusion_matrix, accuracy_score
confusion_matrix = confusion_matrix(y_test, y_predict)
print(confusion_matrix)
print('\n')
print('Correct Predictions : ',confusion_matrix[0][0] + confusion_matrix[1][1])
print('InCorrect Predictions : ',confusion_matrix[0][1] + confusion_matrix[1][0])


# <a id="5.2"></a><br>
# # 5.2 Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=1000, random_state=42)
model.fit(x_train, y_train)
y_predict = model.predict(x_test)

accuracy2 = model.score(x_test, y_test)

print('Accuracy : ',accuracy2)


# In[ ]:


if accuracy1 > accuracy2:
    print('Logistic Regression is Better approach in this Case ')
else:
    print('Random Forest Classifier is Better approach in this Case ')

