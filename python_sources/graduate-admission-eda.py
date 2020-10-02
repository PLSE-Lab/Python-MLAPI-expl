#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# https://www.kaggle.com/biphili/university-admission-in-era-of-nano-degrees
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# ## Knowing the dataset

# In[ ]:


dataset = pd.read_csv("/kaggle/input/graduate-admissions/Admission_Predict_Ver1.1.csv")


# #### Reviewing 1st five data in the dataset

# In[ ]:


dataset.head()


# #### Removig Serial No. column bcoz its not adding any value to datatset. 

# In[ ]:


dataset.drop(['Serial No.'], axis=1,  inplace=True)


# In[ ]:


column_names = {'GRE Score': 'gre_score', 'TOEFL Score': 'toefl_score', 'University Rating': 'university_rating',                 'SOP': 'sop', 'LOR': 'lor', 'CGPA': 'cgpa',                'Research': 'research', 'Chance of Admit ': 'chance_of_admit'}


# #### Changing column names

# In[ ]:


dataset = dataset.rename(columns = column_names)
dataset.head()


# #### Reviewing last five data in the dataset

# In[ ]:


dataset.tail()


# #### Size of dataset

# In[ ]:


dataset.shape


# In[ ]:


dataset.dtypes


# In[ ]:


for data in dataset.columns:
    print(data)
    print(dataset[data].unique())
    print("="*80)


# #### Five point summury of dataset

# In[ ]:


dataset.describe()


# #### Checking for any null value in dataset

# In[ ]:


dataset.isnull().any()


# #### Ploting histogram on dataset

# In[ ]:


plt.subplots(figsize=(10, 5))
sns.heatmap(dataset.corr(), cmap="YlGnBu", annot=True, fmt= '.0%')
plt.show()


# #### Ploting correlation bar graph based on target variable in ascending order. 

# In[ ]:


plt.subplots(figsize=(10, 5))
dataset.corr().loc['chance_of_admit'].sort_values(ascending=False).plot(kind='bar')


# In[ ]:


sns.pairplot(dataset, corner=True, diag_kind="kde")


# #### How important is Research to get an Admission?

# In[ ]:


print(f"{dataset['research'].value_counts()/len(dataset)}")
print("="*80)
sns.countplot(dataset['research'])


# #### CGPA vs GRE Score Analysis

# In[ ]:


sns.scatterplot(y="cgpa", x="gre_score", hue="university_rating", data=dataset)


# In[ ]:


sns.scatterplot(y="cgpa", x="gre_score", hue="research", data=dataset)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ## Model building
# #### Create the x and y variable for linear regression problem solving
# #### Import the standard scaler library and scale the x data
# #### Import the train test split function frm sklearn and create train and test variable with 30% hold out dataset
# #### Initialize and fit the linear regression
# #### Predict the values from tet dataset
# #### Plot the predicated and relevent actual values
# #### Create a hitogram of residuals
# #### What is R^2 value and what is the interpretation of it
# #### Create a small dataset with column and their respective coefficient values
# #### Create a datset with predication, target, and residual

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




