#!/usr/bin/env python
# coding: utf-8

# # Problem statement

# - We need to predict whether a person has a salary above 50k or less than 50k
# - This is a <b>Classification</b> problem
# - URL of the dataset https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data

# ## Importing Libraries

# In[ ]:


import warnings
warnings.filterwarnings('ignore')

import numpy as np  # I may not be using it

# For EDA and cleaning the data
import pandas as pd

# For visualizations
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

# For building a model
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


# ### Loading the data

# In[ ]:


income_df = pd.read_csv('../input/adult.csv', names=['age', 'workclass', 'fnlwgt', 'education', 'education-num',
                                           'marital-status', 'occupation', 'relationship', 'race',
                                           'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
                                           'native-country', 'salary'])


# ### Miscellaneous

# In[ ]:


income_df.head()


# In[ ]:


income_df.shape


# There are <b>32561</b> rows and <b>15</b> columns in the data

# In[ ]:


income_df.info()


# There are <b>no null</b> values in the dataset<br>
# There are <b>six numerical</b> columns<br>
# There are <b>nine text</b> columns

# In[ ]:


income_df.describe()


# Some statistics about the numerical columns in the data

# ## EDA

# In[ ]:


income_df.columns


# In[ ]:


income_df.head()


# In[ ]:


sns.distplot(income_df.age)


# <b>Age column is normally distributed</b>

# In[ ]:


sns.countplot(income_df.salary)


# <b>There are about 24000 people who have a salary less than 50k and remaining have above 50k</b>

# In[ ]:


sns.countplot(income_df.salary, hue=income_df.sex, palette='rainbow')


# <b>Males have more salary than females</b>

# In[ ]:


sns.barplot(income_df.salary, income_df['capital-gain'])


# <b>People who have a salary above 50k have higher capital-gain</b>

# In[ ]:


income_df.occupation.unique()


# In[ ]:


plt.xticks(rotation=90)
sns.countplot(income_df.occupation, hue=income_df.salary, palette='Blues_r')


# <b>Exec-managerical have highest salary above 50k</b>

# In[ ]:


income_df.relationship.unique()


# In[ ]:


plt.xticks(rotation=90)

sns.countplot(income_df.relationship, hue=income_df.salary, palette='Accent')


# <b>Husbands are earning more than 50k</b><br>
# <b>Not-in Family relationship people are earning less than 50k</b>

# In[ ]:


income_df.workclass.value_counts()


# In[ ]:


plt.xticks(rotation=90)
sns.countplot(income_df.workclass, hue=income_df.salary)


# <b>Private employees are earning more than any other type of employee</b>

# In[ ]:


income_df.race.unique()


# In[ ]:


plt.xticks(rotation=90)
sns.barplot(income_df.workclass, income_df['hours-per-week'], hue=income_df.salary, palette='cool')
sns.set()
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


# In[ ]:


plt.xticks(rotation=90)
sns.countplot(income_df.race, hue=income_df.salary)


# <b>White people are earning more</b>

# In[ ]:


income_df.head()


# ## Removing unnecessary columns

# Removing below columns from data
# - fnlwgt
# - education-num

# In[ ]:


income_df.drop(['fnlwgt', 'education-num'], axis=1, inplace=True)


# In[ ]:


income_df.head()


# ## Converting categorical columns to numerical columns

# Categorical columns
# - workclass
# - education
# - marital-status
# - occupation
# - relationship
# - race
# - sex
# - native-country<br>
# 
# We'll use one-hot encoding to convert categorical columns to numerical columns.

# In[ ]:


dummies = pd.get_dummies(income_df.drop(['salary', 'age', 'capital-gain', 'capital-loss',
                                        'hours-per-week'], axis=1))


# In[ ]:


dummies.shape


# In[ ]:


dummies.head()


# In[ ]:


merged = pd.concat([income_df, dummies], axis=1)


# In[ ]:


merged.shape


# In[ ]:


merged.head()


# In[ ]:


merged.columns[:100]


# In[ ]:


final_df = merged.drop(['workclass', 'education', 'marital-status', 'occupation', 'relationship',
                       'race', 'sex', 'native-country'], axis=1)


# In[ ]:


final_df.head()


# In[ ]:


final_df.shape


# ## Splitting the data

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(final_df.drop('salary', axis=1), final_df.salary, 
                                                   test_size=0.30, random_state=4)


# In[ ]:


X_train.shape


# In[ ]:


X_test.shape


# ## Training the model

# In[ ]:


gbm = GradientBoostingClassifier()


# In[ ]:


gbm.fit(X_train, y_train)


# ## Evaluation

# In[ ]:


predictions = gbm.predict(X_test)


# In[ ]:


predictions


# In[ ]:


print(metrics.classification_report(y_test, predictions))


# In[ ]:


metrics.accuracy_score(y_test, predictions)

