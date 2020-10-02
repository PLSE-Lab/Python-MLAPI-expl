#!/usr/bin/env python
# coding: utf-8

# # EDA on Haberman's Survival Data

# 
# 
# ## Objective:
# 

# The objective is to classify whether the patient survive after operation of breast cancer or not.

# 
# ## Data Description:

# 
# Data is collected from https://www.kaggle.com/gowtamsingulur/habermancsv.
# 
# The data set contains cases from a study that was conducted between 1958 and 1970 at the University of Chicago's
# Billings Hospital on the survival of patients who had undergone surgery for breast cancer.
# 
# 

# Importing Libraries

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Loading data

# In[ ]:


data=pd.read_csv("../input/habermancsv/haberman.csv")
data.head()


# In[ ]:


data.columns


# + 'age'- Age of patient at the time of operation.
# 
# + 'year'- Year of operation(i.e 1900).
# 
# + 'Nodes'- No. of positive Axillary Lymph Nodes(Lymph Nodes are small, bean-shaped organs that acts as filter, which are                  present in underarm. If Lymph Nodes have some cancer cells in them, they are called positive.)
# 
# + 'status'- It is the survival status of patient.

# In[ ]:


data.count()


# In[ ]:


data.isnull().sum()


# In[ ]:


data['status'].value_counts()


# Observations
# 
# + There are 4 columns, out of these 'status' is the output column.
# + In status there are 2 class- 1-> 'survival', 2-> 'Death'.
# + There are total of 306 entity.
# + There is no missing value in the dataset.

# In[ ]:


data['status']=data['status'].apply(lambda x: 'survived' if x==1 else 'died')


# For better understanding let 1-> 'survived' and 2-> 'died'

# In[ ]:


s=sns.FacetGrid(data,hue='status',size=6)
s=s.map(sns.distplot,'age')
s.add_legend()
plt.show()


# In[ ]:


s=sns.FacetGrid(data,hue='status',size=6)
s.map(sns.distplot,'year')
s.add_legend()
plt.show()


# In[ ]:


s=sns.FacetGrid(data,hue='status',size=6)
s.map(sns.distplot,'nodes')
s.add_legend()
plt.show()


# Observation
# 
# + From the 1st plot we can say that there is more chance that the patient having age less than 35 can survived.
# + and patient having age more than 75 have less chance of survival.
# + majority of patient survive have less than 5 postive nodes.
# + But using this plot we can't distinguish 2 classes clearly.

# In[ ]:


sns.set_style('whitegrid')
sns.pairplot(data,hue='status',height=5)
plt.show()


# Observation 
# 
# + By looking the scatter plots we can't distinguish class.

# In[ ]:


sns.boxplot(x='status',y='age',data=data)
plt.show()


# In[ ]:


sns.boxplot(x='status',y='year',data=data)
plt.show()


# In[ ]:


sns.boxplot(x='status',y='nodes',data=data)
plt.show()


# Observation
# 
# + we may conclude the patient having 3-4 or less no. of positive nodes are survived.
# + 75% of survived patient having less than 4 positive nodes.
