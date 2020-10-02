#!/usr/bin/env python
# coding: utf-8

# **Introduction:**
# 
# This dataset contains Environmental Health Inspection Results for Restaurants and Markets in the City of Los Angeles. Los Angeles County Environmental Health is responsible for inspections and enforcement activities for all unincorporated areas and 85 of the 88 cities in the County. This dataset is filtered from County data to include only facilities in the City of Los Angeles. 

# **Objective of this notebook:**
# 
# here, I will do data preprocessing and exploratory analysis.

# In[ ]:


import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
color = sns.color_palette()
import plotly.offline as py
py.init_notebook_mode(connected=True)
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.offline as offline
offline.init_notebook_mode()
import plotly.tools as tls


# Reading Data...

# In[ ]:


df = pd.read_csv('../input/restaurant-and-market-health-violations.csv')

print ('Size of the health-violance data : ' , df.shape)


# In[ ]:


df.head()


# Data with simple info

# In[ ]:


df.info()


# Let find a number of categorical features and numeric features in the dataset.

# In[ ]:


cat_col = df.select_dtypes(include = 'object').columns.tolist()
num_col = df.select_dtypes(exclude='object').columns.tolist()

print ('categorical feature :', cat_col)
print ('\nnumeric feature :' ,num_col)
print ('\nnumber of categorical feature : ' , len(cat_col))
print ('\nnumber of numeric feature : ' , len(num_col))


# Description of categorical features

# In[ ]:


df.describe(include=["O"])


# **Data pre-processing**
# * Missing Data

# In[ ]:


def check_missing(df):
    
    total = df.isnull().sum().sort_values(ascending = False)
    percent = (df.isnull().sum()/df.isnull().count()*100).sort_values(ascending = False)
    missing_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    return missing_data
    

missing_data_df = check_missing(df)
missing_data_df.head()


# Only **program_name** feature have missing values.

# * Redundant features

# In[ ]:


def find_uni(df):
    col_list = df.columns
    redundant_col =[]
    for col in col_list:
        if df[col].nunique() == 1:
            redundant_col.append(col)
    return redundant_col


redundant_col = find_uni(df)
print ('Number of redundant features in data :',len(redundant_col))
print ('Redundant Feature :', redundant_col)


# This dataset contains Health Inspection Results for Restaurants and Markets in same city  Los Angeles and the same state of CA. Hence, these features have unique value one.
# 
# So, these two features are redundant for the ML algorithm. They didn't contribute any useful information. It's better to remove these feature.
# 

# In[ ]:


df.drop(redundant_col,axis=1,inplace =True)


# * Find Duplicate features 

# let's check **service_description** and **service_code**  feature.
# 

# In[ ]:


df['service_description'].value_counts()


# In[ ]:


df['service_code'].value_counts()


# Here, **service_description** feature describes the code for inspection service.
# For machine learning perspective both are the duplicate feature. The duplicate feature doesn't help for machine learning algorithm. hence, it's better to remove any one of the feature.

# In[ ]:


df.drop('service_code' , axis =1 , inplace=True)


# Let's check **program_element_pe** feature and **pe_description** feature.

# In[ ]:


df['program_element_pe'].value_counts()


# In[ ]:


df['pe_description'].value_counts()


# **program_element_pe** feature describe the code of the **pe_description** feature. Again these two features are duplicate. so, remove **program_element_pe** feature.
# 

# In[ ]:


df.drop('program_element_pe' , axis =1, inplace =True)


# **Exploratory Data analysis**

# let's check **score**, **grade** and **points** features

# In[ ]:


df[['score','points',]].describe()


# In[ ]:


df['score'].hist().plot()


# 50%  data have more than 90 score value.

# Let's check unique value in **points** feature

# In[ ]:


df['points'].value_counts()


# In[ ]:


temp = df['grade'].value_counts()
labels = temp.index
sizes = (temp / temp.sum())*100
trace = go.Pie(labels=labels, values=sizes, hoverinfo='label+percent')
layout = go.Layout(title='grade')
data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)


# 86% places got A grade. 

# In[ ]:


le = LabelEncoder()
df['grade'] = le.fit_transform(df['grade'])
df['grade'].corr(df['score'])


# **grade** feature is highly correlated with **score** feature. Grade feature has been calculated from the score value range.

# Top 15 highly health-violated place.

# In[ ]:


top_violated_place = df["facility_name"].value_counts().head(15)
pd.DataFrame({'Count':top_violated_place.values},index = top_violated_place.index)


# In[ ]:


temp = df["facility_name"].value_counts().head(25)

trace = go.Bar(
    x = temp.index,
    y = temp.values,
)
data = [trace]
layout = go.Layout(
    title = "Distribution of facility_name",
    xaxis=dict(
        title='facility_name',
        tickfont=dict(
            size=10,
            color='rgb(107, 107, 107)'
        )
    ),
    yaxis=dict(
        title='how many times health-violations occur',
        titlefont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        ),
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
)
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='facility_name')


# **DODGER STADIUM** is one of the most health-violation place.

# Top 10 **facility_name** which got best score

# In[ ]:


temp1 = df[['facility_name','score']].sort_values(['score'],ascending = False).drop_duplicates()
temp1.head(10)


# Top 10  **facility_name** which got worst score

# In[ ]:


temp1 = df[['facility_name','score']].sort_values(['score']).drop_duplicates()
temp1.head(10)


# In[ ]:




