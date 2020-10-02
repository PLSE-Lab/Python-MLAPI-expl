#!/usr/bin/env python
# coding: utf-8

# # Data Examination and Cleaning

# 
# Tutorial Level : ***Beginner***
# 
# Data Cleaning is detection, fillup or removal of incomplete, inaccurate or vague data.
# The tutorial below discuss the examination of data and performing necessary steps to clean data including 
# 
# * Dealing with Missing Data 
#     *     Removal and Filling of Missing Data for Numerical and Categorical Features
# * Dividing Numerical and Categorical Data and Converting Categorical Data to Numeric Form
# * Inspecting Feature Correlation to study important features
# * Removal of less Corrlated Features
# * Plotting histogram and kde plots to get a butter understanding of data
# 

# Importing required libraries

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # for plots
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import os
import warnings
warnings.filterwarnings('ignore')


# Data being used here for data examination is 'Horse Colic Dataset' which predicts whether a horse can survive or not based on past medical conditions.
# Data is available via following links.
# 1.  [Kaggle](http://www.kaggle.com/uciml/horse-colic)
# 2. [UCI Machine Learning Repository](http://archive.ics.uci.edu/ml/datasets/Horse+Colic)

# Reading data from **CSV file** and saving as **Pandas' Dataframe**

# In[ ]:


print(os.listdir("../input"))
data = pd.read_csv('../input/horse.csv')
data.head()


# ## Shape and Nature of data

# In[ ]:


print("Shape of data (samples, features): ",data.shape)


# **Inspecting nature of data**, it can be seen that data consists of **17** categorical features and rest numerical out of **28**.

# In[ ]:


data.dtypes.value_counts()


# ## Checking missing values for each feature

# In[ ]:


nan_per=data.isna().sum()/len(data)*100
plt.bar(range(len(nan_per)),nan_per)
plt.xlabel('Features')
plt.ylabel('% of NAN values')
plt.plot([0, 25], [40,40], 'r--', lw=1)
plt.xticks(list(range(len(data.columns))),list(data.columns.values),rotation='vertical')


# The graph shows the number of Missing values in each feature, most of the features have less than 40% missing values. 

# ## Dividing Categorical and Numerical Data

# In[ ]:


obj_columns=[]
nonobj_columns=[]
for col in data.columns.values:
    if data[col].dtype=='object':
        obj_columns.append(col)
    else:
        nonobj_columns.append(col)
print(len(obj_columns)," Object Columns are \n",obj_columns,'\n')
print(len(nonobj_columns),"Non-object columns are \n",nonobj_columns)

data_obj=data[obj_columns]
data_nonobj=data[nonobj_columns]


# ## Removing and Filling Missing Values in Numerical and Categorical Data 
# 1.     For columns with more than 40% NAN Value : Remove Columns
# 2.     For columns with less than 40% NAN Value 
#     *       ***For Numerical Data***: Replace NAN values with median value of that particular column
#     *      ** *For Categorical Data***: Replace NAN values with mode value of that particular column

# In[ ]:


print("Data Size Before Numerical NAN Column(>40%) Removal :",data_nonobj.shape)
for col in data_nonobj.columns.values:
    if (pd.isna(data_nonobj[col]).sum())>0:
        if pd.isna(data_nonobj[col]).sum() > (40/100*len(data_nonobj)):
            print(col,"removed")
            data_nonobj=data_nonobj.drop([col], axis=1)
        else:
            data_nonobj[col]=data_nonobj[col].fillna(data_nonobj[col].median())
print("Data Size After Numerical NAN Column(>40%) Removal :",data_nonobj.shape)


# In[ ]:


print("Data Size Before Categorical NAN Column(>40%) Removal :",data_obj.shape)
for col in data_obj.columns.values:
    if (pd.isna(data_obj[col]).sum())>0:
        if pd.isna(data_obj[col]).sum() > (40/100*len(data_nonobj)):
            print(col,"removed")
            data_obj=data_obj.drop([col], axis=1)
        else:
            data_obj[col]=data_obj[col].fillna(data_obj[col].mode()[0])
print("Data Size After Categorical NAN Column(>40%) Removal :",data_obj.shape)


# ## Converting Categorical Data to Numerical and Merging Them

# In[ ]:


for col in data_obj.columns.values:
    data_obj[col]=data_obj[col].astype('category').cat.codes
data_merge=pd.concat([data_nonobj,data_obj],axis=1)

target=data['outcome']
print(target.value_counts())
target=data_merge['outcome']
print(target.value_counts())


# It shall be noted that numeric **0,1,2** are equivalent to **died, euthanized, lived** for outcome.

# ## Inspecting Correlation between various Features and Outcome

# Correlation shows how strongly features are related to each other. We will be checking correlation of each column with outcome. 
# * If correlation value is positive, fetaure is positively correlated to outcome. 
# 1. If correlation value is negative, feature is negatively correlated to outcome. 
# 1. If correlation value is 0, two columns are not correlated. 
# 
#     *  |value| > 0.7 : Hight correlated    
#     *  0.7 < |value| > 0.3 : Moderately correlated    
#     *  0.3 < |value| > 0 : Weakly correlated    

# In[ ]:


train_corr=data_merge.corr()
sns.heatmap(train_corr, vmax=0.8)
corr_values=train_corr['outcome'].sort_values(ascending=False)
corr_values=abs(corr_values).sort_values(ascending=False)
print("Correlation of mentioned features wrt outcome in ascending order")
print(abs(corr_values).sort_values(ascending=False))


# Removing unwanted very less correlated features

# In[ ]:


print("Data Size Before Correlated Column Removal :",data_merge.shape)

for col in range(len(corr_values)):
        if abs(corr_values[col]) < 0.1:
            data_merge=data_merge.drop([corr_values.index[col]], axis=1)
            print(corr_values.index[col],"removed")
print("Data Size After Correlated Column Removal :",data_merge.shape)


# To better understand, how two features are correlated. Let us plot two most correlated (to outcome) features as histogram and kde plot.

# ## 1. Packed Cell Volume & Outcome

# In[ ]:


#packed_cell_volume 
col='packed_cell_volume'
fig,(ax1,ax2)=plt.subplots(1,2, figsize=[20,10])

y=data_merge[col][target==2]
x=data_merge['outcome'][target==2]
y.plot.hist(ax=ax2)
sns.kdeplot(y,ax=ax1)

y=data_merge[col][target==0]
x=data_merge['outcome'][target==0]
y.plot.hist(ax=ax2)
sns.kdeplot(y,ax=ax1)

y=data_merge[col][target==1]
x=data_merge['outcome'][target==1]
y.plot.hist(ax=ax2)
sns.kdeplot(y,ax=ax1)

plt.title(col)
ax1.legend(['lived','died','euthanized'])
ax2.legend(['lived','died','euthanized'])
plt.show()


# The plots show that after approx **50**, outcome is most likely to be **euthanized**, and after **60**, it is likely to be **died**. 

# ## 2. Pulse & Outcome

# In[ ]:


#pulse 
col='pulse'
fig,(ax1,ax2)=plt.subplots(1,2, figsize=[20,10])
y=data_merge[col][target==2]
x=data_merge['outcome'][target==2]
y.plot.hist(ax=ax2)
sns.kdeplot(y,ax=ax1)
y=data_merge[col][target==0]
x=data_merge['outcome'][target==0]
y.plot.hist(ax=ax2)
sns.kdeplot(y,ax=ax1)
y=data_merge[col][target==1]
x=data_merge['outcome'][target==1]
y.plot.hist(ax=ax2)
sns.kdeplot(y,ax=ax1)
plt.title(col)
ax1.legend(['lived','died','euthanized'])
ax2.legend(['lived','died','euthanized'])
plt.show()


# The plots show that after approx **60**, outcome is likely to be **died** which is then replaced by **euthanized** after **100**.  And after **150** , the probability of **died** being the outcome is highest.
