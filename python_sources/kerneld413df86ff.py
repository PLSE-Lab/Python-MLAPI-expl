#!/usr/bin/env python
# coding: utf-8

# 
# # EDA Haber Man Dateset
# 

# ## Description:
# The dataset contains cases from a study that was conducted between 1958 and 1970 at the University of Chicago's Billings Hospital on the survival of patients who had undergone surgery for breast cancer.
# 

# Dataset comprises of following Attributes:
# 
# a: Age of patient at time of operation (numerical)
# 
# b: Patient's year of operation (year - 1900, numerical)
# 
# c: Number of positive axillary nodes detected (numerical)
# 
# d: Survival status (class attribute)
# 
#     1 = the patient survived 5 years or longer 
# 
#     2 = the patient died within 5 year

# In[ ]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


haberman=pd.read_csv("../input/haberman/haberman.csv")


# In[ ]:


print(haberman.shape)


# In[ ]:


haberman.info()


# As we can see all the attributes are of int data type.
# 
# All the attributes are having all non-null values.

# In[ ]:


haberman.shape
#dataset is having 306 datapoints


# In[ ]:


haberman.columns


# In[ ]:


haberman['survival_status'].unique()
haberman['survival_status']=haberman['survival_status'].apply(lambda x:'survive' if x==1 else 'died')
#converting target variable from numerical to categorical data set.


# In[ ]:


haberman.head()


# In[ ]:


haberman.tail()


# In[ ]:


print(haberman["survival_status"].value_counts())


# 1. Dataset is highly Imbalanced.
# 
# 2. In Dataset all the features are of int datatypes and having non-null values.
# 
# 3. More number of women have survived.

# ## Objective:
# 
# The objective of this data analysis is to classify whether a patient gone through surgery for breast cancer will survive more then 5 years or not, using the given features.

# # PairPlots

# In[ ]:


plt.close
sns.set_style("whitegrid")
sns.pairplot(haberman,hue="survival_status",vars=['Age','year','positive_axillary_nodes'],size=3).add_legend()
plt.show()


# 
# 
# 1. Data is highly mixed, it is hard to interpret.
# 
# 2. There is no clear cut separation between the data points of different classes.
# 
# 3. The data points are occurring in between the range but since we can't read the scales properly we will look into it indiviually.

# In[ ]:


sns.set_style('whitegrid')
sns.FacetGrid(haberman,hue="survival_status",size=4)    .map(plt.scatter,"positive_axillary_nodes","Age")    .add_legend()


# 1. The age of the women gone under though surgery is in between 30 to 80.
# 2. Most of the nodes are in the range 0 to 30.
# 3. There are more data points of survive class between range 0 t0 5.

# In[ ]:


sns.set_style('whitegrid')
sns.FacetGrid(haberman,hue="survival_status",size=4)    .map(plt.scatter,"positive_axillary_nodes","year")    .add_legend()


# The years of surgery are in between 56 to 70.
# 

# In[ ]:


sns.set_style('whitegrid')
sns.FacetGrid(haberman,hue="survival_status",size=4)    .map(plt.scatter,"year","Age")    .add_legend()


# The patient had surgery between year 66 to 70 is more likely to survive.
# 

# ## PDF

# In[ ]:


sns.FacetGrid(haberman,hue='survival_status',size=5)    .map(sns.distplot,'year')    .add_legend()


# 1. In year between 58.5 to 62.5 the number of survivors are more.
# 2. In year between 62.5 to 66.5 the number of people under under 5 years is more.
# 3. In plot the proportion of classes vary unevenly.

# In[ ]:


sns.FacetGrid(haberman,hue='survival_status',size=5)    .map(sns.distplot,'Age')    .add_legend()


# The Patient of age between 40 to 55 are more likely to die within 5 yrs.

# In[ ]:


sns.FacetGrid(haberman,hue='survival_status',size=5)    .map(sns.distplot,'positive_axillary_nodes')    .add_legend()


# In[ ]:


sns.FacetGrid(haberman,hue='survival_status',size=5,xlim=[-5,5])    .map(sns.distplot,'positive_axillary_nodes')    .add_legend()


# 1. Between Range 0 to 3 of nodes, the number of survivors are more compare to dead.
# 2. As the number of nodes increases the number of survivor decreases exponentially.
# 3. Patient having number of nodes more then 3 are less likely to survive.

# ## BoxPlot

# In[ ]:


sns.boxplot(x='survival_status',y='positive_axillary_nodes',data=haberman)


# 1. There are large number of outliers in survive class.
# 2. The 75 percentile of survive class is less then 50 percentile of died class.
# 3. The quartile range of survive class is very small.
# 4. The data having nodes value greater then its 75 percentile of class survive is more likely to fall under class died.

# In[ ]:


sns.boxplot(x='survival_status',y='year',data=haberman)


# The median of both the classes are approximatly equal.

# In[ ]:


sns.boxplot(x='survival_status',y='Age',data=haberman)


# Almost all part of the survival class is overlapping the died class.

# # Conclusion

# 1. Most of Data points for Nodes feature are outliers.
# 2. For Most of the data points for a given feature classes are mostly overlapped.
# 3. There is a irregular pattern in a dataset.
# 4. We cannot find a clear cut separation between the classes by using the given features.

# In[ ]:




