#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# import necessary packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# set plot style
sns.set_style()


# In[ ]:


# import dataset from local file
dataset = pd.read_csv("../data/csv/South Asian Wireless Telecom Operator (SATO 2015).csv")
# check dataset shape
print("Number of observations in data ",dataset.shape)
print("\n\nDataset info ",dataset.info())


# In[ ]:


# check head of data
dataset.head()


# In[ ]:


# check distribution of each numeric features
dataset.describe()


# In[ ]:


# check target class is balanced or not
dataset.Class.value_counts()


# In[ ]:


# convert Clas s column to numeric form 
def convert_label_to_numeric_form(row):
    if row.Class == "Churned":
        return 0
    elif row.Class == "Active":
        return 1


# In[ ]:


# add new column as Class_Converted containing class labels in 0 or 1 form
dataset["Class_Converted"]=dataset.apply(convert_label_to_numeric_form,axis=1)


# In[ ]:


dataset[dataset["Class_Converted"]==0]["Class"].value_counts()


# In[ ]:


dataset[dataset["Class_Converted"]==1]["Class"].value_counts()


# ### It seem's like no huge difference between avg complaints made by churned subscriber and active subscriber

# In[ ]:


# lets check whether any relation exists between aggregate complaints counts by customer and customer churn

print("\nComplain count distribition for churned subscriber \n ",dataset[dataset["Class_Converted"]==0]["Aggregate_complaint_count"].describe())
print("\nComplain count distribition for active subscriber \n ",dataset[dataset["Class_Converted"]==1]["Aggregate_complaint_count"].describe())


# In[ ]:


# lets check Complain count distribition for each type of subscriber where complaint count is more than 2

print("\nComplain count distribition for churned subscriber \n ",dataset[(dataset["Class_Converted"]==0)&(dataset["Aggregate_complaint_count"]>2)]["Aggregate_complaint_count"].describe())
print("\nComplain count distribition for active subscriber \n ",dataset[(dataset["Class_Converted"]==1)&(dataset["Aggregate_complaint_count"]>2)]["Aggregate_complaint_count"].describe())


# ### It looks like active customers have greater network age as compare to churned customers

# #### found -ve network age for churned custmers (only 5 entries). removed these rows and check distribution again

# In[ ]:


# lets check whether any relation exists between from how long time customer taking services and customer churn

print("\nNetwork age distribition for churned subscriber \n ",dataset[dataset["Class_Converted"]==0]["network_age"].describe())
print("\nNetwork age distribition for active subscriber \n ",dataset[dataset["Class_Converted"]==1]["network_age"].describe())


# In[ ]:


print("\n-ve Network age distribition for churned subscriber \n ",dataset[(dataset["Class_Converted"]==0)&(dataset["network_age"]<0)]["network_age"])


# In[ ]:


print("\nNetwork age distribition for churned subscriber \n ",dataset[(dataset["Class_Converted"]==0)&(dataset["network_age"]>=0)]["network_age"].describe())
print("\nNetwork age distribition for active subscriber \n ",dataset[dataset["Class_Converted"]==1]["network_age"].describe())


# In[ ]:


# create new object by removing entries for -ve network age for churned customers
dataset_copy = dataset[dataset.network_age>=0].copy()
dataset.shape,dataset_copy.shape


# ### It look's like churned subscribers are more for 2G service as compare to 3G service for both aug and sep month

# #### it looks like subscriber's with 2G service has more churned customers

# In[ ]:


# check user type count for churned and active customers for the month of aug and sep
print("User type count for churned customer for the month of aug \n",
     dataset_copy[dataset_copy.Class_Converted==0].aug_user_type.value_counts())

print("\nUser type count for active customer for the month of aug \n",
     dataset_copy[dataset_copy.Class_Converted==1].aug_user_type.value_counts())


print("\nUser type count for churned customer for the month of sep \n",
     dataset_copy[dataset_copy.Class_Converted==0].sep_user_type.value_counts())

print("\nUser type count for active customer for the month of sep \n",
     dataset_copy[dataset_copy.Class_Converted==1].sep_user_type.value_counts())


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




