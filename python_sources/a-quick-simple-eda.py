#!/usr/bin/env python
# coding: utf-8

# # Exploratory data analysis
# 
# Just trying to get a quick feel for the data set.
# 
# Simple histograms and summary statistics are provided on the Kaggle website and will not be explored here. 
# 
# I want to convert the `soil_type` and the `wilderness_area` from binary columns to one categorical column (dtype = int) so I can look at that data first.<br>
# 

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(rc={'figure.figsize':(11,8)})


# In[ ]:


train_df = pd.read_csv("../input/learn-together/train.csv")
train_df.head(3)


# In[ ]:


soil_type_list = list(train_df.columns[15:-1])
soil_type_EDA_column = train_df[soil_type_list].idxmax(axis=1)
soil_type_EDA_column = soil_type_EDA_column.str.replace('Soil_Type', '')
soil_type_EDA_column = soil_type_EDA_column.astype('int32')
print(soil_type_EDA_column[:3])


# In[ ]:


_ = plt.hist(soil_type_EDA_column, bins=40)
_ = plt.xlabel('Soil Type')
_ = plt.ylabel('No of instances')
plt.show()


# In[ ]:


wilderness_area_list = list(train_df.columns[11:15])
wilderness_area_EDA_column = train_df[wilderness_area_list].idxmax(axis=1)
wilderness_area_EDA_column = wilderness_area_EDA_column.str.replace('Wilderness_Area', '')
wilderness_area_EDA_column = wilderness_area_EDA_column.astype('int32')
print(wilderness_area_EDA_column[:3])


# In[ ]:


_ = plt.hist(wilderness_area_EDA_column)
_ = plt.xlabel('Wilderness Area')
_ = plt.ylabel('No of instances')
plt.show()


# # A quick function and for loop to (box) plot the first ten features

# In[ ]:


features_list = list(train_df.columns[1:10])

def make_boxplot(x, y, data):
    _ = sns.boxplot(x=x, y=y, data=data)
    _ = plt.xlabel('Cover Type')
    _ = plt.ylabel('{}'.format(y))
    plt.show()


# In[ ]:


for i in features_list:
    make_boxplot('Cover_Type', i, train_df)


# # Looks like elevation is a pretty important feature!!

# In[ ]:




