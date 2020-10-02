#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import os
import seaborn as sns


# # Create DataFrame
# Based on the image names, the manufacturer, series and year are extracted and put into a dataframe
# 

# In[ ]:


data_original = os.listdir('../input/the-car-connection-picture-dataset') #


# In[ ]:


#Finds index 
def FindIndexInString(string,desired_counter):
    counter = 0
    from_idx = 0
    to_idx = 0
    for idx, char in enumerate(string): #Goes through each char in string
        if char == '_':
            counter += 1
            if counter == desired_counter-1: #Finds the from_idx, to go from example: "this'_'is_a_string"
                from_idx = idx+1
        if counter == desired_counter: #Finds the to_idx "this_is'_'a_string"
            to_idx = idx
            return from_idx, to_idx


# In[ ]:


#Creates dataframe based on input data
def CreateDataFrame(data):
    df = pd.DataFrame(columns=['Manufacturer','Series','Year'])
    
    for i,column in enumerate(df):
        string_list = []
        for idx, string in enumerate(data):
            from_idx,to_idx = FindIndexInString(string,i+1)
            string = string[from_idx:to_idx]
            string_list.append(string) 
        df[column] = string_list
        
    image_df = pd.DataFrame(data,columns=['Image'])
    result_df = pd.concat([df,image_df],axis=1)
    return result_df


# In[ ]:


df = CreateDataFrame(data_original)
df


# # Plottings
# As can be seen, the dataset is imbalanced. Therefore, if it's desired to classify manufacturer of a car, reducing imbalancement should be considered.

# In[ ]:


sns.set(rc={'figure.figsize':(30,10)})
sns.set(font_scale = 0.6)
sns.countplot(x='Manufacturer',data=df)


# In[ ]:


sns.countplot(x='Year',data=df)


# In[ ]:




