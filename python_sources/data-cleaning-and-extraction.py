#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Import the necessary modules for data cleaning and processing
import numpy as np
import pandas as pd


# In[ ]:


#Load the excel file into your worksheet
df = pd.ExcelFile('../input/lga-sen-districts-dataset/LGA_SEN_Districts.xlsx')


# In[ ]:


#Select the worksheet to work on and save it as data
data = df.parse('LGA_SEN_Districts')


# In[ ]:


#Information about the data
data.info()


# In[ ]:


#Outputs the first 50 rows of the data
data.head(50)


# In[ ]:


#Outputs the last 5 rows of the data
data.tail()


# In[ ]:


#Outputs all the columns in the data
data.columns = ['State', 'Senatorial District', 'Code', 'Composition', 'Collation Center']


# In[ ]:


#Drops all the rows with null values
data = data.dropna()


# In[ ]:


data.info()


# In[ ]:


data.head(20)


# In[ ]:


data.head(80)


# In[ ]:


#Filters the rowa and column to get the clean data
data = data[data.State != 'S/N']


# In[ ]:


data.head(80)


# In[ ]:


#Get the 'State' column from the 'Senatorial District' column
state_list = []
for cell in data['Senatorial District']:
    sp = cell.split(' ')
    state = sp[0]
    state_list.append(state)
data['State'] = state_list


# In[ ]:


data


# In[ ]:


#Reset the index
data = data.reset_index(drop=True)


# In[ ]:


data


# In[ ]:


#For Akwa Ibom, Cross River, and FCT, slice using index and name the State properly
data.iloc[3:6,0] = 'AKWA IBOM'
data.iloc[24:27,0] = 'CROSS RIVER'
data.iloc[-1,0] = 'FCT'


# In[ ]:


data


# In[ ]:


#Export clean data
data.to_csv('clean_file.csv', header = True, index = False)

