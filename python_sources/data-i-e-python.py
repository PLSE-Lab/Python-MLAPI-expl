#!/usr/bin/env python
# coding: utf-8

# # Import pandas library 

# In[ ]:


import pandas as pd


# # Import CSV file

# In[ ]:


data=pd.read_csv('../input/student-performance-dataset/Department_Information.csv')


# In[ ]:


print(data)


# In[ ]:


data.head()


# In[ ]:


data.head(3)


# In[ ]:


data.tail(3)


# # How to export the csv file

# In[ ]:


data.to_csv('../file.csv')


# # Import xlsx file

# In[ ]:


data1=pd.read_excel('../input/student-performance-dataset/Department_Information.xlsx')


# 

# In[ ]:


data1.to_csv('../file.xlsx')


# # Import text file

# In[ ]:


data3=pd.read_csv("../input/student-performance-dataset/Department_Information.txt",sep=",")


# # To Export the Txt File

# In[ ]:


data3.to_csv("../file.txt")


# # To import sas file

# In[ ]:


data4=pd.read_sas("../input/student-performance-dataset/department_information.sas7bdat")


# # Use of head function

# In[ ]:


data4.head(10)


# # To know the types of columns

# In[ ]:


data4.dtypes


# # To know the types of data

# In[ ]:


data4.dtypes


# # To import csv file in specific folder

# In[ ]:


data5=pd.read_csv('../input/marks-csv/Marks1.csv')


# In[ ]:


print(data5)


# In[ ]:


data5.dtypes


# # Note1: object is nothing but the string.
#   # Note2: Machine Learning models can't work on string values so we need to convert into numeric format.

# # To Change datatypes to Float.

# In[ ]:


data5['Id']=data5.Id.astype('float')


# In[ ]:


print(data5['Id'])


# # To Change datatypes to object(i.e. String).

# In[ ]:


data5['Id']=data5.Id.astype('object')


# In[ ]:


data5.dtypes


# In[ ]:




