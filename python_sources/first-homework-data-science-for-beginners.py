#!/usr/bin/env python
# coding: utf-8

# <a id="0"></a> 
# # ANALYZE "BLACK FRIDAY" DATA
# In this homework , I will analyze "Black Friday" data with Python libraries that I learned so far.
# 
# **Content:**
# 1. [Read and Analyze Data by using Pandas library](#1)
# 2. [Filter Data and Add New Columns](#2)
# 3. [Draw Graphs by using Matplotlib library](#3)
# 
# 
# 
# 

# <a id="1"></a> 
# # 1. Read and Analize Data by using Pandas library
# 
# In this session , we will also import all libraries which will be used in homework . You can import in anywhere , but now I prefer to do like this 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns  # visualization tool

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


# read BlackFriday.csv file data from input directory and create dataframe named data
data = pd.read_csv("../input/BlackFriday.csv")


# After reading csv file , we will check data by using **info()** method of dataframe , we can see column names and types . <br>
# If we execute command , we will see there are 537577  entries in file .<br>
# For Product_Category_1 column , 164278 of 537577 are non-null and that means rest are null<br>
# For Product_Category_2 column , 370591 of 537577 are non-null and that means rest are null<br>
# Other columns are all full and there is no empty data 
# 

# In[ ]:


# check information about columns data 
data.info()


# In[ ]:


# Let's see first 10 data of dataframe to have knowledge data itself.
data.head(10)


# we can see there are NaN mvalues for Product_Category_2 and Product_Category_3. I want to fill them with zero 

# In[ ]:


data.Product_Category_2.fillna(0, inplace=True)
data.Product_Category_3.fillna(0, inplace=True)


# Other way to see column names (but only names not information about data ) is using **columns** variable of dataframe

# In[ ]:


# display columns 
data.columns


# By using **describe** method of dataFrame , we can learn some statistical information such as mean(average), max, min etc about data 

# In[ ]:


# check statictical information about numeric data columns 
data.describe()


# In[ ]:


data.corr()


# Seaborn correlation map will help us for selecting features. 
# Parameters for Seaborn heatmap method :
#     * annot = show numbers in boxes 
#     * linewidth = line widths between boxes
#     * fmt = precision after decimal point for correlations
#     * figsize = to size figure, we can draw big or small figure by setting this parametes 
#   
#   In this graph below , we will see there is negative correlation between Product_Category_1 and Product_Category_2. That may mean , if somebody buys  Product_Category_1 then (s)he does not buy Product_Category_2  .<br>
#   There is negative correlation between purchase and Product_Category_1.  Number of person who buys more Product_Category_1 is small . Number of person who buys less Product_Category_1 is big 

# In[ ]:


#correlation map
f,ax = plt.subplots(figsize=(10, 10))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()


# [**Go to Top**](#0)

# <a id="2"></a>
# # 2. Filter Data and Add New Columns
# 
# 

# In[ ]:


# define function
def setAgeCategory (pAge):
    if pAge ==  "0-17" :
        return 1
    elif pAge ==  "18-25" :
        return 2 
    elif pAge ==  "26-35" :
        return 3
    elif pAge ==  "36-45" :
        return 4
    elif pAge ==  "46-50" :
        return 5
    elif pAge ==  "51-55" :
        return 6
    elif pAge ==  "55+" :
        return 7
    else:
        return 0

# add new columns to data 
data["AgeCategory"] = data.apply(lambda row: setAgeCategory( row["Age"]),axis=1)
data["GenderCategory"] = [1 if each == "F" else 0 for each in data.Gender ]
data.info()


# In[ ]:


# filter people between 46-50(AgeCategory =5), Marital_Status=1 , City_Category =A
data1 = data[ (data.AgeCategory ==5 ) & (data.Marital_Status ==1) & (data.City_Category == "A")]
data1.info()


# [**Go to Top**](#0)

# <a id="3"></a>
# # 3. Draw Graphs by using Matplotlib library

# In[ ]:


# Draw histogram 
# age frequency of data !! locks session ??
#plt.hist(data.Age,bins=10)
#plt.xlabel("age")
#plt.ylabel("frequency")
#plt.title("histogram")
#plt.legend()
#plt.show()

# AgeCategory frequency of data  
plt.hist(data.AgeCategory,bins=7)
plt.xlabel("AgeCategory")
plt.ylabel("frequency")
plt.title("histogram")
plt.legend()
plt.show()


# Get **Occupation** keys and count of Product Category1 and  Product Category2 for occupation

# In[ ]:


print(data.groupby(['Occupation']).groups.keys())

sf1= data.groupby('Occupation')['Product_Category_1'].sum()
dataByOcc = pd.DataFrame({'Occupation':sf1.index, 'Product_Category_1':sf1.values})

sf2= data.groupby('Occupation')['Product_Category_2'].sum()
dataByOcc['Product_Category_2'] =sf2

# for ages 
sf1= data.groupby('AgeCategory')['Product_Category_1'].sum()
dataByAge = pd.DataFrame({'AgeCategory':sf1.index, 'Product_Category_1':sf1.values})

sf2= data.groupby('AgeCategory')['Product_Category_2'].sum()
dataByAge['Product_Category_2'] =sf2


# Draw 2x2  grid subplot

# In[ ]:


# draw line graph in subplot
plt.subplot(2,2,1)
plt.plot( dataByOcc.Occupation, dataByOcc.Product_Category_1, color="red", label="Product_Category_1" )
plt.plot( dataByOcc.Occupation, dataByOcc.Product_Category_2, color="blue", label="Product_Category_2" )
plt.xlabel("Occupation")
plt.ylabel("Product_Category count")

# draw scatter graph in subplot
plt.subplot(2,2,2)
plt.scatter(dataByOcc.Occupation ,dataByOcc.Product_Category_1, color="r", label="Product_Category_1")
plt.scatter(dataByOcc.Occupation ,dataByOcc.Product_Category_2, color="b", label="Product_Category_2")
plt.xlabel("Occupation")
plt.ylabel("Product Counts")
plt.title("scatter plot")
plt.show()

# draw line graph in subplot
plt.subplot(2,2,3)
plt.plot( dataByAge.AgeCategory, dataByAge.Product_Category_1, color="red", label="Product_Category_1" )
plt.plot( dataByAge.AgeCategory, dataByAge.Product_Category_2, color="blue", label="Product_Category_2" )
plt.xlabel("AgeCategory")
plt.ylabel("Product_Category count")

# draw scatter graph in subplot
plt.subplot(2,2,4)
plt.scatter(dataByAge.AgeCategory ,dataByAge.Product_Category_1, color="r", label="Product_Category_1")
plt.scatter(dataByAge.AgeCategory ,dataByAge.Product_Category_2, color="b", label="Product_Category_2")
plt.xlabel("AgeCategory")
plt.ylabel("Product Counts")
plt.title("scatter plot")
plt.show()


# [**Go to Top**](#0)
