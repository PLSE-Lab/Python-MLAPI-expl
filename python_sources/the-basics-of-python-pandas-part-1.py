#!/usr/bin/env python
# coding: utf-8

# Here we will take a look at the **basics of Python Pandas** including some useful functions to actually start your exploration in data science.
# First things first, we need to import the pandas library to our workspace

# In[ ]:


import pandas as pd

#Now this is the default way on how we import libararies in python. 
#You may choose any meaningful name in place of 'pd'


# Next we will use make use of the Pandas library
# Initially we need to import our dataset
# I have already attached a sample CSV file which i grabbed from https://support.spatialkey.com/spatialkey-sample-csv-data/
# There is plenty of puclic data sets in Kaggle as well as in other websites for you to play with

# In[ ]:


#Now let's bring in our dataset

realEstTrans = pd.read_csv('../input/Sacramentorealestatetransactions.csv')


# If your dataset is in excel format or .xlsx, then we can use the below method
# 
# > realEstTrans_xlsx = pd.read_excel('../input/Sacramentorealestatetransactions.xlsx')
# 
# If your dataset in tab seperated, space seperated file or .txt, then we use the read.csv method by including a delimiter
# > realEstTrans_txt = pd.read_csv('../input/Sacramentorealestatetransactions.csv', delimiter= '\t')
# 
# Try playing around with the methods and GET ERRORS !!! It's where you will learn.

# In[ ]:


#To view the dataset in your workspace, we can simply call the name of the variable or use the print function

realEstTrans


# > print(realEstTrans)
# 
# > realEstTrans.head() will load the first 5 rows of the dataset
# 
# > realEstTrans.head(n) will load the first n rows of the dataset
# 
# > reatEstTrans.tail(n) will load the bottom n rows of the dataset
# 
# Try playing around with differeng functions to get a hang on them.

# In[ ]:


#Now let's explore our dataset
#To view the headers or features we use the .columns method
realEstTrans.columns #or print(realEstTrans.columns) will work the same


# In[ ]:


#To view a specific column we use the column name we want to see

#realEstTrans['street'] #This will load all the data in the 'street column'
realEstTrans['street'][0:5] #We can also give a range of rows we need from the column we need

#realEstTrans.street even this method works when the header is a single word


# In[ ]:


#We can also view multiple columns by simple specifying the names
realEstTrans[['street','type','price']].head()


# In[ ]:


#To read specific row we can use the .iloc[] method
realEstTrans.iloc[2]
#realEstTrans.iloc[2:4] will give the the 3rd and 4th rows 
#Index begins from 0


# In[ ]:


#We can also use .iloc[] to read a specific data at a given location

realEstTrans.iloc[1,6] #This will give the data in the 2nd row, 7th column


# > Another useful function to locate data is the .loc[] method
# 
# If we want to get the rows that matches a particulat text/numeric we can use this method

# In[ ]:


realEstTrans.loc[realEstTrans['type'] == 'Condo']
#We can also use multiple conditions, try it our yourself


# Let's move on to understanding our dataset
# 

# The most simplest way to get an overall idea of the dataset would be to use the method **.describe()**

# In[ ]:


realEstTrans.describe()


# **The result will provide vital information such as the Standard Deviation, Mean, Mode, etc**

# In[ ]:


#Let's sort our dataset
realEstTrans.sort_values('price').head() 


# * This will return only a copy of the sorted dataset.
# 
# * It is good practice to create a variable to hold our modifications or temporary values
# > sorted_df = realEstTrans.sort_values('price') 
# 
# * This dataframe will hold the sorted version of the dataset
# > sorted_df 

# In[ ]:


realEstTrans.sort_values('price', ascending=False).head()
#Asigning False or 0 to ascending will reverse sort the values


# In[ ]:


realEstTrans.sort_values(['price','beds'], ascending=[0,1]).head()
#We can also use multiple parameters to sort our data
#ascending=[0,1] means 'price' will be in descending and beds will be in ascending orders
#Try out with different parameters to get a good understanding


# * So that's the end of part 1 of this small series
# * Try to explore all commented out codes
# * Play around to get a good knowledge on the basics
# * On the next series we will look into making changes and operations on our dataset
# * If you liked this kernal, please upvote
