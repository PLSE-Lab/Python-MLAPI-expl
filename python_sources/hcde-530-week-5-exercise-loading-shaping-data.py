#!/usr/bin/env python
# coding: utf-8

# # Loading and Shaping Data in Python

# This exercise demonstrates a few more techniques in pandas data shaping. First we will import the required libraries for the data work we need to conduct. 

# In[ ]:


import pandas as pd


# ### Loading the data
# In this case, our data is in a CSV file called 'filtered_WDIData.csv' which we created in the previous lesson, so we will use the pandas method .read_csv() to retrieve it. You can add it from your own datasets, or you can get it from my datasets. Click on Add Data in the right hand tray and search Kaggle for the file: reshaped_US_WDIData.csv  Add it to your notebook.
# 
# Then load it into a pandas dataframe:

# In[ ]:


df = pd.read_csv('../input/filtered-wdidata/filtered_WDIData.csv')


# #### Now that we've loaded data into our dataframe, we can use the pandas method .head() to look at the first 5 rows of data.

# In[ ]:


df.head()


# #### We've picked up an extra column which was the index from our old dataframe. Let's remove it. Since we are just working with the US we can also remove Country Name and Country Code.

# In[ ]:


df = df.drop(['Unnamed: 0', 'Country Name', 'Country Code'], axis=1)


# #### Currently each year is a separate column, while each indicator is a separate row. To make analysis easier in most analytics tools, we will want to make the a column called 'Year', where each row is a year and contains other columns contain indicator values. This is the pivot technique we've learned about, though in pandas it is called pd.melt().

# In[ ]:


df = pd.melt(df, id_vars=['Indicator Name', 'Indicator Code'])


# In[ ]:


df.head()


# #### Let's pick a specific indicator and check out all it's values. We'll look at Access to clean fuels and technologies with Indicator Code EG.CFT.ACCS.ZS.

# In[ ]:


df[df['Indicator Code'] == 'EG.CFT.ACCS.ZS']


# #### Yikes, it looks like there are still many NaN values in this dataset. We can remove all these nulls using the pandas .dropna() method. In this case we will specify that axis=0 so we wild drop rows and how='any' so we will drop any rows that contain a null value. First let's take a look at the shape of the data frame which will give a result in (rows, cols):

# In[ ]:


df.shape


# #### The data has 85,728 rows and 4 columns. Now let's drop those rows with null values

# In[ ]:


df = df.dropna(axis=0, how='any')


# #### Now let's look at the shape again:

# In[ ]:


df.shape


# #### We are now down to 33,076 rows. This means we dropped a total of:

# In[ ]:


str(85728-33075) + ' null rows'


# #### We can use a combination of pandas methods to see how many rows exist for each indicator after we removed the nulls. The .groupby() method will group rows by the values of a column, in this case Indicator Name, the .size() method will count the number of rows in each group, and the .sort_values(ascending=False) will sort the results in descending order.

# In[ ]:


df.groupby('Indicator Name').size().sort_values(ascending=False)


# #### We've now cleaned up our data in a way that will be significantly smaller than when we started and much easier to work with in programs like Tableau. Before we save our results, let's make one final change. The year column is now called 'variable'. Let's change this to 'Year' and rename 'value' to 'Value'.

# In[ ]:


df = df.rename(index=str, columns={"variable": "Year", 'value' : 'Value'})


# #### Now let's output the results to csv.

# In[ ]:


df.to_csv('reshaped_US_WDIData.csv')

