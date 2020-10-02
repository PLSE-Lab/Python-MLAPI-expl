#!/usr/bin/env python
# coding: utf-8

# This notebook covers the cleaning and exploration of data for 'Google Play Store Apps'

# ### Imporing Libraries

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # for plots

import os
print(os.listdir("../input"))


# ### Reading data from the csv file

# In[ ]:


data = pd.read_csv('../input/googleplaystore.csv')
data.head()


# In[ ]:


data.columns = data.columns.str.replace(' ', '_')


# In[ ]:


print("Shape of data (samples, features): ",data.shape)
print("Data Types: \n", data.dtypes.value_counts())


# The data has **12** object and **1** numeric feature i.e. *Rating*. 

# Now Exploring each features individually
# 1. [Size](#size)
# 2. [Installs](#installs)
# 3. [Reviews](#reviews)
# 4. [Rating](#rating)
# 5. [Type](#type)
# 6. [Price](#price)
# 7. [Category](#cat)
# 8. [Content Rating](#content_rating)
# 9. [Genres](#genres)
# 10. [Last Updated](#last_updated)
# 11. [Current Version](#current_version)
# 12. [Android Version](#android_version)

# ## <a id=size>Size</a>

# Lets look into frequency of each item to get an idea of data nature

# In[ ]:


data.Size.value_counts().head()
#please remove head() to get a better understanding 


# It can be seen that data has metric prefixes (Kilo and Mega) along with another string.
# Replacing k and M with their values to convert values to numeric.

# In[ ]:


data.Size=data.Size.str.replace('k','e+3')
data.Size=data.Size.str.replace('M','e+6')
data.Size.head()


# Now, we have some two types of values in our Size data.
# 1. exponential values (not yet converted to string)
# 2. Strings (that cannot be converted into numeric)
# 
# Thus specifing categories 1 and 2 as an boolean array **temp**, to convert category 1 to numeric. 
# 

# In[ ]:


def is_convertable(v):
    try:
        float(v)
        return True
    except ValueError:
        return False
    
temp=data.Size.apply(lambda x: is_convertable(x))
temp.head()


# Now checking unique non numeric values (***~temp***) in Size.

# In[ ]:


data.Size[~temp].value_counts()


# - Replacing 'Varies with Device' by nan and 
# - Converting 1,000+ to 1000, to make it numeric

# In[ ]:


data.Size=data.Size.replace('Varies with device',np.nan)
data.Size=data.Size.replace('1,000+',1000)


# Converting the cleaned Size data to numeric type

# In[ ]:


data.Size=pd.to_numeric(data.Size)


# In[ ]:


data.hist(column='Size')
plt.xlabel('Size')
plt.ylabel('Frequency')


# ## <a id='installs'>Installs</a>

# Checking unique values in Install data

# In[ ]:


data.Installs.value_counts()


# It can be seen that there are 22 unique values, out of which
# - 1 is 0, 
# - 1 is Free(string) , which we will be converting to nan here
# - and rest are numeric but with '+' and ',' which shall be removed to convert these into numeric type. 

# In[ ]:


data.Installs=data.Installs.apply(lambda x: x.strip('+'))
data.Installs=data.Installs.apply(lambda x: x.replace(',',''))
data.Installs=data.Installs.replace('Free',np.nan)
data.Installs.value_counts()


# Checking if data is converted to numeric

# In[ ]:


data.Installs.str.isnumeric().sum()


# Now in Installs, 1 sample is non numeric out of 10841, which is nan (converted from Free to nan in previous step)

# In[ ]:


data.Installs=pd.to_numeric(data.Installs)


# In[ ]:


data.Installs=pd.to_numeric(data.Installs)
data.Installs.hist();
plt.xlabel('No. of Installs')
plt.ylabel('Frequency')


# ## <a id='reviews'>Reviews</a>

# Checking if all values in number of Reviews numeric

# In[ ]:


data.Reviews.str.isnumeric().sum()


# One value is non numeric out of 10841. Lets find its value and id.

# In[ ]:


data[~data.Reviews.str.isnumeric()]


# We could have converted it into interger like we did for <a id='id'>Size</a> but the data for this App looks different. It can be noticed that the entries are entered wrong (i.e. cell backwared). We could fix it by setting **Category** as nan and shifting all the values, but deleting the sample for now. 

# In[ ]:


data=data.drop(data.index[10472])


# To check if row is deleted

# In[ ]:


data[10471:].head(2)


# In[ ]:


data.Reviews=data.Reviews.replace(data.Reviews[~data.Reviews.str.isnumeric()],np.nan)


# In[ ]:


data.Reviews=pd.to_numeric(data.Reviews)
data.Reviews.hist();
plt.xlabel('No. of Reviews')
plt.ylabel('Frequency')


# ## <a id='rating'>Rating</a>

# For entries to be right we need to make sure they fall within the range 1 to 5.

# In[ ]:


print("Range: ", data.Rating.min(),"-",data.Rating.max())


# Checking the type of data, to see if it needs to be converted to numeric

# In[ ]:


data.Rating.dtype


# Data is already numeric, now checking if the data has null values

# In[ ]:


print(data.Rating.isna().sum(),"null values out of", len(data.Rating))


# In[ ]:


data.Rating.hist();
plt.xlabel('Rating')
plt.ylabel('Frequency')


# ## <a id='Type'>Type</a>

# Checking for unque type values and any problem with the data

# In[ ]:


data.Type.value_counts()


# There are only two types, free and paid. No unwanted data here.

# ## <a id='price'>Price</a>

# Checking for unique values of price, along with any abnormalities

# In[ ]:


data.Price.unique()


# Data had **$** sign which shall be removed to convert it to numeric

# In[ ]:


data.Price=data.Price.apply(lambda x: x.strip('$'))


# In[ ]:


data.Price=pd.to_numeric(data.Price)
data.Price.hist();
plt.xlabel('Price')
plt.ylabel('Frequency')


# Some apps have price higher than 350. Out of curiosity I checked the apps to see if there is a problem with data. But no !! they do exist, and Yes !! people buy them.

# In[ ]:


temp=data.Price.apply(lambda x: True if x>350 else False)
data[temp].head(3)


# ## <a id='cat'>Category</a>

# Now lets inspect the category by looking into the unique terms. 

# In[ ]:


data.Category.unique()


# It shows no repetition or false data

# In[ ]:


data.Category.value_counts().plot(kind='bar')


# ## <a id='content_rating'>Content Rating </a>

# Checking unique terms in Content Rating Categories, and for repetitive or abnormal data.

# In[ ]:


data.Content_Rating.unique()


# No abnormalies or repetition found

# In[ ]:


data.Content_Rating.value_counts().plot(kind='bar')
plt.yscale('log')


# ## <a id='genres'>Genres</a>

# Checking for unique values, abnormalitity or repetition in data

# In[ ]:


data.Genres.unique()


# The data is in the format **Category;Subcategory**. Lets divide the data into two columns, one as primary category and the other as secondary, using **;** as separator.

# In[ ]:


sep = ';'
rest = data.Genres.apply(lambda x: x.split(sep)[0])
data['Pri_Genres']=rest
data.Pri_Genres.head()


# In[ ]:


rest = data.Genres.apply(lambda x: x.split(sep)[-1])
rest.unique()
data['Sec_Genres']=rest
data.Sec_Genres.head()


# In[ ]:


grouped = data.groupby(['Pri_Genres','Sec_Genres'])
grouped.size().head(15)


# Generating a two table to better understand the relationship between primary and secondary categories of Genres

# In[ ]:


twowaytable = pd.crosstab(index=data["Pri_Genres"],columns=data["Sec_Genres"])
twowaytable.head()


# For visual representation of this data, lets use stacked columns

# In[ ]:


twowaytable.plot(kind="barh", figsize=(15,15),stacked=True);
plt.legend(bbox_to_anchor=(1.0,1.0))


# ## <a id='last_updated'>Last Updated</a>

# Checking the format of data in Last Updated Dates

# In[ ]:


data.Last_Updated.head()


# Converting the data i.e. string to datetime format for furthur processing

# In[ ]:


from datetime import datetime,date
temp=pd.to_datetime(data.Last_Updated)
temp.head()


# Taking a difference between last updated date and today to simplify the data for future processing. It gives days.

# In[ ]:


data['Last_Updated_Days'] = temp.apply(lambda x:date.today()-datetime.date(x))
data.Last_Updated_Days.head()


# ## <a id='android_version'>Android Version</a>

# Checking unique values, repetition, or any abnormalities.

# In[ ]:


data.Android_Ver.unique()


# Most of the values have a upper value and a lower value (i.e. a range), lets divide them as two new features **Version begin and end**, which might come handy while processing data furthur.

# In[ ]:


data['Version_begin']=data.Android_Ver.apply(lambda x:str(x).split(' and ')[0].split(' - ')[0])
data.Version_begin=data.Version_begin.replace('4.4W','4.4')
data['Version_end']=data.Android_Ver.apply(lambda x:str(x).split(' and ')[-1].split(' - ')[-1])


# In[ ]:


data.Version_begin.unique()


# Representing categorial data as two way table and plotting it as stacked columns for better understanding.

# In[ ]:


twowaytable = pd.crosstab(index=data.Version_begin,columns=data.Version_end)
twowaytable.head()


# In[ ]:


twowaytable.plot(kind="barh", figsize=(15,15),stacked=True);
plt.legend(bbox_to_anchor=(1.0,1.0))
plt.xscale('log')


# In[ ]:


data.Version_end.unique()


# ## <a id='current_version'>Current Version</a>

# In[ ]:


data.Current_Ver.value_counts().head(6)


# Lets convert all the versions in the format **number.number** to simplify the data, and check if the data has null values. Also, we are not considering converting value_counts to nan here due to its high frequency.

# In[ ]:


data.Current_Ver.isna().sum()


# As we have only **8** nans lets replace them with **Varies with data** to simplify 

# In[ ]:


import re
temp=data.Current_Ver.replace(np.nan,'Varies with device')
temp=temp.apply(lambda x: 'Varies with device' if x=='Varies with device'  else  re.findall('^[0-9]\.[0-9]|[\d]|\W*',str(x))[0] )


# In[ ]:


temp.unique()


# Saving the updated current version values as a new column

# In[ ]:


data['Current_Ver_updated']=temp


# In[ ]:


data.Current_Ver_updated.value_counts().plot(kind="barh", figsize=(15,15));
plt.legend(bbox_to_anchor=(1.0,1.0))
plt.xscale('log')


# In[ ]:





# In[ ]:




