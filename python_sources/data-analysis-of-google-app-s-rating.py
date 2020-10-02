#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import os
print(os.listdir("../input"))


# ## Exploratory Data Analysis

# Read Data

# In[ ]:


google_data = pd.read_csv("../input/google-play-store-apps/googleplaystore.csv")


# In[ ]:


type(google_data)


# In[ ]:


google_data.head(10)


# In[ ]:


google_data.tail()


# In[ ]:


google_data.info()


# In[ ]:


google_data.describe()


# In[ ]:


google_data.shape


# In[ ]:


google_data.boxplot()


# In[ ]:


google_data.hist()


# ## Data Cleaning

# Count the number of missing values in the Dataframe

# In[ ]:


google_data.isnull()


# In[ ]:


#count the number of missing values in each column
google_data.isnull().sum()


# Check how many ratings are more that 5 outliers

# In[ ]:


google_data[google_data.Rating > 5]


# In[ ]:


google_data.drop([10472],inplace = True)


# In[ ]:


google_data[10470:10475]


# In[ ]:


google_data.boxplot()


# In[ ]:


google_data.hist()


# Remove columns that are 90% empty 

# In[ ]:


threshold = len(google_data)*0.1 #10% of (my rows = 10841)
threshold


# In[ ]:


google_data.dropna(thresh = threshold,axis = 1,inplace =True)


# In[ ]:


print(google_data.isnull().sum())


# In[ ]:


google_data.shape


# ## Data Impution and Manipulation

# Fill the null values with appropriate values using aggregate functions such as mean,median and mode

# In[ ]:


#Define a function impute median
def impute_median(series):
    return series.fillna(series.median())


# In[ ]:


google_data.Rating = google_data['Rating'].transform(impute_median)


# In[ ]:


#Count the number of null values to each column
google_data.isnull().sum()


# In[ ]:


# modes of Categorical values
print(google_data ['Type'].mode())
print(google_data ['Current Ver'].mode())
print(google_data ['Android Ver'].mode())


# In[ ]:


#fill the missing categorical values with mode
google_data['Type'].fillna(str(google_data['Type'].mode().values[0]),inplace=True)
google_data['Current Ver'].fillna(str(google_data['Current Ver'].mode().values[0]),inplace=True)
google_data['Android Ver'].fillna(str(google_data['Android Ver'].mode().values[0]),inplace=True)          


# In[ ]:


#count the number of null values in each column
google_data.isnull().sum()


# In[ ]:


# Lets convert Price,Reviews and Ratings into Numerical Values


# In[ ]:


# Lets convert Price,Reviews and Ratings into Numerical Values
google_data['Price']= google_data['Price'].apply(lambda x: str(x).replace('$','')if '$' in str(x) else str(x))
google_data['Price']=google_data['Price'].apply(lambda x: float(x))
google_data['Reviews']=pd.to_numeric(google_data['Reviews'],errors = 'coerce')


# In[ ]:


google_data['Installs']=google_data['Installs'].apply(lambda x: str(x).replace('+', '')if '+' in str(x) else str(x))
google_data['Installs']=google_data['Installs'].apply(lambda x: str(x).replace(',' , '')if ',' in str(x) else str(x))
google_data['Installs']=google_data['Installs'].apply(lambda x: float(x))


# In[ ]:


google_data.head(10)


# In[ ]:


google_data.describe() #summary stats After Cleaning


# ## Data Visualization

# In[ ]:


grp = google_data.groupby('Category')
x = grp['Rating'].agg(np.mean)
y = grp['Price'].agg(np.sum)
z = grp['Reviews'].agg(np.mean)
print(x)
print(y)
print(z)


# In[ ]:


plt.figure(figsize=(12,5))
plt.plot(x,'ro')
plt.xticks(rotation=90)
plt.show()


# In[ ]:


plt.figure(figsize =(16,5))
plt.plot(x,'*',color='b')
plt.xticks(rotation=90)
plt.title('Category wise Rating')
plt.xlabel('Categories-->')
plt.ylabel('Rating-->')
plt.show()


# In[ ]:


plt.figure(figsize=(16,5))
plt.plot(x,'r--',color='g')
plt.xticks(rotation=90)
plt.title('Category wise Pricing')
plt.xlabel('Categories-->')
plt.ylabel('Prices-->')
plt.show()


# In[ ]:


plt.figure(figsize =(16,5))
plt.plot(x,'g^',color='m')
plt.xticks(rotation=90)
plt.title('Category wise Reviews')
plt.xlabel('Categories-->')
plt.ylabel('Reviews-->')
plt.show()


# # Conclusion

# On the basis of above Plots we can Conclude :

# * The top Category app that have received a highest Ratings are Event apps then Education and then Art & Design
# 
# * People are mostly buying Finance app from Google play Store apps but the Ratings of Finance app have been a bit lower like 4.15 to 4.20.Then people are buying Family,Lifestyle and then medical apps.
# 
# * Most of the Reviews people have given on Communication apps then Social apps and then on Game apps so these are the top three categories for Reviews.  

# In[ ]:




