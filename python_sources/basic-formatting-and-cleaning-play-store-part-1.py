#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
plt.style.use('ggplot')

import seaborn as sns # for making plots with seaborn
color = sns.color_palette()
sns.set(rc={'figure.figsize':(25,15)})

import warnings
warnings.filterwarnings('ignore')


# # Take a look at the dataset

# Let's load the dataset and name it 'df1' 

# In[ ]:


df1 = pd.read_csv('../input/googleplaystore.csv')
df1.head()


# In[ ]:


len(df1)


# # Remove duplicate apps

# Let's see if there is any duplicated apps in the dataset, we'll remove any apps with the same name and will only keep the first one.

# In[ ]:


df1[df1.duplicated(subset='App', keep='first')].head()


# In[ ]:


# Example of duplicated apps
df1[df1['App']=='Google Ads']


# In[ ]:


# Remove any duplicates app and only keep the first one
df1.drop_duplicates(keep='first', subset='App', inplace=True)


# In[ ]:


len(df1)


# Before, there are 10841 rows in the dataset, and after we remove the duplicate, there are now 9660 rows.

# # Removing Rows with Unappropriate Data

# In[ ]:


df1[df1['Android Ver'].isnull()]


# One app clearly have unappropriate data, it has 'Free' in the 'Install' column when it should have been a number. Let's just remove it.

# In[ ]:


# delete row with index number 10472
df1.drop([10472], inplace=True)


# In[ ]:


len(df1)


# # Data formatting and cleaning

# Before we can do some analysis, let's take a look at the data types of the dataframe series

# In[ ]:


df1.dtypes


# Some of the column that we expect to be numeric type is actually a string, we need to format the column to be numeric. 
# The first one is the 'Installs' column, we want to remove any '+' and ',' sign in all rows in the 'Install' column and then turning it into integer type.

# In[ ]:


# delete  '+' and ',' and convert into integer
df1['Installs'] = df1['Installs'].apply(lambda x: str(x).replace(',', '') if ',' in str(x) else x)
df1['Installs'] = df1['Installs'].apply(lambda x: str(x).replace('+', '') if '+' in str(x) else x)
df1['Installs'] = df1['Installs'].apply(lambda x: int(x))


# The next one is the 'Reviews' column, we simply want this column to be integer type

# In[ ]:


# convert into integer
df1['Reviews'] = df1['Reviews'].apply(lambda x: int(x))


# The next one is the 'Price' column, we'll remove the '$' sign and convert it into float type

# In[ ]:


# delete '$' and convert into float 
df1['Price'] = df1['Price'].apply(lambda x: str(x).replace('$', '') if '$' in str(x) else str(x))
df1['Price'] = df1['Price'].apply(lambda x: float(x))


# The last one is the 'Size' column. For this column, we want all the rows to be in MegaByte(MB), so we can  simply remove any 'M'. But for the row with KiloByte(KB) size, we'll remove the 'K' and divide it by 1000 to convert it to MB. We'll convert rows with 'Varies with device' as the size to Null values.

# In[ ]:


df1['Size'] = df1['Size'].apply(lambda x: str(x).replace('Varies with device', 'NaN') if 'Varies with device' in str(x) else x)
df1['Size'] = df1['Size'].apply(lambda x: str(x).replace('M', '') if 'M' in str(x) else x)
df1['Size'] = df1['Size'].apply(lambda x: str(x).replace(',', '') if 'M' in str(x) else x)
df1['Size'] = df1['Size'].apply(lambda x: float(str(x).replace('k', '')) / 1000 if 'k' in str(x) else x)
df1['Size'] = df1['Size'].apply(lambda x: float(x))


# In[ ]:


df1.dtypes


# In[ ]:


df1.head()


# In[ ]:


# save the cleaned dataset 
df1.to_csv('playstoreapps_cleaned.csv',index=False)


# That's it for. Next, we'll continue using the Google Play Store Apps that we've formatted and cleaned and try to do some EDA
# 
