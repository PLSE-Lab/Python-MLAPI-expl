#!/usr/bin/env python
# coding: utf-8

# In[ ]:



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import warnings #to ignore any warnings
warnings.filterwarnings('ignore')

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# ## Reading Data

# In[ ]:


df = pd.read_csv('/kaggle/input/fertilizers-by-product-fao/FertilizersProduct.csv',encoding = "ISO-8859-1")
df.head()


# In[ ]:


df.tail()


# ## Summary

# In[ ]:


df.info()


# From above we can see that there are no null values in the dataframe. We can also see that all the columns are of the correct datatype
# 
# I like to have consistency in my column names, so I'll change all the column names to the format : firstName_lastName

# In[ ]:


df.columns  = df.columns.str.lower()
df.columns = df.columns.str.replace(' ','_')
df.head()


# Now I will remove the irrelevant columns i.e. the columns that are not needed for this analysis

# In[ ]:


df.drop(['area_code','item_code','element_code','year_code','flag'], axis = 1, inplace = True)
df.head()


# #### Checking for any duplicates

# In[ ]:


df.duplicated().any()


# ## Checking Top 10 Countries with Fertilizer uses

# In[ ]:


df1 = df.area.value_counts().head(10).reset_index()
df1 = df1.rename(columns = {'index':'Countries','area':'No. of Fertilizers'})
fig = px.bar(df1, x = 'No. of Fertilizers', y= 'Countries', orientation='h', title = 'Top 10 Countries with Most Fertilizers used')
fig.update_layout(autosize = False)
fig.show()


# From above we can see that India belongs to the top 10 countries that uses fertilizers. And to be precise, its the 9th top country, right after Spain to produce, export and import fertilizers.

# ## Analysing Indian Market

# In[ ]:


India = df[df.area == 'India']
India.head()


# In[ ]:


India.shape


# In[ ]:


India.element.value_counts()


# #### Different Uses for Each Fertilizer in India

# In[ ]:


plt.figure(figsize=(20,7.5))
sns.countplot(x = 'item', data=India, hue = 'element')
plt.title('Different Uses of Fertilizers', size = 18)

plt.xticks(rotation = 90)
plt.show()


# **Now we will analyse Production, Trade and Aglicuture use of Fertilizers in India separately**

# ### Production Use

# In[ ]:


df_prod = India[India.element == 'Production']

#drop irrelevant columns from this dataframe : area (since we know we are analysing Indian market) and element (since we are analysing only one element)

df_prod.drop(['area','element'],axis = 1, inplace = True)

df_prod.head()


# **Fertilizers produced over the years**

# In[ ]:



fig = px.area(df_prod, x = 'year', y = 'value', color = 'item',title = 'Fertilizers produced over the years 2002-2017 in India', line_group = 'item')
fig.update_layout(legend_orientation = 'h', autosize = False)
fig.show()


# From the above graph we can see that Urea's production in India was the highest but only until 2016. It dropped drastically after 2016.
# 

# ### Agriculural Use

# In[ ]:


df_agr = India[India.element == 'Agricultural Use']

#drop irrelevant columns from this dataframe : area (since we know we are analysing Indian market) and element (since we are analysing only one element)

df_agr.drop(['area','element'],axis = 1, inplace = True)

df_agr.head()


# In[ ]:


fig = px.area(df_agr, x = 'year', y = 'value', color = 'item',title = 'Fertilizers used over the years 2002-2017 in India', line_group = 'item')
fig.update_layout(legend_orientation = 'h',autosize = False, height = 600, width = 800)
fig.show()


# From the above we can see that Urea is the most used fertilizer for agriculture. It should not be too shocking since it was the most produced fertilizer in India as well. Now we need to check if Urea was getting imported or exported as well or not.

# ### Trade - Export Use

# #### Export Quantity

# In[ ]:



#getting all the data where element is export quantity
df_equan = India[India.element == 'Export Quantity']
df_equan.head()


# In[ ]:



top_5_exp = df_equan.groupby('item')['value'].sum().sort_values(ascending = False).head().reset_index()
fig = px.bar(top_5_exp, x = 'value', y = 'item', orientation = 'h', title = 'Top 5 Fertilizers Exported')
fig.show()


# NPK fertlizers were exported the most.
# 
# We can also see that even though Urea is produced the most in India, it is not exported much. Compared to what it is produced, it was exported way too less.

# In[ ]:



#getting the fertilizers name
top_5_exp = top_5_exp['item']

#list to store the dataframes created for top 5 fertilizers
df_to_concat = []

for i in top_5_exp:
    items = df_equan[df_equan.item == i]
    df_to_concat.append(items)
    
result = pd.concat(df_to_concat)

#plotting line graph
fig_equan = px.line(result, x = 'year', y = 'value', color = 'item', title = 'Top 5 Fertilizers exported from 2002-2017')
fig_equan.update_layout(legend_orientation = 'h', height= 600)
fig_equan.show()


# Above we can see that NPK fertilizers were not getting exported until 2008 and then by 2009 it became the most exported fertilizer. Even though it went down in 2010, the quantity kept on increasing after that. There is no pattern as such in the top 5 exported fertilizers.

# #### Import Quantity

# In[ ]:


#getting all the data where element is import quantity
df_iquan = India[India.element == 'Import Quantity']
df_iquan.head()


# In[ ]:


#getting top 5 fertilizers that were imported
top_5_imp = df_iquan.groupby('item')['value'].sum().sort_values(ascending = False).head().reset_index()
fig = px.bar(top_5_imp, x = 'value', y = 'item', orientation = 'h', title = 'Top 5 Fertilizers Imported')
fig.show()


# From above we can see that phosphate rock is imported the most and after that Urea is imported the most. Is Urea being used more than it is being produced?

# In[ ]:



#getting the fertilizers name
top_5_imp = top_5_imp['item']

#list to store the dataframes created for top 5 fertilizers
df_to_concat = []

for i in top_5_imp:
    items = df_equan[df_equan.item == i]
    df_to_concat.append(items)
    
result = pd.concat(df_to_concat)

#plotting line graph
fig_equan = px.line(result, x = 'year', y = 'value', color = 'item', title = 'Top 5 Fertilizers imported from 2002-2017')
fig_equan.update_layout(legend_orientation = 'h', height = 600)
fig_equan.show()


# We can see from above that Urea was getting imported a bit and did not get imported much in 2016. From our 'productions' graph we know that India decreased the production of Urea drastically in 2017, which is exactly when India started importing Urea the most.
# 

# ## What Is Happening in India

# We will look into the fertilizer **Urea** a bit more since it is the most produced, used and one of the most imported fertilizers.

# In[ ]:


df_urea = India[(India.item == 'Urea') & (India.unit == 'tonnes')]
df_urea.head()


# In[ ]:


plt.figure(figsize=(10,5))
df_urea.groupby('element')['value'].sum().plot(kind = 'bar')
plt.title('Urea - Use vs Import vs Export vs Production', size = 18)
plt.xlabel('Different Uses', size = 15)
plt.ylabel('Amount of Urea used', size = 15)
plt.show()


# ## Final Remarks

# 1. India is the 9th top country for fertilizer use.
# 2. We observed that **Urea** in India is the most produced fertilizer, but only until 2016. After the production of Urea decreased after 2016, Urea was getting imported in large ammounts. 
# 3. **Urea** is the most used fertilizer for agriculture in India.
