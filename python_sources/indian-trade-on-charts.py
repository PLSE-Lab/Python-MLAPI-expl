#!/usr/bin/env python
# coding: utf-8

# In[ ]:




import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import plotly.graph_objs as go
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (9.0,9.0)


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# ** Reading the trade export and import data to the notebook**

# In[ ]:


data_export = pd.read_csv('/kaggle/input/india-trade-data/2018-2010_export.csv')
data_import = pd.read_csv('/kaggle/input/india-trade-data/2018-2010_import.csv')
print("export data:",data_export.shape,'\n',"import data:",data_import.shape)


# In[ ]:


#export data
data_export.head()


# In[ ]:


#import data
data_import.head()


# **Missing Value Imputation**

# In[ ]:


data_export['value'].describe() 


# mean of "value" column is 21.56

# In[ ]:


data_import['value'].describe() 


# In[ ]:


data_export.info()


# In[ ]:


data_import.info()


# In[ ]:


print("total null values in export data:",data_export['value'].isnull().sum())
print("total null values in export data:",data_import['value'].isnull().sum())


# In[ ]:


# Droping all the rows with null values by writing a function :cleanup()
def filling_null(data_df):
    #data_df = data_df[data_df.value!=0]
    data_df["value"].fillna(data_df['value'].mean(),inplace = True)
    data_import.year = pd.Categorical(data_import.year)
    return data_df


# In[ ]:


data_import = filling_null(data_import)
data_export = filling_null(data_export)


# In[ ]:


print("total null values in export data;",data_export['value'].isnull().sum())
print("total null values in export data;",data_import['value'].isnull().sum())


# **Dropping all the rows where country is "UNSPECIFIED" **

# In[ ]:


def drop_un(dat):
        dat['country']= dat['country'].apply(lambda x : np.NaN if x == "UNSPECIFIED" else x)
        dat.dropna(inplace=True)
        return dat


# In[ ]:


data_import = drop_un(data_import)
data_export = drop_un(data_export)


# **Import And Export Country Wise**

# In[ ]:


print("total number of countries india exporting commodity:",len(data_export['country'].unique()))
print("total number of countries india importing commodity:",len(data_import['country'].unique()))


# In[ ]:


df2 = data_import.groupby('country').agg({'value':'sum'}).sort_values(by='value', ascending = False)
df2 = df2.head()

df3 = data_export.groupby('country').agg({'value':'sum'}).sort_values(by='value', ascending = False)
df3 = df3.head()


# In[ ]:


sb.set_style('whitegrid')
sb.barplot(df2.index,df2.value, palette = 'dark')
plt.title('country with highest value (import trade)', fontsize = 20)
plt.show()


# In[ ]:


sb.set_style('whitegrid')
sb.barplot(df3.index,df3.value, palette = 'bright')
plt.title('country with highest value (export trade)', fontsize = 20)

plt.show()


# **Commodities in Value**

# In[ ]:


# top 5 most exported commodity 
df6 = data_export.groupby('Commodity').agg({'value':'sum'}).sort_values(by='value', ascending=False)
df6 = df6.head(5)
df6


# In[ ]:


sb.barplot(df6['value'],df6.index, palette = 'bright')
plt.title('Top 5 exporting Commodities', fontsize = 30)
plt.show()


# In[ ]:


# top5 most imported commmodit 
df7 = data_import.groupby('Commodity').agg({'value':'sum'}).sort_values(by='value', ascending=False)
df7 = df7.head(5)
df7


# In[ ]:


sb.barplot(df7['value'],df7.index, palette = 'dark')
plt.title('Top 5 importing Commodities',fontsize=30)
plt.show()


# **Yearly Import and export**

# In[ ]:


# import yearwise
df4 = data_import.groupby('year').agg({'value':'sum'})
# export yearwise
df5 = data_export.groupby('year').agg({'value':'sum'})
# deficite


# In[ ]:


sb.barplot(df4.index,df4.value, palette = 'Reds_d')
plt.title('Yearly Import', fontsize =30)
plt.show()


# In[ ]:


sb.barplot(df5.index,df5.value, palette = 'Blues_d')
plt.title('Yearly Export', fontsize =30)


# **Trades with highest value**

# In[ ]:


df_import = data_import[data_import.value>1000]
df_export = data_export[data_export.value>1000]

df_import.head(10)


# In[ ]:


df_export.head(10)


# In[ ]:


f1 = df_import.groupby(['country']).agg({'value': 'sum'}).sort_values(by='value')
f2 = df_export.groupby(['country']).agg({'value': 'sum'}).sort_values(by='value')


# In[ ]:


sb.heatmap(f1)
plt.title('highest trade import countrywise', fontsize = 20)
plt.show()


# In[ ]:


sb.heatmap(f2)
plt.title('highest trade export countrywise', fontsize = 20)
plt.show()


# Referred from @Shubham singh Gharsele kernel, thank you.
