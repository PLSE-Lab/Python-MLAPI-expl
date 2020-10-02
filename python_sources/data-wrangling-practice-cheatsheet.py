#!/usr/bin/env python
# coding: utf-8

# ### <span style = "color:Blue">The below wrangling methods will be used in this practice session:</span>
# 1. dtypes
# 2. drop()
# 3. astype()
# 4. rename(columns = {})
# 5. min()
# 6. max()
# 7. mean()
# 8. std()
# 9. replace()
# 10. isna()
# 11. np.linspace()
# 12. pd.cut()
# 13. round()
# 14. pd.to_numeric()
# 15. pd.get_dummies()
# 16. pd.concat([df1,df2],axis)
# 17. groupby
# 18. pearsonr
# 19. at[]
# 20. get_group()['colname']
# 21. f_oneway
# 22. .shape
# 23. .values
# 24. map()

# In[ ]:


import pandas as pd
import numpy as np

df = pd.read_csv('../input/ibm-sql-course-chicago-crime-and-public-schools/Census_Data_-_Selected_socioeconomic_indicators_in_Chicago__2008___2012-v2.csv')


# In[ ]:


df.dtypes


# <span style = "color:Blue">Separating the object type from the integers</span>

# In[ ]:


df_num = df.drop(['COMMUNITY_AREA_NAME'], axis = 1)
df_cat = df['COMMUNITY_AREA_NAME'].copy()


# <span style = "color:Blue">Converting all the numeric data types to float64</span>

# In[ ]:


df_num = df_num.astype(np.float64).head()


# <span style = "color:Blue">Renaming the column names</span>

# In[ ]:


df.rename(columns = {"PERCENT HOUSEHOLDS BELOW POVERTY":"household_BPL"},inplace = True )


# In[ ]:


df.rename(columns = {"COMMUNITY_AREA_NUMBER":'comm_area_no'}, inplace = True)


# In[ ]:


df.dtypes


# In[ ]:


df.rename(columns = {"COMMUNITY_AREA_NAME":'comm_area_name'}, inplace = True)
df.rename(columns = {"PERCENT OF HOUSING CROWDED":'percent_housing_crowded'}, inplace=True)
df.rename(columns = {"PERCENT AGED 16+ UNEMPLOYED":'per_16_unemp'}, inplace=True)
df.rename(columns = {"PERCENT AGED 25+ WITHOUT HIGH SCHOOL DIPLOMA":'per_25_without_highschool'}, inplace=True)
df.rename(columns = {"PERCENT AGED UNDER 18 OR OVER 64":'per_under18_over64'}, inplace=True)
df.rename(columns = {"PER_CAPITA_INCOME ":'per_capita_income'}, inplace = True)
df.rename(columns = {"HARDSHIP_INDEX":'hardship_index'}, inplace = True)


# In[ ]:


df.head()


# <span style = "color:Blue">Basic ways of normalizing the data:</span>
# 1. Simple Feature scaling
# 2. Min-max scaling
# 3. z scores

# In[ ]:


#simple feature scaling on per_capita_income - Range 0 to 1
df_norm = df
df_norm['per_capita_income'] = df_norm['per_capita_income']/df_norm['per_capita_income'].max()


# In[ ]:


# min-max scaling
df_min_max = df
df_min_max['per_capita_income'] = (df_min_max['per_capita_income'] - df_min_max['per_capita_income'].min())/(
    df_min_max['per_capita_income'].max() - df_min_max['per_capita_income'].min() )


# In[ ]:


df_min_max.head()


# In[ ]:


# z scores range -3 to 3 usually
df_z = df
df_z["per_capita_income"] = ( df_z["per_capita_income"] - df_z["per_capita_income"].mean() )/df_z["per_capita_income"].std(ddof=1)
df_z.head()


# <span style = "color:blue">Dealing with missing values:</span>
# 1. Ask the team which collected the data if missing values can be collected
# 2. Drop missing values
# 3. Replace the values with mean/mode
# 4. Leave the missing values as they are

# In[ ]:


# drop missing values - dropna()

#df_norm.dropna().info()


# In[ ]:


# Identify the column with missing value and replace the values there
df_z.info()


# hardship_index and comm_area_no seem to have missing values

# In[ ]:


df_z[["comm_area_no"]].isna().tail()


# The last value is NaN. Let us find what the values before it are

# In[ ]:


df_z[["comm_area_no"]].tail()


# So this is basically an incremental value which can be replaced with 78.0

# In[ ]:


df_z["comm_area_no"]  =  df_z["comm_area_no"].replace(np.nan, np.float(78))


# In[ ]:


df_z[["comm_area_no"]].tail()


# Let us tackle the hardship_index now

# In[ ]:


df_z.head()


# In[ ]:


df_z['hardship_index'].isna().tail()


# As we see, the last entry is a NaN

# In[ ]:


df_z['hardship_index'].tail()


# Let us check the mean, median, min and max values and the skewness of the distribution

# In[ ]:


print("The mean is", df_z['hardship_index'].mean(),"and median is", df_z['hardship_index'].median(),
     "and the max is", df_z['hardship_index'].max(), "and the min is", df_z['hardship_index'].min())


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
df_z.hist('hardship_index', bins=25)


# Let us replace the value by its mean since the distribution is uniform

# In[ ]:


mean = round(df_z['hardship_index'].mean(),1)
#df_z['hardship_index'].replace(np.nan, mean)
mean


# In[ ]:


df_z['hardship_index'] = df_z['hardship_index'].astype(np.float64)


# In[ ]:


df_z['hardship_index'].dtypes


# In[ ]:


df_z['hardship_index'].replace(np.nan, mean)


# In[ ]:


df_z.info()


# <span style = "color:Blue">Binning the hardship index into 3 categories</span>

# In[ ]:


bins = np.linspace(min(df_z['hardship_index']),max(df_z['hardship_index']), 4)
print(bins)
#bins = np.linspace(min(df_z['hardship_index']), max(df_z['hardship_index']), 4)


# In[ ]:


group_names = ["Easy", "Medium", "High"]
df_z['hardship_binned'] = pd.cut(df_z['hardship_index'], bins = bins, labels = group_names)


# In[ ]:


df_z.head()


# <span style = "color:Blue">One hot encoding of the categorial variables</span>

# In[ ]:


# One hot encoded dataframe
df_demo = pd.get_dummies(df_z['hardship_binned'])


# In[ ]:


# Column binding it with the original data frame
df_onehot = pd.concat([df_z, df_demo], axis = 1)


# In[ ]:


df_onehot.head()


# In[ ]:


get_ipython().run_cell_magic('capture', '', '\n! pip install seaborn')


# In[ ]:


import seaborn as sns


# In[ ]:


df.columns


# In[ ]:


# Using group_by (Per capita income by community area name)
boxp = df.groupby(['comm_area_name'], as_index=False)[['comm_area_name','per_capita_income']].mean()


# In[ ]:


from scipy import stats


# In[ ]:


#Pearson correlation coefficient

df['hardship_index'] = df['hardship_index'].replace(np.nan,mean)
pearsonr, pvalue = stats.pearsonr(df['per_capita_income'], df['hardship_index'])
print("The Pearson r is:", pearsonr,"and the p-value is:", pvalue)


# In[ ]:


# pre-ANOVA grouping

bins = np.linspace(df['hardship_index'].min(), df['hardship_index'].max(), 4)
df.drop(['hardship_binned'], inplace=True, axis = 1)
df['hardship_binned'] = pd.cut(df['hardship_index'], bins = bins, labels = ["Low", "Medium", "high"], include_lowest=True)
grouped = df[['hardship_binned', 'per_capita_income']].groupby(['hardship_binned'])

# To get a value at 0 index of a column
df.at[0,'per_capita_income']


# In[ ]:


grouped.get_group('Low')['per_capita_income']


# In[ ]:


# ANOVA

fval, pval = stats.f_oneway(grouped.get_group('Low')['per_capita_income'], 
               grouped.get_group('Low')['per_capita_income'],
               grouped.get_group('Low')['per_capita_income'])
print("The f val is:", fval,"and the p-value is", pval)


# In[ ]:


# To find the number of rows and columns of a dataframe

df.shape[0] #rows
df.shape[1] #columns


# In[ ]:


# To find the column names as a list

df.columns.values


# In[ ]:




