#!/usr/bin/env python
# coding: utf-8

# On this kernel we will explore the data at hand and perform some analysis. (This kernel is still a work in progress)
# 
# # Data Exploration

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
from matplotlib import pyplot as plt
import missingno as msno
get_ipython().run_line_magic('matplotlib', 'inline')
from ggplot import *

from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from tqdm import tqdm

from subprocess import check_output

df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
df_members = pd.read_csv('../input/members.csv')
df_songs = pd.read_csv('../input/songs.csv')
#df_sample = pd.read_csv('../input/sample_submission.csv')

type(df_train.iloc[1])


# The following three dataframes are the ones we will be focused on. Let's first have a look at them.

# # Train dataframe

# In[ ]:


print(df_train.shape)
df_train.head()


# In[ ]:


mem = df_train.memory_usage(index=True).sum()
print("Memory consumed by train dataframe : {} MB" .format(mem/ 1024**2)) 


# Converting the datatype of **target** variable from **float** to **int8**.

# In[ ]:


df_train['target'] = df_train['target'].astype(np.int8)
df_test['id'] = df_test['id'].astype(np.int32)


# # Members dataframe

# In[ ]:


print(df_members.shape)
df_members.head()


# In[ ]:


mem = df_members.memory_usage(index=True).sum()
print("Memory consumed by members dataframe : {} MB" .format(mem/ 1024**2))


# Changing the datatypes based on the maximum and minimum values present in each of these columns.

# In[ ]:


df_members['city'] = df_members['city'].astype(np.int8)
df_members['bd'] = df_members['bd'].astype(np.int16)
df_members['registered_via'] = df_members['registered_via'].astype(np.int8)
df_members['registration_init_time'] = df_members['registration_init_time'].astype(np.int32)
df_members['expiration_date'] = df_members['expiration_date'].astype(np.int32)


# # Songs dataframe

# In[ ]:


print(df_songs.shape)
df_songs.head()


# In[ ]:


mem = df_songs.memory_usage(index=True).sum()
print("Memory consumed by songs dataframe : {} MB" .format(mem/ 1024**2))


# In[ ]:


df_songs['song_length'] = df_songs['song_length'].astype(np.int32)

#-- Since language column contains Nan values we will convert it to 0,
#-- After converting the type of the column we will revert it back to nan
df_songs['language'] = df_songs['language'].fillna(0)
df_songs['language'] = df_songs['language'].astype(np.int8)

df_songs['language'] = df_songs['language'].replace(0, np.nan)


# Each of these dataframes are huge. Memory reduction will help us effectively use the memory in this kernel.

# # Merged dataframes:
# 1. Merge **df_train** and **df_songs** based on **song_id**.
# 2. Merge the resulting dataframe with **df_members** based on **msno**.

# In[ ]:


df_train_members = pd.merge(df_train, df_members, on='msno', how='inner')
df_train_merged = pd.merge(df_train_members, df_songs, on='song_id', how='outer')
print(df_train_merged.head())
print(len(df_train_merged.columns))
print('\n')
#--- Performing the same for test set ---
df_test_members = pd.merge(df_test, df_members, on='msno', how='inner')
df_test_merged = pd.merge(df_test_members, df_songs, on='song_id', how='outer')
print(df_test_merged.head())
print(len(df_test_merged.columns))


# Cross checking the number of columns in the merged dataframes

# In[ ]:


print(len(df_train.columns))
print(len(df_songs.columns))
print(len(df_members.columns))
print(len(df_test.columns))

print(len(df_train_merged.columns))
print(len(df_test_merged.columns))


# ## Deleting unwanted dataframes

# In[ ]:


del df_train
del df_test
del df_songs
del df_members


# ## Checking Missing Values

# In[ ]:


print(df_train_merged.isnull().values.any())
print(df_test_merged.isnull().values.any())


# In[ ]:


print(df_train_merged.columns[df_train_merged.isnull().any()].tolist(), '\n')
print(df_test_merged.columns[df_test_merged.isnull().any()].tolist())


# Interesting things to note:
# * Train dataframe has missing values in **msno** and **target** columns. Rows having such missing values can be removed.
# * Test dataframe has missing values in column **id**.
# * Apart from these the other columns tend to be the same across both train and test sets.

# In[ ]:


#--- Removing rows having missing values in msno and target ---
df_train_merged = df_train_merged[pd.notnull(df_train_merged['msno'])]
df_train_merged = df_train_merged[pd.notnull(df_train_merged['target'])]


# ## Visualizing missing values

# In[ ]:


msno.bar(df_train_merged[df_train_merged.columns[df_train_merged.isnull().any()].tolist()],figsize=(20,8),color="#32885e",fontsize=18,labels=True,)


# In[ ]:


msno.matrix(df_train_merged[df_train_merged.columns[df_train_merged.isnull().any()].tolist()],width_ratios=(10,1),            figsize=(20,8),color=(0.2,0.2,0.2),fontsize=18,sparkline=True,labels=True)
 


# 

# # Data Visualization

# ## **target** variable
# 
# I used the `ggplot` library available for python to perform the following:

# In[ ]:


pd.value_counts(df_train_merged['target'])


# In[ ]:


ggplot(df_train_merged, aes(x='target')) + geom_bar()


# From this graph it is clear that nearly half the songs have been revisited within a month atleast once by the user!

# ## **source_type**

# In[ ]:


plt.figure(figsize = (8, 6))
ax = sns.countplot(x = "source_type", data = df_train_merged)
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
plt.tight_layout()
plt.show()


# In[ ]:


ct = pd.crosstab(df_train_merged.source_type, df_train_merged.target)
ct.plot.bar(figsize = (12, 6), stacked=True)
plt.show()


# The online playlist, local playlist and local library are places where most songs are heard.

# ## **source_screen_name**

# In[ ]:


plt.figure(figsize = (8, 6))
ax = sns.countplot(y=df_train_merged['source_screen_name'], data=df_train_merged, facecolor=(0, 0, 0, 0),
                    linewidth=5,
                    edgecolor=sns.color_palette("dark", 3))
plt.show()


# In[ ]:


ct = pd.crosstab(df_train_merged.source_screen_name, df_train_merged.target)
ct.plot.bar(figsize = (12, 6), stacked=True)
plt.show()


# Most songs are listened to through the local playlist and the online playlist.

# ## **source_system_tab**

# In[ ]:


plt.figure(figsize = (8, 6))
ax = sns.countplot(y = "source_system_tab", data = df_train_merged)
plt.show()
''' 
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="left")
plt.tight_layout()
plt.show()
'''


# In[ ]:


ct = pd.crosstab(df_train_merged.source_system_tab, df_train_merged.target)
ct.plot.bar(figsize = (12, 6), stacked=True)
plt.show()


# 

# ## **gender**

# In[ ]:



plt.figure(figsize = (8, 8))
pp = pd.value_counts(df_train_merged.gender)
pp.plot.pie(startangle=90, autopct='%1.1f%%', shadow=False, explode=(0.05, 0.05))
plt.axis('equal')
plt.show()


# In[ ]:


ct = pd.crosstab(df_train_merged.gender, df_train_merged.target)
ct.plot.bar(figsize = (12, 6), stacked=True)
plt.show()


# ## **language**

# In[ ]:


plt.figure(figsize = (8, 6))
ax = sns.countplot(x = "language", data = df_train_merged)
plt.show()


# ## **city**

# In[ ]:


plt.figure(figsize = (8, 6))
ax = sns.countplot(y = "city", data = df_train_merged)
plt.show()


# ## **bd** (age)

# In[ ]:





# ## **registered_via**

# In[ ]:


plt.figure(figsize = (8, 6))
ax = sns.countplot(x = "registered_via", data = df_train_merged)
plt.show()


# In[ ]:


'''
sns.factorplot(y="source_system_tab", hue="gender", data=df_train_merged,
                   size=6, kind="bar", palette="muted")
''' 
df_train_merged.columns


# # Data Analysis
# 
# ## AnalyzingData types
# 
# Distribution of various data types in the combined dataframe

# In[ ]:


ax = sns.countplot(y = df_train_merged.dtypes, data = df_train_merged)


# We can see that all the numerical type columns have become `float` type. We must convert them back to `int` to save memory!

# In[ ]:


df_train_merged['target'] = df_train_merged['target'].astype(np.int8)
df_test_merged['id'] = df_test_merged['id'].astype(np.int32)

df_train_merged['city'] = df_train_merged['city'].astype(np.int8)
df_train_merged['bd'] = df_train_merged['bd'].astype(np.int16)
df_train_merged['registered_via'] = df_train_merged['registered_via'].astype(np.int8)
df_train_merged['registration_init_time'] = df_train_merged['registration_init_time'].astype(np.int32)
df_train_merged['expiration_date'] = df_train_merged['expiration_date'].astype(np.int32)

df_test_merged['city'] = df_test_merged['city'].astype(np.int8)
df_test_merged['bd'] = df_test_merged['bd'].astype(np.int16)
df_test_merged['registered_via'] = df_test_merged['registered_via'].astype(np.int8)
df_test_merged['registration_init_time'] = df_test_merged['registration_init_time'].astype(np.int32)
df_test_merged['expiration_date'] = df_test_merged['expiration_date'].astype(np.int32)

df_train_merged['song_length'] = df_train_merged['song_length'].astype(np.int32)
#-- Since language column contains Nan values we will convert it to 0,
#-- After converting the type of the column we will revert it back to nan
df_train_merged['language'] = df_train_merged['language'].fillna(0)
df_train_merged['language'] = df_train_merged['language'].astype(np.int8)
df_train_merged['language'] = df_train_merged['language'].replace(0, np.nan)

df_test_merged['song_length'] = df_test_merged['song_length'].astype(np.int32)
#-- Since language column contains Nan values we will convert it to 0,
#-- After converting the type of the column we will revert it back to nan
df_test_merged['language'] = df_test_merged['language'].fillna(0)
df_test_merged['language'] = df_test_merged['language'].astype(np.int8)
df_test_merged['language'] = df_test_merged['language'].replace(0, np.nan)

df_test_merged.columns


# Now we will have a changed distribution of datatypes

# In[ ]:


ax = sns.countplot(y = df_test_merged.dtypes, data = df_train_merged)


# ## Data Cleaning
# 
# We have to do the following:
# * Convert date columns from type `integer` to `datetime`.
# * Extract *year*, *month* and *day* information from date columns.

# We have two date columns (**registration_init_time** and **expiration_date**) which are of type `int` and must be converted to `datetime` object.

# In[ ]:


date_cols = ['registration_init_time', 'expiration_date']
for col in date_cols:
    df_train_merged[col] = pd.to_datetime(df_train_merged[col])
    df_test_merged[col] = pd.to_datetime(df_test_merged[col])


# Now to extract *year*, *month* and* day* information for both train and test data.

# In[ ]:


print(len(df_train_merged))
print(df_train_merged['msno'].nunique())


# ## Missing values
# 
# * Before proceeding further we will check whether missing values are present and see how well they can be imputed.
# * There are two possible ways of imputing missing values:
#     1. Replace them with a different category, other than ones already present in the column.
#     2. Replace them proportionally with the ratio of existing categories.
#     
# * I have decided to go with the first one.
# * The following section contains a function for checking presence of missing values in a dataframe and printing the columns having so. We will try it out for both the merged train and test dataframes.

# In[ ]:


#--- Function to check if missing values are present and if so print the columns having them ---
def check_missing_values(df):
    print (df.isnull().values.any())
    if (df.isnull().values.any() == True):
        columns_with_Nan = df.columns[df.isnull().any()].tolist()
    print(columns_with_Nan)
    for col in columns_with_Nan:
        print("%s : %d" % (col, df[col].isnull().sum()))
    
check_missing_values(df_train_merged)
check_missing_values(df_test_merged)


# The following two functions replaces missing values for both columns of type *object* and *float*.

# In[ ]:


#--- Function to replace Nan values in columns of type float with -5 ---
def replace_Nan_non_object(df):
    object_cols = list(df.select_dtypes(include=['float']).columns)
    for col in object_cols:
        df[col]=df[col].fillna(np.int(-5))
       
replace_Nan_non_object(df_train_merged) 
replace_Nan_non_object(df_test_merged)  


# In[ ]:


#--- Function to replace Nan values in columns of type object with 'Others' ---
def replace_Nan_object(df):
    object_cols = list(df.select_dtypes(include=['object']).columns)
    for col in object_cols:
        df[col]=df[col].fillna(' ')
    print (object_cols)

replace_Nan_object(df_train_merged)  
replace_Nan_object(df_test_merged)  
#check_missing_values(cop)
#print(object_cols)


# In[ ]:





# ## Correlations

# In[ ]:


corr_matrix = df_train_merged.corr()


# In[ ]:


''' 
for col in tqdm(cols):
    if df_train_merged[col].dtype == 'object':
        df_train_merged[col] = df_train_merged[col].apply(str)
        df_test_merged[col] = df_test_merged[col].apply(str)

        le = LabelEncoder()
        train_vals = list(df_train_merged[col].unique())
        test_vals = list(df_test_merged[col].unique())
        le.fit(train_vals + test_vals)
        df_train_merged[col] = le.transform(df_train_merged[col])
        df_test_merged[col] = le.transform(df_test_merged[col])
'''        

