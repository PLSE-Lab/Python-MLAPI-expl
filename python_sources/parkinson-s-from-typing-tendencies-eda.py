#!/usr/bin/env python
# coding: utf-8

# # Predicting Parkinson's from Typing Tendencies
# 
# ## Quan Nguyen
# ## March 8, 2018
# 
# <hr>
# 
# ## Introduction
# This notebook explores the keystroke dataset for the study titled **High-accuracy detection of early Parkinson's Disease using multiple characteristics of finger movement while typing**. The notebook goes through various data-cleaning techniques to clean and consolidate the provided data files. A number of observations and visualizations are also included.

# In[35]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import os, gc

import warnings; warnings.filterwarnings('ignore')


# ## About the Data Files
# 
# The provided data files are divided into two sub-folders:
# - Folder 1: `Archived users`, which contains information on the participants' details (gender, year of diagnosis, whether the participant has tremors, etc.)
# - Folder 2: `Tappy Data`, which contains keystroke statistics from specific participants (hold time, current hand, previous hand, etc.)

# ## Reading in User Data (Folder 1)

# Here we are getting a list of all the files present in the `Archived users` folder. However, not all users in the above set have a corresponding typing data in the `Tappy Data` folder. We therefore compute the intersection of these two sets, i.e. the user IDs that both have a user data file in Folder 1, and a typing data file in Folder 2.

# In[37]:


user_file_list = os.listdir('../input/Archived-users/Archived users/')
user_set_v1 = set(map(lambda x: x[5: 15], user_file_list)) # [5: 15] to return just the user IDs


tappy_file_list = os.listdir('../input/Archived-Data/Tappy Data/')
user_set_v2 = set(map(lambda x: x[: 10], tappy_file_list)) # [: 10] to return just the user IDs


user_set = user_set_v1.intersection(user_set_v2)

len(user_set)


# We will now be reading through the files corresponding with the IDs in our user set.

# In[39]:


def read_user_file(file_name):
    f = open('../input/Archived-users/Archived users/' + file_name)
    data = [line.split(': ')[1][: -1] for line in f.readlines()]
    f.close()

    return data


# In[41]:


files = os.listdir('../input/Archived-users/Archived users/')

columns = [
    'BirthYear', 'Gender', 'Parkinsons', 'Tremors', 'DiagnosisYear',
    'Sided', 'UPDRS', 'Impact', 'Levadopa', 'DA', 'MAOB', 'Other'
]

user_df = pd.DataFrame(columns=columns) # empty Data Frame for now

for user_id in user_set:
    temp_file_name = 'User_' + user_id + '.txt' # tappy file names have the format of `User_[UserID].txt`
    if temp_file_name in files: # check to see if the user ID is in our valid user set
        temp_data = read_user_file(temp_file_name)
        user_df.loc[user_id] = temp_data # adding data to our DataFrame

user_df.head()


# Data in a number of columns needs to be processed, cleaned, or reformatted:
# - Changing data in `BirthYear` and `DiagnosisYear` to numeric, if a cell has an invalid data format, change it to NaN. (For example, row 2, `DianosisYear` column.)

# In[43]:


# force some columns to have numeric data type
user_df['BirthYear'] = pd.to_numeric(user_df['BirthYear'], errors='coerce')
user_df['DiagnosisYear'] = pd.to_numeric(user_df['DiagnosisYear'], errors='coerce')


# - Encoding binary data: some columns have True-False data values, here we are converting it to binary data (0s and 1s), for better data processing by machine learning models.

# In[45]:


user_df = user_df.rename(index=str, columns={'Gender': 'Female'}) # renaming `Gender` to `Female`
user_df['Female'] = user_df['Female'] == 'Female' # change string data to boolean data
user_df['Female'] = user_df['Female'].astype(int) # change boolean data to binary data


# Similar cleaning technique is now being applied to a number of columns as below:

# In[53]:


str_to_bin_columns = ['Parkinsons', 'Tremors', 'Levadopa', 'DA', 'MAOB', 'Other'] # columns to be converted to binary data

for column in str_to_bin_columns:
    user_df[column] = user_df[column] == 'True'
    user_df[column] = user_df[column].astype(int)


# - Dummy variables: some categorical data will now be converted to mutually exclusive binary data through dummy variables (aka one-hot encoding)

# In[55]:


# prior processing for `Impact` column
user_df.loc[
    (user_df['Impact'] != 'Medium') &
    (user_df['Impact'] != 'Mild') &
    (user_df['Impact'] != 'Severe'), 'Impact'] = 'None'

to_dummy_column_indices = ['Sided', 'UPDRS', 'Impact'] # columns to be one-hot encoded
for column in to_dummy_column_indices:
    user_df = pd.concat([
        user_df.iloc[:, : user_df.columns.get_loc(column)],
        pd.get_dummies(user_df[column], prefix=str(column)),
        user_df.iloc[:, user_df.columns.get_loc(column) + 1 :]
    ], axis=1)

user_df.head()


# ### Visualizations
# 
# - Number of missing data

# In[56]:


missing_data = user_df.isnull().sum()

g = sns.barplot(missing_data.index, missing_data)
g.set_xticklabels(labels=missing_data.index, rotation=90)

plt.show()


# - Birth year distribution, gender count, and tremor count:

# In[57]:


f, ax = plt.subplots(2, 2, figsize=(20, 10))

sns.distplot(
    user_df.loc[user_df['Parkinsons'] == 0, 'BirthYear'].dropna(axis=0),
    kde_kws = {'label': "Without Parkinson's"},
    ax = ax[0][0]
)
sns.distplot(
    user_df.loc[user_df['Parkinsons'] == 1, 'BirthYear'].dropna(axis=0),
    kde_kws = {'label': "With Parkinson's"},
    ax = ax[0][1]
)

sns.countplot(x='Female', hue='Parkinsons', data=user_df, ax=ax[1][0])
sns.countplot(x='Tremors', hue='Parkinsons', data=user_df, ax=ax[1][1])

plt.show()


# In[58]:


user_df.columns


# - Count for different types in `Sided`, `UPDRS`, and `Impact`:

# In[59]:


f, ax = plt.subplots(2, 2, figsize=(20, 10))

sns.barplot(
    ['Sided_Left', 'Sided_None', 'Sided_Right'],
    user_df[['Sided_Left', 'Sided_None', 'Sided_Right']].sum(),
    ax=ax[0][0]
)
sns.barplot(
    ['UPDRS_1', 'UPDRS_2', 'UPDRS_3', 'UPDRS_4', "UPDRS_Don't know"],
    user_df[['UPDRS_1', 'UPDRS_2', 'UPDRS_3', 'UPDRS_4', "UPDRS_Don't know"]].sum(),
    ax=ax[0][1]
)
sns.barplot(
    ['Impact_Medium', 'Impact_Mild', 'Impact_None', 'Impact_Severe'],
    user_df[['Impact_Medium', 'Impact_Mild', 'Impact_None', 'Impact_Severe']].sum(),
    ax=ax[1][0]
)
ax[1][1].axis('off')

plt.show()


# ## Reading in Typing Data (Folder 2)
# 
# Here we will read in a file in our second folder and explore it. From that we will consequently write a general function to process similar files later on.
# 
# (Note that the data in the `Hold time`, `Latency time`, and `Flight time` columns are in milliseconds.)

# In[60]:


file_name = '0EA27ICBLF_1607.txt' # an arbitrary file to explore


# In[61]:


df = pd.read_csv(
    '../input/Archived-Data/Tappy Data/' + file_name,
    delimiter = '\t',
    index_col = False,
    names = ['UserKey', 'Date', 'Timestamp', 'Hand', 'Hold time', 'Direction', 'Latency time', 'Flight time']
)

df = df.drop('UserKey', axis=1)

df.head()


# Next we will be using the `pd.to_datetime()` and `pd.to_numeric()` functions to force-convert our data to be stored in the correct datatypes.

# In[62]:


df['Date'] = pd.to_datetime(df['Date'], errors='coerce', format='%y%M%d').dt.date
# converting time data to numeric
for column in ['Hold time', 'Latency time', 'Flight time']:
    df[column] = pd.to_numeric(df[column], errors='coerce')

df = df.dropna(axis=0)


# Similarly, below we are dropping any entries that don't have the correct data in the `Hand` and `Direction` columns.

# In[63]:


# cleaning data in Hand
df = df[
    (df['Hand'] == 'L') |
    (df['Hand'] == 'R') |
    (df['Hand'] == 'S')
]

# cleaning data in Direction
df = df[
    (df['Direction'] == 'LL') |
    (df['Direction'] == 'LR') |
    (df['Direction'] == 'LS') |
    (df['Direction'] == 'RL') |
    (df['Direction'] == 'RR') |
    (df['Direction'] == 'RS') |
    (df['Direction'] == 'SL') |
    (df['Direction'] == 'SR') |
    (df['Direction'] == 'SS')
]


# For our purposes, we will only be looking at the mean (average) time of the `Hold time`, `Latency time`, and `Flight time` columns in groups of the same `Direction` data. In other words, we will split our current Data Frame into groups of `LL` direction, of `LS` direction, of `LR` direction, and so on. (`L` denotes left hand, `R` denotes right hand, and `S` denotes the spacebar).
# 
# This calculation could be achived easily by the function `groupby()`:

# In[64]:


direction_group_df = df.groupby('Direction').mean()
direction_group_df


# As we can see, the means of numeric columns are taken with respect to the data in the `Direction` column (there are in total 9 different groupings)--all this data is now stored in the `direction_group_df` variable.
# 
# Recall that this is simply a summary of keystroke data of a specifica user at a specific time period. For the sake of convenience, we will put all the commands above into a function to process this data. Note that the function will return the data in an ordered NumPy array for the sake of runtime.

# In[65]:


def read_tappy(file_name):
    df = pd.read_csv(
        '../input/Archived-Data/Tappy Data/' + file_name,
        delimiter = '\t',
        index_col = False,
        names = ['UserKey', 'Date', 'Timestamp', 'Hand', 'Hold time', 'Direction', 'Latency time', 'Flight time']
    )

    df = df.drop('UserKey', axis=1)

    df['Date'] = pd.to_datetime(df['Date'], errors='coerce', format='%y%M%d').dt.date

    # converting time data to numeric
    #print(df[df['Hold time'] == '0105.0EA27ICBLF']) # for 0EA27ICBLF_1607.txt
    for column in ['Hold time', 'Latency time', 'Flight time']:
        df[column] = pd.to_numeric(df[column], errors='coerce')
    df = df.dropna(axis=0)

    # cleaning data in Hand
    df = df[
        (df['Hand'] == 'L') |
        (df['Hand'] == 'R') |
        (df['Hand'] == 'S')
    ]

    # cleaning data in Direction
    df = df[
        (df['Direction'] == 'LL') |
        (df['Direction'] == 'LR') |
        (df['Direction'] == 'LS') |
        (df['Direction'] == 'RL') |
        (df['Direction'] == 'RR') |
        (df['Direction'] == 'RS') |
        (df['Direction'] == 'SL') |
        (df['Direction'] == 'SR') |
        (df['Direction'] == 'SS')
    ]

    direction_group_df = df.groupby('Direction').mean()
    del df; gc.collect()
    direction_group_df = direction_group_df.reindex(['LL', 'LR', 'LS', 'RL', 'RR', 'RS', 'SL', 'SR', 'SS'])
    direction_group_df = direction_group_df.sort_index() # to ensure correct order of data
    
    return direction_group_df.values.flatten() # returning a numppy array


# In[66]:


file_name = '0EA27ICBLF_1607.txt' # an arbitrary file to explore
tappy_data = read_tappy(file_name)

tappy_data # which corresponds to the DataFrame above in order


# Since a user can have multiple typing data files ranging multiple months (for example, **0EA27ICBLF** has **0EA27ICBLF_1607.txt** and **0EA27ICBLF_1608.txt**), we are now writing a function that takes in a user ID, searches for all typing data files for that user, and returns the mean of corresponding `Direction` and `Time` data.

# In[67]:


def process_user(user_id, filenames):
    running_user_data = np.array([])

    for filename in filenames:
        if user_id in filename:
            running_user_data = np.append(running_user_data, read_tappy(filename))
    
    running_user_data = np.reshape(running_user_data, (-1, 27))
    return np.nanmean(running_user_data, axis=0) # ignoring NaNs while calculating the mean


# In[68]:


filenames = os.listdir('../input/Archived-Data/Tappy Data/')

user_id = '0EA27ICBLF'
process_user(user_id, filenames)


# Note that this array doesn't correspond to the array above, as this array is the mean of the two arrays from the two files mentioned earlier. Next, we will loop through all user IDs we have in our `user_df` DataFrame, calling our `process_user()` function and creating a new DataFrame in the process.

# In[69]:


column_names = [first_hand + second_hand + '_' + time for first_hand in ['L', 'R', 'S'] for second_hand in ['L', 'R', 'S'] for time in ['Hold time', 'Latency time', 'Flight time']]

user_tappy_df = pd.DataFrame(columns=column_names)

for user_id in user_df.index:
    user_tappy_data = process_user(str(user_id), filenames)
    user_tappy_df.loc[user_id] = user_tappy_data

# some preliminary data cleaning
user_tappy_df = user_tappy_df.fillna(0)
user_tappy_df[user_tappy_df < 0] = 0    

user_tappy_df.head()


# ## Combining Data From Two Folders
# 
# Here we are concatenating `user_df` DataFrame, which contains information on the users (year of birth, year of diagnosis, drug use, etc.), and `user_tappy_df` DataFrame, which contains typing data for corresponding users.

# In[70]:


combined_user_df = pd.concat([user_df, user_tappy_df], axis=1)
combined_user_df.head()


# ### Visualizations
# 
# First we will use boxplots to visualize distributions of different time data (hold time, latency time, and flight time) between participants with and without Parkinsons's. Each subplot will contain data in a specific typing switch type--for example, the top left subplot contains typing data when participants go from a left-hand key to another left-hand key (denoted as **LL** above the subplot), while the top right one contains data when participants switch from a left-hand key to a space (**LS**).

# In[71]:


f, ax = plt.subplots(3, 3, figsize=(10, 5))
#f.tight_layout()
plt.subplots_adjust(
    #left  = 0.5,
    right = 3,
    #bottom = 0.5,
    top = 3,
)

for i in range(9):
    temp_columns = column_names[3 * i : 3 * i + 3]
    stacked_df = combined_user_df[temp_columns].stack().reset_index()
    
    stacked_df = stacked_df.rename(columns={'level_0': 'index', 'level_1': 'Type', 0: 'Time'})
    stacked_df = stacked_df.set_index('index')

    for index in stacked_df.index:
        stacked_df.loc[index, 'Parkinsons'] = combined_user_df.loc[index, 'Parkinsons']
    
    sns.boxplot(x='Type', y='Time', hue='Parkinsons', data=stacked_df, ax=ax[i // 3][i % 3]).set_title(column_names[i * 3][: 2], fontsize=20)
    
plt.show()


# **[To be continued]**
# 
# ## Thank-You
# Thank you for reading through this notebook. As always, let me know your feedback on this in the comment section below, and upvote the notebook if you have found this useful.
# 
# ![](https://marketplace.canva.com/MACIiWd5o9E/2/0/thumbnail_large/canva-pink-and-blue-diagonal-wave-line-thank-you-postcard-MACIiWd5o9E.jpg)

# In[ ]:




