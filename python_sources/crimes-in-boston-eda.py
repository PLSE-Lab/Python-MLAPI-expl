#!/usr/bin/env python
# coding: utf-8

# # Crimes In Boston-EDA

# ### Importing libraries and dataset

# In[ ]:


# Importing Libraries
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


# Importing Dataset
crime_df = pd.read_csv('../input/crime.csv', encoding='latin-1')
oc_df = pd.read_csv('../input/offense_codes.csv', encoding='latin-1')


# In[ ]:


crime_df.head()


# In[ ]:


oc_df.head()


# ### Data Manipulation

# In[ ]:


# Renaming the columns
oc_df = oc_df.rename(columns={'CODE': 'OFFENSE_CODE', 'NAME': 'OFFENSE_NAME'})


# In[ ]:


oc_df.head()


# In[ ]:


# Merging the dataframes on OFFENSE_CODE column
df = pd.merge(crime_df, oc_df, on='OFFENSE_CODE')


# In[ ]:


df.head()


# In[ ]:


# Converting to datetime format
df['DATE'] = pd.to_datetime(df['OCCURRED_ON_DATE'])


# Here, we can see that the first record we have was on 15 June 2015 and this dataset have last record on 03 September 2018.

# In[ ]:


print('min/max date: %s / %s' % (df['DATE'].min(), df['DATE'].max()))
print('Number of days: %d' % ((df['DATE'].max() - df['DATE'].min()).days + 1))
print('Shape: %d rows' % df.shape[0])


# We will now try to deal with the missing values. We can see that the 'SHOOTING' column contains a lot of null values. The reason behind this is that not all crimes that were recorded involved shooting. For example, Larency (crime involving the unlawful taking of the personal property of another person or business).

# ### Dealing with the missing data

# In[ ]:


df.isnull().sum()


# Percent of missing data in our data is almost 6%. 

# In[ ]:


total_cells = np.product(df.shape)
total_missing = df.isnull().sum().sum()

print('Percent of Missing Data: %d%s' % ((total_missing/total_cells)*100, '%'))


# Here, we can see that the column 'SHOOTING' has two values only(nan, 'Y'). Which means that the crimes which did not encountered shooting we assigned with null value for 'SHOOTING' column. We can convert this into a binary feature by assigning 0 value for crimes that did not involved shooting and assigning 1 value to the crimes that involved shooting.

# In[ ]:


df['SHOOTING'].unique()


# In[ ]:


df['SHOOTING'] = df['SHOOTING'].apply(lambda x: 1 if x=='Y' else 0)


# In[ ]:


_ = sns.countplot(df['SHOOTING'])


# Okay! So, we have edited 'SHOOTING' column and we have assigned binary values to it. And from the plot above we can see that the crimes that involve shooting are very less frequent than the crimes that do not involve shooting. 

# In[ ]:


df.isnull().sum()


# Now that we have removed the null values from the 'SHOOTING' column, we can shift our attention to other columns which still have null values. 
# 
# We can see that these columns have one thing in common. They are related to the location or address(except 'UCR_PART'). It could be because the location of the crime was not recorded or probably it wasn't revealed (not sure about it). But since we have reporting area, I don't think we need to worry a lot about these nan values in district and street. We can use reporting are to locate the crime location.
# 
# In UCR_PART we can see that there are only a certain number of offenses for which the value for this feature is null. Maybe there is a particular reason for that.

# In[ ]:


df[df['UCR_PART'].isnull()]['OFFENSE_CODE'].unique()


# In[ ]:


total_cells = np.product(df.shape)
total_missing = df.isnull().sum().sum()

print('Percent of Missing Data: %s%s' % ((total_missing/total_cells)*100, '%'))


# But since these rows are too little in number, it should not cause any harm to our analysis if we just drop those rows. 

# In[ ]:


df = df.dropna()


# In[ ]:


df.isnull().sum()


# ### Analysing and Visualising the data

# In[ ]:


df.head()


# #### Offense Code Group
# So, we can see that the most common types of offenses include Larency, Medical Assistance and Investigate Person

# In[ ]:


plt.figure(figsize=(20, 10))
p = sns.countplot(df['OFFENSE_CODE_GROUP'])
plt.title('Offense Code Group')
_ = plt.setp(p.get_xticklabels(), rotation=90)


# #### Days, Months, Years and Districts

# In[ ]:


df_year = df.groupby(['YEAR']).size().reset_index(name='counts')
df_month = df.groupby(['MONTH']).size().reset_index(name='counts')


# From the plot below we can see that there not a direct relation between the number of crimes occurred and day of the week. But we can see some trends in other features. 
# 
# Number of crimes reported increased significantly by the mid of 2015 and stayed high till the mid of 2017 and then it started decreasing.
# 
# In the months we can see that most numbers of crimes reported were in the months of July, August and September.
# 
# We can also see that the districts B2, C11 and D4 have highest number of crimes reported.

# In[ ]:


fig, axs = plt.subplots(2,2)
fig.set_figheight(15)
fig.set_figwidth(15)

p = sns.countplot(df['DAY_OF_WEEK'], ax=axs[0, 0])
q = sns.lineplot(x=df_month['MONTH'], y=df_month['counts'], ax=axs[1, 0], color='r')
r = sns.lineplot(x=df_year['YEAR'], y=df_year['counts'], ax=axs[0,1], color='g')
s = sns.countplot(df['DISTRICT'], ax=axs[1,1])


# #### Hours

# In[ ]:


df_hour = df.groupby(['HOUR']).size().reset_index(name='counts')


# In[ ]:


fig, axs = plt.subplots(1,2)
fig.set_figheight(5)
fig.set_figwidth(15)

p = sns.countplot(df['HOUR'], ax=axs[0])
q = sns.lineplot(x=df_hour['HOUR'], y=df_hour['counts'], ax=axs[1], color='y')


# We can see from the plot above that the crime rates increase significantly during the 16-19 hours.

# #### Dates
# 
# Let's see the count of crime by each day

# In[ ]:


df_date = df.groupby(['OCCURRED_ON_DATE']).size().reset_index(name='counts')
df_date['date'] =df_date.apply(lambda x: pd.to_datetime(x['OCCURRED_ON_DATE'].split(' ')[0]), axis=1)


# In[ ]:


plt.figure(figsize=(20, 10))
p = sns.lineplot(x=df_date['date'], y=df_date['counts'], color='r')


# #### Location of Crimes
# 
# Let's now analyse the locations of the crimes using the coordinates given in the dataset.

# In[ ]:


df.Lat.replace(-1, None, inplace=True)
df.Long.replace(-1, None, inplace=True)


# In[ ]:


plt.figure(figsize=(10, 10))
p = sns.scatterplot(x='Lat', y='Long', hue='DISTRICT',alpha=0.01, data=df)

