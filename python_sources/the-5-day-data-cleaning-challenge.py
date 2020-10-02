#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Required Libraries for the Exercise
import numpy as np              # linear algebra
import pandas as pd             # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns           # plotting
import matplotlib.pyplot as plt # plotting

from scipy import stats         # Box-Cox Tranformation (Day 2)
from mlxtend.preprocessing     import minmax_scaling       # Minimum - Maximum Scaling (Day 2)
import datetime                 # date-time transformations (Day 3)
import chardet                  # character encoding module (Day 4)
import fuzzywuzzy
from fuzzywuzzy     import process              # text mining (Day 5)

# Even though I dont think this is necessary, it seems Rachael gives importance to repoducibility.
np.random.seed(23)


# In[2]:


# Data Import for this Exercises
sf_permits  = pd.read_csv("../input/building-permit-applications-data/Building_Permits.csv")
kickstarter = pd.read_csv("../input/kickstarter-projects/ks-projects-201801.csv")
gunviolence = pd.read_csv("../input/gun-violence-data/gun-violence-data_01-2013_03-2018.csv")
earthquakes = pd.read_csv("../input/earthquake-database/database.csv")
suicide_att = pd.read_csv("../input/pakistansuicideattacks/PakistanSuicideAttacks Ver 11 (30-November-2017).csv", encoding = "Windows-1252") # as Rachael suggests.


# In previous chunks, I have imported required libraries and the related data for the next exercises. Let's start with the questions of day 1. I will follow the steps of [Rachael](https://www.kaggle.com/rtatman) but will not fork any of her kernels since I will compress 5-day challenge in 1 kernel.

# # Day 1: Handling Missing Values
# In the next two chunks, we will have a peek at our data and its structure.

# In[5]:


sf_permits.head(6)


# In[10]:


sf_permits.dtypes


# Next, we are in need of the count of the missing values:

# In[21]:


na_count  = sf_permits.isnull().sum()
row_count = np.product(sf_permits.shape)
na_perc   = 100 *  na_count.sum() / row_count
na_perc   = na_perc.round(2)
print('Overall percentage of NA values in this dataset is %{0}.'.format(na_perc))


# Now it is my turn to look at the data (next chunk provides an overview) and have a guess for the following variables:
# 
# - `Street Number Suffix`: *This column is empty et all. There is no record, it is not been used.*
# - `Zipcode`: *There are some missing values in this column.*

# In[48]:


print('There are {0} NA values in Street Number Suffix variable while {1} NA values in Zipcode variable.'.format(na_count[7], na_count[40]))


# As Rachael suggests, now i will drop NAs by rows and columns, consecutively.

# In[53]:


sf_permits_nonna_rows = sf_permits.dropna()
sf_permits_nonna_cols = sf_permits.dropna(axis = 1)
print('Rows left after dropping rows with at least one NA value: {0} \n'. format(sf_permits_nonna_rows.shape[0]))
print('Columns left after dropping columns with at least one NA value: {0} \n'. format(sf_permits_nonna_cols.shape[1]))


# Next challenge is imputation. First with the value that comes after, if that is NA too, then 0. 

# In[54]:


sf_permits_imputated = sf_permits.fillna(method = 'bfill', axis =0).fillna(0)
sf_permits_imputated.head(6)


# # Day 2: Scaling and Normalization
# 
# **My turn!**
# For the following example, decide whether scaling or normalization makes more sense.
# 
# > You want to build a linear regression model to predict someone's grades given how much time they spend on various activities during a normal school week. You notice that your measurements for how much time students spend studying aren't normally distributed: some students spend almost no time studying and others study for four or more hours every day. Should you scale or normalize this variable?
# 
# ** My answer is *normalizing***
# 
# > You're still working on your grades study, but you want to include information on how students perform on several fitness tests as well. You have information on how many jumping jacks and push-ups each student can complete in a minute. However, you notice that students perform far more jumping jacks than push-ups: the average for the former is 40, and for the latter only 10. Should you scale or normalize these variables?
# 
# ** My answer is *scaling***
# 
# Now it's time to scaling of goal column. It seems this column does not need any scaling.

# In[57]:


goal = kickstarter.goal
scaled_goal = minmax_scaling(goal, columns = [0])

fig, ax = plt.subplots(1,2)
sns.distplot(goal, ax = ax[0])
ax[0].set_title("Original Data")
sns.distplot(scaled_goal, ax = ax[1])
ax[1].set_title("Scaled Data")


# When we check for `pledged` column as it has been asked; we find out that normalisation is needed and a similar pattern is observed to `usd_pledged_real` column.

# In[61]:


index_of_pledged = kickstarter.pledged > 0
positive_pledges = kickstarter.pledged.loc[index_of_pledged]
scaled_pledges = stats.boxcox(positive_pledges)[0]

fig, ax = plt.subplots(1,2)
sns.distplot(positives_pledges, ax = ax[0])
ax[0].set_title("Original Data")
sns.distplot(scaled_pledges, ax = ax[1])
ax[1].set_title("Normalized Data")


# For extra practice, I have chosen Gun Violence Dataset. I will first have a look at some of the variables' distributoin plots and then choose some.

# In[9]:


gunviolence.head()


# In[12]:


fig, ax = plt.subplots(1,2)
sns.distplot(gunviolence.n_killed, ax = ax[0])
ax[0].set_title("Dead People by Gun Violence")
sns.distplot(gunviolence.n_injured, ax = ax[1])
ax[1].set_title("Injured People by Gun Violence")


# Above two variables are, seriously in need of, normalisation. So I will do Box-Cox transformation for both of them. It did not change the distribution much but it is much practice I guess ! :)

# In[23]:


# Figuring out positive values.(Box-Cox only accepts positive values)
killed = gunviolence[gunviolence.n_killed > 0].n_killed
injured = gunviolence[gunviolence.n_injured > 0].n_injured

# Box-Cox Transformation
killed_boxcox = stats.boxcox(killed)[0]
injured_boxcox = stats.boxcox(injured)[0]

# Plot!
fig, ax = plt.subplots(2,2)
sns.distplot(gunviolence.n_killed, ax = ax[0, 0])
ax[0, 0].set_title("Dead")
sns.distplot(killed_boxcox, ax = ax[0, 1])
ax[0, 1].set_title("Dead (Box-Cox)")
sns.distplot(gunviolence.n_injured, ax = ax[1, 0])
ax[1, 0].set_title("Injured")
sns.distplot(injured_boxcox, ax = ax[1, 1])
ax[1, 1].set_title("Injured (Box-Cox)")


# # Day 3: Parsing Dates

# In[8]:


print('In Earthquakes dataset, date column is formatted as {0} by default.'.format(earthquakes['Date'].dtype))


# In[15]:


earthquakes['Date_parsed'] = pd.to_datetime(earthquakes['Date'], infer_datetime_format = True)
print('In Earthquakes dataset, parsed date column is formatted as {0} by using infer_datetime_format option since there were more than 1 type.'      .format(earthquakes['Date_parsed'].dtype))


# Instead of a graph, I have grouped the data by 'month' column and shown sizes.

# In[ ]:


earthquakes['month'] = earthquakes['Date_parsed'].dt.month
earthquakes.groupby(['month']).size()


# # Day 4 :Character Encodings

# In[11]:


with open("../input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv", 'rb') as rawdata:
    result = chardet.detect(rawdata.read(1000000))

print(result)


# In[14]:


killerpolice = pd.read_csv("../input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv", encoding = "Windows-1252")
killerpolice.head()


# In[ ]:


# Now, let's save.
killerpolice.to_csv("PoliceKillingsUS_utf8.csv")


# # Day 5: Inconsistent Data Entry

# In[3]:


suicide_att['City'] = suicide_att['City'].str.lower()
suicide_att['City'] = suicide_att['City'].str.strip()


# In[6]:


matches = fuzzywuzzy.process.extract("kuram agency", suicide_att['City'], limit = 10, scorer = fuzzywuzzy.fuzz.token_sort_ratio)
matches


# In[7]:


def column_fixer(df, col, string_to_match, min_ratio = 90):
    unique_vals = df[col].unique()
    matches = fuzzywuzzy.process.extract(string_to_match, unique_vals, limit = 10, scorer = fuzzywuzzy.fuzz.token_sort_ratio)
    matches_above_min = [matches[0] for matches in matches if matches[1] >= min_ratio] # first column names, second column scores.
    matched_rows = df[col].isin(matches_above_min)
    df.loc[matched_rows] = string_to_match
    print("Voila!")


# In[14]:


column_fixer(df = suicide_att, col = "City", string_to_match = "kuram agency")


# In[16]:


fixed_cities = suicide_att['City'].unique()
fixed_cities.sort()
fixed_cities


# It's the end of another challenge!
