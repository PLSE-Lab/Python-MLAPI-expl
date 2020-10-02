#!/usr/bin/env python
# coding: utf-8

# [](http://) INTRODUCTION

# This dataset contains a user's sequence of web pages and the times between them in their web session. The objective is to take a webpage session (a sequence of webpages attended consequently by the same person) and predict whether it belongs to Alice or somebody else.
# 
# More information can be obtained here: https://www.kaggle.com/danielkurniadi/catch-me-if-you-can
# 
# Identifying a user could be useful for detecting like fraud or some other anomalous behavior. It is common to use a user's webpage sessions to identify them, but additional information would be helpful (geographical location, devices, etc.) to distinguish users individually. 
# 
# I originally used the dataset from Catch Me If You Can, but had to use the mlcourse ai 4 dataset to get the labels for the webpages. In addition, I used the following kernels to assist me in data preparation and exploratory data analysis:
# * https://www.kaggle.com/kerneler/starter-catch-me-if-you-can-7181e865-4
# * https://www.kaggle.com/dariavol/alice-baseline
# 
# I am assuming that target = 1 is Alice's sessions and target = 0 is someone else's.
# This is a classification problem. Our primary question we want to ask is, using the dataset, can we predict which sessions belongs to Alice?

# Below are the packages used for this kernel

# In[ ]:


import numpy as np 
import pandas as pd 
import os
import seaborn as sns
import matplotlib.pyplot as plt
import collections
import pickle
import warnings
warnings.filterwarnings("ignore")
import datetime
from sklearn.linear_model import LogisticRegression


# Below is a list of functions used for this kernel

# In[ ]:


#function that to get a list of the sites by mapping the key to the webpage
def get_site_name(site_key, site_dictionary):
    site_list = []
    for i in range(len(site_key)):
        for key, value in site_dictionary.items():
            if value == site_key[i]:
                site_list.append(key)
    return site_list


# DATA PREPARATION

# In this section, I will examine the contents of the datasets. Then, I checked to see if there were missing values and replaced them. Lastly, I standardize the data types of the columns so that there are in an appropriate format for data analysis.

# The command below will display the number of files in the input directory

# In[ ]:


print(os.listdir("../input"))


# To get an understanding of the columns that make up the dataset, I displayed the first five rows along with the number of rows and columns. 

# In[ ]:


df_train = pd.read_csv("../input/train_sessions.csv")
df_train.head()


# In[ ]:


nrow, ncol = df_train.shape
print('There are %i rows and %i columns in the train dataset.' % (nrow, ncol))


# In[ ]:


df_test = pd.read_csv("../input/test_sessions.csv")
df_test.head()


# In[ ]:


nrow, ncol = df_test.shape
print('There are %i rows and %i columns in the test dataset.' % (nrow, ncol))


# Load the site_dic.pkl file and save the contents to the variable site_dic. The pkl file contains the key-value pairs for the sites.

# In[ ]:


with open('../input/site_dic.pkl', 'rb') as f:
    site_dic = pickle.load(f)


# While looking at the train dataset, there were quite a few Not a Number (NaN) values. We will explore how many are present in each column of the datasets.

# In[ ]:


Nan_num_train= df_train.isna().sum()
Nan_num_test= df_test.isna().sum()
print('The number of NaNs in the training set per column \n', Nan_num_train)
print('The number of NaNs in the testing set per column \n ', Nan_num_test)


# There could a few reasons why there are so many NaN values. For now, I am assuming that they visited fewer 10 sites so I replaced the NaN values with zeros.

# In[ ]:


df_train = df_train.fillna(0)
df_test = df_test.fillna(0)

Nan_num_train= df_train.isna().sum()
Nan_num_test= df_test.isna().sum()
print('The number of NaNs in the training set per column \n', Nan_num_train)
print('The number of NaNs in the testing set per column \n ', Nan_num_test)


# I used the dtype command to display the data types of the training and test sets. The session_id, site1, and target are integers. The time columns are objects. The rest of the site columsns are floats. 

# In[ ]:


df_train.dtypes


# In[ ]:


df_test.dtypes


# Convert all the time columns from text to datetime. This step comes in handy when using the describe function.

# In[ ]:


timelist = ['time%s' % i for i in range(1, 11)]
df_train[timelist] = df_train[timelist].apply(pd.to_datetime)
df_test[timelist] = df_test[timelist].apply(pd.to_datetime)


# Use the dtype command to display the data types of the training and test sets. We see that all of the the time columns are now in datetime format.

# In[ ]:


df_train.dtypes


# In[ ]:


df_test.dtypes


# EXPLORATORY DATA ANALYSIS

# In this section, I am primarily analyzing the training dataset. It is quite long, but I wanted to be a bit thorough with my analysis. 
# 
# Here are some of the questions of interest:
# * Are the features correlated?
# * What are the most popular pages?
# * Which time period (year, month, day of the week, etc.) had the most web page views?

# Use the describe function to look at descriptive statistics. 

# In[ ]:


df_train.describe(include='all')


# In[ ]:


df_test.describe(include='all')


# In[ ]:


df_train[list(df_train)].corr()


# In[ ]:


df_test[list(df_test)].corr()


# When plotting the counts of the target values, the figure below shows there are more sessions that do not belong to Alice (target = 0). This means that the data is skewwed and the classes are unbalanced. 
# 
# This could happen for a number of reasons. Maybe Alice was not as active as a user as this other person. Maybe there was not as much data that was collected for Alice, which common for new users. 

# In[ ]:


sns.set(style="whitegrid")
sns.countplot(x="target", data=df_train).set_title('Counts per Target')


# I have created a sub-dataframe called sites_df. This dataframe has the columns target, variable (site1,...,site10), and sites (site key).
# It will be used to create a distribution plot of the sites and a couple of plots of the most popular websites. 

# In[ ]:


sites = [c for c in df_train if c.startswith('site')]
sites_df = pd.melt(df_train, id_vars='target', value_vars=sites, value_name='sites')
sites_df


# In[ ]:


sns.distplot(sites_df['sites']).set_title('Sites Distribution Plot')


# In[ ]:


popular_sites = collections.Counter(sites_df['sites']).most_common(11)
print(popular_sites)


# Remove the order pair (0.0, 122730) from the list as 0.0 is not a real site. 

# In[ ]:


popular_sites.remove((0.0, 122730))
#print(popular_sites)


# Below shows the popular sites for the overall population, Alice, and a user that are not Alice.

# In[ ]:


site, count = zip(*popular_sites)
site_labels = get_site_name(site,site_dic)
#print(site_labels)

y_pos = np.arange(len(site_labels))
plt.barh(y_pos, count, align='center', alpha=0.5)
plt.yticks(y_pos, site_labels)
plt.xlabel('Count')
plt.title('Top %i Sites Overall' % len(site_labels))

plt.show()


# In[ ]:


alice_sites = sites_df[sites_df['target'] == 1]
alice_sites.head(10)
#sns.distplot(alice_sites['sites']).set_title('Alice\'s Sites Distribution Plot')
alice_popular_sites = collections.Counter(alice_sites['sites']).most_common(11)
#print(alice_popular_sites)

site, count = zip(*alice_popular_sites)
site_labels = get_site_name(site,site_dic)
#print(site_labels)

y_pos = np.arange(len(site_labels))
plt.barh(y_pos, count, align='center', alpha=0.5)
plt.yticks(y_pos, site_labels)
plt.xlabel('Count')
plt.title('Alice\'s Top %i Sites Overall' % len(site_labels))

plt.show()


# In[ ]:


notalice_sites = sites_df[sites_df['target'] == 0]
notalice_popular_sites = collections.Counter(notalice_sites['sites']).most_common(11)
#print(notalice_popular_sites)
notalice_popular_sites.remove((0.0, 122529))

site, count = zip(*notalice_popular_sites)
site_labels = get_site_name(site,site_dic)
#print(site_labels)

y_pos = np.arange(len(site_labels))
plt.barh(y_pos, count, align='center', alpha=0.5)
plt.yticks(y_pos, site_labels)
plt.xlabel('Count')
plt.title('Not Alice Top %i Sites Overall' % len(site_labels))

plt.show()


# In the next lines of code, I am creating dataframes of just the time columns per webpage session for the general population called time_df. It has the columns target, variable (time1,...,time10), and times (timestamp).
# 
# From that set, I will generate dataframes for alice and this unamed person. This dataframe will be used to create smaller dataframes for visualizing trends for a specific measurement of time. 
# 
# Here are a couple of resources I used for creatng the sub-dataframes: 
# * https://docs.python.org/2/library/datetime.html
# * https://www.programiz.com/python-programming/datetime

# In[ ]:


times = [c for c in df_train if c.startswith('time')]
times_df = pd.melt(df_train, id_vars='target', value_vars=times, value_name='times')
times_df


# In[ ]:


alice_timesdf = times_df[times_df['target']==1]
notalice_timesdf = times_df[times_df['target']==0]


# In[ ]:


year_df = times_df['times'].dt.year
year_df.value_counts().plot('bar').set_title('Counts per Year')


# In[ ]:


year_alice = alice_timesdf['times'].dt.year
year_alice.value_counts().plot('bar').set_title('Counts per Year For Alice')


# In[ ]:


year_notalice = notalice_timesdf['times'].dt.year
year_notalice.value_counts().plot('bar').set_title('Counts per Year For Not Alice')


# In[ ]:


month_df = times_df['times'].dt.month
month_df.value_counts().plot('bar').set_title('Counts per Month')


# In[ ]:


month_alice = alice_timesdf['times'].dt.month
month_alice.value_counts().plot('bar').set_title('Counts per Month For Alice')


# In[ ]:


month_notalice = notalice_timesdf['times'].dt.month
month_notalice.value_counts().plot('bar').set_title('Counts per Month For Not Alice')


# In[ ]:


hour_df = times_df['times'].map(lambda x: x.strftime('%H'))
hour_df.value_counts().plot('bar').set_title('Counts per Hour')


# In[ ]:


hour_alice = alice_timesdf['times'].map(lambda x: x.strftime('%H'))
hour_alice.value_counts().plot('bar').set_title('Counts per Hour For Alice')


# In[ ]:


hour_notalice = notalice_timesdf['times'].map(lambda x: x.strftime('%H'))
hour_notalice.value_counts().plot('bar').set_title('Counts per Hour For Not Alice')


# In[ ]:


myr_df = times_df['times'].map(lambda x: x.strftime('%m-%Y'))
myr_df.value_counts().plot('bar').set_title('Counts per Month-Year')


# In[ ]:


myr_alice = alice_timesdf['times'].map(lambda x: x.strftime('%m-%Y'))
myr_alice.value_counts().plot('bar').set_title('Counts per Month-Year For Alice')


# In[ ]:


myr_notalice = notalice_timesdf['times'].map(lambda x: x.strftime('%m-%Y'))
myr_notalice.value_counts().plot('bar').set_title('Counts per Month-Year For Not Alice')


# In[ ]:


wkday_df = times_df['times'].map(lambda x: x.weekday())
wkday_df.value_counts().plot('bar').set_title('Counts per Day of the Week')
#Saturday and Sunday are, respectively. the less active days of the week


# In[ ]:


wkday_alice = alice_timesdf['times'].map(lambda x: x.weekday())
wkday_alice.value_counts().plot('bar').set_title('Counts per Day of the Week For Alice')
#Saturday, Wednesday, and Sunday are, respectively, the less active days of the week


# In[ ]:


wkday_notalice = notalice_timesdf['times'].map(lambda x: x.weekday())
wkday_notalice.value_counts().plot('bar').set_title('Counts per Day of the Week For Not Alice')
#Saturday and Sunday are, respectively, the less active days of the week


# I am creating a dataframe of time deltas. The columns of this new dataset are the difference in seconds between each web page. 

# In[ ]:


timedelta = np.zeros((df_train.shape[0], len(timelist)+1))
#len with be 11, columns 0 to 10
timedelta[:,len(timelist)] = df_train['target']
#column 10 is the target
for i in range(len(timelist)-2):
    timedelta[:,i] = (df_train[timelist[i]] - df_train[timelist[i+1]]).abs().dt.seconds


# In[ ]:


#timedelta.mean(axis=1)
#row mean
alice_timedelta = timedelta[timedelta[:,10] == 1]
alice_avgtimedelta = alice_timedelta.mean(axis=1)
plt.hist(alice_avgtimedelta, 4, facecolor='blue', alpha=0.5)
plt.xlabel('Seconds')
plt.ylabel('Count')
plt.title('Alice\'s Average Time Between Sites')
plt.show()


# In[ ]:


notalice_timedelta = timedelta[timedelta[:,10] == 0]
notalice_avgtimedelta = notalice_timedelta.mean(axis=1)
plt.hist(notalice_avgtimedelta, 4, facecolor='blue', alpha=0.5)
plt.xlabel('Seconds')
plt.ylabel('Count')
plt.title('Not Alice Average Time Between Sites')
plt.show()


# FEATURE ENGINEERING

# The time columns need to be transformed before passing the dataset into a machine learning algorithm.
# 
# There are many different ways to encode the time columns (see link below). I have chosen the day of the week. 
# 
# https://stackoverflow.com/questions/46428870/how-to-handle-date-variable-in-machine-learning-data-pre-processing

# In[ ]:


df_new_train = df_train
df_new_train[timelist] = df_new_train[timelist].applymap(lambda x: x.weekday())
df_new_train.head()


# In[ ]:


df_new_test = df_test
df_new_test[timelist] = df_new_test[timelist].applymap(lambda x: x.weekday())
df_new_test.head()


# MAKING PREDICATIONS

# The new test and train sets are reassigned to the variables: X_train (features set), y_train (labels), X_test (test set). They will be passed into scikit learn's Logistic Regression algorithm to make predications on the user (target) given a vector webpages and timestamps. 
# 
# Its seems there wasn't a target column for df_test to create the variable y_test. This makes a little difficult to check the accuracy/performance of the model.

# In[ ]:


X_train = df_new_train[df_new_train.columns[1:20]]
y_train = df_new_train[df_new_train.columns[21]]
X_test = df_new_test[df_new_test.columns[1:20]]


# In[ ]:


logreg = LogisticRegression(random_state=3, solver='lbfgs').fit(X_train, y_train)
predictions = logreg.predict(X_test)
predictions_probabilities = logreg.predict_proba(X_test)


# In[ ]:


values, counts = np.unique(predictions, return_counts=True)
print(values)
print(counts)


# Possible next steps:
# * Split the original train set into cross validation sets and test sets to correctly measure model's accuracy
# * Explore other machine learning algorithms
# * Research and apply methods to deal with unbalanced classes
# 
