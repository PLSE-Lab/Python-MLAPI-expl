#!/usr/bin/env python
# coding: utf-8

# # Events Data Missing Values Analysis
# 
# In this analysis, we will explore the data in the events_data field, particularly looking at which values are defined and which are missing.
# 
# We'll be using the awesome missingno library, so if you haven't used before, feel free to check it out (afterwards): https://github.com/ResidentMario/missingno.
# 
# In most cases we will show only the top 50 columns by amount of non-null entries as we have limited space and many, many features.

# In[ ]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import json

import missingno as msno

get_ipython().run_line_magic('matplotlib', 'inline')

np.random.seed(0)


# ## Read Data
# 
# Read in all the data and process to get some convenient lists.
# 
# Functions taken from [this kernel](https://www.kaggle.com/braquino/890-features).

# In[ ]:


def read_data():
    print('Reading train.csv file....')
    train = pd.read_csv('/kaggle/input/data-science-bowl-2019/train.csv')
    print('Training.csv file have {} rows and {} columns'.format(train.shape[0], train.shape[1]))

    print('Reading test.csv file....')
    test = pd.read_csv('/kaggle/input/data-science-bowl-2019/test.csv')
    print('Test.csv file have {} rows and {} columns'.format(test.shape[0], test.shape[1]))

    print('Reading train_labels.csv file....')
    train_labels = pd.read_csv('/kaggle/input/data-science-bowl-2019/train_labels.csv')
    print('Train_labels.csv file have {} rows and {} columns'.format(train_labels.shape[0], train_labels.shape[1]))

    print('Reading specs.csv file....')
    specs = pd.read_csv('/kaggle/input/data-science-bowl-2019/specs.csv')
    print('Specs.csv file have {} rows and {} columns'.format(specs.shape[0], specs.shape[1]))

    print('Reading sample_submission.csv file....')
    sample_submission = pd.read_csv('/kaggle/input/data-science-bowl-2019/sample_submission.csv')
    print('Sample_submission.csv file have {} rows and {} columns'.format(sample_submission.shape[0], sample_submission.shape[1]))
    return train, test, train_labels, specs, sample_submission

def encode_title(train, test, train_labels):
    # encode title
    train['title_event_code'] = list(map(lambda x, y: str(x) + '_' + str(y), train['title'], train['event_code']))
    test['title_event_code'] = list(map(lambda x, y: str(x) + '_' + str(y), test['title'], test['event_code']))
    all_title_event_code = list(set(train["title_event_code"].unique()).union(test["title_event_code"].unique()))
    # make a list with all the unique 'titles' from the train and test set
    list_of_user_activities = list(set(train['title'].unique()).union(set(test['title'].unique())))
    # make a list with all the unique 'event_code' from the train and test set
    list_of_event_code = list(set(train['event_code'].unique()).union(set(test['event_code'].unique())))
    list_of_event_id = list(set(train['event_id'].unique()).union(set(test['event_id'].unique())))
    # make a list with all the unique worlds from the train and test set
    list_of_worlds = list(set(train['world'].unique()).union(set(test['world'].unique())))
    # create a dictionary numerating the titles
    activities_map = dict(zip(list_of_user_activities, np.arange(len(list_of_user_activities))))
    activities_labels = dict(zip(np.arange(len(list_of_user_activities)), list_of_user_activities))
    activities_world = dict(zip(list_of_worlds, np.arange(len(list_of_worlds))))
    assess_titles = list(set(train[train['type'] == 'Assessment']['title'].value_counts().index).union(set(test[test['type'] == 'Assessment']['title'].value_counts().index)))
    # replace the text titles with the number titles from the dict
    train['title'] = train['title'].map(activities_map)
    test['title'] = test['title'].map(activities_map)
    train['world'] = train['world'].map(activities_world)
    test['world'] = test['world'].map(activities_world)
    train_labels['title'] = train_labels['title'].map(activities_map)
    win_code = dict(zip(activities_map.values(), (4100*np.ones(len(activities_map))).astype('int')))
    # then, it set one element, the 'Bird Measurer (Assessment)' as 4110, 10 more than the rest
    win_code[activities_map['Bird Measurer (Assessment)']] = 4110
    # convert text into datetime
    train['timestamp'] = pd.to_datetime(train['timestamp'])
    test['timestamp'] = pd.to_datetime(test['timestamp'])
    
    
    return train, test, train_labels, win_code, list_of_user_activities, list_of_event_code, activities_labels, assess_titles, list_of_event_id, all_title_event_code

# read data
train, test, train_labels, specs, sample_submission = read_data()
# get usefull dict with maping encode
train, test, train_labels, win_code, list_of_user_activities, list_of_event_code, activities_labels, assess_titles, list_of_event_id, all_title_event_code = encode_title(train, test, train_labels)


# ## Data Augmentation
# Convert the events data to a dataframe after parsing the json.

# In[ ]:


event_data = pd.DataFrame.from_records(
    train.sample(100000).event_data.apply(json.loads).tolist())
# Sort the most non-null columns at the start
event_data = event_data[event_data.isnull().sum().sort_values().index]


# In[ ]:


event_data.describe()


# ## Events Data Exploration
# 

# In[ ]:


print("Total of {} rows and {} features.".format(*event_data.shape))


# In[ ]:


event_data.describe()


# ### Matrix Plot
# This plot shows a quick view of the data sparsity. It can help us understand what we might want to investigate further.

# In[ ]:


msno.matrix(event_data.iloc[:, :50].sample(250))
fig = plt.gcf()


# ### Bar Plot
# 
# Simply plot the percent of missing values for each feature, as well as the total occurrences at the top.

# In[ ]:


msno.bar(event_data.iloc[:, :50])


# The two biggest would be event_count and event_code which is defined for every event. Following that we have game_time for most events, weird that it's undefined for some, what are they?

# In[ ]:


event_data[event_data.game_time.isnull()].describe()


# OK, so these are all for event_code 2000. This is the event that corresponds to the start of the game, so it makes sense that game_time is undefined!
# 
# Round and coordinates are also pretty big, and these likely correspond to specific games.
# 
# The next group description, media_type, identifier and duration all seem to appear together. We'll see this better with our upcoming plots.
# 
# Overall we see that the vast majority of features are defined in less than 10% of the events. It would be great if we could condense these features down in some way that provides more meaning than simply the event code.

# ### Heatmap
# Before we noticed that some features were likely to occur together, let's check if that's actually true by plotting the nullity correlation of the features. This plot shows how the abscence of one variable affects another.

# In[ ]:


msno.heatmap(event_data.iloc[:, :30])


# As expected, we see that duration, description, identifier and media_type all have high scores of correlation (1, or almost 1). We also note that these variables all have strong negative correlation with coordinates which suggests they are in different types of games.
# 
# Looking at the bottom right we also see various strong correlations such as jar_filled with bottles and bottle. These combinations are all likely from the same game which has specific event information.

# ### Dendrogram
# The dendrogram reveals deeper trends than the pairwise ones shown in the heatmap by applying clustering by their nullity correlation.
# 
# The closer to 0 that variables are linked, the more closely they predict each other's nullity. For example, event_count and event_code are linked at 0 as they exactly match in terms of nullity. This contrasts game_time which is less correlated due to the missing values seen earlier.

# In[ ]:


msno.dendrogram(event_data)


# The dendogram shows many of the relationships we saw above, but now can more readily see the different games as we scan down the list of features from top to bottom. For example, bunched together we see jar and jar_filled; cloud, cloud_size and water_level; rocket, height and launched.

# ## Conclusion
# We've seen there are a lot of different fields in the events data and that the majority are mostly null. We've also seen the relationship between various fields which comes mostly from their co-occurrence in games.
# 
# Some interesting next steps would be trying to combine these fields into some smaller number that could provide good features for our machine learning.
