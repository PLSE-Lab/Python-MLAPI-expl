#!/usr/bin/env python
# coding: utf-8

# Hey there guys, after looking into [managers_id][1] and [buildings_id][2], now I want to focus on the "interest_level" associated with the keywords in the feature column.
# 
# Bear with me for a bit if you have some time, it won't take long.
# 
#   [1]: https://www.kaggle.com/den3b81/two-sigma-connect-rental-listing-inquiries/do-managers-matter-some-insights-on-manager-id
#   [2]: https://www.kaggle.com/den3b81/two-sigma-connect-rental-listing-inquiries/some-insights-on-building-id

# In[ ]:


# load the modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# get the training data 
train_df = pd.read_json('../input/train.json')


# Let's create a big dataframe containing all the dummies for the different keywords in "features".
# The lower() method will allow us to group equal keywords which are written differently, i.e. "Laundry in Building" and "Laundry In Building". 
# 
# The columns of this dataframe will be 1 if the keyword IS used in the entry,  0 otherwise.

# In[ ]:


dummies = train_df['features'].str.join(sep=',').str.lower().str.get_dummies(sep=',')
dummies.head(10)


# In[ ]:


dummies.shape


# 1294 different keywords is quite a lot... and we are not even trying to split things like "$1000. move-in visa giftcard will be handed to new tenants upon rental lease signing" to obtain new ones.
# 
# Anyway, let's focus on the most frequent of the unique entries. 

# In[ ]:


dummies.sum().sort_values(ascending= False).head(40).plot(kind = 'bar', figsize = (10,5))


# You might recognize we have similar results to the wordcloud you've seen in [this][1] top EDA notebook by SRK.
# 
# 
#   [1]: https://www.kaggle.com/sudalairajkumar/two-sigma-connect-rental-listing-inquiries/simple-exploration-notebook-2-connect
# 
# Let's compute the average interest levels for the most frequent 100 keywords, and store this information in a new dataframe.

# In[ ]:


frequent_features = dummies.sum().sort_values(ascending= False).head(100).index;
ff_interest_df = pd.DataFrame(index = frequent_features, columns = ['low','medium','high','count'])

for feature in frequent_features:
    # select index where feature is present
    ixes = dummies[feature].astype(bool)
    temp = pd.concat([dummies[ixes][feature],train_df['interest_level']], axis = 1, join = 'inner')
    ff_interest_df.loc[feature] = temp.interest_level.value_counts()/len(temp)
    ff_interest_df.loc[feature,'count'] = len(temp)
    
print(ff_interest_df.head(5))


# And now... let's see which frequent keywords are associated with higher interest levels by sorting the values of the 'high' column.

# In[ ]:


ff_interest_df['high'].sort_values(ascending = False).plot(kind = 'bar', figsize = (15,5))
plt.gca().set_ylabel('% of high interest')


# I think that is what you were expecting to see from the title of this notebook? Right? :D
# 
# Indeed, almost 40% of entries with "Short Term Allowed" have high interest level. "Actual Apt. Photos" and "Sauna" come 2nd and 3rd with over 30%.

# Plotting the averages of all interest levels for the most frequent features yields

# In[ ]:


ff_interest_df[['low','medium','high']].plot(kind = 'bar', stacked = True, figsize = (15,5))


# .... at first it looks like different keywords raise different interest levels.
# 
# However, before trying to harness this kind of analysis for predictive purposes we have to consider two things: 
# 
# 1) how these values compare with the overall averages of the interest levels
# 
# and 
# 
# 2) how many instances we have for each keyword. 

# In[ ]:


# let's put the overall averages and count in the picture
avg_interest_levels = train_df.interest_level.value_counts()/len(train_df)

ff_interest_df.loc['AVERAGES'] = avg_interest_levels
ff_interest_df.loc['AVERAGES','count'] = len(train_df)
ff_interest_df.tail()


# In[ ]:


# let's plot again and add a second plot with the count
ff_interest_df.sort_values(by = 'count', ascending = False)[['low','medium','high']].plot(kind = 'bar', stacked = True, figsize = (15,5))
plt.figure()
ff_interest_df['count'].sort_values(ascending = False).plot(kind = 'bar', figsize = (15,5))


# As one may expect, it seems like the interest levels for the most frequent keywords are very similar to the average ones. 
# 
# Besides, the keywords with very different average levels are not that numerous, after all.
# 
# Things start to become interesting, i.e. different from AVERAGE, after 'fireplace', which has a rather low interest level and a count of 919 (I did not count 'prewar' as it should be merged with 'pre-war'. Same goes for 'hardwood; which should be merged with 'hardwoo floors'.).

# In[ ]:


ff_interest_df.loc['fireplace']


# 'short term allowed', 'actual apt. photos', and 'sauna' sure raise a lot of interest but their count is really low. 
# 
# Maybe not that interesting after all?
# 

# In[ ]:


ff_interest_df.sort_values(by = 'high', ascending = False).head(3)


# Let me know what you think about it, this is just a preliminary investigation!
# 
# Cheers
