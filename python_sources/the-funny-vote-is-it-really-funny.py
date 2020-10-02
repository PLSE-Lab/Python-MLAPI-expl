#!/usr/bin/env python
# coding: utf-8

# ## Background
# 
# One of the dataset provided by yelp is the review dataset. When I explored the dataset, I was so excited when I saw that reviews in yelp has 3 types of vote: useful, funny, and cool. Yes, I am not an actually avid user of yelp myself, as it is not popular in my country. However, detecting humor is indeed a tricky task in NLP, and this 'funny' voting system may offer a way to retrieve a working training set.
# 
# As what every data scientist do before they try to build a model, (ahem..) we start by doing EDA. Yet, my enthusiasm was cut short, because when I manually run through some of the (supposed-to-be-)funny reviews, my face was like this..
# 
# <img src='https://media.giphy.com/media/2qpCqEkUCRKmY/giphy.gif' alt='man grin'></img>
# 
# So, does the 'funny' vote actually 'useful'? Or is it 'cool' to just shut it off altogether if it is, actually, meaningless?

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns

# options
pd.set_option('display.max_colwidth', -1)

# extra config to have better visualization
sns.set(
    style='whitegrid',
    palette='coolwarm',
    rc={'grid.color' : '.96'}
)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['figure.titlesize'] = 12
plt.rcParams['figure.figsize'] = 12, 7.5


# ## Data Loading

# In[ ]:


# load data
rvw_df = pd.read_csv("../input/yelp_review.csv")
biz_df = pd.read_csv("../input/yelp_business.csv")

# check what is inside dataframe
print('This is the review dataframe columns:')
print(rvw_df.dtypes)


# Our variable of interest is the *funny* column. First, let's explore basic statistics of the column.

# ## Basic Information

# In[ ]:


rvw_df['funny'].describe()


# There are about 5.26 millions reviews. The mean of *funny* votes is 0.5 and unsuprisingly the third quartile is 0 vote. This means less than a third of the reviews have more than 0 votes for funny. Let's see how many.

# In[ ]:


zero_funny = len(rvw_df[rvw_df['funny'] == 0])
zero_funny_prop = (zero_funny * 100) / len(rvw_df)
'There are {:,d} ({:2.3f}% of overall) reviews with 0 funny votes'.format(zero_funny, zero_funny_prop)


# About 1.1 million reviews have *funny* votes more than 0, quite many, we can work with that. 

# ## Exploring Funny Reviews
# 
# Now, prepare yourself to laugh after reading the funniest review, worth the **1,481** votes!

# In[ ]:


print(rvw_df['text'].values[rvw_df['funny'] == max(rvw_df['funny'])][0])


# **Well**, ok, I smile a bit, but definitely not something **really, really** funny that was promised by the **1,481** votes it has accumulated. Even more interesting, we can see the top 10 most voted as funny reviews and observe the business id.

# In[ ]:


rvw_df[['business_id', 'funny']].sort_values(by='funny', ascending=False).head(10)


# Wait, what happened? All of them reviewed the same business id! From the funniest review above, we got a sense that those reviews are intended for a restaurant. I do not know what happened, but what if we see the top 50 then?

# In[ ]:


rvw_df[['business_id', 'funny']].sort_values(by='funny', ascending=False).head(50)


# <img src='https://media.giphy.com/media/9D7dHki8a0eTSmD3vR/giphy.gif' alt='What?'></img>
# 
# I have to go to this place then, the funniest restaurant! who are you  **DN0b4Un8--Uf6SEWLeh0UA**? (in fact, the top 80 funniest reviews were for them).

# In[ ]:


biz_df[biz_df['business_id'] == 'DN0b4Un8--Uf6SEWLeh0UA']


# **Oh no**, they are closed already (*is_open* is 0). I hope that's not because of the *funny* reviews... <br />
# 
# ## Distribution of Funny Vote
# 
# I then start to doubt this *funny* votes, let's see some reviews that are within the *normal* range and see if they are indeed humorous. We seek out first what's the range of *funny* votes. I cut the 78% of reviews that has 0 *funny* votes, remove the seemingly bogus *funny* reviews for ABC restaurant, and then find out the 99th percentile.

# In[ ]:


rvw_df[(rvw_df['funny'] > 0) & (rvw_df['business_id'] != 'DN0b4Un8--Uf6SEWLeh0UA')]['funny'].quantile([.95, .99, .999, 1])


# Let's first observe the top review after we remove ABC restaurant.

# In[ ]:


print(rvw_df['text'].values[rvw_df['funny'] == 216][0])


# **Ok,** I personally think this one actually deserve the funny votes it has gather. Funniest in yelp, tho, hmm that is hard to tell. Now let's see the distribution.

# In[ ]:


ax = rvw_df[(rvw_df.funny > 0) & (rvw_df.funny <= 16)]['funny'].hist(
    bins=range(1, 16, 1), 
    normed=True
)

_ = ax.set_yticklabels(['{:.0f}%'.format(x*100) for x in ax.get_yticks()])
_ = ax.set_title('Normalized Distribution for Funny Votes')
_ = ax.set_xlabel('Number of Funny Votes')


# The 99th percentile is 16 votes, how many are there?

# In[ ]:


'There are {:,d} reviews with 16 votes'.format(len(rvw_df[rvw_df['funny'] == 16]))


# In[ ]:


rvw_df[['text','useful','cool']][rvw_df['funny'] == 16].head(3)


# In social media, popularity often plays important part as how many attention a user get (kind of logical). After reading those samples, and many more others that I do not write here for the sake of simplicity, one can come to ask whether the 3 votes that yelp provided (useful, funny, and cool) are distinctive from the eyes of the user.
# 
# **Like in the sample**, they indeed have some parts quite humorous, but the rest are plain. And *funnily*, the *useful* and *cool* votes are quite close to 16 as well. Is it because user votes more to the reviews they like, not really because they are funny, or cool?

# ## Correlation with Other Vote Type

# I proceed by subsetting the dataset only to those reviews that have *funny* votes between 1 - 43 (99.9% percentile), and see their useful and cool votes number. Why? Because in this analysis, we focus on finding out whether *funny* votes is actually independent. Also, it is better to filter outliers for useful and cool.

# In[ ]:


rvw_df[rvw_df['useful'] > 0]['useful'].quantile(.999)


# In[ ]:


rvw_df[rvw_df['cool'] > 0]['cool'].quantile(.999)


# For simplicity, we will filter review with at least one of the category > 0 and for all categories are < 50.

# In[ ]:


rvw_fun_df = rvw_df[((rvw_df['funny'] > 0) | (rvw_df['useful'] > 0) | (rvw_df['cool'] > 0))
                    & (rvw_df['funny'] <= 50) 
                    & (rvw_df['useful'] <= 50) 
                    & (rvw_df['cool'] <= 50) 
                    & (rvw_df['business_id'] != 'DN0b4Un8--Uf6SEWLeh0UA')
                   ].reset_index()[['funny','useful','cool']].copy(deep=True)
'There are {:,d} reviews that fit the criteria'.format(len(rvw_fun_df))


# In[ ]:


for c in ['funny', 'useful', 'cool']:
    jitter = np.random.normal(0, 0.002, size=len(rvw_fun_df))
    rvw_fun_df['z_' + c] = ((rvw_fun_df[c] - rvw_fun_df[c].mean()) / (rvw_fun_df[c].max() - rvw_fun_df[c].min())) + jitter
rvw_fun_df.head(5)


# In[ ]:


fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(12,20))

_ = rvw_fun_df.plot.scatter(
    ax=ax[0],
    x='z_funny',
    y='z_useful',
    color='indianred',
    alpha=.1
)

_ = rvw_fun_df.plot.scatter(
    ax=ax[1],
    x='z_funny',
    y='z_cool',
    color='violet',
    alpha=.1
)

_ = rvw_fun_df.plot.scatter(
    ax=ax[2],
    x='z_cool',
    y='z_useful',
    color='hotpink',
    alpha=.1
)


# As expected, linear relationship among the three votes. And what are the Pearson correlation?

# In[ ]:


rvw_fun_df[['funny','useful','cool']].corr()


# **Gotcha!**

# ## Conclusion
# 
# **Based on the plots and the score, we can see that they are actually correlated.** Yes, this does not proof that the *funny* vote is totally unusable, we have seen some reviews that deserve the vote. However, I would argue that using 3 different type of votes is not really significant. **The highly voted review tend to get high number of vote for all useful, cool, and funny.**
# 
# Very well then, I need to continue my journey on finding the corpus worth for humor detection... or maybe label them one-by-one... *sigh*

# In[ ]:




