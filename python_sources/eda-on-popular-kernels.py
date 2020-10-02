#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# Kaggle is a very popular website for playing around with data and participating in data science competition.  With the hype around Machine Learning, Deep Learning, and Data Science, it is not surprising that there are many kernels uploaded everyday.  This journal will go through the most popular kernels and perform some EDA.
# 
# ## Questions
# 
# I want to ask the following questions:
# 
# * __[What tags are the most popular?](#poptag)__
# * __[Who has the most kernels?](#mostkern)__
# * __[For the person with the most kernels, what popular tags are associated with them?](#mostkernassoc)__
# * __[Who has made the most revisions?](#mostrev)__
# * __Who has received the most:__
#     * __[Votes?](#mostvotes)__
#     * __[Views?](#mostviews)__
#     * __[Comments?](#mostcomm)__
#     * __[Forks?](#mostfork)__
# * __[What language is most often used with kernels?](#mostlang)__
#     
# __Note:__ The dataset only covers kernels that have had at least 33 votes.

# ## Loading the data
# 
# The first step is to load up the csv file containing the data. We want to get a gist on the data we'll be dealing with.

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv("../input/voted-kaggle-kernels.csv")
df.head()


# Upon looking at the table, the Version History column needs some more cleaning.  As a helpful reference, we'll also create a new column for the number of revisions each kernel has undergone.

# In[21]:


# Need to first separate out all of the versions in each set.
df["Version History"] = df["Version History"].str.split("|")

# Since not all kernels have revisions, we'll need to replace nan with empty lists
df.loc[df["Version History"].isnull(), "Version History"] =     df.loc[df["Version History"].isnull(), "Version History"].apply(lambda x: [])
    
df["Revisions"] = [len(row) for row in df["Version History"]]
df.head()


# In fact, it would best to create a new DataFrame specific for revisions.

# ## Analyzing Revisions
# 
# Since there is a lot of revision data within each kernel, it's best to create a new DataFrame.  In this DataFrame, we'll be retrieving the Kernel ID and the revision number associated with the Kernel ID.

# In[22]:


tmp = {"Kernel ID":[], "Revision":[]}
times = [] # Will be used for indexing
for index, row in df.iterrows():
    for versions in list(row["Version History"]):
        versionNum, versionDate = versions.split(',')
        tmp["Kernel ID"].append(index)
        tmp["Revision"].append(versionNum[versionNum.rfind(" ")+1:])
        # 
        times.append(versionDate)

revDF = pd.DataFrame(tmp, index = pd.DatetimeIndex(times))
revDF.sort_index(inplace=True)
revDF.index.name = "Date"
revDF.head()


# With the new DataFrame created, let's plot out the revision on a yearly basis.

# In[23]:


def nextAxis(maxRow, maxCol):
    curRow, curCol = 0, 0
    while maxRow > curRow:
        yield (curRow, curCol)
        curCol += 1
        if maxCol == curCol:
            curCol = 0
            curRow += 1
            
yearIter = range(2015, 2019)

fig, axes = plt.subplots(nrows=2, ncols=2)

axisGen = nextAxis(2, 2)

for year, color, axisPoints in zip(yearIter, ['b', 'g', 'r', 'y'], axisGen):
    valueDF = revDF['{}'.format(year)].groupby(["Date"])["Kernel ID"].count()
    
    print("{} has {} revisions".format(year, valueDF.sum()))

    # Get the next axis points
    points = [point for point in axisPoints]
    
    # Plot the graph
    valueDF.plot(kind = "line", figsize=(16,16), label=year, ax = axes[points[0], points[1]], color=[color])


# Upon looking at the graphs, 2017 contains the most amount of revisions.  A lot of this is due to hype of Deep Learning and Data Science.
# 
# The number of revisions in 2018 is greater than 2015.  With some more time, the amount of revisions would surpass the amount from 2016.

# ## What tags are the most popular? <a id='poptag'></a>
# 
# Like revisions, it'd be best to separate tags into a separate DataFrame.  Our index will represent the Kernel ID and the tag represents a word.
# 
# __Note: There are multiple words for each index.  So, indexing will return multiple entries__

# In[24]:


tags = df['Tags'].str.strip(',') # Remove unnecessary ','
tags = tags.str.split(',') # Split each tag into a separate element

tmp = {"Tag":[]}
indexes = [] # Will be used for indexing
# Since not all kernels have revisions, we'll need to replace nan with empty lists
index = 0
for row in tags:
    if type(row) != list:
        index += 1
        continue
    for tag in list(row):
        tmp["Tag"].append(tag)
        indexes.append(index)
    index += 1
        
# Now create the tag DataFrame
tagDF = pd.DataFrame(tmp, index = indexes)
tagDF.index.name = "Kernel ID"
tagDF.head()
tagDF.loc[1]


# Now that we separated the tags, we can answer our question.  We'll be omitting tags that appear less than 6 times.

# In[25]:


tagThreshold = 5

tagValueCount = tagDF['Tag'].value_counts()
tagValueCount[tagValueCount > tagThreshold].plot(kind='pie',figsize=(15,15))


# The most common tag is __data visualization__ with __tutorial__ being the second most common.  I don't find this too surprising as many Data Science projects require the use of visualizations to tell stories.

# ## Who has the most kernels? <a id='mostkern'></a>
# 
# Who has created the most kernels in this dataset?  We can query owner to determine the person:

# In[26]:


mostKernels = df['Owner'].value_counts()
mkUser = mostKernels.keys()[0]
mkUser


# The answer is __DanB__, an employee at Kaggle.  Here is a list of his kernels:

# In[27]:


mkuDF = df[df['Owner'] == mkUser]
mkuDF


# ## For the person with the most kernels, what popular tags are associated with them? <a id='mostkernassoc'></a>

# From the previous DataFrame, we notice that there are kernels where Dan defines no tags.  What percentage of his kernels contain tags?

# In[28]:


print("{:.2f}%".format(mkuDF['Tags'].isnull().sum() / len(mkuDF) * 100))


# Now, from the kernels that do contain tags, what is the most popular tag that he uses?

# In[29]:


tagDF.loc[mkuDF.index, 'Tag'].value_counts().plot(kind = 'pie', figsize=(10, 10), fontsize=14)


# The most common tag for Dan is __tutorial__.

# ## Who has made the most revisions? <a id='mostrev'></a>

# Since many kernels have revisions, who made the most amount of revisions?

# In[30]:


revOwn = df.groupby('Owner')['Revisions'].sum().sort_values(ascending=False)
mostRevUser = revOwn.keys()[0]
print("{} with {} revisions".format(mostRevUser, revOwn[mostRevUser]))


# ## Who has received the most

# ### Votes? <a id='mostvotes'></a>

# In[31]:


votesOwner = df.groupby('Owner')['Votes'].sum().sort_values(ascending=False)
mostVotesUser = votesOwner.keys()[0]
print("{} with {} votes".format(mostVotesUser, votesOwner[mostVotesUser]))


# ### Views? <a id='mostviews'></a>

# In[32]:


viewsOwner = df.groupby('Owner')['Views'].sum().sort_values(ascending=False)
mostViewsUser = viewsOwner.keys()[0]
print("{} with {} votes".format(mostViewsUser, viewsOwner[mostViewsUser]))


# ### Comments? <a id='mostcomm'></a>

# In[33]:


commentsOwner = df.groupby('Owner')['Comments'].sum().sort_values(ascending=False)
mostCommentsUser = commentsOwner.keys()[0]
print("{} with {} comments".format(mostCommentsUser, commentsOwner[mostCommentsUser]))


# ### Forks? <a id='mostfork'></a>

# In[34]:


forksOwner = df.groupby('Owner')['Forks'].sum().sort_values(ascending=False)
mostForksUser = forksOwner.keys()[0]
print("{} with {} forks".format(mostForksUser, forksOwner[mostForksUser]))


# ## What language is most often used with kernels? <a id='mostlang'></a>

# In[35]:


df['Language'].value_counts().plot(kind = 'bar')


# While it's not surpising that Python is the most common language for kernels, it's surpising that __R__ is used less often than markdown.

# # Limitations

# The data above is not real-time.  Thus, as time goes on, the statistics in this notebook will no longer be correct.
