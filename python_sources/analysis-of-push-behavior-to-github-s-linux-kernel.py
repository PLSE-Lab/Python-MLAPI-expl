#!/usr/bin/env python
# coding: utf-8

# # Overview
# All programmer know the basic cycle of `git`: `commit`, `push`, `pull`, merge, and `push` again.  The typical belief is that developers should make many small commits, rather than the occasionally massive batch.  However, all though this is what programmers should be doing on projects, like many things, what ought to be is not necessarily the nature of things.  By analyzing open source project push events, we can get an idea of how commits are truly done.

# In[ ]:


import matplotlib as mpl
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from __future__ import division


# In[ ]:


pd.set_option('display.float_format', lambda x: '%.3f' % x) #Remove scientific notation for df.describe()
commits_df = pd.read_csv('../input/LinuxCommitData.csv',
                         header=0,
                         index_col=None,
                         na_values='',
                         parse_dates=['Date'],
                         usecols=range(0,5),
                         names=['Date', 'Commits', 'Additions', 'Deletions', 'UserID'])


# ## Contributing to the Linux Kernel
# This data set contains commits to the Linux kernel project on GitHub from April 7th, 2002 - March 3rd, 2017, spread out over 636 individual days.  On any one of those 636 days, there was an average of 35 unique contributors to the Linux kernel project. March 22nd, 2015 had the largest number of unique users (59 total) contributing on a given day.

# In[ ]:


print(commits_df['Date'].describe())
print('='*10)
print('Average # of unique users contributing on active days: {0:.4f}'.format(22372/636))


# We see that there were 139,566 individual commits made over the nearly 15 year existence of the GitHub project.  Those commits resulted in a total of 18,784,298 alterations to the Linux code base (with 9,778,055 new lines of code, and the removal of 9,006,243 lines).
# 
# ### The Nature of Development (or GitHub's `diff` tools)
# Looking at the net changes to the Linux kernel's code base (new lines - deleted lines), we see that 771,812 lines of code have been added to the Linux kernel as of March 3rd 2017.  This is a far cry from the approx. 9.8 million lines that were added over time.  It simply demonstrates how software development, both for purely commercial and free open source software, is largely defined by deleting, replacing, and refactoring of yours and others code.  It is a testament to the impact of code reviews.

# In[ ]:


user_total_df = commits_df.groupby('UserID').sum()
user_total_df['Net Changes'] = user_total_df['Additions'].subtract(user_total_df['Deletions'])
user_total_df['Total Changes'] = user_total_df['Additions'].add(user_total_df['Deletions'])
user_total_df.sort_values(['Commits', 'Total Changes'],
                          ascending=False,
                          inplace=True)
print('Distribution of Total User Commit Behavior (2002-2017)')
user_total_df.describe()


# ### The Commit Distribution
# The below visualization illustrates the total number of commits made by each one of the 100 contributors.  The median number of commits is 1,061 with 75 of the users being within 600 commits.  The smallest number of total commits by a particular user is 845.75.  The visualization clearly indicates how it is skewed to the right with a few users having a large number of commits.  At the most, a single user made 4,903 commits, but this may simply indicate the occurrence of a `merge`.  This provides us with an insight into how actively developers contribute to the Linux kernel, but what does it say about their adherence to best `git` commit practices?  Nothing.  For that, we will need to look at the daily commit behavior for each user.

# In[ ]:


plt.figure(figsize=(14,2))
ax = sns.violinplot(x=user_total_df['Commits'],
                    inner='quartile',
                    saturation=0.5)
ax.tick_params(axis='both', which='major', labelsize=15)
ax.set_xlabel('# of Commits for Each One of 100 Contributors', fontsize=15)


# In[ ]:


fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1,
                                         figsize=(14,6),
                                         sharex=True)

sns.violinplot(x=user_total_df['Total Changes'],
               inner='quartile',
               saturation=0.5,
               ax=ax1)
ax1.set_title('Total Code Changes for Each One of 100 Contributors', fontsize=15)
ax1.set_xlabel('')

sns.violinplot(x=user_total_df['Additions'],
               inner='quartile',
               saturation=0.2,
               ax=ax2,
               color='g')
ax2.set_title('Total Code Additions', fontsize=15)
ax2.set_xlabel('')

sns.violinplot(x=user_total_df['Deletions'],
               inner='quartile',
               saturation=0.4,
               ax=ax3,
               color='r')
ax3.set_title('Total Code Deletions', fontsize=15)
ax3.tick_params(axis='both', which='major', labelsize=15)
ax3.set_xlabel('')

sns.violinplot(x=user_total_df['Net Changes'],
               inner='quartile',
               saturation=0.4,
               ax=ax4,
               color='y')
ax4.set_title('Net Code Changes (Addition-Deletion)', fontsize=15)
ax4.tick_params(axis='both', which='major', labelsize=15)
ax4.set_xlabel('')


# While the correct number of changes per commit is subjective, let us say that 30 total changes makes for a manageable amount for a developer of limited experience.  By this standard, we see that on about half of the 636 days of activity, contributors are adhering to best commit practices.  At that point, there are roughly 26 total line changes for each commit.  After the halfway point though, we start seeing larger and larger commits.  On 20% of days, contributors were pushing at least 53 edits/commit.  On 10%, at least 82 edits/commit.  In the worst case, there was a day when a user made approx. 2,422 edits/commit (clearly an amount which would violate the best practices).  However, it should be noted that this may reflect a `merge`, and thus justify the violation.

# In[ ]:


date_users_df = commits_df.groupby(['Date', 'UserID']).sum()
date_users_df['Daily Net Changes'] = date_users_df['Additions'].subtract(date_users_df['Deletions'])
date_users_df['Daily Total Changes'] = date_users_df['Additions'].add(date_users_df['Deletions'])
date_users_df.sort_values('Commits',
                          ascending=False,
                          inplace=True)
print('Distribution of Daily User Behavior')
date_users_df.describe(percentiles=[.25, .5, .6, .65, .75, .8, .9])


# In[ ]:


plt.figure(figsize=(14, 6))
ax1 = sns.distplot(date_users_df['Commits'])
ax1.tick_params(axis='both', which='major', labelsize=15)
ax1.set_title('Distribution of Daily Commits', fontsize=15)
ax1.set_xlabel('# of Commits', fontsize=15)

