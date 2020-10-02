#!/usr/bin/env python
# coding: utf-8

# We are going to be using The History of Baseball dataset for our analysis. One of the suggestions for analysis is, "Is there a most error-prone position?" We're going to try and answer that question. While there may be other kernels out there that have already tackled this problem, I have not looked for them.  I am still learning, and I want this analysis to be entirely my own.

# In[ ]:


# First we need to import our relevant libraries

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


# We will be working with the fielding.csv file, so let's read it to a dataframe using Pandas
fielding = pd.read_csv('../input/fielding.csv')


# In[ ]:


# Check the head of the dataframe
fielding.head()


# In[ ]:


# Get information regarding the makeup of the observations
fielding.info()


# In[ ]:


# In order to cleanup the data, we can drop all observations with the position DH, 
# since the DH only bats.
fielding = fielding[fielding['pos']!='DH']


# In[ ]:


# We can check the size of the updated dataframe.
# We still have 18 columns, but we only have 165,477 rows
fielding.shape


# In[ ]:


# The first thing we can do is compare the total number of errors for each position
errors = fielding.groupby(by='pos')['e'].sum()


# In[ ]:


# Let's view the results
sns.barplot(x=errors.sort_values().index, y=errors.sort_values())
plt.title('Number of Errors by Position')
plt.xlabel('Position')
plt.ylabel('Total Errors')


# In[ ]:


# We can see from the plot that the SS position has the most errors overall. But does
# that mean that SS is more error prone? Short-stop is a very important position, so
# maybe they are involved in more plays. We also see that CF, RF, and LF have far fewer
# errors than the other positions. That's because prior to 1954, they were all recorded
# as OF. If we add all the outfield positions together, let's see what we get.
errors['OF'] = errors['CF'] + errors['RF'] + errors['LF'] + errors['OF']


# In[ ]:


#Now let's remove the individual outfield positions.
errors.drop(['CF', 'RF', 'LF'], axis=0, inplace=True)


# In[ ]:


# Let's view the updated results.
sns.barplot(x=errors.sort_values().index, y=errors.sort_values())
plt.title('Number of Errors by Position')
plt.xlabel('Position')
plt.ylabel('Total Errors')


# In[ ]:


# So now it looks like outfielders may have made more errors than shortstops. But perhaps there
# is more to it than total errors. Certain positions may handle the ball more, so we'd expect more
# errors. To account for this, we'll now compare errors per putout for each position.
# First we'll group the dataframe by position, then select the error and putouts columns,
# then we'll sum them.
errors_po = fielding.groupby(by='pos')[['e', 'po']].sum()


# In[ ]:


# Let's check the head to make sure we got what we want.
errors_po.head()


# In[ ]:


# Since we already know about the outfield problem, let's go ahead and sum the results
# of the individual outfield positions and add them to the OF line.
errors_po.loc['OF']['e'] = errors_po.loc['OF']['e'] + errors_po.loc['LF']['e'] + errors_po.loc['CF']['e'] + errors_po.loc['RF']['e']
errors_po.loc['OF']['po'] = errors_po.loc['OF']['po'] + errors_po.loc['LF']['po'] + errors_po.loc['CF']['po'] + errors_po.loc['RF']['po']


# In[ ]:


#Now let's remove the individual outfield positions.
errors_po.drop(['CF', 'RF', 'LF'], axis=0, inplace=True)


# In[ ]:


# We have errors and putouts, but we need to create the column error/putout.
# We can define a function that takes in a dataframe and returns the errors column
# divided by the putouts column.
def e_per_po(df):
    return df['e'] / df['po']


# In[ ]:


# Now we apply our new function to the dataframe.
errors_po['e/po'] = errors_po.apply(e_per_po, axis=1)


# In[ ]:


# Check the head to see if calculated correctly
errors_po.head()


# In[ ]:


# We can once again plot our results, but this time with our new e/po column.
sns.barplot(x=errors_po['e/po'].sort_values().index, y=errors_po['e/po'].sort_values())
plt.title('Number of Errors per Putout by Position')
plt.xlabel('Position')
plt.ylabel('Average Errors per Putout')


# In[ ]:


# Here we see a different picture. Relative to the number of putouts, outfielders
# commit amongst the fewest errors, with third basemen and pitchers leading the way.
# However, we might once again be overlooking something. Pitchers are also responsible
# for strikeouts. Perhaps we should add strikeouts and putouts to determine the total
# number of outs produced. We'll need the pitching table
pitching = pd.read_csv('../input/pitching.csv')


# In[ ]:


pitching.head()


# In[ ]:


pitching.info()


# In[ ]:


so = pitching['so'].sum()


# In[ ]:


errors_po.loc['P']


# In[ ]:


errors_to = errors_po.copy()


# In[ ]:


errors_to['so'] = 0


# In[ ]:


errors_to


# In[ ]:


errors_to.reset_index(inplace=True)


# In[ ]:


errors_to


# In[ ]:


def set_so(row):
    if row == 'P':
        return so
    else:
        return 0


# In[ ]:


errors_to['so'] = errors_to['pos'].apply(set_so)


# In[ ]:


errors_to


# In[ ]:


def set_to(df):
    return df['so'] + df['po']


# In[ ]:


errors_to['to'] = errors_to.apply(set_to, axis=1)


# In[ ]:


errors_to


# In[ ]:


# We have errors and putouts, but we need to create the column error/putout.
# We can define a function that takes in a dataframe and returns the errors column
# divided by the putouts column.
def e_per_to(df):
    return df['e'] / df['to']


# In[ ]:


# Now we apply our new function to the dataframe.
errors_to['e/to'] = errors_to.apply(e_per_to, axis=1)


# In[ ]:


# We can once again plot our results, but this time with our new e/po column.
errors_to.sort_values(by='e/to', inplace=True)
sns.barplot(x=errors_to['pos'], y=errors_to['e/to'])
plt.title('Number of Errors per Out by Position')
plt.xlabel('Position')
plt.ylabel('Average Errors per Out')


# In[ ]:


errors_to['pos'].sort_values(by='e/to')


# In[ ]:




