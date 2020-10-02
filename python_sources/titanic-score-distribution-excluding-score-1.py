#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
pd.set_option('display.max_rows', 110) # Set max rows to display to 110
import plotly.express as px # Data Visualisation

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# ## Background
# 
# A suspiciously high number of participants in the Titanic Machine Learning return an accuracy score of 100%.  It is highly likely that these perfect scores are achieved through use of publicly available data listing the names of all those who survived and perished.  This makes a mockery of the leaderboard, and frustrates those keen to assess their performance against others who have participated honestly.  
# 
# To provide some insight into the performance of those adhering to the competition rules, this notebook calculates score percentiles, whilst excluding all participants with an accuracy of 1.  
# 
# ## Data
# 
# The public leaderboard was downloaded and saved as an Excel file.  This download covered all submissions made since the competition was introduced in 2013.  The file was then imported using Pandas.

# In[ ]:


# Import public leaderboard (date of download 24/12/2019)
df = pd.read_excel('/kaggle/input/titanic-publicleaderboard.xlsx')
df.head()


# ## Drop Score=1
# 
# All submissions that achieved a score of 1 were dropped from the dataset. 

# In[ ]:


# Drop all records for which Score=1
df = df.loc[df['Score']!=1]
df.head()


# ## Calculate Percentiles
# 
# Next, the score percentiles were calculated.

# In[ ]:


# Calculate score percentiles (1% intervals)
percentiles = np.percentile(df['Score'], np.arange(0, 100, 1)) 

# Create labels for each percentile (1% intervals)
labels = np.arange(0,100,1)


# Finally, the percentiles were put into a DataFrame to enable easy review.

# In[ ]:


# Put percentiles and percentile labels into dataframe
percentiles_df = pd.DataFrame({'Percentile(%)':labels, 'Percentile_Score':percentiles})

# Display percentile DataFrame
percentiles_df


# From the above, it appears that a score above 0.784680 places a participant in the top 20%.  Meanwhile, a score above 0.832530 in the leading 1%.  Participants adhering to competition rules can therefore be reassured that a good score is likely significantly lower than the public leaderboard might suggest.   
# 
# To complete this post, the percentiles were visualised using Plotly.

# In[ ]:


# Plot percentiles on line chart
fig = px.line(percentiles_df, x="Percentile(%)", y="Percentile_Score", title='Distribution of Scores: Titanic ML Competition')
fig.show()

