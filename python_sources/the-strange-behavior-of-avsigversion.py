#!/usr/bin/env python
# coding: utf-8

# In this kernel we will investigate AvSigVersion. 
# Most likely the top culprit for the differences between leaderboard and CV scores.
# Here we will look at the first four digits of AvSigVersion, and see what that gets us.
# 
# In this Kernel we will quickly do some data digestion and wrangling, plot the key results then discuss.

# # Data input

# In[ ]:


import os
import pandas as pd


# In[ ]:


train_columns = ['AvSigVersion', 'HasDetections']
test_columns = ['AvSigVersion']

train_path = '../input/train.csv'  # path to training file
test_path = '../input/test.csv'  # path to testing file

# Note: We are only keeping the columns we actually need.
train = pd.read_csv(train_path, usecols=train_columns)
test = pd.read_csv(test_path, usecols=test_columns)


# # Combining data
# 
# There is probably a cleaner way to do this, I just did something quick and dirty

# In[ ]:


train['TopVersion']=train['AvSigVersion'].apply(lambda x:  x.replace('.', '')[0:4] )

avsig_train = train[['TopVersion', 'HasDetections']].groupby('TopVersion').agg(['mean', 'count'])
avsig_train.columns = avsig_train.columns.droplevel()
avsig_train.rename({'count':'train count', 'mean':'HasDetections mean' }, axis='columns', inplace=True)

test['TopVersion']=test['AvSigVersion'].apply(lambda x:  x.replace('.', '')[0:4]  )

avsig_test = test['TopVersion'].value_counts()
avsig_test.name='Submission file count'

combined = pd.merge(avsig_train, pd.DataFrame(avsig_test), how='outer', left_index=True, right_index=True )
combined = combined.reset_index().rename({'index':'major_version'}, axis='columns')

print(combined.head(5))


# Cleanup...

# In[ ]:


combined.query("not major_version== '12&#' " , inplace=True)
combined['major_version'] = pd.to_numeric( combined['major_version'] )
combined = combined.query("major_version>1221")
combined[['train count', 'Submission file count']] = combined[['train count', 'Submission file count']].fillna(0)


# # Create the plot
# Ok, let's start plotting

# In[ ]:


import plotly.plotly 
import plotly.graph_objs as go

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)


# In[ ]:


scatter1 = go.Scatter(y=combined['train count'], x=combined['major_version'],  
                      name='Train count', 
#                      mode='markers',
                      yaxis='y1')
scatter2 = go.Scatter(y=combined['Submission file count'], x=combined['major_version'], 
                      name='Test count')
scatter3 = go.Scatter(y=combined['HasDetections mean'], x=combined['major_version'], 
                      mode='markers',
                      yaxis='y2',
                      name='Mean HasDetections')

plotly_data = [scatter1 , scatter2, scatter3]

layout= go.Layout(
        title="AvSigVersion counts and mean 'HasDetections'",
        yaxis2=dict(
                title='HasDetections mean',
                side='right',
                overlaying='y',
                ),
        xaxis=dict(title='Major Version number (first four digits)'),
        yaxis=dict(title='Number of observations (train and test)')
            )

figure = go.Figure(plotly_data, layout)

iplot(figure)


# # Discussion
# 
# 

# We note that we have no observations for the two most recent top versions. They are only included in the test set. But they are also strongly represented in the test set, with approximately 2.4 million rows.
# 
# Mean HasDetections is clearly high for version 1273 and 1275, and lower for 1265 through 1271.
# This is really interesting as these versions represents most of the observations in the train set.
# 
# With significant differences between mean values for HasDetections for very large parts of the dataset, AvSigVersion is clearly important for modelling.
# 
# The trend here is surprising. I would expect newer version of AntiVirus to protect better against virus, not worse. Or maybe it is an interaction effect - newer antivirus is better at detecting virus? Also the trend is clearly non-linear, so it is really hard to interpolate.
# 
# In conclusion I think the most interesting hypothesis is an interaction effect where newer AntiVirus is both better at protecting against malware and detecting malware.
# 
# 
