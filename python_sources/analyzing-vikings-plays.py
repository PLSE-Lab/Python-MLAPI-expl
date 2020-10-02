#!/usr/bin/env python
# coding: utf-8

# Being a Packers fan, I was disheartened by the Packers tough loss to the Vikings tonight, so I am going to try to analyze the Vikings tendencies.

# In[ ]:


import sys
import numpy as np # linear algebra
import pandas as pd
# Random Forests are good here because very small amount of data
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import train_test_split # split data
import seaborn as sns; # plotting
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df = pd.read_csv("../input/nflplaybyplay2015.csv", low_memory=False)
df.columns.values


# First, I want to analyze the Vikings offensive plays. I will grab every play that they are on offense.

# In[ ]:


mn_o = df[df.posteam == 'MIN'] 


# In[ ]:


mn_o.PlayType.unique()


# So the only plays we really care about are standard plays. From the list above I consider standard plays to be when the 'PlayType' is either a Run, Pass, or Sack.

# In[ ]:


valid_plays = ['Pass', 'Run', 'Sack']
mn_ov = mn_o[mn_o.PlayType.isin(valid_plays)] # mn_ov -> offensive plays considered in model


# Because 4th down is a constrained down, I am also going to remove 4th down plays from the model.

# In[ ]:


mn_ovn = mn_ov[mn_ov.down.isin([1,2,3])]
mn_ovn = mn_ovn[mn_ovn.TimeSecs>120] # Last two minutes too situational


# In[ ]:


mn_ovn.describe()


# Now that we have what I would consider a valid play set, we need to identify which variables we can actually use in the model. Because teams tend to game plan against their opponents, let's limit the plays to one's that are against the packers, for now.

# In[ ]:


mn_gb = mn_ovn[mn_ovn.DefensiveTeam == 'GB'] 


# In[ ]:


len(mn_gb)


# It is a pretty small sample size, but I am willing to bet that we can still get a decent prediction.

# In[ ]:


# create a column that has 1 for pass/sack, 0 for run
pass_plays = ['Pass', 'Sack']
mn_gb['is_pass'] = mn_gb['PlayType'].isin(pass_plays).astype('int')
mn_gb_pred = mn_gb[['down','yrdline100','ScoreDiff', 'PosTeamScore', 'DefTeamScore',
             'ydstogo','TimeSecs','ydsnet','Drive', 'yrdln','is_pass']]

# train/test split on data
X, test = train_test_split(mn_gb_pred, test_size = 0.2)
# pop the classifier off the sets.
y = X.pop('is_pass')
test_y = test.pop('is_pass')


# In[ ]:


# raise number of n_estimators so that it generates more data via bootstrapping
clf = RandomForestClassifier(n_estimators=100000)


# In[ ]:


clf.fit(X,y)


# In[ ]:


clf.score(test,test_y)


# In[ ]:


sns.barplot(x = clf.feature_importances_, y = X.columns)
sns.despine(left=True, bottom=True)


# So we can see that when it comes to playing the Packers, the time remaining in the game is critical in determining what type of play the Vikings will call. This test runs from about 68-80% accurate. One thing that I think would bring the accuracy up is analyzing the result of the previous play.

# In[ ]:




