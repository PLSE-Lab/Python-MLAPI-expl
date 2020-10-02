#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import matplotlib.pyplot as plt


# In[ ]:


lb_data = pd.read_csv("../input/leaderboard-4-days-out/jigsaw-unintended-bias-in-toxicity-classification-publicleaderboard.csv")


# In[ ]:


lb_data = lb_data.set_index("SubmissionDate")


# In[ ]:


#skimming off the top 15
top_15_teams = lb_data.groupby("TeamId").max().sort_values("Score")[-15:]["TeamName"].values


# In[ ]:


top_15_subs = lb_data.loc[lb_data["TeamName"].isin(top_15_teams)]


# In[ ]:


top_15_subs = top_15_subs.drop("TeamId", axis = 1)


# In[ ]:


top_15_subs.pivot(columns="TeamName", values="Score")


# In[ ]:


#looking at peoples top score over time. Interesting to see when people made their rise. Sometimes coincides with information going public in discussion
top_15_subs.pivot(columns="TeamName", values="Score").interpolate().plot(legend = True, ylim = (.93, .95), figsize = (12,12))


# In[ ]:


#viewing prints of peoples rising. Recent activity and the size of peoples jumps is interesting
for i in top_15_subs.pivot(columns="TeamName", values="Score").interpolate():
    print(top_15_subs.pivot(columns="TeamName", values="Score")[i].dropna())


# In[ ]:


#graphs showing individual teams trends. More easily readable than the above graph
for i in top_15_subs.pivot(columns="TeamName", values="Score").interpolate():
    top_15_subs.pivot(columns="TeamName", values="Score")[i].dropna().plot(legend = True, ylim = (.93, .95), figsize = (12,12), title = str(i))
    plt.show()


# In[ ]:


top_15_subs.index = pd.to_datetime(top_15_subs.index)


# In[ ]:


top_15_subs_last_7 = top_15_subs.loc[top_15_subs.index > '2019-6-15']


# In[ ]:


#looking at teams last 7 days worth of submissions. Some are active and yuval has no activity. 
for i in top_15_subs_last_7.pivot(columns="TeamName", values="Score").interpolate():
    top_15_subs_last_7.pivot(columns="TeamName", values="Score")[i].dropna().plot(legend = True, ylim = (.93, .95), figsize = (12,12), title = str(i))
    plt.show()


# In[ ]:




