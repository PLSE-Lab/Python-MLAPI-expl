#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import bar_chart_race as bcr


# In[ ]:


get_ipython().system('pip install bar-chart-race')


# In[ ]:





# In[ ]:


df= bcr.load_dataset("covid19_tutorial")


# In[ ]:


df


# In[ ]:


bcr.bar_chart_race(df, orientation="v")


# In[ ]:


bcr.bar_chart_race(df, orientation="v", sort="asc")


# In[ ]:


bcr.bar_chart_race(df, orientation="v", sort="asc", steps_per_period=20, period_length=200
                   )


# In[ ]:


bcr.bar_chart_race(df, orientation="v", sort="asc", interpolate_period=True)


# In[ ]:


bcr.bar_chart_race(df, orientation="v", sort="asc", figsize=(5,3), title="Covid19 Deaths by Country", interpolate_period=True)

