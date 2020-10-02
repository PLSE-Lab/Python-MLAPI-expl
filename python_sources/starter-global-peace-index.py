#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd

import altair as alt
alt.themes.enable('default')
alt.renderers.enable('kaggle');


# In[ ]:


START_YEAR = 2008
END_YEAR = 2019
YEARS = range(START_YEAR, END_YEAR + 1)


# In[ ]:


gpi = pd.read_csv(f'../input/global-peace-index/gpi-{START_YEAR}-{END_YEAR}.csv')


# ## The most and the least peaceful countries

# In[ ]:


gpi.head()


# In[ ]:


gpi.tail()


# ## Total hostility score

# In[ ]:


score_columns = [f'{year} score' for year in YEARS]

total_score = pd.DataFrame({
    'year': list(map(str, YEARS)),
    'hostility': gpi[score_columns].sum()
})

alt.Chart(total_score).mark_line().encode(
    x='year',
    y=alt.Y('hostility', scale=alt.Scale(zero=False))
)


# ## Make DS, not war!
