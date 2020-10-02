#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


import sqlite3, pandas as pd, numpy as np


# In[ ]:


# Look at earnings data overall

# Read data from the SQLite datbase
con = sqlite3.connect('../input/database.sqlite')
earnings = pd.read_sql_query("""
SELECT opeid, count_wne_p10, count_nwne_p10, mn_earn_wne_p10, md_earn_wne_p10, pct10_earn_wne_p10, pct25_earn_wne_p10, pct75_earn_wne_p10, pct90_earn_wne_p10
FROM Scorecard
WHERE count_wne_p10 IS NOT NULL""", con)

earnings.head()


# In[ ]:


for col in earnings.columns:
    earnings.loc[earnings[col] == 'PrivacySuppressed', col] = None


# In[ ]:


np.sum(earnings.mn_earn_wne_p10.isnull())


# In[ ]:


earnings.describe()


# In[ ]:


from matplotlib import pyplot as plt
import seaborn
seaborn.set()


# In[ ]:


earnings.count_nwne_p10.hist();


# In[ ]:


earnings.shape


# In[ ]:


plt.plot(earnings.count_wne_p10, earnings.md_earn_wne_p10, 'o');

