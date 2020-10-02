#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import sqlite3 
import seaborn as sns
import math
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
con = sqlite3.connect('../input/database.sqlite')
data = pd.read_sql("select INSTNM, UNITID, cast(mn_earn_wne_p6 as int) mn_earn_wne_p6, cast(mn_earn_wne_p10 as int) mn_earn_wne_p10 from Scorecard where cast(mn_earn_wne_p6 as int)>0 and cast(mn_earn_wne_p10 as int)>0",con)


# In[ ]:


data['CAGR'] = np.power(data.mn_earn_wne_p10/data.mn_earn_wne_p6,1/4)
data.sort_values(by='CAGR').tail(30)[['INSTNM','CAGR']]


# Medicine/healthcare generally shows some most promising income growth path.
