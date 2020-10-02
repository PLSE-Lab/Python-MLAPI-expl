#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pylab as plt
import sqlite3


# In[ ]:


# Any results you write to the current directory are saved as output.
con = sqlite3.connect('../input/database.sqlite')
cursor = con.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
print(cursor.fetchall())


# In[ ]:


db = sqlite3.connect('../input/database.sqlite')
#table = pd.read_sql_query("SELECT is_screener FROM patients_train WHERE patient_state = 'TX';",db)
#print(table.sum()/table.shape[0])

sdf = pd.read_sql_query("SELECT patient_state, is_screener FROM patients_train;",db)
pdf = sdf.groupby('patient_state').apply(lambda x: x.is_screener.sum()/x.shape[0])
pdf.sort_values(inplace = True,ascending = False)
print(pdf)


# In[ ]:


pdf.plot()


# In[ ]:



cm = plt.get_cmap('RdBu')
colors = [cm(x) for x in pdf.values]
pdf.plot(kind='barh', figsize=(18, 14), grid=False, color=colors)
plt.title('Percentage of screeners by state')
plt.ylabel('State')
plt.show()


# In[ ]:


pos = 1.5*np.arange(len(colors)) + 1.5
fig = plt.figure(figsize=(18,14))
ax = fig.add_subplot(111, axisbg='w', frame_on=False)
fig.suptitle('Percentage of screeners by state', fontsize=20)
plt.barh(pos,pdf.values, height = 1.2, align = 'center', color = colors, tick_label = pdf.index)
plt.show()


# In[ ]:




