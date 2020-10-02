#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import re

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import numpy as np
import seaborn as sns
import os
os.chdir('../input')
sns.set_style("darkgrid")


# In[ ]:


with open('us_canada.xml') as f:
    f = f.read()
    these_regex="<title.*?>(.+?)</title>"
    pattern=re.compile(these_regex)
    titles=re.findall(pattern,f)


# In[ ]:


with open('us_canada.xml') as f:
    f = f.read()
    these_regex="<pubDate.*?>(.+?)</pubDate>"
    pattern=re.compile(these_regex)
    pubdates=re.findall(pattern,f)


# In[ ]:


pubdates


# In[ ]:


print(len(pubdates))


# In[ ]:


pubdates = [x[5:16] for x in pubdates]


# In[ ]:


title_date_dict = {}
for i in range(len(titles)):
    title_date_dict[titles[i]] = pubdates[i]
    
len(title_date_dict)


# In[ ]:


pubdate_title_pairs = list(zip(title_date_dict.values(), title_date_dict.keys()))


# In[ ]:


pubdate_title_pairs[:5]


# In[ ]:


date_china_dict = {}
for elem in pubdate_title_pairs:
    if "China" in elem[1] or "Chinese" in elem[1]:
        date_china_dict[elem[0]] = date_china_dict.get(elem[0], 0) + 1
date_china_dict


# In[ ]:


sorted(date_china_dict)
date_china_list = []
for key in sorted(date_china_dict):
    date_china_list.append([key, date_china_dict[key]])
    
date_china_list


# In[ ]:


fig = plt.figure()
ax = plt.axes()
fig.suptitle('Mentions of "China" or "Chinese" in Headlines of US-and-Canada Articles from the SCMP', fontsize=20)
fig.set_size_inches(16, 9)

x = np.linspace(0, 10, 1000)
x_axis = [x[0] for x in date_china_list]
y_axis = [x[1] for x in date_china_list]
ax.plot(x_axis, y_axis, linestyle='-', marker='o', color='b')


# In[ ]:


for pair in pubdate_title_pairs:
    if pair[0] == '24 Jul 2019':
        if 'China' in pair[1] or 'Chinese' in pair[1]:
            print(pair[1])


# In[ ]:




