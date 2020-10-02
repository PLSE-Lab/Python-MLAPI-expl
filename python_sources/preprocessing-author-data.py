#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import numpy as np
import pandas as pd
from itertools import cycle, chain
aminer_dir = '../input/'


# # Process Author Data
# Here we read in the author table as described here
# ```
# #index ---- index id of this author
# #n ---- name  (separated by semicolons)
# #a ---- affiliations  (separated by semicolons)
# #pc ---- the count of published papers of this author
# #cn ---- the total number of citations of this author
# #hi ---- the H-index of this author
# #pi ---- the P-index with equal A-index of this author
# #upi ---- the P-index with unequal A-index of this author
# #t ---- research interests of this author  (separated by semicolons)
# ```

# In[ ]:


with open(os.path.join(aminer_dir, 'AMiner-Author.txt'), 'r') as f:
    dict_list = []
    c_dict = {}
    for i, line in enumerate(f):
        c_line = line.strip()[1:].strip()
        if len(c_line)<1:
            if len(c_dict)>0:
                dict_list += [c_dict]
            c_dict = {}
        else:
            c_frag = c_line.split(' ')
            c_dict[c_frag[0]] = ' '.join(c_frag[1:])


# In[ ]:


author_df = pd.DataFrame(dict_list)
author_df.rename({'a': 'Affiliation',
                 'n': 'Author', 
                 'pc': 'Papers',
                 'cn': 'Citations',
                  'hi': 'H-index',
                  't': 'research interests'
                 }, axis=1, inplace=True)
author_df.to_csv('author_combined.csv')
author_df.sample(3)


# In[ ]:


print(author_df.shape[0], 'authors')
zrh_match = author_df['Affiliation'].map(lambda x: 'ZURICH' in x.upper())
uzh_match = author_df['Affiliation'].map(lambda x: 'UNIVERSITY OF ZURICH' in x.upper())
print(zrh_match.sum(), 'in zurich')
print(uzh_match.sum(), 'at uzh')


# In[ ]:


author_df[uzh_match].sample(3)


# In[ ]:


major_keywords = author_df[uzh_match]['research interests'].    map(lambda x: x.split(';')).    values.tolist()
major_keywords = pd.DataFrame({'keyword': list(chain(*major_keywords))})
major_keywords.    groupby('keyword').    size().    reset_index(name='count').    sort_values('count', ascending=False).    head(12).    plot.bar(x='keyword', y='count')


# In[ ]:


author_df[uzh_match].sample(5)['Affiliation'].values


# In[ ]:


author_df[uzh_match].head(10)


# In[ ]:


author_df[zrh_match & author_df['Author'].map(lambda x: 'KEVIN' in x.upper())]


# In[ ]:




