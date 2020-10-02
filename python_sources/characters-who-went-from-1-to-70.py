#!/usr/bin/env python
# coding: utf-8

# ## How many characters went from level 1 to 70 in 2008?

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

wowah_df = pd.read_csv('../input/wowah_data.csv')


char_and_level = wowah_df[['char', ' level']]

chars1 = char_and_level[char_and_level[' level'] == 1].set_index('char').to_dict()[' level']

chars70 = char_and_level[(char_and_level[' level'] == 70)].char.tolist()

test_group = [c for c in set(chars70) if c in chars1]

print(len(test_group))

