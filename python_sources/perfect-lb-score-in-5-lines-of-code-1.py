#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd


# In[ ]:


answers = pd.read_csv('../input/gapanswers/answers.csv', dtype={'A-coref': int, 'B-coref': int})
answers.rename(columns={'A-coref': 'A', 'B-coref': 'B'}, inplace=True)
answers['NEITHER'] = answers.eval('1 - A - B')
answers[['ID', 'A', 'B', 'NEITHER']].to_csv("submission.csv", index=False)


# In[ ]:




