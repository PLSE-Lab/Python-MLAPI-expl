#!/usr/bin/env python
# coding: utf-8

# I subscribe to various kinds of websites and podcast channels related to data science and machine learning, and check them almost everyday.
# In this kernel, I'll explore the 2018 Kaggle ML & DS Survey Challenge dataset, specifically Q38:
# ## Q38: Who/what are your favorite media sources that report on data science topics?

# In[ ]:


import os
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, Markdown

sns.set()
warnings.filterwarnings('ignore')


# In[ ]:


ffr_df = pd.read_csv('../input/freeFormResponses.csv')
q38 = ffr_df[ffr_df['Q38_OTHER_TEXT'].notnull()]['Q38_OTHER_TEXT'].iloc[1:]
print(f'Number of non-null answers: {q38.shape[0]}')


# In[ ]:


print(*q38.tolist()[:100], sep='\n')


# In[ ]:


infos = [
    # name, matching pattern, url
    ('Data Elixir', r'data\s*elixir', 'https://dataelixir.com/'),
    ('Data Machina', r'data\s*machina', 'https://www.getrevue.co/profile/datamachina'),
    ('Analytics Vidhya', r'analytics\s*vid[h]*ya', 'https://www.analyticsvidhya.com/'),
    ('DataCamp', r'data\s*camp', 'https://www.datacamp.com'),
    ('Data Science Weekly', r'data\s*science\s*weekly', 'https://www.datascienceweekly.org/'),
    ('SuperDataScience', r'super\s*data\s*science', 'https://www.superdatascience.com/'),
    ('Data Science Central', r'data\s*science\s*central', 'https://www.datasciencecentral.com/'),
    ('This Week in Machine Learning and AI Podcast', r'twiml', 'https://twimlai.com/'),
    ('DataFramed', r'data\s*framed', 'https://www.datacamp.com/community/tags/dataframed'),
    ('Towards Data Science', r'towards\s*data', 'https://towardsdatascience.com/'),
    ('R Bloggers', r'r[\s-]*blogger', 'https://www.r-bloggers.com/'),
    ('ods.ai', r'ods.ai|open\s*data\s*science', 'https://opendatascience.slack.com'),
    ('Two Minute Papers', r'two\s*minutes*\s*paper', 'https://www.youtube.com/channel/UCbfYPyITQ-7l4upoX8nvctg'),
    ('Machine Learning Mastery', r'machine\s*learning\s*mastery', 'https://machinelearningmastery.com/'),
    ('sentdex', 'sentdex', 'https://www.youtube.com/user/sentdex'),
    ('Habrahabr', 'habr', 'https://habr.com/'),
    ('Medium', 'medium', 'https://medium.com/'),
    ('GitHub', 'github', 'https://github.com/'),
    ('LinkedIn', 'linkedin', 'https://www.linkedin.com/'),
    ('Facebook', 'facebook', 'https://www.facebook.com/'),
    ('Stack Overflow', 'stack\s*overflow', 'https://stackoverflow.com/'),
]

# get the number of matches
matches = [q38[q38.str.lower().str.contains(pat)].shape[0] for _, pat, _ in infos]

# sort the results
infos = [x for _, x in sorted(zip(matches, infos), reverse=True)]
names = [name for name, _, _ in infos]
matches = sorted(matches, reverse=True)

# display the results in markdown
md_str = '''
|Name|Matches|URL|
|-|-|-|
'''

for match, (name, _, url) in zip(matches, infos):
    row = f'|{name}|{match}|{url}|\n'
    md_str += row

display(Markdown(md_str))


# In[ ]:


x = list(reversed(range(len(matches))))
plt.figure(figsize=(8, 10))
plt.barh(x, matches)
plt.yticks(x, names, fontsize=12)
plt.xlabel('Number of Matches')
plt.ylabel('Media Source');


# We find:
# - ods.ai and Analytics Vidhya dominate
# - People use LinkedIn a lot
# - Some youtube and podcast channels rank in

# In[ ]:




