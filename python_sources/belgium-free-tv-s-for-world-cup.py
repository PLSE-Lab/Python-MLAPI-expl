#!/usr/bin/env python
# coding: utf-8

# # Belgium Free TV's for World Cup
# 
# As you might have read in the news, a Belgian electronics chain offered a promotion at the start of the FIFA World Cup:
# Every television that was bought in the months prior to the World Cup (1) would be reimbursed if the Belgian team would score more than 15 goals during the cup.
# 
# The Belgian team obtained a third place and scored... 16 goals. A lot of media picked this story up and published articles about it, like for example:
# http://dailyhive.com/vancouver/world-cup-report-belgium-free-tv-2018
# 
# The chain called 'Krefel', now claimed they were well prepared for this to happen. As I was wondering how likely this was, I webscraped the Fifa website for yearly statistics per country since FIFA started recording stats (1934). 
# 
# Although it's very short kernel, I thought it would be intresting to share/
# 

# In[ ]:


import pandas as pd

df = pd.read_csv('../input/world_cup_goals.csv')

print(df.shape)
print(df.head(5))


# In[ ]:


print(df['Country'].unique())


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df['WC'] = df['WC'].apply(lambda x: x[-4:])

x_small = df[df['Goals'] < 16]['WC']
y_small = df[df['Goals'] < 16]['Country']

x_big = df[df['Goals'] > 15]['WC']
y_big = df[df['Goals'] > 15]['Country']

fig, axes = plt.subplots(figsize=(10,30))


axes.scatter(x_small, y_small, color="red", marker='x')
axes.scatter(x_big, y_big, color="green")

plt.xticks(rotation=70)
plt.gca().invert_yaxis()
axes.legend(['15 or less', 'More than 15'])


# In[ ]:


df[df['Goals'] > 15].shape[0]


# In[ ]:


df['Goals'].count()


# In[ ]:


df[df['Goals'] > 15].shape[0] / df['Goals'].count()


# # Conclusion
# 
# Turns out, in the 395 participations (excluding this World Cup) only 15 times a team accomplished this feat, which translates to 3,80 %.
# It was very unlikely Krefel anticipated this result.
