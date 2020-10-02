#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from numpy import mean


# In[ ]:


data = pd.read_csv('../input/golden-globe-awards/golden_globe_awards.csv')
data.head()


# Lets start by looking at the 10 most nominated actors...

# In[ ]:


plt.figure(figsize=(15,10))
df = data['nominee'].value_counts().nlargest(10)
sb.barplot(x=df.index, y=df.values)
plt.xticks(rotation=90, fontsize=12)
plt.ylabel("# of Nominations", fontsize=15)
plt.yticks(fontsize=16)
plt.title("10 Most Nominated")


# Meryl Streep leads the number of nominations by a huge margin. It's cool to see all the legend in this list.
# 
# Lets see how many of these people are on the list of top 10 winners in Golden Globes history...

# In[ ]:


plt.figure(figsize=(15,10))
df = data.loc[data['win']==True]
df = df['nominee'].value_counts().nlargest(10)
sb.barplot(x=df.index, y=df.values)
plt.xticks(rotation=90, fontsize=12)
plt.ylabel("# of Wins", fontsize=15)
plt.yticks(fontsize=16)
plt.title("10 Most Wins")


# Meryl Streep leads this list as well, followed by Jane Fonda, who interestingly didn't make the list of top 10 nominations, pointing to a very high win percentage...
# 
#  What is that percentage?

# In[ ]:


df = data.loc[data['nominee']=='Jane Fonda']
df['win'].value_counts().plot.pie(autopct='%.1f%%', label="Jane Fonda Wins vs Nominations")


# Lets look at the Meryl Streep movies nominated in the Golden Globes and see which of her movies won an award...

# In[ ]:


plt.figure(figsize=(18,10))
df = data.loc[data['nominee']=="Meryl Streep"]
sb.countplot(x='film', hue='win', data=df)
plt.xticks(rotation=90, fontsize=12)
plt.xlabel("Film", fontsize=15)
plt.ylabel("Count", fontsize=15)
plt.yticks(fontsize=16)
plt.title("Meryl Streep Movies", fontsize=15)


# It'd be interesting to see how the number of nominations for The Golden Globes changed over the history of the event...

# In[ ]:


plt.figure(figsize=(15,10))
df = data['year_award'].value_counts()
sb.lineplot(x=df.index, y=df.values)
plt.xticks(fontsize=16)
plt.xlabel("Year", fontsize=15)
plt.ylabel("Nominations", fontsize=15)
plt.yticks(fontsize=16)


# Lets also look at how the number of Wins in relation to this...

# In[ ]:


plt.figure(figsize=(15,10))
df2 = data.loc[data['win']==True]
df2 = df2['year_award'].value_counts()
sb.lineplot(x=df.index, y=df.values, label="Nominations")
sb.lineplot(x=df2.index, y=df2.values, label="Wins")
plt.xticks(fontsize=16)
plt.xlabel("Year", fontsize=15)
plt.ylabel("Count", fontsize=15)
plt.yticks(fontsize=16)
plt.title("Nominations Vs Wins")


# Its interesting that in the last decade the number of wins has leveled off. The number of categories appeared to have reached a stable level.
# 
# Lets look at the most nominated shows/movies in Golden Globes history...

# In[ ]:


plt.figure(figsize=(15,10))
df = data['film'].value_counts().nlargest(10)
df = data[data['film'].isin(df.index)]
sb.countplot(x='film', hue='win', data=df)
plt.xticks(rotation=90, fontsize=16)
plt.xlabel("Film", fontsize=15)
plt.ylabel("Count", fontsize=15)
plt.yticks(fontsize=16)
plt.title("Most Nominated Shows/Movies")


# Will & Grace, the Cleveland Browns of The Golden Globes. The most Nominated show in Golden Globes history, yet they have never won a single time. 0-30. Interestingly, these are all tv shows, which makes sense, since they are on for multiple seasons, they have a better chance of getting nominated multiple times...
