#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import requests
from geopy import geocoders
from urllib.parse import urlparse 
import tqdm
import sys
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS 
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df = pd.read_csv('/kaggle/input/startup-investments-crunchbase/investments_VC.csv', encoding = 'unicode_escape')
pd.set_option('display.max_columns', None)
df = df.dropna()


# In[ ]:


df.head()


# ### How many startups are still active?

# In[ ]:


labels = "acquired", "operating", "closed"
sizes = [len(df[df["status"] == "acquired"])/len(df)*100, len(df[df["status"] == "operating"])/len(df)*100, len(df[df["status"] == "closed"])/len(df)*100]
explode = (0, 0, 0.1)  # only "explode" the closed status startups

fig1, ax1 = plt.subplots(figsize=(18,8))
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()


# Seems like 95% of all the startups in the dataset are either operating or acquired. Well thats not very ideal to make a predictive model, so.... I guess we just do the other cool thing: plot graphs :P

# ### Which countries are the startups mainly from?

# After i dropped rows containing NaN values, the only countries left behind are USA and CANADA, obviously.

# In[ ]:


labels = "USA", "Canada"
sizes = [len(df[df["country_code"] == "USA"])/len(df)*100, len(df[df["country_code"] == "CAN"])/len(df)*100]

fig1, ax1 = plt.subplots(figsize=(18,8))
ax1.pie(sizes,labels=labels, autopct='%1.1f%%', startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()


# ### Which markets contain the most number of startups?

# These are the top 10 markets based on the number of startups.
# > What I'm thinking:
# * Software came up on top which was predicatable.
# * I assumed ecommerce would come up higher because lifestyle business are huge now but i guess this data is a bit old.

# In[ ]:


sizes = { market : len(df[df[" market "] == market])/len(df)*100 for market in df[" market "].unique()}
labels = sorted(sizes, key=sizes.get, reverse=True)[:10]
sizes = [len(df[df[" market "] == market])/len(df)*100 for market in df[" market "].unique()]
sizes.sort(reverse=True)
sizes = sizes[:10]

fig1, ax1 = plt.subplots(figsize=(18,8))
ax1.pie(sizes,labels=labels, autopct='%1.1f%%', startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()


# ### Does more money equal more suceess?

# From my analysis, yes. The acquired startups had the highest amount of investments and the startups that closed had the least. 
# > What I'm thinking:
# * My analysis might be completely wrong.
# * But if it is not, I have a theory why. The startup culture in general is getting to a point where all they focus on is raising the next round of funding instead of servings its customers properly, mainly because the investors want them to keep spending their investments by getting multiple offices and huge team. I think its because even if the startup fails it would acquied by some other company since it has a lot of assets,therefore the investors win anyway. So the companies that get acquired usually have a huge amount investment.
# * **Thats just what i think... let me know what you think in the comments below.**

# In[ ]:


data = {'acquired':  df[df["status"] == "acquired"][" funding_total_usd "].str.strip().str.replace(',', '').str.replace('-', '0').astype(int).sum()/len( df[df["status"] == "acquired"]), 'operating': df[df["status"] == "operating"][" funding_total_usd "].str.strip().str.replace(',', '').str.replace('-', '0').astype(int).sum()/len( df[df["status"] == "operating"]), 'closed': df[df["status"] == "closed"][" funding_total_usd "].str.strip().str.replace(',', '').str.replace('-', '0').astype(int).sum()/len( df[df["status"] == "closed"])}
names = list(data.keys())
values = list(data.values())

fig1, ax1 = plt.subplots(figsize=(18,8))
ax1.bar(names, values)
# axs[0].bar(names, values)
# axs[2].plot(names, values)
fig1.suptitle('Categorical Plotting')


# #### Just some calculation to figure out the funding duration of each startup

# In[ ]:


df['first_funding_at'] =  pd.to_datetime(df['first_funding_at'], format='%Y-%m-%d', errors = 'coerce')
df['last_funding_at'] =  pd.to_datetime(df['last_funding_at'], format='%Y-%m-%d')
df['funding_period'] = df['last_funding_at'] - df['first_funding_at']
df[" funding_total_usd "] = df[" funding_total_usd "].str.strip().str.replace(',', '').str.replace('-', '0').astype(int)
df = df.fillna(0)
df["funding_period"] =  df["funding_period"].astype(int)
df[" funding_total_usd "] = df[" funding_total_usd "]/len(df[" funding_total_usd "])


# ##### Funding period calucluated in days.

# In[ ]:


df.head()


# ### Should a startup raise money fast or slow?

# It looks like startups that tried to raise money really fast ended up closing, but there are startups that were successfull even though they raised the money quickly. So i really cant conclude on anything. Let me know what you think about it down below.

# In[ ]:


df[df["status"] == "acquired"].plot.scatter(x='funding_period',y=' funding_total_usd ', figsize=(8,7), ylim=[0, 250000], xlim=[0.0, 1e18], title="Acquired startups")
df[df["status"] == "operating"].plot.scatter(x='funding_period',y=' funding_total_usd ', figsize=(8,7), ylim=[0, 250000], xlim=[0.0, 1e18], title="Operating startups")
df[df["status"] == "closed"].plot.scatter(x='funding_period',y=' funding_total_usd ', figsize=(8,7), ylim=[0, 250000], xlim=[0.0, 1e18], title="Closed startups")


# ### Startups that take grants, are they all sucessfull?
# 
# Definition: Startup grants consist of a sum of money that groups offer to small companies and nonprofits to help them with their work. Grants aren't like loans. Organizations don't have to put up collateral or pay late fees or interest. In fact, organizations don't have to pay back grants at all
# 
# >What I'm thinking: 
# * Well 92% of them are still operating and only 2.5% of them closed. So I think its safe to say they are mostly successfull.
# * Grants by their very own definition are usually given to highly promising startups so this was sort of predictable i guess.

# In[ ]:


grant_df = df[df['grant'] != 0]

labels = "acquired", "operating", "closed"
sizes = [len(grant_df[grant_df["status"] == "acquired"])/len(grant_df)*100, len(grant_df[grant_df["status"] == "operating"])/len(grant_df)*100, len(grant_df[grant_df["status"] == "closed"])/len(grant_df)*100]
explode = (0, 0, 0.1)  # only "explode" the closed status startups

fig1, ax1 = plt.subplots(figsize=(18,8))
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()


# * ### Biotechnology and Education tend me to be the most common markets that gets grants, which makes sense since they affect the society in a more impactful manner.

# In[ ]:


grant_cat = grant_df[' market ']
grant_cats = grant_cat.tolist()
unique_string=(" ").join(grant_cats)
wordcloud = WordCloud(width = 1000, height = 500).generate(unique_string)
plt.figure(figsize=(15,8))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()


# ### Cities with the largest numbers of startups

# In[ ]:


city_wise_spread = pd.DataFrame.from_dict({"city":[],"count":[], "acquired-count":[],"operating-count":[], "closed-count":[]})

for city, group in df.groupby("city"):
    data = [city, len(group), len(group[group["status"] == "acquired"]), len(group[group["status"] == "operating"]), len(group[group["status"] == "closed"])]
    city_wise_spread = city_wise_spread.append(pd.Series(data, index=["city","count", "acquired-count","operating-count", "closed-count"] ), ignore_index=True)

city_wise_spread.sort_values(by=['count'], ascending=False, inplace=True)


# In[ ]:


city_wise_spread.head()


# Obviously, San francisco ranks first but its intersting to see new york over palo alto.

# In[ ]:


sample = city_wise_spread[:10]
labels = sample["city"]
sizes = [value for value in sample["count"]]


fig1, ax1 = plt.subplots(figsize=(18,8))
ax1.pie(sizes,labels=labels, autopct='%1.1f%%', startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()


# **Thats all folks. Feel free to fork this kernel. Also if you like dit dont forget to upvote.**

# In[ ]:




