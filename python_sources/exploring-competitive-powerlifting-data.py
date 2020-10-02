#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
plt_kwargs = {'figsize': (10, 6)}


# In[6]:


meets = pd.read_csv("../input/meets.csv", index_col=0)
meets.head()


# In[11]:


meets['Federation'].value_counts().head(10).plot.bar(
    title='Top 10 Meet-Organizing Federations', **plt_kwargs
)


# Competitive powerlifting events happen under the auspices of specific federations. As with any aggregate dataset like this one, there are likely to be biases. In this case, some federations do a much better job exposing their data than others. `NSF` is a Norwegian powerlifting association, and its dominance in the list of included meets indicates this pretty clearly.
# 
# Unless you're an industry insider these names are otherwise probably pretty uninformative. `USAPL` stands for "USA Powerlifters". `NASA` is the "Natural Athlete Strength Association" (and I kind of doubt their choice of acronym is accidental...). For a better reference on what the names mean, see [the meetlist](http://www.openpowerlifting.org/meetlist.html).

# In[21]:


(pd.to_datetime(meets['Date'])
     .to_frame()
     .assign(n=0)
     .set_index('Date')
     .resample('AS')
     .count()
     .plot.line(title='Meets Included in the Dataset by Year', **plt_kwargs))


# Interestingly enough the dataset has some meets going back pretty far.

# In[25]:


meets['MeetCountry'].value_counts().head(10).plot.bar(**plt_kwargs, title="Meets by Country")


# This dataset is *extremely* heavily biased towards English-speaking countries in general, the USA specifically, and, weirdly, Norway (I wonder if the person running this website is Norwegian...?).

# In[29]:


meets['MeetState'].value_counts().head(10).plot.bar(**plt_kwargs, title="Powerlifting Meet State-by-State Representation")


# This is going to mostly follow population curves. I would love to believe that red-bellied Texas enjoys its powerlifting _way more_ than blue-bellied California, even though California has way more people total, but given the biases in this dataset, I'm pretty skeptical that holds water.

# In[31]:


len(" ".join(meets['MeetName'].values))


# What words are organizers using to describe their powerlifting meets?

# In[34]:


from wordcloud import WordCloud
wordcloud = WordCloud().generate(" ".join(meets['MeetName'].values))
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, figsize=(12, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")


# I enjoy "Raw", "Pro", "Iron", and "Bash". Note how this result is littered with Norwegian words I don't know. =)
# 
# That about does it for the meets. The real "meat" of the dataset, however, is in the lifts, and the lifters performing them. That's in the `openpowerlifting.csv` file.

# In[35]:


competitors = pd.read_csv("../input/openpowerlifting.csv", index_col=0)
competitors.head()


# For one thing, what's the gender distribution like?

# In[37]:


competitors['Sex'].value_counts() / len(competitors)


# You know what? For one of the most testosterone-fueled sports out there, 80-20 is really not that bad. I was expecting worse, more like 90-10.

# In[42]:


competitors['Equipment'].value_counts().plot.bar(**plt_kwargs, title='Assistive Equipment Used')


# Powerlifters make a big deal these days about lifting "raw", that is, unassisted by hand wraps etcetera which makes it slightly easier to lift the weight (by keeping it aligned for you). This shows in the methods attempted.

# In[50]:


import seaborn as sns
fig, ax = plt.subplots(1, figsize=(10, 6))
sns.kdeplot(competitors['Age'])
plt.suptitle("Lifter Age Distribution")


# ...huh. Looks like a lot of powerlifting meet competitors are in their early 20s.

# In[79]:


competitors.query('Sex == "M"')['WeightClassKg'].str.replace("+", "").astype(float).dropna().value_counts().sort_index().plot.line(**plt_kwargs)
competitors.query('Sex == "F"')['WeightClassKg'].str.replace("+", "").astype(float).dropna().value_counts().sort_index().plot.line()
plt.suptitle("Male (Blue) and Female (Orange) Weight Classes")
plt.gca().set_xlabel("Weight Class (kg)")
plt.gca().set_ylabel("N")


# This is admitedly not a very well-designed graphic, but it gets the point across. For the Americans in the room, some of the male powerlifters are competiting in 250 pound weight class! These guys are *tanks*.
# 
# The maximum female weight class represented in ~200 pounds, while the minimum is a lithe 120 pounds or so. Small but mighty.
# 
# I'm going to pick my favorite exercise, the squat, and see how I place against powerlifting competitors. I'm 160 lbs and can do my bodyweight, 160 lbs,  as a one-rep max.. Wish me luck!

# In[109]:


(competitors
     .query('Sex == "M"')
     .loc[:, ['WeightClassKg', 'BestSquatKg']]
     .dropna()
     .pipe(lambda df: df.assign(WeightClassKg=df.WeightClassKg.map(lambda v: np.nan if "+" in v else np.nan if float(v) < 0 else v)))
     .dropna()
     .astype(float)
     .groupby("WeightClassKg")
     .agg([np.max, np.median])
     .plot.line(**plt_kwargs)
)
plt.gca().set_xlabel("Competitor Weight")
plt.gca().set_ylabel("Weight Lifted")
plt.suptitle("Male Powerlifting Competitor Median and Maximum Squats")
plt.plot(70, 70, 'go')


# As you can see I, uh, have a long way to go =).
# 
# That's all folks! Hopefully you enjoyed this short EDA kernel!
