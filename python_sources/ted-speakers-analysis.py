#!/usr/bin/env python
# coding: utf-8

# # A Targetted Analysis of Text
# 
# ## How to use Python to discover the most common openings to Ted speeches
# 
# Ted Talks are a multifaceted source of inspiration. I'm sure most everyone has come across one Talk or another whose content inspired them to think differently or live differently. I've been further impressed by the way that stand-out Ted Talk speakers can eloquently and captivatingly engage listeners. Maybe part of the reason is because I always have such difficulty starting my own speeches. First impressions count for so much, yet I have difficulty figuring out how to engage my audience right from the start. You should have seen how long it took me to write this paragraph.
# 
# While browsing the Kaggle Cache, one of my favorite weekly email inbox deliveries, I found this dataset. Just by looking at the nature of the data, I realized that it would be a really interesting investigation to **see how Ted Talk speakers open their speeches**. Maybe it won't be the most scientific or technical process, but at least I might learn something. That's good enough for me!
# 
# ## Explore the Data

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import operator
from string import punctuation
import re
from collections import Counter

ts = pd.read_csv("../input/transcripts.csv")
ts.head(10)


# ## Clean Up
# 
# Before progressing, I think it will benefit us to remove all the instances of the following meta and sound effect cues from the start of the transcripts:
# - (...Music...)
# - (...Applause...)
# - (...Video...)
# - (...Sing...)
# - (...Cheer...)
# 
# I actually only realized that this step was necessary after writing the "Learn from the Best" section and seeing these 

# In[ ]:


def removeMusicAndApplause(transcript):
    r = re.compile(r"(\(Music.*?\))|(\(Applause.*?\))|(\(Video.*?\))|(\(Sing.*?\))|(\(Cheer.*?\))", re.IGNORECASE)
    return r.sub("", transcript).strip()

ts['raw_transcript'] = ts['transcript']
ts['transcript'] = ts['transcript'].apply(removeMusicAndApplause)

ts.head()


# ## Insights
# 
# ### Opening Word
# 
# Right away, I can see a number of "classes" of openings. There's the opening thanks, the time-of-day greeting ("good morning!"), and more than one person who starts with Number One ("I am..."). My eventual goal is to classify all of these openings, even if there's a class of "unique" openings at the end. First, let's just see how many different ways there are to start a speech.

# In[ ]:


punctuation_table = str.maketrans({key: None for key in punctuation})
ts['first_word'] = ts['transcript'].map(lambda x: x.split(" ")[0]).map(lambda x: x.translate(punctuation_table))
opening_counts = ts.first_word.value_counts()
opening_counts.head(30).plot(kind='barh', figsize=(6,6), title='Most Common Opening Words', color='g')


# I'm just getting started, but it's already obvious that opening with **a reference to oneself** is by far the most common way to open. Interestingly, in this Top 30 list, two forms of **"There..."** are very close to each other in frequency. Just as interesting is the appearance of **"Chris"** in this list. I'll need to look into that.
# 
# ### Opening Phrases
# I want to a similar analysis using the top phrases that appear in the opening remarks. I'll say that the first 250 characters worth of transcript is a large enough window to capture opening phrases, since Ted Talk-ers are often pressed to get past the pleasantries into their content.

# In[ ]:


r = re.compile(r'[.!?:-]'.format(re.escape(punctuation)))
ts['first_250'] = ts['transcript'].map(lambda x: x[0:250])
phrases = []
for x in ts['first_250'].tolist():
    openings = r.split(x)
    phrases.append(openings[0])
    
phrase_count = Counter(phrases)
phrase_count = sorted(phrase_count.items(), key=operator.itemgetter(1))

phrase, count = zip(*phrase_count)
phrase = [x for _,x in sorted(zip(count,phrase), reverse=True)]
count = sorted(count, reverse=True)
y_pos = np.arange(len(phrase))

number_of_phrases = 20

plt.figure(figsize=(6,6))
plt.title('Most Common Opening Phrases')
plt.yticks(y_pos[:number_of_phrases], phrase[:number_of_phrases])
plt.xticks(np.arange(25))
plt.barh(y_pos[:number_of_phrases], count[:number_of_phrases],color='g')
plt.show()


# Part of being a great speaker is innovating your openings and creating an engaging hook. Yet, more basic than that is simply greeting your audience. Clearly this graph shows that you should not forget to **greet your audience first**.
# 
# But then what? Then, grab your audience with something that will cause them to pay attention. Phrases like **"I have a confession to make"**, **"Let me tell you a story"**, and **"I have a question"** are all great candidates to get you started.
# 
# There are a few details to note here:
# 
# - We get more insight into our **Chris** mystery, by now discovering his full name, likely preceeding a colon. A quick Google search indicates that **Chris Anderson** is a popular TED speaker, even the author of a book, *TED Talks: The Official TED Guide to Public Speaking*. **Pat Mitchell** is the same case.
# - The presence of the **long phrase about SOPA** exceeds our 100 character cut-off and appears twice. It is possible that two speeches begin with this same phrase, such as in the case of a reciting of some document. But this seems unlikely, since it's not a very good speech opener. It's possible that this notice was spoken or inserted by a non-TED-talker source in a couple of recordings. Or, it may be that this is actually a single speech, and that one or more talks in this data set are duplicated in some way.
# - Overall, this data set is **not as clean** as we once thought. If I hadn't done the initial data cleaning, we would see the appearance of metadata here, like "(Music)". And some transcripts begin with the speaker's name, like "Chris Anderson:".

# ## Learn from the Best
# 
# Since we have at least 20 speeches by Chris Anderson, and since he claims to have enough mastery over the art of public speaking to write a book about it, let's see how he begins his speeches. I'll first enhance the data set further by pulling off into a new column any names that appear in the transcript as a prefix to the actual speech. Then we'll see what hooks Chris uses in his speeches.

# In[ ]:


ts['speaker'] = ts['transcript'].map(lambda x: x.split(":")[0])
mask = ts['speaker'].map(lambda x: x.split(" ")).map(lambda x: len(x)) >= 4
ts.loc[mask, 'speaker'] = 'unknown'

ts[ts['speaker'] != "unknown"]['speaker'].head(10)


# Looks like this method wasn't perfect, but we got names for some of these. Now to see how Chris does his speeches.

# In[ ]:


# Sanity check that there we caught the 20 rows with Chris Anderson as the annotated speaker.
by_Chris = ts[ts['speaker'] == "Chris Anderson"]
len(by_Chris)


# In[ ]:


r = re.compile(r'[.!?:-]'.format(re.escape(punctuation)))
raw_openings = by_Chris['transcript'].apply(lambda x: x[0:250])
phrases = []
for x in raw_openings.tolist():
    openings = r.split(x)
    phrases.append(openings[1]) # Skip the "Chris Anderson:"
    
phrase_count = Counter(phrases)
phrase_count = sorted(phrase_count.items(), key=operator.itemgetter(1))

phrase, count = zip(*phrase_count)
phrase = [x for _,x in sorted(zip(count,phrase), reverse=True)]
count = sorted(count, reverse=True)
y_pos = np.arange(len(phrase))

number_of_phrases = 21

plt.figure(figsize=(6,6))
plt.title('Chris Anderson\'s Most Common Opening Phrases')
plt.yticks(y_pos[:number_of_phrases], phrase[:number_of_phrases])
plt.xticks(np.arange(25))
plt.barh(y_pos[:number_of_phrases], count[:number_of_phrases],color='g')
plt.show()


# Ah, **he's never repeated himself twice** in 21 speeches. That's not really that surprising. 
# 
# In fact, he seems to not really be making speeches at all. These are more like **introductions and segways for other speakers**. An interesting find for non-regular TED talk viewers, but not very insightful for our purposes.
# 
# ## Conclusions
# 
# From this investigation, we have found that **"I have a confession to make"** as a phrase is the most popular way to engage audiences from the start. Other popular hooks **start with "I"** but immediately **address "you"**, such as:
# 
# - __I__ have a question for __you__...
# - __I__ know what __you__'re thinking...
# - __I__ am going to tell __you__ about...
# 
# We also learned that **Chris Anderson** is a potential resource for getting better at speaking like a TED talker. Together, these tips can help a speaker who is inspired by the skill of TED talkers. By crafting more engaging openings, we can set a positive tone for our speeches and draw in audiences more effectively.