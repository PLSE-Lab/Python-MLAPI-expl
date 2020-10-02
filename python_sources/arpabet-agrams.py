#!/usr/bin/env python
# coding: utf-8

# # Creating a New Word "Game"
# We have all heard of anagrams, where you take a word or phrase and shuffle the letters to make a new one. However, I was wondering what if, instead of letters, what if we shuffled the words base sounds around. The ARPAbet is a set of phonetic transcription codes developed by Advanced Research Projects Agency (ARPA) in the 1970s. It allows us to break down words farther into visual representation of speech sounds. Now what if we shuffled these around to make, what I am calling, an ARPAbet-agram. This is a bit more complex than anyone would ever care to deal with (It makes anagrams look pretty easy by comparasion). Calling it a word "game" is somewhat generous, but we can make a computer do all that teadious work to find ARPAbetagrams and we can just enjoy the results. Feel free to play with this notebook or just download the outputed .CSV file of ~135,000 words and their possible ARPAbetagrams. 
# 
# ## Defining a ARPAbetagram
# 
# I will define an ARPAbetagram as a word (or phrase) whose formed by rearranging the ARPAbet phones of another. 2 additional rules:
# 
# -The ARPAbetagram cannot be a homophone
# 
# -Stresses can be ignored. Since I'm making this up, I can do that. It is like ignoring spaces or capitalization in anagrams. I find it makes for more interseting results. The ARPAbet-agrams are sparse enough without them.
# 
# As an example, the word 'accounts' has a pronuncation in ARPAbet as "AH K AW N T S" and the word 'countess' has a pronuncation in ARPAbet as "K AW N T AH S". Note that both words use the same phones (AH AW K N S T) so are concidered an ARPAbetagram of each other.

# In[1]:


import pandas as pd
import numpy as np
import os
print(os.listdir("../input"))


# In[27]:


dictionary = open('../input/cmudict.dict', 'r')


# # Process ARPAbet dictionary
# First we'll reformat the dictionary into a Dataset with the word and it's pronunciation. I am removing numbers from the set as numbers only indicate minor stress points for vowels in ARPAbet. Try as I might, I am unable to here the difference between these stresses so I am discounting them in this exercise.

# In[28]:


get_ipython().run_cell_magic('time', '', '\nwith dictionary as f:\n    phonics = [line.rstrip(\'\\n\') for line in f]\n\nword = []\npronunciation = []\npronunciation_sorted = []\n\nfor x in phonics:\n    x = x.split(\' \')\n    word.append(x[0])\n    p = \' \'.join(x[1:])\n    # removing numbers from pronunciation\n    p = p.replace(\'0\',\'\')\n    p = p.replace(\'1\',\'\')\n    p = p.replace(\'2\',\'\')\n    pronunciation.append(p)\n    a = p.split(\' \')\n    a.sort()\n    a = \' \'.join(a)\n    pronunciation_sorted.append(a)\n\ndf = pd.DataFrame({\n        "word": word,\n        "pronunciation": pronunciation,\n        "pronunciation_sorted": pronunciation_sorted\n    })\n\n# add placeholder columns\ndf[\'ARPAbetagrams\'] = \'\'\ndf[\'index\'] = df.index\ndf[:10]')


# # Find all ARPAbetagram
# Note: This runs a but slow but gets the job done. Takes ~1 hour to complete. The result will be a new column listing all the ARPAbetagrams of that word.

# In[ ]:


get_ipython().run_cell_magic('time', '', "def fillARPAbetagrams(line):\n    word = line[0]\n    cp = line[1]\n    cpa = line[2]\n    p = 0\n    i = line[3]\n    if i % 1350 == 0:\n        print(str(i/1350)+'% done')\n    \n    pg = df.loc[(df['pronunciation_sorted'] == cpa) & (df['pronunciation'] != cp)]['word'].values.tolist()\n    \n    pg = ','.join(pg)\n    h = ''\n    return pg\ndf['ARPAbetagrams'] = df[['word', 'pronunciation', 'pronunciation_sorted', 'index']].apply(fillARPAbetagrams, axis = 1)\n\ndf.drop(['index'], axis=1)")


# # Look at the Results
# As you can see, ARPAbetagrams are pretty rare. Most words have none. Many words only have a few because the dataset inculdes some questionable words. That being said, there are some pretty interesting and unexpected ARPAbetagrams mixed throughout. Making a program that can go through a phrase and find ARPAbetagrams of it might be phase 2 of this notebook, but I will leave it here for now.
# 

# In[25]:


# df.loc[(df['word'] == 'accord')]
df[:50]


# # Output the CSV File
# Enjoy going through the dataset

# In[ ]:


df.to_csv("ARPAbetagrams_Dataset.csv", index=False, header=True)

