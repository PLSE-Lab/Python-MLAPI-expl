#!/usr/bin/env python
# coding: utf-8

# # ARPAbet-agrams (Phrases)
# This is a quick follow up notebook to my ARPAbetagrams notebook. The difference is while that one was about finding ARPAbetagrams for individual *words* this one deals with finding ARPAbetagram of *phrases*. This is much more computationally expensive so we will only be able deal with one phrase at a time, rather than a whole dataset. See the original ARPAbet-agrams and download the full ~135,000 word dataset here: https://www.kaggle.com/valkling/arpabet-agrams
# 
# For reveiw, here is what an ARPAbetagram is:
# 
# # Creating a New Word "Game"
# We have all heard of anagrams, where you take a word or phrase and shuffle the letters to make a new one. However, I was wondering what if, instead of letters, what if we shuffled the words base sounds around. The ARPAbet is a set of phonetic transcription codes developed by Advanced Research Projects Agency (ARPA) in the 1970s. It allows us to break down words farther into visual representation of speech sounds. Now what if we shuffled these around to make, what I am calling, an ARPAbet-agram. This is a bit more complex than anyone would ever care to deal with (It makes anagrams look pretty easy by comparasion). Calling it a word "game" is somewhat generous, but we can make a computer do all that teadious work to find ARPAbetagrams and we can just enjoy the results.
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

# In[ ]:


import pandas as pd
import numpy as np
import os
print(os.listdir("../input"))


# In[ ]:


dictionary = open('../input/cmu-pronouncing-dictionary/cmudict.dict', 'r')


# # Process ARPAbet dictionary
# First we'll reformat the dictionary into a Dataset with the word and it's pronunciation. I am removing numbers from the set as numbers only indicate minor stress points for vowels in ARPAbet.

# In[ ]:


get_ipython().run_cell_magic('time', '', '\nwith dictionary as f:\n    phonics = [line.rstrip(\'\\n\') for line in f]\n\nword = []\npronunciation = []\npronunciation_sorted = []\n\nfor x in phonics:\n    x = x.split(\' \')\n    word.append(x[0])\n    p = \' \'.join(x[1:])\n    # removing numbers from pronunciation\n    p = p.replace(\'0\',\'\')\n    p = p.replace(\'1\',\'\')\n    p = p.replace(\'2\',\'\')\n    pronunciation.append(p)\n    a = p.split(\' \')\n    a.sort()\n#     a = \' \'.join(a)\n    pronunciation_sorted.append(a)\n\ndf = pd.DataFrame({\n        "word": word,\n        "pronunciation": pronunciation,\n        "pronunciation_sorted": pronunciation_sorted\n    })\n\nprint(df.shape)')


# # Merge a Second Dataset
# Unlike the original ARPAbetagrams notebook, I am going to merge a second dataset. This is not to make *more* words to work with but rather make *less*. Many "words" in the pronunciation dict are not really words but rather common sounds. This was fine for the single word ARPAbetagrams because most words have only a handfull anyways so a few extra hits with sorta words and sounds is still kind of interesting. However, with phrases the results will be really massive without them anyways. merging the dict with this frequency one will filler it to words that are only common between the 2 datasets. That being said, this frequency dataset is large and has uncommon and/or non-words too so it does not cut *too* deep and we will still see some really complete results.

# In[ ]:


unigram_freq = pd.read_csv('../input/english-word-frequency/unigram_freq.csv')

# unigram_freq = unigram_freq.loc[unigram_freq['word'].isin(word)]
# df = df.loc[df['word'].isin(fword)]
df = pd.merge(df,unigram_freq, on='word')


# And the dataset goes from 135,010 to 92433 words

# In[ ]:


print(df.shape)


# # Pick and Prepare a Phrase
# So lets find all ARPAbetagrams for a phrase. The sample line will be "a dog ate a taco". (Trying different lines is great but be aware of computational explosion!)
# 

# In[ ]:


Phrase = "a dog ate a taco"
Phrase = Phrase.lower().split()
Data_List = df[['word','pronunciation_sorted']].values.tolist()

ARPA_Phonics = []

for x in Phrase:
    Word_Array = df.loc[df['word'] == x]['pronunciation_sorted'].values[0]
    ARPA_Phonics += Word_Array

print(ARPA_Phonics)


# # Find All ARPAbetagrams for a Phrase
# While finding all the ARPAbetagrams for "a dog ate a taco" might seem simple, we are dealing with a lot of possible combinations. The 11 sounds in "a dog ate a taco" can be ordered almost 40 million different ways. To check that agains combinations of 92,433 words is quite a bit. Still, this code will work in ~1 minute to find over 1 million ARPAbetagrams for that phrase. (on the full cmudict.dict dataset, this jumps to almost 3 million)

# In[ ]:


get_ipython().run_cell_magic('time', '', 'def ARPA_Phrases(Data, phrase, word_total, First):\n    count = []\n    for i ,line in enumerate(Data):\n#         if First and i % 1 == 0:\n#             print(i)\n        word = line[0]\n        pron = line[1]\n        curword_total = word_total.copy()\n        curphrase = phrase.copy()\n        if all(x in curphrase for x in pron):\n            try:\n                for x in pron:\n                    del curphrase[curphrase.index(x)]\n                curword_total.append(word)\n                if curphrase == []:\n                    count.append(curword_total)\n                else:\n                    nlist = []               \n                    for nline in Data[i:]:\n                        if all(x in curphrase for x in nline[1]):\n                            nlist.append(nline)\n                    count += ARPA_Phrases(nlist, curphrase, curword_total, False)\n            except:\n                word\n    return count\n\nNew_List = []               \nfor line in Data_List:\n    if all(x in ARPA_Phonics for x in line[1]):\n        New_List.append(line)\n\nData_List = New_List\ncount = ARPA_Phrases(Data_List, ARPA_Phonics, [], True)\n\nprint(len(count))')


# # Look at the Results
# Unlike the single words, where ARPAbet-agrams are pretty rare, we end up with tons of ARPAbetagrams from such a short phrase. This is consitant with what we expect with computational explosion. Now the results do have a lot of sorta words still and alternate spellings make up a lot. Purging the dataset of these sorta words might make the results easier but, for now, I'm keeping it to this more complete list.
# 

# In[ ]:


count[:100]

