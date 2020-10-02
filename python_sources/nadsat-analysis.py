#!/usr/bin/env python
# coding: utf-8

# **Prepping Data**<br>
# This is to seperate the Nadsat words from their English counterparts

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

def data_prep(raw):
    out_y = raw.values[...,0,]
    batch_size = raw.shape[0]
    out_x = raw.values[...,1,]
    return list(out_x), list(out_y)


# **Reading in Data** <br>This is getting the data from the filesystem and calling the above function to prep the data for analysis

# In[ ]:


file = '../input/nadsat vocab.csv'
raw_data = pd.read_csv(file)
english, nadsat = data_prep(raw_data)


# **Nadsat Frequency chart ** <br>We can see here that in Nadsat, one word generally has one definition with one exception: Razrez. Razrez can mean both "Anger" and "Tear".

# In[ ]:


raw_data['word'].value_counts().head(30).plot.bar()


# **English Frequency chart**<br>
# This chart shows that the Nadsat word-to-definition ratio is **not** one-to-one. Many different (Up to 4) Nadsat words can mean one English word.

# In[ ]:


raw_data['definition'].value_counts().head(30).plot.bar()


# ****

# **Analysis of Nadsat  (3-letter) Prefixes**<br>
# We can take away from this analysis that some of the Nadsat prefixes can be traced to their English definition.

# In[ ]:


#Word Object
class Word:
    def __init__(self, word, count):
        self.word = word
        self.count = count
    def __repr__(self):
        return "{"+str(self.word)+ ", " +str(self.count)+"}"
    def __eq__(self, obj):
        return self.word == obj.word and self.count == obj.count
    def __hash__(self):
        return hash(self.word)

#Getting all prefixes of word
def gas(input_string):
  length = len(input_string)
  return [input_string[:i+1] for i in range(length)]

freq = []
subs = []

#Sorting prefixes into frequency list
for word in nadsat:
    for s in gas(word):
        if len(s) == 3:
            subs.append(s)
for s in subs:
    freq.append(Word(s, subs.count(s)))
freq = sorted(set(freq), key = lambda x: x.count)[::-1]

perm = []
temp = []

#Connecting frequency list with english definition list
for s in freq:
    temp = []
    temp.append(s.word)
    for i in range(len(nadsat)):
        if s.word in gas(nadsat[i]):
            temp.append(english[i])
    perm.append(temp)

print("Prefix ", " English counterparts to Nadsat words with prefix")
#Printing results
for i in range(10):
    res = ""
    res += perm[i][0] + ":     "
    for w in range(len(perm[i])):
        if w != 0:
            res += perm[i][w] + " | "
    print(res)
        


# **Analysis of Nadsat  (3-letter) Suffixes**<br>
# Unlike prefixes, suffixes seem to be more connected to the sound of the word than the meaning. (Ex. "Noisy" and "Crazy" are connected and "Speak" and "Drink")

# In[ ]:


#Word Object
class Word:
    def __init__(self, word, count):
        self.word = word
        self.count = count
    def __repr__(self):
        return "{"+str(self.word)+ ", " +str(self.count)+"}"
    def __eq__(self, obj):
        return self.word == obj.word and self.count == obj.count
    def __hash__(self):
        return hash(self.word)

#Getting all prefixes of word
def gas(input_string):
  length = len(input_string)
  return [input_string[i:] for i in range(length)]

freq = []
subs = []

#Sorting prefixes into frequency list
for word in nadsat:
    for s in gas(word):
        if len(s) == 3:
            subs.append(s)
for s in subs:
    freq.append(Word(s, subs.count(s)))
freq = sorted(set(freq), key = lambda x: x.count)[::-1]

perm = []
temp = []

#Connecting frequency list with english definition list
for s in freq:
    temp = []
    temp.append(s.word)
    for i in range(len(nadsat)):
        if s.word in gas(nadsat[i]):
            temp.append(english[i])
    perm.append(temp)

print("Suffix ", " English counterparts to Nadsat words with suffix")
#Printing results
for i in range(10):
    res = ""
    res += perm[i][0] + ":     "
    for w in range(len(perm[i])):
        if w != 0:
            res += perm[i][w] + " | "
    print(res)
        


# **Vowel to Consonant Analysis**<br>
# From this data we can see that Nadsat has a much higher consonant to vowel ratio than english.
# 

# In[ ]:


def v_count(string):
    c = 0
    for i in string:
        if i in "aeiouAEIOU":
            c += 1
    return c

def c_count(string):
    c = 0
    for i in string:
        if i not in "aeiouAEIOU":
            c += 1
    return c

english_c = 0
english_v = 0

for i in english:
    english_c += c_count(i)
    english_v += v_count(i)

nadsat_v = 0
nadsat_c = 0

for i in nadsat:
    nadsat_c += c_count(i)
    nadsat_v += v_count(i)

print("Nadsat consonant to vowel ratio:", round(nadsat_c / nadsat_v, 2))
print("English consonant to vowel ratio:", round(english_c / english_v, 2))


# **Average Length of Word Analysis**<br>
# The average Nadsat word has one more letter than the average English word

# In[ ]:


nadsat_len = 0
english_len = 0

for i in english:
    english_len += len(i)

for i in nadsat:
    nadsat_len += len(i)
    
print("Average Nadsat word length:", round(nadsat_len / len(nadsat), 2))
print("Average English word length:", round(english_len / len(english), 2))


# **Conclusion**<br>
# This may seem like a stretch, but hear me out. Anthony Burgess was a music composer. Music is all about communicating without language (Excluding lyrics). Burgess may have wanted to try to do the same thing but in writing. I belive that Nadsat is his way of doing this. Nadsat is far enough away from English that by itself, a reader wouldn't understand it. Nadsat is close enough to English in terms of following similar patterns, being of similar length, and having similar consistancy within itself to allow the brain to relate to it in a similar fashion to how it relates to English. This is much like how sounds in music can convey emotion to us by following patterns that are familiar to our brain that were developed in spoken language.
# 
# 
# 
# 
# 
