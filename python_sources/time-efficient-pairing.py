#!/usr/bin/env python
# coding: utf-8

# # Time Efficient Pairing
# ## _Work smarter not harder_

# In[ ]:


import pandas as pd
import re
from math import ceil


# At first, I wondered what made this a data science challenge rather than a cryptanalysis competition. Then, having solved the first cipher, I realised that I still had to search 108755 possible plaintexts to find the right one to pair with it.
# 
# What makes this hard is the random padding at the start and ends. Without this padding, you could create a straight forward hash table for quick lookups. One approach would be to systematically strip off the padding; this is possible, but not the easiest approach.
# 
# Let's look at an example: how do we know the following two messages are a pair?

# In[ ]:


pt = "9]8pVnN4n,DaA6[XNib4K2yVIn[jk[MW0VTo5?J62P?'.0HbpEnter, from one side, LUCIUS, IACHIMO, and  the Roman Army: from the other side, the  British Army, POSTHUMUS LEONATUS following,  like a poor soldier. They march over and go  out. Then enter again, in skirmish, IACHIMO  and POSTHUMUS LEONATUS he vanquisheth and disarmeth IACHIMO, and then leaves himvqlhyWqM4ilXEv]dElTRiO2XBC!)9rl(Iy($HLn'd]ktE6b58y"
print(pt)


# In[ ]:


plain = "Enter, from one side, LUCIUS, IACHIMO, and  the Roman Army: from the other side, the  British Army, POSTHUMUS LEONATUS following,  like a poor soldier. They march over and go  out. Then enter again, in skirmish, IACHIMO  and POSTHUMUS LEONATUS he vanquisheth and disarmeth IACHIMO, and then leaves him"
print(plain)


# Well, assuming we've decrypted the ciphertext perfectly, the correct plaintext should be a substring:

# In[ ]:


plain in pt


# This, however, would require 27158x108755 (~11 billion) substring tests to pair all difficulty 1 ciphers. In complexity terms, this is of quadratic order, O(n**2).
# 
# A first observation is that we can reduce the number of possible plaintexts to test against by grouping by size. Let's see how much that would help:

# In[ ]:


train = pd.read_csv('../input/train.csv')
train.text.apply(lambda x: 100*ceil(len(x)/100)).value_counts()


# So for the short messages, that's still **a lot** of comparisons to make. Now at this point, I probably could rewrite my code in C, parallelise it, and chuck it up to the cloud...but that doesn't sound like much fun. I'd prefer to come up with a solution that works on a single core of my 10 year old laptop, and takes no longer than me to make a cup of tea. **Let's work smarter rather than harder.**
# 
# The aim here is to make a [time-memory trade off](https://en.wikipedia.org/wiki/Time/memory/data_tradeoff_attack) such that pairing is quicker at the expense of memory. Grouping by length was a good approach, but the problem was there wasn't enough groups (only 8) and the messages were not spread evenly across them (99.9% in one!). Normally for this type of problem, my go to solution is [Locality Sensitive Hashing](https://en.wikipedia.org/wiki/Locality-sensitive_hashing) which will ensure that messages end up in the right bucket with high probabiliy, but a probabilistic approach feels massively overkill here. 
# 
# Instead, let's design our own 'hash' function that will place messages into buckets. When I say 'hash', you might automatically think of [cryptographic hashing](https://en.wikipedia.org/wiki/Cryptographic_hash_function), but actually that's not want we want here: we do not want to encorporate the entire string into the hash (remember the padding), and hash collisons are desirable here.
# 
# I'll present two hashing techniques I've tried out, and you can comment below if you can think of a better one. First off, notice that most of the words are the same in both strings I'm trying to pair: it's just the padding at the start and end which differs. Perhaps I could just pick out a single word, and use that as my hash value?

# In[ ]:


train[train.text.str.contains('vanquisheth')]


# As I chose a rare word from the message, this produced just a single result (i.e. 'no hash collisions'). If I went for a more common word, I'd have lots of collisions:

# In[ ]:


len(train[train.text.str.contains('from')])


# Let's hash messages into buckets based on 'rare' words, and see how that works. If we were just to chose one rare word to hash based on, there's a chance that that word is the first or last of the message and therefore is garbled by the padding. So, we'll hash each message three times based on the three least common words within it, just to be on the safe side.

# In[ ]:


wordlist = {}
word_rex = re.compile('[A-Za-z]{2,}')
for i,t in train.iterrows():
    for w in word_rex.findall(t.text):
        if w not in wordlist:
            wordlist[w] = 1
        else:
            wordlist[w] += 1
print("Built wordlist frequencies")            


# In[ ]:


rare_map = {}
for i,t in train.iterrows():
    fs = []
    for w in word_rex.findall(t.text):
        fs.append((w, wordlist[w]))
    if len(fs) == 0:
        continue
    fs.sort(key=lambda x:x[1])
    for rare_w,_ in fs[:3]:
        if rare_w not in rare_map:
            rare_map[rare_w] = [t]
        else:
            rare_map[rare_w].append(t)
print("Built hash table")


# Let's see how well balanced the hash table is:

# In[ ]:


pd.Series([len(v) for v in rare_map.values()]).describe()


# Looks like most of the buckets only have one message in, and the biggest bucket isn't too big. We could probably do better (maybe a [tf-idf](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) optimisation approach?), but let's see if this is good enough...

# In[ ]:


def find(pt):
    fs = []
    for w in word_rex.findall(pt):
        if w in rare_map.keys():
            fs.append((w, wordlist[w]))
    if len(fs)==0:
        return None
    fs.sort(key=lambda x:x[1])
    for rare_w, _ in fs[:5]: #We'll check up to 5 rare words, just to be safe.
        for t in rare_map[rare_w]:
            if t.text in pt:
                return t


# In[ ]:


pt = "9]8pVnN4n,DaA6[XNib4K2yVIn[jk[MW0VTo5?J62P?'.0HbpEnter, from one side, LUCIUS, IACHIMO, and  the Roman Army: from the other side, the  British Army, POSTHUMUS LEONATUS following,  like a poor soldier. They march over and go  out. Then enter again, in skirmish, IACHIMO  and POSTHUMUS LEONATUS he vanquisheth and disarmeth IACHIMO, and then leaves himvqlhyWqM4ilXEv]dElTRiO2XBC!)9rl(Iy($HLn'd]ktE6b58y"
find(pt).text


# It works! With this approach, my elderly laptop decrypts and pairs all all difficulty 1 messages in 95 seconds (excluding the pre-computation time). My kettle cannot boil that quick.
# 
# The second idea is to use the capitalised names instead of rare words, and to map all of them rather than just three words. Let's try that:

# In[ ]:


cap_map = {}
cap_rex = re.compile('[A-Z]{5,}')
for i,t in train.iterrows():
    for capw in cap_rex.findall(t.text):
        if capw not in cap_map:
            cap_map[capw] = [t]
        else:
            cap_map[capw].append(t)
print("Built hash table")


# In[ ]:


pd.Series([len(v) for v in cap_map.values()]).describe()


# A lot less buckets than before, and on average bigger buckets.

# In[ ]:


def find2(pt):
    for cap_w in cap_rex.findall(pt):
        if cap_w in cap_map: #We should only need to check one of the capitalised words
            for t in cap_map[cap_w]:
                if t.text in pt:
                    return t


# In[ ]:


find2(pt).text


# With this approach, my elderly laptop decrypts and pairs all all difficulty 1 messages in 86 seconds (excluding the pre-computation time).
# 
# I hope you've found this kernel useful; **let me know your thoughts and suggestions below** (there's probably a better approach that I've completely missed!), and give me an upvote if you've learnt something new :)
