#!/usr/bin/env python
# coding: utf-8

# # Hapax Legomena, Cabbages and Kings
# 
# Out of the roughly 1.3 million words used in *The Simpsons*, about 47,000 are unique. Of those, about 23,000 have been used only once. What do I say to that? **CRANTASTIC**! The Frinkian term for a word used only once in a corpus of text is *hapax legomenon*, the plural being *hapax legomena*. So these methods would let you see a character's hapax, but I also used them to distill his or her unique, vital essence.
# 
# Thanks to William C. and Harizo R. for supplying the data and ideas.

# In[ ]:


import numpy as np
import pandas as pd
from collections import Counter


# There are a lot of characters who appeared on the show:

# In[ ]:


df_char_index = pd.read_csv("../input/simpsons_characters.csv")
print("Nb. of distinct characters: %d " % df_char_index.shape[0])


# So let's extract a subset of handpicked characters from this index. Just look at the *characters.csv* file, and add to this dict if you want to check out a character not here:

# In[ ]:


custom_char_index = {"Maggie": 105, "Marge": 1, "Bart": 8, "Lisa": 9, "Moe": 17,  "Seymour": 3,
                     "Ned": 11, "Grampa": 31, "Wiggum": 71, "Milhouse": 25, "Smithers": 14,
                     "Nelson": 101, "Edna": 40, "Selma": 22, "Barney": 18, "Patty": 10, "Martin": 38,
                     "Todd": 5, "Rod": 121, "Homer": 2, "Cletus": 1413, "Gil": 2369,
                     "Moleman": 963, "Duffman": 2277, "Apu": 208, "Burns": 15, "Dr. Nick": 349,
                     "Dr. Hibbert": 332, "Sideshow Bob": 153,"Krusty": 139, "Fat Tony": 568, "Snake": 518,
                     "Ralph": 119}


# Next, we load the script lines into a DataFrame:

# In[ ]:


data_script_lines = pd.read_csv("../input/simpsons_script_lines.csv",
                    error_bad_lines=False,
                    warn_bad_lines=False,
                    low_memory=False)


# The following code gets all the words used  over all the episodes:

# In[ ]:


dialogue = data_script_lines[['normalized_text']].dropna()
raw_words = []
for row in dialogue['normalized_text']:
    these_words = str(row).split()
    for word in these_words:
        raw_words.append(word)
# lots of dashes in the scripts      
words = [x for x in raw_words if x != '--']
# sometimes they're attached to the ends of words
words = [word[:-2] if word.endswith('--') else word for word in words]
# sometimes to the front
words = [word[2:] if word.startswith('--') else word for word in words]
# numbers - if you want them, comment out the following line:
words = [x for x in words if not (x.isdigit())]
words = sorted(words)    


# So, how many words are we talking about? And how many unique words?

# In[ ]:


print("Number of words: %d " % len(words))
simpsons_vocabulary = set(words)
corpus_vocab = Counter(words)
print("Size of vocabulary in the Simpsons: %d " % len(simpsons_vocabulary))


# In[ ]:


# Get all the words the occur only once for the given wordlist
def hapax(words):
    desired_value = 1
    myDict = dict(Counter(words))
    hapax_legomena_unsorted = [k for k, v in myDict.items() if v == desired_value]
    hapax_legomena = sorted(hapax_legomena_unsorted)
    return hapax_legomena


# Let's get the number of one-time words for the entire corpus:

# In[ ]:


corpus_hapax = hapax(words)
print("Hapax legomena in the corpus: %d " % len(corpus_hapax))


# Now, let's look at individual characters. The following function, **char_words**, does the following:
# 
#  1. Gets all the lines from the character of interest,
#  2. Splits up those lines to get all the words,
#  3. Gets that word count,
#  4. Gets the unique word count

# In[ ]:


def char_words(character, data):
    chosen_char_id = custom_char_index[character]
    df_charac = data[data["character_id"]==str(chosen_char_id)]
    charac_lines = list(df_charac["normalized_text"].values.astype(str))
    # Transform into one big string:
    charac_lines_one_str = ' '.join(charac_lines)
    raw_words = []
    these_words = charac_lines_one_str.split(" ")
    these_words = [x for x in these_words if x != '--']
    these_words = [word[:-2] if word.endswith('--') else word for word in these_words]
    these_words = [word[2:] if word.startswith('--') else word for word in these_words]
    # numbers as words- if you want to keep them, comment out the following line:
    these_words = [x for x in these_words if not (x.isdigit())]
    print("%s: " % character)
    print("Number of words: %d " % len(these_words))
    unique_words = set(these_words)
    print("Number of unique words: %d " % len(unique_words))
    return these_words


# In[ ]:


homer_words = char_words("Homer", data_script_lines)


# Let's look at the words only this character uses with this function, **unique_vocab**:

# In[ ]:


def unique_vocab(character, words, corpus_vocab):
    vocab = Counter(words)
    unique_vocab = []
    for word in vocab.keys():
        if vocab[word] == corpus_vocab[word]:
            unique_vocab.append(word)
    print("Unique vocabulary count: %d " % len(unique_vocab))
    return sorted(unique_vocab)


# In[ ]:


homer_unique_vocab = unique_vocab("Homer", homer_words, corpus_vocab)


#  The function, **char_hapax**, does the following:
# 
#  1. Gets the hapax legomena for the character, i.e., words he or she said only once.
#  2. Gets that character's unique set of hapax, i.e., the hapax words which only that character has spoken.

# In[ ]:


def char_hapax(character, data, corpus_hapax):   
    this_hapax = hapax(data)
    print("Hapax count: %d " % len(this_hapax))
    unique_hapax = sorted(set(corpus_hapax) & set(this_hapax))
    print ("Unique hapax count: %d " % len(unique_hapax))
    return unique_hapax


# In[ ]:


homer_hapax = char_hapax("Homer", homer_words, corpus_hapax)


# Let's look at a sample of them:

# In[ ]:


print(homer_hapax[3000:3100])


# So what are Homer's words, his go-to terms? Remember, these are words that only he says, but they are words that he has said more than once. This next function, **char_common_vocab**, removes the hapax terms from a character's unique vocabulary, leaving only his or her more common terms.

# In[ ]:


def char_common_vocab(unique_vocab, unique_hapax):
    char_words = [x for x in unique_vocab if x not in unique_hapax]
    return char_words


# **Groin-grabbingly** interesting stuff! "5-3226' is Mr. Plow's telephone number : KLondike 5-3226.

# In[ ]:


homer_talk = char_common_vocab(homer_unique_vocab, homer_hapax)
print(homer_talk)


# With these functions we can easily take a look at the other characters:

# In[ ]:


marge_words = char_words("Marge", data_script_lines)
marge_unique_vocab = unique_vocab("Marge", marge_words, corpus_vocab)
marge_hapax = char_hapax("Marge", marge_words, corpus_hapax)
marge_talk = char_common_vocab(marge_unique_vocab, marge_hapax)
print(marge_talk)


# In[ ]:


bart_words = char_words("Bart", data_script_lines)
bart_unique_vocab = unique_vocab("Bart", bart_words, corpus_vocab)
bart_hapax = char_hapax("Bart", bart_words, corpus_hapax)
bart_talk = char_common_vocab(bart_unique_vocab, bart_hapax)
print(bart_talk)


# In[ ]:


lisa_words = char_words("Lisa", data_script_lines)
lisa_unique_vocab = unique_vocab("Lisa", lisa_words, corpus_vocab)
lisa_hapax = char_hapax("Lisa", lisa_words, corpus_hapax)
lisa_talk = char_common_vocab(lisa_unique_vocab, lisa_hapax)
print(lisa_talk)


# In[ ]:


krusty_words = char_words("Krusty", data_script_lines)
krusty_unique_vocab = unique_vocab("Krusty", krusty_words, corpus_vocab)
krusty_hapax = char_hapax("Krusty", krusty_words, corpus_hapax)
krusty_talk = char_common_vocab(krusty_unique_vocab, krusty_hapax)
print(krusty_talk)


# In[ ]:


burns_words = char_words("Burns", data_script_lines)
burns_unique_vocab = unique_vocab("Burns", burns_words, corpus_vocab)
burns_hapax = char_hapax("Burns", burns_words, corpus_hapax)
burns_talk = char_common_vocab(burns_unique_vocab, burns_hapax)
print(burns_talk)


# In[ ]:


ralph_words = char_words("Ralph", data_script_lines)
ralph_unique_vocab = unique_vocab("Ralph", ralph_words, corpus_vocab)
ralph_hapax = char_hapax("Ralph", ralph_words, corpus_hapax)
ralph_talk = char_common_vocab(ralph_unique_vocab, ralph_hapax)
print(ralph_talk)

