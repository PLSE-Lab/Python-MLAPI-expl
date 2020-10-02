#!/usr/bin/env python
# coding: utf-8

# ![](http://i.imgur.com/4WYIeJh.png)

# Markov chains describe probabilisitc systems which evolution has no memory. This means that the next state of the system depends only on the actual state and not on any of the previous states. A bit like if what you will do tomorrow is decided entirely on today not on your past: yesterday and prior are irrelevant in that choice. Few systems really have no memory but these models are really useful in modelisation and applications such as generating text. The Markov chain generates text, in a words by words fashion (words mean state here and the number of word in a state is a free parameter called `state_size`). The generation comes after the model has read transition probabilities from some text corpus.
# 
# We present here an example showing generation of arXiv article titles and abstract from existing ones. The end result could fool the laymen, as we will see.

# ### Importing the libraries

# In[ ]:


import numpy as np
import pandas as pd
import markovify #ready-to-use text Markov chain


# ### Importing data

# In[ ]:


df = pd.read_json("../input/arxivData.json")

#concatenating titles and abstract in new column
df["all_text"] = df["title"] + ". " + df["summary"]
df["all_text"] = df["all_text"].map(lambda x : x.replace("\n", " "))
df.head(5)


# ### What's a Markov Chain text generator?

# ### Building Markov chain models

# In[ ]:


#number of words defining a state in the text Markov chain
STATE_SIZE = 2

#generating a model for all the text and one only for titles
text_model = markovify.Text( df["all_text"], state_size=STATE_SIZE)
title_model = markovify.Text( df["title"], state_size=STATE_SIZE)


# * We picked states composed of two words. It's a compromise between variance, "creativity" of the word generation, and the requirement of a sensible output, respectively small and large `state_size`.

# ### Generating random article titles + abtracts

# In[ ]:


def findnth( str, char=" ", n=2):
    """
    Returns position of n-th occurence of pattern in a string
    """
    
    index_from_beg = 0
    while n >= 1:
        index = str.find( char)
        str = str[index+1:]
        index_from_beg += index + len(char)
        n -= 1
    return index_from_beg

sample_size = 7
successes = 0
while successes < sample_size:
    try: #some make_sentence calls raise a KeyError exception for misunderstood reasons
        #first generating a title
        _title = title_model.make_sentence()
        _end_of_title = " ".join( _title.split()[-STATE_SIZE:])

        #generating abstract from the end of the tile
        _abstract = text_model.make_sentence_with_start( _end_of_title)
        
        #concatenating both
        index = findnth( _abstract, " ", 2)
        _abstract = _abstract[index:]
        _full_article_description = _title + " " + _abstract
        print( _full_article_description, end="\n\n")
        successes += 1

    except:
        pass


# After playing with this a bit we find a variety of length with abstracts ranging from 1 to 10 lines. At first read most of these abstract sound "plausible" and could fool the non specialized audience. They could maybe be shared on  [snarXiv](http://snarxiv.org/).
