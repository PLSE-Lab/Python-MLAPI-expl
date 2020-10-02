#!/usr/bin/env python
# coding: utf-8

# ## Loading and playing the music
# First load the data using pickle. Here we are only loading 200 (out of 7134) due to memory.
# 

# In[ ]:


get_ipython().system('pip install pygame pypianoroll')
import pickle
import random
with open("../input/music/music.pk", "rb") as f:
    music = pickle.load(f)
music = random.sample(music, 200)


# In[ ]:


music[0]


# In[ ]:


from collections import Counter
import matplotlib.pyplot as plt
c = Counter([i["composer"] for i in music]).items()
fig, ax = plt.subplots(figsize=(40, 20))
ax.bar(*zip(*c))


# The `play` function will not work inside the kernel but should work if you run it locally. Couldn't find a great way to do it but using pygame does the job.

# In[ ]:


from pypianoroll import Multitrack, Track
import pygame
def write_midi(arr, filename):
    Multitrack(tracks=[Track(arr*127)]).write(filename)
def play(filename):
    pygame.init()
    pygame.mixer.music.load(filename)
    pygame.mixer.music.play()
    


# Ok... the next bit looks a bit complicated but all it does is concatentate all composers pieces into one big piece per composer then splits each big piece into 1024 sized chunks, with each chunk having the label of the composer.

# In[ ]:


import itertools
from scipy.sparse import vstack
import tqdm
num_composers = 40
chunk_size = 1024
groups = itertools.groupby(sorted(music, key=lambda x: x["composer"]), lambda x: x["composer"])
segments = []
for composer, pieces in tqdm.tqdm_notebook(groups, total=num_composers):
    pieces_list = list(i["piece"].tocsr() for i in pieces)
    n = sum([i.shape[0] for i in pieces_list])//chunk_size
    if n!=0:
        trimmed_concat  = vstack(pieces_list)[:chunk_size*n]
        composer_segs = [(trimmed_concat[i:i+chunk_size], composer) for i in range(0,n*chunk_size,chunk_size)]
        segments.extend(composer_segs)
random.shuffle(segments)


# Lets see the distribution of the number segments per composers

# In[ ]:


c = Counter(seg[1] for seg in segments).items()
fig, ax = plt.subplots(figsize=(40, 20))
ax.bar(*zip(*c))


# Here is a fun game. Guess the composer from the snippet played

# In[ ]:


def test(num):
    answers = []
    for seg, comp in segments[:num]:
        write_midi(seg.toarray(), "temp.mid")
        play("temp.mid")
        inp = input("Who was it?")
        if inp=="quit":
            break
        if len(inp)>=3 and inp.lower() in comp.lower():
            print(f"Correct the composer was {comp}")
            answers.append((comp, True))
        else:
            print(f"Incorrect the composer was {comp}")
            answers.append((comp, False))
    return answers
        
    


# In[ ]:


#test(10) will work locally

