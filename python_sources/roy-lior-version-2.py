#!/usr/bin/env python
# coding: utf-8

# <font size="5">We chose to analyze **"The Marvel Universe Social Network"**<font/>

# Marvel Comics, originally called Timely Comics Inc., has been publishing comic books for several decades. "The Golden Age of Comics" name that was given due to the popularity of the books during the first years, was later followed by a period of decline of interest in superhero stories due to World War ref. In 1961, Marvel relaunched its superhero comic books publishing line. This new era started what has been known as the Marvel Age of Comics. Characters created during this period such as Spider-Man, the Hulk, the Fantastic Four, and the X-Men, together with those created during the Golden Age such as Captain America, are known worldwide and have become cultural icons during the last decades. Later, Marvel's characters popularity has been revitalized even more due to the release of several recent movies which recreate the comic books using spectacular modern special effects. Nowadays, it is possible to access the content of the comic books via a digital platform created by Marvel, where it is possible to subscribe monthly or yearly to get access to the comics. More information about the Marvel Universe can be found here.

# <font size="3">**The dataset:**<font/>

# The dataset contains heroes and comics, and the relationship between them. The dataset is divided into three files:
# 
# * **nodes.csv**: Contains two columns (node, type), indicating the name and the type (comic, hero) of the nodes.
# * **edges.csv**: Contains two columns (hero, comic), indicating in which comics the heroes appear.
# * **hero-edge.csv**: Contains the network of heroes which appear together in the comics. This file was originally taken from http://syntagmatic.github.io/exposedata/marvel/

# In[1]:


import pandas as pd

#Data importing
edges=pd.read_csv('../input/edges.csv', header=0)
nodes=pd.read_csv('../input/nodes.csv', header=0)
hero_network=pd.read_csv('../input/hero-network.csv', header=0)


# In[2]:


#edges.csv data sample:
edges.sample(5,random_state=12228)


# In[3]:


#nodes.csv data sample:
nodes.sample(5,random_state=12228)


# In[4]:


#hero_network.csv data sample:
hero_network.sample(5,random_state=12228)


# <font size ="4">**Our research goal**</font>
# would be to identify the main heroes cliques- groups of heroes that apeeared in multiple comics together.

# 
