#!/usr/bin/env python
# coding: utf-8

# Hey, I am trying to explore the Pokemon Go battles here. Let's start!
# ---------------------------------------------------------------------

# First, let's import the packages we'll be using in this kernel.

# In[ ]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as pl


# I'll read the file using Pandas

# In[ ]:


poke = pd.read_csv('../input/pokemonGO.csv')


# Let's take a look at the data

# In[ ]:


poke.head()


# I'm not gonna be using the Image URL and Pokemon No. fields. So, I'll drop them

# In[ ]:


poke = poke.drop(['Image URL','Pokemon No.'],1)


# Let's start using Seaborn to visualize the Pokemon battle stats.
# Below is an example of a [jointplot][1] 
# 
# 
#   [1]: http://seaborn.pydata.org/generated/seaborn.jointplot.html
# I am comparing Max CP and Max HP here

# In[ ]:


sns.jointplot(x="Max CP", y="Max HP", data=poke);


# Isn't that something cool. Let's try some more examples. 
# Below is an example of a [boxplot][1]
# 
# 
#   [1]: http://seaborn.pydata.org/generated/seaborn.boxplot.html

# In[ ]:


sns.boxplot(data=poke);


# The  below [swarmplot][1] is somewhat similar to jointplot. Isn't it?
# 
# 
#   [1]: http://seaborn.pydata.org/generated/seaborn.swarmplot.html#seaborn.swarmplot

# In[ ]:


sns.swarmplot(x="Max CP", y="Max HP", data=poke);


# Tired of dots and swarms. Let's do something interesting with the [barplot][1]
# 
# 
#   [1]: http://seaborn.pydata.org/generated/seaborn.barplot.html#seaborn.barplot
# 
# Doesn't it look like an equalizer?

# In[ ]:


sns.barplot(x="Max CP", y="Max HP", data=poke);


# Let's do one more and call it a day. Here's the '[pointplot][1]
# 
# 
#   [1]: http://seaborn.pydata.org/generated/seaborn.pointplot.html

# In[ ]:


sns.pointplot(x="Max CP", y="Max HP", data=poke);


# Thank you for reading! I hope you enjoyed the show. 
# Keep learning, sharing and of course, visualizing!
# 

# In[ ]:




