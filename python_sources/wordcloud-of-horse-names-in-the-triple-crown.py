#!/usr/bin/env python
# coding: utf-8

# # Introduction
# In this short kernel, I am going to create a wordcloud based on the names of all the horses that competed in the Triple Crown races between 2005 and 2019 to see if there are any consistent patterns of how thoroughbreads are named.

# Importing relevant libraries.

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from wordcloud import WordCloud, STOPWORDS
from PIL import Image
import seaborn as sns
from os import path, getcwd
import os
from matplotlib import cm
from matplotlib.colors import ListedColormap


# Loading the data

# In[ ]:


df = pd.read_csv("../input/triple-crown-of-horse-races-2005-2019/TripleCrownRaces_2005-2019.csv")


# Identifying the words in the horse names.

# In[ ]:


horse_words = " "
stopwords = set(STOPWORDS)
for val in df['Horse']:
    val = str(val)
    tokens = val.split()
    
    for i in range(len(tokens)):
        tokens[i] = tokens[i].lower()
    for words in tokens:
        horse_words = horse_words + words + ' '


# Creating a custom colormap, just because I like how it looks.

# In[ ]:


brg = cm.get_cmap('gist_heat', 256)
new_colors = brg(np.linspace(0, 1, 256))
new_colors = new_colors[:-50]
newcmp = ListedColormap(new_colors)


# Create the wordcloud object and import an image of a horse sillhouette as a mask for the wordcloud. I uploaded the horse picture manually, so I had to first find where it was by using `os.listdir`.

# In[ ]:


print(os.listdir("../input"))


# This actually creates the wordcloud object. The mask parameter takes the image that I imported and will fit the wordcloud within its bounds. It took me some time to find one that worked well, but it seems like having an image with a white background does the trick. The image itself does not have to be black and white. I also had trouble getting it to work with .png files and images with a smaller size.

# In[ ]:


mask = np.array(Image.open('../input/horsejpg/horse.jpg'))

wordcloud = WordCloud(width = 2400, height = 1800, mask = mask, background_color = 'white', colormap = newcmp,
                      contour_width = 0.5, contour_color = 'black')
wordcloud.generate(horse_words)


# Now, finally I can plot the wordcloud.

# In[ ]:


plt.figure(figsize = (10, 8))
plt.imshow(wordcloud, interpolation = "quadric", aspect = 'auto', origin = 'upper')
plt.axis("off")
plt.tight_layout(pad = 0)
plt.savefig('wordcloud.png', facecolor='w', bbox_inches='tight')


# # Wrapping up
# Looks like owners like to name their horses in a very macho way. Names including 'Man', 'King', 'War', and 'Daddy' popped up quite a lot! I still need to polish this graph a bit to get it looking better. I can't quite figure out why the image is getting truncated on the edges. Also, I need to see if there is a way I can smooth out the outline of the cloud. Thanks for following along with me!
