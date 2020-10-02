#!/usr/bin/env python
# coding: utf-8

# Hey everyone! Today we are going to learn how to create simple WordClouds. We will be using the wordcloud library and hopefully by the end of this tutorial you all will have a simple understanding to start building your own wordclouds!

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud,STOPWORDS


# **Some Great Quote by Yoda to start off the Kernel**

# In[ ]:


from IPython.display import Image
Image(filename='../input/yoda-pics-1/The-greatest-teacher-failure-is.-Master-Yoda-Star-Wars.png')


# 
# 
#  *THE GREATEST TEACHER, FAILURE IS* - Yoda, The Last Jedi

# First of all, lets start by importing the text files into the notebook. We will be using the "r" method, i.e. read method to (you guessed it!) read the files.

# In[ ]:


eiv=open('../input/star-wars-movie-scripts/SW_EpisodeIV.txt','r')


# I'll use the Episode IV scripts here only for simplicity but i highly suggest to try on Episode V and Episode VI  too! 
# 
# Bonus points if you use it on your own datasets!

# In[ ]:


eiv=eiv.read()


# In[ ]:


print(eiv[:500])


# Hmm, looks good. But I am sure we can make it look better. Let's try to split the text as soon as a newline characters comes up.

# In[ ]:


eiv1=eiv.split("\n")


# In[ ]:


eiv1[:10]


# Just for curiosity, let see how many lines of script are in the movie. We should subtract one from the count because of the '"character" "dialogue"' text in the start.

# In[ ]:


print(len(eiv1)-1)


# Creating a "mask" on which the WordCloud will be made.

# In[ ]:


from PIL import Image
mask = np.array(Image.open("../input/star-wars-movie-scripts/wordcloud_masks/yoda.png"))


# (Note: The IPython.display and PIL have different Image utilities!)

# In[ ]:


stop_words=set(STOPWORDS)
eiv_wc=WordCloud(width=800,height=500,mask=mask,random_state=21, max_font_size=110,stopwords=stop_words).generate(eiv)


# Plotting our WordCloud with matplotlib.

# In[ ]:


fig=plt.figure(figsize=(16,8))
plt.imshow(eiv_wc)


# Looks like we have made a WordCloud. And it looks pretty great!

# So looks like you have reached the end of this kernel. This was an extremely basic implementation of WordCloud and there is a lot you can play around with. Since I am by no means an expert, feel free to correct me if there is any mistake in the kernel.
# 
# Also, if you liked it, please drop an upvote, it really motivates me to make even better kernels. Cheers!

# In[ ]:




