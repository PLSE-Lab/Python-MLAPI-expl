#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install wordcloud')
get_ipython().system('wget -O ../working/abril_fatface.zip "https://fonts.google.com/download?family=Abril%20Fatface"')
get_ipython().system('unzip ../working/abril_fatface.zip -d ../working')


# In[ ]:


import os

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import Image, display
from wordcloud import WordCloud

print(os.listdir("../input"))


# In[ ]:


get_ipython().run_line_magic('cd', '../input')
plots = pd.read_csv("wiki_movie_plots_deduped.csv")
plots.head()


# In[ ]:


wordcloud = WordCloud(width=1600, height=800, font_path="../working/AbrilFatface-Regular.ttf", background_color="white", contour_color="white")     .generate(" ".join(plots.Plot))     .to_file("../working/word_cloud.png")


# In[ ]:


get_ipython().run_line_magic('cd', '../working')

display(Image(filename="word_cloud.png"))


# In[ ]:


get_ipython().system('rm abril_fatface.zip AbrilFatface-Regular.ttf OFL.txt ')

