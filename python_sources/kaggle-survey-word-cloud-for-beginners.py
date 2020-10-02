#!/usr/bin/env python
# coding: utf-8

# # 100% GENUINE PYTHON CODE WORD CLOUD

# ## HELLO
# 
# Today I'm going to show you how to make those fancy word clouds from 100% Python code using Kaggle data. No need to use online word cloud generators anymore!

# ## Step One 
# 
# Import the necessary packages

# In[ ]:


from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as image 
import pandas as pd
import re
from wordcloud import WordCloud, STOPWORDS
from IPython.display import Image as im


# ## Step Two
# 
# Read in the files

# In[ ]:


kaggle = pd.read_csv('../input/kaggle-survey-2017/freeformResponses.csv')
kaggle.head()


# ## Step Three
# 
# CLEAN ALL THE DATA

# In[ ]:


titles = kaggle.CurrentJobTitleFreeForm.tolist()
titles[0:10]


# Wait... what are nans? We need to get rid of them!

# In[ ]:


clean_titles = [t for t in titles if t != "nan"]
clean_titles[0:10]


# Hmm, that looks weird. Let's examine closer to what the title format is.

# In[ ]:


type(clean_titles[0])


# Interesting! As we can see, the type is a float, meaning we cannot do any string computations on it. 

# In[ ]:


clean_title_str = [str(t) for t in titles]
type(clean_title_str[0])


# In[ ]:


im(filename="drake.jpg")


# Now that it's in a string format, we can clean it!

# In[ ]:


complete_title_str = [t for t in clean_title_str if t != "nan"]


# In[ ]:


complete_title_str[10:20]


# In[ ]:


zstring = ' '.join(str(e) for e in complete_title_str)
print(zstring[750:900])


# We don't want non-alphabetical characters messing up our data!

# In[ ]:


non_alphabet = re.compile('[^A-Za-z ]+')
zstring = re.sub(non_alphabet, ' ', zstring)
print(zstring[90:250])


# In[ ]:


zkeys = zstring.split()
zkeys[0:10]


# In[ ]:


zkeys = [w for w in zkeys if len(w) > 3]
zkeys = [w.lower() for w in zkeys]
zkeys = ' '.join(zkeys)


# In[ ]:


zkeys[0:50]


# ## Step Four
# 
# VISUALIZE ALL THE DATA!

# Pick any of your favorite images to use as a background image! (Hint: works better with larger images with a white background)

# In[ ]:


pi_mask = np.array(Image.open('../input/pisymbol/pisymbol.jpeg'))


# In[ ]:


im(filename="pisymbol.jpeg")


# Create the word cloud instance:

# In[ ]:


wc = WordCloud(background_color="white", max_words=3000, mask=pi_mask,
               stopwords=STOPWORDS)

wc.generate(zkeys)


# In[ ]:


wc.to_file("kaggle.png")


# In[ ]:


im(filename="kaggle.png")


# Ta-da! We've now got our own word cloud made from custom Kaggle survey information from 100% pure Python code!
