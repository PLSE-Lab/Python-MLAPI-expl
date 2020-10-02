#!/usr/bin/env python
# coding: utf-8

# # Generating Data from the Chai
# 
# While helping out Sanyam with this dataset, I wrote a little script that converts our raw text into something that can work for ML frameworks, so let's get into it!

# In[ ]:


import pandas as pd
import re


# Let's take a look at some of the files we're working with. We'll use the raw `.txt`'s:

# In[ ]:


f = open('../input/chai-time-data-science/Raw Subtitles/E75.txt')


# In[ ]:


f.readline()


# In[ ]:


f.readline()


# In[ ]:


f.readline()


# So we can see that we've got the speaker, a timestamp, and the text, before a `\n` at the end. So let's write a function to do this:

# In[ ]:


def extract_transcript(fn, save=False, save_path=''):
    "Takes transcript and converts it to `DataFrame`"
    pat = r'([A-Za-z]|\s+)\s([0-9]{0,2}:{0,1}[0-9]{1,2}:[0-9][0-9])'
    f = open(fn, "r")
    t = True
    df = pd.DataFrame(columns = ['Time', 'Speaker', 'Text'])
    i = 0
    first = True
    while t:
        line = f.readline()
        if line == '': t = False
        i += 1
        line = re.split(pat, line[:-1])
        if len(line) == 4:
            is_new = 1
            speak = line[0]
            time = line[2]
        while is_new == 1:
            if first:
                line = f.readline()
                for i in range(6):
                    l_c = f.readline()
                    if speak not in l_c and time not in l_c:
                        line += l_c
                i += 1
                first = False
            else:
                line = f.readline()
                i += 1
            if len(line) > 2 and line != '\n':
                line = line[:-1]
                df.loc[i] = [time, speak, line]
                df.reset_index()
            else:
                is_new = 0
    df.reset_index(drop=True, inplace=True)
    df['Text'] = df['Text'].replace('\n', '')
    if save:
        df.to_csv(save_path+fn.name[:-3] + 'csv', index=False, sep='|')
    return df


# In our function we're using regex to extract the speaker name and the time, and if it's the first row, we grab the next 6 `readlines` due to how Sanyam has his wonderful interviews formatted. Let's try that one:
# 
# > Also, do note the seperator for our `csv`'s are a pipe (|), this is due to how we have many commas in our transcripts, so NumPy won't play nice

# In[ ]:


df = extract_transcript('../input/chai-time-data-science/Raw Subtitles/E73.txt')


# In[ ]:


df.head()


# In[ ]:


df.tail()


# So now we have an annotated format! Let's convert all of them:
# 
# We'll use `fastai`'s helpful additions to pathlib as well:

# In[ ]:


from fastai.vision import *


# In[ ]:


path = Path('../input/chai-time-data-science/Raw Subtitles/')


# In[ ]:


path.ls()[:5]


# In[ ]:


for fn in path.ls():
    extract_transcript(fn, '../output/kaggle/working/')


# And now we have working CSV's for the entire dataset, have fun!

# In[ ]:


df = pd.read_csv('./E70.csv', delimiter='|')


# In[ ]:


df.head()


# In[ ]:


df.tail()


# In[ ]:




