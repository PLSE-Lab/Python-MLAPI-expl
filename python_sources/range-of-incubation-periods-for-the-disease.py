#!/usr/bin/env python
# coding: utf-8

# # ****Range of incubation periods for the disease(Covid-19) in humans****

# In[ ]:


import numpy as np
import pandas as pd
import json
import re
import os
from matplotlib import style
from tqdm import tqdm
style.use('fivethirtyeight')
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams.update({'figure.figsize':(10,8),'figure.dpi':(70)})
plt.style.use(['fivethirtyeight'])


# # ****These are the directories where the articles are saved.****

# In[ ]:


root_dirs = ['/kaggle/input/CORD-19-research-challenge']
dirs = ['biorxiv_medrxiv', 'comm_use_subset', 'noncomm_use_subset', 'custom_license']
docs = list()


# The below snippet parses the JSON files using the key and values parsion and then saving the Title, Abstract and the Text of the Scholarly Article

# In[ ]:


for rd in root_dirs:
    for d in dirs:
        for file in (os.listdir(f"{rd}/{d}/{d}")):
            file_path = f"{rd}/{d}/{d}/{file}"
            j = json.load(open(file_path,'rb'))
            title = j['metadata']['title']
            
            try:
                abstract = j['abstract'][0]['text']
            except:
                abstract = ''
                
            full_text = ''
            for text in j['body_text']:
                full_text = full_text + text['text'] + '\n\n'
            docs.append([title, abstract, full_text])
df = pd.DataFrame(docs, columns = ['Title', 'Abstract', 'full_text'])
keyword = "incubation"
incubation = df[df['full_text'].str.contains(keyword)]


# In[ ]:


incubation.head()


# In[ ]:


text = incubation['full_text'].values


# Utilizing the Regular Expression rather a naive method to get the days in the numeric form and append the incubation days in the list

# In[ ]:


keyword_time = list()
for t in text:
    for sentences in t.split(". "):
        if keyword in sentences:
            single_day = re.findall(r" \d{1,2} day", sentences)
            
            if len(single_day) == 1:
                num = single_day[0].split(" ")
                keyword_time.append(float(num[1]))
print(keyword_time)


# In[ ]:


print("The mean projected incubtion time is:",np.mean(keyword_time))


# Plotting the Days in the Histogram

# In[ ]:


plt.hist(keyword_time, bins = 50, density = True,  width = 8.0, color = 'skyblue')
plt.ylabel("Bin Counts")
plt.xlabel("Incubation Time")


# In[ ]:




