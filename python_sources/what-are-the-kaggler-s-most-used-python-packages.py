#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# Let us read the py_packages file and analyze the data to find the top python packages.

# In[ ]:


import json

from pathlib import Path
from collections import Counter
from wordcloud import WordCloud

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


pwd


# In[ ]:


folderName = Path('../input/packages-used-in-kernels/')
fileName = 'py_packages.json'

print(folderName)
print(folderName / fileName)

with open(folderName / fileName) as f:
    data = json.load(f)


# In this notebook we answer the following questions
# 
# *  How many packages are imported per kernel
# 
# *  What are the most frequently used packages
# 
# Finally we have word cloud of the packages used.

# ### Number of Packages per Kernel

# In[ ]:


import_lens = [len(kernel_data) for kernel_data in data]
N = max(import_lens)
print(
    f'Maximum number of packages: {N} and Minimum number of packages: {min(import_lens)}')


# In[ ]:


def bins_labels(bins, **kwargs):
    bin_w = (max(bins) - min(bins)) / (len(bins) - 1)
    plt.xticks(np.arange(min(bins)+bin_w/2, max(bins), bin_w), bins, **kwargs)
    plt.xlim(bins[0], bins[-1])


# In[ ]:


figsize = (16, 8)
bins = list(range(0, N))

plt.style.use('ggplot')
plt.figure(figsize=figsize)
plt.hist(import_lens, bins=bins, density=True)
bins_labels(bins, fontsize=20)
plt.title('Number of Imported Packages')
plt.xlabel('Number of Packages')
plt.ylabel('Frequency')


# In[ ]:


figsize = (8, 6)
plt.style.use('ggplot')
plt.figure(figsize=figsize)
plt.boxplot(import_lens, showmeans=True)
plt.xticks(ticks=[1], labels=['No of Packages'])
plt.title('Number of Imported Packages')


# ### Most used packages

# In[ ]:


# %%timeit -o
count = Counter([])
for kernel_data in data:
    count.update(list(elem[1] for elem in kernel_data))


# In[ ]:


complete_text = []
for kernel_data in data:
    complete_text.append(' '.join(list(elem[1] for elem in kernel_data)))


# In[ ]:


package_freq = pd.Series(count)
package_freq


# In[ ]:


top_10_packages = package_freq.sort_values(ascending=False)[0:10]
top_10_packages


# In[ ]:


figsize = (16, 8)

plt.style.use('ggplot')
plt.figure(figsize=figsize)
plt.bar(top_10_packages.index, top_10_packages/sum(package_freq))
plt.title('Top 10 packages')
plt.xlabel('Package names')
plt.ylabel('Frequency')


# ### WordCloud

# In[ ]:


wordcloud = WordCloud(width=800, height=800, collocations=False,
                      background_color='white',
                      min_font_size=10)


# In[ ]:


cloud = wordcloud.generate(' '.join(complete_text))

plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(cloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
plt.show() 


# This clearly shows numpy, matplotlib, pandas are the most used packages. 
# 
# Surprisingly there is no mention of torch in this list!

# In[ ]:




