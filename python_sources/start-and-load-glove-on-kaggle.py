#!/usr/bin/env python
# coding: utf-8

# ## Introduction
# In this notebook we would learn using Glove for NLP tasks
# This notebook aims at giving an intutive understanding to vectors present in glove and how they look

# In[ ]:


from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# Displaying the contents of root folder
# 

# In[ ]:


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# ### Loading the Glove file

# In[ ]:


embeddings_index = {}
with open('/kaggle/input/glove.6B.100d.txt','r') as f:
    for line in f:
        values=line.split()
        word=values[0]
        vectors=np.asarray(values[1:],'float32')
        embeddings_index[word]=vectors
f.close()


# * #### Total number of word tokens

# In[ ]:


print('Total %s word vectors.' % len(embeddings_index))


# ### Exploring words and their vectors

# In[ ]:


first_20_words_in_glove=list(embeddings_index.keys())[:20]
print('first 20 tokens')
print(first_20_words_in_glove)


# In[ ]:


embeddings_index['the']


# In[ ]:


embeddings_index['earthquake']


# ## Conclusion
# This concludes your starter analysis! To go forward from here, click the blue "Edit Notebook" button at the top of the kernel. This will create a copy of the code and environment for you to edit. Delete, modify, and add code as you please. Happy Kaggling!

# In[ ]:




