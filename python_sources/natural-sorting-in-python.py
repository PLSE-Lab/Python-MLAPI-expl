#!/usr/bin/env python
# coding: utf-8

# # Few Riddles in CTDS contest and Natural Sorting in Python

# natsort (https://github.com/SethMMorton/natsort)
# 
# * natsort comes handy for all natural sorting needs in python.
# 
# * In this CTDS data exploration, natsort comes handy while dealing with episode ids etc where there is sorting requirement for strings that contains numbers
# 
# * There are multiple ways to do such sorting and natsort is one such way. Just thought about sharing this notebook incase it helps few..
# 
# * There are couple of riddles below related to CTDS show..If interested please have a look..

# Installing the library

# In[ ]:


get_ipython().system('pip install natsort')


# In[ ]:


import os
from natsort import natsorted


# In[ ]:


def get_filenames(path):
    st_fname = []
    for f_name in os.listdir(f'{path}'):
        st_fname.append(f_name)
    return st_fname


# In[ ]:


path = '../input/chai-time-data-science/Cleaned Subtitles/'


# <b>Getting list of all files within the path. Took Cleaned Subtitles folder here for illustration..</b>

# In[ ]:


episode_files = get_filenames(path)


# In[ ]:


type(episode_files)


# # Default Sorting

# In[ ]:


episode_files


# <b>As we can see above, ordering is lexical when there is string with numbers.</b>

# # Natural Sorting

# In[ ]:


episode_id = natsorted(episode_files)


# In[ ]:


episode_id


# We got the file names in our required order.
# For further analysis on trend etc, this library will be handy

# <b><u>Tiny bit complex examples</u></b>

# In[ ]:


list1 = ['episode69.1.Masala', 'episode63.2.Paan_rose', 'episode48.3.Ginger2', 'episode44.4.Herbal', 'episode35.5.Ginger1']
list2 = ['sanyam', 2.75, 'E49', '4.12', 'E27', 29.33, 6.89, 'jeremy']


# In[ ]:


natsorted(list1)


# In[ ]:


natsorted(list2)


# <H3>* <u><b>Above lists does not contain random strings/numbers.</b> They have some significance related to CDTS show. </u><br></H3>
#  
# * To find their relevance kindly have a look at my submission notebook https://www.kaggle.com/vinothsuku/1-year-of-ctds-journey-and-what-we-infer

# There are totally numerous ways to handle sorting, just thought this notebook might help some...:)
# 
# Please have a look at https://github.com/SethMMorton/natsort for more details and examples

# In[ ]:




