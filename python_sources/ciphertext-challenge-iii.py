#!/usr/bin/env python
# coding: utf-8

# ## Import Packages

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')

pd.options.display.max_rows = 16
pd.options.display.max_columns = 32

os.listdir('../input')


# In[ ]:


plain_text = pd.read_csv('../input/train.csv')
cipher_text = pd.read_csv('../input/test.csv')
sample_df = pd.read_csv('../input/sample_submission.csv')


# In[ ]:


plain_text.head()


# In[ ]:


cipher_text.head()


# In[ ]:


plain_text.tail()


# In[ ]:


cipher_text.tail()


# In[ ]:


cipher_text.difficulty.value_counts()


# Let's test the amount of information contained in the text.

# In[ ]:


plain_text['word_list'] = plain_text.text.str.split()
plain_text['word_num'] = plain_text['word_list'].map(len)
plain_text.head(10)


# In[ ]:


ciph_1 = cipher_text[cipher_text.difficulty == 1]
ciph_1.set_index('ciphertext_id',inplace=True)
ciph_1.drop(columns = 'difficulty')


# In[ ]:


ciph_2 = cipher_text[cipher_text.difficulty == 2]
ciph_2.set_index('ciphertext_id',inplace=True)
ciph_2.drop(columns = 'difficulty')


# In[ ]:


ciph_3 = cipher_text[cipher_text.difficulty == 3]
ciph_3.set_index('ciphertext_id',inplace=True)
ciph_3.drop(columns = 'difficulty')


# In[ ]:


ciph_4 = cipher_text[cipher_text.difficulty == 4]
ciph_4.set_index('ciphertext_id',inplace=True)
ciph_4.drop(columns = 'difficulty')


# In[ ]:


ciph_1['ciphertextlist'] = ciph_1.ciphertext.str.split()
ciph_1.head(10)


# In[ ]:


ciph_2['ciphertextlist'] = ciph_2.ciphertext.str.split()
ciph_2.head(10)


# In[ ]:


ciph_3['ciphertextlist'] = ciph_3.ciphertext.str.split()
ciph_3.head(10)


# In[ ]:


ciph_4['ciphertextlist'] = ciph_4.ciphertext.str.split()
ciph_4.head(10)


# In[ ]:


plain_text['word_num'].value_counts()


# In[ ]:


ciph_1['length'] = ciph_1.ciphertextlist.map(len)
ciph_1.length.value_counts()


# In[ ]:


ciph_2['length'] = ciph_2.ciphertextlist.map(len)
ciph_2.length.value_counts()


# In[ ]:


ciph_3['length'] = ciph_3.ciphertextlist.map(len)
ciph_3.length.value_counts()


# In[ ]:


ciph_4['length'] = ciph_4.ciphertextlist.map(len)
ciph_4.length.value_counts()

