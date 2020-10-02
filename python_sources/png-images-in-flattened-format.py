#!/usr/bin/env python
# coding: utf-8

# # Image Format

# Images can be found in attached datasets. They are flattened so you no longer need to do `**/*`

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import random
import os
from PIL import Image
print(os.listdir("../input"))


# In[ ]:


print(os.listdir("../input/siim-png-images/input"))


# In[ ]:


Image.open(os.path.join("../input/siim-png-images/input/train_png", random.choice(os.listdir("../input/siim-png-images/input/train_png"))))


# In[ ]:


Image.open(os.path.join("../input/siim-png-images/input/train_png", random.choice(os.listdir("../input/siim-png-images/input/train_png"))))


# In[ ]:


Image.open(os.path.join("../input/siim-png-images/input/train_png", random.choice(os.listdir("../input/siim-png-images/input/train_png"))))


# In[ ]:


Image.open(os.path.join("../input/siim-png-images/input/train_png", random.choice(os.listdir("../input/siim-png-images/input/train_png"))))


# In[ ]:




