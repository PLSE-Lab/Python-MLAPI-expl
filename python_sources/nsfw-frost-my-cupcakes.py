#!/usr/bin/env python
# coding: utf-8

# ## Found in the `train_curated` set
# 
# ## !Warning Not Safe For Work!

# In[ ]:


import os
from scipy.io import wavfile
import IPython.display as ipd


# In[ ]:


ipd.Audio(wavfile.read("../input/train_curated/7f409e1a.wav")[1], rate=44100)

