#!/usr/bin/env python
# coding: utf-8

# ## I've listened to the entire test set
# 
# A few observations:
# 
# 1. ** There's a lot of silent recordings (not silence) **
# 
#     We can normalize the recordings, but is it really a good idea? I mean, we'll lose the information that is in my opinion present, that keyboard writing is quieter than shouting. This is almost a rule, but there are some "pre-normalized" samples.
# 
#     One should definitely take care of that phenomena. 
# 
# 2. ** Test set is similar to train_curated **
# 
#     You can easily verify this. 
# 
# 3. ** Test set is not similar to train_noisy **
# 
#     These recordings are indeed out of the domain. The compression, differs a lot, source sampling rate were different (I mean recordings are oversampled)
# 
# 4. ** Test set is specific **
# 
#     There is a lot of harmonicas, accordion, guitar sounds, but none of the f.e violin. There are American Italian, and some Spanish speakers, but there's no let's say Polish. 
#     
#     What I mean is the test set is not trying to represent all the sounds in the world, it's focused on some specific sounds.
#     Maybe the intention was to choose sounds that can be confused.

# In[ ]:


import os
from scipy.io import wavfile
import IPython.display as ipd


# ## Bonus - the weirdest things I've heard recently
# 
# Among the test examples, I noted some test examples that "turned me in my ears".
# 
# ** Sorry if anyone founds it inappropriate**
# 
# I don't know if there's any ML value in that, but here we go:

# In[ ]:


chosen_test = ["0b73077e.wav", "0bb86b6c.wav", "1a1edda6.wav", "1a2580e1.wav", "1a362723.wav", "1c6d56db.wav", "1d6f2b16.wav", "1d2459ce.wav", "1dcc4dd4.wav",
               "1e8fe3d2.wav", "2d7ae271.wav", "2d21dbf4.wav", "2df0cf1f.wav", "2ee9f139.wav", "2ee3ef28.wav", "07da8ebf.wav", "08b06642.wav", "10e6db66.wav", "18fda5a4.wav",
               "25e23528.wav", "26c01e3f.wav", "34d0b2ca.wav", "34d4cfda.wav", "34f2c99d.wav", "34f2c99d.wav", "36beb3fd.wav", "36c459de.wav", "37c95c1c.wav", "38e43355.wav",
               "38f6aa5e.wav", "040c5c0c.wav", "40dd1274.wav", "41af688a.wav", "065b49ec.wav", "084e5784.wav", "191adc5f.wav", "280c07e4.wav", "395df625.wav", "0837e041.wav",
               "2462f236.wav", "2733bb3f.wav", "4037cd65.wav", "07603ad3.wav", "20801c98.wav", "12264142.wav", "35930583.wav"]
chosen_test = iter(chosen_test)


# **"ear rape warning" - as suggested by Khoi Nguyen in the comments - turn down the speakers!**
# 

# In[ ]:


ipd.Audio(wavfile.read(os.path.join("../input/test", next(chosen_test)),)[1], rate=44100)


# In[ ]:


ipd.Audio(wavfile.read(os.path.join("../input/test", next(chosen_test)),)[1], rate=44100)


# In[ ]:


ipd.Audio(wavfile.read(os.path.join("../input/test", next(chosen_test)),)[1], rate=44100)


# In[ ]:


ipd.Audio(wavfile.read(os.path.join("../input/test", next(chosen_test)),)[1], rate=44100)


# In[ ]:


ipd.Audio(wavfile.read(os.path.join("../input/test", next(chosen_test)),)[1], rate=44100)


# In[ ]:


ipd.Audio(wavfile.read(os.path.join("../input/test", next(chosen_test)),)[1], rate=44100)


# In[ ]:


ipd.Audio(wavfile.read(os.path.join("../input/test", next(chosen_test)),)[1], rate=44100)


# In[ ]:


ipd.Audio(wavfile.read(os.path.join("../input/test", next(chosen_test)),)[1], rate=44100)


# In[ ]:


ipd.Audio(wavfile.read(os.path.join("../input/test", next(chosen_test)),)[1], rate=44100)


# In[ ]:


ipd.Audio(wavfile.read(os.path.join("../input/test", next(chosen_test)),)[1], rate=44100)


# In[ ]:


ipd.Audio(wavfile.read(os.path.join("../input/test", next(chosen_test)),)[1], rate=44100)


# In[ ]:


ipd.Audio(wavfile.read(os.path.join("../input/test", next(chosen_test)),)[1], rate=44100)


# In[ ]:


ipd.Audio(wavfile.read(os.path.join("../input/test", next(chosen_test)),)[1], rate=44100)


# In[ ]:


ipd.Audio(wavfile.read(os.path.join("../input/test", next(chosen_test)),)[1], rate=44100)


# In[ ]:


ipd.Audio(wavfile.read(os.path.join("../input/test", next(chosen_test)),)[1], rate=44100)


# In[ ]:


ipd.Audio(wavfile.read(os.path.join("../input/test", next(chosen_test)),)[1], rate=44100)


# In[ ]:


ipd.Audio(wavfile.read(os.path.join("../input/test", next(chosen_test)),)[1], rate=44100)


# In[ ]:


ipd.Audio(wavfile.read(os.path.join("../input/test", next(chosen_test)),)[1], rate=44100)


# In[ ]:


ipd.Audio(wavfile.read(os.path.join("../input/test", next(chosen_test)),)[1], rate=44100)


# In[ ]:


ipd.Audio(wavfile.read(os.path.join("../input/test", next(chosen_test)),)[1], rate=44100)


# In[ ]:


ipd.Audio(wavfile.read(os.path.join("../input/test", next(chosen_test)),)[1], rate=44100)


# In[ ]:


ipd.Audio(wavfile.read(os.path.join("../input/test", next(chosen_test)),)[1], rate=44100)


# In[ ]:


ipd.Audio(wavfile.read(os.path.join("../input/test", next(chosen_test)),)[1], rate=44100)


# In[ ]:


ipd.Audio(wavfile.read(os.path.join("../input/test", next(chosen_test)),)[1], rate=44100)


# In[ ]:


ipd.Audio(wavfile.read(os.path.join("../input/test", next(chosen_test)),)[1], rate=44100)


# In[ ]:


ipd.Audio(wavfile.read(os.path.join("../input/test", next(chosen_test)),)[1], rate=44100)


# In[ ]:


ipd.Audio(wavfile.read(os.path.join("../input/test", next(chosen_test)),)[1], rate=44100)


# In[ ]:


ipd.Audio(wavfile.read(os.path.join("../input/test", next(chosen_test)),)[1], rate=44100)


# In[ ]:


ipd.Audio(wavfile.read(os.path.join("../input/test", next(chosen_test)),)[1], rate=44100)


# In[ ]:


ipd.Audio(wavfile.read(os.path.join("../input/test", next(chosen_test)),)[1], rate=44100)


# In[ ]:


ipd.Audio(wavfile.read(os.path.join("../input/test", next(chosen_test)),)[1], rate=44100)


# In[ ]:


ipd.Audio(wavfile.read(os.path.join("../input/test", next(chosen_test)),)[1], rate=44100)


# In[ ]:


ipd.Audio(wavfile.read(os.path.join("../input/test", next(chosen_test)),)[1], rate=44100)


# In[ ]:


ipd.Audio(wavfile.read(os.path.join("../input/test", next(chosen_test)),)[1], rate=44100)


# In[ ]:


ipd.Audio(wavfile.read(os.path.join("../input/test", next(chosen_test)),)[1], rate=44100)


# In[ ]:


ipd.Audio(wavfile.read(os.path.join("../input/test", next(chosen_test)),)[1], rate=44100)


# In[ ]:


ipd.Audio(wavfile.read(os.path.join("../input/test", next(chosen_test)),)[1], rate=44100)


# In[ ]:


ipd.Audio(wavfile.read(os.path.join("../input/test", next(chosen_test)),)[1], rate=44100)


# In[ ]:


ipd.Audio(wavfile.read(os.path.join("../input/test", next(chosen_test)),)[1], rate=44100)


# In[ ]:


ipd.Audio(wavfile.read(os.path.join("../input/test", next(chosen_test)),)[1], rate=44100)


# In[ ]:


ipd.Audio(wavfile.read(os.path.join("../input/test", next(chosen_test)),)[1], rate=44100)


# In[ ]:


ipd.Audio(wavfile.read(os.path.join("../input/test", next(chosen_test)),)[1], rate=44100)


# In[ ]:


ipd.Audio(wavfile.read(os.path.join("../input/test", next(chosen_test)),)[1], rate=44100)


# In[ ]:


ipd.Audio(wavfile.read(os.path.join("../input/test", next(chosen_test)),)[1], rate=44100)


# In[ ]:


ipd.Audio(wavfile.read(os.path.join("../input/test", next(chosen_test)),)[1], rate=44100)


# In[ ]:


ipd.Audio(wavfile.read(os.path.join("../input/test", next(chosen_test)),)[1], rate=44100)


# In[ ]:




