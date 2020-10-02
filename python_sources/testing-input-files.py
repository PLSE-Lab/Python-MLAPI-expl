#!/usr/bin/env python
# coding: utf-8

# Here's a quick demonstration of where and how to read in all of the wav files from `../input/notes`.

# In[ ]:


from os import listdir
from os.path import isfile, join
from scipy.io import wavfile


# In[ ]:


path_to_files = "../input/notes"
sound_files = [f for f in listdir(path_to_files) if isfile(join(path_to_files, f))]

print(sound_files)


# In[ ]:


# testing reading in one of the wav files
sampFreq, X = wavfile.read(path_to_files + "/" + sound_files[1])


# It seems like there could be an issue going on with the wav file headers. It would be great if someone could demo how to read them.
