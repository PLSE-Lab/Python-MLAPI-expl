#!/usr/bin/env python
# coding: utf-8

# # Example video clips

# ## Read metadata

# In[ ]:


import glob
import os
import json

        
metafiles = sorted(glob.glob('/kaggle/input/*/part*/*/*.json'))
metafile = metafiles[0]
print(metafile)

dirname = os.path.dirname(metafile)
with open(metafile) as fd:
    meta = json.load(fd)
    
real = next(iter(meta))
print(real)


# ## Play an original

# In[ ]:


from IPython.display import HTML
from base64 import b64encode

def play(filenames):
    html = ''
    for name in filenames:
        video = open(name,'rb').read()
        src = 'data:video/mp4;base64,' + b64encode(video).decode()
        html += '<video width=352 controls autoplay loop><source src="%s" type="video/mp4"></video>' % src 
    return HTML(html)

real_path = os.path.join(dirname, real)
play([real_path])


# ## Play the corresponding fakes

# In[ ]:


fakes = meta[real]
print(fakes)
fake_paths = [os.path.join(dirname, fake) for fake in fakes]
play(fake_paths)


# ## Example with multiple faces in the same video

# In[ ]:


metafiles = sorted(glob.glob('/kaggle/input/*/multi-part*/*/*.json'))
metafile = metafiles[0]
print(metafile)

dirname = os.path.dirname(metafile)
with open(metafile) as fd:
    meta = json.load(fd)
    
real = next(iter(meta))
print(real)


# In[ ]:


real_path0 = os.path.join(dirname, real[:-5] + '0.mp4')
real_path1 = os.path.join(dirname, real[:-5] + '1.mp4')
play([real_path1, real_path0])

