#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('cp /kaggle/input/decord/install.sh . && chmod  +x install.sh && ./install.sh ')


# In[ ]:


import os, sys
sys.path.insert(0,'/kaggle/working/reader/python')

from decord import VideoReader
from decord import cpu, gpu
from decord.bridge import set_bridge
import glob


# In[ ]:


set_bridge('torch')


# In[ ]:


filenames = glob.glob('../input/deepfake-detection-challenge/test_videos/*.mp4')


# ### GPU version. Return torch.uint8 located on gpu. Have proble with memory

# In[ ]:


get_ipython().run_cell_magic('time', '', '## Be carefull GPU memory leak\nshapes = []\nfor filename in filenames:\n    video = VideoReader(filename, ctx=gpu())\n    data = video.get_batch(range(len(video)))\n    shapes += [data.size()]\n    del video, data')


# ### CPU version. Return torch.uint8

# In[ ]:


get_ipython().run_cell_magic('time', '', '## slower but stable\nshapes = []\nfor filename in filenames:\n    video = VideoReader(filename, ctx=cpu())\n    data = video.get_batch(range(len(video)))\n    shapes += [data.size()]\n    del video, data')


# In[ ]:


get_ipython().system('rm -r reader && rm install.sh')


# In[ ]:


get_ipython().system('cp /kaggle/input/deepfake-detection-challenge/sample_submission.csv ./submission.csv')

