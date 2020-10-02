#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd, numpy as np, librosa as lb
from pathlib import Path
from IPython.display import Audio
import warnings
warnings.filterwarnings("ignore") # Filter annoying librosa warnings


# In[ ]:


DATA_ROOT = Path("../input/birdsong-recognition")


# # The dataset

# In[ ]:


class BirdDataset:
    """Fastly load and sample the audio file in order to get same wave size for batch items.
    
    Parameters:
    ----------
    sr: int
        The sample rate, defaults to librosa's 22050 Hz.
        
    nseconds: int
        Targetted duration in seconds. The wave will right-padded if it lasts less than `nseconds`.
        This is useful when batching.
    """
    def __init__(self, sr = 22050, nseconds=5):
        self.sr = sr
        self.nseconds = nseconds
        self.df = pd.read_csv(DATA_ROOT/"train.csv")
        self.df.sort_values(["ebird_code", "filename"], inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, i):
        """Load the ith wave file."""
        x = self.load(self.ith_file(i))
        
        return  self.sample(x),BIRDS_MAP[self.df.loc[i, "ebird_code"]]
    
    
    def ith_file(self, i):
        row = self.df.loc[i]
        filename = "{}/{}".format(row["ebird_code"], row["filename"])
        return filename
    
    def load(self, filename, res_type = 'kaiser_best'):
        """Load the wave file by name."""
        filename = DATA_ROOT/"train_audio"/filename
        y, _ = lb.load(filename.as_posix(), sr = self.sr, res_type=res_type)
        return y
    
    def display(self, audio):
        return Audio(self.load(audio) if isinstance(audio, str) else audio, rate=self.sr)
    
    
    def sample(self, x):
        """Sample the wave file in order to make it last exactly `self.nseconds`.
        The wave will be right-padded if it's shorter.
        """
        max_frames = self.nseconds*self.sr
        nframes = len(x)
        if max_frames < nframes:
            offset = np.random.choice(nframes - max_frames)
            x = x[offset:offset + max_frames]
        elif max_frames>nframes:
            x = np.concatenate([np.concatenate([x]*(max_frames//nframes)), x[-max_frames%nframes:]])
        return x
    
    
    def sample_on_load(self,filename, duration, res_type = 'kaiser_best'):
        """Fastly and directly sample the wave file on load time in order to make it last 
        exactly `self.nseconds`. The wave will be right-padded if it's shorter.
        """
        target_duration = self.nseconds
        filename = DATA_ROOT/"train_audio"/filename
        
        if duration > target_duration:
            offset = np.random.choice(duration - target_duration)
            x, sr = lb.load(filename, offset=offset, duration=target_duration, res_type= res_type, sr= self.sr)
        else:
            x, sr = lb.load(filename, sr=self.sr)
            nframes = len(x)
            target_frames = self.nseconds*self.sr
            x = np.concatenate([np.concatenate([x]*(target_frames//nframes)), x[-target_frames%nframes:]])
        return x


# # Benchmark

# In[ ]:


bds = BirdDataset(nseconds=5) # Sample 5 seconds  lasting audio from wave files


# In[ ]:





# <h2 style="background-color:gray; color:white"> A half-minute long wave file </h2>

# In[ ]:


i = 105
row = bds.df.loc[i]
bird_file = "{}/{}".format(row.ebird_code, row.filename)
duration = row.duration
bird_file,duration


# In[ ]:


get_ipython().run_cell_magic('time', '', '\nx = bds.sample(bds.load(bird_file))\nprint(x.shape)\nbds.display(x)')


# In[ ]:


get_ipython().run_cell_magic('time', '', '\nx = bds.sample_on_load(bird_file, duration)\nprint(x.shape)\nbds.display(x)')


# > Sampling on laod is **4** times faster for a  **30 s** long wave file.

# In[ ]:





# <h2 style="background-color:gray; color:white"> A one-minute long wave file </h2>

# In[ ]:


i = 8390
row = bds.df.loc[i]
bird_file = "{}/{}".format(row.ebird_code, row.filename)
duration = row.duration
bird_file,duration


# In[ ]:


get_ipython().run_cell_magic('time', '', '\nx = bds.sample(bds.load(bird_file))\nprint(x.shape)\nbds.display(x)')


# In[ ]:


get_ipython().run_cell_magic('time', '', '\nx = bds.sample_on_load(bird_file, duration)\nprint(x.shape)\nbds.display(x)')


# > Sampling on laod is **7** times faster for a  **60 s** long wave file. Indeed, longer the wave and higher the gain !

# In[ ]:





# <h2 style="background-color:gray; color:white"> A 10 minutes long wave file </h2>

# In[ ]:


i = 1341
row = bds.df.loc[i]
bird_file = "{}/{}".format(row.ebird_code, row.filename)
duration = row.duration
bird_file,duration


# In[ ]:


get_ipython().run_cell_magic('time', '', '\nx = bds.sample(bds.load(bird_file))\nprint(x.shape)\nbds.display(x)')


# In[ ]:


get_ipython().run_cell_magic('time', '', '\nx = bds.sample_on_load(bird_file, duration)\nprint(x.shape)\nbds.display(x)')


# > Sampling on laod is about **25** times faster for a  **10  minutes** long wave file. As there are some waves longer than 30 minutes, sampling on load is a really good hack.

# In[ ]:





# <h2 style="text-align:center">kkiller</h2>

# In[ ]:




