#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


train_data = pd.read_csv("https://s3.amazonaws.com/google-landmark/metadata/train.csv")


# In[3]:


landmarks = train_data['landmark_id'].value_counts().sort_values(ascending=False).to_frame(name="Landmark's samples qty")


# In[4]:


percentiles = list(range(10, 110, 10))


# <h1>Landmark's qty distribution</h1>

# <h2>All landmarks</h2>

# In[5]:


hist = landmarks.hist()


# <h2>Landmarks with samples qty < 100</h2>

# In[6]:


hist = landmarks[landmarks<100].hist()


# <h2>Landmarks with samples qty < 50</h2>

# In[7]:


hist = landmarks[landmarks<50].hist()


# <h2>Landmark's qty percentiles</h2>

# In[8]:


pd.DataFrame(np.percentile(landmarks, percentiles), index=percentiles, columns = ["Landmark's samples qty"])


# <h3>50 % of landmarks have less then 10 samples</h3>

# <h1>Landmark's samples qty distribution</h1>

# In[16]:


landmarks_qty = train_data['landmark_id'].map(landmarks["Landmark's samples qty"]).to_frame(name="Landmark's samples qty")


# <h2>All landmarks</h2>

# In[17]:


hist = landmarks_qty.hist()


# <h2>Landmarks with samples qty < 100</h2>

# In[18]:


hist = landmarks_qty[landmarks_qty<100].hist()


# <h2>Landmarks with samples qty < 50</h2>

# In[19]:


hist = landmarks_qty[landmarks_qty<50].hist()


# <h2>Landmark's qty percentiles</h2>

# In[20]:


pd.DataFrame(np.percentile(landmarks_qty, percentiles), index=percentiles, columns = ['Landmark images qty'])


# <h3>50 % of samples belong to landmarks 
# which have less then 50 samples</h3>

# <h1>90 % of landmarks have less then 50 samples and contain 50% of train data</h1>
