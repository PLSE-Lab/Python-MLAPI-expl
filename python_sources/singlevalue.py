#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Let p be probability of fake.
#When 0.4 is assign to all output we get a result of: 0.71355
#So p log 0.4 + (1-p) log 0.6 = -0.71355
#Or p log (0.6667) = 0.71355 - log 0.6 = -0.20272437623
#or p = 0.499979831 That is on final dataset there is a 50-50 split of fake and real videos!


# In[ ]:





# In this notebook, I am simply making a submission which assigns 0.5 probability for each video.
# Sadly when I submit it it errors: "Status: Submission Scoring Error"
# 
# I believe it has something to do with the way I am taking input.
# 
# 
# Could someone please helpm me fix this?
# 
# Thanks

# In[ ]:


import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
#print("Starting")
#for dirname, _, filenames in os.walk('./'):
#    for filename in filenames:
#        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

import zipfile
import csv


# In[ ]:


testVidPath = "/kaggle/input/deepfake-detection-challenge/test_videos/"


# In[ ]:


mylist = []
for _, _, files in os.walk(testVidPath):
    for fName in files:
        if not fName.endswith("mp4"):
            continue
        vidId = fName.split("/")[-1]#.split(".")[0]
        mylist.append(vidId)


# In[ ]:


df = pd.DataFrame(data={"filename": mylist, "label": [0.4]*len(mylist)})
df.to_csv("./submission.csv", sep=',',index=False)


# In[ ]:




