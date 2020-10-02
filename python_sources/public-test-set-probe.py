#!/usr/bin/env python
# coding: utf-8

# # What's wrong with this submission
# I forked [this kernel](https://www.kaggle.com/zaharch/public-test-errors) from @zaharch. I changed the last part to confirm if all of  corrupted videos are fake. But always got the "Submission Scoring Error".
# **Anyone knowns the reason. Thanks.**

# # 27 public test videos fail on conversion to PIL image

# I had several spare submissions so I tried to hunt down the public test errors that were plaguing the village. The conclusion so far is that there are 27 corrupted videos in the public test, and the failures for me are specifically at `Image.fromarray` call. Some more details below.

# In[ ]:


import os
import glob
import cv2
from PIL import Image
import pandas as pd
import numpy as np


# In[ ]:


filenames = glob.glob('/kaggle/input/deepfake-detection-challenge/test_videos/*.mp4')


# In[ ]:


submission = pd.read_csv("/kaggle/input/deepfake-detection-challenge/sample_submission.csv", index_col=0)
submission['label'] = 0.5


# In[ ]:


print(len(filenames))
for filename in filenames:
    try:
        name = os.path.basename(filename)
        v_cap = cv2.VideoCapture(filename)
        v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for j in range(v_len):
            success = v_cap.grab()
            if j == (v_len-1):
                success, vframe = v_cap.retrieve()
                vframe = Image.fromarray(vframe) # this line fails for 27 videos
        v_cap.release()
        submission.loc[name, 'label'] = 0.5
    except:
        submission.loc[name, 'label'] = 1


# In[ ]:


submission.to_csv('submission.csv')


# In the code above I am submitting the same values for all videos, and we know that there are exactly 2000 fakes and 2000 real videos in the public test, so the expected score we can calculate with this function:

# In[ ]:




