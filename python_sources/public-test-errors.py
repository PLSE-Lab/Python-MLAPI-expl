#!/usr/bin/env python
# coding: utf-8

# # 27 public test videos fail on conversion to PIL image

# I had several spare submissions so I tried to hunt down the public test errors that were plaguing the village. The conclusion so far is that there are 27 corrupted videos in the public test, and the failures for me are specifically at `Image.fromarray` call. Some more details below.

# In[ ]:


import glob
import cv2
from PIL import Image
import pandas as pd
import numpy as np


# In[ ]:


filenames = glob.glob('/kaggle/input/deepfake-detection-challenge/test_videos/*.mp4')


# In[ ]:


print(len(filenames))
count_failed = 0
for filename in filenames:
    try:
        v_cap = cv2.VideoCapture(filename)
        v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for j in range(v_len):
            success = v_cap.grab()
            if j == (v_len-1):
                success, vframe = v_cap.retrieve()
                vframe = Image.fromarray(vframe) # this line fails for 27 videos
        v_cap.release()
    except:
        count_failed += 1

submission = pd.read_csv("/kaggle/input/deepfake-detection-challenge/sample_submission.csv")
submission['label'] = 0.01 + 0.003*count_failed
submission.to_csv('submission.csv', index=False)


# In the code above I am submitting the same values for all videos, and we know that there are exactly 2000 fakes and 2000 real videos in the public test, so the expected score we can calculate with this function:

# In[ ]:


def score(x):
    return np.floor(1e5*(-0.5*np.log(x) - 0.5*np.log(1-x)))/1e5


# In[ ]:


for x in [0.01, 0.01+26*0.003, 0.01+27*0.003, 0.01+28*0.003]:
    print('submitting', x, 'scores', score(x))


# This kernel scores `1.24615`, so we conclude that 27 of the videos have crashed. The score is back to `2.30761` if one remarks `vframe = Image.fromarray(vframe)` line.
