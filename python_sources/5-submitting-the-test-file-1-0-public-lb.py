#!/usr/bin/env python
# coding: utf-8

# # 5. 1.0 submission: submitting the test file
# ### Airbus Ship Detection Challenge - A quick overview for computer vision noobs
# 
# &nbsp;
# 
# Hi, and welcome! This is the fifth kernel of the series `Airbus Ship Detection Challenge - A quick overview for computer vision noobs.` This short and trivial kernel creates a 1.0 submission using the data shared by the Challenge organizers in response to the discovery of an important data leak.
# 
# 
# The full series consist of the following notebooks:
# 1. [Loading and visualizing the images](https://www.kaggle.com/julian3833/1-loading-and-visualizing-the-images)
# 2. [Understanding and plotting rle bounding boxes](https://www.kaggle.com/julian3833/2-understanding-and-plotting-rle-bounding-boxes) 
# 3. [Basic exploratory analysis](https://www.kaggle.com/julian3833/3-basic-exploratory-analysis)
# 4. [Exploring public models](https://www.kaggle.com/julian3833/4-exploring-models-shared-by-the-community)
# 5. *[1.0 submission: submitting the test file](https://www.kaggle.com/julian3833/5-1-0-submission-submitting-the-test-file)*
# 
# This is an ongoing project, so expect more notebooks to be added to the series soon. Actually, we are currently working on the following ones:
# * Understanding and exploiting the data leak
# * A quick overview of image segmentation domain
# * Jumping into Pytorch
# * Understanding U-net
# * Proposing a simple improvement to U-net model

# ## 1. Context

# As we will discuss in a *future notebook* of the series, a severe [data leakage](https://www.kaggle.com/c/airbus-ship-detection/discussion/64355) was reported on August 28th, trivializing the full competition. The organizers [recognized the leak ](https://www.kaggle.com/c/airbus-ship-detection/discussion/64388) very fast and proposed a plan to address the situation. Following that plan, they [released the segmentations for the test set](https://www.kaggle.com/c/airbus-ship-detection/discussion/64702) and will soon provide a new test set and reset the scoreboard.
# 
# Until this reset, we can trivially generate a 1.0 scoring submission just resending the `test_ship_segmentations.csv` file as follows.

# ## 2. Submission

# In[ ]:


import os
import pandas as pd

test_files = [f for f in os.listdir("../input/test/")]
df = pd.read_csv("../input/test_ship_segmentations.csv")
df = df[df['ImageId'].isin(test_files)].drop_duplicates(subset="ImageId")
df.to_csv("submission.csv", index=False)
len(df)


# In[ ]:


get_ipython().system('head submission.csv')


# ### References
# * [Severe leak](https://www.kaggle.com/c/airbus-ship-detection/discussion/64355)
# * [Test Segmentations Now Available](https://www.kaggle.com/c/airbus-ship-detection/discussion/64702)
# * [Data Leak and Next Steps](https://www.kaggle.com/c/airbus-ship-detection/discussion/64388)
# 
# 
# ### What's next?
# This is the last notebook  of the series for now. Stay tuned, we are currently working on a notebook which aims to understand and exploit the leak (not submitting the shared solution but generating it from the leakage itself)!

# In[ ]:




