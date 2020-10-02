#!/usr/bin/env python
# coding: utf-8

# This kernel produces a simple average of the three best public submissions. 
# 
# It is possible to add new submission files and change the threshold. The latter is expressed in the form of "minimum number of times that a pixel has been counted as positive in order to be included in the final prediction". 
# 
# At the moment, the final average solution does not produce a better score than the individual best submission.

# In[ ]:


import numpy as np
import pandas as pd
import os
from glob import glob
import sys
import skimage.measure
from tqdm import tqdm, tqdm_notebook

sys.path.insert(0, os.path.join("/", "kaggle", "input", "siim-acr-pneumothorax-segmentation"))
from mask_functions import rle2mask, mask2rle


# In[ ]:


# read all submissions into daframes and store them in a list
df_sub_list = [pd.read_csv(f) for f in glob(os.path.join("/", "kaggle", "input", "*", "*.csv"))]


# In[ ]:


# create a list of unique image IDs
iid_list = df_sub_list[0]["ImageId"].unique()
print(f"{len(iid_list)} unique image IDs.")


# Create average prediction mask for each image

# In[ ]:


# set here the threshold for the final mask
# min_solutions is the minimum number of times that a pixel has to be positive in order to be included in the final mask
min_solutions = 3 # a number between 1 and the number of submission files
assert (min_solutions >= 1 and min_solutions <= len(df_sub_list)),     "min_solutions has to be a number between 1 and the number of submission files"


# In[ ]:


# create empty final dataframe
df_avg_sub = pd.DataFrame(columns=["ImageId", "EncodedPixels"])
df_avg_sub_idx = 0 # counter for the index of the final dataframe

# iterate over image IDs
for iid in tqdm_notebook(iid_list):
    # initialize prediction mask
    avg_mask = np.zeros((1024,1024))
    # iterate over prediction dataframes
    for df_sub in df_sub_list:
        # extract rles for each image ID and submission dataframe
        rles = df_sub.loc[df_sub["ImageId"]==iid, "EncodedPixels"]
        # iterate over rles
        for rle in rles:
            # if rle is not -1, build prediction mask and add to average mask
            if "-1" not in str(rle):
                avg_mask += rle2mask(rle, 1024, 1024) / float(len(df_sub_list))
    # threshold the average mask
    avg_mask = (avg_mask >= (min_solutions * 255. / float(len(df_sub_list)))).astype("uint8")
    # extract rles from the average mask
    avg_rle_list = []
    if avg_mask.max() > 0:
        # label regions
        labeled_avg_mask, n_labels = skimage.measure.label(avg_mask, return_num=True)
        # iterate over regions, extract rle, and save to a list
        for label in range(1, n_labels+1):
            avg_rle = mask2rle((255 * (labeled_avg_mask == label)).astype("uint8"), 1024, 1024)
            avg_rle_list.append(avg_rle)
    else:
        avg_rle_list.append("-1")
    # iterate over average rles and create a row in the final dataframe
    for avg_rle in avg_rle_list:
        df_avg_sub.loc[df_avg_sub_idx] = [iid, avg_rle]
        df_avg_sub_idx += 1 # increment index


# In[ ]:


df_avg_sub.shape


# In[ ]:


df_avg_sub["ImageId"].nunique()


# In[ ]:


df_avg_sub.head()


# In[ ]:


df_avg_sub.to_csv("average_submission.csv", index=False)

