#!/usr/bin/env python
# coding: utf-8

# I exported and merged my image blurrness score extraction. wish It would be helpful to everyone! 

# In[2]:


import pandas as pd
import numpy as np
import gc


# In[4]:


train = pd.read_csv('../input/avito-demand-prediction/train.csv')
test = pd.read_csv('../input/avito-demand-prediction/test.csv')

train_sorted = train.sort_values(by = ["image"])
test_sorted = test.sort_values(by = ["image"])

#Load image blurrness for train

tr_1 = pd.read_csv("../input/train-data-image-blurrness/1.csv")
tr_2 = pd.read_csv("../input/train-data-image-blurrness/2.csv")
tr_3 = pd.read_csv("../input/train-data-image-blurrness/3.csv")
tr_4 = pd.read_csv("../input/train-data-image-blurrness/4.csv")
tr_5 = pd.read_csv("../input/train-data-image-blurrness/5.csv")
tr_6 = pd.read_csv("../input/train-data-image-blurrness/6.csv")
tr_7 = pd.read_csv("../input/train-data-image-blurrness/7.csv")
tr_8 = pd.read_csv("../input/train-data-image-blurrness/8.csv")
tr_9 = pd.read_csv("../input/train-data-image-blurrness/9.csv")
tr_10 = pd.read_csv("../input/train-data-image-blurrness/10.csv")
tr_11 = pd.read_csv("../input/train-data-image-blurrness/11(12_13.5).csv")
tr_12 = pd.read_csv("../input/train-data-image-blurrness/last.csv")

frames = [tr_1, tr_2, tr_3, tr_4, tr_5, tr_6, tr_7, tr_8, tr_9, tr_10, tr_11, tr_12]
new = pd.concat(frames)
new["File"] = new["File"].apply(lambda x : x.split("/")[-1].split(".")[0])
new = new.sort_values(by = ["File"])
scores = list(new["Score"].values) + [-1] * (len(train)-len(new))
train_sorted["image_blurrness_score"] = scores
train = train_sorted.sort_index()


##Testing
te_1 = pd.read_csv("../input/image-blurrness-test/test_1.csv")
te_2 = pd.read_csv("../input/image-blurrness-test/test_2.csv")
te_3 = pd.read_csv("../input/image-blurrness-test/test_3.csv")
te_4 = pd.read_csv("../input/image-blurrness-test/test_4.csv")
te_5 = pd.read_csv("../input/image-blurrness-test/test_5.csv")

frames_te = [te_1, te_2, te_3, te_4, te_5]
new_te = pd.concat(frames_te)
new_te["File"] = new_te["File"].apply(lambda x : x.split("/")[-1].split(".")[0])
new_te = new_te.sort_values(by = ["File"])
scores_te = list(new_te["Score"].values) + [-1] * (len(test)-len(new_te))

test_sorted["image_blurrness_score"] = scores_te
test = test_sorted.sort_index()

train.head()


# In[9]:


train[["item_id", "image_blurrness_score"]].to_csv("train_blurrness.csv", index = False)
test[["item_id", "image_blurrness_score"]].to_csv("test_blurrness.csv", index = False)

