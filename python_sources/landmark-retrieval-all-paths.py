#!/usr/bin/env python
# coding: utf-8

# # Load Libraries

# In[ ]:


import pandas as pd
import cv2, os
from tqdm import tqdm
import numpy as np


# In[ ]:


train = pd.read_csv("../input/landmark-retrieval-2020/train.csv")


# # Define a function that can read all paths.

# In[ ]:


def get_paths(sub):
    index = ["0","1","2","3","4","5","6","7","8","9","a","b","c","d","e","f"]

    paths = []

    for a in index:
        for b in index:
            for c in index:
                try:
                    paths.extend([f"../input/landmark-retrieval-2020/{sub}/{a}/{b}/{c}/" + x for x in os.listdir(f"../input/landmark-retrieval-2020/{sub}/{a}/{b}/{c}")])
                except:
                    pass

    return paths


# # Get the training paths

# In[ ]:


train_path = train

rows = []
for i in tqdm(range(len(train))):
    row = train.iloc[i]
    path  = list(row["id"])[:3]
    temp = row["id"]
    row["id"] = f"../input/landmark-retrieval-2020/train/{path[0]}/{path[1]}/{path[2]}/{temp}.jpg"
    rows.append(row["id"])
    
rows = pd.DataFrame(rows)
train_path["id"] = rows


# **Save paths without KFold**

# In[ ]:


train_path.to_csv("train_paths.csv",index = False)


# # Apply KFold with 10 Folds

# In[ ]:


from sklearn.model_selection import KFold

kf = KFold(n_splits=10,shuffle = True,random_state = 15)
kf.get_n_splits(train_path)

folds = np.zeros(len(train_path))

for fold, (train_index, test_index) in enumerate(kf.split(train_path)):
    folds[list(train_index)] = fold
    
train_path["Fold"] = folds.astype(int)


# **Save it.**

# In[ ]:


train_path.to_csv("train_paths_KFold.csv",index = False)
pd.DataFrame(get_paths("test")).to_csv("test_paths.csv",index = False)
pd.DataFrame(get_paths("index")).to_csv("index_paths.csv",index = False)

