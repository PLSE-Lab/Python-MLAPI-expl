#!/usr/bin/env python
# coding: utf-8

# Hello everyone!
# 
# Today I'd like to show you that not all of splitting methods are good for our tasks.

# In[ ]:


get_ipython().system('pip install iterative-stratification')
get_ipython().system('pip install scikit-multilearn')


# In[ ]:


import os
import time
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from skmultilearn.model_selection import IterativeStratification


# In[ ]:


DATASET_PATH = "/kaggle/input/bengaliai-cv19/"
KEYS = ["grapheme_root", "vowel_diacritic", "consonant_diacritic"]
TEST_SIZE = 0.1
SEED = 69


# In[ ]:


def get_csv(dataset_path, name):
    return pd.read_csv(os.path.join(dataset_path, f"{name}.csv"))


def count(df):
    return df.groupby(KEYS).size().reset_index().rename(columns={0: "size"})


def split(dataset_path, test_size, stratification):
    df = get_csv(dataset_path, name="train")
    img_ids = df["image_id"]

    if stratification == "sklearn_random":
        train_set, valid_set = train_test_split(df[KEYS], test_size=test_size,
                                                random_state=SEED, shuffle=True)
    elif stratification == "sklearn_stratified":
        splitter = StratifiedShuffleSplit(n_splits=1,
                                          test_size=test_size,
                                          random_state=SEED)

        train_indcs, valid_indcs = next(splitter.split(X=img_ids, y=df[KEYS]))
        train_set = df.loc[df.index.intersection(train_indcs)].copy()
        valid_set = df.loc[df.index.intersection(valid_indcs)].copy()
        
    elif stratification == "iterstrat":

        splitter = MultilabelStratifiedShuffleSplit(n_splits=1,
                                                    test_size=test_size,
                                                    random_state=SEED)

        train_indcs, valid_indcs = next(splitter.split(X=img_ids, y=df[KEYS]))
        train_set = df.loc[df.index.intersection(train_indcs)].copy()
        valid_set = df.loc[df.index.intersection(valid_indcs)].copy()

    elif stratification == "skmultilearn":
        
        splitter = IterativeStratification(n_splits=2, order=2, 
                                           sample_distribution_per_fold=[
                                               test_size, 1.0-test_size])
        
        train_indcs, valid_indcs = next(splitter.split(X=img_ids, y=df[KEYS]))
        train_set = df.loc[df.index.intersection(train_indcs)].copy()
        valid_set = df.loc[df.index.intersection(valid_indcs)].copy()
        
    else:
        raise ValueError("Try something else :)")

    return train_set, valid_set

def eval(train, valid):
    
    train_count, val_count = count(train), count(valid)
    
    total = train_count["size"] + val_count["size"]
    train_part = train_count["size"] / total
    val_part = val_count["size"] / total
    relative = val_part / train_part
    
    
    for k, v in {"Train": train_part, "Valid": val_part, 
                 "Valid relative to train": relative}.items():
        print("-------------------------------------------------------------------")
        print(k)
        print(v)
        print(",".join([f"{m}: {f(v):.2}" 
                        for m, f in {"min": np.min, "max": np.max, 
                                     "mean": np.mean, "std": np.std}.items()]))
        print("-------------------------------------------------------------------")


# Let's try the most famous random train_test_split and look what are the partitions for every class triplet.

# In[ ]:


method = "sklearn_random"


start = time.time()

train, valid = split(dataset_path=DATASET_PATH, 
                     test_size=TEST_SIZE, 
                     stratification=method)

print(f"Dataset split done for {time.time() - start} seconds")


# In[ ]:


eval(train, valid)


# As you can see, some validation triplets represent from 3% to 20% of the whole set, which seems to me not very stratified :)
# 
# Let's see how other method works.

# In[ ]:


method = "iterstrat"

start = time.time()

train, valid = split(dataset_path=DATASET_PATH, 
                     test_size=TEST_SIZE, 
                     stratification=method)

print(f"Dataset split done for {time.time() - start} seconds")


# In[ ]:


eval(train, valid)


# Next method works veeeeery slow and also wrong, that's why I deleted it, but I left the code for you to try.

# In[ ]:


# method = "skmultilearn"

# start = time.time()

# valid, train = split(dataset_path=DATASET_PATH, 
#                      test_size=TEST_SIZE, 
#                      stratification=method)

# print(f"Dataset split done for {time.time() - start} seconds")


# And last method which stratify dataset which as for me suits much better.

# In[ ]:


method = "sklearn_stratified"

start = time.time()

train, valid = split(dataset_path=DATASET_PATH, 
                     test_size=TEST_SIZE, 
                     stratification=method)

print(f"Dataset split done for {time.time() - start} seconds")


# In[ ]:


eval(train, valid)


# As you can see only last method with sklearn.StratifiedShuffleSplit splits train and val in appropriate stratified sizes.
