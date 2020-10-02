#!/usr/bin/env python
# coding: utf-8

# ## Introduction
# 
# In this notebook, one of the last techniques that we applied is shown. A simple post-processing technique, as described in the associated [post](https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification/discussion/160986). The score gain is relatively low compared to other techniques we applied. However, it gave a steady increase (~0.0001) for each of languages es/tr/fr/ru both in public LB as private LB. This also secured our first place.
# 
# 
# Here, I present an example of how to use our earlier Russian subs to achieve the gain in score: going from public LB 9549 to 9550, and private LB 9532 to 9533. This be done in similar fashion with the other languages.

# # Imports

# In[ ]:


# General imports
import numpy as np
import pandas as pd
import os


# In[ ]:


# Specify lang
LANG = "ru"
DIR = f"../input/{LANG}-changed-subs/"
WEIGHT = 1 # we kept WEIGHT between 1-2


# In[ ]:


submission = pd.read_csv("../input/jigsaw-multilingual-toxic-comment-classification/sample_submission.csv")
test = pd.read_csv("../input/jigsaw-multilingual-toxic-comment-classification/test.csv")
sub_best = pd.read_csv(os.path.join(DIR, "sub-LB-9549.csv"))


# In[ ]:


files_sub = os.listdir(DIR)
files_sub = sorted(files_sub)
print(len(files_sub))
files_sub


# In[ ]:


for file in files_sub:
    test[file.replace(".csv", "")] = pd.read_csv(os.path.join(DIR, file))["toxic"]


# In[ ]:


test = test.loc[test["lang"]==LANG].reset_index(drop=True)
test.head(1)


# In[ ]:


# Derive the given sub increases or decreases in score
test["diff_good1"] = test[f"{LANG}-9397"] - test[f"{LANG}-9373"]
test["diff_good2"] = test[f"{LANG}-9476"] - test[f"{LANG}-9475"]
test["diff_good3"] = test[f"{LANG}-9529"] - test[f"{LANG}-9510"]
test["diff_good4"] = test[f"{LANG}-9544"] - test[f"{LANG}-9543"]

test["diff_bad1"] = test[f"{LANG}-9545"] - test[f"{LANG}-9543-from-9545"]


# In[ ]:


test["sub_best"] = test["sub-LB-9549"]
col_comment = ["id", "content", "sub_best"]
col_diff = [column for column in test.columns if "diff" in column]
test_diff = test[col_comment + col_diff].reset_index(drop=True)

test_diff["diff_avg"] = test_diff[col_diff].mean(axis=1) # the mean trend


# In[ ]:


# Apply the post-processing technique in one line (as explained in the pseudo-code of my post.
test_diff["sub_new"] = test_diff.apply(lambda x: (1+WEIGHT*x["diff_avg"])*x["sub_best"] if x["diff_avg"]<0 else (1-WEIGHT*x["diff_avg"])*x["sub_best"] + WEIGHT*x["diff_avg"] , axis=1)


# In[ ]:


submission["toxic"] = sub_best["toxic"]
submission.loc[test["id"], "toxic"] = test_diff["sub_new"].values
submission.to_csv("submission.csv", index=False)

