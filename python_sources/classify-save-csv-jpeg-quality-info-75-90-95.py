#!/usr/bin/env python
# coding: utf-8

# This notebook aims to classify all JPG images by its quality (75, 90, 95) and save it into a csv file.
# 

# In[ ]:


get_ipython().system('apt-get update')
get_ipython().system('apt -y install imagemagick')


# In[ ]:


from typing import List
from glob import glob
import os
import pandas as pd

working_dir: str = "../input/alaska2-image-steganalysis/"

    
training_images: List[str] = list(glob(os.path.join(working_dir, "*", "*.jpg")))
df_train = pd.DataFrame({"file_path": training_images})
df_train["image"] = df_train["file_path"].apply(lambda x: os.path.basename(x))
df_train["kind"] = df_train["file_path"].apply(lambda x: os.path.split(os.path.dirname(x))[-1])
df_train


# In[ ]:


from typing import Optional

def func(file_path: str) -> Optional[float]:
    output = os.popen(f"identify -format '%Q' {file_path}").read()
    return output


# In[ ]:


from multiprocessing import Pool
with Pool(4) as p:
    ret = list(p.map(func, training_images))


# In[ ]:


df_train["quality"] = ret
df_train["quality"] = df_train["quality"].astype("float")
df_train.sort_values("image", inplace=True)
df_train[["image", "kind", "quality"]].to_csv("image_quality.csv", index=False, float_format='%.0f')


# In[ ]:


df_train["quality"].value_counts(normalize=False)

