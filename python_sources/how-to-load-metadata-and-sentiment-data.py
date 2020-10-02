#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import json
import numpy as np
import pandas as pd
from pathlib import Path
from pprint import pprint
input_path = Path("../input")


# In[ ]:


ls ../input


# In[ ]:


train = pd.read_csv(input_path.joinpath("train/train.csv"))
test = pd.read_csv(input_path.joinpath("test/test.csv"))


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


sample_id = train.PetID[0]
print(sample_id)


# In[ ]:


with open(input_path.joinpath("train_sentiment/{}.json".format(sample_id)), "r") as f:
    sample = json.load(f)


# In[ ]:


pprint(sample)


# In[ ]:


sample_meta_data = {}
for path in input_path.joinpath("train_images/").glob("{}*".format(sample_id)):
    with open(input_path.joinpath("train_metadata/{}.json".format(path.stem)), "r") as f:
        sample_meta_data[path.stem] = json.load(f)


# In[ ]:


pprint(sample_meta_data[path.stem])


# In[ ]:




