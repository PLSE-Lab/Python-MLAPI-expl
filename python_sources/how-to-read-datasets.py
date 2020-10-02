#!/usr/bin/env python
# coding: utf-8

# Introduction
# ------------
# 
# This is a quick way to look at the different Datasets....

# In[ ]:



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# First, look at everything.
from subprocess import check_output
print(check_output(["ls", "../input/data"]).decode("utf8"))


# In[ ]:


#  Pick a Dataset you might be interested in.
#  Say, all airline-safety files...
import zipfile

Dataset = "airline-safety"

# Will unzip the files so that you can see them..
with zipfile.ZipFile("../input/data/"+Dataset+".zip","r") as z:
    z.extractall(".")


# In[ ]:


from subprocess import check_output
print(check_output(["ls", "airline-safety"]).decode("utf8"))


# In[ ]:


# There's only one file above...we'll select it.
d=pd.read_csv(Dataset+"/airline-safety.csv")
d.head()

