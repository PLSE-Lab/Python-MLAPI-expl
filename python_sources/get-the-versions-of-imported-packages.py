#!/usr/bin/env python
# coding: utf-8

# Knowing exactly what version of packages you're using in your work (especially in a fast-moving field like machine learning!) is a crucial part of debugging. This is a short example of one way to check the version of the packages you've imported.

# In[17]:


# import some packages to check
import numpy as np 
import pandas as pd
import tensorflow as tf


# The code below is from [this answer](https://stackoverflow.com/questions/40428931/package-for-listing-version-of-packages-used-in-a-jupyter-notebook) on Stack Overflow provdied by [Alex P. Miller](https://stackoverflow.com/users/2628402/alex-p-miller).

# In[18]:


import pkg_resources
import types
def get_imports():
    for name, val in globals().items():
        if isinstance(val, types.ModuleType):
            # Split ensures you get root package, 
            # not just imported function
            name = val.__name__.split(".")[0]

        elif isinstance(val, type):
            name = val.__module__.split(".")[0]

        # Some packages are weird and have different
        # imported names vs. system names
        if name == "PIL":
            name = "Pillow"
        elif name == "sklearn":
            name = "scikit-learn"

        yield name
imports = list(set(get_imports()))

requirements = []
for m in pkg_resources.working_set:
    if m.project_name in imports and m.project_name!="pip":
        requirements.append((m.project_name, m.version))

for r in requirements:
    print("{}=={}".format(*r))

