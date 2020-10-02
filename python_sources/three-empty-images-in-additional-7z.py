#!/usr/bin/env python
# coding: utf-8

# In[ ]:




get_ipython().run_line_magic('matplotlib', 'inline')
import os
import numpy as np 
import pandas as pd 
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.misc import imresize

for base_dir in ["train", "additional"]:
    print(base_dir)    
    type1 = glob.glob("../input/{}/Type_1/*.jpg".format(base_dir))
    type2 = glob.glob("../input/{}/Type_2/*.jpg".format(base_dir))
    type3 = glob.glob("../input/{}/Type_3/*.jpg".format(base_dir))
    print("Type 1:", len(type1))
    print("Type 2:", len(type2))
    print("Type 3:", len(type3))
    print("--")
plt.imshow(np.hstack([imresize(plt.imread(x), (320,240)) for x in type1[:5]]))


# In[ ]:


plt.imshow(np.hstack([imresize(plt.imread(x), (320,240)) for x in type2[:5]]))


# In[ ]:


plt.imshow(np.hstack([imresize(plt.imread(x), (320,240)) for x in type3[:5]]))


# In[ ]:


for base_dir in ["train", "additional"]:
    for x in glob.glob("../input/{}/**/*.jpg".format(base_dir)):
        s = os.stat(x)
        if s.st_size == 0:
            print("WARNING: corrupted file: ", x)


# In[ ]:




