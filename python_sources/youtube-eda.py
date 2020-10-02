#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as mp
path = "../input/USvideos.csv"
df = pd.read_csv(path)
df.boxplot('views','likes')


# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as mp
path = "../input/USvideos.csv"
df = pd.read_csv(path)
df.plot.scatter("views","likes")




# In[2]:


import pandas as pd
import numpy as np
import statistics as s
path = "../input/USvideos.csv"
df = pd.read_csv(path)
x=df["views"]
print("Mean : ",s.mean(x))
print("Median : ",s.median(x))
print("Mode : ",s.mode(x))


# In[ ]:


import pandas as pd
path = "../input/USvideos.csv"
df = pd.read_csv(path)
print(df)

