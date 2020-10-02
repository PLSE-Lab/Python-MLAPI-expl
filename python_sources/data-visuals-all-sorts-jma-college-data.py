#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 
import os 
import pandas as pd

dat = pd.read_csv('../input/college-data/data.csv')
dat.head()


# In[ ]:


sns.lmplot(x="apps", y="accept",  data=dat);


# In[ ]:


sns.lmplot(x="accept", y="grad_rate",  data=dat);


# In[ ]:


f, ax = plt.subplots(figsize=(6, 6))
sns.kdeplot(dat.top10perc, dat.top25perc, ax=ax)
sns.rugplot(dat.top10perc, color="g", ax=ax)
sns.rugplot(dat.top25perc, vertical=True, ax=ax);


# In[ ]:


f, ax = plt.subplots(figsize=(6, 6))
cmap = sns.cubehelix_palette(as_cmap=True, dark=0, light=1, reverse=True)
sns.kdeplot(dat.top10perc, dat.top25perc, cmap=cmap, n_levels=60, shade=True);


# In[ ]:


import matplotlib.pyplot as plt
import numpy as np

plt.scatter(dat.top10perc, dat.top25perc, c="red", alpha=0.4)


# In[ ]:


sns.boxplot( y=dat["apps"] )


# In[ ]:


sns.boxplot( x=dat["accept"] )


# In[ ]:


sns.boxplot(data=dat.ix[:,1:4])


# In[ ]:


sns.pairplot(dat, kind="reg")
plt.show()


# In[ ]:


sns.pairplot(dat, kind="scatter")
plt.show()


# In[ ]:


from matplotlib import pyplot as plt
import pandas as pd
import numpy as np


# In[ ]:


import seaborn as sns
plt.plot( 'apps', 'accept', data=dat, marker='o', color='mediumvioletred')
plt.show()

