#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

plt.style.use('bmh')


df = pd.read_csv('../input/wages-data/Griliches.csv')
df.head()


# In[ ]:


# library
 import matplotlib.pyplot as plt
 import numpy as np

 plt.stem(df.age)
 plt.show()
 
 plt.stem(df.tenure)
 plt.show()


 plt.stem(df.age, df.tenure)
 plt.show()


# In[ ]:


(markerline, stemlines, baseline) = plt.stem(df.age,df.tenure)
plt.setp(baseline, visible=False)
plt.show()


# In[ ]:


# library
import matplotlib.pyplot as plt
import numpy as np
 
# plot with no marker
plt.stem(df.expr, markerfmt=' ')
#plt.show()
 
# change color and shape and size and edges
(markers, stemlines, baseline) = plt.stem(df.expr)
plt.setp(markers, marker='D', markersize=10, markeredgecolor="orange", markeredgewidth=2)
#plt.show()

