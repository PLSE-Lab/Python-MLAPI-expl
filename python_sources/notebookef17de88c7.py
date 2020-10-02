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

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns

df = pd.read_csv('../input/xAPI-Edu-Data.csv')
df.head()


# In[ ]:


print(df.shape)


# In[ ]:


df.isnull().sum()


# In[ ]:


import pandas as pd

df = pd.DataFrame()
a_group_desc = df.groupby('Class').describe()
print a_group_desc


# In[ ]:


import pandas as pd

df = pd.DataFrame()
a_group_desc = df.groupby('Class').describe()
print a_group_desc()


# In[ ]:


import pandas as pd

df = pd.DataFrame()
a_group_desc = df.groupby('Class').describe()
print (a_group_desc)


# In[ ]:


df = pd.DataFrame()
a_group_desc = df.groupby('Class').describe()
print (a_group_desc)


# In[ ]:


ax = sns.boxplot(x="Class", y="Discussion", data=df)
plt.show()


# In[ ]:


import mathplotlib as plt
ax = sns.boxplot(x="Class", y="Discussion", data=df)
plt.show()


# In[ ]:


import mathplotlib.pyplot as plt
ax = sns.boxplot(x="Class", y="Discussion", data=df)
plt.show()


# In[ ]:


import matplotlib.pyploy as plt
ax = sns.boxplot(x="Class", y="Discussion", data=df)
plt.show()


# In[ ]:


import matplotlib.pyplot as plt
ax = sns.boxplot(x="Class", y="Discussion", data=df)
plt.show()


# In[ ]:


import matplotlib.pyplot as plt
sns.countplot(x="Topic", data=df, palette="muted");
plt.show()

