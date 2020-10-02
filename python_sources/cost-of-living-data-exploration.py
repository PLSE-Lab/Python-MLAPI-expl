#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import pandas as pd
cost_of_living = pd.read_csv("../input/cost-of-living/cost-of-living.csv",index_col=0)
 


# In[ ]:


a= cost_of_living.T.reset_index()
a = a.rename(columns={'index':'City'})
a.head()


# In[ ]:


a.info()


# In[ ]:


import seaborn as sns
sns.lineplot(x="City",y="Meal, Inexpensive Restaurant",data=a) 


# In[ ]:


a.describe()


# In[ ]:


c=a.sort_values("McMeal at McDonalds (or Equivalent Combo Meal)")
c


# In[ ]:


Meal_greater_than15 = a[a["Meal, Inexpensive Restaurant"]>20
                       ]
Meal_greater_than15


# In[ ]:


table=pd.pivot_table(a, values='Meal, Inexpensive Restaurant', index=["City"],
                     aggfunc=np.sum)
table


# In[ ]:


a.max(axis = 0)


# In[ ]:


Max = a[a["Domestic Beer (0.5 liter draught)"]==11.13
                       ]
Max


# In[ ]:


Max = a[a["McMeal at McDonalds (or Equivalent Combo Meal)"]==12.97
                       ]
Max


# In[ ]:


b=a.iloc[3:35,0:3]
b


# In[ ]:


import seaborn as sns
sns.heatmap(b)


# In[ ]:


sns.clustermap(b)


# In[ ]:


sns.pairplot(b)


# In[ ]:


sns.lineplot(data=b)


# In[ ]:


sns.countplot( x="Onion (1kg)",data=a)


# In[ ]:


x=a["Tomato (1kg)"]
y=a["Banana (1kg)"]
sns.jointplot(x, y, kind="hex", color="#4CB391")


# In[ ]:


e=a.iloc[30:35,10:14]
g = sns.PairGrid(e, diag_sharey=False)

g.map_upper(sns.scatterplot)
g.map_lower(sns.kdeplot, colors=None)
g.map_diag(sns.kdeplot, lw=2)


# In[ ]:


import seaborn as sns
sns.set(style="ticks")
sns.pairplot(b, hue="City")

