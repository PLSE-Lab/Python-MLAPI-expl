#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

ds = pd.read_csv("/kaggle/input/youtube-new/USvideos.csv")


# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


ds.tail(11)


# In[ ]:


ds.loc[ds.channel_title == 'The Deal Guy']


# In[ ]:


ds.iloc[5000]


# In[ ]:


ds.loc[ds.likes <= 5000000]


# In[ ]:


sum(ds.likes[ds.channel_title == 'iHasCupquake'])


# In[ ]:


plt.figure()
plt.title('Grafico del canal iHasCupquake')
y = ds.like[ds.channel_title == 'iHasCupquake']
x = ds.trending_date[ds.channel_title = 'iHasCupquake']
plt.plot(x,y)


# In[ ]:


import pandas as pd
CAvideos = pd.read_csv("../input/youtube-new/CAvideos.csv")
DEvideos = pd.read_csv("../input/youtube-new/DEvideos.csv")
FRvideos = pd.read_csv("../input/youtube-new/FRvideos.csv")
GBvideos = pd.read_csv("../input/youtube-new/GBvideos.csv")
INvideos = pd.read_csv("../input/youtube-new/INvideos.csv")
JPvideos = pd.read_csv("../input/youtube-new/JPvideos.csv")
KRvideos = pd.read_csv("../input/youtube-new/KRvideos.csv")
MXvideos = pd.read_csv("../input/youtube-new/MXvideos.csv")
RUvideos = pd.read_csv("../input/youtube-new/RUvideos.csv")
USvideos = pd.read_csv("../input/youtube-new/USvideos.csv")

