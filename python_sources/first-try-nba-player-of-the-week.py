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
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
df= pd.read_csv('../input/NBA_player_of_the_week.csv',)
#display(df)
#correlation map
f,ax = plt.subplots(figsize=(18, 18))
sns.set(font_scale=1)
sns.heatmap(df.corr(), annot=True, linewidths=1.5, fmt= '.2f' ,ax=ax)
plt.show()


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
df= pd.read_csv('../input/NBA_player_of_the_week.csv',
                usecols=['Age','Seasons in league','Position'])
df.head(20)

plt.figure(figsize=(16,6))
sns.boxplot(x=df['Position'],y= df['Age'],palette='gist_earth')
plt.show()

plt.figure(figsize=(16,6))
sns.boxplot(x=df['Position'],y= df['Seasons in league'],palette='gist_earth')
plt.show()


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
df= pd.read_csv('../input/NBA_player_of_the_week.csv',
                usecols=['Age','Seasons in league','Position'])
df.head(20)

df_Age = df['Age']
Q1 = df_Age.quantile(.25)
Q3 = df_Age.quantile(.75)
IIQ = Q3 - Q1
Down_limit = Q1 - 1.5 * IIQ
UP_limit = Q3 + 1.5 * IIQ
df_Age_Select = (df_Age >= Down_limit) & (df_Age <= UP_limit)
df_Age2 = df_Age[df_Age_Select]

plt.figure(figsize=(16,6))
sns.boxplot(x=df['Position'],y= df_Age2,palette='gist_earth')
plt.show()


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
df= pd.read_csv('../input/NBA_player_of_the_week.csv',
                usecols=['Age','Seasons in league','Position'])
df.head(20)

df_Seasons_in_leadue = df['Seasons in league']
Q1 = df_Seasons_in_leadue.quantile(.25)
Q3 = df_Seasons_in_leadue.quantile(.75)
IIQ = Q3 - Q1
Down_limit = Q1 - 1.5 * IIQ
UP_limit = Q3 + 1.5 * IIQ
df_Seasons_in_leadue_Select = ((df_Seasons_in_leadue >= Down_limit) 
                               & (df_Seasons_in_leadue <= UP_limit))
df_Seasons_in_leadue_2 = df_Seasons_in_leadue[df_Seasons_in_leadue_Select]


plt.figure(figsize=(16,6))
sns.boxplot(x=df['Position'],y= df_Seasons_in_leadue_2,palette='gist_earth')
plt.show()

