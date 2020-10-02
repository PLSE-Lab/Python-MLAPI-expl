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
games = pd.read_csv("../input/chess/games.csv")


# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


games


# In[ ]:


plt.hist(games["turns"], color="green", edgecolor="black", linewidth="2")
plt.xlabel("turns")
plt.ylabel("frequency")
plt.title("Turns per Game")


# The historgraph shows the number of turns per chess game. The range of the data set is 1-349 and the width is 348. The data has a center at 50 and it is skewed to the right.

# In[ ]:


import seaborn as sns

sns.boxplot(x="white_rating", data=games)


# The boxplot shows the frequencies of the ratings of the white peices. The data is centered around the 1500 mark, has a range of 750 to 2750, and has a width of 2000. The data is slightly skewed to the right, with some outliers near the 2750 mark. 

# In[ ]:


games["winner"].value_counts()


# In[ ]:


games['winner'].value_counts().head(10).plot.pie()

plt.gca().set_aspect('equal')


# The pie graph shows the frequncey that each side wins. The data is distrubuted pretty evenly with white taking up 50%, black taking up 40-45% and draw with 5-10%. 

# In[ ]:


labels="mate", "resign"
sizes=[11232.48,6418.56]

plt.bar(labels, sizes)


# The bar chart shows the results of chess matches. The data shows that more of the mathces end in a check mate than a resign, with check mate at around 11000 and resign around 6000. 

# In[ ]:


sns.boxplot(x="turns", y="victory_status", data=games)

