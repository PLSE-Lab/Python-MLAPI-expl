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


import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import requests as rq
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# Reading data #
urban_data = pd.read_csv("../input/urban_dictionary.csv", sep=',')
urban_data.head()


# In[ ]:


# removing unnessery colomns #
urban_data.drop(['tags','date'], axis=1, inplace=True)


# In[ ]:


# plot b/w words vs up-vote distribution #
fig=plt.figure()
ax=fig.add_subplot(1,1,1)
ax.hist(urban_data['up'],bins=7)
plt.title('up- vote distribution')
plt.xlabel('up')
plt.ylabel('word')
plt.show()


# In[ ]:


# plot b/w words vs down-vote distribution #
fig=plt.figure()
ax=fig.add_subplot(1,1,1)
ax.hist(urban_data['down'],bins=7)
plt.title('down_vote distribution')
plt.xlabel('down')
plt.ylabel('word')
plt.show()


# In[ ]:


var=urban_data.groupby('author').up.mean()
fig=plt.figure()
ax1=fig.add_subplot(1,1,1)
ax1.set_xlabel('author')
ax1.set_ylabel('mean of up-votes')
ax1.set_title("author wise mean of up votes")
var.plot(kind='bar')
plt.show()
plt.show(max('author'))


# In[ ]:


fig=plt.figure()
ax=fig.add_subplot(1,1,1)
ax.scatter(urban_data['down'],urban_data['up'])
plt.xlabel('down')
plt.ylabel('up')
plt.show()


# In[ ]:


#Columna con la diferencia entre likes y dislikes
urban_data['diff_likes'] = urban_data['up'] - urban_data['down']
urban_data.head()
urban_data['diff_likes']


# In[ ]:


urban_data.sort_values(['diff_likes'], ascending=True).head()


# In[ ]:


#Boolean column if (up>=down): True
urban_data['positive_likes'] = urban_data['up'] > urban_data['down']
urban_data[['word','up','down','positive_likes']].head(30)


# In[ ]:


# Total number of true positive likes & False positive likes#
urban_data_positive_likes = urban_data['positive_likes'].value_counts()
urban_data_positive_likes

