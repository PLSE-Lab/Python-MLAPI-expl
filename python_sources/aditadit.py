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


# 

# In[ ]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
df = pd.read_csv('/kaggle/input/uisummerschool/Online_sales.csv')
df


# In[ ]:


df.info()
korelasi = df[['Date','Avg. Price']]


# In[ ]:


df.describe()


# In[ ]:


df.head()


# In[ ]:


df.tail()


# In[ ]:


korelasi


# 

# In[ ]:


korelasi.hist()


# In[ ]:


korelasi.plot(kind='line', x='Date')


# **Insight**: Dalam periode 1 tahun, dapat dicari bagaimana kecenderungan orang "berani" mengeluarkan uang pada setiap bulannya.
# Dari diagram ini dapat dilihat bahwa orang-orang cenderung "berani" dan "mampu" untuk mengeluarkan uang yang lebih besar pada masa akhir tahun. Hal ini kemungkinan besar karena terdapat banyak acara besar di akhir tahun, seperti libur panjang, tahun baru, natal, thanksgiving (di luar negeri), dll.
# Namun, orang-orang cenderung "berhemat" ketika masa pertengahan tahun. Hal ini mungkin disebabkan banyaknya kebutuhan pada pertengahan tahun, seperti biaya masuk kuliah, sekolah (elementary school & high school), dll.
# 
# **Action**: Kurang tepat apabila menyediakan stok dan menjual barang yang harganya mahal pada masa pertengahan tahun. Strategi yang dapat dilakukan untuk menjual barang yang harganya tinggi yaitu dengan menyiapkan stok dan menjualnya di masa akhir tahun.
