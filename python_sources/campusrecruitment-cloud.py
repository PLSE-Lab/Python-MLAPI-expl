#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


'/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv'


# In[ ]:


df = pd.read_csv('/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')
df.head()


# In[ ]:


df.info()


# # Rata-rata gaji berdasarkan jurusan 

# In[ ]:


#pake mean & grouping, tipe : bar

gaji = df.groupby(['degree_t'])['salary'].mean()

plt.bar(gaji.index, gaji)
plt.xlabel("Jurusan")
plt.ylabel('Gaji')

plt.show()


# # Jumlah murid yang ketrima kerja berdasarkan gender

# In[ ]:


#
sns.countplot(df['status'], hue=df['gender'])


# # Jumlah mahasiswa yang memiliki tipe spesialisasi tertentu MBA berdasarkan degree nya

# In[ ]:


sns.countplot(df['specialisation'], hue=df['degree_t'])


# # Jurusan yang diambil mahasiswa berdasarkan spesialisasi pendidikan secondary mereka 

# In[ ]:


sns.countplot(y="degree_t", hue="hsc_s", data=df)


#  # Jumlah mahasiswa yang dipecah berdasarkan jurusan yang diambil

# In[ ]:


#barh + count
mhs_jurusan = df.groupby(['degree_t'])['gender'].count()

plt.barh(mhs_jurusan.index, mhs_jurusan)

plt.ylabel("Jurusan")
plt.xlabel('Jumlah Mahasiswa')

plt.show()


# # Grafik line hasil tes kemampuan kerja yang diurutkan berdasarkan nomer seri murid

# In[ ]:


# pakai plot
df.plot(x = 'sl_no', y = 'etest_p')


# # awh

# In[ ]:


fig, ax = plt.subplots()
df.groupby(['gender']).plot(x = 'sl_no', y = 'mba_p', ax=ax)


# # Grafik besar gaji yang didapat berdasarkan nilai etest

# In[ ]:


fig, ax = plt.subplots()

ax.scatter(df['etest_p'], df['salary'])


# # Perbandingan besar gaji yang didapat berdasarkan nilai etest dengan nilai mba_p

# In[ ]:


fig, ax = plt.subplots()

ax.scatter(df['etest_p'], df['salary'], color='coral')
ax.scatter(df['mba_p'], df['salary'], color='lightgreen')


