#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
print("Setup Complete")

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        datafile_path = os.path.join(dirname, filename)
        data = pd.read_csv(datafile_path)

# Any results you write to the current directory are saved as output.
data.head()


# **Depict the first 5 records of the data.**

# In[ ]:


print("Number of unique resgistrations: ", data["Registration"].nunique())
print("Number of unique names of companies or individuals: ", data["Name"].nunique())
print("Number of unique national IDs of companies or individuals: ", data["National_ID"].nunique())


# **It could be interpreted as follow:**
# * Each company or individual could have submitted more than one registration.
# * Several company (or maybe individual) names could belong to a certain national ID.

# In[ ]:


plt.figure(figsize=(18,12))
plot = sns.scatterplot(x=data["Registration"], y=data["Amount"]*data["Currency_to_rial"]/10000000, hue=data["Currency"])
plot.set(xlabel='Registration Number', ylabel='Amount (unit: 10,000,000 Rials)')
plt.show()


# **Depict the distribution of registrations by their amount in Rials and colored based on the type of currency they have.**
# It can be understood that:
# * Most of outliers (those with a more than 1 Trillion Rials amount) are in the South Korean Won.
# * The second frequent outliers (More than 1 Trillion Rials) are in Euro.
# * The earliest and the latest registrations have more huge payments than the middle ones. (Assuming that the less a registration number is, the earlier it has been submitted.)
# 

# In[ ]:


amount_in_rial = []
for record in data.itertuples():
    amount_in_rial.append(record.Amount*record.Currency_to_rial/pow(10,7))
        
data["Amount_in_rial"] = amount_in_rial

plot = sns.catplot(x="Currency", y="Amount_in_rial", data = data, height = 10, aspect = 2)
plot.set(xlabel='Currency', ylabel='Amount (unit: 10,000,000 Rials)')
plt.show()


# **Depict the distribution of registrations by their amount in Rials, classified by the type of currency they have.**
# Possible conslusions are the same as those of the previous plot.

# In[ ]:


plt.figure(figsize=(24,8))
plot = sns.countplot(x="Currency", data=data)
plot.set(xlabel='Currency', ylabel='Number of registrations')
plt.show()


# **Depict number of registrations classified by the currency they demand.**
# It could be concluded that:
# * Most of the submitted registrations demand Euro.
# * The second most frequently demanded currency is China's Yuan.
# * The number of submitted registrations demanding USA Dollor is significantly small.
# 

# In[ ]:


plt.figure(figsize=(18,6))
amount_in_rial = []
for record in data.itertuples():
    amount_in_rial.append(record.Amount*record.Currency_to_rial/pow(10, 10))
        
data["Amount_in_rial"] = amount_in_rial

grouped_data = data.groupby('Currency', as_index = False).agg({'Amount_in_rial':'sum'})
grouped_data = grouped_data.reindex(index=grouped_data.index[::-1])

plot = sns.barplot(x="Currency", y="Amount_in_rial", data=grouped_data)
plot.set(xlabel='Currency', ylabel='Total Amount (unit: 10 Billion Rials)')
plt.show()


# **Depict how much of each currency has been registered.**

# In[ ]:


plt.figure(figsize=(18,6))
amount_in_rial = []
for record in data.itertuples():
    amount_in_rial.append(record.Amount*record.Currency_to_rial/pow(10, 7))
        
data["Amount_in_rial"] = amount_in_rial

correlated_data = data.corr()
sns.heatmap(correlated_data, annot=False)


# **Depict the correlation between any pair of features.**
# Noticeable correlations could be listed as follow:
# * The (time of submitting a ) registration does not have a significant relationship with demanded amount in rial.(Assuming that the less a registration number is, the earlier it has been submitted)
# * The later a registration is submitted, the more the value of demanded currency is. (Assuming that the less a registration number is, the earlier it has been submitted)
# * Naturally, the more the demanded amount is, the less the value of demanded currency is.
# * The more the demanded amount is, the more value it has in rial.
