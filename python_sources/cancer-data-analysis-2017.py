#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # EXPLORATORY DATA ANALYSIS OF CANCER IN USA, 2017

# This dataset constitutes the information of nine different cancers of Brain/ nervous system, Female breast, Colon and rectum, Leukemia, Liver, Lung and Bronchus, Non Hodgkin Lymphoma, Ovary, Pancreas, Prostrate in 51 states of America, during the year 2017. 
# 
# Since, there is not description of the values in the dataset, I assume it to be the new incidents of different cancer types in various states suring the year 2017 in USA.
# 
# The present analysis will report the findings through statistical analysis and various graph plotting to gain insight on the incidents of cancer throughout United States of America in 2017.

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import matplotlib.style as style
from scipy import stats


# # 1. Data description and representation

# In[ ]:


cancer_data = pd.read_csv('../input/cancer-data-2017/cancer2017.csv', encoding='ISO-8859-2')


# In[ ]:


cancer_data.info()


# This clearly shows that the data is complete with no null values

# In[ ]:


cancer_data.head(10)


# In[ ]:


#cancer_data.head(10)
cancer_data


# In[ ]:


cancer_data.isnull()


# ## **1. Data Cleaning and Preparation**
# 
# i) Although the dataset shows that the dataset is complete, but upon visualizing showed few missing values. Hence, the dataset requires cleaning. These missing values represent either no reported incidents or values not obtained. Either ways, the missing values requires to be handled.
# 
# ii) The datatype of value is presented as object, hence requires conversion to numbers 

# In[ ]:


# Handling the missing values represented by a symbol "?" in the dataset by replacing with NA
cancer_data.replace({r'[^\x00-\x7F]+':np.nan}, regex=True, inplace=True)


# In[ ]:


cancer_data.head()


# In[ ]:


cancer_data.info()


# In[ ]:


cancer_data.columns


# In[ ]:


cancer_data.shape


# In[ ]:


#removing commas
for i in range(0,51):
    for j in range(1,11):
        if ',' in str(cancer_data.iloc[i][j]):
            cancer_data.iloc[i][j]=cancer_data.iloc[i][j].replace(',','')
cancer_data.head()


# In[ ]:


# converting the datatypes of the dataset

cancer_data_num=cancer_data.apply(pd.to_numeric, errors='ignore')
cancer_data_num.info()


# In[ ]:


# Preparing a working dataset
cd = cancer_data_num


# In[ ]:


cd


# ## 2. STATISTICAL DATA ANALYSIS

# In[ ]:


cd2 = cd.dropna()
cd2.shape


# In[ ]:


cd2


# In[ ]:


# Computing the Descriptive statistics for the cancer data

cd_stat = cd2.describe()


# In[ ]:


cd_stat


# In[ ]:


Mean = cd_stat.iloc[1,:]
Mean


# In[ ]:


Mean.min()


# In[ ]:


colnames = []
for col in cd_stat.columns:
    colnames.append(col)
    
colnames


# In[ ]:


fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111)

# For Labels 
colnames = []
for col in cd_stat.columns:
    colnames.append(col)

# Plotting mean and standard deviation
x = np.arange(1, 11, 1)
Mean = cd_stat.iloc[1,:]
Stdev = cd_stat.iloc[2,:]

#style.use('fivethirtyeight')
plt.errorbar(x, Mean, Stdev, linestyle='None', marker='o', markersize = 10, capsize = 3)

ax.set_xticks(np.arange(len(colnames)+1))
ax.set_xticklabels([' ','Brain/ nervous system',
 'Female breast',
 'Colon & rectum',
 'Leukemia',
 'Liver',
 'Lung & bronchus',
 'Non-Hodgkin Lymphoma',
 'Ovary',
 'Pancreas',
 'Prostate'])
plt.xticks(rotation=70)
plt.title('Cancer incidents reported across US - Mean and Standard deviation')
plt.xlabel('Cancer Types')
plt.ylabel('Reported Incidents')

plt.grid()
plt.show()


# This shows that the reported cases for Lung and Bronchus are relatively high than the other cancer types in USA although the standard deviation is high.

# Box plots to show the distribution of the data

# In[ ]:


f, ax = plt.subplots(figsize=(11, 6))
sns.set(style="whitegrid")

# Draw a violinplot with a narrower bandwidth than the default
sns.violinplot(data=cd2, palette="Set3", bw=.2, cut=1, linewidth=1)
plt.xticks(rotation=70)

plt.title('Cancer incidents reported across US - Mean and Standard deviation')
plt.xlabel('Cancer Types')
plt.ylabel('Reported Incidents')

plt.grid()
plt.show()


# Plotting the density plots will provide information on the distribution of cancer cases across USA

# In[ ]:


plt.rcParams['figure.figsize'] = (20.0, 10.0)
plt.rcParams['font.family'] = "serif"
cd2.plot(kind='bar', stacked=True)

# For Labels 
labels = []
for r in cd2.iloc[:,0]:
    labels.append(r)
    
plt.xticks(np.arange(45), labels, rotation=70)
plt.title('Cancer incidents reported across US - Mean and Standard deviation')
plt.xlabel('Cancer Types')
plt.ylabel('Reported Incidents')


# This plot shows that the total lung and bronchus cancer cases are high in number, expecially in California, Florida, New York, Ohio and Texas 

# In[ ]:




cd2['Total'] = cd2.sum(axis = 1)
cd2['Total'].plot(kind = 'barh', color = 'r')

# For Labels 
labels = []
for r in cd2.iloc[:,0]:
    labels.append(r)

plt.yticks(np.arange(45), labels, rotation=1)
plt.title('Total Cancer incidents reported in different states across US')
plt.xlabel('Total Cancer cases')
plt.ylabel('States of America')


# In[ ]:


data = cd2.iloc[:,1:-1]
sns.clustermap(data, metric="correlation", cmap="Oranges")

plt.title('Heatmap of Cancer cases reported in different states across US')


# From the above heatmap, it is observed that the incidents of Lung and Bronchus cance, liver, Prostrate, Pancreas, leukemia, Female breast and Colon & rectum cancers have similar patterns and hence checking their correlation through pairplot might shed light on occurence of the incidents.

# In[ ]:


# Pairplots to study the correlation between the above mentioned incidents of cancers inn USA, 2017

data = cd2[['Brain/ nervous system', 'Female breast', 'Leukemia', 'Liver', 'Lung & bronchus', 'Pancreas', 'Prostate']]

# with regression
sns.pairplot(data, kind="reg")
plt.show()


# The above clearly shows that there is a correlation between the incidents in cancer report in various states of USA.

# Similarly, plotting of the reported cases in other states is also done

# In[ ]:


# Pairplots to study the correlation between the above mentioned incidents of cancers inn USA, 2017

data = cd2[['Colon & rectum', 'Non-Hodgkin Lymphoma', 'Ovary']]

# with regression
sns.pairplot(data, kind="reg")
plt.show()


# It is observed that there is a correlation between the reported cancer cases of Colon & rectum, Non-Hodgkin Lymphoma and Ovary in different states of USA

# In[ ]:




