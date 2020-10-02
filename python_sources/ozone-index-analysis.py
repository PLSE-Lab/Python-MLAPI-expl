#!/usr/bin/env python
# coding: utf-8

# <h1><center>Ozone Index Analysis</center></h1>

# <h3> In this notebook I will try to explain the change in ozone layer and ozone index transformation during the last century. After the lockdown, a notable pattern is visible in transformation of the ozone indices and thus this historical data is a good referrence for the present situation, in order to when we must be the most aware to protect the ozone layer and by what means </h3>

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt #for all graphical and plotting purposes
import seaborn as sns


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # 1. Pre-processing data from dataframe

# <h4>In this segment we will reshape the available data in such a manner that it is easy to use and understand. Here, I will remove unnecessary columns, null values (if any) and do all the required modifications to make the data-model a good fit</h4> 

# In[ ]:


df = pd.read_csv('../input/ozone-data/ozone.csv')
df.head()


# In[ ]:


ndf = df.drop(['Unnamed: 0'], axis=1)
print(ndf.head())


# # 2. Comparing a monthly analysis of data for each pair of months 

# <h4>Comparison of Ozone indices from between January and February</h4>

# In[ ]:


sns.lmplot(x="Jan",y="Feb",data=ndf)


# <h4>Comparison of Ozone indices from between February and March</h4>

# In[ ]:


sns.lmplot(x="Feb",y="Mar",data=ndf)


# <h4>Comparison of Ozone indices from between March and April</h4>

# In[ ]:


sns.lmplot(x="Mar",y="Apr",data=ndf)


# # 3. Boxplot demonstrating relationship between ozone indices of two consequetive months

# In[ ]:


plt.figure(figsize=(30,20))
plt.subplot(2,2,1)
sns.boxplot(x="Aug",y="Sep",data=ndf)
plt.subplot(2,2,2)
sns.boxplot(x="Sep",y="Oct",data=ndf)
plt.subplot(2,2,3)
sns.boxplot(x="Oct",y="Nov",data=ndf)
plt.subplot(2,2,4)
sns.boxplot(x="Nov",y="Dec",data=ndf)


# <h4>The pink boxes in first two graphs denote the time when the maximum variances in ozone layer density took place.</h4>
# <h4>The green and blue boxplots in last two figures denote their subsequent densities of ozone</h4>

# # 4. Area plot of January to April

# In[ ]:


ndf.plot.area(y=['Jan','Feb','Mar','Apr'],alpha=0.4,figsize=(16,8))


# <h4>We can see an anomaly in the density of ozone layer in April. This is because in April, the temperature soars up and so the Ozone layer detriorates due to increased CFC emmision in atmosphere from air-conditioners and refrigerators</h4>

# # 5. Jointplot of Year and Annual distribution of ozone indices

# In[ ]:


g = sns.jointplot(x="Year", y="Annual", data=ndf, kind="kde", color="m")
g.plot_joint(plt.scatter, c="w", s=30, linewidth=1, marker="+")
g.ax_joint.collections[0].set_alpha(0)
g.set_axis_labels("$Year$", "$Annual$");


# <h4>This shows how the intensity of ozone layer depletion has risen in past 70-80 years, after industrial revolution and mass production of high heat and chemical emitting machines, increased use of coolant gases like ammonia and Chloro-fluro-carbon(CFC) after the invention and rapid use of air-conditioner and refrigerators in past 2 decades.</h4>

# # 6.Correlation and Heatmap of all data of Ozone indices

# In[ ]:


print(ndf.corr())


# In[ ]:


plt.subplots(figsize=(16,16))
sns.heatmap(ndf.corr(),annot = True,fmt="f").set_title("Correlation of Ozone Indices")
plt.show()


# <h4>The heatmap and correlational plotted data signifies how the ozone indices are related to the respective months of a year. Shown in orane boxes, the weather in April to August having a correlational constant less than 0.5 signifies the highest affected season of Ozone layer depletion.</h4>

# # From the above analysis of ozone layer indexing and layer health, we can easily notice that  the most harm is caused to the ozone layer during the months of April to August, when the temperature in tropical regions soar up and thus people tend to use more cooling devices in homes, offices and industries. These cooling equipments like air-conditioners(in homes, vehicles and offices), cold-storages, refrigerators cause a huge emission of CFC which harms the ozone layer. 

# # The best solution is to plant trees and reduce global warming, (a similar situation is observed in 2020 due to lockdowns and less vehicular emmissions).

# In[ ]:




