#!/usr/bin/env python
# coding: utf-8

# # Introduction
# * This kernel is notes that I took for Seaborn. This is not a tutorial you can think that kernel as a cheatsheet for Seaborn. While coding a kernel open this in a new tab and copy paste. 
# 
# ### This kernel is a part of a big series:
# * [Data Science Notes1: Matplotlib](https://www.kaggle.com/mrhippo/data-science-notes1-matplotlib)
# * Data Science Notes2: Seaborn 
# * [Data Science Notes3: Plotly](https://www.kaggle.com/mrhippo/data-science-notes3-plotly?scriptVersionId=38663418)
# * [Data Science Notes4: Machine Learning (ML)]()
# * [Data Science Notes5: Deep Learning: ANN](https://www.kaggle.com/mrhippo/data-science-notes5-deep-learning-ann) 
# * [Data Science Notes6: Deep Learning: CNN](https://www.kaggle.com/mrhippo/data-science-notes6-deep-learning-cnn) 
# * [Data Science Notes7: Deep Learning: RNN and LSTM](https://www.kaggle.com/mrhippo/data-science-notes7-deep-learning-rnn-and-lstm)
# 
# ### This kernel will be updated
# 
# ## Content:
# * [Imports and Datasets](#1)
# * [Normal Plots](#2)
# * [Styled Plots](#3)
# * [Conclusion](#4)

# <a id="1"></a> <br>
# # Imports and Datasets

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date, timedelta, datetime
import warnings
warnings.filterwarnings('ignore')
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


data1 = pd.read_csv("/kaggle/input/heart-disease-uci/heart.csv")
data1.head()


# In[ ]:


data2 = pd.read_csv("/kaggle/input/videogamesales/vgsales.csv")
data2.head()


# In[ ]:


data3 = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv")
data3.head()


# In[ ]:


data_last = data3.tail(1)
data_last_day = data3[data3["ObservationDate"] == data_last["ObservationDate"].iloc[0]] 
country_list = list(data_last_day["Country/Region"].unique())
confirmed = []
deaths = []
recovered = []
for i in country_list:
    x = data_last_day[data_last_day["Country/Region"] == i]
    confirmed.append(sum(x["Confirmed"]))
    deaths.append(sum(x["Deaths"]))
    recovered.append(sum(x["Recovered"]))
data3_country = pd.DataFrame(list(zip(country_list,confirmed,deaths,recovered)),columns = ["Country","Confirmed","Deaths","Recovered"])
data3_country.head()


# In[ ]:


date_list1 = list(data3["ObservationDate"].unique())
confirmed = []
deaths = []
recovered = []
for i in date_list1:
    x = data3[data3["ObservationDate"] == i]
    confirmed.append(sum(x["Confirmed"]))
    deaths.append(sum(x["Deaths"]))
    recovered.append(sum(x["Recovered"]))
data3 = pd.DataFrame(list(zip(date_list1,confirmed,deaths,recovered)),columns = ["Date","Confirmed","Deaths","Recovered"])
data3.head()


# In[ ]:


from datetime import date, timedelta, datetime
data3["Date"] = pd.to_datetime(data3["Date"])
data3.info()


# <a id="2"></a> <br>
# # Normal Plots

# In[ ]:


fig = plt.figure(figsize = (12,8))
sns.lineplot(data = data3["Confirmed"])
plt.title("title of plot")
plt.xlabel("days")
plt.ylabel("confirmed")
plt.show()


# In[ ]:


fig = plt.figure(figsize = (12,8))
sns.scatterplot(x = data1["oldpeak"], y = data1["trestbps"])
plt.title("scatterplot")
plt.show()


# In[ ]:


fig = plt.figure(figsize = (12,8))
sns.barplot(x = data3_country.head(10)["Country"],y = data3_country.head(10)["Confirmed"])
plt.xticks(rotation = 45)
plt.title("barplot")
plt.show()


# In[ ]:


fig = plt.figure(figsize = (12,8))
sns.distplot(data1["age"],kde=False)
plt.title("histogram")
plt.show()


# In[ ]:


fig = plt.figure(figsize = (12,8))
sns.distplot(data1["age"])
plt.title("histogram with kdeplot")
plt.show()


# In[ ]:


fig = plt.figure(figsize = (12,8))
sns.distplot(data1["age"],hist = False)
plt.title("kdeplot")
plt.show()


# In[ ]:


fig = plt.figure(figsize = (12,8))
sns.kdeplot(data1["age"],shade = True)
plt.title("kdeplot shade")
plt.show()


# In[ ]:


fig = plt.figure(figsize = (12,8))
sns.jointplot(x=data1["age"], y=data1["thalach"])
plt.title("jointplot")
plt.show()


# In[ ]:


fig = plt.figure(figsize = (12,8))
sns.jointplot(x=data1["age"], y=data1["thalach"], kind="hex")
plt.title("jointplot hexagon")
plt.show()


# In[ ]:


fig = plt.figure(figsize = (12,8))
sns.jointplot(x=data1["age"], y=data1["thalach"], kind="kde")
plt.title("jointplot kde")
plt.show()


# In[ ]:


fig = plt.figure(figsize = (12,8))
sns.pairplot(data3)
plt.title("pairplot")
plt.show()


# In[ ]:


fig = plt.figure(figsize = (12,8))
sns.boxplot(x = data1["sex"], y = data1["age"], data=data1,hue = "sex")
plt.title("boxplot")
plt.show()


# In[ ]:


fig = plt.figure(figsize = (12,8))
sns.violinplot(y = data1["chol"])
plt.title("violinplot")
plt.show()


# In[ ]:


fig = plt.figure(figsize = (12,8))
sns.violinplot(x="sex", y="trestbps", hue="sex",
                    data=data1, split=True)
plt.title("violinplot split")
plt.show()


# In[ ]:


f,ax = plt.subplots(figsize=(12, 12))
sns.heatmap(data1.corr(), annot=True,annot_kws = {"size": 12}, linewidths=0.5, fmt = '.2f', ax=ax)
plt.title("heatmap")
plt.show()


# In[ ]:


fig = plt.figure(figsize = (12,8))
sns.countplot(data1["sex"])
plt.title("countplot")
plt.show()


# In[ ]:


fig = plt.figure(figsize = (12,8))
sns.swarmplot(x="slope", y="trestbps",hue="sex", data=data1)
plt.title("swarmplot")
plt.show()


# In[ ]:


fig = plt.figure(figsize = (12,8))
sns.lmplot(x="chol", y="trestbps", data=data1)
plt.title("lmplot")
plt.show()


# <a id="3"></a> <br>
# # Styled Plots

# In[ ]:


fig = plt.figure(figsize = (12,8))

cmap = sns.cubehelix_palette(dark=.3, light=.8, as_cmap=True)
sns.scatterplot(x = data1["oldpeak"],
                y = data1["trestbps"],
                hue = data1["oldpeak"],
                size = data1["trestbps"],
                sizes = (20,200),
                marker = "s")
plt.title("styled scatter")
plt.show()


# In[ ]:


fig = plt.figure(figsize = (12,8))
sns.barplot(x = data3_country.head(10)["Country"],y = data3_country.head(10)["Confirmed"],palette="Blues_d")
plt.xticks(rotation = 45)
plt.title("styled barplot")
plt.show()


# In[ ]:


fig = plt.figure(figsize = (12,8))
sns.pairplot(data2.head(20),hue = "Platform",markers = "D")
plt.title("pairplot")
plt.show()


# In[ ]:


f,ax = plt.subplots(figsize=(12, 12))
sns.heatmap(data2.corr(), annot=True,annot_kws = {"size": 12},linecolor = "black", linewidths=0.5, fmt = '.2f', ax=ax,cmap = "BrBG")
plt.title("styled heatmap")
plt.show()


# <a id="4"></a> <br>
# # Conclusion
# * **If there is something wrong with this kernel please let me know in the comments.**
# 
# ### My other kernels: https://www.kaggle.com/mrhippo/notebooks
# * **References:**
# * https://seaborn.pydata.org/
# * https://www.kaggle.com/kanncaa1/seaborn-tutorial-for-beginners
