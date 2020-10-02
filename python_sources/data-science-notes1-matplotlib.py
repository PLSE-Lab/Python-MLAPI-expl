#!/usr/bin/env python
# coding: utf-8

# # Introduction
# * This kernel is notes that I took for **Matplotlib**. This is not a tutorial you can think that kernel as a cheatsheet for Matplotlib. While coding a kernel open this in a new tab and copy paste. 
# 
# <img src = "https://matplotlib.org/_static/logo2_compressed.svg" height = "422" width = "750">
# 
# ### This kernel is a part of a big series:
# * Data Science Notes1: Matplotlib
# * [Data Science Notes2: Seaborn](https://www.kaggle.com/mrhippo/data-science-notes2-seaborn?scriptVersionId=38602288)
# * [Data Science Notes3: Plotly](https://www.kaggle.com/mrhippo/data-science-notes3-plotly?scriptVersionId=38663418)
# * [Data Science Notes4: Machine Learning (ML)](https://www.kaggle.com/mrhippo/data-science-notes4-machine-learning?scriptVersionId=39376804)
# * [Data Science Notes5: Deep Learning: ANN](https://www.kaggle.com/mrhippo/data-science-notes5-deep-learning-ann) 
# * [Data Science Notes6: Deep Learning: CNN](https://www.kaggle.com/mrhippo/data-science-notes6-deep-learning-cnn) 
# * [Data Science Notes7: Deep Learning: RNN and LSTM](https://www.kaggle.com/mrhippo/data-science-notes7-deep-learning-rnn-and-lstm)
# 
# ### This kernel will be updated
# 
# ## Content
# * [Imports and Datasets](#1)
# * [Normal Plots](#2)
# * [Texts](#3)
# * [Styled Plots](#4)
# * [Subplots](#5)
# * [Zooming](#6)
# * [Styles](#7)
# * [Conclusion](#8)

# <a id="1"></a> <br>
# # Imports and Datasets

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from datetime import date, timedelta, datetime
import warnings
warnings.filterwarnings('ignore')

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

# ## Anatomy of a Plot
# <img src = "https://matplotlib.org/3.1.1/_images/sphx_glr_anatomy_001.png" height = "422" width = "750">

# In[ ]:


plt.style.use("default")
plt.plot(range(10,400),range(100,490))
plt.show()


# In[ ]:


fig = plt.figure(figsize = (12,8))
plt.plot(data3["Confirmed"],data3["Deaths"], label = "Label of Plot")
plt.legend()
plt.title("Title of Plot")
plt.xlabel("X Label")
plt.ylabel("Y Label")
plt.grid(True, alpha = 0.4)
plt.show()


# In[ ]:


fig = plt.figure(figsize = (12,8))
plt.scatter(data1["trestbps"],data1["chol"], color = "r")
plt.title("scatter plot")
plt.show()


# In[ ]:


fig = plt.figure(figsize = (12,8))
plt.bar(data1["age"],data1["chol"])
plt.title("bar plot")
plt.show()


# In[ ]:


fig = plt.figure(figsize = (12,8))
plt.hist(data1["age"])
plt.title("histogram (age)")
plt.xlabel("age")
plt.ylabel("frequency")
plt.show()


# In[ ]:


labels = ["Deaths","Recovered","Confirmed"]
sizes = [data3["Deaths"].iloc[len(data3)-1],data3["Recovered"].iloc[len(data3)-1],data3["Confirmed"].iloc[len(data3)-1]]
fig = plt.figure(figsize = (12,8))
plt.pie(sizes, labels = labels)
plt.title("piechart")
plt.show()


# In[ ]:


fig = plt.figure(figsize = (12,8))
plt.boxplot(data1["age"])
plt.ylabel("age")
plt.title("boxplot")
plt.show()


# In[ ]:


x = ["A","B","C","D","E","F"]
y = [10,20,11,22,14,18]

fig = plt.figure(figsize = (10,6))
plt.polar(x,y)
plt.title("polar lineplot")
plt.show()


# In[ ]:


x = ["A","B","C","D","E","F"]
y = [10,20,11,22,14,18]

fig = plt.figure(figsize = (10,6))
plt.polar(x,y,"ro")
plt.title("polar lineplot")
plt.show()


# In[ ]:


x = ["A","B","C","D","E","F"]
y = [10,20,11,22,14,18]

fig = plt.figure(figsize = (10,6))
ax = plt.subplot(111, projection = "polar")
ax.bar(x,y)
ax.set_title("polar barplot")
plt.show()


# In[ ]:


fig = plt.figure(figsize = (10,6))
plt.imshow(plt.imread("/kaggle/input/stanford-dogs-dataset/images/Images/n02109961-Eskimo_dog/n02109961_5772.jpg"))
plt.title("showing images with matplotlib")
plt.show()


# In[ ]:


# 3d plot
fig = plt.figure(figsize = (12,8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(data3["Confirmed"],data3["Deaths"],data3["Recovered"],label = "label")
ax.set_xlabel("confirmed")
ax.set_ylabel("deaths")
ax.set_zlabel("recovered")
plt.title("3D lineplot")
plt.legend()

plt.show()


# In[ ]:


fig = plt.figure(figsize = (12,8))
plt.plot(data3["Date"],data3["Confirmed"], label = "confirmed")
plt.plot(data3["Date"],data3["Recovered"], color = "green", label = "recovered")
plt.title("2 plots together")
plt.xlabel("dates (converted fron object to datetime)")
plt.legend()
plt.show()


# <a id="3"></a> <br>
# # Texts

# In[ ]:


fig = plt.figure(figsize = (12,8))
plt.text(0.30, 0.5, "this is a text, fontsize = 20", fontsize=20)
plt.text(0.20, 0.5, "this is a text, rotation = 45", fontsize=15, rotation = 45)
plt.text(0.20, 0.3, "this is a text, color = red", fontsize=18, color = "r")
plt.text(0.30, 0.1, 'boxed text, pad = how big box is', fontsize = 25,
        bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 25})
plt.text(0.60, 0.3, "this is a text, style = italic",style = "italic", fontsize=18)
plt.annotate('annotate', xy=(0.1, 0.3), xytext=(0.08,0.08),
            arrowprops=dict(facecolor='black', shrink=0.5))

plt.text(0.2, 0.4, 'an equation: $E=mc^2$', fontsize=20)

plt.show()


# <a id="4"></a> <br>
# # Styled Plots

# In[ ]:


# styled lineplot
fig = plt.figure(figsize = (12,8))
plt.plot(np.arange(1,len(data3)+1),data3["Deaths"],"-.",alpha = 0.6,linewidth = 3,color = "#cc0000",label = "deaths",marker = "o",markerfacecolor = "#0066ff",markersize = 10,markevery = [-1])
plt.legend(loc='center')
plt.title("styled lineplot (fontsize = 25)",fontsize = 25)
plt.ylim(0,8*10**5)
plt.annotate('today total: '+str(data3.tail(1)["Deaths"].iloc[0]), xy=(len(data3), data3.tail(1)["Deaths"]), xytext=(len(data3)-65, data3.tail(1)["Deaths"]),
             arrowprops=dict(facecolor='black', shrink=0.05),
             fontsize = 15)
plt.xticks(rotation = 45)
plt.tick_params(labelsize = 14)   
plt.xlabel("days",fontsize = 15)
plt.ylabel("deaths, limit = 0,8*10**5",fontsize = 20, color = "blue")
plt.grid(True, alpha = 0.4, color = "#33cc33")
plt.fill_between(np.arange(1,len(data3)+1),data3["Deaths"],color = "red",alpha = 0.3)
plt.show()


# In[ ]:


#styled scatter
fig = plt.figure(figsize = (12,8))
plt.scatter(data1["oldpeak"],data1["chol"], marker = "^", s = 40, alpha = 0.6, label = "oldpeak")
plt.scatter(data1["oldpeak"],data1["chol"], marker = "o", s = data1["trestbps"]*3, alpha = 0.4,facecolors = "none",edgecolor = "red", label = "trestbps") 
colors = np.random.rand(len(data1["thalach"])) 
plt.scatter(data1["oldpeak"],data1["thalach"],c = colors, marker = "X",s = 50,label = "thalach")
plt.legend()
plt.yscale("log")
plt.title("styled scatterplot", fontsize = 20)
plt.grid(which="both",alpha = 0.2)
plt.xlabel("oldpeak")
plt.show()

"""
MARKERS:
"."  point
","  pixel
"o"  circle
"v"  triangle_down
"^"  triangle_up
"<"  triangle_left
">"  triangle_right
"1"  tri_down
"2"  tri_up
"3"  tri_left
"4"  tri_right
"8"  octagon
"s"  square
"p"  pentagon
"P"  plus (filled)
"*"  star
"h"  hexagon1
"H"  hexagon2
"+"  plus
"x"  x
"X"  x (filled)
"D"  diamond
"d"  thin_diamond
"|"  vline
"_"  hline

"""


# In[ ]:


#styled barplot
fig = plt.figure(figsize = (12,8))
data_bar = data3.head(10)
plt.barh(data_bar["Date"],data_bar["Recovered"], color = "green",linewidth = 2,edgecolor = "black")
plt.title("styled barplot",fontsize = 15)
plt.show()


# In[ ]:


#styled histogram
fig = plt.figure(figsize = (12,8))
plt.grid(True, alpha = 0.4)
plt.hist(data1["oldpeak"],cumulative = True,label = "cumulative hist",color = "#00ffff")
plt.legend()
plt.title("styled histogram", fontsize = 20)
plt.xlabel("oldpeak")
plt.show()


# <a id="5"></a> <br>
# # Subplots
# <img src = "https://www.stat.berkeley.edu/~nelle/teaching/2017-visualization/figures/subplot-grid.png" height = "315" width = "550">

# In[ ]:


#subplots
fig = plt.figure(figsize=(12, 7)) # initialize figure

ax = [None for _ in range(6)] # list to save many ax for setting parameter in each

ax[0] = plt.subplot2grid((3,4), (0,0), colspan=4)
ax[1] = plt.subplot2grid((3,4), (1,0), colspan=1)
ax[2] = plt.subplot2grid((3,4), (1,1), colspan=1)
ax[3] = plt.subplot2grid((3,4), (1,2), colspan=1)
ax[4] = plt.subplot2grid((3,4), (1,3), colspan=1,rowspan=2)
ax[5] = plt.subplot2grid((3,4), (2,0), colspan=3)

for ix in range(6): 
    ax[ix].set_title('ax[{}]'.format(ix)) # make ax title for distinguish:)
    ax[ix].set_xticks([]) # to remove x ticks
    ax[ix].set_yticks([]) # to remove y ticks

plt.show()


# In[ ]:


fig, ax = plt.subplots(2,2, figsize = (12,8))
fig.suptitle('subplots title',fontsize = 20)
ax[0,0].scatter(data1["age"],data1["thalach"])
ax[0,0].set_title("1. title")
ax[0,1].bar(data1["age"],data1["slope"])
ax[0,1].set_title("2. title")
ax[1,0].plot(data1["age"],data1["age"])
ax[1,0].set_title("3. title")
ax[1,1].pie([list(data1["sex"].values).count(1),list(data1["sex"].values).count(0)])
ax[1,1].set_title("4. title")
plt.show()


# In[ ]:


fig = plt.figure(figsize = (12,8))

ax1 = plt.subplot(212)
ax1.plot(data1["age"], data1["age"])
ax1.set_title("big plot")

ax2 = plt.subplot(221)
ax2.plot(data1["age"], data1["age"])
ax2.set_title("little plot")

ax3 = plt.subplot(222)
ax3.plot(data1["age"], data1["age"])
ax3.set_title("little plot")

plt.show()


# <a id="6"></a> <br>
# # Zooming

# In[ ]:


fig = plt.figure(figsize = (12,8))
ax1 = plt.subplot(212)
ax1.margins(0.05)           # Default margin is 0.05, value 0 means fit
ax1.plot(data3["Confirmed"], data3["Deaths"])

ax2 = plt.subplot(221)
ax2.margins(2, 2)           # Values >0.0 zoom out
ax2.plot(data3["Confirmed"], data3["Deaths"])
ax2.set_title('Zoomed out')

ax3 = plt.subplot(222)
ax3.margins(x=0, y=-0.25)   # Values in (-0.5, 0.0) zooms in to center
ax3.plot(data3["Confirmed"], data3["Deaths"])
ax3.set_title('Zoomed in')

plt.show()


# <a id="7"></a> <br>
# # Styles

# In[ ]:


#styles
styles = ["bmh",
          "classic",
          "dark_background",
          "fivethirtyeight",
          "ggplot",
          "grayscale",
          "seaborn-bright",
         "seaborn-colorblind",
         "seaborn-dark",
         "seaborn-dark-palette",
         "seaborn-darkgrid",
         "seaborn-deep",
         "seaborn-muted",
         "seaborn-notebook",
         "seaborn-paper",
         "seaborn-pastel",
         "seaborn-poster",
         "seaborn-talk",
         "seaborn-ticks",
         "seaborn-white",
         "seaborn-whitegrid"]
for i in styles:
    fig = plt.figure(figsize = (12,8))
    plt.style.use(i)
    plt.plot([1,2,3,4],[1,2,3,4])
    plt.title(i, fontsize = 20)
    
plt.show()


# <a id="8"></a> <br>
# # Conclusion
# * **If there is something wrong with this kernel please let me know in the comments.**
# 
# ### My other kernels: https://www.kaggle.com/mrhippo/notebooks
# 
# * **References:**
# * https://matplotlib.org/3.2.2/contents.html
