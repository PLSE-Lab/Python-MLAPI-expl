#!/usr/bin/env python
# coding: utf-8

# <h1 class="list-group-item list-group-item-success" data-toggle="list"  role="tab" aria-controls="home">EDA - OSIC Pulmonary Fibrosis Progression</h1>

# <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcSroWe00EY1yelEbhqMg9L8-yjHQWTB5amO8w&usqp=CAU" alt="Meatball Sub" width="800"/>

# **Pulmonary fibrosis** : Pulmonary fibrosis is a lung disease that occurs when lung tissue becomes damaged and scarred. This thickened, stiff tissue makes it more difficult for your lungs to work properly. As pulmonary fibrosis worsens, you become progressively more short of breath.
# 
# The scarring associated with pulmonary fibrosis can be caused by a multitude of factors. But in most cases, doctors can't pinpoint what's causing the problem. When a cause can't be found, the condition is termed idiopathic pulmonary fibrosis. [link](https://www.mayoclinic.org/diseases-conditions/pulmonary-fibrosis/symptoms-causes/syc-20353690#:~:text=Pulmonary%20fibrosis%20is%20a%20lung,your%20lungs%20to%20work%20properly.)

# In[ ]:


from IPython.display import YouTubeVideo
YouTubeVideo('AfK9LPNj-Zo', width=800, height=300)


# In[ ]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objects as go
from ipywidgets import widgets
from ipywidgets import *

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")
from scipy.signal import find_peaks


# In[ ]:


train = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/train.csv",delimiter=",",encoding="latin", engine='python')
test = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/test.csv",delimiter=",",encoding="latin", engine='python')
train.head(10)


# In[ ]:


print(train.info())


# In[ ]:


train.dtypes.value_counts()


# In[ ]:


test.head(10)


# ## 1 - Quik analyse 

# In[ ]:


count = train['Patient'].value_counts() 
print(count) 


# In[ ]:


print("Number of Patient in the train set {}".format(len( train['Patient'].unique()))) 
print("Number of Patient in the test set {}".format(len( test['Patient'].unique()))) 


# <div class="alert alert-block alert-warning">  
# <b> Observation 1 :</b> Number of Patient in the train set = 176 and the number of patient in the test set = 5. 
# </div>

# ### 2 - FVC (Forced Vital Capacity)

# **Forced vital capacity (FVC) is the total amount of air exhaled during the FEV test.**
# 
# Forced expiratory volume and forced vital capacity are lung function tests that are measured during spirometry. Forced expiratory volume is the most important measurement of lung function. It is used to:
# - Diagnose obstructive lung diseases such as asthma and chronic obstructive pulmonary disease (COPD). A person who has asthma or COPD has a lower FEV1 result than a healthy person.
# - See how well medicines used to improve breathing are working.
# - Check if lung disease is getting worse. Decreases in the FEV1 value may mean the lung disease is getting worse.
# 

# In[ ]:


YouTubeVideo('BmYCAp4dRuA', width=800, height=300)


# **How are FEV1 and FVC Measured?**
# 
# Forced expiratory volume in one second (FEV1) and forced vital capacity (FVC) are measured during a pulmonary function test. A diagnostic device called a spirometer measures the amount of air you inhale, exhale and the amount of time it takes for you to exhale completely after a deep breath. For pulmonary function tests, the spirometer attaches to a machine that records your lung function measurements.
# 
# **What is FVC?** 
# 
# The forced vital capacity (FVC) measurement shows the amount of air a person can forcefully and quickly exhale after taking a deep breath.
# Determining your FVC helps your doctor diagnose a chronic lung disease, monitor the disease over time and understand the severity of the condition. In general, doctors compare your FVC measurement with the predicted FVC based on your age, height and weight.
# 
# **What is FEV1?** 
# 
# Forced expiratory volume is measured during the forced vital capacity test. The forced expiratory volume in one second (FEV1) measurement shows the amount of air a person can forcefully exhale in one second of the FVC test. In addition, doctors can measure forced expiratory volume during the second and third seconds of the FVC test.
# Determining your FEV1 measurement helps your doctor understand the severity of disease. Typically, lower FEV1 scores show more severe stages of lung disease.
# 

# In[ ]:


fig = plt.figure(figsize = (20, 10))
ax = fig.add_subplot()
i = 0 
for id_patient in train["Patient"].unique()[0:6] : 
    y = train[train["Patient"] == id_patient]["FVC"].reset_index(drop=True)
    df = train[train["Patient"] == id_patient].reset_index(drop=True)
    max_peaks_index, _ = find_peaks(y, height=0) 
    doublediff2 = np.diff(np.sign(np.diff(-1*y))) 
    min_peaks_index = np.where(doublediff2 == -2)[0] + 1
    ax.plot(y, color = "blue", alpha = .6)

    if i == 0:
        ax.scatter(x = y[max_peaks_index].index, y = y[max_peaks_index].values, marker = "^", s = 150, color = "green", alpha = .6, label = "Peaks")
        ax.scatter(x = y[min_peaks_index].index, y = y[min_peaks_index].values, marker = "v", s = 150, color = "red", alpha = .6, label = "Troughs")
    else :
        ax.scatter(x = y[max_peaks_index].index, y = y[max_peaks_index].values, marker = "^", s = 150, color = "green", alpha = .6)
        ax.scatter(x = y[min_peaks_index].index, y = y[min_peaks_index].values, marker = "v", s = 150, color = "red", alpha = .6)
    for max_annot in max_peaks_index[:] :
        for min_annot in min_peaks_index[:] :

            max_text = df.iloc[max_annot]["FVC"]
            min_text = df.iloc[min_annot]["FVC"]

            max_text_w = df.iloc[max_annot]["Weeks"]
            min_text_w = df.iloc[min_annot]["Weeks"]

            ax.text(df.index[max_annot], y[max_annot] + 50, s = max_text, fontsize = 12, horizontalalignment = 'center', verticalalignment = 'center')
            ax.text(df.index[min_annot], y[min_annot] + 50, s = min_text, fontsize = 12, horizontalalignment = 'center', verticalalignment = 'center')

            ax.text(df.index[max_annot], y[max_annot] - 50, s = "Week : " + str(max_text_w), fontsize = 12, horizontalalignment = 'center', verticalalignment = 'center')
            ax.text(df.index[min_annot], y[min_annot] - 50, s = "Week : " + str(min_text_w), fontsize = 12, horizontalalignment = 'center', verticalalignment = 'center')
    ax.text(df.index[0], y[0] + 30, s = id_patient, fontsize = 10, horizontalalignment = 'center', verticalalignment = 'center')
    i = i + 1
    ax.legend(loc = "upper left", fontsize = 10)


# In[ ]:


train['FVC_mean'] = train['FVC'].groupby(train['Patient']).transform('mean')
train['FVC_max'] = train['FVC'].groupby(train['Patient']).transform('max')
train['FVC_min'] = train['FVC'].groupby(train['Patient']).transform('min')
train['FVC_std'] = train['FVC'].groupby(train['Patient']).transform('std')


# In[ ]:


fig = plt.figure(figsize = (12, 6))
ax = fig.add_subplot(111) 

for Smoking in sorted(list(train["SmokingStatus"].unique())):
    Age = train[train["SmokingStatus"] == Smoking]["Age"]
    FVC_mean = train[train["SmokingStatus"] == Smoking]["FVC_mean"]
    ax.scatter(Age, FVC_mean, label = Smoking, s = 10)

ax.spines["top"].set_color("None") 
ax.spines["right"].set_color("None")
ax.set_xlabel("Age") 
ax.set_ylabel("FVC_mean")
ax.set_title("Scatter plot of Age vs FVC_mean.")
ax.legend(loc = "upper left", fontsize = 10)


# In[ ]:


fig = plt.figure(figsize = (12, 6))
ax = fig.add_subplot(111) 

for Smoking in sorted(list(train["SmokingStatus"].unique())):
    Percent = train[train["SmokingStatus"] == Smoking]["Percent"]
    FVC_mean = train[train["SmokingStatus"] == Smoking]["FVC_mean"]
    ax.scatter(Percent, FVC_mean, label = Smoking, s = 10)

ax.spines["top"].set_color("None") 
ax.spines["right"].set_color("None")

ax.set_xlabel("Percent") 
ax.set_ylabel("FVC_mean")

ax.set_title("Scatter plot of Percent vs FVC_mean.")
ax.legend(loc = "upper left", fontsize = 10)


# In[ ]:


import squarify
label_value = train["SmokingStatus"].value_counts().to_dict()
labels = ["{} has {} obs".format(class_, obs) for class_, obs in label_value.items()]
colors = [plt.cm.Spectral(i/float(len(labels))) for i in range(len(labels))]
plt.figure(figsize = (10, 5))
squarify.plot(sizes = label_value.values(), label = labels,  color = colors, alpha = 0.8)
plt.title("Smoking Status")


# In[ ]:


fig, axes = plt.subplots(1, 3, figsize=(20, 9))
p = sns.boxplot(x='Sex', y='Age', hue='SmokingStatus', data=train, ax=axes[0])
p.set_title('train')

p = sns.boxplot(x='Sex', y='FVC_mean', hue='SmokingStatus', data=train, ax=axes[1])
p.set_title('train')

p = sns.boxplot(x='Sex', y='FVC_std', hue='SmokingStatus', data=train, ax=axes[2])
p.set_title('train')


# In[ ]:


fig, axes = plt.subplots(1, 2, figsize=(20, 9))


for s in train["SmokingStatus"].unique():
    x = train[train["SmokingStatus"] == s]["Percent"]
    g1 = sns.distplot(x, kde = True, label = "{}".format(s), ax=axes[0])
    g1.set_title('Percent vs SmokingStatus')
g1.legend()


for s in train["SmokingStatus"].unique():
    x = train[train["SmokingStatus"] == s]["Age"]
    g2 = sns.distplot(x, kde = True, label = "{}".format(s), ax=axes[1])
    g2.set_title('Age vs SmokingStatus')
g2.legend()
   


# In[ ]:


fig, axes = plt.subplots(1, 2, figsize=(20, 9))


for s in train["Sex"].unique():
    x = train[train["Sex"] == s]["Percent"]
    g1 = sns.distplot(x, kde = True, label = "{}".format(s), ax=axes[0])
    g1.set_title('Percent vs Sex')
g1.legend()


for s in train["Sex"].unique():
    x = train[train["Sex"] == s]["Age"]
    g2 = sns.distplot(x, kde = True, label = "{}".format(s), ax=axes[1])
    g2.set_title('Age vs Sex')
g2.legend()
   


# In[ ]:


fig, axes = plt.subplots(2, 2, figsize=(20, 11))


for s in train["SmokingStatus"].unique():
    x = train[train["SmokingStatus"] == s]["FVC_mean"]
    g1 = sns.distplot(x, kde = True, label = "{}".format(s), ax=axes[0,0])
    g1.set_title('FVC_mean vs SmokingStatus')
g1.legend()


for s in train["SmokingStatus"].unique():
    x = train[train["SmokingStatus"] == s]["FVC_std"]
    g2 = sns.distplot(x, kde = True, label = "{}".format(s), ax=axes[0,1])
    g2.set_title('FVC_std vs SmokingStatus')
g2.legend()

for s in train["SmokingStatus"].unique():
    x = train[train["SmokingStatus"] == s]["FVC_max"]
    g3 = sns.distplot(x, kde = True, label = "{}".format(s), ax=axes[1,0])
    g3.set_title('FVC_max vs SmokingStatus')
g3.legend()

for s in train["SmokingStatus"].unique():
    x = train[train["SmokingStatus"] == s]["FVC_min"]
    g4 = sns.distplot(x, kde = True, label = "{}".format(s), ax=axes[1,1])
    g4.set_title('FVC_min vs SmokingStatus')
g4.legend()


# In[ ]:


fig, axes = plt.subplots(2, 2, figsize=(20, 11))


for s in train["Sex"].unique():
    x = train[train["Sex"] == s]["FVC_mean"]
    g1 = sns.distplot(x, kde = True, label = "{}".format(s), ax=axes[0,0])
    g1.set_title('FVC_mean vs Sex')
g1.legend()


for s in train["Sex"].unique():
    x = train[train["Sex"] == s]["FVC_std"]
    g2 = sns.distplot(x, kde = True, label = "{}".format(s), ax=axes[0,1])
    g2.set_title('FVC_std vs Sex')
g2.legend()

for s in train["Sex"].unique():
    x = train[train["Sex"] == s]["FVC_max"]
    g3 = sns.distplot(x, kde = True, label = "{}".format(s), ax=axes[1,0])
    g3.set_title('FVC_max vs Sex')
g3.legend()

for s in train["Sex"].unique():
    x = train[train["Sex"] == s]["FVC_min"]
    g4 = sns.distplot(x, kde = True, label = "{}".format(s), ax=axes[1,1])
    g4.set_title('FVC_min vs Sex')
g4.legend()


# ### To be continued ...
