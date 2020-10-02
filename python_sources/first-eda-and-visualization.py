#!/usr/bin/env python
# coding: utf-8

# **Hello everybody. This is my first data analysis.
# Data set World Happiness Report
# **
# 

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import os
print(os.listdir("../input"))
import warnings
warnings.filterwarnings("ignore")


# In[ ]:


data = pd.read_csv("../input/2017.csv")


# In[ ]:


data.columns


# In[ ]:


data.head()


# In[ ]:


data.tail()


# In[ ]:


data.info()


# In[ ]:


data.describe()


# **Countries happiness scores**

# In[ ]:


pd.melt(frame=data,id_vars=["Country"],value_vars=["Happiness.Score"])


# **Correlation map**
# 

# In[ ]:


f,ax = plt.subplots(figsize=(8,8))
sns.heatmap(data.corr().iloc[1:,1:],annot=True,linewidths=.5,fmt=".2f",ax=ax)
plt.show()


#  **Whisker.high** and **Whisker.low**  is descreased **Happy Score** falling

# In[ ]:


plt.figure(figsize=(28,7))

sns.pointplot(data["Happiness.Rank"],data["Whisker.high"]/max(data["Whisker.high"]),alpha=0.4,color="purple")
sns.pointplot(data["Happiness.Rank"],data["Whisker.low"]/max(data["Whisker.low"]),color="green",alpha=0.4)

plt.grid()


# In[ ]:


plt.figure(figsize=(9,4))
sns.boxplot(data["Economy..GDP.per.Capita."])
plt.show()


# **Classify according to economical power
# **

# In[ ]:


economyQ3 = data["Economy..GDP.per.Capita."].quantile(0.75)
economyQ1 = data["Economy..GDP.per.Capita."].quantile(0.25)
economyQ2 = data["Economy..GDP.per.Capita."].quantile(0.50)

EconomyQuality = []

for i in data["Economy..GDP.per.Capita."]:
    if i >= economyQ3 :
        EconomyQuality.append("VeryHigh")
    elif i < economyQ3 and i >= economyQ2:
        EconomyQuality.append("High")
    elif i < economyQ2 and i >= economyQ1 :
        EconomyQuality.append("Normal")
    else:
        EconomyQuality.append("Low")
        
data["EconomyQuality"] = EconomyQuality
data.head()


# **And Happy classification**

# In[ ]:


happyQ3 = data["Happiness.Score"].quantile(0.75)
happyQ1 = data["Happiness.Score"].quantile(0.25)
happyQ2 = data["Happiness.Score"].quantile(0.50)

happyQuality = []

for i in data["Happiness.Score"]:
    if i >= happyQ3 :
        happyQuality.append("VeryHigh")
    elif i < happyQ3 and i >= happyQ2:
        happyQuality.append("High")
    elif i < happyQ2 and i >= happyQ1 :
        happyQuality.append("Normal")
    else:
        happyQuality.append("Low")
        
data["HappyQuality"] = happyQuality
data.head()


# In[ ]:


plt.figure(figsize=(7,7))
sns.barplot(data.EconomyQuality.unique(),data.EconomyQuality.value_counts(),palette = sns.cubehelix_palette(len(data.EconomyQuality.unique())))
plt.show()


# In[ ]:


plt.figure(figsize=(7,7))
sns.barplot(data.HappyQuality.unique(),data.HappyQuality.value_counts(),palette = sns.cubehelix_palette(len(data.EconomyQuality.unique())))

plt.show()


# In[ ]:


data.groupby("EconomyQuality").mean().sort_values(by=['Happiness.Rank'])


# In[ ]:


data.groupby("HappyQuality").mean().sort_values(by=['Happiness.Rank'])


# **HappyQuality and EconomyQuality classification together
# **

# In[ ]:


pd.melt(frame=data,id_vars=["HappyQuality","EconomyQuality"],value_vars=["Country"])


# In[ ]:


plt.figure(figsize=(7,7))
sns.boxplot(data["HappyQuality"],data["Economy..GDP.per.Capita."])
plt.show()


# **Happiness and economic correlation**
# 
# 

# In[ ]:


sns.jointplot(data["Happiness.Score"],data["Economy..GDP.per.Capita."],kind="hex",size=7,ratio=3)
plt.show()


# In[ ]:


dataCorr = data.loc[:,["Economy..GDP.per.Capita.","Happiness.Score"]]
plt.figure(figsize=(7,7))
sns.heatmap(dataCorr.corr(),annot=True)
plt.show()


# **Countries with bad economy are not happy
# **

# In[ ]:


plt.figure(figsize=(7,7))
plt.scatter(x=data.HappyQuality, y=data.EconomyQuality)
plt.xlabel("Happy")
plt.ylabel("Economy")
plt.show()


# In[ ]:


plt.figure(figsize=(7,7))
sns.swarmplot(data["HappyQuality"],data["Economy..GDP.per.Capita."],hue=data["EconomyQuality"])
plt.show()


# **HappyQuality rates in economy classes
# **

# In[ ]:


filt1 = data.EconomyQuality == "Low" 
filt2 = data.EconomyQuality == "Normal" 
filt3 = data.EconomyQuality == "High" 
filt4 = data.EconomyQuality == "VeryHigh" 

dataX1 = data[filt1]
dataX2 = data[filt2]
dataX3 = data[filt3]
dataX4 = data[filt4]

colors = ["#9b59b6", "#3498db", "#95a5a6", "#2ecc71"]

explode = [0,0,0,0]
explode2 = [0,0]

plt.figure(figsize=(8,8))

plt.subplot(221)

count = dataX1["HappyQuality"].value_counts()
countDict = dict(count)

dictKey = [i for i,j in countDict.items()]
plt.pie(count,labels =countDict,explode=explode2 ,colors=colors,autopct="%1.1f%%")
plt.xlabel("Low Economy")
plt.ylabel("HappyQuality percent")

plt.subplot(222)

count = dataX2["HappyQuality"].value_counts()
countDict = dict(count)
dictKey = [i for i,j in countDict.items()]

plt.pie(count,labels =countDict,explode=explode ,colors=colors,autopct="%1.1f%%")
plt.xlabel("Normal Economy")
plt.ylabel("HappyQuality percent")

plt.subplot(223)

count = dataX3["HappyQuality"].value_counts()
countDict = dict(count)
dictKey = [i for i,j in countDict.items()]

plt.pie(count,labels =countDict,explode=explode ,colors=colors,autopct="%1.1f%%")
plt.xlabel("High Economy")
plt.ylabel("HappyQuality percent")

plt.subplot(224)

count = dataX4["HappyQuality"].value_counts()
countDict = dict(count)
dictKey = [i for i,j in countDict.items()]

plt.pie(count,labels =countDict,explode=explode2 ,colors=colors,autopct="%1.1f%%")
plt.xlabel("VeryHigh Economy")
plt.ylabel("HappyQuality percent")

plt.show()


# **Averages of features according to happiness score
# **

# In[ ]:


happinessMean = data["Happiness.Score"].mean()

print(data.loc[data["Happiness.Score"] > happinessMean,["Dystopia.Residual"]].mean())
print(data.loc[data["Happiness.Score"] < happinessMean,["Dystopia.Residual"]].mean())
print(data.loc[data["Happiness.Score"] > happinessMean,["Freedom"]].mean())
print(data.loc[data["Happiness.Score"] < happinessMean,["Freedom"]].mean())
print(data.loc[data["Happiness.Score"] > happinessMean,["Trust..Government.Corruption."]].mean())
print(data.loc[data["Happiness.Score"] < happinessMean,["Trust..Government.Corruption."]].mean())


# In[ ]:


plt.figure(figsize=(7,7))
sns.scatterplot(data["Trust..Government.Corruption."],data["Dystopia.Residual"],hue=data["HappyQuality"],style=data["HappyQuality"],s=75)
plt.show()


# **No problem as long as the economy grows
# **

# In[ ]:


plt.figure(figsize=(7,7))
sns.scatterplot(data["Dystopia.Residual"],data["Economy..GDP.per.Capita."],hue=data["HappyQuality"],style=data["HappyQuality"],s=75)
plt.show()


# **There is no clear effect of Dystopia.Residual
# **

# In[ ]:


d1 = data.groupby("Dystopia.Residual").max().sort_values(by = "Dystopia.Residual",ascending=True).head(15)
d2 = data.groupby("Dystopia.Residual").min().sort_values(by = "Dystopia.Residual",ascending=False).head(15)
sns.scatterplot(d1.HappyQuality,d1.index,hue=d1.EconomyQuality)
plt.show()
sns.scatterplot(d2.HappyQuality,d2.index,hue=d2.EconomyQuality)
plt.show()


# **Freedom and economic power bring happiness
# **

# In[ ]:


plt.figure(figsize=(7,7))
sns.scatterplot(data["Freedom"],data["Economy..GDP.per.Capita."],hue=data["HappyQuality"],style=data["HappyQuality"],s=75)
plt.show()


# **No happiness if the health expectant is low
# **

# In[ ]:


plt.figure(figsize=(7,7))
sns.scatterplot(data["Health..Life.Expectancy."],data["Economy..GDP.per.Capita."],hue=data["HappyQuality"],style=data["HappyQuality"],s=75)
plt.show()


# In[ ]:


plt.figure(figsize=(7,7))
sns.scatterplot(data["Family"],data["Health..Life.Expectancy."],hue=data["HappyQuality"],style=data["HappyQuality"],s=75)
plt.show()


# **Even if everything is low, people will be happy as long as they are free
# **

# In[ ]:


filter1 = data["Health..Life.Expectancy."] < data["Health..Life.Expectancy."].mean()
filter2 = data["Family"] < data["Family"].mean()

common = [(i & j) for i,j in zip(filter1,filter2)]

commonData = data[common]
commonData[commonData["HappyQuality"] == "High"]

