#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


dataFrame = pd.read_csv("../input/2015.csv")
dataFrame.head()


# In[ ]:


dataFrame.columns


# In[ ]:


dataFrame.info()


# In[ ]:


dataFrame.describe()


# In[ ]:


dataFrame.plot(figsize=(16, 5)) # not gonna show good result
plt.show() #for removing "<matplotlib.axes ..." 


# In[ ]:


plotting_frame = dataFrame.drop(["Happiness Rank", "Happiness Score"], axis=1)
plotting_frame.plot(figsize=(16,8)) 
plt.show()


# # Scatter plot

# In[ ]:


#economy_health_df = dataFrame["Economy (GDP per Capita)", "Health (Life Expectancy)"]
plt.figure(figsize=(16,8))
sns.scatterplot("Economy (GDP per Capita)","Health (Life Expectancy)", data = dataFrame, color="green" )
plt.show()


# In[ ]:


dataFrame.plot.scatter("Family","Health (Life Expectancy)", color= "red",figsize=(16,8)) #another way to show scatter plot
plt.show()


# # Bar Plot

# In[ ]:


#Economy (GDP per Capita) regional
region_list = list(dataFrame["Region"].unique())
regional_records = []
for each in region_list:
    x = dataFrame[dataFrame["Region"] == each]
    regional_records.append(sum(x["Economy (GDP per Capita)"])/len(x))

df_eco_per_region = pd.DataFrame({"Region": region_list, "Economy" : regional_records })
new_index = (df_eco_per_region["Economy"].sort_values(ascending=False)).index.values
df_eco_sorted = df_eco_per_region.reindex(new_index)

plt.figure(figsize=(12,8))
sns.barplot("Region", "Economy", data= df_eco_sorted )
plt.xticks(rotation=45)
plt.xlabel("Regions")
plt.ylabel("Economy")
plt.title("Regions Economy")
plt.show()


# In[ ]:


freedom_regional_records = []
for each in region_list:
    x = dataFrame[dataFrame["Region"] == each]
    fredom_rate = sum(x["Freedom"])/len(x)
    freedom_regional_records.append(fredom_rate)

df_freedom_per_region = pd.DataFrame({"Region": region_list, "Freedom" : freedom_regional_records })
new_index = (df_freedom_per_region["Freedom"].sort_values(ascending=False)).index.values
df_freedom_per_region_sorted = df_freedom_per_region.reindex(new_index)

plt.figure(figsize=(12,8))
sns.barplot("Region", "Freedom", data= df_freedom_per_region_sorted )
plt.xticks(rotation=60)
plt.xlabel("Regions")
plt.ylabel("Freedom")
plt.title("Regions Freedom")
plt.show()


# # Point Plot

# In[ ]:


f,ax1 = plt.subplots(figsize =(12,8))
sns.pointplot("Region","Freedom", data = df_freedom_per_region_sorted, color="green")
sns.pointplot("Region", "Economy", data = df_eco_sorted, color="blue")
plt.xticks(rotation=45)
plt.xlabel("Happinnes Rank")
plt.ylabel("Values")
plt.grid()


# # Joint plot

# In[ ]:


plt.figure(figsize=(12,8))
sns.jointplot("Freedom", "Economy (GDP per Capita)", data= dataFrame, kind="kde", height=7)
plt.show()


# In[ ]:


plt.figure(figsize=(12,8))
sns.jointplot("Freedom", "Economy (GDP per Capita)", data= dataFrame, height=8, color="r", ratio=3)
plt.show()


# # LM Plot

# In[ ]:


sns.lmplot(x="Economy (GDP per Capita)", y="Health (Life Expectancy)", data=dataFrame)
plt.show()


# # Heatmap

# In[ ]:


f,ax = plt.subplots(figsize=(10, 10))
sns.heatmap(dataFrame.corr(), annot=True, linewidths=0.5,linecolor="red", fmt= '.1f',ax=ax)
plt.show()


# # Pairplot

# In[ ]:


data_for_pair = dataFrame.drop(["Happiness Score", "Standard Error"], axis= 1)
sns.pairplot(data_for_pair)
plt.show()


# In[ ]:





# In[ ]:




