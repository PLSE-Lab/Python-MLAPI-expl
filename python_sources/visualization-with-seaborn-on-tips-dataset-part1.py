#!/usr/bin/env python
# coding: utf-8

# # Data Visualization with Seaborn

# ### 1- Import and Meet DataSet
# ### 2 - Visualization
# ####  2.1 - Visualizing statistical relationships
# ####  2.2 - Plotting with categorical data
# ###### Note : This is first part, i have one more part visualization examples on same dataset. 

# ## 1- Import and Meet DataSet

# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# In[ ]:


tips = sns.load_dataset("tips")
df = tips.copy()
df.sample(7)


# In[ ]:


df.info()


# In[ ]:


df.describe().T


# In[ ]:


df.isnull().sum()


# In[ ]:


df.corr()


# ## 2 -Visualization

# ### 2.1 Visualizing statistical relationships

# In[ ]:


sns.set(style="darkgrid")
sns.relplot(x = "total_bill", y = "tip", data = df);


# In[ ]:


sns.relplot(x = "total_bill", y = "tip", hue= "smoker", data = df);


# In[ ]:


sns.scatterplot(x = "total_bill", y = "tip", hue= "sex", data = df);


# In[ ]:


sns.scatterplot(x = "total_bill", y = "tip", hue= "smoker", style= "smoker", data = df);


# In[ ]:


sns.relplot(x = "total_bill",
            y = "tip",
            hue = "smoker",
            style = "time",
            height = 6,
            data = tips);


# In[ ]:


sns.relplot(x = "total_bill", y = "tip", hue = "size", height = 7, data = df);


# In[ ]:


sns.relplot(x = "total_bill", y = "tip", size = "size", sizes = (20,100), hue = "size", data = df);


# ###  2.2 - Plotting with categorical data

# In[ ]:


sns.relplot(x = "total_bill", y = "tip", hue = "smoker", col = "time", data = df);


# In[ ]:


sns.relplot(x = "total_bill", y = "tip", hue = "smoker", col = "sex", data = df);


# In[ ]:


sns.relplot(x = "total_bill", y = "tip", hue = "smoker", col = "day", data = df);


# In[ ]:


sns.catplot(x = "day", y = "total_bill", data = df);


# In[ ]:


sns.catplot(x = "day", y = "total_bill", hue = "sex", data = df);


# In[ ]:


sns.catplot(x = "day", y = "total_bill", jitter = False, hue = "sex", alpha = .33, data = df);


# In[ ]:


sns.swarmplot(x = "day", y = "total_bill", data = df);


# In[ ]:


sns.swarmplot(x = "day", y = "total_bill", hue = "sex", alpha = .75, data = df);


# In[ ]:


sns.swarmplot(x ="size", y = "total_bill", data = df);


# In[ ]:


sns.swarmplot(x ="size", y = "total_bill", hue = "sex", alpha =.7, data = df);


# In[ ]:


sns.catplot(x = "smoker", y = "tip", order = ["No", "Yes"], data = df);


# In[ ]:


sns.catplot(x = "day", y = "total_bill", hue = "time", alpha = .5, data = df);


# In[ ]:


sns.swarmplot(x = "day", y = "total_bill", hue = "time", alpha = .5, data = df);


# In[ ]:


sns.boxplot(x = "day", y = "total_bill", data = df);


# In[ ]:


sns.boxplot(x = "day", y = "total_bill", hue = "sex", data = df);


# In[ ]:


sns.boxplot(x = "day", y = "total_bill", hue = "smoker", data = df);


# In[ ]:


df["weekend"] = df["day"].isin(["Sat","Sun"])
df.sample(5)


# In[ ]:


sns.boxplot(x = "day", y = "total_bill", hue = "weekend", data = df);


# In[ ]:


sns.boxenplot(x= "sex", y = "tip", hue = "smoker", data = df);


# In[ ]:


sns.violinplot(x ="day", y = "total_bill", hue = "time", data = df);


# In[ ]:


sns.violinplot(x ="day", y = "total_bill", hue = "time", bw = .15, data = df);


# In[ ]:


sns.violinplot(x ="day", y = "total_bill", hue = "smoker", bw = .25, split = True, data = df);


# In[ ]:


sns.violinplot(x="day", y="total_bill", hue="smoker", bw=.25, split=True, palette= "pastel", inner= "stick", data=df);


# In[ ]:


sns.violinplot(x = "day", y = "total_bill", inner = None, data = df)
sns.swarmplot(x = "day", y = "total_bill", color = "k", size = 3, data = df);


# In[ ]:


sns.barplot(x = "sex", y= "total_bill", hue = "smoker", data = df);


# In[ ]:


sns.barplot(x = "day", y= "tip", hue = "smoker", palette = "ch:.25", data = df);


# In[ ]:


sns.countplot(x = "day", hue ="sex", data = df);


# In[ ]:


sns.countplot(x = "sex", hue = "smoker", palette = "ch:.25", data = df);


# In[ ]:


sns.countplot(x = "day", hue = "size", palette = "ch:.25", data = df);


# In[ ]:


sns.pointplot(x = "day", y = "tip", data= df);


# In[ ]:


sns.pointplot(x = "day", y = "size", hue = "sex", linestyles = ["-", "--"], data= df);


# In[ ]:


f, ax = plt.subplots(figsize = (7,3))
sns.countplot(x = "day", hue= "smoker", data = df);


# In[ ]:


sns.catplot(x="day", y = "total_bill", hue = "smoker", col = "time", data = df);


# In[ ]:


sns.catplot(x = "day", y = "total_bill", col = "sex", kind="box", data = df);


# In[ ]:


# to be continued.

