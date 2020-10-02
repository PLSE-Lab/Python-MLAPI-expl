#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import pandas as pd

pathfile = '../input/diabetes.csv'
mydata = pd.read_csv(pathfile)


# In[ ]:


mydata.head()


# In[ ]:


mydata.tail()


# In[ ]:


mydata.describe()


# In[ ]:


mydata.info()


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[ ]:


#mdata= mydata.drop['Age', 'Outcome', 'DiabetesPedigreeFunction']


# In[ ]:


plt.figure(figsize=(16,6))
#sns.lineplot(data=mydata)


# In[ ]:


y = mydata.DiabetesPedigreeFunction


# In[ ]:


mydata.columns


# In[ ]:


features=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'Age', 'Outcome']
   
X=mydata[features]


# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_valid, y_train,  y_valid= train_test_split(X,y,random_state=1)


# In[ ]:


from sklearn.ensemble import RandomForestRegressor


# In[ ]:


marammodel = RandomForestRegressor()
marammodel.fit(X_train, y_train)


# In[ ]:


y_predict = marammodel.predict(X_valid)


# In[ ]:


from sklearn.metrics import mean_absolute_error
print("MAE=", mean_absolute_error(y_valid, y_predict))


# **Visualization data with LinePlot**

# In[ ]:


# Set the width and height of the figure
plt.figure(figsize=(14,6))

# Add title
plt.title("Lineplot")

# Line chart showing BloodPressurev
sns.lineplot(data=mydata['BloodPressure'], label="Blood")

# Line chart showing Age'
sns.lineplot(data=mydata['Age'], label="Age")


# In[ ]:


#Figure Size
plt.figure(figsize= (16,4))

#title
plt.title('Trends of Skin Thickness and Age')
sns.lineplot(x="Age", y="SkinThickness", data= mydata)

plt.show()


# In[ ]:


sns.lineplot(data=mydata['DiabetesPedigreeFunction'], label="Diabetes Pedigree Function")


# In[ ]:





# **Bar chart**
# 
#  create a bar chart showing the  Age and  DiabetesPedigreeFunction

# In[ ]:


# Set the width and height of the figure
plt.figure(figsize=(10,6))

# Add title
plt.title("Relationship between Age and  DiabetesPedigreeFunction")

# Bar chart 
sns.barplot(x=mydata.head()['Age'], y=mydata.head()['DiabetesPedigreeFunction'])

# Add label for vertical axis
plt.ylabel("DiabetesPedigreeFunction")


# **Heatmap**
# 
# In the code cell below, we create a heatmap to quickly visualize patterns in dataset. Each cell is color-coded according to its corresponding value.

# In[ ]:


# Set the width and height of the figure
plt.figure(figsize=(14,7))

# Add title
plt.title("Heatmap of Dataset")

# Heatmap 
sns.heatmap(data=mydata.head(), annot=True)

# Add label for horizontal axis
plt.xlabel("Airline")


# **Scatter plots**
# 
# 
# To create a simple scatter plot, we use the sns.scatterplot command and specify the values

# In[ ]:


# Set the width and height of the figure
plt.figure(figsize=(14,7))

sns.scatterplot(x=mydata['DiabetesPedigreeFunction'], y=mydata['Age'])


#  the strength of the relationship,  add a regression line, or the line that best fits the data. We do this by changing the command to sns.regplot

# In[ ]:


# Set the width and height of the figure
plt.figure(figsize=(14,7))

sns.regplot(x=mydata['DiabetesPedigreeFunction'], y=mydata['Age'])


# In[ ]:





# In[ ]:


# Set the width and height of the figure
plt.figure(figsize=(14,7))

sns.scatterplot(x=mydata['DiabetesPedigreeFunction'], y=mydata['Age'], hue=mydata['Glucose'])


# In[ ]:


plt.figure(figsize=(14,7))

sns.scatterplot(x=mydata['BloodPressure'], y=mydata['Glucose'], hue=mydata['Age'])


# In[ ]:


# Set the width and height of the figure
#plt.figure(figsize=(14,7))

sns.scatterplot(x=mydata['DiabetesPedigreeFunction'], y=mydata['Age'], hue=mydata['Outcome'])


# In[ ]:


# Set the width and height of the figure
plt.figure(figsize=(16,4))

sns.lmplot(x="DiabetesPedigreeFunction", y="Age", hue="Outcome", data=mydata)


# In[ ]:


sns.swarmplot(x=mydata['DiabetesPedigreeFunction'], y=mydata['Age'])


# **Histograms**
# 
# 
# create a histogram to see how BloodPressure varies in dataset. We can do this with the sns.distplot command.

# In[ ]:


# Histogram 
sns.distplot(a=mydata['BloodPressure'], kde=False)


# # KDE plot 
# 

# In[ ]:


sns.kdeplot(data=mydata['BloodPressure'], shade=True)


# In[ ]:


# 2D KDE plot
sns.jointplot(x=mydata['BloodPressure'], y=mydata['Age'], kind="kde")


# In[ ]:


# Histograms for each species
sns.distplot(a=mydata['Glucose'], label="Glucose", kde=False)
sns.distplot(a=mydata['BloodPressure'], label="BloodPressure", kde=False)
sns.distplot(a=mydata['SkinThickness'], label="SkinThickness", kde=False)

# Add title
plt.title("Histogram of data")

# Force legend to appear
plt.legend()


# In[ ]:


# Histograms for each species
sns.kdeplot(data=mydata['Glucose'], label="Glucose", shade=True)
sns.kdeplot(data=mydata['BloodPressure'], label="BloodPressure", shade=True)
sns.kdeplot(data=mydata['SkinThickness'], label="SkinThickness", shade=True)

# Add title
plt.title("Histogram of data")

# Force legend to appear
plt.legend()


# **PairPlot**

# In[ ]:


sns.set(style="ticks")

#df = sns.mydata("")
sns.pairplot(mydata, hue="Age")


# **Violinplot**

# In[ ]:


plt.figure(figsize=(14,7))
# Draw a violinplot 
sns.violinplot(data=mydata.head(), palette="Set3", bw=.5, cut=1, linewidth=2)


# **Striplot**

# In[ ]:


# Initialize the figure
f, ax = plt.subplots()
sns.despine(bottom=True, left=True)

# Show each observation with a scatterplot
sns.stripplot(x="Age", y="BloodPressure", hue="Outcome",
              data=mydata.head(35), dodge=True, jitter=True,
              alpha=.25, zorder=1)

# Show the conditional means
sns.pointplot(x="Age", y="BloodPressure", hue="Outcome",
              data=mydata.head(35), dodge=.532, join=False, palette="dark",
              markers="d", scale=.75, ci=None)


# **#FacePlot**

# In[ ]:


# Set up a grid of axes with a polar projection
g = sns.FacetGrid(mydata.head(), col="Age", hue="Outcome",
                  subplot_kws=dict(projection='polar'), height=4.5,
                  sharex=False, sharey=False, despine=False)

# Draw a scatterplot onto each axes in the grid
g.map(sns.scatterplot, "BloodPressure", "Glucose")


# boxplot

# In[ ]:


# Draw a nested boxplot 
sns.boxplot(x="BloodPressure", y="Age",
            hue="Outcome", palette=["m", "g"],
            data=mydata.head(25))
sns.despine(offset=10, trim=True)


# **Discovering structure in heatmap data**
# 

# In[ ]:




# Draw the full plot
sns.clustermap(mydata.corr(), center=0, cmap="vlag",
            
               linewidths=.75, figsize=(13, 13))


# In[ ]:




