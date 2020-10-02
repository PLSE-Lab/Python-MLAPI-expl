#!/usr/bin/env python
# coding: utf-8

# # **Seaborn Tutorial: Count Plots**

# ----------
# # Data Preparation & Feature Classification

# In[ ]:


from pandas import read_csv
data = read_csv("../input/xAPI-Edu-Data.csv")


# In[ ]:


target = "Class"
features = data.drop(target,1).columns


# In[ ]:


features_by_dtype = {}

for f in features:
    dtype = str(data[f].dtype)
    if dtype not in features_by_dtype.keys():
        features_by_dtype[dtype] = [f]
    else:
        features_by_dtype[dtype] += [f]


# In[ ]:


keys = iter(features_by_dtype.keys())
k = next(keys)
l = features_by_dtype[k]
categorical_features = l
k = next(keys)
l = features_by_dtype[k]
numerical_features = l


# In[ ]:


categorical_features, numerical_features
features, target
pass


# # Categorical Features Preview

# In[ ]:


data[categorical_features].head()


# ----------
# # Seaborn's Count Plot

# In[ ]:


from seaborn import countplot
from matplotlib.pyplot import figure, show


# ## Default Settings with Minimal Code

# In[ ]:


figure()
countplot(data=data,x=target)
show()


# ## Horizontal count plot

# In[ ]:


figure()
countplot(data=data,y=target)
show()


# ## Change the size

# In[ ]:


width=12
height=6
figure(figsize=(width,height))
countplot(data=data,x=target)
show()


# ## Manually change the order of subclasses within a category.

# In[ ]:


figure(figsize=(12,6))

order=["L","M","H"]

countplot(data=data,x=target,order=order)
show()


# ## Use descending order 

# In[ ]:


descending_order = data[target].value_counts().sort_values(ascending=False).index

figure(figsize=(12,6))
countplot(data=data,x=target,order=descending_order)
show()


# ## Colour all bars with a single colour

# In[ ]:


figure(figsize=(12,6))
countplot(data=data,x=target,color="tomato")
show()


# ## Manually set the order of colors for each bar with "palette".

# In[ ]:


colours = ["maroon", "navy", "gold"]

figure(figsize=(12,6))
countplot(data=data,x=target,palette=colours)
show()


# Possible "color" string values are from [HTML color names or HEX values][1].
# 
#   [1]: https://www.w3schools.com/colors/colors_names.asp

# # Create a side-by-side countplot with "hue" parameter. Choose another categorical variable.

# In[ ]:


figure(figsize=(12,6))
countplot(data=data,x=target, hue="StageID")
show()


# ## Rename bar labels

# In[ ]:


figure(figsize=(12,6))
ax = countplot(data=data,x=target)
ax.set_xticklabels(["1","2","3"])
show()


# ## Rename X label

# In[ ]:


figure(figsize=(12,6))
ax = countplot(data=data,x=target)
ax.set_xlabel("X Label Renamed!")
show()


# ## Rename Y Label

# In[ ]:


figure(figsize=(12,6))
ax = countplot(data=data,x=target)
ax.set_ylabel("Y Label Renamed!")
show()


# ## Add title to figure

# In[ ]:


from matplotlib.pyplot import suptitle
figure(figsize=(12,6))
suptitle("Enter Title Here")
ax = countplot(data=data,x=target)
show()


# ## Change yticks

# In[ ]:


fig = figure(figsize=(12,6))
ax = countplot(data=data,x=target)
ax.set_yticks([t*15 for t in range(0,16)])
show()


# ## Increase font size

# In[ ]:


from seaborn import set

set(font_scale=1.4)
fig = figure(figsize=(12,6))
ax = countplot(data=data,x=target)
show()


# ## Remove top and right spine/border.

# In[ ]:


from seaborn import despine

fig = figure(figsize=(12,6))
ax = countplot(data=data,x=target)
despine()
show()


# ## Remove grid

# In[ ]:


from seaborn import axes_style

with axes_style({'axes.grid': False}):
    fig = figure(figsize=(12,6))
    ax = countplot(data=data,x=target)
show()


# In[ ]:


# Change background colour


# In[ ]:


from seaborn import axes_style

with axes_style({'axes.facecolor': 'gold'}):
    fig = figure(figsize=(12,6))
    ax = countplot(data=data,x=target)
show()


# # Change grid line colour

# In[ ]:


from seaborn import axes_style

with axes_style({'grid.color': "red"}):
    fig = figure(figsize=(12,6))
    ax = countplot(data=data,x=target)
show()


# # Rotate x Labels

# In[ ]:


from matplotlib.pyplot import xticks

figure(figsize=(12,6))
countplot(data=data,x=target)
xticks(rotation=90)


# In[ ]:




