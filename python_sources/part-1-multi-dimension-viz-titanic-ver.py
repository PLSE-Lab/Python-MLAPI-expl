#!/usr/bin/env python
# coding: utf-8

# ## Multi-Dimension Visualization
# 
# ![](https://upload.wikimedia.org/wikipedia/commons/d/db/Titanic-Cobh-Harbour-1912.JPG)
# 
# 
# - **main library : matplotlib + seaborn**
# - **sub library: missingno**
# 
# > I am writing an information visualization book using a Python library in Korea. This notebook was created to test the visualizations in the book.
# 
# There are 7 kinds of information visualization. [link](http://www.interactiondesign.us/courses/2011_AD690/PDFs/Shneiderman_1996.pdf)
# 
# - 1D
# - 2D
# - 3D
# - Multi-dimension
# - Temporal (time series)
# - Tree 
# - Network
# 
# Here I am going to do 5 visualizations as follows, grouped by task a little differently.
# 
# - N dimension(1,2,3,..)
# - Time Series 
# - Tree
# - Network
# - Geographic
# 
# Now let's focus on the visualization library and see how we can achieve the desired visualization.

# ## 1. Why visualize information?
# 
# There are two main reasons for visualization.
# 
# 1. Insights to readers(including you)
# 2. Fast decision making
# 
# Currently, there is a lot of visualization/infographic related to **Corona 19**, and I think everyone knows the insights that visualization provides.
# 
# They give intuition and help that can't be achieved through figures such as *warnings, good / bad growth, cross-country comparisons, policy effectiveness, and etc.*
# 
# Also, for kagglers (machine learning / deep learning developers), it can help with feature extraction, feature selection, and result verification.
# 
# 
# ## 2. What would you like to show?
# 
# Representatively, we see a total of four.
# 
# - Composition
# - Distribution
# - Comparison
# - Relationship
# 
# <figure>
# <img src = "https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F2332181%2Fa79590d04a5b86c4178f56f2d6e655ac%2FScreen%20Shot%202019-09-02%20at%2015.55.29.png?generation=1567428955573342&alt=media" width=70%/> 
#     <figcaption> <a href="https://extremepresentation.typepad.com/files/choosing-a-good-chart-09.pdf">original link</a></figcaption>
# </figure>
# 
# And here are only three charts you should know.
# 
# - **Bar graph**
# - **Line graph**
# - **Scatter plot**
# 
# Speaking a little differently, it means that the visualization proceeds with three components: **point, line, and plane**.
# 
# Data visualization is, after all, **a many-to-many mapping of data and graphic elements.** Now let's go through the visualization to see how to give meaning.

# ## 3. A Quick look at Data
# 
# We should always look at the data first. Here's what you'll see at the start:
# 
# - **Missing value**: It is possible to visualize the biased content, and there is a high possibility of errors in the library.
# - **Number of data**: Error may occur depending on time and library
# - **Number of features**: Comparable visualization
# - **Data type**: Select the appropriate visualization
#     - **Numerical** 
#         - continuous
#         - discrete
#     - **Categorical**
#         - norminal
#         - ordinal
#         
# 
# ### 3-1. Library for default setting

# In this notebook, we use `matplotlib` and `seaborn` for visualization.
# 
# Since `seaborn` is based on `matplotlib`, it is all possible with `matplotlib`, but I will use `seaborn`, which is useful for statistical information visualization.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib as mpl
import matplotlib.pyplot as plt 
from matplotlib import gridspec # flexible multi figure size
 
import seaborn as sns

print('matplotlib : ', mpl.__version__)
print('seaborn : ', sns.__version__)
print('pandas : ', pd.__version__)

plt.rcParams['figure.dpi'] = 200


# In[ ]:


# load dataset by pandas's read_csv function
data = pd.read_csv('/kaggle/input/titanic/train.csv')
print(data.shape)


# There are a total of 12 features.
# 
# The table is also a visualization. Let's take a look at some of the data and see how it is organized.

# In[ ]:


data.head()


# ### 3-2. Missing data check
# 
# - [Exploring incomplete data using visualization techniques](https://personal.eur.nl/alfons/publications/ADAC2012.pdf)
# - [Missingno: a missing data visualization suite](https://joss.theoj.org/papers/10.21105/joss.00547)
#     - [github/missingno](https://github.com/ResidentMario/missingno)
# 
# There are many ways to visualize missing values.
# 
# I think the easiest and most intuitive way to do this is with **matrix plot**, and the `missingno` library makes it easy.

# In[ ]:


import missingno as msno
msno.matrix(data)


# You can see certain missing values in **Age**, **Cabin**, and **Embarked**.
# 
# Such data needs to be careful **not only in visualization, but also in forecasting.**
# 
# 

# In[ ]:


import missingno as msno
msno.matrix(data, sort='descending')


# It is also helpful later to sort by the number of missing values and check the row with the most null values.
# 
# Now let's check the data type and statistical information.

# In[ ]:


data.info()


# In pandas, `object` usually means when it contains non-numeric data.
# 
# `object` is likely to be categorical and can have a lot of work to preprocess.

# ## 4. 1-Dimension Visualization
# 
# > The dimension will vary depending on the definition, but I will define it as the number of features used in this notebook.
# 
# A single feature is good for looking at the composition and distribution of data.
# 
# So, before that, let's divide the type of data specifically.
# 
# - Categorical
#     - Norminal
#         - `Survived` : most important info
#         - `Sex`
#         - `Embarked`
#     - Ordinal
#         - `Pclass`
# - Numerical
#     - Contnuous
#         - `Age`
#         - `Fare`
#     - Discreate
#         - `SibSp`
#         - `Parch`
# - Etc
#     - Meaningless to viz : *Because it is all individual information*
#         - `Name` : Title information can be extracted, but left as a future task
#         - `Ticket` 
#         - `PassengerId`
#         - `Cabin`
# Let's visualize it by looking at the details of each data type.

# ### 4-1. Bar plot / Pie Chart
# 
# > [What's the difference between a graph, a chart, and a plot?](https://english.stackexchange.com/questions/43027/whats-the-difference-between-a-graph-a-chart-and-a-plot)
# 
# To see the overall composition, **Pie Chart** also gives visual fun, but it is **difficult to compare** because the axes are not fixed.
# 
# So, let's compare the two by applying them to `survived` feature.

# In[ ]:


# count the value first!
survived_count = data['Survived'].value_counts()
print(survived_count)


# In[ ]:


# No Custom Version
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].bar(survived_count.index, survived_count) # bar(x_label, y_label)
axes[1].pie(survived_count) # pie chart

plt.show()


# You can use the following elements to customize it a bit.
# 
# - color palette
# - x,y axis
# - width
# - little notation

# In[ ]:


# Custom Version
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Custom Color Pallete

color = ['gray', 'lightgreen']  # To express the meaning of survival
new_xlabel = list(map(str, survived_count.index))

# Axes[0] : Bar Plot Custom
axes[0].bar(new_xlabel, # redefine for categorical x labels 
            survived_count, 
            color=color,# color
            width=0.65, # bar width 
            edgecolor='black', # bar color
            # linewidth=1.5 # edge width
        ) 

axes[0].margins(0.2, 0.2) # margin control (leftright, topbottom)
axes[0].set_xlabel('Survived') # label info

# Axes[0] : Pie Chart Custom
explode = [0, 0.05]

axes[1].pie(survived_count,
            labels=new_xlabel,
            colors=color, # color
            explode=explode, # explode
            textprops={'fontsize': 12, 'fontweight': 'bold'}, # text setting
            autopct='%1.1f%%', # notation
            shadow=True # shadow
           )

fig.suptitle('[Titanic] Bar Plot vs Pie Chart', fontsize=15, fontweight='bold') # figure scale title

plt.show()


# - In addition to labels, **color is used to improve readability**.
# 
# - You can highlight the goals of your visualization by **adding titles** to **figures or axes**.
# 
# - In the case of bar plots, it is also good to **make margins** appropriately because too few margins are difficult to read.
# 
# - Pie charts can see the **whole as 1**, so indicating **percentages** is a bit easier to understand.
# 
# - Pie charts can use `explode` to highlight what you want to emphasize.
# 
# > There is no right answer for data visualization. If you have 100 readers, you may have 100 different tastes, so you need to practice constantly thinking and customizing.

# ### 4-2. Categorical features' Distribution
# 
# Let's look at the distribution of the remaining categorical data as in the above method.
# 
# - `Survived`
# - `Sex` : Similar to Survived
# - `Embarked` : Order does not matter (Sort by size is recommended)
# - `Pclass` : Order matter
# 
# First, let's count and proceed.

# In[ ]:


categorical_features = ['Sex', 'Embarked','Pclass']

for feature in categorical_features:
    print(f'[{feature}]')
    print(data[feature].value_counts(), '\n')


# In order to visualize with matplotlib, you need to count by `value_counts` and set it separately on x-axis and y-axis. 
# 
# This process is a bit annoying, so let's visualize it more conveniently using `seaborn`.
# 
# `countplot` makes existing preprocessing very easy.

# In[ ]:


# Custom Version
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

# Sex
sns.countplot(data['Sex'], ax=axes[0])

# Embarked
sns.countplot(data['Embarked'], ax=axes[1])

# Pclass
sns.countplot(data['Pclass'], ax=axes[2])

    
plt.show()


# The color is automatically selected from the color pallete, and the ticks are also well written.
# 
# However, there are some inconveniences to readability, and we will fix it.
# 
# The fix also has the functionality of seaborn, but it can also be done with matplotlib.
# 
# - **margin**
# - **Overlap of yticks** 
#     - Since the numbers have similar scales, let's keep only the leftmost y-axis.
#     - use `sharey` parameter or use `ylim`(yticks range) parameter
# - **Repeat color palette** (In the end, you have to set it yourself.)
#     - You can keep `sex` and `embarked`, but repetition of the same color palette decreases readability. However, if you use too many colors, it can also reduce readability, so when visualizing multiple graphs, keeping it in a single color can be one way.
#     - The color of the pclass seems to adjust the brightness gradually to express the order rather than making the hue completely different.

# In[ ]:


sns.set_style("whitegrid")

# Custom Version
fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)

# Sex
# New Palette
sns.countplot(x='Sex', data=data, ax=axes[0], palette="Set2", edgecolor='black') 

# Embarked
# Fixed Color
sns.countplot(data['Embarked'], ax=axes[1], color='gray', edgecolor='black') 

# Pclass
# Gradient Palette
sns.countplot(data['Pclass'], ax=axes[2], palette="Blues", edgecolor='black') 

# Margin & Label Custom
for ax in axes : 
    ax.margins(0.12, 0.15)
    # you can set axis setting like this
    ax.xaxis.label.set_size(12)
    ax.xaxis.label.set_weight('bold')
    
# figure title    
fig.suptitle('[Titanic] Categorical Distribution', 
             fontsize=16, 
             fontweight='bold',
             x=0.05, y=1.08,
             ha='left' # horizontal alignment
            ) 

plt.tight_layout()
plt.show()


# - `seaborn` can be drawn by selecting the x-axis when there is a dataset. (in `pandas DataFrame`)
# - Color can be set by color or palette.
#     - [Named Colors](https://matplotlib.org/gallery/color/named_colors.html)
#     - [Choosing Colormaps in Matplotlib](https://matplotlib.org/tutorials/colors/colormaps.html)
#     - Alternatively, you can use a custom palette after creating a color or rgb format according to the palette format.   
# - The location of the title can be customized as desired.
#     - All text in matplotlib can be customized with coordinates.
#     - Text is easy to customize if you understand weight, size, and family. (Close to web development / PPT)
# - `plt.tight_layout()` can narrow the distance between axes to eliminate the obscure margin.
# - Both seaborn and matplotlib can pick the theme early.
#     - The overall palette, grid, and font size are set.
#     - `plt.style.use()` : ggplot ...
#     - `sns.set_style()` : whitegrid ...
#     - Even if you choose this well, you can proceed with a pretty visualization.
#     - You can choose the style you want by looking at it yourself!
#     - You can also initialize the settings using `mpl.style.use ('default')`.
#     
# Last reset and proceed to the next turn.

# In[ ]:


mpl.style.use ('default')


# ### 4-3. Numerical features Distribution
# 
# Numerical data, among other things, data with **continuous** meaning must be visualized in order.
# 
# This is the detailed difference between **histogram** and **bar plot**. [more read](https://keydifferences.com/difference-between-histogram-and-bar-graph.html)
# 
# The histogram can be drawn with `hist` in `matplotlib` or `distplot` in `seaborn`.
# 
# I prefer to use any good tools, so I'll use `seaborn`.
# 
# - `Age` : Null is included, but it is omitted by itself.
# - `Fare` 
# 
# --- 
# 
# - `SibSp`
# - `Parch`

# In[ ]:


# No custom
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.distplot(data['Age'], ax=axes[0], kde=False)
sns.distplot(data['Fare'],  ax=axes[1], kde=False)
plt.tight_layout()
plt.show()


# This distribution can be seen, but there are various difficulties.
# 
# - The division is unclear.
# - If the range is long and there are many divided sections, it is difficult to see.
#     - Just adjust the `bins` parameter.
# 
# However, it seems that there is nothing that can be of great help to the details of the visualization here.
# 
# The title and margins have been touched above, so I'll skip this one first.
# 
# Let's cover this a little more in 2D visualization.

# ## 5. 2-Dimension Visualization
# 
# In fact, it's easy to look at the distribution of a feature in the data. Custom can also improve readability a bit, but it doesn't mean much.
# Now you need to be able to look at your data to better fit your goals.
# 
# The reason to look at the data is **the relationship between features and the relationship between features and targets.**
# Now let's take a look at the data more interestingly.
# 
# - 2 axis
#     - swarm plot, violin plot, box plot (categorical * numerical)
#     - scatter plot (numerical & numerical)
#         - joint plot
# - use `hue` + `label` 
# - heatmap
# - Multi-Plot

# ### 5-1. 2 Axis : Scatter Plot, Swarm plot, Violin Plot, Box Plot
# 
# The easiest way to view two or more features is by matching feature 1 on one axis(x) and feature 2 on the other axis(y).
# 
# However, here, categorical variables can lead to misunderstanding, so first, draw the ordered variables.
# 
# In order to view the distribution by category, the order may not matter. Let's first look at the relationship between **categorical and numerical** features.

# In[ ]:


fig, axes = plt.subplots(1, 2, figsize=(9, 4))

sns.scatterplot(x='Sex', y='Age', data=data, ax=axes[0]) # ax.scatter(data['Sex'], data['Fare'])
sns.scatterplot(x='Sex', y='Age', data=data, ax=axes[1], alpha=0.05)

for ax in axes : ax.margins(0.3, 0.1)
plt.show()


# This allows you to look at the original **density**.
# 
# In fact, it is difficult to understand or distribute the data because it is simply printed in *a straight line*.
# 
# Of course, you can adjust the transparency with `alpha` as shown on the right, but it is still poorly readable.
# 
# Now, I will introduce plots that can solve this.

# In[ ]:


sns.set_style("whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(9, 10), sharey=True)

sns.stripplot(x='Sex', y='Age', data=data, ax=axes[0][0])
sns.swarmplot(x='Sex', y='Age', data=data, ax=axes[0][1])
sns.violinplot(x='Sex', y='Age', data=data, ax=axes[1][0])
sns.boxplot(x='Sex', y='Age', data=data, ax=axes[1][1])

# Tips for turning multiple plots into loops
# use reshape!
for ax, title in zip(axes.reshape(-1), ['Strip Plot', 'Swarm Plot', 'Violin Plot', 'Box Plot'] ): 
    ax.set_title(title, fontweight='bold')
    
plt.show()


# - `Strip Plot` : scatter with width and overlap 
# - `Swarm Plot` : scatter + no overlap (kde : kernel density estimate)
# - `Violin Plot` : swarm to kde
# - `Box Plot` : kde to rectangle (quartile, outliers notation)
# 
# First of all, among these, `stripplot` is rarely used. Usually, `violinplot` or `boxplot` is used to check the distribution.
# 
# 

# ## It is still being updated. If you like it, please upload it.

# In[ ]:





# In[ ]:




