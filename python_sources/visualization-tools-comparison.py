#!/usr/bin/env python
# coding: utf-8

# # Visualization Practice for Beginner 

# ### My kernel will crash when I run plotly. I want to leave it unexcuted unitil I figure it out. As I concluded in the last paragraph, plotly is nice and cool but eats a lot of memory and crash my kernel all the time. 

# ## Why I want to do this
# [Mercari Price Suggestion Challenge](https://www.kaggle.com/c/mercari-price-suggestion-challenge#description)  is my first real Kaggle project. I want to treat it seriously. But I found I am still straggling with a lot things from EDA to visulization, regex and model construction. The first problem I encountered is that there are many ways to plot a figure. For example, if I want to make a barplot, I can use plt, pandas, plotly and seaborn. All of these modules have function to plot barplot. But which one is the easiest or best in demonstration? In the following notebook, I will try to make the same plot using different methods and summarize my feelings of them. 
# 
# I want to thank BuryBuryZymon for his amazing code at  https://www.kaggle.com/maheshdadhich/i-will-sell-everything-for-free-0-55
# 
# Also I want to thank ThyKhueLy for her excellent code about plotly https://www.kaggle.com/thykhuely/mercari-interactive-eda
# 
# Please let me know if I have made some stupid mistakes or can simplify the codes. I am eager to learn. Thanks.

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from scipy.stats import norm
from scipy import stats
from time import time
import plotly.offline as py
import plotly.graph_objs as go
py.init_notebook_mode(connected=True)
import plotly.tools as tls


# ## Load Data

# In[ ]:


train = pd.read_csv('../input/train.tsv', sep = '\t')
test = pd.read_csv('../input/test.tsv', sep = '\t')


# In[ ]:


train.describe()


# In[ ]:


train.head()


# ### First plot
# The first plot will be the log(price+1) distribution plot. In this plot, ThyKhuely didn't use plotly. She used pandas.DataFrame.plot.hist(). I think that's because there is not many interactive infomation here. I will try to use plotly to draw it.

# In[ ]:


fig, (ax1, ax2) = plt.subplots(1,2, figsize = (18,6))
#Using pandas.DataFrame.plot.hist() to draw histgram
start1 = time()
np.log(train['price']+1).plot.hist(bins=50, ax = ax1, edgecolor='white')
ax1.set_xlabel('log(price+1)', fontsize=12)
ax1.set_ylabel('frequency', fontsize=12)
ax1.tick_params(labelsize=15)
end1 = time()
#Using Seaborn to draw the distribution plot. The shape is the same. Just change frequency to density
start2 = time()
sns.distplot(np.log(train['price']+1),bins = 50, ax = ax2,kde = False)
ax2.tick_params(labelsize=15)
ax2.set_xlabel('log(price+1)', fontsize=12)
ax2.set_ylabel('frequency', fontsize=12)
end2 = time()


# In[ ]:


start3 = time()
data = [go.Histogram(x = np.log(train['price']+1), nbinsx = 50)]
py.iplot(data, filename = 'price_bar')
end3 = time()


# In[ ]:


print ('the time to excute plot1 is {}'.format(end1 - start1))
print ('the time to excute plot2 is {}'.format(end2 - start2))
print ('the time to excute plot3 is {}'.format(end3 - start3))


# ### Comparison
# 
# At first, I want to draw all three plots at the same row, but I couldn't use `plt.subplots` to set up the axes for plotly. I don't know if it's possible. Thus, I put plotly plot under the two easy plots. 
# 
# The excution times show that the seaborn used least time while the plotly is the most time consuming code. Also I looked at the size of saved iPynb file which is 100+ Mb for only one plotly plot. The interactive plot is good and cool, but does it worth the time and memory. Also I found my Mac was slow after running the plotly. I don't know if it's because of it. 
# 
# Another think I found confusing is that I set `nbinsx = 50` in plotly. But the shape of the plot is not what I have seen in the first two plots where I used `bins = 50`. I didn't add xlabel and ylabel in plotly plot because I don't want to wait another 16 seconds. 
# 
# To my surprice, the excution time for `pandas.DataFrame.plot.hist()` is 12s which is way larger than the excution time for `seaborn`. I don't know why. 

# ## Category count plot
# 
# The next plot I want to compare is to use `seaborn`, `pands.DataFrame.plot()` and `plotly` to draw number of items by different catagory. What I did first is to use regex to extract categories from `category_name`. I noticed there are three`\` in this column which corresponds to category1,2 and 3. The first category is the main category. 

# In[ ]:


train['cat1'] = train['category_name'].str.extract('([A-Z]\w{0,})',expand = True)
train['cat2'] = train['category_name'].str.extract('/(.*)/', expand = True)
train['cat3'] = train['category_name'].str.extract('/.+/(.*)', expand = True)


# In[ ]:


train.head()


# In[ ]:


cat1 = pd.DataFrame(train['cat1'].value_counts())
cat1_mean = pd.DataFrame(train.groupby('cat1').mean()['price'], index = cat1.index)
main_cat_sum =pd.concat([cat1, cat1_mean], axis = 1)
main_cat_sum.head()


# In[ ]:


start1 = time()
fig, (ax1,ax2) = plt.subplots(1,2,figsize = (18,6))
width = 0.8
main_cat_sum['cat1'].plot.bar(ax = ax1, width = width, color = 'b')
ax1.set_ylabel('count')
ax1.set_xlabel('Category')
ax1.set_title('Number of items by Category1')
end1 = time()

start2 = time()
sns.barplot(x= cat1.index, y= cat1['cat1'], ax = ax2)
ax2.set_ylabel('count')
ax2.set_xlabel('Category')
ax2.set_title('Number of items by Category1')
end2 = time()


# In[ ]:


start3 = time()
trace1 = go.Bar(x= cat1.index, y= cat1['cat1'] )
layout = dict(title= 'Number of Items by Main Category',
              yaxis = dict(title='Count'),
              xaxis = dict(title='Category'))
fig=dict(data=[trace1], layout=layout)
py.iplot(fig)
end3 = time()


# In[ ]:


print ('the time to run pandas.DataFrame.plot() is {}'.format(end1 - start1))
print ('the time to excute seaborn is {}'.format(end2 - start2))
print ('the time to excute plotly is {}'.format(end3 - start3))


# In[ ]:


train['log(price+1)'] = np.log(train['price']+1)


# In[ ]:


fig, (ax1, ax2) = plt.subplots(1,2,figsize = (18,6))
current_palette = sns.color_palette()
start1 = time()
train.boxplot(by = 'cat1', column = 'log(price+1)', ax = ax1, rot = 45)
ax1.set_ylabel('log(price+1)')
ax1.set_xlabel('Category')
ax1.set_title('log(price+1) of items by Category1')
end1 = time()
start2 = time()
sns.boxplot(y = 'cat1', x = 'log(price+1)',ax = ax2, data = train)
ax2.set_xlabel('log(price+1)')
ax2.set_ylabel('Category')
ax2.set_title('log(price+1) of items by Category1')
end2 = time()


# In[ ]:


start3 = time()
general_cats = train['cat1'].unique()
x = [train.loc[train['cat1']==cat, 'price'] for cat in general_cats]
data = [go.Box(x=np.log(x[i]+1), name=general_cats[i]) for i in range(len(general_cats))]
layout = dict(title="Price Distribution by General Category",
              yaxis = dict(title='Frequency'),
              xaxis = dict(title='Category'))
fig = dict(data=data, layout=layout)
py.iplot(fig)
end3 = time()


# In[ ]:


print ('the time to run pandas.DataFrame.plot() is {}'.format(end1 - start1))
print ('the time to excute seaborn is {}'.format(end2 - start2))
print ('the time to excute plotly is {}'.format(end3 - start3))


# I am sorry that I don't know how to fill the colors in to boxplot and how to convert the orientation to horizontal in `pandas.DataFrame.boxplot()`. If someone knows how to do it, please feel free to show me in the comments. I found running of plotly here to plot price boxplot is extremely unstable on my Mac. The notebook shut down automatically after 2 seconds of running the last chunk of code. I recorded the running time, clearly, the running time of plotly is increadible long. Although it looks very cool and attractive, I don't recommand plot such a large data using plotly since it may crash your computer. Or maybe it's just my laptop is too old. 

# ## Conclusion
# 
# To sum up, I used `pandas.DataFrame.plot()`, `seaborn` and `plotly` to plot three figures seperately. I wanted to try more plots but somehow my Mac ran very slowly after excuting three plotly plots. I have 8 Gb of memory and the python used about 2.6G even when I am not running any cells. If you want to do the EDA just for yourself, seeaborn, plt and plot function in pandas are all very convinient and quick. If you want to present your visualization products to someone else, the plotly can give you a excellent interactive interface where you can embed a lot of information in it. But in that way, you will sacrafise a lot of memory and time. Maybe some expert on Kaggle will have a better understanding than me. I hope you can point out my mistakes in this kernel since this is my first share. At last, hope you all get good rank on LB. 

# In[ ]:




