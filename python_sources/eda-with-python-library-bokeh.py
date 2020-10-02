#!/usr/bin/env python
# coding: utf-8

# There are already many versions of EDA kernels done by many Kagglers for Avito competition. This is my first attempt at EDA kernel in Kaggle. I have tried not to repeat few basics which are present in other Kernel.
# I have used Bokeh for the Visualization in Python. I am not an expert in Bokeh, i have started learning it for this kernel. I will continue to refine this kernel with more features and explorations. Please feel free to point out any corrections or improvements.

# # <a id='lib'>1. Load libraries</a>

# In[4]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from bokeh.models import LinearAxis, Range1d
from bokeh.transform import dodge
from bokeh.core.properties import value

from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.io import output_notebook, show

from wordcloud import WordCloud

import os
# print(os.listdir("../input"))


# In[5]:


np.set_printoptions(suppress=True)


# In[6]:


output_notebook()


# # <a id='data'>2. Import data</a>

# In[7]:


train = pd.read_csv('../input/train.csv', parse_dates = ['activation_date'])
test = pd.read_csv('../input/test.csv', parse_dates = ['activation_date'])
periods_train = pd.read_csv('../input/periods_train.csv', parse_dates = ['activation_date', 'date_from', 'date_to'])
# train_active = pd.read_csv('../input/train_active.csv')


# In[8]:


print('Number of Observations in train is {0} and number of columns is {1}'.format(train.shape[0], train.shape[1]))
print('Number of Observations in periods_train is {0} and number of columns is {1}'.format(periods_train.shape[0], periods_train.shape[1]))


# # <a id='overview'>3. Data Overview</a>

# In[9]:


print('A sample of train data')
train.head()


# In[10]:


print('A sample of period_train data')
periods_train.head()


# In[11]:


""" 
Function to highlight rows based on data type
"""
def dtype_highlight(x):
    if x['type'] == 'object':
        color = '#2b83ba'
    elif (x['type'] == 'int64') | (x['type'] == 'int32'):
        color = '#abdda4'
    elif (x['type'] == 'float64') | (x['type'] == 'float32'):
        color = '#ffffbf'
    elif x['type'] == 'datetime64[ns]':
        color = '#fdae61'
    else:
        color = ''
    return ['background-color : {}'.format(color) for val in x]

train_dtypes = pd.DataFrame(train.dtypes.reset_index())
train_dtypes.columns = ['column', 'type']
periods_train_dtypes = pd.DataFrame(periods_train.dtypes.reset_index())
periods_train_dtypes.columns = ['column', 'type']

train_dtypes.style.apply(dtype_highlight, axis = 1)


# Take Aways:
# 1. More Categorical independent features than numeric features

# In[14]:


periods_train_dtypes.style.apply(dtype_highlight, axis = 1)


# Since Image_top_1 is a classification code for image, i am considering it as descrete values

# In[15]:


train.image_top_1 = train.image_top_1.astype(object)


# # <a id='summary'>4. Basic Summary</a>

# ### <a id='non_num'>4.1 Non-Numeric columns summary</a>

# Let's get basic summary of categorical columns in train. For simplicity, i have ordered the result based on count of unique values in each column.

# In[ ]:


desc = train.describe(include=['O']).sort_values('unique', axis = 1)


# In[ ]:


def highlight_row(x):
    if x.name == 'unique':
        color = 'lightblue'
    else:
        color = ''
    return ['background-color: {}'.format(color) for val in x]


# In[ ]:


desc.style.apply(highlight_row, axis =1)


# Take Aways:
#     1. Only 2 columns with less than 10 distinct values and rest have 28 to 1733 distinct values - So target encoding with smoothing might help as mentioned in this kernel from Porto Seguro competition- https://www.kaggle.com/aharless/xgboost-cv-lb-284
#     2. Significant missing values in param columns and not all items have description and image

# In[ ]:


# Since image column is a ID code for image, let's remove it for time being
train.drop('image', axis = 1, inplace = True)


# ### <a id='num'>4.2 Numeric columns summary</a>

# In[ ]:


def color_zero_red(val):
    """
    Takes a scalar and returns a string with
    the css property `'background-color: red'` for negative
    strings, black otherwise.
    """
    color = 'red' if val == 0 else ' '
    return 'background-color: %s' % color


# In[ ]:


# Summary of numeric columns
train.describe().style.applymap(color_zero_red)


# Missing Values in each column

# In[ ]:


# Missing values in each column
train.isnull().sum().sort_values(ascending =False)


# # <a id='target'>5. Target column distribution</a>

# In[ ]:


train.deal_probability.describe()
(train.deal_probability ==0).sum()/train.shape[0]


# Exactly 64.8% of the values in deal_probability are zero. Since these zero values skews a lot, let's look at the distribution after removing 0

# In[ ]:


non_zero_probability = train.deal_probability[train.deal_probability !=0]

hist, edges = np.histogram(non_zero_probability, 
                               bins = 50, 
                               range = [0, 1])

hist_edges = pd.DataFrame({'#items': hist, 
                       'left': edges[:-1], 
                       'right': edges[1:]})
hist_edges['cumulative_items'] = hist_edges['#items'].cumsum()
hist_edges['p_interval'] = ['%.2f to %.2f' % (left, right) for left, right in zip(hist_edges['left'], hist_edges['right'])]

src = ColumnDataSource(hist_edges)


# In[ ]:


hover1 = HoverTool(tooltips=[
    ("probability interval", "@p_interval"),
    ("#Items", "$y")
])

p1 = figure(title="deal_probability histogram",  y_axis_label='No.of.Items', x_axis_label='probability', tools = [hover1], background_fill_color="#E8DDCB")
p1.title.align = 'center'
p1.left[0].formatter.use_scientific = False
p1.below[0].formatter.use_scientific = False
p1.quad(top='#items', bottom=0, left='left', right='right',
        fill_color="#036564", line_color = "#2B2626", source =src)
show(p1)


# In[ ]:


hover2 = HoverTool(tooltips=[
    ("probability <=", "@right"),
    ("#Items", "$y")
])

p2 = figure(title="deal_probability cumulative",  y_axis_label='No.of.Items', x_axis_label='probability', tools = [hover2], background_fill_color="#E8DDCB")
p2.title.align = 'center'
p2.left[0].formatter.use_scientific = False
p2.below[0].formatter.use_scientific = False
p2.quad(top='cumulative_items', bottom=0, left='left', right='right',
        fill_color="#036564", line_color = "#2B2626", source =src)
# p.line('left', 'cumulative_items', line_color="#9E3030", source = src)
show(p2)


# Take Aways:
#     1. After removing zero probability, we can clearly see that the distribution is bimodal with values concentrated around 0.15 and 0.80

# # <a id='user_item'>6. user_id and item_id distribution</a>

# In[3]:


user_item = train.groupby('user_id')['item_id'].count().sort_values()
user_item_dist = user_item.value_counts().reset_index()
user_item_dist.columns = ['No.of Ads', 'No.of Users']

user_item_dist['pct'] = user_item_dist['No.of Users']/user_item_dist['No.of Users'].sum()


# Distribution of number of Users for each count of Ad post

# In[ ]:


user_item_dist.head()


# In[ ]:


user_item_dist.tail()


# In[ ]:


user_item_dist[user_item_dist['No.of Ads'] == 1].loc[:,'pct']


# Take Aways:
# 1. It can be observed that there is very high skew in the distribution. 68.9 % of the users have posted exactly once. Similarly, a exact 1080 Ads have been posted by exactly 1 User.

# In[ ]:


user_item_dist[user_item_dist['No.of Users'] == 1].loc[:,'No.of Ads'].max()


# A maximum 1080 of Ads have been posted by exactly one user

# In[ ]:


user_item_dist[user_item_dist['No.of Users'] == 1].loc[:,'No.of Ads'].min()


# A minimum of 101 Ads have been posted by exactly one user

# Plot the distribution after removing outlier cases

# In[ ]:


user_item_dist2 = user_item_dist[(user_item_dist['No.of Ads'] > 1) & (user_item_dist['No.of Users'] > 1)]

hover3 = HoverTool(tooltips=[
    ("No.of Ads ==", "@right"),
    ("Percent of Users", "$y")
])

p3 = figure(title="Distribution of No. of Ads posted by users",  y_axis_label='No.of.Users', x_axis_label='No.of Ads posted', tools = [hover3], background_fill_color="#E8DDCB")
p3.title.align = 'center'
p3.left[0].formatter.use_scientific = False
p3.below[0].formatter.use_scientific = False
p3.quad(top= user_item_dist2['pct'], bottom=0, left= user_item_dist2['No.of Ads'][:-1], right= user_item_dist2['No.of Ads'][1:],
        fill_color="#036564", line_color = "#2B2626")
show(p3)


# # <a id='no.ads_prob'>7. Distribution of deal probablity wrt No.of Ads posted by users</a>

# Let's add the No.of Ads posted feature as along with the train and check how deal probability varies with it. 
# 
# Note: While creating No.of Ads features i am not taking activation_date into consideration, i am using the entire data and creating the feature. In a way it indicates if one is a spammer/frequent/occasional user or not

# In[ ]:


user_item = user_item.reset_index()
user_item.columns = ['user_id', 'No.of Ads']


# In[ ]:


train = train.merge(user_item, on = 'user_id', how = 'left')
train['No.of Ads bin'] = pd.cut(train['No.of Ads'], 20, labels = range(20))

Ads_bin_prob = train.groupby(['No.of Ads bin'])['deal_probability'].mean().reset_index()
Ads_bin_prob.columns = ['No.of Ads bin', 'avg_deal_probability']


# In[ ]:


hover4 = HoverTool(tooltips=[
    ("Ads bin ", "@right"),
    ("avg_deal_probability", "$y")
])

p4 = figure(title="Distribution of No. of Ads posted by users",  y_axis_label='No.of.Users', x_axis_label='No.of Ads posted', tools = [hover4], background_fill_color="#E8DDCB")
p4.title.align = 'center'
p4.left[0].formatter.use_scientific = False
p4.below[0].formatter.use_scientific = False
p4.quad(top= Ads_bin_prob['avg_deal_probability'], bottom=0, left= Ads_bin_prob['No.of Ads bin'][:-1], right= Ads_bin_prob['No.of Ads bin'][1:],
        fill_color="#036564", line_color = "#2B2626")
show(p4)


# Take Aways:
#     1. User belonging to group which posts less than 10 Ads overall has highest average deal_probability followed by users who posts 70-80 Ads
#     
# It remains to be seen if this features will come out significant in the model

# In[ ]:


train.region.nunique()


# # <a id='usr_typ_deal'>8. User Type histogram and deal probability</a>

# Only user_type and parent_category_name had less than 10 distinct values

# In[ ]:


f = {'deal_probability':['mean'], 'item_id': ['size']}
user_hist_prob = train.groupby('user_type').agg(f).reset_index()
user_hist_prob.columns = ['user_type','avg_deal_probability', 'user_type_count']


# In[ ]:


user_hist_prob


# In[ ]:


user_type_source = ColumnDataSource(data=user_hist_prob)


# In[ ]:


p5 = figure(x_range = list(user_hist_prob.user_type), plot_width=800, plot_height=400, title = 'user_type distribution and deal_probability mean')
p5.vbar(x = dodge('user_type', -0.20, range=p5.x_range), top = 'user_type_count', width=.4, color='#f45666', source = user_type_source, legend = value('user_type_count'))
p5.y_range =  Range1d(0, user_hist_prob.user_type_count.max())
p5.extra_y_ranges = {"avg_deal_probability": Range1d(start=0, end=1)}
p5.xaxis.axis_label = 'user_type'
p5.yaxis.axis_label = 'user_type_count'
p5.add_layout(LinearAxis(y_range_name="avg_deal_probability", axis_label= 'avg deal_probability'), 'right')
p5.vbar(x = dodge('user_type', 0.20, range=p5.x_range), top = 'avg_deal_probability', y_range_name='avg_deal_probability', width = 0.4, color='lightblue', source = user_type_source, legend = value('avg_deal_probability'))
p5.legend.location = "top_left"
p5.legend.orientation = "horizontal"
show(p5)


# # <a id='par_cat_deal'>9. parent_category_name histogram and deal probability</a>

# In[ ]:


f = {'deal_probability':['mean'], 'item_id': ['size']}
parent_cat_hist_prob = train.groupby('parent_category_name').agg(f).reset_index()
parent_cat_hist_prob.columns = ['parent_category_name','avg_deal_probability', 'parent_category_count']


# In[ ]:


parent_cat_hist_prob


# In[ ]:


parent_cat_source = ColumnDataSource(data=parent_cat_hist_prob)


# In[ ]:


p6 = figure(x_range = list(parent_cat_hist_prob.parent_category_name), plot_width=800, plot_height=400, title = 'parent_category distribution and deal_probability mean')
p6.vbar(x = dodge('parent_category_name', -0.16, range=p6.x_range), top = 'parent_category_count', width=.3, color='#f45666', source = parent_cat_source, legend = value('parent_category_count'))
p6.y_range =  Range1d(0, parent_cat_hist_prob.parent_category_count.max())
p6.extra_y_ranges = {"avg_deal_probability": Range1d(start=0, end=1)}
p6.xaxis.axis_label = 'parent_category_name'
p6.yaxis.axis_label = 'parent_category_count'
p6.add_layout(LinearAxis(y_range_name="avg_deal_probability", axis_label= 'avg deal_probability'), 'right')
p6.vbar(x = dodge('parent_category_name', 0.16, range=p6.x_range), top = 'avg_deal_probability', y_range_name='avg_deal_probability', width = 0.3, color='lightblue', source = parent_cat_source, legend = value('avg_deal_probability'))
p6.legend.location = "top_left"
p6.legend.orientation = "horizontal"
show(p6)


# In[ ]:


train.region.value_counts()

