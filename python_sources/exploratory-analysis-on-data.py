#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt #plotting library
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns    #plotting library
sns.set(color_codes=True)


# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
train_data = pd.read_json("../input/train.json")
test_data = pd.read_json("../input/test.json")
display_count = 2
target = 'interest_level'


# In[ ]:


train_data.iloc[0]


# ###Calculate number of occurrences of a particular category###

# In[ ]:


def get_value_counts(col, df):
    return pd.DataFrame(df[col].value_counts())


# In[ ]:


target_values = list(train_data[target].unique())
target_groups = train_data.groupby(target)


# In[ ]:


global_chart_settings = {
    'height' : 4,             # height of chart
    'width' : 8,              # width of chart
    'bar_width' : 0.9,        # width of bar
    'title' : 'Number of occurrences of {0}', #default title
    'ylabel' : 'Occurrence',  #label of y axis
    'alpha' : None,           # alpha of chart(transparency factor)
    'lbl_fontsize' : 13,      # font size of labels
    'title_fontsize' : 13     # font size of title
}


# ### Function to plot a bar chart with customied settings ###

# In[ ]:


def plot_distributions(xcol, huecol, data, width, height):
    plt.figure(figsize=(width, height))
    sns.countplot(x=xcol, hue=huecol, data=data)
    plt.ylabel('Number of Occurrences', fontsize=12)
    plt.xlabel(xcol, fontsize=12)
    plt.show()
    plt.close()


# In[ ]:


def plot_seaborn_bar(df, column, ax, i, color, title, chart_styles):
    n = len(df.index)
    bar_locations = np.arange(n)
    ax[i, 0].bar(bar_locations, df[column], color=color)
    ax[i, 0].set_xticks(bar_locations + 0.1 / 2)
    ax[i, 0].set_xticklabels(df.index)
    ax[i, 0].set_ylabel(chart_styles['ylabel'], fontsize=chart_styles['lbl_font'])
    ax[i, 0].set_title(title, fontsize=chart_styles['title_font'])
    #ax[i, 0].set_facecolor('white')
    
    for x,y in zip(bar_locations, df[column]):
        ax[i, 0].text(x + 0.05, y + 0.01, '%.0f' % y, ha='center', va= 'bottom')


# In[ ]:


def plot_chart(subplot_count, dataframes, columns, colors, chart_styles, titles, chart_types):
    
    width = chart_styles['width']
    height = chart_styles['height']
    fig, ax = plt.subplots(subplot_count, figsize = (width, height), facecolor='w', squeeze=False)
    for i in range(subplot_count):
        if chart_types[i] == 'bar':
            plot_seaborn_bar(dataframes[i], columns[i], ax, i, colors[i], titles[i], chart_styles)
plt.show()
plt.close()


# ###Analyse bathrooms###

# In[ ]:


subplot_count = 2
colors = ['rosybrown', 'salmon']
columns = ['bathrooms', 'bathrooms']
chart_types = ['bar', 'bar']
dataframes = []
df = get_value_counts('bathrooms', train_data)
dataframes.append(df)
dataframes.append(df[df.index >= 3])
chart_styles = {
    'height' : 10,
    'width' : 13,
    'ylabel' : 'Count',
    'lbl_font' : 15,
    'title_font' : 15
}
titles = ['Occurrences of bathrooms', 'Occurrences of bathrooms from 3 and above']
plot_chart(subplot_count, dataframes, columns, colors, chart_styles, titles, chart_types)


# ***Occurrence is very high for 1 and 2 bathrooms***

# ***Occurrence count for 3 bathrooms and above are much smaller that 1 and 2 bathrooms.Factors contributing to this needs deeper exploration.***

# In[ ]:


subplot_count = len(target_values)
colors = ['blueviolet', 'plum', 'mediumvioletred']
columns = ['bathrooms', 'bathrooms', 'bathrooms']
chart_types = ['bar', 'bar', 'bar']
dataframes = []
titles = []
title = 'Bathroom count for target({})'

for value in target_values:
    df = get_value_counts('bathrooms', target_groups.get_group(value))
    dataframes.append(df)
    titles.append(title.format(value))
    
chart_styles = {
    'height' : 15,
    'width' : 10,
    'title' : 'Occurrences of {0}',
    'ylabel' : 'Count',
    'lbl_font' : 15,
    'title_font' : 15
}
plot_chart(subplot_count, dataframes, columns, colors, chart_styles, titles, chart_types)


# ***People have high inquiries for listings with 1 and 2 bathrooms.Similarly very low inquiries again for 1 or 2 bathrooms.We have to dig deeper into those listings***

# ###Analyse bedrooms###

# In[ ]:


subplot_count = 1
colors = ['indianred']
columns = ['bedrooms', 'bathrooms']
chart_types = ['bar']
dataframes = []
df = get_value_counts('bedrooms', train_data)
dataframes.append(df)
chart_styles = {
    'height' : 4,
    'width' : 8,
    'title' : 'Occurrences of {0}',
    'ylabel' : 'Count',
    'lbl_font' : 15,
    'title_font' : 15
}
titles = ['Occurrence count for bedrooms']
plot_chart(subplot_count, dataframes, columns, colors, chart_styles, titles, chart_types)


# In[ ]:


subplot_count = len(target_values)
colors = ['deepskyblue', 'royalblue', 'dodgerblue']
columns = ['bedrooms', 'bedrooms', 'bedrooms']
chart_types = ['bar', 'bar', 'bar']
dataframes = []
titles = []
title = 'Bedroom count for target({})'

for value in target_values:
    df = get_value_counts('bedrooms', target_groups.get_group(value))
    dataframes.append(df)
    titles.append(title.format(value))
    
chart_styles = {
    'height' : 15,
    'width' : 10,
    'title' : 'Occurrences of {0}',
    'ylabel' : 'Count',
    'lbl_font' : 15,
    'title_font' : 15
}
plot_chart(subplot_count, dataframes, columns, colors, chart_styles, titles, chart_types)


# In[ ]:


subplot_count = 3
colors = ['mediumpurple', 'plum', 'fuchsia']
columns = ['price', 'price', 'price']
chart_types = ['bar', 'bar', 'bar']

dataframes = []
df1 = train_data.groupby('bedrooms').agg({'price' : np.mean})
df2 = train_data.groupby('bathrooms').agg({'price' : np.mean})
df3 = train_data.groupby(target).agg({'price' : np.mean})
dataframes.append(df1)
dataframes.append(df2)
dataframes.append(df3)
chart_styles = {
    'height' : 14,
    'width' : 12,
    'ylabel' : 'Count',
    'lbl_font' : 15,
    'title_font' : 15
}
titles = ['Mean price across bedrooms', 'Mean price across bathrooms', 'Mean price across interest level']
plot_chart(subplot_count, dataframes, columns, colors, chart_styles, titles, chart_types)

