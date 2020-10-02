#!/usr/bin/env python
# coding: utf-8

# In[16]:


import pandas
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


# In[17]:


df = pandas.read_csv("../input/march18_myspeed.csv")


# Setting 'na' values for Signal Strength to None. Some of the values for Signal Strength are 'na' which means not available since the device was not able to capture signal level.

# In[18]:


# set na to None
df.loc[df['Signal_strength'] == 'na', 'Signal_strength'] = None
df.loc[df.isnull()['Signal_strength']]

# convert 'Signal_strength' to float
df['Signal_strength'] = pandas.to_numeric(df.loc[:,'Signal_strength'])


# In[19]:


print(df.head())


# In[20]:


print(df.info())


# In[21]:


print(df.describe())


# In[22]:


df.isnull().sum()


# In[23]:


columns = ['Technology', 'Test_type', 'Service Provider', 'LSA']

for c in columns:
    v = df[c].unique()
    g = df.groupby(by=c)[c].count().sort_values(ascending=True)
    r = np.arange(len(v))
    print(g.head())
    plt.figure(figsize = (6, len(v)/2 +1))
    plt.barh(y = r, width = g.head(len(v)))
    total = sum(g.head(len(v)))
    print(total)
    for (i, u) in enumerate(g.head(len(v))):
        plt.text(x = u + 0.2, y = i - 0.08, s = str(round(u/total*100, 2))+'%', color = 'blue', fontweight = 'bold')
    plt.margins(x = 0.2)
    plt.yticks(r, g.index)
    plt.show()    


# Basic bar plots of various categories in dataset. Number of times a category appears in given column.
# 
# * In technologies, 4G comprises of almost 90% of the dataset while 3G is only 10%. Hence number of people with 3G are much smaller compared to 4G users.
# * Upload and Download tests are almost equal as they should be.
# * In Service Providers JIO is currently dominating with 55.35% of whole dataset that is greater than all other service providers put together.
# * State wise, most samples occur in Maharashtra while least samples occur in Chennai.

# In[24]:


def sel(df, column_name, value):
    data = df.loc[(df[column_name] == value)]
    return data

pandas.DataFrame.mask = sel


# In[25]:


def plot_graphs(provider, state):
    # plot distributions of speeds
    #columns = ['Technology', 'Test_type', 'Service Provider', 'LSA']
    data4g = df.mask('Test_type', 'Download').mask('Technology', '4G')
    data3g = df.mask('Test_type', 'Download').mask('Technology', '3G')
    
    if provider != 'All':
        data4g = data4g.mask('Service Provider', provider)
        data3g = data3g.mask('Service Provider', provider)
    
    if state != 'All':
        data4g = data4g.mask('LSA', state)
        data3g = data3g.mask('LSA', state)
        
    x1 = data4g['Data Speed(Mbps)']
    x2 = data3g['Data Speed(Mbps)']

    fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (9, 5))
    #print(x1)
    axes[0].clear()
    axes[0].hist(x1, bins=100, label = '4G', normed = True)
    axes[0].axvline(x1.mean(), color = 'k', linewidth = 1, label = 'avg:'+str(round(x1.mean(), 2)))
    axes[0].legend(loc = 'upper right')
    axes[0].set_xlabel('Count')


    # print(x2)
    axes[1].clear()
    axes[1].hist(x2, bins=100, label = '3G', color = 'g', normed = True)
    axes[1].axvline(x2.mean(), color = 'k', linewidth = 1, label = 'avg:'+str(round(x2.mean(), 2)))
    axes[1].legend(loc = 'upper right')
    axes[1].set_xlabel('Count')

    fig.canvas.set_window_title('Provider-' + provider + ' ' + 'State-' + state)
    #plot both histogram in one figure
    # plt.figure(figsize = (9, 5))
    # plt.hist(x1, bins=100, label = '4G', alpha = 0.5, normed = True)
    # plt.axvline(x1.mean(), linewidth = 1, color = 'b', label = 'avg:'+str(round(x1.mean(), 2)))
    # plt.legend(loc = 'upper right')
    # plt.hist(x2, bins=100, label = '3G', color = 'g', alpha = 0.5, normed = True)
    # plt.axvline(x2.mean(), linewidth = 1, color = 'g', label = 'avg:'+str(round(x.mean(), 2)))
    # plt.legend(loc = 'upper right')

    plt.suptitle('State-' + state + "    " + 'Provider-' + provider, fontsize = 16)
    plt.show()


# Widgets are added to plot distributions of 4g and 3g speeds for different states and providers. The vertical line gives average speed of the distribution. This distribution gives answer to certain anamolies that appear in last section where we do visualisation of average speeds for various service providers.
# 
# From seeing the distributionns, it is obvious that the distribution of speeds over customers decreases exponentially as Download Speeds increases. Hence more customers are provided with lower speeds while the number of customers with higher speeds is much less. As speed increases, the number of customer having that speed decreases rapidly.
# 
# Some providers do not provide 4g, hence an empty graph is plotted with average speed set to nan. Example include Aircel, Cellone etc. which do not provide 4g. Similarly Jio that do not provide 3g.

# In[26]:


import ipywidgets as widgets
from ipywidgets import HBox
state_select = widgets.Dropdown(
    options=['All', 'North East', 'Kolkata', 'Bihar', 'Chennai', 'Jammu & Kashmir', 'Delhi',
       'Tamil Nadu', 'Maharashtra', 'Punjab', 'UP East', 'Rajasthan',
       'Gujarat', 'West Bengal', 'Mumbai', 'Kerala', 'Andhra Pradesh',
       'UP West', 'Orissa', 'Assam', 'Madhya Pradesh', 'Karnataka', 'Haryana',
       'Himachal Pradesh'],
    value='All',
    description='States:',
    disabled=False,
)

provider_select = widgets.Dropdown(
    options=['All', 'JIO', 'VODAFONE', 'AIRTEL', 'IDEA', 'CELLONE', 'UNINOR', 'DOLPHIN', 'AIRCEL'],
    value='All',
    description='Provider:',
    disabled=False,
)


def on_change(change):
    if change['type'] == 'change' and change['name'] == 'value':
        plot_graphs(provider_select.value, state_select.value)


state_select.observe(on_change)
provider_select.observe(on_change)

hb = HBox([state_select, provider_select])
display(hb)
plot_graphs(provider_select.value, state_select.value)


# In[27]:


#2d histograms
#columns = ['Technology', 'Test_type', 'Service Provider', 'LSA']
import matplotlib.colors as colors

data = df.dropna().mask('Test_type', 'Download').mask('Technology', '4G')
#print(data.isnull().sum())
x = data['Signal_strength']
y = data['Data Speed(Mbps)']
plt.hist2d(x, y, bins = 40, norm=colors.LogNorm())
plt.ylabel('Data Speed(Mbps)')
plt.xlabel('Signal_strength')
plt.show()


# Here is a 2D histogram plotted for Download Speed of 4g Networks vs Signal Strength.
# 
# * The max Speed increases as Signal Strength increases.
# * Optimal values of max speeds occur in interval [-80, -70] of signal strength.
# * Most of the sample lies in [-110, -80] (Yellow Color), but the speeds of most sample is much lower. In this interval most of the customers are having much lower speeds. As speed increases, the number of customers decreases. Hence it is less likely to get a higher speed with signal strength in this interval.
# * While in interval [-80, -70], which is also an optimal interval for max speeds, as you move upwards towards increasing speeds, number of customers do not decrease exponentially. Hence this interval can be said to be optimal in terms of speed distribution. As you are more likely to get a higer speed in this interval.
# * After  -70, max speeds again start decreasing.
# * The various rows where signal strength is set to None are dropped in this Histogram.

# In[28]:


# avg speeds of states and service providers
#columns = ['Technology', 'Test_type', 'Service Provider', 'LSA']

state = 'LSA'
service = 'Service Provider'
speed = 'Data Speed(Mbps)'

values = df[state].unique()
r = np.arange(len(values))

plt.figure(figsize = (8, len(values)/2 +1))
plt.xlabel(speed)


# 4g
data = df.mask('Test_type', 'Download').mask('Technology', '4G')
group = data.groupby(by=state)[speed].mean().sort_values(ascending = True)
plt.barh(y = r, width = group.head(len(values)), label = '4G')
for (i, v) in enumerate(group.head(len(values))):
    plt.text(x = v + 0.2, y = i - 0.1, s = str(round(v, 2)), color = 'blue', fontweight = 'bold')
plt.yticks(r, group.index)

# 3g
data = df.mask('Test_type', 'Download').mask('Technology', '3G')
temp = data.groupby(by=state)[speed].mean()
# get correct positions of width according to previous sorting
for v in group.index:
    group.head(len(values))[v] = temp.head(len(values))[v]
    

plt.barh(y = r, width = group.head(len(values)), color = 'y', label = '3G')
for (i, v) in enumerate(group.head(len(values))):
    plt.text(x = v + 0.2, y = i - 0.1, s = str(round(v, 2)), color = 'yellow', fontweight = 'bold')


plt.margins(x = 0.15)
plt.legend(loc = 'lower right')
plt.show()


# Average Download Speeds for different states for both 4g and 3g technologies.
# * Average speeds of 3g networks lies between 1-3 Mbps. This is a lot less than what a 3g network should offer.
# * Average speeds of 4g networks lies between 7-26 Mbps.
# * The highest 4g network speed occurs for Himachal Pradesh which is explained by the speed distribtution of Himachal Pradesh for all providers. Since the number of samples are less, hence the average speeds hit the max value compared to other states.
# * The lowest 4g speed occurs for north east states.

# In[29]:


# avg speeds of states and service providers
#columns = ['Technology', 'Test_type', 'Service Provider', 'LSA']

#4g
print(df[service].unique())
data = df.mask('Test_type', 'Download').mask('Technology', '4G')
values = data[service].unique()
group = data.groupby(by=service)[speed].mean().sort_values(ascending = True)
r = np.arange(len(values))

plt.figure(figsize = (6, len(values)/2 +1))
plt.barh(y = r, width = group.head(len(values)), label = '4G')
for (i, v) in enumerate(group.head(len(values))):
    plt.text(x = v + 0.2, y = i - 0.1, s = str(round(v, 2)), color = 'blue', fontweight = 'bold')
plt.yticks(r, group.index)
plt.xlabel(speed)
plt.margins(x = 0.15)
plt.legend(loc = 'lower right')

#3g
data = df.mask('Test_type', 'Download').mask('Technology', '3G')
values = data[service].unique()
group = data.groupby(by=service)[speed].mean().sort_values(ascending = True)
r = np.arange(len(values))

plt.figure(figsize = (6, len(values)/2 +1))
plt.barh(y = r, width = group.head(len(values)), color = 'g', label = '3G')
for (i, v) in enumerate(group.head(len(values))):
    plt.text(x = v + 0.02, y = i - 0.1, s = str(round(v, 2)), color = 'green', fontweight = 'bold')
plt.yticks(r, group.index)
plt.xlabel(speed)
plt.margins(x = 0.15)
plt.legend(loc = 'lower right')
plt.show()


# Finally bar plots for average 4g and 3g network speeds for each provider.
# * In 4g networks, jio is the one with max average speed followed by airtel.
# * In 3g networks, Aircel is the one with max average speed, which is doubtful. But the speed distribution of aircel reveals very less numbers of customers, hence the higher speeds.
# * Other than Jio all the service providers are giving average 4g speeds of 6-8 Mbps which is much lower than speeds that they should be giving.
# * For jio also, most of the customers have lower speeds, giving an exponential decrease in number of customers as we increase network speeds.

# This is my first visaulisation.
# If found any bug in code or any mistake, Please let me know. Thanks.
