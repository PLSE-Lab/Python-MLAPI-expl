#!/usr/bin/env python
# coding: utf-8

# # Analyzing the Worst Drivers in the USA
# 
# The dataset for this kernel was first collected via <a href="https://fivethirtyeight.com/features/which-state-has-the-worst-drivers/">Five Thirty Eight</a>.

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
plt.style.use("seaborn")
import dataframeoverview as dfo # user-defined utility script


# In[ ]:


df = pd.read_csv('../input/fivethirtyeight-bad-drivers-dataset/bad-drivers.csv')


# In[ ]:


dfo.feature_info(df)


# Therefore, our data has :
# 1. No Missing Values
# 2. 1 Categorical feature, 7 continuous features

# In[ ]:


# rename the features for ease

new_features = ["state", "num_driver_fatal", "percent_speeding", "percent_alcohol", "percent_not_dis", "percent_no_prev", "insur_prem", "loss_per_insur_driver"]
df=df.rename(columns=dict(zip(df.columns,new_features)))


# In[ ]:


# Function to find categorical and non-categorical features
def cat_finder(df):
    cat = []
    cont = []
    cols = list(df.columns)
    for col in cols:
        t = df[col].dtype #returns the type of the feature
        if (t=='O'):
            cat.append(col)
        else :
            cont.append(col)
      
    return (cat, cont)

# Using the function above


cat, cont = cat_finder(df)
print("\nCategorical Features:", cat)
print("\nContinuous Features:\n", cont)


# In[ ]:


# All distributions of the continuous features

def plot_dist(df, cont_features):
    for f in cont_features:
        plt.figure(figsize=(6,3))
        sns.kdeplot(df[f], legend=False, color="red", shade=True)
        plt.title("Distribution of "+f)
        plt.show()
        
plot_dist(df, cont)
        


# ### Insights
# 1. The number of drivers involved in fatal collisions per billion miles is **normally distributed** with a mean of 15.79 collisions per billion miles and **standard deviation** of 4.12.
# 2. The proportion of drivers involved in fatal collisions who were not distracted is **left-skewed** and depicts the large proportion of non-distracted drivers involved in fatal colissions.

# ## The Feature Correlations

# In[ ]:


# Correlation between the features

corr_list = cont
corr = df[corr_list].corr()

# Visualize using heatmap
plt.figure(figsize=(7, 4)) 
plt.title("The Correlations of the Features")
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"):
    ax = sns.heatmap(corr, annot=True, fmt=".2f", linewidths=.5, mask=mask, cmap="inferno")


# ### Insight
# 1. There exists a strong positive correlation between the insurance amount paid by drivers and the losses incurred to compabies due to collisions.

# In[ ]:


# loss_per_insur_driver vs insur_prem

sns.lmplot(x='loss_per_insur_driver',y='insur_prem',fit_reg=False,scatter_kws={"color":"blue","alpha":0.5,"s":100},data=df)
plt.xlabel('\nLosses incurred by insurance companies per insured driver ($)',size=12)
plt.ylabel('Car insurance premiums ($)',size=12)
plt.title("Relation between Insurance Premiums and Losses to Insurance Companies\n", fontsize=17)
plt.show()


# ### Insight 
# 1. Higher the Car insurance premium amounts, greater the losses for the insurance companies when an insured driver is involved in an accident.

# ## Speed Kills! But is it the greatest contributor of fatalities?

# In[ ]:


# Speed kills

plt.figure(figsize=(10,9))
sns.scatterplot(x="percent_speeding", y="state", data=df, color="red")
plt.xlabel("\nPercentage of drivers involved in fatal collisions who were speeding", fontsize=15)
plt.ylabel("State", fontsize=15)
plt.title("Proportions of collisions due to speeding across states\n", fontsize=17)
plt.show()


# ### Insight
# 1. Pennysylvania and Hawaii were the only 2 states that had "Speeding" cause 50% or more of the fatal accidents they had. 
# 2. Nebraska had the least proportion of drivers involved in fatal collisions while speeding. 
# 3. In most of the states, speeding contributes to less than 40% of the total fatal accidents caused. Therefore, "Speed kills. But it is definitely not the scariest factor for drivers on the road"!

# ## How safe are you even if you are not distracted?

# In[ ]:


# Non-distracted drivers' analysis
sns.violinplot(df['percent_not_dis'])
plt.xlabel("\nPercentage of drivers involved in fatal collisions who were not distracted", fontsize=15)
plt.title("How safe are you even when you follow the rules?\n", fontsize=17)
plt.show()


# ### Insight
# 1. The violin plot depicts a large concentration of data after 80%. This shows that even if you are not distracted during driving, you still are not immune to being a part of fatal accidents. 
# 2. It's evident that mistakes of one person can lead to the loss of several other lives too.

# ## The Top 10s of Everything (Not a good list to top!)

# In[ ]:


# Top 10s of all continuous features

def top_10(df, features):
    for f in features:
        l = df[["state",f]].sort_values(by=f,ascending=False).head(10)
        plt.figure(figsize=(6,3))
        sns.barplot(y="state", x=f, data=l)
        plt.title("Top 10 - "+f)
        plt.show()

top_10(df, cont)


# ## Insights
# 1. South Carolina, North Dakota and West Virginia had the most number of fatal collisions per billion miles of travel.
# 2. The state with the largest proportion of fatal collisions due to speeding was Hawaii, closely followed by Pennysylvania.
# 3. Montana has the highest proportion of inebriated drivers causing fatal colissions.
# 4. Texas is present within the top 10 states in all the 3 main features related to collisions (number per billion miles, proportion that was speeding and proportion that was under the influence of alcohol)
# 5. The drivers of New Jersey paid the highest amounts of car insurance premiums ($), followed by Louisiana and the District of Columbia. 

# ## Interactive Visualizations using Bokeh

# Interactive Visualizations help tell the story in a much faster, efficient way. So, I have tried a couple of them here using the Bokeh library.

# In[ ]:


from bokeh.io import show, output_notebook
from bokeh.plotting import figure
from bokeh.layouts import widgetbox
from bokeh.models.widgets import Button
from bokeh.models import Panel, Tabs
from bokeh.models import ColumnDataSource
from bokeh.models.tools import HoverTool
from bokeh.palettes import Category10
from bokeh.transform import factor_cmap


# In[ ]:


source = ColumnDataSource(df)
p = figure()
p.circle(x='insur_prem', y='loss_per_insur_driver', source=source, size='num_driver_fatal', color='red', line_color='black', alpha=0.5)
p.title.text = 'Relation between Insurance Premiums and Losses to Insurance Companies'
p.xaxis.axis_label = 'Insurance Premiums ($)'
p.yaxis.axis_label = 'Losses to Insurance Companies ($)'
hover = HoverTool()
hover.tooltips=[
    ('State', '@state')
]
p.add_tools(hover)
show(p)
output_notebook()


# ### Using the Plot
# 1. X-axis : Insurance Premium Amount in USD
# 2. Y-axis : Losses to Insurance Companies in USD
# 3. Each circle depicts a state and the radius of each circle is proportional to the Number of fata collisions in the state
# 4. Hover over each of the points to know the corresponding State

# In[ ]:


# Top 10 of the 2 main features related to fatal collisions - 'num_driver_fatal', 'percent_speeding', 'percent_alcohol'

l1 = df[["state",'num_driver_fatal']].sort_values(by='num_driver_fatal',ascending=True).tail(10)
source = ColumnDataSource(l1)
states = source.data['state'].tolist()
p1 = figure(y_range=states)
p1.hbar(y='state', right='num_driver_fatal', source=source, height=0.70, color='red')
p1.title.text = "Number of drivers involved in fatal collisions per billion miles"
p1.yaxis.axis_label = 'State'
p1.xaxis.axis_label = 'Number of drivers involved in fatal collisions per billion miles'
hover = HoverTool()
hover.tooltips=[
    ('Value', '@num_driver_fatal')
]
p1.add_tools(hover)
tab1 = Panel(child=p1, title="Total number of drivers")

l2 = df[["state",'percent_speeding']].sort_values(by='percent_speeding',ascending=True).tail(10)
source = ColumnDataSource(l2)
states = source.data['state'].tolist()
p2 = figure(y_range=states)
p2.hbar(y='state', right='percent_speeding', source=source, height=0.70, color='red')
p2.title.text = "Percentage of drivers who were speeding during fatal collision"
p2.yaxis.axis_label = 'State'
p2.xaxis.axis_label = 'Proportion of drivers speeding during fatal collision'
hover = HoverTool()
hover.tooltips=[
    ('Value', '@percent_speeding'+'%')
]
p2.add_tools(hover)
tab2 = Panel(child=p2, title="Speeding")

l3 = df[["state",'percent_alcohol']].sort_values(by='percent_alcohol',ascending=True).tail(10)
source = ColumnDataSource(l3)
states = source.data['state'].tolist()
p3 = figure(y_range=states)
p3.hbar(y='state', right='percent_alcohol', source=source, height=0.70, color='red')
p3.title.text = "Percentage of drivers who were alcohol-impaired during fatal collision"
p3.yaxis.axis_label = 'State'
p3.xaxis.axis_label = 'Proportion of drivers under influence of alcohol during fatal collision'
hover = HoverTool()
hover.tooltips=[
    ('Value', '@percent_alcohol'+'%')
]
p3.add_tools(hover)
tab3 = Panel(child=p3, title="Alcohol-Impaired")

tabs = Tabs(tabs=[ tab1, tab2, tab3 ])
show(tabs)
output_notebook()


# ### Using the Plot
# 1. Navigate between each tab to view the corresponding graph
# 2. Hover over each bar for the value

# ## That's all for now. Hope you had a good read!
