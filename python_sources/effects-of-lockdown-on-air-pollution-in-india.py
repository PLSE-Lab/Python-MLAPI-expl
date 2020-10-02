#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
get_ipython().run_line_magic('matplotlib', 'inline')

data = pd.read_csv('../input/air-quality-data-in-india/city_day.csv')
data['Date'] = pd.to_datetime(data['Date'])
cities = ['Kolkata','Chennai', 'Delhi', 'Mumbai', 'Bengaluru']
data = data[data['City'].isin(cities)]
data = data[(data['Date'].dt.year>2018)]
data['month'] = data['Date'].dt.month
data['year'] = data['Date'].dt.year
data['day'] = data['Date'].dt.day


# In[ ]:


fig = plt.figure(figsize=(23,30))
for city,num in zip(cities, range(1,6)):
    df = data[data['City']==city]
    ax = fig.add_subplot(5,2,num)
    ax.plot(df['Date'], df['AQI'])
    ax.axvline(pd.to_datetime('2020-03-24'), c='k')
    ax.set_title('Year wise mean AQI for {}'.format(city))
    ax.set_ylabel('Mean AQI')


# In[ ]:


fig,ax = plt.subplots(figsize=(15, 7))

for city in cities: 
    sns.lineplot(x="Date", y="AQI", data=data[data['City']==city].iloc[::30],label = city)

ax.set_xticklabels(ax.get_xticklabels(cities), rotation=30, ha="left")
ax.axvline(pd.to_datetime('2020-03-24'), c='k')
ax.set_title('AQI values in cities')
ax.legend()


# In[ ]:


perc = data.loc[data['Date'].dt.year==2019,["month","City",'AQI']]
perc['mean_AQI'] = perc.groupby([perc.City,perc.month])['AQI'].transform('mean')
perc.drop('AQI', axis=1, inplace=True)
perc = perc.drop_duplicates()
perc = perc.sort_values("month")

fig=px.bar(perc,x='City', y="mean_AQI", animation_frame="month", 
           animation_group="City", color="City", hover_name="City", range_y=[0,600])
fig.update_layout(showlegend=False)
fig.show()


# # Comparing 2019 data to 2020 data

# In[ ]:


df_2019 = data[(data['Date'].dt.year==2019) & (data['Date'].dt.month>=3) & (data['Date'].dt.month<5)]
df_2020 = data[(data['Date'].dt.year==2020) & (data['Date'].dt.month>=3) & (data['Date'].dt.month<5)]


# In[ ]:


#source: http://nicolasfauchereau.github.io/climatecode/posts/drawing-a-gauge-with-matplotlib/

from matplotlib.patches import Circle, Wedge, Rectangle

def degree_range(n):
    start = np.linspace(0,180,n+1, endpoint=True)[0:-1]
    end = np.linspace(0,180,n+1, endpoint=True)[1::]
    mid_points = start + ((end-start)/2.)
    return np.c_[start, end], mid_points

def rot_text(ang): 
    rotation = np.degrees(np.radians(ang) * np.pi / np.pi - np.radians(90))
    return rotation


# In[ ]:


#source: http://nicolasfauchereau.github.io/climatecode/posts/drawing-a-gauge-with-matplotlib/
from matplotlib.patches import Circle, Wedge, Rectangle
def gauge(labels=['GOOD','SATISFACTORY','MODERATE','POOR','VERY POOR','EXTREME'],           colors='jet_r', arrow=1, title='', fname=False): 
    
    """
    some sanity checks first
    

"""
    
    N = len(labels)
    
    if arrow > N: 
        raise Exception("\n\nThe category ({}) is greated than         the length\nof the labels ({})".format(arrow, N))
 
    
    """
    if colors is a string, we assume it's a matplotlib colormap
    and we discretize in N discrete colors 
    """
    
    if isinstance(colors, str):
        cmap = cm.get_cmap(colors, N)
        cmap = cmap(np.arange(N))
        colors = cmap[::-1,:].tolist()
    if isinstance(colors, list): 
        if len(colors) == N:
            colors = colors[::-1]
        else: 
            raise Exception("\n\nnumber of colors {} not equal             to number of categories{}\n".format(len(colors), N))

    """
    begins the plotting
    """
    
    fig, ax = plt.subplots()

    ang_range, mid_points = degree_range(N)

    labels = labels[::-1]
    
    """
    plots the sectors and the arcs
    """
    patches = []
    for ang, c in zip(ang_range, colors): 
        # sectors
        patches.append(Wedge((0.,0.), .4, *ang, facecolor='w', lw=2))
        # arcs
        patches.append(Wedge((0.,0.), .4, *ang, width=0.10, facecolor=c, lw=2, alpha=0.5))
    
    [ax.add_patch(p) for p in patches]

    
    """
    set the labels (e.g. 'LOW','MEDIUM',...)
    """

    for mid, lab in zip(mid_points, labels): 

        ax.text(0.35 * np.cos(np.radians(mid)), 0.35 * np.sin(np.radians(mid)), lab,             horizontalalignment='center', verticalalignment='center', fontsize=14,             fontweight='bold', rotation = rot_text(mid))

    """
    set the bottom banner and the title
    """
    r = Rectangle((-0.4,-0.1),0.8,0.1, facecolor='w', lw=2)
    ax.add_patch(r)
    
    ax.text(0, -0.05, title, horizontalalignment='center',          verticalalignment='center', fontsize=22, fontweight='bold')

    """
    plots the arrow now
    """
    
    pos = mid_points[abs(arrow - N)]
    
    ax.arrow(0, 0, 0.225 * np.cos(np.radians(pos)), 0.225 * np.sin(np.radians(pos)),                  width=0.04, head_width=0.09, head_length=0.1, fc='k', ec='k')
    
    ax.add_patch(Circle((0, 0), radius=0.02, facecolor='k'))
    ax.add_patch(Circle((0, 0), radius=0.01, facecolor='w', zorder=11))

    """
    removes frame and ticks, and makes axis equal and tight
    """
    
    ax.set_frame_on(False)
    ax.axes.set_xticks([])
    ax.axes.set_yticks([])
    ax.axis('equal')
    plt.tight_layout()
    if fname:
        fig.savefig(fname, dpi=200)


# In[ ]:


#display("Delhi's AQI levels")

gauge(labels=['Good','Satisfactory','Moderate','Poor','Very Poor','Extreme'],       colors=['#007A00','#90EE90','#ffff00','#FF9900','#ff0000','#CC0000'], arrow=5, title='Delhi AQI in 2019') 

gauge(labels=['Good','Satisfactory','Moderate','Poor','Very Poor','Extreme'],       colors=['#007A00','#90EE90','#ffff00','#FF9900','#ff0000','#CC0000'], arrow=3, title='Delhi AQI in 2020') 


# In[ ]:


gauge(labels=['Good','Satisfactory','Moderate','Poor','Very Poor','Extreme'],       colors=['#007A00','#90EE90','#ffff00','#FF9900','#ff0000','#CC0000'], arrow=4, title='Kolkata AQI in 2019') 

gauge(labels=['Good','Satisfactory','Moderate','Poor','Very Poor','Extreme'],       colors=['#007A00','#90EE90','#ffff00','#FF9900','#ff0000','#CC0000'], arrow=3, title='Kolkata AQI in 2020') 


# In[ ]:


gauge(labels=['Good','Satisfactory','Moderate','Poor','Very Poor','Extreme'],       colors=['#007A00','#90EE90','#ffff00','#FF9900','#ff0000','#CC0000'], arrow=4, title='Bengaluru AQI in 2019') 

gauge(labels=['Good','Satisfactory','Moderate','Poor','Very Poor','Extreme'],       colors=['#007A00','#90EE90','#ffff00','#FF9900','#ff0000','#CC0000'], arrow=2, title='Bengaluru AQI in 2020') 


# In[ ]:


fig = plt.figure(figsize=(15,30))
for city,num in zip(cities, range(1,6)):
    df1 = df_2019[df_2019['City']==city].reset_index(drop=True).reset_index()
    df2 = df_2020[df_2020['City']==city].reset_index(drop=True).reset_index()
    ax = fig.add_subplot(5,2,num)
    ax.plot(df1['index'], df1['PM2.5'], label='2019', c='k')
    ax.plot(df2['index'], df2['PM2.5'], label='2020', c='r')
    ax.set_xlabel('Days')
    ax.set_ylabel('PM2.5')
    ax.set_title('PM2.5 values in {}'.format(city))
    ax.legend()


# In[ ]:


fig = plt.figure(figsize=(15,30))
for city,num in zip(cities, range(1,6)):
    df1 = df_2019[df_2019['City']==city].reset_index(drop=True).reset_index()
    df2 = df_2020[df_2020['City']==city].reset_index(drop=True).reset_index()
    ax = fig.add_subplot(5,2,num)
    ax.plot(df1['index'], df1['NOx'], label='2019', c='k')
    ax.plot(df2['index'], df2['NOx'], label='2020', c='r')
    ax.set_xlabel('Days')
    ax.set_ylabel('NOx')
    ax.set_title('NOx values in {}'.format(city))
    ax.legend()


# In[ ]:


fig = plt.figure(figsize=(15,30))
for city,num in zip(cities, range(1,6)):
    df1 = df_2019[df_2019['City']==city].reset_index(drop=True).reset_index()
    df2 = df_2020[df_2020['City']==city].reset_index(drop=True).reset_index()
    ax = fig.add_subplot(5,2,num)
    ax.plot(df1['index'], df1['SO2'], label='2019', c='k')
    ax.plot(df2['index'], df2['SO2'], label='2020', c='r')
    ax.set_xlabel('Days')
    ax.set_ylabel('SO2')
    ax.set_title('SO2 values in {}'.format(city))
    ax.legend()


# In[ ]:


fig = plt.figure(figsize=(15,30))
for city,num in zip(cities, range(1,6)):
    df1 = df_2019[df_2019['City']==city].reset_index(drop=True).reset_index()
    df2 = df_2020[df_2020['City']==city].reset_index(drop=True).reset_index()
    ax = fig.add_subplot(5,2,num)
    ax.plot(df1['index'], df1['Benzene'], label='2019', c='k')
    ax.plot(df2['index'], df2['Benzene'], label='2020', c='r')
    ax.set_xlabel('Days')
    ax.set_ylabel('Benzene')
    ax.set_title('Benzene values in {}'.format(city))
    ax.legend()


# # AQI distribution in cities

# In[ ]:


df_Mumbai = data[data['City']== 'Mumbai']
df_Bengaluru = data[data['City']== 'Bengaluru']
df_Delhi = data[data['City']== 'Delhi']
df_Chennai = data[data['City']== 'Chennai']
df_Kolkata = data[data['City']== 'Kolkata']


# In[ ]:


fig,ax=plt.subplots(figsize=(10, 7))
sns.despine(fig, left=True, bottom=True)
sns.set_context("notebook", font_scale=2, rc={"lines.linewidth": 2})
sns.distplot(df_Delhi['AQI'].iloc[::30], color="y",label = 'Delhi')
sns.distplot(df_Mumbai['AQI'].iloc[::30], color="b",label = 'Mumbai')
sns.distplot(df_Chennai['AQI'].iloc[::30], color="black",label = 'Chennai')
sns.distplot(df_Bengaluru['AQI'].iloc[::30], color="g",label = 'Bengaluru')
sns.distplot(df_Kolkata['AQI'].iloc[::30], color="r",label = 'Kolkata')
labels = [item.get_text() for item in ax.get_xticklabels()]
ax.set_xticklabels(ax.get_xticklabels(labels), rotation=30,ha="left")
plt.rcParams["xtick.labelsize"] = 15
ax.set_title('AQI DISTRIBUTIONS FROM SELECTED CITIES')
ax.legend(fontsize = 14)
plt.show()


# # Nov-Dec 2019 vs Mar-Apr 2020

# In[ ]:


df_nov_dec = data[(data['year']==2019) & (data['month']>10) & (data['month']<=12)]
df_mar_apr = data[(data['year']==2020) & (data['month']>2) & (data['month']<=4)]


# In[ ]:


df_new = pd.concat([df_nov_dec,df_mar_apr])


# In[ ]:


dd = df_new.loc[:,['PM2.5','City','year']]
dd['meanPM'] = dd.groupby(['City','year'])['PM2.5'].transform('mean')
dd.drop('PM2.5',axis=1, inplace=True)
dd = dd.drop_duplicates()

plt.figure(figsize=(10,7))
sns.barplot(x="City", y="meanPM", hue="year", data=dd)
plt.title('PM2.5 levels in Cities')
plt.ylabel('PM2.5')
plt.xlabel('')
plt.show()


# In[ ]:


dd = df_new.loc[:,['NH3','City','year']]
dd['meanNH3'] = dd.groupby(['City','year'])['NH3'].transform('mean')
dd.drop('NH3',axis=1, inplace=True)
dd = dd.drop_duplicates()

plt.figure(figsize=(10,7))
sns.barplot(x="City", y="meanNH3", hue="year", data=dd)
plt.title('NH3 levels in Cities')
plt.ylabel('NH3')
plt.xlabel('')
plt.show()


# In[ ]:


dd = df_new.loc[:,['CO','City','year']]
dd['meanCO'] = dd.groupby(['City','year'])['CO'].transform('mean')
dd.drop('CO',axis=1, inplace=True)
dd = dd.drop_duplicates()

plt.figure(figsize=(10,7))
sns.barplot(x="City", y="meanCO", hue="year", data=dd)
plt.title('CO levels in Cities')
plt.ylabel('CO')
plt.xlabel('')
plt.show()


# In[ ]:


dd = df_new.loc[:,['O3','City','year']]
dd['meanO3'] = dd.groupby(['City','year'])['O3'].transform('mean')
dd.drop('O3',axis=1, inplace=True)
dd = dd.drop_duplicates()

plt.figure(figsize=(10,7))
sns.barplot(x="City", y="meanO3", hue="year", data=dd)
plt.title('O3 levels in Cities')
plt.ylabel('O3')
plt.xlabel('')
plt.show()


# In[ ]:




