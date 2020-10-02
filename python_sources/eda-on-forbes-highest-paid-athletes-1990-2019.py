#!/usr/bin/env python
# coding: utf-8

# # The Highest Paid Athletes of the last 29 years according to Forbes
# 
# ![Lebron James](https://specials-images.forbesimg.com/imageserve/5dfd3fda0e40fc00080e70b2/1920x0.jpg?fit=scale)
# 
# This is a comprehensive list of highest paid athletes in the world over the last twenty-nine years. The list has been compiled by Forbes Magazing and shows the highest earning athletes for the period of 1990 - 2019. Given the time span, the information extractable from this list can show us which athletes have managed to sustain their career longetivity rather than burning fast and dying out early. It will also give us insight into which athletes are benefiting the most from their sporting skill as well as their ability to pull endorsements from big-name companies in order to boost their overall earnings. 
# 
# As you go through this EDA, the name of legendary basketball player, Kobe Bryant will feature more than once. Kobe was and still remains one of the best players to grace the game of basketball. May his legacy live on as a result of more than just his earnings but also his sheer greatness, discipline, dedication to his sport and athleticism. 
# 
# **Rest in Power, Kobe and Gina.**
# 
# ![Kobe_Gina](https://img.buzzfeed.com/buzzfeed-static/static/2020-01/26/23/asset/dfdc5f88d1c3/sub-buzz-2438-1580080422-1.jpg?downsize=800:*&output-format=auto&output-quality=auto)

# In[ ]:


import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go

import matplotlib.pyplot as plt
import matplotlib.ticker as tk

import matplotlib.animation as animation
from IPython.display import HTML


# In[ ]:


path = '../input/forbes-highest-paid-athletes-19902019/'
df = pd.read_csv(path+'Forbes Richest Atheletes (Forbes Richest Athletes 1990-2019).csv')


# As is customary for an exploratory data analysis, we get a preview of the table and get basic statistics on it. 

# In[ ]:


df.head(11)


# In[ ]:


df.tail(11)


# In[ ]:


year_groups = df.groupby('Year')
'Of course, the records span over %d years.' %len(year_groups)


# To start with, one may be interested in knowing which athletes have enjoyed longetivity in their careers. In other words, who has appeared on this list more than once and how many times have they dominated? <br>
# 
# Below, the athletes are listed in order of appearance on the top 20 list during its 30 year history.

# ## Number of appearances for an athlete during the 29 year period

# In[ ]:


list_len = 15
athletes = df.groupby(['Name','Nationality'])
df_appearances = athletes['Name'].count()                .reset_index(name='Count')                .sort_values(['Count'], ascending=0)                .head(list_len)
df_appearances_9_or_more = df_appearances[df_appearances['Count'] >= 9]
df_appearances_9_or_more


# In[ ]:


fig = go.Figure( go.Bar(
              x = df_appearances_9_or_more.Count[::-1],
              y = df_appearances_9_or_more.Name[::-1],
              orientation='h', 
              opacity=0.5, 
              marker=dict(color='rgba(207, 0, 15, 1)')
                        )
               )

fig.update_traces(marker_line_color='rgb(10,48,107)')
fig.update_layout(title_text='Number of Appearances on the Forbes List between 1990-2019', 
                  xaxis_title='Number of Appearances', 
                  yaxis_title='Athletes') 


# Apparently, the USA has produced 4/5 of the top 5 highest paid athletes of the last three decades. 
# Does this equate to their dominance in producing the majority of highly paid athletes overall?

# ## The countries with the high-earning athletes

# In[ ]:


# This is a standard operation which can be generalised
# The keys are set to upper to reduce case-sensitive duplicates 
def top_list(key, list_len):
    countries = df.groupby(df[key].str.upper())
    return countries[key].count()                .reset_index(name='Count')                .sort_values(['Count'], ascending=0)                .head(list_len)

df_countries = top_list( 'Nationality', 10 )
df_countries


# To emphasise this point even more, these stats can be visualised on a simple pie chart.

# In[ ]:


px.pie(df_countries, 'Nationality', 'Count', 
       color_discrete_sequence=px.colors.sequential.Viridis, 
       title="Tallied Representation of a Nation's Athletes on the Forbes List between 1990-2019")


# ## Sports that produce the highest earning athletes

# In[ ]:


df_sports = top_list( 'Sport', 10 )
df_sports


# In[ ]:


px.pie(df_sports, 'Sport', 'Count', 
       color_discrete_sequence=px.colors.sequential.Cividis, 
       title="Tallied Representation of Sports on the Forbes List between 1990-2019")


# Basketball and boxing have the highest paid athletes in the world. This seems to correlate rather nicely with the fact that 3/5 of the top 5 earners have all made their money from basketball: Michael Jordan, Kobe Bryant and Lebron James. <br> 
# 
# But how much did they actually earn?

# ## Highest earners on the list in the 29 year period

# In[ ]:


list_len = 10
athletes = df.groupby(['Name'])
df_earnings = athletes['earnings ($ million)'].sum().reset_index(name='Sum')                .sort_values(['Sum'], ascending=0)                .head(list_len)
df_earnings


# In[ ]:


fig = go.Figure( go.Bar(
              x = df_earnings.Sum[::-1],
              y = df_earnings.Name[::-1],
              orientation='h', 
              opacity=0.5, 
              marker=dict(color='rgba(50, 171, 96, 0.6)')
                        )
               )

fig.update_layout(title_text='Career Earnings for Athletes on the Forbes List between 1990-2019', 
                 xaxis_title='Total Career Earnings (US$ million)', 
                 yaxis_title='Athletes')     


# Floyd Mayweather is the second highest earner of all time despite not having a number of appearances in the annual list similar Tiger Woods, Lebron James or Michael Jordan yet his career has spanned over much less time than the afore mentioned veterans. How could this be?

# In[ ]:


df_mayweather = athletes.get_group('Floyd Mayweather')
'Despite earning this much over his career, Floyd Mayweather has only appeared on the list %d times.'%len(df_mayweather)
df_mayweather


# It appears that Mayweather made gigantic leaps in his annual earnings in less than a decade.

# ## How quickly Floyd Mayweather's earnings rose

# In[ ]:


fig = go.Figure( go.Scatter(x=df_mayweather.Year, 
                            y=df_mayweather['earnings ($ million)'], 
                            line=dict(color='firebrick', width=3)
                           )
               )

#very customised layouts
fig.add_layout_image(
        dict(
            source="https://1284474717.rsc.cdn77.org/wp-content/uploads/2018/10/014_Floyd_Mayweather_vs_Conor_McGregor.jpg",
            xref="x",
            yref="y",
            x=2008.7, 
            y=315,
            sizex=300,
            sizey=300,
            opacity=0.33,
            layer="below")
)

fig.update_layout(
    title_text="Floyd Mayweather's Earnings throughout the years", 
    xaxis_title='Year', 
    yaxis_title='Earnings (US$ million)', 
    autosize=False,
    height=600, 
    width=800
)


# After a steady rise in his earnings between 2010 and 2014, he almost tripled is annual earnings in 2015 after winning the "Fight of the Century" against Manny Pacquiao in a unaminous 12-round victory on the 2nd May 2015. 

# ## Earnings by year
# 
# We can take a look at the top earners for each year and their earnings with the passage of time. Racing bar plots have been rather popular on social media lately. @pratapvardhan put together a matplotlib version of this which I borrowed for this task. We get an animated view of the changing Top 10 Forbes top-earning athlete's list every year, how much they earned during that year and their country of origin. 

# In[ ]:


df_race_bar = df[['Name', 'Year', 'Nationality', 'earnings ($ million)' ]]
dff = df_race_bar.groupby(['Year'])


# In[ ]:


def draw_barchart(year):
    dff_year = dff.get_group(year)                  .sort_values(by='earnings ($ million)', ascending=1)
    colors = dict(zip(
        ['USA', 'Germany', 'Switzerland', 'Portugal',
         'Portugal', 'Argentina', 'Brazil', 'Australia', 
        'Italy', 'Finland', 'Canada', 'UK', 'France', 
        'Ireland', 'Austria', 'Russia', 'Dominican', 
        'Philippines', 'Spain', 'Serbia', 'Mexico', 
        'Northern Ireland'],
        ['#b6c4ef', '#ffb3ff', '#90d595', '#e48381',
         '#aafbff', '#f7bb5f', '#eafb50', '#e58285', 
        '#16a085', '#f54565', '#f39c12', '#bfc9ca', 
          '#e1dbdb', '#a9dfbf', '#f9e79f', '#d6eaf8', 
         '#d7bde2', '#f7dc6f', '#2980b9', '#9a45b5', 
        '#4377cf', '#0e9921']
    ))
    
    group_lk = df.set_index('Name')['Nationality'].to_dict()
    
    ax.clear()
    ax.barh(dff_year['Name'], dff_year['earnings ($ million)'],
            color=[colors[group_lk[x]] for x in dff_year['Name']])
    
    for i, (value, name) in enumerate(zip(dff_year['earnings ($ million)'], dff_year['Name'])):
        ax.text(value-0.5, i,     name,            ha='right', size=14, fontweight='bold')  # Athlete's name
        ax.text(value-0.5, i-.3, group_lk[name],  ha='right')  # Nationality
        ax.text(value+0.2, i,     value,      ha='left', size=14)   # Earnings    
    
    ax.text(1, 0.4, year, transform=ax.transAxes, color='#777777', size=46, ha='right', weight=800)
    ax.text(0, 1.06, 'Earnings ($ millions)', transform=ax.transAxes, size=16, color='#777777')
    ax.xaxis.set_major_formatter(tk.StrMethodFormatter('{x:,.0f}'))
    ax.xaxis.set_ticks_position('top')
    ax.tick_params(axis='x', colors='#777777', labelsize=12)
    ax.set_yticks([])
    ax.margins(0, 0.01)
    ax.grid(which='major', axis='x', linestyle='-')
    ax.set_axisbelow(True)
    ax.text(0, 1.12, 'The highest paid Athletes from 1990 to 2019',
            transform=ax.transAxes, size=20, weight=600, ha='left')
    ax.text(1, 0, 'by @pratapvardhan; credit @jburnmurdoch', transform=ax.transAxes, ha='right',
            color='#777777', bbox=dict(facecolor='white', alpha=0.8, edgecolor='white'))
    plt.box(False)


# In[ ]:


fig, ax = plt.subplots(figsize=(15,7))
frames = [ i for i in range(1990,2020) if i != 2001 ]
def init():
    ax.clear()
    ax.set_yticks([])
    ax.set_xticks([])
animator = animation.FuncAnimation(fig, draw_barchart, frames=frames, blit=False, init_func=init)
HTML(animator.to_jshtml(fps=1.5))


# In[ ]:




