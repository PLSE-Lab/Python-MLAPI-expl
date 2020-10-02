#!/usr/bin/env python
# coding: utf-8

# # Brazil Temperature Change (1961-2019)

# <img src="https://s7.gifyu.com/images/brazil_temperature_change_1961_2019-1.gif" data-canonical-src="https://s7.gifyu.com/images/brazil_temperature_change_1961_2019-1.gif" width="500" height="500" />

#    # Load dataset

# In[ ]:


import pandas as pd
df = pd.read_csv("/kaggle/input/conventional-weather-stations-brazil/conventional_weather_stations_inmet_brazil_1961_2019.csv", sep=";")


# ### Show all fields

# In[ ]:


df.dtypes


# ### Show first rows

# In[ ]:


df.head()


# ### Configure date index

# In[ ]:


df.index = pd.to_datetime(df['Data'])


# # Calculate Average Monthly Temperature (1961-2019)

# In[ ]:


average_monthly_temperature = df['Temp Comp Media'].groupby(df.index.month).mean()
average_monthly_temperature


# # Filter values by year 

# In[ ]:


data_2019 = df[df.index.year == 2019]
data_2019.head()


# # Calculate deviation from the average temperature

# In[ ]:


data_2019_monthly_average = data_2019
data_2019_monthly_average = data_2019_monthly_average    .groupby(data_2019_monthly_average.index.month)['Temp Comp Media']    .mean() - average_monthly_temperature
data_2019_monthly_average


# # Create polar coordinate system

# In[ ]:


import matplotlib.pyplot as plt
fig = plt.figure(figsize=(8,8))
ax1 = plt.subplot(111, projection='polar')


# # Preparing data for polar plotting

# In[ ]:


import numpy as np
fig = plt.figure(figsize=(8,8))
ax1 = plt.subplot(111, projection='polar')
r = data_2019_monthly_average
theta = np.linspace(0, 2*np.pi, 12)

ax1.plot(theta, r)


# In[ ]:


import numpy as np


fig = plt.figure(figsize=(9,9))
ax1 = plt.subplot(111, projection='polar')


theta = np.linspace(0, 2*np.pi, 12)

ax1.grid(False)
ax1.set_title("Brazil Temperature Change (1961-2019)", color='white', fontdict={'fontsize': 30})
lines, labels = plt.thetagrids((90, 60, 30, 0, 330, 300, 260, 230, 210, 180, 150, 120), labels=('Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'), fmt=None)
ax1.set_ylim(-3, 3)

ax1.tick_params(axis='x', colors='white', labelsize=18)

fig.set_facecolor("#323331")
ax1.set_facecolor('#000100')

year = 1961

year_data = df[df.index.year == year]

r  = year_data.groupby(year_data.index.month)['Temp Comp Media'].mean() - average_monthly_temperature

year_label = ax1.text(0,0,year, color='white', size=30, ha='center')

ax1.plot(theta, r)
plt.tight_layout()


# # Adding temperature rings

# In[ ]:


full_circle_thetas = np.linspace(0, 2*np.pi, 1000)
blue_line_one_radii = [-1.0]*1000
blue_line_two_radii = [0.0]*1000
red_line_one_radii = [1.5]*1000

ax1.plot(full_circle_thetas, blue_line_one_radii, c='blue')
ax1.plot(full_circle_thetas, blue_line_two_radii, c='blue')
ax1.plot(full_circle_thetas, red_line_one_radii, c='red')


# In[ ]:


ax1.text(np.pi/2, -1.0, "-1.0 C", color="blue", ha='center', fontdict={'fontsize': 20})
ax1.text(np.pi/2, 0.0, "0.0 C", color="blue", ha='center', fontdict={'fontsize': 20})
ax1.text(np.pi/2, 1.5, "1.5 C", color="red", ha='center', fontdict={'fontsize': 20})


# # Generating The GIF Animation

# In[ ]:


import sys
from matplotlib.animation import FuncAnimation

years = list(range(1961, 2020))

def update(i):
    year = years[i]
    
    year_data = df[df.index.year == year]

    r = year_data.groupby(year_data.index.month)['Temp Comp Media'].mean() - average_monthly_temperature

    year_label.set_text(year)
    ax1.set_facecolor('#000100')
    ax1.plot(theta, r)
    plt.tight_layout()

anim = FuncAnimation(fig, update, frames=len(years), interval=200)

gif_output = '/kaggle/working/brazil_temperature_change_1961_2019.gif'

anim.save(gif_output, dpi=120, writer='imagemagick', savefig_kwargs={'facecolor': '#323331'})

from IPython.display import Image
Image(gif_output)

