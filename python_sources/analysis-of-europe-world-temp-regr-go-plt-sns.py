#!/usr/bin/env python
# coding: utf-8

# # [blog.osmanballi.com](https://blog.osmanballi.com/)

# ![](https://khosann.com/wp-content/uploads/2016/01/16-2.jpg)

# * The World Meteorological Organization announced that the 10 years ending as of the end of 2019 are the hottest 10 years in the world history.
# 
# * In addition, 2019 was the second hottest year measured from 1850 on calculations using data from the US Department of Aeronautics and Space (NASA), the US Department of Ocean and Atmosphere (NOAA) and the British Meteorological Service.
# 
# * When looking at the five-year series of the last 170 years, it was stated that the five years we left behind were the hottest series, while 2019 was 1.1 degrees warmer than the pre-industrial period.
#  
# * The World Meteorological Organization stated that by 2020, extreme weather events such as fires in Australia will be more common.
#  
# * Petteri Taalas, head of the World Meteorological Organization, said in a statement that Australia is experiencing the hottest and dryest writing it has seen, leading to major fires.
# 

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import plotly.graph_objs as go
import plotly.offline as pyoff
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


df = pd.read_csv("/kaggle/input/daily-temperature-of-major-cities/city_temperature.csv")
df["index"]=np.arange(len(df))
df.head()


# In[ ]:


df.shape


# In[ ]:


print(df[df["Country"]== "Turkey"]["City"].unique())


# In[ ]:


#Count of Null Value
print(df.isnull().sum())


# In[ ]:


#Count of -99 Value in Regions
mistake_values=df[df["AvgTemperature"]==-99]
group_mistake_values=mistake_values.groupby("Region")
for name,group in group_mistake_values:
    print(name)
    print(len(group))


# In[ ]:


#Cleaned of Nan and -99 Values
df.drop("State",axis=1,inplace=True)
_list=list(df[df["AvgTemperature"]==-99].axes)
df.drop(index=_list[0][:],inplace=True)
print(df.shape)


# In[ ]:


len(df[df["AvgTemperature"]==-99])


# In[ ]:


#Count of Null Value
print(df.isnull().sum())


# In[ ]:


#Count of Country and Region
print(len(df["Country"].unique()))
print(len(df["Region"].unique()))


# In[ ]:


#Name of Region
print(df["Region"].unique())


# In[ ]:


#Maximum Tempraturee
print(df[df["AvgTemperature"]==df["AvgTemperature"].max()])


# # Temperature Averages of European 

# In[ ]:


#number of data in months
for month in np.arange(1,13):
    print("Number of Data in "+str(month)+".Month= "+str(len(df[(df["Region"]=="Europe")& (df["Year"]>1994) & (df["Month"]==month)])))


# In[ ]:


def mounth_hot_avg(month):
    europe_data = df[(df["Region"]=="Europe") & (df["Month"]==month) & (df["Year"]>1994)]
    europa_avg_mounth_deg = europe_data.groupby("Year")["AvgTemperature"].mean()
    return europa_avg_mounth_deg
def plot(x,y,z,mounts,season):
    fig,axs = plt.subplots(3,figsize=(15,10))
    fig.suptitle('European '+ season +' Month Temperature Averages by Years',color='g',fontsize=15)
    
    axs[0].plot(x.axes[0], x, 'o-', color = '#FF7F00')
    axs[0].grid(color='r', linestyle='dotted', linewidth=1)
    axs[0].set_title(str(mounts[0])+'. Month')
    axs[0].set_xlabel("Year")
    axs[0].set_ylabel("Average Temperature")
    
    axs[1].plot(y.axes[0], y, 'o-', color = '#9900CC')
    axs[1].grid(color='r', linestyle='dotted', linewidth=1)
    axs[1].set_title(str(mounts[1])+'. Month')
    axs[1].set_xlabel("Year")
    axs[1].set_ylabel("Average Temperature")
    
    axs[2].plot(z.axes[0], z, 'o-', color = '#66FF00')
    axs[2].grid(color='r', linestyle='dotted', linewidth=1)
    axs[2].set_title(str(mounts[2])+'. Month')
    axs[2].set_xlabel("Year")
    axs[2].set_ylabel("Average Temperature")
    for ax in axs.flat:
        ax.label_outer()
    


# In[ ]:


#Winter mounts avarege temp
season = "Winter"
mounts =[12,1,2]
_1mounth_of_season = mounth_hot_avg(mounts[0])
_2mounth_of_season = mounth_hot_avg(mounts[1])
_3mounth_of_season = mounth_hot_avg(mounts[2])
plot(_1mounth_of_season,_2mounth_of_season,_3mounth_of_season,mounts,season)


# In[ ]:


#Spring mounts avarege temp
season = "Spring"
mounts =[3,4,5]
_1mounth_of_season = mounth_hot_avg(mounts[0])
_2mounth_of_season = mounth_hot_avg(mounts[1])
_3mounth_of_season = mounth_hot_avg(mounts[2])
plot(_1mounth_of_season,_2mounth_of_season,_3mounth_of_season,mounts,season)


# In[ ]:


#Summer mounts avarege temp
season = "Summer"
mounts =[6,7,8]
_1mounth_of_season = mounth_hot_avg(mounts[0])
_2mounth_of_season = mounth_hot_avg(mounts[1])
_3mounth_of_season = mounth_hot_avg(mounts[2])
plot(_1mounth_of_season,_2mounth_of_season,_3mounth_of_season,mounts,season)


# In[ ]:


#Summer mounts avarege temp
season = "Autumn"
mounts =[9,10,11]
_1mounth_of_season = mounth_hot_avg(mounts[0])
_2mounth_of_season = mounth_hot_avg(mounts[1])
_3mounth_of_season = mounth_hot_avg(mounts[2])
plot(_1mounth_of_season,_2mounth_of_season,_3mounth_of_season,mounts,season)


# In[ ]:


europe_data = df[(df["Region"]=="Europe") & (df["Year"]>1994) & (df["Year"]<2020)]
europa_avg_year_deg = europe_data.groupby("Year")["AvgTemperature"].mean()
europa_avg_year_deg 
x=europa_avg_year_deg.axes[0]
y=europa_avg_year_deg
plt.figure(figsize=(15,5))
plt.plot(x,y,"o-",linestyle='solid',label="Temperature")
plt.xticks(x, x, rotation=75)
plt.ylabel("Average Temperature",fontsize=16)
plt.xlabel("Years",fontsize=16)
plt.legend()
plt.title("Europe Temperature Average of per Year",fontdict={'fontsize': 20, 'fontweight': 'medium'},color='r')
plt.grid(color='r', linestyle='dotted', linewidth=0.5)
plt.show()


# # Temperature Averages of Regions

# In[ ]:


#Temperature Averages of Regions
Years = df[df["Year"]>1994]
regions_avg_deg = Years.groupby("Region")["AvgTemperature"].mean()
regions_avg_deg
plt.figure(figsize=(15, 5))
plt.title("Density of the Average Number of Temperatures (1995-2020)")
sns.distplot(regions_avg_deg)


# In[ ]:


#Temperature Averages of Regions
Years = df[df["Year"]==2019]
regions_avg_2019_deg = Years.groupby("Region")["AvgTemperature"].mean()
regions_avg_2019_deg
plt.figure(figsize=(15, 5))
plt.title("Density of the Average Number of Temperatures (2019)")
sns.distplot(regions_avg_2019_deg)


# **Overall the temperature average for 2019 appears to be above the temperature average for all years.**

# In[ ]:


plot_data = [
    go.Bar(x=regions_avg_deg.axes[0], y=regions_avg_deg, name='1995-2020',width=0.4,marker_color='rgb(139,136,120)'),
    go.Bar(x=regions_avg_2019_deg.axes[0], y=regions_avg_2019_deg, name='2019',width=0.4,marker_color='rgb(0,139,139)'), 
]
plot_layout = go.Layout(
        title='Temperature Averages of Regions',
        yaxis_title='Average Temperature',
        xaxis_title='Region'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
pyoff.iplot(fig)


# In[ ]:


world_data = df[(df["Year"]>1994) & (df["Year"]<2020)]
world_avg_deg = world_data.groupby("Year")["AvgTemperature"].mean()
plt.figure(figsize=(16, 9))
plt.scatter(world_avg_deg.axes[0],world_avg_deg)
plt.xlabel("Years",fontsize=16)
plt.ylabel("Average Temperature",fontsize=16)
plt.title("Temperature Averages of World by Years",fontdict={'fontsize': 20, 'fontweight': 'medium'},color='r')
plt.grid(color='r', linestyle='dotted', linewidth=0.5)
plt.show()


# # Polinomial Regression

# In[ ]:


x = world_avg_deg.axes[0].values.reshape(-1,1)
y = world_avg_deg.values.reshape(-1,1)
print(x.shape)
print(y.shape)


# In[ ]:


from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
Poly_reg=PolynomialFeatures(degree=2) 
x_poly=Poly_reg.fit_transform(x)
Lin_reg = LinearRegression()
Lin_reg.fit(x_poly,y)


# In[ ]:


plt.figure(figsize=(16, 9))
plt.scatter(x,y)
plt.xlabel("Years",fontsize=16)
plt.ylabel("Average Temperature",fontsize=16)
plt.title("Temperature Averages of World by Years",fontdict={'fontsize': 20, 'fontweight': 'medium'},color='r')
plt.grid(color='r', linestyle='dotted', linewidth=0.5)
y_pred=Lin_reg.predict(x_poly)
plt.plot(x,y_pred,color="green",label="Polinomial Regression Model")
plt.legend()
plt.show()


# Polynomial regression is applied but looks like linear regression
