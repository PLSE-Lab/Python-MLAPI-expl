#!/usr/bin/env python
# coding: utf-8

# <h1 style="text-align: center;font-size: 30px;">Weather of Bangladesh Analysis & Visualization</h1>
# 
# ---
# 
# <center><a href="https://www.animatedimages.org/cat-weather-148.htm"><img src="https://www.animatedimages.org/data/media/148/animated-weather-image-0012.gif" border="0" alt="animated-weather-image-0012" /></a> </center>
# 
# ---

# <h4>Bangladesh is called the land of six seasons (Sadartu). It has a temperate climate because of its physical location. Though the climate of Bangladesh is mainly sub-tropical monsoon, ie warm and humid; Bangla calendar year is traditionally divided into six seasons: Grisma (summer), Barsa (rainy), Sarat (autumn), Hemanta (late autumn), Shhit (winter) and Basanta (spring). Each season comprises two months, but some seasons flow into other seasons, while others are short. Actually, Bangladesh has three distinct seasons: the pre-monsoon hot season from March through May, rainy monsoon season which lasts from June through October, and a cool dry winter season from November through February. However, March may also be considered as the spring season, and the period from mid-October through mid-November may be called the autumn.</h4>
# 

# In[ ]:


from IPython.display import Image
import os
get_ipython().system('ls ../input/')


# In[ ]:


Image("../input/finalimg/IMG_20200715_102837.jpg",width=950,height=50)


# <h4>Summer (grisma) Comprises Baishakh and Jyaistha (mid-April to mid-June), the two Bangla calendar months, when days are hot and dry. But the influence of summer is usually felt from mid-March. The heat of the sun dries up the waterbodies including the rivers, canals and the wetlands. The summer days are longer than the nights. At this time the southerly or southwesterly monsoons flow over the country. </h4>

# In[ ]:


Image("../input/finalimg/IMG_20200715_102743.jpg",width=950,height=100)


# <h4>The rainy season (barsa) Traditionally spreads over Asadh and Shraban (mid-June to mid-August). However, the rainy season may start from the end of Baishakh and last up to the beginning of Kartik (mid-May to late-October). During the rainy season, the southwest monsoon winds bring plenty of rainfall (70 to 85 percent of the annual total) and occasionally lasting for days without end without any respite.
# Autumn (sharat) Lasts during Bhadra and Ashvin (mid-August to mid-October). This is traditionally the season when housewives put out clothes, musty and damp because of the rains, to air and dry in the hot sun of Bhadra. However, the bright day is often punctuated by sudden showers. The dark clouds in a grey sky, characteristic of the rainy season, are replaced by white clouds floating in a blue sky. Though at the beginning of this season, the days can be hot and sultry, towards the end of the season the nights and mornings become cool. </h4>

# In[ ]:


Image("../input/finalimg/IMG_20200715_102954.jpg",width=950,height=50)


# <div class="alert alert-block alert-info">
# <b></b> Now let's get to the main point.Let's analyze our dataset,The Weather of Bangladesh..
# </div>
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import numpy as np
import pandas as pd
import pandas_profiling
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import plotly.express as px
import plotly.io as pio
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df = pd.read_csv("/kaggle/input/bangladesh-weather-dataset/Temp_and_rain.csv")


# In[ ]:


df.head()


# In[ ]:


df.info()


# In[ ]:


df.describe()


# <h3>Let's see the whole Profile of our Dataset</h3>

# In[ ]:


profile = pandas_profiling.ProfileReport(df)
profile


# <h3>Let's see is there any null values or not</h3>

# In[ ]:


df.isnull().sum()


# In[ ]:


sns.set()
n = msno.bar(df,color="purple")


# <h3>First let's Slice our dataset into first 50 Years,and analyze the weather of these Years</h3>

# In[ ]:


df1 = df.loc[0:599,:]
df1.tail()


# <h3>Let's find the amount of  Rain with respect to Year(1901 to 1950)<h3>

# In[ ]:


sns.set()
plt.figure(figsize=(16,8))
sns.regplot(x = "Year",y="rain",fit_reg = False,data=df1)
plt.show()


# In[ ]:


pio.templates.default = "plotly_dark"
fig = px.line(df1,x='Year',y='rain',color="Year",title="Measurement of Rain according to Year")
fig.show()


# In[ ]:


df1.rain.max()


# <h4> We can see that year between 1901 to 1950 , 1919 has the maximum amount of rain
#  Let's see relation between Month & Rain </h4>

# In[ ]:


pio.templates.default = "plotly_dark"
fig = px.line(df1,x='Month',y='rain',color="Year",title="Measurement of Rain according to Month")
fig.show()


# In[ ]:


pio.templates.default = "plotly_dark"
fig = px.bar(df1[0:100],x='Month',y='rain',color="Year",title="Measurement of Rain according to Month")
fig.show()


# In[ ]:


fig = px.pie(data_frame=df1,values="rain",names='Month',labels= {"Month"},
             title="Pie chart of  Rain according to  Month")
fig.update_traces(textposition ='inside',textinfo='percent+label')
fig.show()


# <h4>Here we can see that between year 1901 to 1950, Month 6,7 & 8 that means June,July & August  has the Maximum Percentage of Rain , And we  also know that these three month's are the Rainy Season in Bangladesh</h4>

# <h3>Now let's see relation  Between Month and Temperature</h3>

# In[ ]:


pio.templates.default = "plotly_dark"
fig = px.bar(df1,x='Month',y='tem',color="Year",title="Measurement of Temperature according to Month")
fig.show()


# In[ ]:


pio.templates.default = "plotly_dark"
fig = px.scatter(df1,x='Month',y='tem',color="Year",title="Measurement of Temperature according to Month")
fig.show()


# In[ ]:


fig = px.pie(data_frame=df1,values="tem",names='Month',labels= {"Month"},
             title="Pie chart of Temperature according to  Month")
fig.update_traces(textposition ='inside',textinfo='percent+label')
fig.show()


# <h4>Here we can see that  Month 1,2,11,12 that means January,February,November & December has the minimum percentage of Temperature & we also know that these 4 month's are the Winter Season & though the rest of the month's are consists of different seasons but  all  the People of Bangladesh know that,here almost all of the Month's it feels like Summer</h4>

# <h3>Let's see Relation between Temperature & Rain</h3>

# In[ ]:


pio.templates.default = "plotly_dark"
fig = px.scatter(df1,x='tem',y ='rain',color="Month",title="Measurement of Rain according to Temperature")
fig.show()


# <h4>Here we can see that as the temperature getting Increased the amount of rain  is getting increased.So we can say that the relation between Temperature & Rain is kind of Proportional</h4>

# <h3>Now let's see changes of 'rain' according to 'temperature' throughout the 'year'</h3>

# In[ ]:


pio.templates.default = "plotly_dark"
fig = px.scatter(df1,x='tem',y ='rain',animation_frame ='Year',animation_group = 'Month',
                 title="Changes of Rain according to Temperature During the Year",size="rain",color="tem",
                 hover_name="Month",log_x = True,size_max = 55)
fig.show()


# <div class="alert alert-block alert-info">
# <b></b> !!..........................Click the Play Button.........................!!
# </div>

# <h3>Now let's analyze the leftover years which is 1951 to 2015</h3>

# In[ ]:


df2 = df.loc[600:,:]
df2.tail()


# <h3>Let's find the amount of  Rain with respect to Year(1951 to 2015)</h3>

# In[ ]:


pio.templates.default = "plotly_dark"
fig = px.line(df2,x='Year',y='rain',color="Year",title="Measurement of Rain according to Year")
fig.show()


# In[ ]:


df2.rain.max()


# <h4> We can see that year between 1951 t0 2015 , year 2011 has the maximum amout of rain</h4>

# <h3>Let's see relation between Month & Rain</h3>

# In[ ]:


pio.templates.default = "plotly_dark"
fig = px.line(df2,x='Month',y='rain',color="Year",title="Measurement of Rain according to Month")
fig.show()


# In[ ]:


pio.templates.default = "plotly_dark"
fig = px.bar(df2,x='Month',y='rain',color="Year",title="Measurement of Rain according to Month")
fig.show()


# In[ ]:


fig = px.pie(data_frame=df2,values="rain",names='Month',labels= {"Month"},
             title="Pie chart of  Rain according to  Month")
fig.update_traces(textposition ='inside',textinfo='percent+label')
fig.show()


# <h4> Here we can also see that year between  1951 to 2015, Month 6,7 & 8 that means June,July & August  has the Maximum Percentage of Rain , And we  also know that these three month's are the Rainy Season in Bangladesh </h4>

# <h3>Now let's see relation  Between Month and Temperature</h3>

# In[ ]:


pio.templates.default = "plotly_dark"
fig = px.bar(df2,x='Month',y='tem',color="Year",title="Measurement of Temperature according to Month")
fig.show()


# In[ ]:


fig = px.pie(data_frame=df2,values="tem",names='Month',labels= {"Month"},
             title="Pie chart of Temperature according to  Month")
fig.update_traces(textposition ='inside',textinfo='percent+label')
fig.show()


# <h4>We can see that, it's the same, compared to year between 1901 to 1950, Month's 1,2,11&12 that means January,February,November & December has the minimum percentage of Temperature,& we also know that these 4 month's are the Winter Season & though the rest of the month's are consists of different seasons but  all  the People of Bangladesh know that,here almost all of the Month's it feels like Summer </h4>

# <h3>Let's see Relation between Temperature & Rain (1951 - 2015)</h3>

# In[ ]:


pio.templates.default = "plotly_dark"
fig = px.scatter(df1,x='tem',y ='rain',color="Month",title="Measurement of Rain according to Temperature")
fig.show()


# <h4> Here we can also see the same that as the temperature getting Increased the amount of rain  is getting increased.So here we can also say that the relation between Temperature & Rain is kind of Proportional</h4>

# <h3>Now let's see changes of 'rain' according to 'temperature' throughout the 'year'</h3>

# In[ ]:


pio.templates.default = "plotly_dark"
fig = px.scatter(df2,x='tem',y ='rain',animation_frame ='Year',animation_group = 'Month',
                 title="Changes of Rain according to Temperature During the Year",size="rain",color="tem",
                 hover_name="Month",log_x = True,size_max = 55)
fig.show()


# <div class="alert alert-block alert-info">
# <b></b> ....................................Click on the Play Button...........................................
# </div>

# <h3>Now let's separate our Month's according to different Seasons of Bangladesh</h3>

# In[ ]:


mapping = {1 : "Winter",2:"Winter",12: "Winter",3:"Spring",4:"Spring",10:"Late autumn",
           11:"Late autumn",8:"Autumn",9:"Autumn",6:"Rainy",7:"Rainy",4:"Summer",5:"Summer"}
df["Season"] = df["Month"].map(mapping).astype(str)


# In[ ]:


df["Season"].value_counts()


# In[ ]:


df.head()


# <h3>Now let's see the weather of Bangladesh according to the 6 seasons (1901-1950)</h3>

# In[ ]:


df3 = df.loc[0:599,:]
df3.tail()


# <h3>Let's see amount of rain according to Season</h3>

# In[ ]:


pio.templates.default = "plotly_dark"
fig = px.bar(df3,x='Season',y='rain',color="Year",title="Rain according to Season(1901-1950)")
fig.show()


# In[ ]:


fig = px.pie(data_frame=df3,values="rain",names='Season',labels= {"Season"},
             title="Pie chart of Rain according to  Season")
fig.update_traces(textposition ='inside',textinfo='percent+label')
fig.show()


# <h4>So we can see that Rainy Season has the  heighest percentage of Rain,Autumn also got a high percentage ,that's because half of august is considered Rainy season,but in the dataset ,i considered full August as the Autumn Season,that's why it also got a high percentage</h4>

# <h3>Let's see live changes  of rain,temperature & Season During the Year(1901-1950)</h3>

# In[ ]:


pio.templates.default = "plotly_dark"
fig = px.scatter(df3,x='tem',y ='rain',animation_frame ='Year',animation_group = 'Month',
                 title="Relation between rain,temperature & Season During the Year(1901-1950)",size="rain",color="Season",
                 hover_name="Month",log_x = True,size_max = 55)
fig.show()


# <h3>Now let's see the weather of Bangladesh according to the 6 seasons (1951-2015)</h3>

# In[ ]:


df4 = df.loc[600:,:]
df4.tail()


# <h3>Let's see amount of rain according to Season(1951-2015)</h3>

# In[ ]:


pio.templates.default = "plotly_dark"
fig = px.bar(df4,x='Season',y='rain',color="Year",title="Rain according to Season(1951-2015)")
fig.show()


# In[ ]:


fig = px.pie(data_frame=df4,values="rain",names='Season',labels= {"Season"},
             title="Pie chart of Rain according to  Season (1951-2015)")
fig.update_traces(textposition ='inside',textinfo='percent+label')
fig.show()


# <h4>Here we can see that, The percentage is almost Same compared to year between(1901-1950). The Rainy Season has the  heighest percentage of Rain</h4>

# <h3>Let's see live changes  of rain,temperature & Season During the Year(1951-2015)</h3>

# In[ ]:


pio.templates.default = "plotly_dark"
fig = px.scatter(df4,x='tem',y ='rain',animation_frame ='Year',animation_group = 'Month',
                 title="Relation between rain,temperature & Season During the Year(1951-2015)",size="rain",color="Season",
                 hover_name="Month",log_x = True,size_max = 55)
fig.show()


#  <div class="alert alert-block alert-info">
# <b></b> ..........................If you like the Notebook then do UpVote,Your appreciation means a lot!!..........
#     
#     
# </div>
#  

# In[ ]:




