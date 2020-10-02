#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


df=sns.load_dataset("tips")
df.head()


# # 1.relplots:
#    *To view relationship between stastical variables relplots are used*       
#    *relplot refer to relational plots i guess..it has two flavours
#     scatterPlots and linePlots*

# ## 1.1 RelPlots : Scatter plots
#    *The scatter plot is a mainstay of statistical visualization. It depicts the joint distribution of two variables using a cloud of points, where each point represents an observation in the dataset. This depiction allows the eye to infer a substantial amount of information about whether there is any meaningful relationship between them.*

# In[ ]:


#1.relplot(kind='scatter') when not mentioned
sns.relplot(x='total_bill',y='tip',data=df)  


# In[ ]:


# 2.For 3 variables, add hue
sns.relplot(x='total_bill',y='tip',hue='sex',data=df)


# In[ ]:


#3.For 4 variables, add hue + style
sns.relplot(x='total_bill',y='tip',hue='sex',style='time',data=df)


# ### Color Schemes
# > random colors are chosen when representing categorical datas in hue .for example male/female,lunch/dinner..etc
# > however when a float value is used in hue the color scheme changes to say one color and their intensity to represent the values..

# In[ ]:


#4. when numeric data type is used in hue it show one color & their intensity to represent the value
sns.relplot(x='total_bill',y='tip',hue='size',data=df)


# In[ ]:


#5.if you are more of bigger the better type ...then
sns.relplot(x='total_bill',y='tip',hue='sex',size='size',data=df)


# In[ ]:


#6.well the circles arent big enough so to buff it up..
sns.relplot(x='total_bill',y='tip',hue='sex',size='size',sizes=((15, 200)),data=df) # 15 & 200 can be replaced with min and max values too


# In[ ]:


#7.to see tip vs total_bill for male and female individual.add col
sns.relplot(x='total_bill',y='tip',hue='smoker',col='sex',data=df)


# In[ ]:


#8.to see tip vs total_bill for male smoker and non smoker and female smoker and non smoker,add row
sns.relplot(x='total_bill',y='tip',hue='smoker',col='sex',row='smoker',data=df)


# In[ ]:


#9.UI lets make them thinner by reducing the aspect
sns.relplot(x='total_bill',y='tip',hue='smoker',col='sex',aspect=.4,data=df)


# In[ ]:


#10.well each fig got thinner but all stackup on eachother.lets wrap them up using col_wrap
#sns.relplot(x='total_bill',y='tip',hue='smoker',col='sex',row='size',aspect=.4,col_wrap=5,data=df)
#row and col_wrap cannot be used..Noted


# ## 1.2. Scatter plots Summary:-
#     relplots(
#         x=variable 1
#         y=variable 2
#         hue=variable 3(color change depends on var 3 data type)
#         style=variable 4
#         size=to show variables value in size
#         sizes=to change min and max variable for the parameter('size')
#         col=to see relationship between two variables at two different events(col=thatDifferentEventCol)
#         row=same as cow  except can see two variables at four differnt events(00,01,10,11)
#         
#         col_wrap=to put all side by side in the fig (no of fig u want in a row)        
#         aspect=aspectRatio of the fig
#         height=height of the fig
#         )

# # 2.catplots():
#    ***if type(variable1),type(variable2) ==numeric then scatter plots are used which makes sense
#    but if
#    type(variable1)==categorical and type(variable2)==numeric then scatter plots can be used but makes less sense
#    hence we are moving to specialized plots ->catplots***

# In[ ]:


#11.see we can plot like this But can't see more of any pattern
sns.relplot(x="day", y="total_bill", data=df)


# ## 2.1 catplots : Stripplot

# In[ ]:


#12.But this looks visually appealing
sns.catplot(x="day", y="total_bill",kind='strip', data=df) #no need to mention strip as it is the default


# ## 2.2 Catplots : Swarmplots

# In[ ]:


#13. Lets try 'Swarm'
sns.catplot(x="day", y="total_bill",kind='swarm', data=df) 
#although it looks same to striplot it doesn't overlap datas.so we can see the distribution more clearly in swarm but using it large data set might look aweful.


# In[ ]:


#14.Nothing fancy just added an another variable 'sex' using hue.
sns.catplot(x="day", y="total_bill",hue='sex',kind='swarm', data=df) 


# In[ ]:


#15.what if i want to reverse the order of days,add order which accepts list..
sns.catplot(x="day", y="total_bill",hue='sex',kind='strip',order=["Sun","Thur","Fri","Sat"],data=df)


# In[ ]:


#16. we can filterout certain category here i kicked out persons whose team size=3. using query function ..but do know it is a dataframe function
sns.catplot(x="size", y="total_bill",kind='swarm',data=df.query('size!=3'))


# ### Distributions of observations within categories
#    *As the size of the dataset grows,, categorical scatter plots become limited in the information they can provide about the distribution of values within each category. When this happens, there are several approaches for summarizing the distributional information in ways that facilitate easy comparisons across the category levels*

# ## 2.3 Catplots : Box & Boxen plots

# In[ ]:


#17.we'll try box plot to see the distribution of values
sns.catplot(x="day", y="total_bill",hue='sex',data=df,kind='box')


# ### box plot meaning:
# 
#    *The colored block (orange and blue) shows where 50% of the data are. The lower boundary is the 25-percentile, the upper boundary is the 75-percentile. The line in the middle is the mean.
# The distance between the 75-percentile and the 25-percentile is called "interquartile range" (IQR). 1.5 times the IQR from the 25-percentile border is the lower whisker. Everything lower than that is an outlier and only represented by a dot. The same for the upper whisker*

# In[ ]:


#18.to stack up hue value use dodge=false
sns.catplot(x="day", y="total_bill",hue='sex',dodge=False,data=df,kind='box')


# In[ ]:


#19.boxen plots is similar to box but it shows more distribution of data in larger data set then box plots
sns.catplot(x="day", y="total_bill",hue='sex',data=df,kind='boxen')


# ## 2.4 Catplots : ViolinPlots

# In[ ]:


sns.catplot(x="day", y="total_bill",hue='sex',data=df,kind='violin')


# ### violinplot meaning:
# 
#    *the bulged area shows how much the values are distributed in a range..here for example on thursday more person ate in the range 5 to 25 dollars and less people ate in the range of 30 to 50 and the white dot thing is the mean value and the box and whiskers are from box plots*

# In[ ]:


#20 .Violin plots looks good but sometimes when u have negative values then u need to cut extreme negative ends using 'cut'
sns.catplot(x="day", y="total_bill",hue='sex',data=df,kind='violin',bw=.10,cut=10)


# > bw refers to bandwidth and cut refers to the scale but i dont understand both as of now

# In[ ]:


#20 .to save visual space both values in hue can be shown together using 'split'
sns.catplot(x="day", y="total_bill",hue='sex',data=df,kind='violin',split=True,bw=.10,cut=10)


# In[ ]:


#20 .instead of the box plot inside you can see the actual distribution using 'inner'
sns.catplot(x="day", y="total_bill",hue='sex',data=df,kind='violin',split=True,inner='sticks')


# In[ ]:


#21.u can combine swarm plot with violin plots too..

violinPlot = sns.catplot(x="day", y="total_bill", kind="violin", inner='sticks', data=df) # put inner =None as it looks terrible  with swarms already
sns.catplot(x="day", y="total_bill",color="k",kind='swarm', data=df,ax=violinPlot.ax)  #add color to differtiate from background


# ## 2.5 Catplots : BarPlots & CountPlots

# In[ ]:


#22.the most common plot used all over..
sns.catplot(x="day", y="size",data=df,kind='bar')


# In[ ]:


#22.it is used to estimate the number of values it contains itself ..meaning:how many cats ,dogs,donkeys present in a animals column assuming only these 3 animals exist in it
sns.catplot(x='day',data=df,kind='count')


# ## 2.5 Catplots : Point plots
#   *This function also encodes the value of the estimate with height on the other axis, but rather than showing a full bar, it plots the point estimate and confidence interval. Additionally, pointplot() connects points from the same hue category. This makes it easy to see how the main relationship is changing as a function of the hue semantic, because your eyes are quite good at picking up on differences of slopes:*

# In[ ]:


sns.catplot(x="size", y="tip",hue='sex',data=df,kind='point')

