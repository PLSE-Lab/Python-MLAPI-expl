#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


data = pd.read_csv("../input/international-football-results-from-1872-to-2017/results.csv")


# In[ ]:


data.corr()


# In[ ]:


data.head(10)


# In[ ]:


data.columns


# In[ ]:


data.home_score.plot(kind= 'line', color= 'g', label= 'home_score', linewidth= 2, alpha= 0.5, grid= True, linestyle= ':')
data.away_score.plot(color= 'r', label= 'away_score', linewidth= 2, alpha= 0.5, grid= True, linestyle= ':')
plt.legend(loc= 'upper right')          #legend= puts label into plot
plt.xlabel('x axis')                    #label= name of label
plt.ylabel('y axis')                    #label= name of label
plt.title('Line Plot')                  #title= title of plot
plt.show()


# In[ ]:


data.plot(kind = 'scatter', x = 'home_score', y= 'away_score', alpha = 0.5, color = 'red')
plt.xlabel('home_score')
plt.ylabel('away_score')
plt.title('Home score - Away score Scatter Plot')
plt.show()


# In[ ]:


data.away_score.plot(kind = 'hist',bins = 50,figsize = (12,12))
plt.show()


# In[ ]:


data.away_score.plot(kind = 'hist',bins = 50)
plt.clf()


# In[ ]:


series = data['home_score']        
print(type(series))     
data_frame = data[['away_score']]  
print(type(data_frame))


# In[ ]:


x = data['away_score']>15 
data[x]


# In[ ]:


data[(data['home_score']>5) & (data['away_score']>5 )]

