#!/usr/bin/env python
# coding: utf-8

# ## Explanatory Analysis

# In[ ]:


# Importing all packages and setting plots to be embedded inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import statsmodels.api as sm
from sklearn.cluster import KMeans

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# Loading the dataset into a pandas dataframe
df=pd.read_csv('../input/1987_modified.csv')


# #### Explanatory Polishing
# 
# > Distribution of total delays has been explored using histogram.
# 
# > Distribution of unique carrier with total delays.
# 
# > Barplot has been plotted for unique carrier vs total on monthly basis to get a clear insight for every month.
#  
# > Relating total delays with distance.
# 
# > Exploration for distance vs actual elapsed time reveals some sort of linear relation between them.
# 
# > Relating distance vs total delay on month basis.

# ### Distribution of total delay
# > The plot shows that majority of the delays are within 25 minutes,and the number decreases significantly as the duration increases.
# 
# > This shows that major delays are few and if we reduce the minor delays air transport will be optimised and those delays with high values are actually outliers.

# In[ ]:


plt.figure(figsize=[8,5])
bins=np.arange(0,df['Total_Delay'].max()+10,25)
plt.hist(data=df,x='Total_Delay',bins=bins,rwidth=0.7)
plt.title('Distribution of Total delay')
plt.xlabel('Delay duration in minutes')
plt.xlim(0,400)
plt.xticks([0,25,50,75,100,125,150,175,200,225,250,275,300,325,350,375]);


# ### Distribution of unique carrier with total delay 
# 
# > From the barplot it clear that PS plane carrier has major contribution the delays followed by CO and AS.

# In[ ]:


plt.figure(figsize=[8,5])
sb.barplot(data=df,x='UniqueCarrier',y='Total_Delay')
plt.title('Unique Carrier vs Total delay')
plt.xlabel('Carrier Code')
plt.ylabel('Count of Delays');


# ### Barplot for distribution of unique carrier with total on monthly basis
# 
# > The graph shows that most of the delays in the month of October and November are of flights belonging to PS.
# Only in the month of December CO carrier has the majority of delays.The result seems quite obvious as seen above that PS flight carrier has the major contribution in the total delay followed by CO carrier.
# 
# > Also one interesting fact that the out of the three months Month 11 i.e. November has the most number of delays. 

# In[ ]:


g=sb.FacetGrid(data=df,col='Month',margin_titles=True,sharex=False,sharey=False,height=5)
g.map(sb.barplot,'UniqueCarrier','Total_Delay')
g.add_legend();


# ### Distance vs Total Delay
# 
# > The scatterplot does not depict any clear relation between distance and total delay as the data points are present as clusters.The correlation coefficient comes out to 0.05 approximately, which clearly shows that there's hardly any correlation between distance and total delay.
# 
# > The majority of delays are clustered between 0 to 3000 miles distance.

# In[ ]:


plt.figure(figsize=[8,5])
sb.regplot(data=df,x='Total_Delay',y='Distance',fit_reg=False)
plt.title('Distance vs Total delay')
plt.xlabel('Total delay in minutes')
plt.ylabel('Distance in miles');


# ### Distance vs Actual elapsed time
# 
# > The scatterplot for distance vs actual elapsed time does depict some sort of linear relation.
# 
# > The correlation coefficient calculated above also has a high value of 0.97 which shows that there's strong positive correlation between distance and actual elapsed time.

# In[ ]:


sb.regplot(data=df,y='Distance',x='ActualElapsedTime',fit_reg=False)
plt.title('Distance vs Actual elapsed time')
plt.xlabel('Actual elapsed time in minutes')
plt.ylabel('Distance in miles');


# ### Distance vs total delay month wise
# 
# > This graph depicts distance vs total delay on monthly basis.For each of the scatter plot correlation coefficient has been calculated.The correlation coefficient for each of them is quite close to zero which depicts that there is no relation between distance and total delay.

# In[ ]:


g=sb.FacetGrid(data=df,col='Month',margin_titles=True,sharex=False,sharey=False)
g.map(plt.scatter,'Total_Delay','Distance')
g.add_legend();


# ### Websites referred
# 
# > https://stackoverflow.com/
# 
# > https://pandas.pydata.org/
