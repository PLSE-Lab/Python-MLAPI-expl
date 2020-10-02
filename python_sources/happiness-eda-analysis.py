#!/usr/bin/env python
# coding: utf-8

# # LOAD THE LIBRARIES

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly.offline import iplot
from scipy.stats import pearsonr
import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from sklearn.preprocessing import scale
from sklearn.preprocessing import normalize
import scipy.cluster.hierarchy as shc
from scipy import stats
sns.set_style("darkgrid")
plt.style.use("fivethirtyeight")
import warnings
warnings.filterwarnings('ignore')


# # LOAD DATA

# In[ ]:


df=pd.read_csv('../input/world-happiness/2019.csv')


# # SEE THE COLUMNS AND DATA AND ALSO DO BASIC STATISTICS

# In[ ]:


df.columns


# In[ ]:


df.head(10)


# In[ ]:


df.shape


# it has 156 rows and 9 columns.

# In[ ]:


df.min()


# In[ ]:


df.style.background_gradient(cmap='Blues')


# darker the color gradient higher the index.

# In[ ]:


df.info()


# In[ ]:


df.describe().style.background_gradient(cmap='Blues')


# # FINDING MISSING VALUE

# In[ ]:


df.isna().sum()


# we can see that there is no null value*.*

# # correlation in between the columns

# In[ ]:


df.corr().style.background_gradient(cmap="Greens")


# # BASIC VISUALIZATION

# HEATMAP

# In[ ]:


plt.figure(figsize=(20,8))
sns.heatmap(df.corr(), annot=True);


# # REORGNISED THE COLUMNS NAME

# In[ ]:


df.rename(columns = {'Overall rank':'RANK', 'Country or region':'REGION','GDP per capita':'GDP','Social support':'SCOICL','Healthy life expectancy'
                    :'HIE','Freedom to make life choices':'FTMLC','Perceptions of corruption':'CORUP'}, inplace = True)


# In[ ]:


df.columns


# In[ ]:


df1=df[['Score','RANK','GDP','CORUP']]


# In[ ]:


df1.head(3)


# In[ ]:


cormat=df1.corr()


# In[ ]:


sns.heatmap(cormat, annot=True);


# In[ ]:


differ=df['Score'].max()-df['Score'].min()
z=round(differ/3,3)
low=df['Score'].min()+z
mid=low+z


# In[ ]:


cat=[]
for i in df.Score:
    if(i>0 and i<low):
        cat.append('Low')
    elif(i>low and i<mid):
         cat.append('Mid')
    else:
         cat.append('High')
df['Category']=cat  


# In[ ]:


color = (df.Category == 'High' ).map({True: 'background-color: lightblue',False:'background-color: red'})
df.style.apply(lambda s: color)


# # COUNT THE FEATURES FOR EACH COUNTRIES IN BAR PLOT

# In[ ]:


fig = px.bar(df, x='REGION', y='Score',
             hover_data=['RANK', 'GDP', 'SCOICL', 'HIE', 'FTMLC'], color='GDP')
fig.show()


# 1. above barplot give us a clean idea about score for each countries in terms of social supprt, freedom, health, rank. color is varied for gdp. if gdp is high color is yellow and blue for low gdp.
# 2. please hover the mouse for detail view.

# # Visualization of the percentage of happiness according to regions and social support affecting it.
# With the joint plot we can see the correlation between the two features.

# In[ ]:


ndf=pd.pivot_table(df, index = 'REGION', values=["Score","SCOICL"])
ndf["Score"]=ndf["Score"]/max(ndf["Score"])
ndf["SOCIAL"]=ndf["SCOICL"]/max(ndf["SCOICL"])
sns.jointplot(ndf.SCOICL,ndf.Score,kind="kde",height=10,space=0.5)
plt.savefig('graph.png')
plt.show()


# In[ ]:


df.columns


# In[ ]:


fig = px.scatter_matrix(df,dimensions=['RANK', 'GDP', 'SCOICL', 'HIE', 'FTMLC','CORUP'],color='Category')
fig.show()


# # the above pair plot we can visualise the correlation in between the parameters like gdp, social supprt, freedom, health, rank, corruption. and how all this parameters varies in the high, mid, low happy countries. 

# # building a ordinary linear regression model

# In[ ]:


y = df['GDP']
x =  df[['RANK', 'SCOICL', 'HIE', 'FTMLC','CORUP']]
model = sm.OLS(y, x)
results = model.fit()
print(results.summary())


# In[ ]:


print('Parameters: ', results.params)
print('Standard errors: ', results.bse)


# in the above model we can observed the t test value, p value, std err, skewness, kurtosis and all the statistical parameters. 

# # Geographical Visualization of Score

# In[ ]:


data = dict(type = 'choropleth', 
           locations = df['REGION'],
           locationmode = 'country names',
           colorscale='RdYlGn',
           z = df['Score'], 
           text = df['REGION'],
           colorbar = {'title':'Score'})

layout = dict(title = 'Geographical Visualization of Happiness Score', geo = dict(showframe = True, projection = {'type': 'azimuthal equal area'}))

choromap3 = go.Figure(data = [data], layout=layout)
iplot(choromap3)


# # clustering

# In[ ]:


df2=df.drop(['REGION', 'RANK','Category'], axis=1)


# In[ ]:


df2.head()


# In[ ]:


fig = ff.create_dendrogram(df2, color_threshold=1)
fig.update_layout(width=2000, height=500)
fig.show()


# in clustering the x axis represent the serial no of countries according their rank.

# In[ ]:




