#!/usr/bin/env python
# coding: utf-8

#    # loading the libraries

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly.offline import iplot, init_notebook_mode
import plotly.offline as py
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


# # load data

# In[ ]:


df=pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv',parse_dates=['Last Update'])
df.rename(columns={'Province/State':'State','Country/Region':'Country'}, inplace=True)
df = df.drop(columns = ['SNo', "Last Update"])
dfdeath = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv")
dfdeath.rename(columns={'Province/State':'State','Country/Region':'Country'}, inplace=True)
dfcnf = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv")
dfcnf.rename(columns={'Province/State':'State','Country/Region':'Country'}, inplace=True)


# # some basic insight of the data

# In[ ]:


print("the no of rows and columns are:",df.shape)
print("the no of rows and columns are:",dfcnf.shape)
print("the no of rows and columns are:",dfdeath.shape)


# In[ ]:


print("information about the data:", df.info())


# In[ ]:


print("a brief and basic statistical description of the data frame: ")
df.describe().style.background_gradient(cmap='Blues')


# In[ ]:


print("a brief and basic statistical description of the data frame: ")
dfcnf.describe().style.background_gradient(cmap='Blues')


# In[ ]:


print("a brief and basic statistical description of the data frame: ")
dfdeath.describe().style.background_gradient(cmap='Blues')


# In[ ]:


df.head(10)


# In[ ]:


print("convert ObservationDate,Last Update to datetime as they are object and set all the other column except region and sate into numerical values")
df['ObservationDate'] = pd.to_datetime(df['ObservationDate'])
df['Confirmed']=df['Confirmed'].astype('int')
df['Deaths']=df['Deaths'].astype('int')
df['Recovered']=df['Recovered'].astype('int')
df.info()


# In[ ]:


df.tail(10)


# In[ ]:


print("finding the missing values: ", df.isna().sum())


#     as we can see that only province column has some missing value and it is acceptble.

# In[ ]:


print("finding missing values: ", dfcnf.isna().sum())


# In[ ]:


print("finding missing values: ", dfdeath.isna().sum())


# # basic analysis start here: 

# In[ ]:


print(" lets start with symptoms for covid-19\n Souce is taken for  wiki and othe website for symptoms")


# In[ ]:


symptoms={'symptoms':['Fever','Dry cough','Fatigue','Sputum production','Shortness of breath','Muscle pain',
        'Sore throat','Headache','Chills','Nausea or vomiting','Nasal congestion','Diarrhoea','Haemoptysis',
        'Conjunctival congestion'],'percentage':[87.9,67.7,38.1,33.4,18.6,14.8,13.9,13.6,11.4,5.0,4.8,3.7,0.9,0.8]}


# In[ ]:


symptoms=pd.DataFrame(data=symptoms,index=range(14))
symptoms.style.background_gradient(cmap='Blues')


# In[ ]:


fig = px.bar(symptoms[['symptoms', 'percentage']].sort_values('percentage'), y="percentage", x="symptoms", color='symptoms', 
             log_y=True, title='Symptoms of covid-19')
fig.show()


# # now work on the data set:

# In[ ]:


df.columns


# In[ ]:


df.corr().style.background_gradient(cmap="Greens")


# In[ ]:


py.init_notebook_mode(connected=True)


# In[ ]:


group = df.groupby(['ObservationDate','Country'])['Confirmed', 'Deaths', 'Recovered'].max()
group = group.reset_index()
group


# In[ ]:


group['Date'] = pd.to_datetime(group['ObservationDate'])
group['Date'] = group['Date'].dt.strftime('%m/%d/%Y')
group['Active'] = group['Confirmed'] - group['Recovered'] - group['Deaths']
group['Country'] =  group['Country']

fig = px.choropleth(group, locations="Country", locationmode='country names', 
                     color="Active", hover_name="Country",hover_data = [group.Recovered,group.Deaths,group.Active],projection="mercator",
                     animation_frame="Date",width=1000, height=700,
                     color_continuous_scale='Reds',
                     range_color=[100,10000],title='World Map of Coronavirus')

fig.update(layout_coloraxis_showscale=True)
py.offline.iplot(fig)


# # heat map of the world.

# In[ ]:


fig = px.treemap(df, path=['Country'], values='Confirmed',color='Confirmed', hover_data=['Country'], color_continuous_scale='burgyl')
fig.show()


# # visualization of correlation in between the parameters in a pairwise format

# In[ ]:


fig = px.scatter_matrix(df,dimensions=['Confirmed', 'Deaths','Recovered'],color='Recovered')
fig.show()


# # building a ordinary linear regression model

# In[ ]:


y = df['Deaths']
x =  df[['Confirmed','Recovered']]
model = sm.OLS(y, x)
results = model.fit()
print(results.summary())


# In[ ]:


print('Parameters: ', results.params)
print('Standard errors: ', results.bse)


# In[ ]:


fig = px.bar(df[['Country', 'Recovered']].sort_values('Recovered', ascending=False), y="Recovered", x="Country", color='Country', 
             log_y=True, template='ggplot2', title='Recovered Cases')
fig.show()


# In[ ]:


fig = px.bar(df[['Country', 'Deaths']].sort_values('Deaths', ascending=False),y="Deaths", x="Country", color='Country', 
             log_y=True, template='ggplot2', title='Death')
fig.show()


# In[ ]:


top5 = df.groupby(['Country']).sum().nlargest(5,['Confirmed'])
top5
print("Top 5 Countries were affected most")
print(top5)


# In[ ]:


fig = px.scatter(df, x="Country", y="Recovered", color="Recovered")
fig.show()


# # this is a very basic analysis. please upvote if like it.

# In[ ]:




