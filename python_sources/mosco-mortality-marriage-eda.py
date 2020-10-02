#!/usr/bin/env python
# coding: utf-8

# ![irina-grotkjaer-dyPNRXLevJY-unsplash.jpg](attachment:irina-grotkjaer-dyPNRXLevJY-unsplash.jpg)

# Thank you Vitaliy Malcev for providing this dataset. 
# 
# **Objectives**:
# * To find the seasonality of Birth and Death Rate in Moscow
# * To find seasonality of Marriages and Divorces in Moscow

# In[ ]:



#importing Libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)    #THIS LINE IS MOST IMPORTANT AS THIS WILL DISPLAY PLOT ON 
#NOTEBOOK WHILE KERNEL IS RUNNING
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")



import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


df = pd.read_csv('../input/mortaliy-moscow-20102020/moscow_stats.csv')
df.head()


# In[ ]:


df.info()


# # **Birth Vs Death In Moscow**

# In[ ]:


dfpopy = df.copy()
dfpopy = dfpopy.groupby(['Year'])['NumberOfBirthCertificatesForBoys', 
                                  'NumberOfBirthCertificatesForGirls','StateRegistrationOfBirth',
                                  'StateRegistrationOfDeath',].sum().reset_index()


dfpopy.style.background_gradient(cmap='Greens')


# **Inference**
# 
# * The Year 2016 has the most Total Number of Births 
# 
# * The Year 2010 has the most Number of Death

# In[ ]:


dfpopm = df.copy()
dfpopm = dfpopm.groupby(['Month'])['NumberOfBirthCertificatesForBoys', 
                                   'NumberOfBirthCertificatesForGirls','StateRegistrationOfBirth',
                                   'StateRegistrationOfDeath',].sum().reset_index()
dfpopm.style.background_gradient(cmap='Greens')


# **Inference**
# 
# * The month of July has the Most Number of Births
# 
# * The month of January has the Most Number of Deaths
# 

# In[ ]:




fig = go.Figure(data=go.Heatmap(
                   z= df['StateRegistrationOfBirth'],
                   x=df['Month'],
                   y= df['Year'],
                   hoverongaps = False))
fig.update_layout(
    title_text= ' Birth in Moscow',
    yaxis = dict(
        tickmode = 'array',
        tickvals = [2010, 2011, 2012, 2013, 2014, 
                    2015, 2016, 2017 , 2018, 2019, 2020],
        ticktext = ['2010', '2011', '2012', '2013', '2014', 
                    '2015', '2016', '2017' , '2018', '2019', '2020'],),
    paper_bgcolor='rgb(233,233,233)',
    
    )
fig.show()


# The given dataset has only 4 months data of 2020
# 
# **Inference**
# 
# * The months of July and August has most number of Births over the years.

# In[ ]:




fig = go.Figure(data=go.Heatmap(
                   z= df['StateRegistrationOfDeath'],
                   x=df['Month'],
                   y= df['Year'],
                  
                   hoverongaps = False))
fig.update_layout(
    title_text= ' Death in Moscow',
    yaxis = dict(
        tickmode = 'array',
        tickvals = [2010, 2011, 2012, 2013, 2014, 2015, 
                    2016, 2017 , 2018, 2019, 2020],
        ticktext = ['2010', '2011', '2012', '2013', '2014', 
                    '2015', '2016', '2017' , '2018', '2019', '2020'],),
    paper_bgcolor='rgb(233,233,233)',
    
    )
fig.show()


# **Inference**
# 
# * The Month of January has the most number of Death over the years
# 
# * For the year 2010 , July and August has the highest number of Deaths

# **Comparing Birth Rate and Death Rate**

# **The crude birth rate (CBR) is equal to the number of live births (b) in a year divided by the total midyear population (p), with the ratio multiplied by 1,000 to arrive at the number of births per 1,000 people.**

# In[ ]:


df2 = df.copy()
df2['BirthRate'] = df2['StateRegistrationOfBirth'] / df2['TotalPopulationThisYear'] * 1000
df2['DeathRate'] = df2['StateRegistrationOfDeath']/df2['TotalPopulationThisYear'] * 1000
df2['BirthRateF'] = df2['NumberOfBirthCertificatesForBoys'] / df2['TotalPopulationThisYear'] * 1000
df2['BirthRateM'] = df2['NumberOfBirthCertificatesForGirls']/df2['TotalPopulationThisYear'] * 1000


# In[ ]:


df3 = df2.groupby(['Year'])['BirthRate','DeathRate','BirthRateF','BirthRateM' ,].sum().reset_index()
df4 = df2.groupby(['Month'])['BirthRate','DeathRate','BirthRateF','BirthRateM' ,].sum().reset_index()


# **Comparing Birth Rate of Male and Female Yearwise**

# In[ ]:


fig = go.Figure(data=[
    go.Bar(name='Birth Rate Male', x= df3['Year'], y= df3['BirthRateF']),
    go.Bar(name='Birth Rate Female', x= df3['Year'], y=df3['BirthRateM'])
])
# Change the bar mode
fig.update_layout(barmode='group',
                 xaxis = dict(
        tickmode = 'array',
        tickvals = [2010, 2011, 2012, 2013, 2014, 2015, 
                    2016, 2017 , 2018, 2019, 2020],
        ticktext = ['2010', '2011', '2012', '2013', '2014', 
                    '2015', '2016', '2017' , '2018', '2019', '2020'],),
        paper_bgcolor='rgb(233,233,233)',
        title_text= 'Birth Rate of Male & Female Year Wise ',)
fig.show()


# **Inference**
# 
# * Comparatively Female birth rate was always less than Male Birth rate
# 
# * Male Birth Rate was Highest in 2016 and was lowest in 2011.
# 
# * Female Birth Rate was Highest in 2016 and was lowest in 2011.
# 
# 2016 seems to be good year for Moscow

# Comparing Birth Rate of Male and Female Monthwise

# In[ ]:


fig = go.Figure(data=[
    go.Bar(name='Birth Rate Male', x= df4['Month'], y= df4['BirthRateF']),
    go.Bar(name='Birth Rate Female', x= df4['Month'], y=df4['BirthRateM'])
])
# Change the bar mode
fig.update_layout(barmode='group',
                 xaxis={'categoryorder':'array', 
                        'categoryarray':['January','February',
                                         'March','April', 'May', 
                                         'June', 'July', 'August', 
                                         'September', 'October', 'November', 'December',]},
                 paper_bgcolor='rgb(233,233,233)',
                 title_text= ' Birth Rate of Male & Female Monthwise',)
fig.show()


# **Inference**
# 
# * Comparatively Female birth rate was less than Male Birth rate across the years
# 
# * Male Birth rate was highest in month of July and was lowest in month of February
# 
# * Female Birth rate was highest in month of July  and was lowest in month of May
# 
# 

# Comparing Birth Rate and Death Rate Yearwise

# In[ ]:


fig = go.Figure(data=[
    go.Bar(name='Birth Rate', x= df3['Year'], y= df3['BirthRate']),
    go.Bar(name='Death Rate', x= df3['Year'], y=df3['DeathRate'])
])
# Change the bar mode
fig.update_layout(barmode='group',
                 xaxis = dict(
        tickmode = 'array',
        tickvals = [2010, 2011, 2012, 2013, 2014, 
                    2015, 2016, 2017 , 2018, 2019, 2020],
        ticktext = ['2010', '2011', '2012', '2013', 
                    '2014', '2015', '2016', '2017' , 
                    '2018', '2019', '2020'],),
        paper_bgcolor='rgb(233,233,233)',
        title_text= ' Birth Rate & Death Rate Yearwise',)
fig.show()


# **Inference**
# 
# * Birth rate seems to be always greater than Death rate over the years except for year 2010 & 2020
# 
# * Birth rate was highest in 2016 and was lowest in 2011
# 
# * Death rate was highest in 2010 and was lowest in 2010

# Comparing Death Rate and Birth Rate Month Wise

# In[ ]:


fig = go.Figure(data=[
    go.Bar(name='Birth Rate', x= df4['Month'], y= df4['BirthRate']),
    go.Bar(name='Death Rate', x= df4['Month'], y=df4['DeathRate'])
])
# Change the bar mode
fig.update_layout(barmode='group',
                  xaxis={'categoryorder':'array', 
                         'categoryarray':['January','February','March',
                                          'April', 'May', 'June', 'July', 
                                          'August', 'September', 'October',
                                          'November', 'December',]},
                 paper_bgcolor='rgb(233,233,233)',
                 title_text= ' Birth Rate & Death Rate Month wise',)
fig.show()


# **Inference**
# 
# * Except the month of January over the years death rate was always less than birth rate.
# 
# * Birth rate is highest in month of July and lowest in month of February.
# 
# * Death rate is highest in month of January   and lowest in month of June.

# # **Marriage and Divorce in Moscow**

# **Marriages over the years**

# In[ ]:


dfm = df.groupby(['Year'])['StateRegistrationOfMarriage'].sum().reset_index()
#dfm = dfm.loc[dfm['Year'] != 2020]
sns.set_context("talk")
plt.style.use("fivethirtyeight")
plt.figure(figsize = (14,10))

ax = sns.barplot(x = 'Year', y = 'StateRegistrationOfMarriage' , data = dfm, ci = None, palette = 'bright')
plt.title("Marriages in Moscow over the years")
plt.ylabel('Marriages')
for p in ax.patches:
             ax.annotate("%.f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),
                 ha='center', va='center', fontsize=15, color='black', xytext=(0, 8),
                 textcoords='offset points')


# **Inference**
# 
# * 2014 had the highest number of marriages (100483) whereas 2019 had the lowest number of marriages (83010)

# In[ ]:


fig = go.Figure(data=go.Heatmap(
                   z= df['StateRegistrationOfMarriage'],
                   x=df['Month'],
                   y= df['Year'],
                   hoverongaps = False))
fig.update_layout(
    title_text= ' Marriages in  Moscow',
    yaxis = dict(
        tickmode = 'array',
        tickvals = [2010, 2011, 2012, 2013, 2014, 
                    2015, 2016, 2017 , 2018, 2019, 2020],
        ticktext = ['2010', '2011', '2012', '2013', 
                    '2014', '2015', '2016', '2017' , 
                    '2018', '2019', '2020'],),
    paper_bgcolor='rgb(233,233,233)',
    
    )
fig.show()


# **Inference**
# 
# * Over the years the months from June to September has more number of marriages.
# 
# * In 2013 and 2014 , August had the most number of marriages.

# **Divorces over the years**

# In[ ]:


dfd = df.groupby(['Year'])['StateRegistrationOfDivorce'].sum().reset_index()
#dfd = dfd.loc[dfd['Year'] != 2020]
sns.set_context("talk")
plt.style.use("fivethirtyeight")
plt.figure(figsize = (14,10))

ax = sns.barplot(x = 'Year', y = 'StateRegistrationOfDivorce', data = dfd, ci=None, palette = 'bright')
plt.title("Divorces in Moscow over the years")
plt.ylabel('Divorces')
for p in ax.patches:
             ax.annotate("%.f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),
                 ha='center', va='center', fontsize=15, color='black', xytext=(0, 8),
                 textcoords='offset points')


# **Inference**
# * 2019 had the highest number of Divorces (47995) and 2012 had the lowest number of Divorces (41928)

# In[ ]:


fig = go.Figure(data=go.Heatmap(
                   z= df['StateRegistrationOfDivorce'],
                   x=df['Month'],
                   y= df['Year'],
                   hoverongaps = False))
fig.update_layout(
    title_text= ' Divorces in Moscow',
    yaxis = dict(
        tickmode = 'array',
        tickvals = [2010, 2011, 2012, 2013, 2014, 
                    2015, 2016, 2017 , 2018, 2019, 2020],
        ticktext = ['2010', '2011', '2012', '2013', 
                    '2014', '2015', '2016', '2017' , 
                    '2018', '2019', '2020'],),
    paper_bgcolor='rgb(233,233,233)',
    
    )
fig.show()


# **Inference**
# 
# * Over the years March has the most number of Divorces. after that December has the most number of Divorces

# **Adoption and Name Change **

# **Adoptions over the Years**

# In[ ]:


df6 = df.groupby(['Year'])['StateRegistrationOfAdoption'].sum().reset_index()
df6 = df6.loc[df6['Year'] != 2020]
sns.set_context("talk")
plt.style.use("fivethirtyeight")
plt.figure(figsize = (14,10))
ax = sns.barplot(x = 'Year' , y = 'StateRegistrationOfAdoption', data = df6, ci = None, palette = 'dark')
plt.title('Adoptions in Moscow over the years')
plt.ylabel('Adoptions')

for p in ax.patches:
             ax.annotate("%.f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),
                 ha='center', va='center', fontsize=15, color='black', xytext=(0, 8),
                 textcoords='offset points')


# **Inference**
# 
# * Over the years number of adoptions are gradually decreasing 

# **Name Change over the years**

# In[ ]:


df7 = df.groupby(['Year'])['StateRegistrationOfNameChange'].sum().reset_index()
df7 = df7.loc[df7['Year'] != 2020]
sns.set_context("talk")
plt.style.use("fivethirtyeight")
plt.figure(figsize = (14,10))
ax = sns.barplot(x = 'Year' , y = 'StateRegistrationOfNameChange', data = df7, ci = None, palette = 'dark')
plt.title('Name Changes in Moscow over the years')
plt.ylabel('Name Changes')
for p in ax.patches:
             ax.annotate( "%.f" %p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),
                 ha='center', va='center', fontsize=15, color='black', xytext=(0, 8),
                 textcoords='offset points')


# **Inference**
# * Over the years the number of Name Changes are gradually Increasing.

# In[ ]:


df8 = df.groupby(['Year'])['StateRegistrationOfPaternityExamination'].sum().reset_index()
df8 = df8.loc[df8['Year'] != 2020]
sns.set_context("talk")
plt.style.use("fivethirtyeight")
plt.figure(figsize = (14,10))
ax = sns.barplot(x = 'Year' , y = 'StateRegistrationOfPaternityExamination', data = df8, ci = None, palette = 'dark')
plt.title('Paternity Examinations in Moscow over the years')
plt.ylabel('No. of parenity examination')
for p in ax.patches:
             ax.annotate( "%.f" %p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),
                 ha='center', va='center', fontsize=15, color='black', xytext=(0, 8),
                 textcoords='offset points')


# # Conclusion
# 
# * In Moscow 2016 has the most number of Births.
# 
# * Month of July has the most of Number of Births.
# 
# * In Moscow 2010 has the most number of Deaths.
# 
# * Month of January has the most number of Deaths.
# 
# * In Moscow most of the most couple marry in months of June to September.
# 
# * In Moscow couple generally divorce in Month of March and December
# 
# * In Moscow Adoptions are decreasing whereas Name changes are Increasing over the years. Further studies need to be done to find out the reason.
