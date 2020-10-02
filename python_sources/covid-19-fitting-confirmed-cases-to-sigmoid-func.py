#!/usr/bin/env python
# coding: utf-8

# # Importing basic libraries

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams, pyplot as plt, style as style
from scipy.optimize import curve_fit


# In[ ]:


path="../input/covid19-global-forecasting-week-1/train.csv"
train=pd.read_csv(path)


# In[ ]:


pd.set_option('display.max_rows',50)
pd.set_option('display.max_columns',None)
pd.set_option('display.width',None)
pd.set_option('display.max_colwidth',-1)


# ### RENAMING COLUMNS

# In[ ]:


train.rename(columns={"ConfirmedCases":"Confirm","Fatalities":"Deaths"},inplace=True)
   


# In[ ]:


train.columns


# # COUNTRY WISE GROUPING

# In[ ]:


country = train.fillna('N/A').groupby(['Country/Region'])['Confirm', 'Deaths'].sum().sort_values(by='Confirm', ascending=False)


# In[ ]:


country['Country']=country.index


# #### TOTAL ACTIVE CASES= CONFIRMED CASES-DEATHS

# In[ ]:


country['Active']=country['Confirm']-country['Deaths']


# In[ ]:


country.head()


# In[ ]:


top_10=country.index[:10]
country_top=country[country['Country'].isin(top_10)]
country_top.head()


# In[ ]:


country_melted = pd.melt(country_top, id_vars=['Country'] , value_vars=['Confirm','Deaths','Active'])


# In[ ]:


country_melted


# ### BAR GRAPH OF 10 COUNTRIES WITH MOST CONFIRMED CASES

# In[ ]:


style.use('ggplot')
rcParams['figure.figsize'] = 15,10
ax = sns.barplot(x = 'Country', hue="variable", y="value", data=country_melted)


# # DATE WISE GROUPING OF DATA

# #### The growth of confirmed cases with increasing no of days.
# I plotted the curve for total number of confirmed cases per increasing days

# In[ ]:


date_wise=train.groupby(["Date"],as_index=False)["Confirm","Deaths"].max()

no_days=len(np.array(date_wise['Date'].unique()))
days=np.arange(1,no_days+1)
#d=sns.barplot(x='Date',y='Confirm',data=date_wise)

confirmed=np.array(date_wise['Confirm'])#converting the total deaths to an numpy array
deaths=np.array(date_wise['Deaths'])
confirmed.reshape(1,no_days)
deaths.reshape(1,no_days)
date_wise=train.groupby(["Date"],as_index=False)["Confirm","Deaths"].max()

no_days=len(np.array(date_wise['Date'].unique()))
days=np.arange(1,no_days+1)
#d=sns.barplot(x='Date',y='Confirm',data=date_wise)
#print(days)
confirmed=np.array(date_wise['Confirm'])
deaths=np.array(date_wise['Deaths'])
confirmed.reshape(1,no_days)
deaths.reshape(1,no_days)


# ### FITTING THE DATA TO SIGMOID CURVE

# In[ ]:


import numpy as np
import pylab
from scipy.optimize import curve_fit


ydata = confirmed#np.array([400, 600, 800, 1000, 1200, 1400, 1600])
xdata = days#np.array([0, 0, 0.13, 0.35, 0.75, 0.89, 0.91])

def sigmoid(x, L ,x0, k, b):
    y = L / (1 + np.exp(-k*(x-x0)))+b
    return (y)

p0 = [max(ydata), np.median(xdata),1,min(ydata)] # this is an mandatory initial guess

popt, pcov = curve_fit(sigmoid, xdata, ydata,p0, method='dogbox')
'''def sigmoid(x, L ,x0, k, b):
    y = L / (1 + np.exp(-k*(x-x0)))+b
    return (y)

p0 = [max(ydata), np.median(xdata),1,min(ydata)] # this is an mandatory initial guess

popt, pcov = curve_fit(sigmoid, xdata, ydata,p0, method='dogbox')


popt, pcov = curve_fit(fsigmoid, xdata, ydata, method='dogbox', bounds=([0., 600.],[0.01, 1200.]))'''
plt.plot(xdata,ydata,'o')

plt.plot(xdata,sigmoid(xdata,*popt))


# In[ ]:


print(popt)


# THE EQUATION THAT FITS THE CURVE IS:  $$Y=\frac{6.81100908e+04}{(1+e^{(-2.31999685e-01(x-1.96462423e+01)-3.08015353e+02)})}$$

# In[ ]:





# In[ ]:




