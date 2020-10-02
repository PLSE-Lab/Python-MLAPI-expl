#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd 
import matplotlib.pyplot as plt
from fbprophet import Prophet
from fbprophet.plot import plot_plotly, add_changepoints_to_plot
import plotly.offline as py


# In[ ]:


data=pd.read_csv("../input/covid19-in-italy/covid19_italy_region.csv")

data_groupby_date1 = data.groupby("Date")[['TotalPositiveCases', 'Deaths', 'Recovered','TestsPerformed','HospitalizedPatients','TotalHospitalizedPatients']].sum().reset_index()
dgd1 = data_groupby_date1

pr_data_cm = dgd1.loc[:,['Date','TotalPositiveCases']]
pr_data_cm.columns = ['ds','y']

m=Prophet()
m.fit(pr_data_cm)
future=m.make_future_dataframe(periods=365)
forecast_cm=m.predict(future)

cnfrm = forecast_cm.loc[:,['ds','trend']]
cnfrm = cnfrm[cnfrm['trend']>0]
cnfrm.head()
cnfrm=cnfrm.head(42)
cnfrm=cnfrm.tail(30)
cnfrm.columns = ['Date','Confirm']

fig_cm = plot_plotly(m, forecast_cm)
py.iplot(fig_cm) 

fig_cm = m.plot(forecast_cm,xlabel='Date',ylabel='Confirmed Count')

