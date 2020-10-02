#!/usr/bin/env python
# coding: utf-8

# <h1> Italy Coronavirus trend and forecasting </h1>
# <p> This notebook wants to show Coronavirus trend in Italy; data are retrieved from official site of "Protezione Civile" (Italian Emergency Agency). These data are published every day at 6 PM at this link </p>
# https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-andamento-nazionale/dpc-covid19-ita-andamento-nazionale.csv 
# 
# At the end of this notebook a future projection of some trend is provided using Facebook Prophet library; **please, consider that ML predictions are based on trend retrieved from the past. It means that, if past data show a growing trend, it implies that even in the future you will see similar trend. This kind of analysis will be useful when, due to current restrictions recomended by Italian governemt, trends will begin to decline**. In that case, a correct timeseries forecast of the trend will help to figure out how many days it will require to come back to "normality".
# 
# [National data of last 6 days](#National-data-of-last-6-days)
# 
# Here below forecasting trends
# 
# [Overall forecasting of daily deaths for next 7 days](#Overall-forecasting-of-daily-deaths-for-next-7-days)
# 
# [From 21/3 forecasting of daily deaths for next 7 days](#From-21/3-forecasting-of-daily-deaths-for-next-7-days)
# 
# [Overall forecasting of daily healed for next 7 days](#Overall-forecasting-of-daily-healed-for-next-7-days)
# 
# [Overall forecasting of daily intensive care for next 7 days](#Overall-forecasting-of-daily-intensive-care-for-next-7-days)
# 
# [Overall forecasting of daily increment of new positve cases for next 7 days](#Overall-forecasting-of-daily-increment-of-new-positve-cases-for-next-7-days)
# 
# [From 21/3 forecasting of daily increment of new positve cases for next 7 days](#From-21/3-forecasting-of-daily-increment-of-new-positve-cases-for-next-7-days)
# 
# [Overall forecasting of daily increment of new total positve cases for next 7 days](#Overall-forecasting-of-daily-increment-of-new-total-positve-cases-for-next-7-days)
# 
# [From 21/3 forecasting of daily increment of new total positve cases for next 7 days](#From-21/3-forecasting-of-daily-increment-of-new-total-positve-cases-for-next-7-days)

# In[ ]:


#import basic python libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#read in a dataframe from remote url
git_hub_url="https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-andamento-nazionale/dpc-covid19-ita-andamento-nazionale.csv"
df = pd.read_csv(git_hub_url,  parse_dates=True, index_col='data',sep=',')


# In[ ]:


df.plot(figsize=(18,8))


# In[ ]:


df.drop(columns=["ricoverati_con_sintomi","tamponi"]).plot(figsize=(18,8))


# In[ ]:


#remove not useful column and show last 3 days
df=df.drop(columns=["stato","note_it","note_en"])
df_delta=df.drop(df.columns,axis=1)


# # National data of last 6 days

# In[ ]:


df.tail(6)


# In[ ]:


#let's define a couple of python functions useful for next steps
def delta(dfxx,df1):
  for c in dfxx.columns:
    new_name="delta_"+c
    df1[new_name]=(dfxx[c]-dfxx[c].shift()).fillna(0)
  return df1
def plot(dfx,cols=None,sizex=(18,8)):
    dff=dfx
    if cols:  
      dff=dfx[cols]
    dff.plot(figsize=sizex)


# In[ ]:


plot(df,["deceduti","dimessi_guariti","terapia_intensiva"])


# In[ ]:


plot(df,["totale_positivi","totale_casi"])


# # Day by day analysis #

# In[ ]:


df_delta=delta(df,df_delta)


# ## National data increments of last 6 days

# In[ ]:


df_delta.tail(6)


# In[ ]:


plot(df_delta,["delta_deceduti","delta_dimessi_guariti","delta_terapia_intensiva","delta_totale_positivi","delta_totale_casi"])


# In[ ]:


#df_delta=df_delta.drop(columns=["delta_tamponi"],axis=1)
plot(df_delta,["delta_deceduti","delta_dimessi_guariti","delta_terapia_intensiva"])
#df_delta.plot(figsize=(18,8))


# In[ ]:


plot(df_delta,["delta_totale_positivi","delta_totale_casi"])


# # Timeseries forecasting #
# Let's assume (it's a strong but optimistic assumption) that max values for "deltas" have been reached on 21/03, then we can consider two kinds of prediction:
# 1. **Overall prediction**, we use in our model all data. For sure in this case we will see a still growing trend due to the fact that until 21 data were growing.
# 2. **From delta peak prediction,** we use in our model only data from 21/03. In this case we are excluding growing trend and assuming that (I hope) from 21/3 data will reach the peak point (pay attention the delta peak point is 21/3 but data were still growing) but delta data will tend to zero.  
# 
# Let's prepare data accordingly with Facebook Prophet conventions

# In[ ]:


from fbprophet import Prophet
pdf=df_delta.reset_index()
pdf["ds"]=pdf["data"]


# In[ ]:


#pdf.tail(5)


# ## Let's define a forecast function on daily base,it is able to forecast values for a given column belonging to the dataframe

# In[ ]:


def forecast(ds_forecast,column,days=7):
    ds_forecast["y"]=ds_forecast[column]
    pro_df=ds_forecast[["ds","y"]]
    my_model = Prophet()
    my_model.fit(pro_df)
    future_dates = my_model.make_future_dataframe(periods=days, freq='D')
    forecast = my_model.predict(future_dates)
    return my_model,forecast 


# # Overall forecasting of daily deaths for next 7 days
# Black dots are observed past true values, blue line is forecast for next 7 days

# In[ ]:


model,fc=forecast(pdf,"delta_deceduti")
model.plot(fc,uncertainty=True);


# # From 21/3 forecasting of daily deaths for next 7 days

# In[ ]:


pdf_tmp=pdf[pdf["ds"]>'2020-03-21'].copy()
model,fc=forecast(pdf_tmp,"delta_deceduti")
model.plot(fc,uncertainty=True);


# # Overall forecasting of daily healed for next 7 days
# 

# In[ ]:


model,fc=forecast(pdf,"delta_dimessi_guariti")
model.plot(fc,uncertainty=True);


# # Overall forecasting of daily intensive care for next 7 days

# In[ ]:


model,fc=forecast(pdf,"delta_terapia_intensiva")
model.plot(fc,uncertainty=True);


# # Overall forecasting of daily increment of new positve cases for next 7 days

# In[ ]:


model,fc=forecast(pdf,"delta_totale_positivi")
model.plot(fc,uncertainty=True);


# # From 21/3 forecasting of daily increment of new positve cases for next 7 days

# In[ ]:


model,fc=forecast(pdf_tmp,"delta_totale_positivi")
model.plot(fc,uncertainty=True);


# # Overall forecasting of daily increment of new total positve cases for next 7 days

# In[ ]:


model,fc=forecast(pdf,"delta_totale_casi")
model.plot(fc,uncertainty=True);


# # From 21/3 forecasting of daily increment of new total positve cases for next 7 days

# In[ ]:


model,fc=forecast(pdf_tmp,"delta_totale_casi")
model.plot(fc,uncertainty=True);


# In[ ]:




