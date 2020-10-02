#!/usr/bin/env python
# coding: utf-8

# **Current State**
# 
# The accumulated confirmation count in Indonesia is increasing exponentially after 2nd of March 2020. There are many reason for infection but most cases have similar reasons that they have contacted with foreigner in Indonesia or have visited foreign countries. Indonesia has raised its coronavirus alert to the "Darurat Nasional" until 29 May 2020, as confirmed case numbers keep rising. As the data set says the Java island is more affected, while Jakarta, the capital city of Indonesia is highly infected by the coronavirus.

# Libraries

# In[ ]:


import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.express as px
from datetime import date, timedelta
from sklearn.cluster import KMeans
from fbprophet import Prophet
from fbprophet.plot import plot_plotly, add_changepoints_to_plot
import plotly.offline as py
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.api as sm
from keras.models import Sequential
from keras.layers import LSTM,Dense
from keras.layers import Dropout
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator


# Reading Dataset

# In[ ]:


cases = pd.read_csv("../input/indonesia-coronavirus-cases/cases.csv")
confirmed_acc = pd.read_csv("../input/indonesia-coronavirus-cases/confirmed_acc.csv")
keywordtrend = pd.read_csv("../input/indonesia-coronavirus-cases/keywordtrend.csv")
patient = pd.read_csv("../input/indonesia-coronavirus-cases/patient.csv")


# Looking into patient data

# In[ ]:


patient.head()


# In[ ]:


cases.tail()


# Age distribution of the confirmed

# In[ ]:


plt.figure(figsize=(10,6))
sns.set_style("darkgrid")
plt.title("Age distribution of the confirmed")
sns.kdeplot(data=patient['age'], shade=True)


# State distribution by age and sex

# In[ ]:


sns.set_style("whitegrid")
sns.FacetGrid(patient, hue='gender', size = 10).map(plt.scatter, 'current_state', 'age').add_legend()
plt.title('State by age and gender')
plt.show()


# State distribution by province and origin

# In[ ]:


sns.set_style("darkgrid")
sns.FacetGrid(patient, hue='nationality', size = 10).map(plt.scatter, 'current_state', 'province').add_legend()
plt.title('State by province and nationality')
plt.show()


# Number of Patients in Each Province

# In[ ]:


plt.figure(figsize=(15,5))
plt.title('Number patients in province')
patient.province.value_counts().plot.bar();


# In[ ]:


import plotly.express as px
import plotly.offline as py
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

fig = px.pie( values=patient.groupby(['province']).size().values,names=patient.groupby(['province']).size().index)
fig.update_layout(
    font=dict(
        size=15,
        color="#242323"
    )
    )   
    
py.iplot(fig)


# Number of Patients in Each Hospital

# In[ ]:


plt.figure(figsize=(15,5))
plt.title('Number patients in hospital')
patient.hospital.value_counts().plot.bar();

