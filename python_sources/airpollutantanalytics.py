#!/usr/bin/env python
# coding: utf-8

# # imports

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# # File import

# In[ ]:


data = pd.read_csv("../input/airpollutantsample/AirQuality.csv")
data.head(10)


# In[ ]:


del data['lastupdate']
data.head()


# # Plotting avg,min and max pollution

# In[ ]:


plt.plot(data['Avg'])
plt.xlabel('cities')
plt.ylabel('amount')
plt.title('Average Pollution Data')


# In[ ]:


plt.plot(data['Max'])
plt.xlabel('cities')
plt.ylabel('amount')
plt.title('Maximum Pollution Data')


# In[ ]:


plt.plot(data['Min'])
plt.xlabel('cities')
plt.ylabel('amount')
plt.title('Minimum Pollution Data')


# # Checking which State have most Pollution

# In[ ]:


plt.figure(figsize=(20,10), dpi = 80)
sns.countplot(x='State',data=data)
plt.xlabel('State')
plt.tight_layout()


# ### Conclusion
# #### We got that Delhi is Most Polluted and Gujrat and Jharkhand are least polluted

# # Plotting Min,Max and Avg Pollution (Pollutant wise)

# In[ ]:


data_p1=data[data.Pollutants=='PM2.5']
data_p1[['Max','Avg','Min']].plot()
plt.xlabel('cities')
plt.ylabel('amount')
plt.title('PM2.5')


# In[ ]:


data_p2=data[data.Pollutants=='PM10']
data_p2[['Max','Avg','Min']].plot()
plt.xlabel('cities')
plt.ylabel('amount')
plt.title('PM10')


# In[ ]:


data_p3=data[data.Pollutants=='NO2']
data_p3[['Max','Avg','Min']].plot()
plt.xlabel('cities')
plt.ylabel('amount')
plt.title('NO2')


# In[ ]:


data_p4=data[data.Pollutants=='NH3']
data_p4[['Max','Avg','Min']].plot()
plt.xlabel('cities')
plt.ylabel('amount')
plt.title('NH3')


# In[ ]:


data_p5=data[data.Pollutants=='SO2']
data_p5[['Max','Avg','Min']].plot()
plt.xlabel('cities')
plt.ylabel('amount')
plt.title('SO2')


# In[ ]:


data_p6=data[data.Pollutants=='CO']
data_p6[['Max','Avg','Min']].plot()
plt.xlabel('cities')
plt.ylabel('amount')
plt.title('CO')


# In[ ]:


data_p7=data[data.Pollutants=='OZONE']
data_p7[['Max','Avg','Min']].plot()
plt.xlabel('cities')
plt.ylabel('amount')
plt.title('OZONE')


# # Plotting Min,Max And Avg (State wise)

# In[ ]:


from pandas import DataFrame
df =DataFrame(data.State)
DataFrame.drop_duplicates(df)


# In[ ]:


data_state1=data[data.State=='Andhra_Pradesh']
data_state1[['Min','Avg','Max']].plot()
plt.xlabel('cities')
plt.ylabel('Amount')
plt.title('Andhra_Pradesh')


# In[ ]:


data_state2=data[data.State=='Bihar']
data_state2[['Min','Avg','Max']].plot()
plt.xlabel('cities')
plt.ylabel('Amount')
plt.title('Bihar')


# In[ ]:


data_state3=data[data.State=='Delhi']
data_state3[['Min','Avg','Max']].plot()
plt.xlabel('cities')
plt.ylabel('Amount')
plt.title('Delhi')


# In[ ]:


data_state4=data[data.State=='Gujarat']
data_state4[['Min','Avg','Max']].plot()
plt.xlabel('cities')
plt.ylabel('Amount')
plt.title('Gujarat')


# In[ ]:


data_state5=data[data.State=='Haryana']
data_state5[['Min','Avg','Max']].plot()
plt.xlabel('cities')
plt.ylabel('Amount')
plt.title('Haryana')


# In[ ]:


data_state6=data[data.State=='Jharkhand']
data_state6[['Min','Avg','Max']].plot()
plt.xlabel('cities')
plt.ylabel('Amount')
plt.title('Jharkhand')


# In[ ]:


data_state7=data[data.State=='Karnataka']
data_state7[['Min','Avg','Max']].plot()
plt.xlabel('cities')
plt.ylabel('Amount')
plt.title('Karnataka')


# In[ ]:


data_state8=data[data.State=='Kerala']
data_state8[['Min','Avg','Max']].plot()
plt.xlabel('cities')
plt.ylabel('Amount')
plt.title('Kerala')


# In[ ]:


data_state9=data[data.State=='Madhya Pradesh']
data_state9[['Min','Avg','Max']].plot()
plt.xlabel('cities')
plt.ylabel('Amount')
plt.title('Madhya Pradesh')


# In[ ]:


data_state10=data[data.State=='Maharashtra']
data_state10[['Min','Avg','Max']].plot()
plt.xlabel('cities')
plt.ylabel('Amount')
plt.title('Maharashtra')


# In[ ]:


data_state11=data[data.State=='Odisha']
data_state11[['Min','Avg','Max']].plot()
plt.xlabel('cities')
plt.ylabel('Amount')
plt.title('Odisha')


# In[ ]:


data_state12=data[data.State=='Punjab']
data_state12[['Min','Avg','Max']].plot()
plt.xlabel('cities')
plt.ylabel('Amount')
plt.title('Punjab')


# In[ ]:


data_state13=data[data.State=='Rajasthan']
data_state13[['Min','Avg','Max']].plot()
plt.xlabel('cities')
plt.ylabel('Amount')
plt.title('Rajasthan')


# In[ ]:


data_state14=data[data.State=='TamilNadu']
data_state14[['Min','Avg','Max']].plot()
plt.xlabel('cities')
plt.ylabel('Amount')
plt.title('TamilNadu')


# In[ ]:


data_state15=data[data.State=='Telangana']
data_state15[['Min','Avg','Max']].plot()
plt.xlabel('cities')
plt.ylabel('Amount')
plt.title('Telangana')


# In[ ]:


data_state16=data[data.State=='Uttar_Pradesh']
data_state16[['Min','Avg','Max']].plot()
plt.xlabel('cities')
plt.ylabel('Amount')
plt.title('Utter_Pradesh')


# In[ ]:


data_state17=data[data.State=='West_Bengal']
data_state17[['Min','Avg','Max']].plot()
plt.xlabel('cities')
plt.ylabel('Amount')
plt.title('West_Bengal')


# # Plotting Mean Pollutant Amount

# In[ ]:


data_pollu=data.groupby('Pollutants')
data_pollu.mean()
plt.figure(figsize=(20,10) , dpi=100)
plt.plot(data_pollu.mean())
plt.legend(['Max','Avg','Mean'])
plt.xlabel('Amount')
plt.ylabel('Pollutant')
plt.title('Pollutant Amounts')


# #### Conclusion 
# ##### PM2.5 is Most Spreaded in Air

# # Plotting Mean State Pollution

# In[ ]:


data_states=data.groupby('State')
data_states.mean()


# In[ ]:


plt.figure(figsize=(18,5) , dpi=100)
plt.plot(data_states.mean())
plt.legend(['Max','Avg','Min'])
plt.xlabel('States')
plt.ylabel('Amount')
plt.tight_layout()


# #### Conclusion
# ##### Delhi And UP are Most Polluted

# # Plotting mean  City Pollution

# In[ ]:


data_city=data.groupby('city')
data_city.mean()


# In[ ]:


plt.figure(figsize=(150,50) , dpi=100)
plt.plot(data_city.mean())
plt.xlabel('City')
plt.ylabel('Amount')
plt.show()


# # Geospatial plotting
