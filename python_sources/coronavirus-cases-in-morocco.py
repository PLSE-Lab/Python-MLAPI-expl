#!/usr/bin/env python
# coding: utf-8

# **Coronavirus (COVID-19) Morocco Data , insights and Predictions**
# 
# The coronavirus pandemic of 2019-20 was reported to have spread to Morocco on 2 March 2020, when the first case of COVID-19 was confirmed in Casablanca. It involved a Moroccan expatriate living in Italy and returning from Italy on 27 February 2020.

# In[ ]:


import pandas as pd
import plotly.graph_objects as go

df = pd.read_csv("../input/covid19-morocco/covid19-Morocco.csv")

trace0 = go.Scatter(x = df['Date'], y = df['Confirmed'], name='Infections')
trace1 = go.Scatter(x = df['Date'], y = df['Deaths'], name='Deaths')
trace2 = go.Scatter(x = df['Date'], y = df['Recovered'], name='Deaths')
layout = go.Layout(title= "Timeline of Infections, Deaths and Recovered ")

data = [trace0,trace1, trace2]

fig = go.Figure(data = data,layout=layout )
fig.show()


# In[ ]:


total_confirmed = df['Confirmed'].sum()
total_deaths = df['Deaths'].sum()
total_recovered = df['Recovered'].sum()
active_cases = total_confirmed - total_deaths - total_recovered

labels = ['Active Cases','Recoveries','Deaths']
values = [active_cases, total_recovered, total_deaths]
figure = go.Figure(data=[go.Pie(labels=labels, values=values)])

figure.show()


# In[ ]:


import plotly.graph_objects as go  
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 

#plt.plot(x,y, 'Confirmed', 'Recovered')
gas = pd.read_csv('../input/covid19-morocco/covid19-Morocco.csv')  

#gas = pd.read_csv('covid19-Morocco.csv')  
plt.figure(figsize=(20,20), dpi=80) 
plt.title('Timeline of Infections, Deaths and Recovered ', fontdict={'fontname': 'Comic Sans MS', 'fontsize': 20})

plt.ylabel('Confirmed') 
plt.ylabel('Recovered') 
plt.xlabel('Deaths')  
 
# for Recovered in gas: 
# 	if Recovered !='Date': 
# 	plt.plot(gas.Date, gas[Recovered], marker='.')
# print(gas.Date[::4]) 
x2 = np.arange(0,4.5,0.5)

plt.plot(gas['Date'],gas['Confirmed'],'r--')  
plt.plot(gas['Date'],gas['Recovered']) 
plt.plot(gas['Date'],gas['Deaths'])   

#plt.plot(x2[5:], x2[5:]**2,'r--' )


#print (gas.Date[::1])
plt.legend()  
plt.xticks(gas.Date[::4])
plt.show() 


# In[ ]:





# In[ ]:




