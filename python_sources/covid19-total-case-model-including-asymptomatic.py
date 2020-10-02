#!/usr/bin/env python
# coding: utf-8

# # COVID-19 _ A model for correation between confirmed number and mitigation
# 
# Here, I would propose a simple model for understanding the correation between the total confirmed number and types of mitigation.
# 
# In this model, we classify infected people into three categories: confirmed (Na), symptomatic (Nb), and asymptomatic (Nc).
# For a newly infected person, a certain percentage will be detected, and a certain percentage will not be detected immediately. Also, we assume that 25% is asymptomatic but has infectivity in the first seven days. 
# 
# With daily changes, Na will no longer infect other people (diagnosed patients are properly isolated), and Nb and Nc will infect others with a percentage. In addition, Nb may show symptoms and be detected. Nc will not be detected but it will recover on its own and lose its infection.
# 
# There are two methods that can be taken. The first is to shut down (reduce people's contact) and the second is to increase the test rate.
# 
# 
# **In this model.**
# 
# **When we use 'shut down,' the infection rate will change from a constant to an exponential decline.**
# 
# **When we increase the test rate, we increase the proportion of infected people on the confirmed list.**
# 
# 
# 

# # **First, I used this model to simulate real cases (South Korea, Italy, Germany, Spain, France, and Japan).**
# 
# For the reliability of the model, the parameters need to coincide with "Confirmed cases" and "R7"
# 
# ,where R7 = (confirmed case at day 7-confirmed case at day 0) / confirmed case at day 0
# 
# 
# The result looks like a good fit.
# 
# Since I am not an expert in infectious diseases, I will not interpret the parameters too much here.

# In[ ]:



import numpy as np 
import pandas as pd
from sklearn import metrics
import lightgbm as lgb
from xgboost import XGBRegressor

train = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv')

train_con = train.groupby(['Country/Region']).sum()

daily = train_con.iloc[:,2:].copy()

daily = daily.sort_values(by=daily.columns[-1:].tolist(), ascending=False)

daily = daily[daily>400]

daily10 = daily.head(30)
daily10 = daily10.transpose()

s = 7
case0 = daily10[:-s]
case7 = daily10[s:]

R7 = (case7.values-case0)/case0.values

R7_x = R7.reset_index().drop(['index'],axis=1)

for col in R7_x:
    x=R7[col].values
    x = x[~np.isnan(x)]
    R7_x[col]=np.nan
    R7_x[col][:len(x)]=x
    

daily10_x = daily10.reset_index().drop(['index'],axis=1)

for col in daily10_x:
    x=daily10[col].values
    x = x[~np.isnan(x)]
    daily10_x[col]=np.nan
    daily10_x[col][:len(x)]=x


daily10 = daily10_x.copy()
R7 = R7_x.copy() 


def com_model(N0,ra,rc,af0,aft,sep,na0):

    rb=1-ra-rc
    
    Na = [na0]
    Nb = [rb*N0]
    Nc = [rc*N0]
    
    day = 100
    for i in range(0,day):
        if i >= sep:
            af = af0*np.exp(-(i-sep)/aft)
        else:
            af = af0
        Na1 = Na[i]+ra*Nb[i]*(1+af)+ra*af*Nc[i]
        Nb1 = Nb[i]*(1+af)*rb+rb*af*Nc[i]
        Nc1 = Nb[i]*(1+af)*rc+Nc[i]*6/7+rc*af*Nc[i]
        
        Na.append(Na1)
        Nb.append(Nb1)
        Nc.append(Nc1)
    
    Na = np.array(Na)
    Nb = np.array(Nb)
    Nc = np.array(Nc)
    
    total_NO = pd.DataFrame()
    
    total_NO['Na'] = Na
    total_NO['Nb'] = Nb
    total_NO['Nc'] = Nc
    
    s = 7
    case0 = total_NO[:-s]
    case7 = total_NO[s:]
    
    R7_model = (case7.values-case0)/case0.values
    return total_NO, R7_model




aaf1 = [1000.0, 0.18073303474021774, 0.6280982274084522, 5.71892473354724]
daily10_model, R7_model = com_model(aaf1[0],aaf1[1],0.25,aaf1[2],aaf1[3],3.5,433)



aaf2 = [1370.4678055543616, 0.1838035946577493, 0.478029454040578, 27.6993430169006]
daily10_model_2, R7_model_2 = com_model(aaf2[0],aaf2[1],0.25,aaf2[2],aaf2[3],8,453)

#Germany
aaf3 = [865.2796397013346, 0.11261479700447499, 0.5390848997603882, 4.913298085389435]
daily10_model_3, R7_model_3 = com_model(aaf3[0],aaf3[1],0.25,aaf3[2],aaf3[3],10.5,482)


#Spain
aaf5 = [1339.8537548059298, 0.19098358950663832, 0.7647970998427716, 10.865896088753793]
daily10_model_5, R7_model_5 = com_model(aaf5[0],aaf5[1],0.25,aaf5[2],aaf5[3],2.5,500)

#France
aaf6 = [1805.3490537183832, 0.11822501182971824, 0.4469633683947488, 19.194623547613222]
daily10_model_6, R7_model_6 = com_model(aaf6[0],aaf6[1],0.25,aaf6[2],aaf6[3],5,656)

#Japan
aaf7 = [2429.4807472136063, 0.03302756790866209, 0.1195843026300669, 8580.06536922646]
daily10_model_7, R7_model_7 = com_model(aaf7[0],aaf7[1],0.25,aaf7[2],aaf7[3],1.5,420)
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

#plt.figure(0)


# In[ ]:


daily10[['Korea, South','Italy','Germany']][:30].plot(style='o')
daily10_model['Na'][:30].plot(style='-',label='Korea_simulation')
daily10_model_2['Na'][:30].plot(style='-',label='Italy_simulation')
daily10_model_3['Na'][:30].plot(style='-',label='Germany_simulation')

plt.xlabel('Time (Days)')
plt.ylabel('Confirmed cases')
plt.legend()
#plt.yscale('log')
plt.show()

daily10[['Spain','France','Japan']][:30].plot(style='o')
daily10_model_5['Na'][:30].plot(style='-',label='Spian_simulation')
daily10_model_6['Na'][:30].plot(style='-',label='France_simulation')
daily10_model_7['Na'][:30].plot(style='-',label='Japan_simulation')
plt.xlabel('Time (Days)')
plt.ylabel('Confirmed cases')
plt.legend()
#plt.yscale('log')
plt.show()


# In[ ]:


R7[['Korea, South','Italy','Germany']][:30].plot(style='o')
R7_model['Na'][:30].plot(style='-',label='Korea_simulation')
R7_model_2['Na'][:30].plot(style='-',label='Italy_simulation')
R7_model_3['Na'][:30].plot(style='-',label='Germany_simulation')
plt.xlabel('Time (Days)')
plt.ylabel('R7')
plt.legend()
#plt.yscale('log')
plt.show()


R7[['Spain','France','Japan']][:30].plot(style='o')
R7_model_5['Na'][:30].plot(style='-',label='Spian_simulation')
R7_model_6['Na'][:30].plot(style='-',label='France_simulation')
R7_model_7['Na'][:30].plot(style='-',label='Japan_simulation')
plt.xlabel('Time (Days)')
plt.ylabel('R7')
plt.legend()
#plt.yscale('log')
plt.show()


# # **Next, I want to know the impact of various strategies on confirmed cases.**
# 
# Let's take the average of Korean and Italian parameters to make a demo model.
# 
# i.e. the initial case = 1500, test rate = 0.14, initial groth rate = 0.5, decay constant = 20, 
#      and take 'shut down' at day 7
# 
# # Demo

# In[ ]:



def com_model(N0,ra,rc,af0,aft,sep,na0,rca):

    rb=1-ra-rc
    
    Na = [na0]
    Nb = [rb*N0]
    Nc = [rc*N0]
    
    day = 60
    for i in range(0,day):
        if i >= sep:
            af = af0*np.exp(-(i-sep)/aft)
        else:
            af = af0
        Na1 = Na[i]+ra*Nb[i]*(1+af)+ra*af*Nc[i]+ra*Nc[i]*(1+af)*rca/2
        Nb1 = Nb[i]*(1+af)*rb+rb*af*Nc[i]
        Nc1 = Nb[i]*(1+af)*rc+Nc[i]*6/7+rc*af*Nc[i]-ra*Nc[i]*(1+af)*rca/2
        
        Na.append(Na1)
        Nb.append(Nb1)
        Nc.append(Nc1)
    
    Na = np.array(Na)
    Nb = np.array(Nb)
    Nc = np.array(Nc)
    
    total_NO = pd.DataFrame()
    
    total_NO['Na'] = Na
    total_NO['Nb'] = Nb
    total_NO['Nc'] = Nc
    
    s = 7
    case0 = total_NO[:-s]
    case7 = total_NO[s:]
    
    R7_model = (case7.values-case0)/case0.values
    return total_NO, R7_model[['Na']]

daily10_model, R7_model = com_model(1500,0.14,0.25,0.5,20,7,0,0)

def take_data(daily_data):
    daily_array_10 = [daily_data[10],daily_data[20],daily_data[30],daily_data[40],daily_data[50],daily_data[60]]
    return np.array(daily_array_10).astype(int)

aa = take_data(daily10_model['Na'].values)
final = pd.DataFrame()
final['Demo']=aa


# In[ ]:



daily10_model.plot(style='-')
plt.title('Demo: confirmed (Na), symptomatic (Nb), and asymptomatic (Nc)')
plt.xlabel('Time (Days)')
plt.ylabel('Confirmed cases')
#plt.yscale('log')
plt.show()


# # Early shun down (at day 0)

# In[ ]:


daily10_model, R7_model = com_model(1500,0.14,0.25,0.5,20,0,0,0)
daily10_model.plot(style='-')
plt.title('Early shun down (at day 0)')
plt.xlabel('Time (Days)')
plt.ylabel('Confirmed cases')
#plt.yscale('log')
plt.show()

aa = take_data(daily10_model['Na'].values)
final['Early shun down']=aa


# # 50% increase in test rate

# In[ ]:


daily10_model, R7_model = com_model(1500,0.21,0.25,0.5,20,7,0,0)
daily10_model.plot(style='-')
plt.title('50% increase in test rate')
plt.xlabel('Time (Days)')
plt.ylabel('Confirmed cases')
#plt.yscale('log')
plt.show()

aa = take_data(daily10_model['Na'].values)
final['50% increase in test rate']=aa


# # Try to find asymptomatic
# *Suppose test rate is half of those with symptoms*

# In[ ]:


daily10_model, R7_model = com_model(1500,0.14,0.25,0.5,20,7,0,1)
daily10_model.plot(style='-')
plt.title('Try to find asymptomatic')
plt.xlabel('Time (Days)')
plt.ylabel('Confirmed cases')
#plt.yscale('log')
plt.show()

aa = take_data(daily10_model['Na'].values)
final['Try to find asymptomatic']=aa


# # Strict implementation of shut down
# *decay constant = 5*
# 
# *note: the value for fit with Korea' data is 5.7*

# In[ ]:


daily10_model, R7_model = com_model(1500,0.14,0.25,0.5,5,7,0,0)
daily10_model.plot(style='-')
plt.title('Strict implementation of shut down')
plt.xlabel('Time (Days)')
plt.ylabel('Confirmed cases')
#plt.yscale('log')
plt.show()

aa = take_data(daily10_model['Na'].values)
final['Strict implementation of shut down']=aa


# # All the above strategies

# In[ ]:


daily10_model, R7_model = com_model(1500,0.21,0.25,0.5,5,0,0,1)
daily10_model.plot(style='-')
plt.title('All the above strategies')
plt.xlabel('Time (Days)')
plt.ylabel('Confirmed cases')
#plt.yscale('log')
plt.show()

aa = take_data(daily10_model['Na'].values)
final['All the above strategies']=aa


# # SUMMARY
# 
# The table is the number of confirmed diagnoses for all the above strategies after 10, 20, 30, 40, 50, 60 days.
# 
# I believe that the actual situation must be more complicated, but maybe this model can lead to some useful ideas. Let the disaster end soon.

# In[ ]:


final.index = [10,20,30,40,50,60]
final


# **Ratio**

# In[ ]:



final_ratio = final.copy()
for col in final:
    final_ratio[col] = final_ratio[col]/final['Demo']
final_ratio    


# In[ ]:





# In[ ]:




