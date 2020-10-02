#!/usr/bin/env python
# coding: utf-8

# # Relation between growth rate, $R_0(t)$, containment and mitigation mesures.
# 
# In this notebook, I show how there is a correlation between mitigation mesures, containment and growth rates,$R_0(t)$.
# To do this, we will observe the strategy of each country(China, US, Germany,South Korea, italy, France...) to fight against covid19 and the impact of this strategy on the growth rate and $R_0(t)$. We will need three growth rates also $R_0(t)$.
# 
# > **growth rate positive cases**
# 
# > **growth rate recovered**
# 
# > **growth rate death**
# 
# > **relation between growth rate current positive case with** $R_0(t)$
# 
# > **Adequate contact rate and incidence, SIRF Model with standard incidence adapted, Practical: SIRF approximated,  Estimate $\beta(t), \gamma(t), \delta(t)$, Behaviour of Covid 19 disease**
# 
# And the end of this work, I give some conclusion. Let's start.

# # SARS Cov 2 transmission rate ($\beta$), recovered rate ($\gamma$) and fatalities rate ($\delta$)
# 
# ## Adequate contact rate and incidence
# 
# **Contact rate $U(N)$** is the number of individuals contacted by infective per unit of time. Suppose that the probability of infection by each contact is $\beta_0$, then the **adequate contact rate** is $\beta_0U(N)$. 
# 
# The mean adequate contact rate of an infected individual to a susceptible is $\beta_0U(N)\dfrac{S}{N}$. This rate is called an **infection rate**. Then the total new infectives infected by all individuals in the infected compartiment per unit of time, at time t is $(\beta_0U(N)\dfrac{S}{N})I$, which is called **incidence** of disease.
# 
# - If $U(N) = kN$ that is, the contact rate is proportional to the total population size, the incidence is $\beta(t)S(t)I(t)$, where $\beta = \beta_0k$ is called the transmission coefficient(transmission rate). This type of incidence is called **bilinear incidence**
# - If $U(N) = k^{'}$, that is, the contact rate is a constant in this case, the incidence become $\beta I\dfrac{S}{N}$, where $\beta = \beta_0k^{'}$, and it is called **standard incidence**.
# 
# **Extract from: Zhien Ma, Jia Li - Dynamical Modeling and Anaylsis of Epidemics-World Scientific Publishing Company (2009)**

# ### SIRF Model with standard incidence  adapted
# 
# **Can we find the model that explain well the spreading of covid 19 in the world?**
# 
# We know that covid19 have many importants variables but our data, we have four  **ConfirmedCases(TotalpositiveCases), CurrentConfirmedCases(CurrentpositiveCases), Recovered and Deaths**. How can we obtain the dynamics system equation for these variables? To answer this question, we are going to use the SIRF Model with standard incidence:
# 
# The SIRF model with standard incidence  is a classic model in epidemiology, it contain 04 subpopulations, the susceptibles **S**, the infectives **I** and recovered individuals **R**, fatalities **F**:
# 
# > Susceptiles 
# 
# > Infective
# 
# > Recovered
# 
# > Fatalities
# 
# The susceptible can become infective, and the infectives can become recovered or Fatalities, but no other transitions are considered.
# The population $N = S + I + R + F$ remains constant. The model describes the movement between the classes by the system of differential equations.
# 
# > $\dfrac{dS}{dt} = -\beta I\dfrac{S}{N}$, $\qquad$ $\dfrac{dI}{dt} = \beta I\dfrac{S}{N} -(\gamma +\delta) I$, $\qquad$ $\dfrac{dR}{dt} = \gamma I$ $\qquad$ $\dfrac{dF}{dt} = \delta I$.  Where  $\beta$ is the transmission rate, $\gamma$ is the recovery rate, $\delta$ is fatalities rate and $R_{0}=\dfrac{\beta }{\gamma+\delta}$

# ### Practical:  SIRF approximated
# 
# In the context of sars cov 2 in the world, we need to adapt SIRF model to our data such that we can make some approximation on behavior of disease and define transmission rate and others. If we consider **(N)**  the number of population in some fixed surface ($Km^{2}$) at time t. We know that there will exist some confirmed cases population and non confirmed cases population.
# 
# **population size = totalpositivecases + totalnegativecases** and **totalpositivecases = currentpositivecases + (recovered + death)**
# 
# hence,
# 
# **population size = totalnegativecases + currentpositivecases + recovered + death**  (1)
# 
# From (1) we can make some identification:
# 
# > population size can be a total Population (N).
# 
# > totalnegativecases can be a Susceptible (S)
# 
# > currentpositivecases can be an Infective (I) 
# 
# > recovered + death can be a Recovered individuals (R) + Fatalities (F)
# 
# We can write again:
# 
# $S = N  - S_c \rightarrow \dfrac{S}{N} = 1 - \dfrac{S_c}{N}$ if $  \dfrac{S_c}{N} << 1 $ we have $S \approx N$ and SIRF Model with standard  incidence become:
# 
# $\dfrac{dI}{dt} = (\beta - \gamma - \delta)I$, $\qquad$ $\dfrac{dR}{dt} = \gamma I$ $\qquad$ $\dfrac{dF}{dt} = \delta I$

# ### Estimate $\beta(t), \gamma(t), \delta(t)$
# 
# > $\beta(t) = \dfrac{the \:  number \: of \:  daily \:  currentConfirmed \:  covid19 \:  patients \:  at \:  time \:  t}{the \:  number \:  of \:  accummulated \:  confirmed \:  covid19 \:  patients \:  at \:  time \:  t}$
# 
# > $\gamma(t) = \dfrac{the \:  number \: of \:  daily \:  recovered \:  covid19 \:  patients \:  at \:  time \:  t}{the \:  number \:  of \:  accummulated \:  confirmed \:  covid19 \:  patients \:  at \:  time \:  t}$
# 
# > $\delta(t) = \dfrac{the \:  number \: of \:  daily \:  deaths \:  covid19 \:  patients \:  at \:  time \:  t}{the \:  number \:  of \:  accummulated \:  confirmed \:  covid19 \:  patients \:  at \:  time \:  t}$
# 
# **Source: Zhien Ma, Jia Li - Dynamical Modeling and Anaylsis of Epidemics-World Scientific Publishing Company (2009)**

# ## Behaviour of Covid 19 disease (relation between growth rate current positive case with $R_0(t)) 
# 
# The behaviour of disease depends on the state of ratio reproductive number $R_0(t)$ over time. Disease have three behaviours following the state of $R_0(t)=\dfrac{\beta(t) }{\gamma(t)+\delta(t)}$.
# 
# > If $\dfrac{dI}{Idt} > 0$ then $R_{0}(t) > 1 $ **the disease outbreaks again**.
# 
# > If $\dfrac{dI}{Idt} \approx 0$ then $R_{0}(t) \approx 1 $, also $ \beta(t) \approx \gamma(t)+\delta(t)$ **the disease remains constant over time, it is the buffer state of the disease**.
# 
# > If $\dfrac{dI}{Idt} < 0$  then $R_{0}(t) < 1 $ **the disease die out**.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# # Prepare data

# In[ ]:


pd.set_option('display.max_rows', 10000)
pd.set_option('display.max_colwidth', 150)
media = pd.read_csv('/kaggle/input/covid19-containment-and-mitigation-measures/COVID 19 Containment measures data.csv')
covid19 = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv')


# In[ ]:


media.head(2)


# In[ ]:


policy = media[['Country','Date Start','Description of measure implemented']].sort_values('Date Start',ascending=True)


# In[ ]:


policy.head(3)


# In[ ]:


# we see different country
policy.Country.unique()


# In[ ]:


covid19.head(3)


# In[ ]:


covid19 = covid19.groupby(['Country/Region', 'ObservationDate'])[['Confirmed', 'Deaths', 'Recovered']].agg('sum')
covid19['CurrentConfirmed'] = covid19['Confirmed'] - covid19['Recovered'] - covid19['Deaths']
covid19.head()


# In[ ]:


country = covid19.reset_index()
country.info()


# In[ ]:


country.head(3)


# In[ ]:


country.loc[:, 'ObservationDate'] = pd.to_datetime(country.loc[:, 'ObservationDate'])
country.loc[:, 'Confirmed'] = pd.to_numeric(country.loc[:, 'Confirmed'], errors='coerce')
country.loc[:,'Recovered'] = pd.to_numeric(country.loc[:,'Recovered'], errors='coerce')
country.loc[:,'Deaths'] = pd.to_numeric(country.loc[:,'Deaths'], errors='coerce')
policy.loc[:, 'Date Start'] = pd.to_datetime(policy.loc[:, 'Date Start'])


# In[ ]:


country['Country/Region'].unique()


# In[ ]:


def growth_rate(data=None):
    x = []
    x.append(0)
    for i in range(data.shape[0]-1):
        a = data.iloc[i+1]-data.iloc[i]
        if data.iloc[i] == 0:
            v = 0.0
        else:
            v = a/data.iloc[i]
        #v=v*100
        x.append(v)
        
    return np.array(x)


# In[ ]:


def compute_growth_rate(data=None):
    """
        :params data
        
    """
    for c in ['Confirmed', 'Recovered', 'Deaths','CurrentConfirmed']:
        r = 'growth_rate_{}'.format(c)
        data.loc[:,r] = growth_rate(data.loc[:,c])
        
    return data.copy()


# # China containment and mitigation mesures

# In[ ]:


china = policy[policy.Country == 'China'].sort_values(by=['Date Start'])
mainland_china = country[country['Country/Region'] == 'Mainland China']


# In[ ]:


# to see date that policy has started at China
china['Date Start'].unique()


# ## before

# In[ ]:


china[china['Date Start'] < '2020-02-22'].style.set_properties(**{'background-color': 'black',
                            'color': 'white',
                            'border-color': 'lawngreen'})


# In[ ]:


gr_china = compute_growth_rate(mainland_china)


# In[ ]:


cols = list(set(gr_china.columns) - set(['Confirmed', 'Deaths', 'Recovered','Country/Region','CurrentConfirmed',
                                        'growth_rate_CurrentConfirmed']))
icols = ['ObservationDate', 'Confirmed', 'Recovered', 'Deaths']


# ## After 

# In[ ]:


china[china['Date Start'] >= '2020-02-22'].style.set_properties(**{'background-color': 'black',
                            'color': 'lawngreen',
                            'border-color':'white'})


# ### Growth rate

# In[ ]:


mainland_china[cols].plot(x='ObservationDate', figsize=(15,5))
plt.title('The effect of containment and mitigation mesures on growth rate confirmed, recovered,deaths in China ')
plt.ylabel('growth rate')


# In[ ]:


mainland_china.plot(x='ObservationDate', y = 'growth_rate_CurrentConfirmed', figsize=(15,5))
plt.hlines(0, mainland_china.ObservationDate.min(), mainland_china.ObservationDate.max(), 
           linestyles='dashdot', colors='black',
              label='limit (relation between growth rate and R0)')
plt.title('The effect of containment and mitigation mesures on growth rate current confirmed in China')
plt.legend(loc='best')
plt.ylabel('growth rate')


# In[ ]:


mainland_china[['ObservationDate','Confirmed','Recovered', 'Deaths']].plot(x='ObservationDate', figsize=(15,5))
plt.title('Control disease state in China')


# # Italy Containment and Mitigation mesures

# In[ ]:


italy = policy[policy['Country'] == 'Italy'].sort_values(by=['Date Start'])
c_italy = country[country['Country/Region'] == 'Italy']


# ## Before

# In[ ]:


italy[italy['Date Start'] < '2020-02-24'].style.set_properties(**{'background-color': 'black',
                            'color': 'white',
                            'border-color': 'lawngreen'})


# In[ ]:


gr_italo = compute_growth_rate(c_italy)


# ## After

# In[ ]:


italy[italy['Date Start'] >= '2020-02-24'].style.set_properties(**{'background-color': 'black',
                            'color': 'lawngreen',
                            'border-color': 'white'})


# In[ ]:


gr_italo[cols].plot(x='ObservationDate', figsize=(15,5))
plt.title('The effect of containment and mitigation mesures on growth rate confirmed,Recovered and death in Italy')
plt.ylabel('growth rate')


# In[ ]:


gr_italo.plot(x='ObservationDate', y = 'growth_rate_CurrentConfirmed', figsize=(15,5))
plt.hlines(0, gr_italo.ObservationDate.min(), gr_italo.ObservationDate.max(), 
           linestyles='dashdot', colors='black',
              label='limit (relation between growth rate and R0)')
plt.title('The effect of containment and mitigation mesures on growth rate current confirmed in Italy')
plt.legend(loc='best')
plt.ylabel('growth rate')


# In[ ]:


fig = plt.figure(figsize=(15,5))
fig.subplots_adjust(hspace=0.4, wspace=0.1)
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)
c_italy.plot(x='ObservationDate', y = 'Confirmed', ax=ax1)
ax1.set_title('Increasing confirmed in Italy')
c_italy[['ObservationDate','Deaths', 'Recovered']].plot(x='ObservationDate', ax=ax2)
ax2.set_title('Increasing recovered and deaths in Italy')


# **To continuous with Italy see** https://www.kaggle.com/lumierebatalong/italy-space-time-spreading-of-covid19

# # Germany containment and mitigation mesures

# In[ ]:


german = policy[policy['Country'] == 'Germany'].sort_values(by=['Date Start'])
germany = country[country['Country/Region'] == 'Germany']


# ## Before

# In[ ]:


german[german['Date Start'] <= '2020-02-28'].style.set_properties(**{'background-color': 'black',
                            'color': 'white',
                            'border-color': 'lawngreen'})


# In[ ]:


gr_german = compute_growth_rate(germany)


# # After

# In[ ]:


german[german['Date Start'] > '2020-02-28'].style.set_properties(**{'background-color': 'black',
                            'color': 'lawngreen',
                            'border-color': 'white'})


# In[ ]:


gr_german[cols].plot(x='ObservationDate', figsize=(15,5))
plt.title('The effect of containment and mitigation mesures on growth rate confirmed, recovered,deaths in Germany')
plt.ylabel('growth rate')


# In[ ]:


gr_german.plot(x='ObservationDate', y = 'growth_rate_CurrentConfirmed', figsize=(15,5))
plt.hlines(0, gr_german.ObservationDate.min(), gr_german.ObservationDate.max(), 
           linestyles='dashdot', colors='black',
              label='limit (relation between growth rate and R0)')
plt.title('The effect of containment and mitigation mesures on growth rate current confirmed in Germany')
plt.legend(loc='best')
plt.ylabel('growth rate')


# In[ ]:


fig = plt.figure(figsize=(15,5))
fig.subplots_adjust(hspace=0.4, wspace=0.1)
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)
germany.plot(x='ObservationDate', y = 'Confirmed', ax=ax1)
ax1.set_title('Increasing confirmed in Germany')
germany[['ObservationDate','Deaths', 'Recovered']].plot(x='ObservationDate', ax=ax2)
ax2.set_title('Increasing recovered and deaths in Germany')


# # France 

# In[ ]:


french = policy[policy['Country'] == 'France'].sort_values(by=['Date Start'])
france = country[country['Country/Region'] == 'France']


# In[ ]:


french.style.set_properties(**{'background-color': 'black',
                            'color': 'lawngreen',
                            'border-color': 'white'})


# In[ ]:


gr_france = compute_growth_rate(france)


# In[ ]:


gr_france[cols].plot(x='ObservationDate', figsize=(15,5))
plt.title('The effect of containment and mitigation mesures on growth rate confirmed, recovered,deaths in France')


# In[ ]:


gr_france.plot(x='ObservationDate', y = 'growth_rate_CurrentConfirmed', figsize=(15,5))
plt.hlines(0, gr_france.ObservationDate.min(), gr_france.ObservationDate.max(), 
           linestyles='dashdot', colors='black',
              label='limit (relation between growth rate and R0)')
plt.title('The effect of containment and mitigation mesures on growth rate current confirmed in France')
plt.legend(loc='best')
plt.ylabel('growth rate')


# In[ ]:


fig = plt.figure(figsize=(15,5))
fig.subplots_adjust(hspace=0.4, wspace=0.1)
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)
france.plot(x='ObservationDate', y = 'Confirmed', ax=ax1)
ax1.set_title('Increasing confirmed in France')
france[['ObservationDate','Deaths', 'Recovered']].plot(x='ObservationDate', ax=ax2)
ax2.set_title('Increasing recovered and deaths in France')


# # Iran

# In[ ]:


tehran = policy[policy['Country'] == 'Iran'].sort_values(by=['Date Start'])
iran = country[country['Country/Region'] == 'Iran']


# In[ ]:


tehran.style.set_properties(**{'background-color': 'black',
                            'color': 'lawngreen',
                            'border-color': 'white'})


# In[ ]:


gr_iran = compute_growth_rate(iran)


# In[ ]:


gr_iran[cols].plot(x='ObservationDate',figsize=(15,5))
plt.title('The effect of containment and mitigation mesures on growth rate confirmed, recovered, death in Iran')
plt.ylabel('growth rate ')


# In[ ]:


gr_iran.plot(x='ObservationDate', y = 'growth_rate_CurrentConfirmed', figsize=(15,5))
plt.hlines(0, gr_iran.ObservationDate.min(), gr_iran.ObservationDate.max(), 
           linestyles='dashdot', colors='black',
              label='limit (relation between growth rate and R0)')
plt.title('The effect of containment and mitigation mesures on growth rate current confirmed in Germany')
plt.legend(loc='best')
plt.ylabel('growth rate')


# In[ ]:


iran[icols].plot(x='ObservationDate', figsize=(15,5))
plt.title('The fight against disease controlled by Iran')


# # Egypt containment and mitigation mesures

# In[ ]:


cairo = policy[policy.Country == 'Egypt']
egypt = country[country['Country/Region'] == 'Egypt']


# In[ ]:


cairo.style.set_properties(**{'background-color': 'black',
                            'color': 'lawngreen',
                            'border-color': 'white'})


# In[ ]:


gr_egypt = compute_growth_rate(egypt)


# In[ ]:


gr_egypt[cols].plot(x='ObservationDate', figsize=(15,5))
plt.title('The effect of containment and mitigation mesure on growth rate confirmed,recovered and deaths in Egypt')
plt.ylabel('growth rate ')


# In[ ]:


gr_egypt.plot(x='ObservationDate', y = 'growth_rate_CurrentConfirmed', figsize=(15,5))
plt.hlines(0, gr_egypt.ObservationDate.min(), gr_egypt.ObservationDate.max(), 
           linestyles='dashdot', colors='black',
              label='limit (relation between growth rate and R0)')
plt.title('The effect of containment and mitigation mesures on growth rate current confirmed in Egypt')
plt.legend(loc='best')
plt.ylabel('growth rate')


# In[ ]:


egypt[icols].plot(x='ObservationDate', figsize=(15,5))
plt.title('The fight against disease controlled by Egypt')


# **You can see the relationship between growth rate and containment, mitigation measures very well.
# we notice that this mitigation measures can be efficient if government and population fight together against this disease. (e.g. China, ..)**

# ## UpNext
