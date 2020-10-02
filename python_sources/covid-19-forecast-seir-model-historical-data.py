#!/usr/bin/env python
# coding: utf-8

# # COVID-19 forecast: SEIR model + historical data
# In this notebook I investigate the spread of COVID-19 by fitting a SEIR model the historical data. This has allowed me to make some prediction of the spread here in Italy, such as when the expected peak will show up. According to my findings - at least in Italy - it seems that we are currently experiencing the peak of the spread. However, social-distancing restrictions must be followed in order to contain the COVID-19 spread. 

# In[ ]:


import pandas as pd
import numpy as np
import datetime as dt
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import plotly.graph_objects as go
import plotly.express as px
from IPython.display import Image


# # Load data
# The data is taken from [Novel Corona Virus 2019 Dataset](https://www.kaggle.com/sudalairajkumar/novel-corona-virus-2019-dataset), where it is available in a time-series shape.
# 
# I will focus on the analysis of the spread in my country - that is, Italy - for the sake of simplicity, but this notebook should easily be adapted to any other country.
# 
# Three types of data is available:
# - confirmed cases;
# - deaths;
# - recovered.
# 
# Let's load one of those and explore the dataframe.

# In[ ]:


df = pd.read_csv('../input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv')
df.head()


# Ok, nice. Each row reports the (cumulative) number of individuals for each day. Moreover, the first 4 columns contains geographical data.
# 
# Here, I load the data from each .csv file, for Italy.

# In[ ]:


country = 'Italy'

# Confirmed cases
df = pd.read_csv('../input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv')
cols = df.columns[4:]
infected = df.loc[df['Country/Region']==country, cols].values.flatten()

# Deaths
df = pd.read_csv('../input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv')
deceased = df.loc[df['Country/Region']==country, cols].values.flatten()

# Recovered
df = pd.read_csv('../input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv')
recovered = df.loc[df['Country/Region']==country, cols].values.flatten()


# Let's see the raw data we have just imported.

# In[ ]:


dates = cols.values
x = [dt.datetime.strptime(d,'%m/%d/%y').date() for d in dates]

fig = go.Figure(data=go.Scatter(x=x, y=infected,
                               mode='lines+markers',
                               name='Infected'))
fig.add_trace(go.Scatter(x=x, y=deceased,
                    mode='lines+markers',
                    name='Deceased'))
fig.add_trace(go.Scatter(x=x, y=recovered,
                    mode='lines+markers',
                    name='Recovered'))
fig.update_layout(title='COVID-19 spread in Italy',
                   xaxis_title='Days',
                   yaxis_title='Number of individuals')
fig.show()


# A quick data cleaning allows us to focus on the period when the epidemic is actually spreading. We are looking for values greater than zero.

# In[ ]:


infected


# By looking at the data, and by reading the [news](https://www.corriere.it/cronache/20_gennaio_30/coronavirus-italia-corona-9d6dc436-4343-11ea-bdc8-faf1f56f19b7.shtml) of that period, it seems that the values of 2-3 referes to isolated case in Rome, which were securely hospitalized.
# 
# So, I exclude those values from the data, focusing only on the exponential-growth values.

# In[ ]:


infected_clean = infected[30:]
deceased_clean = deceased[30:]
recovered_clean = recovered[30:]


# # Modelling the epidemic spread
# ## The SEIR model
# The modelling of an infectious desease is commonly performed with the so-called **SEIR model**  ([here](https://shorturl.at/bmA25) and [here](https://en.wikipedia.org/wiki/Compartmental_models_in_epidemiology)). The population is divided into compartments:
# 
# - Susceptible
# - Exposed: individuals which are *infected*, but not yet *infectious* 
# - Infected
# - Recovered: deceased or recovered individuals
# 
# The individuals move through each compartment as shown in the figure below ([source](https://www.idmod.org/docs/hiv/model-seir.html)).

# In[ ]:


Image('../input/seir-model/SEIR.png')


# This is mathematically modelled with a set of ordinary differential equations, shown below.
# 
# $$\left\{ \begin{array}{lcr}
# \frac{dS}{dt}=-\beta\frac{IS}{N}\\
# \frac{dE}{dt}=\beta\frac{IS}{N}-\sigma E\\
# \frac{dI}{dt}=\sigma E-\gamma I\\
# \frac{dR}{dt}=\gamma I
# \end{array}\right.$$
# 
# It is clear that the behaviuor of the dynamic system relies on the parameters $\beta, \gamma, \sigma$. Their roles are:
# - **Infectious** rate $\beta$: controls the rate of spread, which represents the probability of transmitting disease between a susceptible and an infectious individual
# - **Recovery** rate $\gamma$: is determined by the average duration of infection.
# - **Incubation** rate $\sigma$: is the rate of latent individuals becoming infectious. The average duration of incubation is $1/\sigma$.
# 
# The reproduction number $R_0$ is defined as $R_0=\frac{\beta}{\gamma}$.
# 
# As suggested in this [notebook](https://www.kaggle.com/anjum48/seir-model-with-intervention-final), the social-distancing effect can be modeled as a decrease in the reproduction numer $R_0$ of the desease.
# 
# Here, I choose to model this behaviour with an exponential decay on $\beta$:
# 
# $$\beta\left(t> t_q\right)=\beta\cdot\exp\left(-\alpha\cdot(t-t_q)\right)$$
# 
# where $t_q$ is the time at which the social-distancing takes place and $\alpha$ defines the decay.

# In[ ]:


def SEIR_q(t, y, beta, gamma, sigma, alpha, t_quarantine):
    """SEIR epidemic model.
        S: subsceptible
        E: exposed
        I: infected
        R: recovered
        
        N: total population (S+E+I+R)
        
        Social distancing is adopted when t>=t_quarantine.
    """
    S = y[0]
    E = y[1]
    I = y[2]
    R = y[3]
    
    if(t>t_quarantine):
        beta_t = beta*np.exp(-alpha*(t-t_quarantine))
    else:
        beta_t = beta
    dS = -beta_t*S*I/N
    dE = beta_t*S*I/N - sigma*E
    dI = sigma*E - gamma*I
    dR = gamma*I
    return [dS, dE, dI, dR]


# This function can be integrated with the *scipy.integrate.solve_ivp* command ([link]()). 

# In[ ]:


N = 100
beta, gamma, sigma, alpha = [2, 0.4, 0.1, 0.5]
t_q = 10
y0 = np.array([99, 0, 1, 0])
sol = solve_ivp(SEIR_q, [0, 100], y0, t_eval=np.arange(0, 100, 0.1), args=(beta, gamma, sigma, alpha, t_q))

fig = go.Figure(data=go.Scatter(x=sol.t, y=sol.y[0], name='Susceptible, with intervention',
                               line=dict(color=px.colors.qualitative.Plotly[0])))
fig.add_trace(go.Scatter(x=sol.t, y=sol.y[1], name='Exposed, with intervention',
                        line=dict(color=px.colors.qualitative.Plotly[1])))
fig.add_trace(go.Scatter(x=sol.t, y=sol.y[2], name='Infected, with intervention',
                        line=dict(color=px.colors.qualitative.Plotly[2])))
fig.add_trace(go.Scatter(x=sol.t, y=sol.y[3], name='Recovered, with intervention',
                        line=dict(color=px.colors.qualitative.Plotly[3])))


beta, gamma, sigma, alpha = [2, 0.4, 0.1, 0.0]
t_q = 10
y0 = np.array([99, 0, 1, 0])
sol = solve_ivp(SEIR_q, [0, 100], y0, t_eval=np.arange(0, 100, 0.1), args=(beta, gamma, sigma, alpha, t_q))

fig.add_trace(go.Scatter(x=sol.t, y=sol.y[0], name='Susceptible, no intervention',
                               line=dict(color=px.colors.qualitative.Plotly[0], dash='dash')))
fig.add_trace(go.Scatter(x=sol.t, y=sol.y[1], name='Exposed, no intervention',
                        line=dict(color=px.colors.qualitative.Plotly[1], dash='dash')))
fig.add_trace(go.Scatter(x=sol.t, y=sol.y[2], name='Infected, no intervention',
                        line=dict(color=px.colors.qualitative.Plotly[2], dash='dash')))
fig.add_trace(go.Scatter(x=sol.t, y=sol.y[3], name='Recovered, no intervention',
                        line=dict(color=px.colors.qualitative.Plotly[3], dash='dash')))

fig.update_layout(title='SEIR epidemic model',
                 xaxis_title='Days',
                 yaxis_title='Percentage of population')
fig.show()


# The intervention, parameterized by $\alpha$ in this analysis, shows a clear positive effect both on the number of total infected and exposed individuals, and on the end of the crisis.
# 
# Now that the behaviour of the system is clear, it is possible to use the historical data to fit the model (i.e. to find the parameters $\beta, \gamma, \sigma, \alpha$).
# 
# # Fit model to data
# As said earlier, we will use the historical data to find the optimal parameters of the SEIR model. This objective will be achieved throught a *mono-objective optimization*.
# The objective function is the *mean squared error* between the prediction of the model and historical data.
# 
# Note that some correspondencies exists between historical data and the results of the SEIR model. IN detail:
# - the number of *infected* individuals is equal to the cumulative sum of E and I;
# - the number of *recovered* and *deceased* individuals is equal to R. 

# In[ ]:


def fit_to_data(vec, t_q, N, test_size):
    beta, gamma, sigma, alpha = vec
    
    sol = solve_ivp(SEIR_q, [0, t_f], y0, args=(beta, gamma, sigma, alpha, t_q), t_eval=t_eval)
    
    split = np.int((1-test_size) * infected_clean.shape[0])
    
    error = (
        np.sum(
            5*(deceased_clean[:split]+recovered_clean[:split]-sol.y[3][:split])**2) +    
        np.sum(
            (infected_clean[:split]-np.cumsum(sol.y[1][:split]+sol.y[2][:split]))**2)
    ) / split
    
    return error


# #### Initial conditions and other assumptions
# A few assumptions must be made in order to proceed with the simulation.
# Whereas the initial conditions could be defined by looking at the first element of the data, it is not clear what to choose as population size.
# 
# In Italy, there is a high mortality rate, much higher than the ones recorded in other countries. This is due to a high bias of the tested population: in fact, test are performed only on likely-positive individuals.
# 
# [Here](https://www.ispionline.it/en/publication/covid-19-and-italys-case-fatality-rate-whats-catch-25586), it is discussed that the mortality rate, despite the 10% recorded, is actually somewhere around 1.1%. Keep in mind that China's and Germany's Case Fatality Rates hovered around 4% and 0.5%, respectively.
# 
# To remove the bias introduced by testing policy, the italian population is therefore divided by 10/1.1.

# In[ ]:


N = 60e6 / (10/1.1)
N = np.int(N)
t_q = 7 # quarantine takes place
t_f = infected_clean.shape[0]
y0 = [N-infected_clean[0], 0, infected_clean[0], 0]
t_eval = np.arange(0,t_f,1)
test_size = 0.1

opt = minimize(fit_to_data, [2, 1, 0.8, 0.3], method='Nelder-Mead', args=(t_q, N, test_size))
beta, gamma, sigma, alpha = opt.x
sol = solve_ivp(SEIR_q, [0, t_f], y0, args=(beta, gamma, sigma, alpha, t_q), t_eval=t_eval)


# In[ ]:


fig = go.Figure(data=go.Scatter(x=x[30:], y=np.cumsum(sol.y[1]+sol.y[2]), name='E+I',
                               marker_color=px.colors.qualitative.Plotly[0]))
fig.add_trace(go.Scatter(x=x[30:], y=infected_clean, name='Infected', mode='markers', 
                         marker_color=px.colors.qualitative.Plotly[0]))
fig.add_trace(go.Scatter(x=x[30:], y=sol.y[3], name='R', mode='lines', 
                         marker_color=px.colors.qualitative.Plotly[1]))
fig.add_trace(go.Scatter(x=x[30:], y=deceased_clean+recovered_clean, name='Deceased+recovered', 
                         mode='markers', 
                         marker_color=px.colors.qualitative.Plotly[1]))
fig.add_trace(go.Scatter(x=[x[37], x[37]], y=[0, 100000], name='Quarantine', mode='lines',
                        marker_color='darkgrey'))
fig.update_layout(title='''Model's predictions vs historical data''',
                   xaxis_title='Days',
                   yaxis_title='Number of individuals')

fig.show()


# Ok, the predictions does not look so bad. In my opinion, a better definition of the error will lead to a better result. Moreover, the data chosen as test (the last 5-ish samples) contains a lot of information on the behaviour of the system, which is constantly changing, day in day out.
# So, let's fit the model on the whole historical data and use this to make some predictions.

# In[ ]:


test_size = 0

opt = minimize(fit_to_data, [2, 1, 0.8, 0.3], method='Nelder-Mead', args=(t_q, N, test_size))
beta, gamma, sigma, alpha = opt.x
sol = solve_ivp(SEIR_q, [0, t_f], y0, args=(beta, gamma, sigma, alpha, t_q), t_eval=t_eval)


# In[ ]:


fig = go.Figure(data=go.Scatter(x=x[30:], y=np.cumsum(sol.y[1]+sol.y[2]), name='E+I',
                               marker_color=px.colors.qualitative.Plotly[0]))
fig.add_trace(go.Scatter(x=x[30:], y=infected_clean, name='Infected', mode='markers', 
                         marker_color=px.colors.qualitative.Plotly[0]))
fig.add_trace(go.Scatter(x=x[30:], y=sol.y[3], name='R', mode='lines', 
                         marker_color=px.colors.qualitative.Plotly[1]))
fig.add_trace(go.Scatter(x=x[30:], y=deceased_clean+recovered_clean, name='Deceased+recovered', 
                         mode='markers', 
                         marker_color=px.colors.qualitative.Plotly[1]))
fig.add_trace(go.Scatter(x=[x[37], x[37]], y=[0, 100000], name='Quarantine', mode='lines',
                        marker_color='darkgrey'))
fig.update_layout(title='''Model's predictions vs historical data''',
                   xaxis_title='Days',
                   yaxis_title='Number of individuals')

fig.show()


# # Predictions
# Now it is possible to use the fitted model to make some predictions. I am looking for the day when the peak will be reached.

# In[ ]:


days_ahead = 45
new_x = x[30:] + [x[-1]+dt.timedelta(days=day) for day in range(1, days_ahead)]
t_eval = np.arange(0,t_f+days_ahead,1)
sol = solve_ivp(SEIR_q, [0, t_f+days_ahead], y0, args=(beta, gamma, sigma, alpha, t_q), t_eval=t_eval)

peak = new_x[np.argmax(sol.y[2])]

fig = go.Figure(data=go.Scatter(x=new_x, y=sol.y[1], name='E'))
fig.add_trace(go.Scatter(x=new_x, y=sol.y[2], name='I'))
fig.add_trace(go.Scatter(x=new_x, y=sol.y[3], name='R'))
fig.add_trace(go.Scatter(x=[peak, peak], y=[0, 5e4], name='Predicted peak', mode='lines',
             line=dict(color=px.colors.qualitative.Plotly[3], dash='dot')))
fig.update_layout(title='''Model's predictions''',
                   xaxis_title='Days',
                   yaxis_title='Number of individuals')
fig.show()


# In[ ]:


fig = go.Figure(data=go.Scatter(x=new_x, y=np.cumsum(sol.y[1]+sol.y[2]), name='Infected'))
fig.add_trace(go.Scatter(x=new_x, y=sol.y[3], name='Deceased+recovered'))

fig.update_layout(title='''Model's predictions''',
                   xaxis_title='Days',
                   yaxis_title='Number of individuals')
fig.show()


# We can roughly predict number of deceased with the current death rate (among infected), which is somewhere around 45% in Italy ([source](https://www.worldometers.info/coronavirus/country/italy/)).

# In[ ]:


death_rate=.457

fig = go.Figure(data=go.Scatter(x=new_x, y=sol.y[3]*death_rate, name='Deceased (predicted)',
                               line=dict(color=px.colors.qualitative.Plotly[2])))
fig.add_trace(go.Scatter(x=x[30:], y=deceased_clean, name='Historical', mode='markers',
                        marker_color=px.colors.qualitative.Plotly[3]))
fig.update_layout(title='Predicted deaths and historical data',
                   xaxis_title='Days',
                   yaxis_title='Number of individuals')
fig.show()


# # Proof of concept
# The parameters of SEIR model allow to define some intrinsic characteristics of COVID-19, such as the **reproductive number** $R_0$ and the **mean incubation period**.
# From [literature](https://www.worldometers.info/coronavirus/coronavirus-incubation-period/) these shows high variance, but mean values seems to be around $\bar{R_0}=2.5$ and $\bar{t_i}=3.0 days$.

# In[ ]:


R_0 = beta / gamma
incubation = 1 / sigma

print('Estimated reproductive number: {:.2f}'.format(R_0))
print('Estimated mean incubation period: {:.2f}'.format(incubation))


# Which seems to be fairly reasonable.

# ## On the need of social-distancing
# Here, I present a simple model which shows what is likely to happen if the social-distance restriction were not followed. In the previous model, I add a time $t_{stop}$ at which (oversimplifing) $\beta$ returns to its initial value.

# In[ ]:


def SEIR_q_stop(t, y, beta, gamma, sigma, alpha, t_quarantine, t_stop):
    """SEIR epidemic model.
        S: subsceptible
        E: exposed
        I: infected
        R: recovered
        
        N: total population (S+E+I+R)
        
        Social distancing is adopted when t>t_quarantine and t<=t_stop.
    """
    S = y[0]
    E = y[1]
    I = y[2]
    R = y[3]
    
    if(t>t_quarantine and t<=t_stop):
        beta_t = beta*np.exp(-alpha*(t-t_quarantine))
    else:
        beta_t = beta
    dS = -beta_t*S*I/N
    dE = beta_t*S*I/N - sigma*E
    dI = sigma*E - gamma*I
    dR = gamma*I
    return [dS, dE, dI, dR]


# In[ ]:


N = 100
beta, gamma, sigma, alpha = [2, 0.4, 0.1, 0.5]
t_q = 10
t_stop = 30
y0 = np.array([99, 0, 1, 0])
sol = solve_ivp(SEIR_q_stop, [0, 100], y0, t_eval=np.arange(0, 100, 0.1), args=(beta, gamma, sigma, alpha, t_q, t_stop))

fig = go.Figure(data=go.Scatter(x=sol.t, y=sol.y[0], name='Susceptible, interrupted',
                               line=dict(color=px.colors.qualitative.Plotly[0])))
fig.add_trace(go.Scatter(x=sol.t, y=sol.y[1], name='Exposed, interrupted',
                        line=dict(color=px.colors.qualitative.Plotly[1])))
fig.add_trace(go.Scatter(x=sol.t, y=sol.y[2], name='Infected, interrupted',
                        line=dict(color=px.colors.qualitative.Plotly[2])))
fig.add_trace(go.Scatter(x=sol.t, y=sol.y[3], name='Recovered, interrupted',
                        line=dict(color=px.colors.qualitative.Plotly[3])))

t_stop = 200
sol = solve_ivp(SEIR_q_stop, [0, 100], y0, t_eval=np.arange(0, 100, 0.1), args=(beta, gamma, sigma, alpha, t_q, t_stop))

fig.add_trace(go.Scatter(x=sol.t, y=sol.y[0], name='Susceptible, continuous',
                               line=dict(color=px.colors.qualitative.Plotly[0], dash='dash')))
fig.add_trace(go.Scatter(x=sol.t, y=sol.y[1], name='Exposed, continuous',
                        line=dict(color=px.colors.qualitative.Plotly[1], dash='dash')))
fig.add_trace(go.Scatter(x=sol.t, y=sol.y[2], name='Infected, continuous',
                        line=dict(color=px.colors.qualitative.Plotly[2], dash='dash')))
fig.add_trace(go.Scatter(x=sol.t, y=sol.y[3], name='Recovered, continuous',
                        line=dict(color=px.colors.qualitative.Plotly[3], dash='dash')))

fig.update_layout(title='SEIR epidemic model - effect of social-distancing',
                 xaxis_title='Days',
                 yaxis_title='Percentage of population')
fig.show()


# # Conclusions
# In this notebook I have applied the SEIR model to fit the historical data of the spread of COVID-19 in Italy. My results shows that we are experiencing the peak in these days, and that for the next 30 days an increment is to be expected in the number of infected, deceased and recovered individuals.
# A proof of the model is given by the estimates of both the reproductive number and mean incubation period obtained, which are similar to the values present in literature. A more solid study should be done, estimating interval of confidence by considering the effect of bias introducted in historical data with the italian testing policy.
# 
# Althought this notebook models the situation in Italy, it should be able to adapt to every other country.
# 
# *Note that the predictions are based on the assumptions that social-distancing restrictions are stricly followed. So please, #StayHome and #StaySafe*.
