#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np, pandas as pd, scipy as sp
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta
from time import strptime


# ## Credits
# 
# First and foremost, thank you to the following kernels provided by **datasaurus** and **Patrick Sanchez** for motivating the study of SEIR models and providing useful functionalities (I'm trying to learn how to properly tag these users as this is my first kaggle notebook)
# 
# https://www.kaggle.com/saga21/covid-global-forecast-sir-model-ml-regressions
# 
# https://www.kaggle.com/anjum48/seir-model-with-intervention
# 
# 
# **The first** extension I want to provide from the previous notebooks are a detailed explanation of the ODE parameters. While this notebook only obtains point-wise estimates of the model parameters as motivated by the above notebooks, understanding the parameters in detail will help in formulating a Bayesian model as the next step. As more data comes in, both times series and medical statistics related to the virus, it would be interesting to see how much insight we can get when modeling the ODE parameters with some suitable prior distributions and obtain histograms rather than point estimates. In such a Bayesian model we have the flexibility to also consider grouping parameters geographically, as neighbouring countries may take similar measures due to governmental/social similarities. **(Bayes model discussed at the end).**
# 
# **The second** extension is to split $R$ into two groups (details in parameters section). Since we are very interested in the total confirmed cases and fatalities, we can divide the R group into two mutually exclusive groups: Recovered (R) and Deceased (D). This will yield the model of the form $S \to E \to I \to R / D$ and allow us to model the $D$ curve explicitly.
# 

# # Vanilla Model
# 
# The SEIR model is a widely known epidemiological model of disease transmission within a population. It considers four main groups: Susceptible, Exposed, Infectious, Recovered (S, E, I, R). It is an extension of the SIR model in that the exposed group has come into contact with an infectious member but have not yet become infectious themselves, thereby considering the realistic assumption that the disease has some incubation period. If we consider $S_t,E_t,I_t,R_t$ to be the number of members of these respective groups at any time $t$, the evolution of these four groups can be modeled with the following set of Ordinary Differential Equations (ODEs):
# 
# \begin{align}
# &\frac{dS_t}{dt} =-\alpha S_t I_t \label{eq1}\tag{1} \\
# &\frac{dE_t}{dt} =\alpha S_t I_t -\beta E_t \label{eq2}\tag{2} \\
# &\frac{dI_t}{dt} = \beta E_t - \gamma I_t \label{eq3}\tag{3} \\
# &\frac{dR_t}{dt} =\gamma I_t \label{eq4}\tag{4}\\
# &N = S_t + E_t + I_t + R_t \label{eq5}\tag{5}
# \end{align}
# 
# where each equation explains the change in size of each group per unit time as they transition from one to another over time; the transitions through the states are strictly forward $(S \to E \to I \to R)$. Note that group $R$ includes both recoveries and fatalities together.
# 
# This model assumes a fixed population size $N$ with no immigration, no disease intervention, that natural birth and death rates are equal, and of course the most strict assumption that all members are uniformly well mixed.
# 
# For further reading:
# 
# https://en.wikipedia.org/wiki/Compartmental_models_in_epidemiology#The_SEIR_model
# 
# 
# ## Parameter interpretations
# 
# ## $\alpha$
# 
# In equation $(1)$ $\alpha$ describes the rate at which susceptible members become truly exposed per unit time. $\alpha$ from $(1)$ can be expanded further as a function of a few things. If we consider:
# 
# $r$ is the average number of people that an infectious person comes into contact with during infection
# <br>
# $\frac{S}{N}$ is the percentage of the population currently susceptible
# <br>
# $\rho$ is the probability that the disease is transmitted from an infectious member to a susceptible member given that they contacted
# <br>
# $T$ is the average time it takes for an infectious member to transition into the $R$ group
# 
# It follows that given $I_t$ infectious members and $S_t$ members currently susceptible, we would expect the number of suseptible members becoming exposed per unit time to be
# 
# \begin{align}
#     \frac{\rho \times r \times \frac{S_t}{N} \times I_t}{T}
# \end{align}
# 
# which implies that
# 
# \begin{align}
#     \alpha=\frac{\rho r}{N T}
# \end{align}
# 
# To implement disease intervention we can consider $r$ no longer to be constant, but some decay function overtime described above, $r_t$. In particular we consider the Hill decay function as contributed by datasaurus. But we can also modify it to ensure that number of contacts will never be reduced to 0 as that is unrealistic for humans.
# 
# \begin{align}
#     r(t;k,L)=r_0 \left( \frac{c}{1+(\frac{t}{L})^k} +1-c \right)
# \end{align}
# 
# We now have a time dependent $\alpha$
# 
# \begin{align}
#     \alpha_t = \frac{\rho \times r_t }{NT}
# \end{align}
# 
# In this notebook we will fix $k=2.5$ and $L=25$ and $c=1$ strictly for illustration because this is what seemed to work in a lot of cases with the data so far.
# 

# In[ ]:


k, L = 2.5, 25
hill = lambda t : 1 / (1 + (t/L)**k)
times = np.arange(100)

plt.plot(hill(times))
plt.ylabel('Decay')
plt.xlabel('Days')
plt.title('Hill function with k=2.5, L=25')
plt.show()


# To implement extension 2) we can consider splitting $R$ into two mutually exclusive groups: $R$ (recoveries from the disease) and $D$ (fatalities from the disease). If we consider
# 
# $T_R$: the average time for $I$ member to recover
# <br>
# $T_F$: the average time to fatality of $I$ member
# <br>
# $p_R$: probability of infectious member recovering
# <br>
# $p_F$: probability of infectious member dying
# 
# It follows that given $I_t$ infectious members (who will recover) and $S_t$ members currently susceptible, we would expect the number of suseptible members becoming exposed per unit time by future survivors to be
# 
# \begin{align}
#     \frac{\rho \times r_t \times \frac{S_t}{N} \times p_R \times I_t}{T_R}
# \end{align}
# 
# which implies that
# 
# \begin{align}
#     \alpha_t^R = \frac{\rho r_t p_R }{NT_R}
# \end{align}
# 
# And we would expect the number of suseptible members becoming exposed per unit time by those who will die in the future to be
# 
# \begin{align}
#     \frac{\rho \times r_t \times \frac{S_t}{N} \times p_F \times I_t}{T_F}
# \end{align}
# 
# which implies that
# 
# \begin{align}
#     \alpha_t^F = \frac{\rho r_t p_F }{NT_F}
# \end{align}
# 
# ## $\beta$
# 
# In equation $(2)$ $\beta$ describes the rate at which exposed members become infected per unit time. Given that an exposed member has successfully been transmitted the disease, then assuming they will become infectious after incubation period $T_I$ with probability 1:
# 
# \begin{align}
#     \beta = \frac{1}{T_I}
# \end{align}
# 
# ## $\gamma$
# 
# In equation $(3)$ $\gamma$ describes the rate at which infectious members transition into $R$. We know some will be recoveries and some will be fatalities, so the ones who will recover will do so on average in $T_R$ days with probability $p_R$ and the ones who will die will do so on average in $T_F$ days with probability $p_F$
# 
# \begin{align}
#     \gamma = \frac{p_R}{T_R} + \frac{p_F}{T_F}
# \end{align}
# 
# 
# ## Final Model
# 
# \begin{align}
# &\frac{dS_t}{dt} = -\frac{\rho r_t p_R }{NT_R} S_t I_t - \frac{\rho r_t p_F }{NT_F} S_t I_t\\
# &\frac{dE_t}{dt} =\frac{\rho r_t p_R }{NT_R} S_t I_t + \frac{\rho r_t p_F }{NT_F} S_t I_t -\frac{1}{T_I}E_t\\
# &\frac{dI_t}{dt} = \frac{1}{T_I}E_t - \frac{p_R}{T_R}I_t - \frac{p_F}{T_F}I_t\\
# &\frac{dR_t}{dt} = \frac{p_R}{T_R}I_t\\
# &\frac{dD_t}{dt} = \frac{p_F}{T_F}I_t\\
# &N = S_t + E_t + I_t + R_t + D_t\\
# &p_F + p_R =1
# \end{align}
# 
# We will be modeling the fatalities $D$ and the total confirmed cases, the total number of distinct confirmed cases (survivals and fatalities) at time $t$, $C_t$, can be computed as
# 
# \begin{align}
#     C_t = \sum_{\tau=1}^t I_{\tau} + R_{\tau} + D_{\tau} - I_{\tau-1} - R_{\tau-1} - D_{\tau-1}
# \end{align}
# 
# $(S_0,E_0,I_0,R_0,D_0)$ are the initial conditions at $t=0$.
# 

# # Data
# 
# Populaton data is provided by datasaurus

# In[ ]:


train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/train.csv')
populations = pd.read_csv('/kaggle/input/covid19-population-data/population_data.csv')
populations = populations.drop(columns=['Type']).set_index('Name').transpose()
populations = populations.to_dict()
train.columns = ['Id', 'State', 'Country', 'Date', 'ConfirmedCases', 'Fatalities']


# # Define some functions

# In[ ]:


# visualization function for later

def multi_plot(M, susceptible = True, labels=False, interventions=False):
    n = M.shape[0]
    CC0 = M[2,0]+M[3,0]+M[4,0]
    CCases = np.diff(M[2]+M[3]+M[4], prepend=CC0).cumsum()
    Deaths = M[4]
    fig = plt.figure()
    ax1 = fig.add_axes([0.1, 1, 1.25, 1], ylabel='# of People')
    ax2 = fig.add_axes([0.1, 0, 1.25, 1], ylabel='# of People')    
    if susceptible == True:
        rows=range(0,n)
    else:
        rows=range(1,n)
    for ii in rows:
        if labels == False:
            ax1.plot(M[ii])
        else:
            ax1.plot(M[ii], label = labels[ii])
    if interventions==False:
        ax1.set_title('Time Evolution without intervention')
    else:
        ax1.set_title('Time Evolution with intervention')
        for action, day in zip(list(interventions.keys()), [interventions[kk]['day'] for kk in list(interventions.keys())]):
            ax1.axvline(x=day,label=action, linestyle='--')
            ax2.axvline(x=day,label=action, linestyle='--')
    ax1.legend(loc='best')
    ax2.plot(CCases, label='ConfirmedCases', color='brown')
    ax2.plot(Deaths, label='Deaths', color='black')
    ax2.legend(loc='best')
    ax2.set_xlabel('Days')
    plt.show()

categories = ['Susceptible','Exposed','Infected','Recovered','Deceased']


# In[ ]:


# SEIRD model for simulation

def reproduction(t):
    intervention_days = [interventions[kk]['day'] for kk in list(interventions.keys())]
    reproduction_rates = [interventions[kk]['reproduction_rate'] for kk in list(interventions.keys())]
    ix=np.where(np.array(intervention_days)<t)[0]
    
    if len(ix)==0:
        return R0
    else:
        return reproduction_rates[ix.max()]


def dS_dt(S, I, reproduction_t, alpha1, alpha2):
    return -alpha1*reproduction_t*S*I -alpha2*reproduction_t*S*I

def dE_dt(S, I, E, reproduction_t, alpha1, alpha2, beta):
    return alpha1*reproduction_t*S*I + alpha2*reproduction_t*S*I - beta*E

def dI_dt(E, I, beta, gamma, psi):
    return beta*E - gamma*I - psi*I

def dR_dt(I, gamma):
    return gamma*I

def dD_dt(I, psi):
    return psi*I


def ODE_model(t, y, Rt, alpha1, alpha2, beta, gamma, psi):

    if callable(Rt):
        reproduction_t = Rt(t)
    else:
        reproduction_t = Rt
    
    S, E, I, R, D = y
    St = dS_dt(S, I, reproduction_t, alpha1, alpha2)
    Et = dE_dt(S, I, E, reproduction_t, alpha1, alpha2, beta)
    It = dI_dt(E, I, beta, gamma, psi)
    Rt = dR_dt(I, gamma)
    Dt = dD_dt(I, psi)
    return [St, Et, It, Rt, Dt]


# # Simulation
# 
# Here we will look at what the evolution of SEIRD will look like under intervention. Interventions such as quarantines and social distancing reduce the number of members an infected person has contact with which can result in a slower rate of reproduction of the disease overtime.
# 

# In[ ]:


# Fix some population parameters for simulation (refer to previous sections for interpretations)

N = 1000 # population size
T_inc = 10 # days for average incubation
T_rec = 14 # days for average recovery
T_die = 10 # days for average infection duration given death
R0 = 5 # average number of contacts an infected person has per day
tau = 0.2 # probability of transmission given S <-> I contact
p_live = 0.95 # average survival rate
p_die = 0.05 # average pmortality rate
ndays = 100 # number of days simulated

# ODE parameters
alpha1 = tau*p_live/N # average % of susceptible people who get infected by survivor
alpha2 = tau*p_die/N # average % of susceptible people who get infected by non-survivor
beta = 1/T_inc # transition rate of incubation to infection
gamma = p_live/T_rec # transition rate of infection to recovery
psi = p_die/T_die # transition rate of infection to mortality

y0 = [N-1, 1, 0, 0, 0] # initial conditions
t_span = [0, ndays] # dayspan to evaluate
t_eval = np.arange(ndays) # days to evaluate


# In[ ]:


# Here we look at the evolution given no intervention

interventions = {}

solution = sp.integrate.solve_ivp(fun = ODE_model, t_span = t_span, t_eval = t_eval, y0 = y0, 
                                  args = (R0, alpha1, alpha2, beta, gamma, psi))

Y = np.maximum(solution.y,0)

multi_plot(Y, labels=categories)


# In[ ]:


# In this simulation, we look at the evolution given 2 interventions
# we may consider what happens to the process after some intervention, which
# reduces the average number of contacts an infected person has per day
# such as social distancing and lockdown

interventions = {'social_distancing':{'day':25, 'reproduction_rate':2},
                 'lockdown':{'day':35, 'reproduction_rate':.5}}

solution = sp.integrate.solve_ivp(fun = ODE_model, t_span = t_span, t_eval = t_eval, y0 = y0, 
                                  args = (reproduction, alpha1, alpha2, beta, gamma, psi))

Y = np.maximum(solution.y,0)

multi_plot(Y, labels=categories, interventions=interventions)


# One of the key things to notice is that the effects of any measure will show a lag, and this is partly due to the incubation period of the disease. At the time when lockdown is implemented, total number of infectious people was less than 200 while those who were exposed and incubating the virus was already well over 200, this is why we see the confirmed cases continue to increase for some time after intervention. 
# 
# In the first scenario with no intervention, we can see that the system had reached equilibrium well before day 100, in other words the entire population became infectious at some point. With intervention, we see that equilibrium has not yet been reached as the rates of transmission have significantly reduced. In fact, after 500 days there will be a portion of the population that never became exposed at all!!!

# In[ ]:


ndays = 500
t_span = [0, ndays] # dayspan to evaluate
t_eval = np.arange(ndays) # days to evaluate

solution = sp.integrate.solve_ivp(fun = ODE_model, t_span = t_span, t_eval = t_eval, y0 = y0, 
                                  args = (reproduction, alpha1, alpha2, beta, gamma, psi))

Y = np.maximum(solution.y,0)

multi_plot(Y, labels=categories, interventions=interventions)


# # Define Extended Model

# In[ ]:


def dS_dt(S, I, alpha1_t, alpha2_t):
    return -alpha1_t*S*I -alpha2_t*S*I

def dE_dt(S, I, E, alpha1_t, alpha2_t, beta):
    return alpha1_t*S*I + alpha2_t*S*I - beta*E

def dI_dt(E, I, beta, gamma, psi):
    return beta*E - gamma*I - psi*I

def dR_dt(I, gamma):
    return gamma*I

def dD_dt(I, psi):
    return psi*I


def ODE_model(t, y, alpha1t, alpha2t, beta, gamma, psi):

    alpha1_t = alpha1t(t)
    alpha2_t = alpha2t(t)
    
    S, E, I, R, D = y
    St = dS_dt(S, I, alpha1_t, alpha2_t)
    Et = dE_dt(S, I, E, alpha1_t, alpha2_t, beta)
    It = dI_dt(E, I, beta, gamma, psi)
    Rt = dR_dt(I, gamma)
    Dt = dD_dt(I, psi)
    return [St, Et, It, Rt, Dt]


# # Model Fit

# In[ ]:


def loss(theta, data, population, k, L, nforecast=0, error=True):
    alpha1_0, alpha2_0, beta, gamma, psi = theta
    
    Infected_0 = data.ConfirmedCases.iloc[0]
    ndays = nforecast
    ntrain = data.shape[0]
    y0 = [(population-Infected_0)/population, 0, Infected_0/population, 0, 0]
    t_span = [0, ndays] # dayspan to evaluate
    t_eval = np.arange(ndays) # days to evaluate
    
    def a1_t(t):
        return alpha1_0 / (1 + (t/L)**k)

    def a2_t(t):
        return alpha2_0 / (1 + (t/L)**k)

    sol = sp.integrate.solve_ivp(fun = ODE_model, t_span = t_span, t_eval = t_eval, y0 = y0, 
                                 args = (a1_t, a2_t, beta, gamma, psi))
    
    pred_all = np.maximum(sol.y, 0)
    ccases_pred = np.diff((pred_all[2] + pred_all[3] + pred_all[4])*population, n = 1, prepend = Infected_0).cumsum()
    deaths_pred = pred_all[4]*population
    ccases_act = data.ConfirmedCases.values
    deaths_act = data.Fatalities.values
    
    if ccases_act[-1]<ccases_act[-2]:
        ccases_act[-1]=ccases_act[-2]
    if deaths_act[-1]<deaths_act[-2]:
        deaths_act[-1]=deaths_act[-2]
    
    weights =  np.exp(np.arange(data.shape[0])/10)/np.exp((data.shape[0]-1)/10) 

    ccases_rmse = np.sqrt(mean_squared_error(ccases_act, ccases_pred[0:ntrain], sample_weight=weights))
    deaths_rmse = np.sqrt(mean_squared_error(deaths_act, deaths_pred[0:ntrain], sample_weight=weights))

    loss = np.mean((ccases_rmse, deaths_rmse))
    
    if error == True:
        return loss
    else:
        return loss, ccases_pred, deaths_pred


# ### DISCLAIMER: 
# First as this is a fairly recent outbreak, there is still very little data available (as of March 25 of writing this). Second as this notebook is only considering $\textbf{one model}$ for illustrative purposes only, there will be no validation, only fitting on the available training data. This is a work in progress and I will play around with a few more models as time goes on.

# In[ ]:


train['location'] = train['State'].fillna(train['Country'])
locations=list(train['location'].drop_duplicates())
train.set_index(['location', 'Date'], inplace=True)


# In[ ]:


train.head()


# In[ ]:


parms0 = [1.5, 1.5, 0.5, 0.05, 0.001]
bnds = ((0.001, None), (0.001, None), (0, 10), (0, 10), (0, 10))


# In[ ]:


def fit_ODE_model(location, k, L):
        
    dat = train.loc[location].query('ConfirmedCases > 0')
    nforecast = 75
    population = populations[location]['Population']
    n_infected = train['ConfirmedCases'].iloc[0]
        
    res = sp.optimize.minimize(fun = loss, x0 = parms0, 
                               args = (dat, population, k, L, nforecast),
                               method='L-BFGS-B', bounds=bnds)
    
    dates_all = [str(datetime.strptime(dat.index[0], '%Y-%m-%d') + timedelta(days = ii))[0:10] for ii in range(nforecast)]
    
    err, ccases_pred, deaths_pred = loss(theta = res.x, data = dat, population = population, k=k, L=L, 
                                         nforecast=nforecast, error=False)
    
    predictions = pd.DataFrame({'ConfirmedCases': ccases_pred,
                                'Fatalities': deaths_pred}, index=dates_all)
    
    train_true = dat[['ConfirmedCases',  'Fatalities']]
    predictions.columns = ['ConfirmedCases_pred',  'Fatalities_pred']

    plot_df = pd.merge(predictions,train_true,how='left', left_index=True, right_index=True)

    plt.plot(plot_df.ConfirmedCases_pred.values, color='green',linestyle='--', linewidth=0.5, label='Confirmed Cases (pred)')
    plt.plot(plot_df.Fatalities_pred.values, color='blue',linestyle='--', linewidth=0.5, label='Fatalities (pred)')
    plt.plot(plot_df.Fatalities.values, color='red', label='Fatalities (real)')
    plt.plot(plot_df.ConfirmedCases.values, color='orange', label='Confirmed Cases (real)')
    plt.title(location)
    plt.xlabel('Days since first case in '+location)
    plt.ylabel('Confirmed Cases')
    plt.legend(loc='best')
    plt.show()        
    
    print(res.x)


# # Vizualize

# In[ ]:


fit_ODE_model('Korea, South', k, L)


# Just an observation that the parameter corresponding to $\frac{p_F}{T_F}=0.0037784$. This may be in line with some current estimates that mortality rates can be as high as $5 \%$ and incubation time is 14 days $(0.05/14 = 0.0035714)$. Much more analysis must be done to estimate uncertainties in these parameters from the data. Also keep in mind that both the numerator and denominator here can change substantially across countries (different countries implement different interventions and have different medical resources, population demographics, etc) so I am very open to this similarity being just chance.

# In[ ]:


fit_ODE_model('Hubei', k, L)


# In[ ]:


fit_ODE_model('Netherlands', k, L)


# In[ ]:


fit_ODE_model('Spain', k, L)


# In[ ]:


fit_ODE_model('Poland', k, L)


# # Conclusions
# 
# In practice, fitting ODE models can be very tricky. They are very sensitive to initial conditions and especially in this case because while the initial condition can be only 1 infected person, we are solving the ODE system with a susceptile (group S) population size of sometimes 50M+ people, and so the data we see so far is only a very recent evolution that has only reached a small portion of the total population, and given only a few weeks of more data we may start to see some very different dynamics.
# 
# The model, while considering intervention, does not account for many other significant factors such as (sub)population demographics, population densities in different areas, how populations move in reaction to intervention announcement, health resources (ICU beds, ventilators), how temperatures and local conditions can affect the spread, the list goes on.
# 

# # Bayesian Model
# 
# The next step will be to infer parameters from posterior sampling. As new data comes in regarding viral characteristics we can start to form some prior distributions over the parameters.
# 
# In terms of a likelihood function, we can assume that the variance of the confirmed cases errors to some extent varies over time, which may be slightly more realtistic than assuming the variance is constant because one might expect that as the nuber of cases grows, then the so does the margin of error, etc. In other words, if we consider some error with respect to cumulative cases $(\epsilon_t=C_t^{real}-C_t(\Theta))$, where $C_t$ is the solution to the ODE parameterized by $\Theta=(T_F,T_R,T_I,p_F,p_R,\rho, L,k)$, we can impose a Gaussian error model of the form
# 
# \begin{align}
#     \epsilon_t \overset{iid}{\sim} \mathcal{N} \left(0, \sigma_{t}^2 \right)\\
# \end{align}
# 
# If we consider instead $\sigma_{t}^2 = \frac{1}{2 w_t}$ then we have
# 
# \begin{align}
#     \epsilon_t \overset{iid}{\sim} \mathcal{N} \left(0, \frac{1}{2 w_t} \right)\\
# \end{align}
# 
# It turns out that maximizing this likelihood is equivalent to minimizing a weigthed sum of squared error loss function where $w_t$ is some weight assigned to each error at time $t$ (which is similar to what was done when fitting the model above already).
# 
# If we have some prior distribution such that $\Theta \sim \pi(\Theta)$ then we can sample $\Theta$ from the posterior distribution where
# 
# \begin{align}
#     p(\Theta | \mathcal{D}) \propto L(\Theta | \mathcal{D})\pi(\Theta)
# \end{align}
# 
# Expanding on this we would have a posterior that we sample of the form 
# 
# \begin{align}
#     p(\Theta | \mathcal{D}) \propto \prod_{t} e^{-w_t\left( C_t^{real}-C_t(\Theta) \right)^2} \pi(\Theta)
# \end{align}
# 
# 
# From an optimization perspective, this would be equivalent to minimizing the negative log-posterior
# 
# \begin{align}
#     min_{\Theta} \quad -p(\Theta | \mathcal{D}) = \sum_{t}w_t \left(C_t^{real}-C_t(\Theta) \right)^2 -nlog \pi(\Theta) \\
# \end{align}
# 

# In[ ]:




