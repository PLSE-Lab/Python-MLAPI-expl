#!/usr/bin/env python
# coding: utf-8

# # Auquan & The HELP Project's Modified SEIR model to forecast coronavirus infection peaks 
# 
# ## Introduction
# In March my company Auquan began working on a community project to model coronavirus infections and predict future impact. Initially the scope was just to model the health impacts (deaths, infections, critical care capacity), but we are now working to create a unified model for explaining the health and economic impacts of different lockdown exit strategies. The disease model that we created is, as far as I'm aware, the only community created disease model to be used by the CDC (Center for Disease Control in the US) as part of there ensemble forcasting model. 
# 
# **You can see the CDC ensemble components here:** https://www.cdc.gov/coronavirus/2019-ncov/covid-data/forecasting-us.html
# 
# **You can see the final version of our model here:** https://covid19-infection-model.auquan.com/
# I would encourage you to select different countries / adjust the parameters to see how the virus impact changes.
# 
# As this is a community effort that aims to share information with people, we've decided to release this notebook containing an early version of the CDC model free to all. This notebook doesn't contains the exact CDC model as we've iterated from here a couple times and added complexity. It is the same structure and details our modified SEIR approach, so hopefully shows some of the stages required to create a robust disease model.
# 
# The community effort currently contains people from diverse companies such as Google, Microsoft, Goldman Sachs, Startups and Acedemia. If you're interested in helping work on this community effort, there are more details and a signup form here: **https://links.quant-quest.com/helpproject**
# 
# The full credit for this effort goes to the whole team, but especially [Chandini](https://www.linkedin.com/in/chandinijain/) & [Vishal](https://www.linkedin.com/in/vishaltomar28/) for their massive input on this specific model. 
# 
# David
# 
# ----
# 
# # Notebook
# 
# In this notebook, we are going to build a model to predict the actual number of infected people at any given point in any given geography based on current deceased cases, population of the country and virus spread
# 
# Parameters for virus spread is copied from here: https://github.com/midas-network/COVID-19/tree/master/parameter_estimates/2019_novel_coronavirus
# 
# We will use this to further predict lockdowns or policy actions and impact on portfolio

# ## Setup
# 
# ### Importing required packages

# In[ ]:


import pandas as pd
import numpy as np
import math

from datetime import timedelta

from scipy.optimize import curve_fit
from scipy.integrate import odeint
from scipy.interpolate import interp1d
from IPython.core.debugger import set_trace

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
plt.style.use('default')


# ### Downloading and preparing the data
# 
# We are using the John Hopkins repository to get the data for number of infections and deaths.
# 
# If running this notebook for the first time, you will need to uncomment this next section to clone the github repo.

# In[ ]:


get_ipython().system('rm -rf COVID-19 ')
get_ipython().system('git init ')
root_git = 'https://github.com/CSSEGISandData/COVID-19.git'
get_ipython().system('git clone $root_git')
get_ipython().system("git pull './COVID-19'")


# In[ ]:


data_dir = './COVID-19/csse_covid_19_data/csse_covid_19_time_series/'


# In[ ]:


# !ls -alrth $data_dir


# ### Loading the data
# 
# Here we're just making some small changes to the format of the data from the CSV and changing the names to make them smaller.

# In[ ]:


### Loading the data
rename = {'Country/Region': 'zone', 
          'Province/State': 'sub_zone'}

df_recovery = pd.read_csv(data_dir + 'time_series_covid19_recovered_global.csv').rename(columns=rename)
df_deaths = pd.read_csv(data_dir + 'time_series_covid19_deaths_global.csv').rename(columns=rename)
df_confirmed = pd.read_csv(data_dir + 'time_series_covid19_confirmed_global.csv').rename(columns=rename)
print(df_confirmed.shape)
df_confirmed.head(100).T.head().T


# In[ ]:


rename = {'Country/Region': 'zone', 
          'Province/States': 'sub_zone'}
data_dir_who ='./COVID-19/who_covid_19_situation_reports/who_covid_19_sit_rep_time_series/'
df_confirmed_who = pd.read_csv(data_dir_who + 'who_covid_19_sit_rep_time_series.csv').rename(columns=rename)
print(df_confirmed_who.shape)
df_confirmed_who.head(100).T.head().T


# In[ ]:


zones = ['China','Korea, South', 'Italy', 'Iran', 'France', 'Spain','US', 'United Kingdom']
#print(df_confirmed.zone.unique())
#[sz for sz in df_confirmed.zone if sz in zones]

def clean_df(df):
  df = df.sum(axis=0)
  df = df[[d for d in df.index if d.find('/')>0]].T
  df.index = pd.to_datetime(df.index)
  return df

def extract_cols(df, zones):
  dic_comp = {
      z: clean_df(df.query("zone == '{zone}'".format(zone=z))) for z in zones}
  return pd.DataFrame(dic_comp)

df_select_conf = extract_cols(df_confirmed, zones=zones)
df_select_conf_who = extract_cols(df_confirmed_who, zones=zones)
df_select_death = extract_cols(df_deaths, zones=zones)
df_select_reco = extract_cols(df_recovery, zones=zones)
pd.merge(df_select_conf, df_select_death,
             left_index=True, right_index=True,
             how='outer', suffixes=(' new cases', ' new death')).diff().tail()


# ## Estimating confirmed Case Fatality Rate and Infection Fatality Rate
# 
# The first step we need to take is to create an estimate for the severity of COVID-19 infection. There are a couple of statistics for doing this and each is important in different situations. The thing that most people want to know the most is how likely someone is to die once they've caught the disease. The problem in answering this question comes in defining and measuring 'someone'. Whilst most global media use various definitions without explanation, it's important that we understand what precisely we are measuring.
# 
# We are going to look at two metrics for disease fatality:
# 
# - Confirmed Case Fatality Rate (cCFR)
# - Infection Fatality Rate (IFR)
# 
# #### Confirmed Case Fatality Rate (cCFR)
# cCFR is a measure of the proportion of confirmed cases that result in a patient's death. The key point here is that the total population that is being considered is only those who've been diagnosed with COVID-19 and had their disease status confirmed (e.g. by PCR Test). This has the benefit that the total population is known and can be confirmed by counting all positive test results. [Paper calculating CFR](https://www.thelancet.com/journals/laninf/article/PIIS1473-3099%2820%2930246-2/fulltext)
# 
# #### Infection Fatality Rate (IFR)
# IFR is a measure of the proportion of total infections that result in a patient's death. The total population here includes the confirmed cases above, plus everyone else that was infected and not diagnosed. A diagnosis might not be confirmed if, for example, the patient is asymptomatic, they have mild symptoms and choose not to seek medical care, testing systems are overwhelmed so can't test people who are suspected to have the condition. This measure provides a much better picture of how dangerous an infectious disease is, as it represents the 'true' fatality rate. However, in the case of COVID-19 it is/was extremely hard to measure as all three of the examples above were true. Some countries did manage to roll out widespread testing and can provide a CFR that is closer to the IFR value - including: South Korea and Iceland. [Paper calculating IFR](https://www.thelancet.com/journals/laninf/article/PIIS1473-3099%2820%2930246-2/fulltext)
# 
# ### In our approach
# We use data from China for cCFR. cCFR can be estimated as 
# - deaths/total cases or
# - deaths/closed cases, where closed cases = (recovered cases + deaths)
# 
# Since every case will eventually either recover or die, these two should eventually converge to the same number. We take the average of that asymptotic value as cCFR
# 
# We use data from S. Korea for IFR, since they have done extensive testing. IFR can be estimated as 
# - deaths/total cases
# 
# ### Calculating cCFR and IFR
# 
# Before we can actually calculate the cCFR and IFR we need to make some adjustments to our data. In order to compare cases and case outcomes (death or recoveries), we need to align the data. This is because the people who die from a disease today arn't the same people who were recorded as infected today. Diseases take some time to progress to their outcomes and during this period more people are being infected. 
# 
# Ideally we could avoid this problem by using individual level data, and only considering cases that have reached their outcomes, however this data is not readily avaliable. Instead what we need to do is take the outcomes and compare them to the number of infections at T-x and T-y (respectively). In this approach we used figures published by the WHO at the time:

# In[ ]:


lag_onset_death = 15  ## estimate as reported by WHO
lag_onset_recovery = 21 ## estimate as reported by WHO


# In[ ]:


df_select_death.shift(-lag_onset_death) # deaths today must have been confirmed lag days before
df_select_reco.shift(-lag_onset_recovery)  # recovered today must have been confirmed lag days before


# ### Calculating mortality estimate from confirmed cases (CFR)
# 
# CFR: estimate using last x data points from China. We've selected the value of fit_points based on today, in the future you will need to change this to ensure you get a good fit.
# 

# In[ ]:


### CFR
## estimate using last x data points from China
fit_points = 110
death_rate_estimate_conf = (df_select_death.shift(-lag_onset_death                                                 )/df_select_conf.replace(0,np.nan))[                                                -fit_points-lag_onset_death:-lag_onset_death] ##estimate from confirmed cases


# ### Calculating mortality estimate from closed cases (CFR)
# 

# In[ ]:


death_rate_estimate_closed = (df_select_death.shift(-lag_onset_death                                                   )/(df_select_death.shift(-lag_onset_death)+                                                      df_select_reco.shift(-lag_onset_recovery)).replace(                                                        0, np.nan))[-fit_points -lag_onset_recovery:                                                                    -lag_onset_recovery] ##estimate from closes cases


# ### Fitting a curve to mortality rate (CFR)

# In[ ]:


### Fitting a curve to mortality rate
def func(x, a, b, c):
    return a * np.exp(-b * x) + c
#   return a * np.log(b * x) + c

def func2(x, a, b, c):
    return a * (2**(-b * x)) + c

popt, pcov = curve_fit(func, range(fit_points), [np.log(death_rate_estimate_conf['China'].iloc[x]) for x in range(fit_points)])

popt2, pcov2 = curve_fit(func2, range(len(death_rate_estimate_closed['China'])), death_rate_estimate_closed['China'])


# In[ ]:


plt.figure(figsize=(10,8))

ax1 = plt.subplot(1, 1, 1)
plt.title('Fatality Rate by confirmed vs closed cases in China', size=18)

plt.plot(range(len(death_rate_estimate_closed['China'])), death_rate_estimate_closed['China'], 'bo', label='Closed')
plt.plot(range(fit_points), np.exp(func(range(fit_points), *popt)), 'r--', label="Fitted Curve")
# Plots the data
plt.plot(range(len(death_rate_estimate_conf['China'])), death_rate_estimate_conf['China'], 'ko', alpha=.65, label='Confirmed')
plt.plot(range(fit_points), func2(range(fit_points), *popt2), 'g--', label="Fitted Curve")
plt.legend()


# We see that the two rates converge. We take the average value as asymptotic cCFR

# In[ ]:


asymptotic_fatality_rate = (np.exp(func(100, *popt)) + func2(100, *popt2))/2.0
asymptotic_fatality_rate


# ### IFR estimate from confirmed cases in Korea
# 
# Here we're going to use data from South Korea as they were the first to introduce widespread population testing to identify COVID-19. We make sure to only consider data after this program was rolled out.

# In[ ]:


date_cutoff = '2020-02-22'
popt3, pcov3 = curve_fit(func2, range(len(death_rate_estimate_conf.loc[date_cutoff:,'Korea, South'])),                         death_rate_estimate_conf.loc[date_cutoff:,'Korea, South'])


# In[ ]:



plt.figure(figsize=(14,8))

ax1 = plt.subplot(1, 1, 1)
plt.title('Infection Fatality Rate by confirmed cases in Korea after Patient 31', size=18)

# plt.plot(death_rate_estimate_closed['Korea, South'], lw=2, alpha=.75, label='Closed')
# # Plots the data
plt.plot(range(len(death_rate_estimate_conf.loc[date_cutoff:,'Korea, South'])),death_rate_estimate_conf.loc[date_cutoff:,'Korea, South'], 'Pr', alpha=.65, label='Confirmed')
plt.plot(range(150), func2(range(150), *popt3), 'g--', label="Fitted Curve")
plt.legend()


# ### Our estimates of cCFR and IFR (these will be used later in the model)

# In[ ]:


cCFR = asymptotic_fatality_rate
IFR = func2(100, *popt3)
cCFR, IFR


# ## We are now going to try and fit a modified SEIR model to the data
# 
# First we take some estimates reported by WHO as a starting point for our model. We will later let the model alter these to find the values it thinks leads to a good fit.

# In[ ]:


### estimates as reported by WHO
symptom_onset_to_death = lag_onset_death
symptom_onset_to_death_lb = 13.1
symptom_onset_to_death_ub = 17.7
doubling_time = 6.2
doubling_time_lb = 5.8
doubling_time_ub = 7.1
incubation_period = 5


# ### Defining our modified SEIR model
# 
# We use a modified SEIR for modeling the COVID-19 disease dynamics. A typical SEIR model breaks $N$, the total population, into four comparments.
# - S Susceptible
# - E Exposed
# - I Infected
# - R Recovered
# 
# We modify the model by breaking down infected category, $I$ into two:
# - the reported number of infections, I<sub>r</sub>
# - the unreported number of infections, I<sub>u</sub>
# 
# This is because testing is different across countries and a lot of countries are not doing enough tests. Additionally, people who are reported to be infected don't have the same rate of infecting others as the unreported infections. So use different $\beta$ parameters for these classes. We also break down the recovered class, $R$ into deaths $D$ and cured $C$. We assume that all the unreported infections only get cured (C<sub>u</sub>) because if they were to develop serious symptoms they would probably get reported.
# The model parameters are:
# 
# - N Total Population
# - S Susceptible
# - E Exposed
# - I<sub>u</sub> Infected Unreported
# - I<sub>r</sub> Infected reported
# - C<sub>u</sub> Cured unreported
# - C Cured reported
# - D Death reported
# - $\beta_1$ = the average number of contacts per infected person (reported) per time. Typical time between contacts is  Tc2 = 1/ $\beta_1$ 
# - $\beta_2$ = the average number of contacts per infected person (unreported) per time. Typical time between contacts is  Tc2 = 1/ $\beta_2$ 
# - $\alpha$ = $1/incubation_period$, Time it takes for a person to become infectious after exposure to virus
# - $\epsilon$ Ratio of exposed people who show symptoms and are reported as infectious
# - $\delta$ = 1/time_to_death, Time it takes for an reported infected person to die
# - $\eta$ = 1/time_to_recovery_unreported, Time it takes for an unreported infected person to recover
# - $\zeta$ = 1/time_to_recovery_reported, Time it takes for an reported infected person to recover
# 
# The differential equation governing this model are:
# 
# $\frac{dS}{dt}= -(\beta_1 I_r + \beta_2 I_u) * \frac{S}{N}$
# 
# $\frac{dE}{dt}= (\beta_1 I_r + \beta_2 I_u) * \frac{S}{N} - \alpha * E$
# 
# $\frac{Ir}{dt}= \epsilon * \alpha * E - \delta * cCFR * I_r - \zeta * (1 - cCFR) * I_r$
# 
# $\frac{Iu}{dt}= (1 - \epsilon) * \alpha * E - \eta * I_u$
# 
# $\frac{dD}{dt}= \delta * cCFR * I_r$
# 
# $\frac{dC}{dt}= \zeta * (1 - cCFR) * I_r$
# 
# $\frac{dCu}{dt}= \eta * I_u$
# 
# The function below describes the dynamics of this model

# ### Implementing our modified SEIR model
# 
# First, we need to recreate the SEIR model differential equations from above.

# In[ ]:


# The SEIR model differential equations.
def seir_deriv(y, t, N, beta1, beta2, alpha, epsilon, delta, zeta, eta, cCFR):
#     alpha = 1/incubation_period
#     delta = 1/lag_onset_death
#     zeta = 1/lag_onset_recovery
#     eta = 1/lag_onset_recovery
    S, E, Ir, D, C, Iu, Cu= y
    dSdt = -((beta1 * Iu) + (beta2 * Ir)) * S / N # Susecptible
    dEdt = ((beta1 * Iu) + (beta2 * Ir)) * S / N - alpha * E # Exposed
    dIudt = (1-epsilon) * alpha * E - eta * Iu # Infected but unreported
    dCudt = eta * Iu # Cured but unreported
    dIrdt = epsilon * alpha * E - delta * cCFR * Ir - zeta * (1-cCFR) * Ir # Infected and reported
    dDdt = delta * cCFR * Ir # Deaths
    dCdt = zeta *(1-cCFR) * Ir # Reported and Cured
    return dSdt, dEdt, dIrdt, dDdt, dCdt, dIudt, dCudt


# #### To setup the model, we need some initial estimates
# - We estimate initial reported deaths $D_0$ as deaths reported on the first day of simulation
# - We estimate initial reported recoveries $C_0$ as recoveries reported on the first day of simulation
# - We estimate initial reported infections $I_r$$_0$ as (Confirmed Cases - recoveries - deaths) reported on the first day of simulation
# - Using $IFR$, reported deaths in *time lag from symtom onset to death*(15 days), we estimate the total infected population, $I_0$ today. We then estimate initial unreported infections $I_u$$_0$ as $I_0$ - $I_r$$_0$
# - Similarly, using $IFR$, reported deaths today, we estimate the total infected population, $I_{-15}$ 15 days prior to today. We estimate initial unreported recoveries $C_u$$_0$ as $I_{-15}$ - $I_r$$_0$ - $D_0$ - $C_0$ (everybody unreported from 15 days prior must have recovered)
# - We estimate initial exposed $E$ as 100*Confirmed Cases today
# 
# *Note: In later versions of the model, instead of using estimates for these values, we allowed the model to find the values within certain bounds. We optimised this still further by giving random start points to avoid local minimas. Feel free to try and implement these changes yourself.*

# In[ ]:


### Function to setup the evolution for these models
def setup_SEIR(beta1, beta2, alpha, delta, zeta, cCFR, E0, S0, Iu0, N, df_conf, df_reco, df_death, t0, forward, generating_curve = False):
    if generating_curve:
        Ir0 = df_conf.iloc[-1] - df_reco.iloc[-1] - df_death.iloc[-1]
        D0 = df_death.iloc[-1]
        C0 = df_reco.iloc[-1]
    else:
        Ir0 = df_conf.iloc[0] - df_reco.iloc[0] - df_death.iloc[0]
        D0 = df_death.iloc[0]
        C0 = df_reco.iloc[0]
    Cu0 = 0.1*Iu0

def setup_SEIR(beta1, beta2, alpha, epsilon, delta, zeta, eta, cCFR,               N, df_conf, df_reco, df_death, t0, IFR, forward, generating_curve = False):
    
    if generating_curve: ##generate from 15 days prior to today
        D0 = df_death.iloc[-16] # intial number of deaths we know
        C0 = max(df_reco.iloc[-16], .10*df_conf.iloc[-16])   # intial number of recoveries we know
        Ir0 = df_conf.iloc[-16] - D0 - C0 ## initial reported infections

        Iu0 = df_death.iloc[-1]/IFR - Ir0 ## estimate unreported infections based on deaths in 15 days
        Cu0 = np.maximum(0, df_death.iloc[-16]/IFR - df_conf.iloc[-16]) ## assume everybody unreported from 15 days prior is recovered
        E0 =30*df_conf.iloc[-16]
    else:
        # We use an optimization method to best fit these numbers 
        D0 = df_death.iloc[0] # intial number of deaths we know
        C0 = max(df_reco.iloc[0], .10*df_conf.iloc[0])  # intial number of recoveries we know
        Ir0 = df_conf.iloc[0] - D0 - C0 ## initial reported infections
        Cu0 = np.maximum(0, df_death.iloc[0]/IFR - df_conf.iloc[0]) ## assume everybody unreported from 15 days prior is recovered
        Iu0 = 10*Cu0#df_death.iloc[15]/IFR - Ir0 ## estimate unreported infections based on deaths in 15 days
        E0 =30*df_conf.iloc[0]
    S0 = N - E0 - Ir0 - Iu0 - C0 - D0 - Cu0 # rest are susceptible
    
                                          
    # Initial conditions vector
    y0 = S0, E0, Ir0, D0, C0, Iu0, Cu0
#     print('Initial S0, E0, Ir0, D0, C0, Iu0, Cu0')
#     print(y0)

    forward_period = len(df_death) + forward
    # A grid of time points (in days)
    t = np.linspace(0, forward_period, forward_period)

    # Integrate the SIR equations over the time grid, t.
    ret = odeint(seir_deriv, y0, t, args=(N, beta1, beta2, alpha, epsilon, delta, zeta, eta, cCFR))
    S, E, Ir, D, C, Iu, Cu0= ret.T
    return S, E, Ir, D, C, Iu, Cu0, forward_period


# ### Fitting model to current data
# We now fit this model to the Reported deaths and Reported confirmed cases data we have so far on a country. At the moment, we are using WHO's estimates of incubation_period (5 days), time_to_death (15 days), and time_to_recovery (21 days) to fix $\alpha$, $\delta$, $\eta$  and  $\zeta$. We then fit the model to estimate $\beta_1$, $\beta_2$, $\epsilon$ and $cCFR$
# 
# The function returned below is the residual, which is the loss function we're minimising in the fitting process. You can see that here we are fitting to confirmed cases and number of deaths.  You can change what data is being used to calculate the fit, or the ratio in which they're used. We recommend not fitting to recovery data as this is very low quality

# In[ ]:


### Function to calculate residuals for estimating parameters
from scipy import optimize

def resid_seir(params, N, df_conf, df_reco, df_death, t0, IFR):
    S, E, Ir, D, C, Iu, Cu0, _ = setup_SEIR(params[0], params[1], params[2], params[3], 
                                            params[4], params[5], params[6], params[7],
                                            N, df_conf, df_reco, df_death, t0, IFR,0)
    true_Ir = df_conf - df_reco - df_death
    true_D = df_death
    true_C = df_reco
    fit_days = len(true_D)
    return np.nan_to_num(np.array((
#                                    np.abs(((Ir)[:len(true_Ir)] - true_Ir)) + \
                                   .3* np.abs(((Ir+D+C)[:len(true_Ir)] - df_conf)) + \
#                                     .5*np.abs((C[:len(true_C)] - true_C)) + \
                                   np.abs((D[len(true_D)-fit_days:len(true_D)]-true_D.iloc[-fit_days:]))+ \
                                 0)).astype(float))


# In[ ]:


## Functions for plotting curves
def plot_curves_seir(t0, forward_period, Ir, E, D, C, true_Ir, true_D, true_C, Iu, Iu_long, Ir_long, D_long):
    sim_dates = pd.date_range(start=t0, periods=(forward_period))
    sim_dates2 = pd.date_range(start=t0, periods=(len(Iu_long)))
    
    fig = plt.figure(figsize=(10, 32))
    ax1 = fig.add_subplot(811)  # , axis_bgcolor='#dddddd', axisbelow=True)

    ax1.plot(sim_dates, Ir, 'b', alpha=0.5, lw=2, label='Projected Reported Active Infections')
    ax1.plot(true_Ir.loc[t0:],              'ko', markersize=8, label='Reported Active Infections')

    ax2 = fig.add_subplot(812)
    ax2.plot(sim_dates, D, 'g', alpha=0.5, lw=2, label='Projected Deaths')
    ax2.plot(true_D,              'ko', markersize=8, label='Reported Deaths')

    ax3 = fig.add_subplot(813)
    ax3.plot(sim_dates, Ir+D+C, 'r', alpha=0.5, lw=2, label='Projected Confirmed Cases')
    ax3.plot(true_Ir+ true_D + true_C,              'ko', markersize=8, label='Reported Confirmed Cases')

    ax4 = fig.add_subplot(814)
    ax4.plot(sim_dates2, (Iu_long + Ir_long), 'k', alpha=0.5, lw=2, label='Projected Total Infections')
    ax4.plot(sim_dates2, (Ir_long), 'g', alpha=0.5, lw=2, label='Projected reported Infections')
    
    
    ax5 = fig.add_subplot(815)
    ax5.plot(sim_dates2, (Ir_long), 'g', alpha=0.5, lw=2, label='Long Term Reported Infections')


    ax6 = fig.add_subplot(816)
    ax6.plot(sim_dates2, (Ir_long), 'g', alpha=0.5, lw=2, label='Projected reported Infections')
    ax6.plot(sim_dates2, (D_long), 'b', alpha=0.5, lw=2, label='Long Term Death')
    
    ax7  = fig.add_subplot(817)
    ax7.plot(sim_dates, Ir, 'b', alpha=0.5, lw=2, label='Projected Reported Active Infections')
    ax7.plot(sim_dates, Iu, 'm', alpha=0.5, lw=2, label='Projected Unreported Active Infections')
    ax7.plot(sim_dates, D, 'g', alpha=0.5, lw=2, label='Projected Deaths')
    ax7.plot(sim_dates, C, 'r', alpha=0.5, lw=2, label='Projected Recoverd')
    ax7.plot(sim_dates, (E), 'k', alpha=0.5, lw=2, label='Projected Exposed')
    
    ax8  = fig.add_subplot(818)
    ax8.plot(sim_dates2, Ir_long, 'b', alpha=0.5, lw=2, label='Projected Reported Active Infections')
    ax8.plot(sim_dates2, Iu_long, 'm', alpha=0.5, lw=2, label='Projected Unreported Active Infections')
    ax8.plot(sim_dates2, D_long, 'g', alpha=0.5, lw=2, label='Projected Deaths')
    ax8.plot(sim_dates2, C_long, 'r', alpha=0.5, lw=2, label='Projected Recoverd')
    ax8.plot(sim_dates2, E_long, 'k', alpha=0.5, lw=2, label='Projected Exposed')

    for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]:
        ax.set_xlabel('Dates')
        ax.set_ylabel('Population')
        ax.yaxis.set_tick_params(length=0)
        ax.xaxis.set_tick_params(length=0)
        ax.grid(b=True, which='major', c='w', lw=2, ls='-')
        legend = ax.legend()
        legend.get_frame().set_alpha(0.5)
        for spine in ('top', 'right', 'bottom', 'left'):
            ax.spines[spine].set_visible(False)


# ### Fitting our modified SEIR to different countries
# 
# First we are going to add some information on some of the countries we are interested in modelling. We can add other countries here as well.
# 
# We also want to limit the model to only train after the 10th death. The reason for this decision is that early infection data will be highly affected by noise and individual variation. By making the model wait untill this point to start the fitting process we're hoping to get a better fit to the underlying dynamics.

# In[ ]:


country = 'United Kingdom'

population_map = {'China' : 1400000000,
                 'Korea, South' : 51500000,
                'Italy': 60550000,
                'Iran': 80000000,
                'France': 67000000,
                'Spain': 46700000,
                'US': 327000000,
                'United Kingdom': 66400000,
                  'Belgium':11400000,
                'Austria':8820000,
                  'India':1250000000,
}

N = .7*population_map[country]


# In[ ]:


df_conf_all = clean_df(df_confirmed[(df_confirmed['zone']==country) ])
df_reco_all = clean_df(df_recovery[(df_recovery['zone']==country) ])
df_death_all = clean_df(df_deaths[(df_deaths['zone']==country) ])


# ### Only fit to datapoints after tenth death - fit to lst 14 to 21 days
# 
# Here we are fiiting to all data after the 10th death. This isn't the best way to do this as the model parameters change over time (due to government intervention such a social distancing). In our final model we train over the previous 2 weeks data. This is something you can experiment with.

# In[ ]:


t0 = df_death_all[df_death_all>29999].index[0]#df_hubei_conf[df_hubei_conf>2].index[0]
df_conf = df_conf_all.loc[df_conf_all.index>=t0]
df_reco = df_reco_all.loc[df_reco_all.index>=t0]
df_death = df_death_all.loc[df_death_all.index>=t0]


# In[ ]:


t0, len(df_conf), df_conf.iloc[0], df_reco.iloc[0], df_death.iloc[0]


# ### Estimate the parameters, by fitting to the curve
# 
# For initial estimates, we use 
# - $\beta_1$, $\beta_2$ calculated using estimates of $R_0$, *doubling time* and *time_to_death*
# - $\epsilon$ = 0.2 
# - $cCFR$ estimated from China's data

# In[ ]:


# Contact rate, beta, and mean recovery rate, gamma, (in 1/days).
R = 2.68
gamma = 1/lag_onset_death
growth_rate = 2**(1/doubling_time) - 1
beta = R*gamma#(growth_rate + gamma)
relative_contact_rate = 0
beta = beta * (1-relative_contact_rate) 


# In[ ]:


# N = 58000000
incubation_period = 5
print('Population: ',N)
beta1= beta
beta2 = beta
alpha = 1/incubation_period
epsilon = 0.2
delta = 1/lag_onset_death
zeta = 1/lag_onset_recovery
eta = 1/lag_onset_recovery
print('Initial Estimates')
print('beta1, beta2, alpha, epsilon, delta, zeta, eta, cCFR')
print('%.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f'%(beta1, beta2, alpha, epsilon, delta, zeta, eta, cCFR))
E_up = N - df_conf.iloc[0]
S_up = N - df_conf.iloc[0]
Cu_up = N - df_conf.iloc[0]

res = optimize.least_squares(resid_seir, [beta1, beta2, alpha, epsilon, delta, zeta, eta, cCFR], 
                                 bounds=([0.05, 0.01, 1/6.0, 0.01, 1/21, 1/25, 1/25, 0.03], \
                                        [0.25, 0.2, 1/4.0, .95, 1/10, 1/21, 1/21, 0.150]),\
                             loss='linear', xtol = 3e-16, ftol = 3e-16,\
                             args=(N, df_conf, df_reco, df_death, t0, IFR))
#beta1, beta2, alpha, epsilon, delta, zeta, eta, cCFR)
# try: guass newton or bfgs
# try calc gradient or hessian


# In[ ]:


### check parameters
res['x']


# In[ ]:


print('Final Estimates')
print('R0_1, R0_2, incubation period, epsilon, time_to_death, time_to_recovery, time_to_recovery, cCFR')
print('%.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f'      %(res['x'][0]/gamma, res['x'][1]/gamma, 1/res['x'][2],        res['x'][3], 1/res['x'][4], 1/res['x'][5], 1/res['x'][6], res['x'][7]))


# ### Run simulation using optimum parameters derived from fitting to data
# 
# Now we've done all the hard work to estimate the paramaters describing the disease spread we can run the model to create forcasts of what will happen looking forward.
# 
# Remember, there are improvements we'd need to make to get a really good fit. I've mentioned some of these as we've gone through. But let's see how it looks:

# In[ ]:


S, E, Ir, D, C, Iu, Cu, _ = setup_SEIR(res['x'][0], res['x'][1], res['x'][2], res['x'][3],
                                       res['x'][4], res['x'][5], res['x'][6], res['x'][7],
                                       N, df_conf, df_reco, df_death, t0, IFR,0)
# plt.plot(Ir)
distancing = 0
S_long, E_long, Ir_long, D_long, C_long, Iu_long, Cu_long, _ = setup_SEIR(res['x'][0]*(1-distancing), 
                                                                          res['x'][1]*(1-distancing), 
                                                                          res['x'][2], res['x'][3],
                                                                   res['x'][4], res['x'][5], res['x'][6], res['x'][7],
                                                                   N, df_conf, df_reco, df_death, t0, IFR,180, generating_curve=True)
plt.plot(Cu_long)


# In[ ]:


true_C = df_reco
true_D = df_death
true_R = true_C + true_D
true_Ir = df_conf- true_R#
get_ipython().run_line_magic('matplotlib', 'inline')

forward_period = len(df_conf)
plot_curves_seir(df_conf.index[0], forward_period, Ir, E, D, C, true_Ir, true_D, true_C, Iu, Iu_long, Ir_long, D_long)


# In[ ]:


print('Estimated Final Infection Mortality Rate is: %.3f'%(D_long[-1]/(C_long[-1]+D_long[-1]+Cu_long[-1])))
print('Estimated Final Observed Mortality Rate is: %.3f'%(D_long[-1]/(C_long[-1]+D_long[-1])))
print('Estimated Final Deaths are: %i'%(D_long[-1]))


# ## Final remarks
# 
# As this isn't the final version of our model, it's performance can definately be improved. We've made these fixes in newer versions but you can see if you can do the same to this notebook. We can see despite this we did manage to achieve a decent fit for deaths.
# 
# This model is very sensitive to the starting point and the assumptions here are chosen to work for the UK. If you want it to work for other countries, you'll have to find assumptions that work, or implement the V2 or V3 fixes I outlined in the notebook.
# 
# I hope you've found this interesting and informative. If you have any questions, please leave a comment.
# 
# If you're interested in joining the HELP project, please sign up here: https://links.quant-quest.com/helpproject
# 
# 
