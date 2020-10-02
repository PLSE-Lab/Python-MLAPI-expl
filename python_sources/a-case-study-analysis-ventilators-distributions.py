#!/usr/bin/env python
# coding: utf-8

# # A case study analysis: Ventilators necessity over Italy
# 
# **By**: Kaike Wesley Reis
# 
# **Contact**: kaikewesley@hotmail.com
# 
# ![corona illustration](https://user-images.githubusercontent.com/32513366/79788955-d3459900-831f-11ea-883c-2eba6c4dadfa.gif)
# 
# ## Task - Which populations have contracted COVID-19 and require ventilators? And when does this occur in their disease course?
# 
# But why ventilators are important? Well if you get the disease and present serious symptoms those equipaments would be necessary to help you breath. So it's important for a country to present a large number of ventilators (to prevent population death).
# 
# ## Modules

# In[ ]:


# Standard modules
import numpy as np
import warnings

# Data modules
import pandas as pd

# Modelling modules
from scipy import integrate
from scipy import optimize

# Plot modules
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.offline as pyo
import plotly.graph_objs as go

# Notebook commands
warnings.filterwarnings('ignore')
pyo.init_notebook_mode()
get_ipython().run_line_magic('matplotlib', 'inline')


# # Case Study: Italy
# ![italy-flag](https://user-images.githubusercontent.com/32513366/80289653-5eef6900-8716-11ea-9e12-4e3681b34a59.jpg)
# 
# ## Why Italy?
# At the beginning of the pandemic, many countries including european, did not consider the SARS-COV-2 virus to be harmful. Italy's situation can be considered as the first western shock and what has caused many countries to take action. Given this fact, I chose an in-depth study in Italy to show what the impacts would be on other countries if no action was taken.

# # Study assumptions
# 
# ## Ventilators
# Based in this reference from Imperial College:
# ![image](https://user-images.githubusercontent.com/32513366/80289717-d7562a00-8716-11ea-9c94-899bcb159bd0.png)
# 
# I assume that:
# - at least 80% (from 10%) of hospitalized pacients will require ventilators
# - 100% of ICU patients will require ventilators
# - I will not evalute different ventilators i.e. I will evaluate only required units
# - I will assume that if a pacient require a ventilator and doesn't have it, he dies.
# 
# Given that, I will considerer that **13%**(5% from ICU patients + 8% from hospitalized patients) of my infected cases will require ventilators.
# 
# 
# ## Mathematical approach
# In the next section I show my modelling approach. To understand my results, I present my major assumption: I assume a **static** approach for this case study, which means that my calculated parameters will not change over time. Basically my results here present a specific scenario:
# 
# ### What would happen if Italy doesn't take any action after ending of March
# 
# This is important to say because most of the world are doing a **dynamic** approach:
# - Measure the curve by chosen parameters
# - Make predictions to understand better the techniques they need to apply such as isolation or lockdown
# - Apply a technique and study if your situation has improved
# - Evaluate if your model start to show higher errors compared to reality
# - Restart the circle

# # SEIR Model Explanation
# 
# SEIR is a set (system) of ODE (Ordinary Differencial Equations) which models the dynamics of four main dependent variables over time during a pandemic:
# - **S(t)**: Suscetible cases - Population fraction that can get a disease
# - **E(t)**: Exposed cases - Population fraction that got the virus, but doesn't present any symptom
# - **I(t)**: Infected cases - As exposed, but presents any symptom
# - **R(t)**: Recovered cases - Population fraction that got infected and recovered after some time
# 
# The mathematical ODE system is presented bellow:
# ![seir](https://user-images.githubusercontent.com/32513366/80239696-101ed200-8637-11ea-8ace-918fb5d98076.PNG)
# 
# The dynamics between them are correlated and reminds SIR model. For more modelling information I strongly recommend the [MAA SIR reading](https://www.maa.org/press/periodicals/loci/joma/the-sir-model-for-spread-of-disease-the-differential-equation-model) and [Numberphile - The corona curve](https://www.youtube.com/watch?v=k6nLfCbAzgo) to understand SIR model (it helps to understand this model too) and a [SEIR Model doc](http://www.public.asu.edu/~hnesse/classes/seir.html).
# 
# In red, I specify the parameters that control SEIR curves:
# - **er**: exposed rate
# - **ir**: infection rate
# - **rr**: recovery rate
# 
# So when you hear:
# 
# ### "Flatten the Corona curve" 
# 
# The objective is to decrease **infection rate** to flatten the Infection curve. This is achieve by lockdown/isolation techniques.
# 
# A higher **recovery rate** is desired too, because the number of people in hospitals tends to decrease.
# 
# 
# This ODE system present your dependent variables (s, i, e and r) in lowercase. This statement is important because for my solution I evaluated fraction values related to population size (N):
# ![SEIR_CR](https://user-images.githubusercontent.com/32513366/80240551-65a7ae80-8638-11ea-8784-6edad80e0ba2.PNG)
# 
# ## But how I will develop this model?
# Given the fact that I have:
# - A model approach: SEIR
# - Initial curves for Infected and Recovered cases in each state
# 
# I will implement an **optimization technique** to find the actual parameters that minimize my error for both initial curves: i(t) and r(t).
# 
# ## Defined functions

# In[ ]:


# FUNCTION - Define SEIR Model ODE System
def seir_model_ode(y, t, params): 
    '''
    Arguments:
    - y: dependent variables
    - t: independent variable (time)
    - params: Model params
    '''
    # Parameters to find
    infection_rate = params[0]
    recovery_rate = params[1]
    exposed_rate = params[2]
    
    # Y variables
    s = y[0]
    e = y[1]
    i = y[2]
    r = y[3]
    
    # SEIR EDO System 
    dsdt = -exposed_rate*s*i
    dedt = (exposed_rate*s*i) - (infection_rate*e)
    didt = (infection_rate*e) - (recovery_rate*i)
    drdt = recovery_rate*i
    
    # Return our system
    return (dsdt, dedt, didt, drdt)


# In[ ]:


# FUNCTION - Calculate SEIR Model in t (time as days) based on given parameters
def calculate_seir_model(params, t, initial_condition):
    # Create an alias to our ode model to pass guessed params
    seir_ode = lambda y,t:seir_model_ode(y, t, params)
    
    # Calculate ode solution, return values to each
    ode_result = integrate.odeint(func=seir_ode, y0=initial_condition, t=t)
    
    # Return results
    return ode_result


# In[ ]:


# FUNCTION - Auxiliar function to find the best parameters
def fit_seir_model(params_to_fit, t, initial_condition, i_r_true):
    # Calculate ODE solution for possible parameter, return values to each dependent variable:
    # (s, e, i and r)
    fit_result = calculate_seir_model(params_to_fit, t, initial_condition)
    
    # Calculate residual value between predicted VS true
    ## Note: ode_result[0] is S result
    residual_i = i_r_true[0] - fit_result[:,2]
    residual_r = i_r_true[1] - fit_result[:,3]

    # Create a np.array of all residual values for both (i) and (r)
    residual = np.concatenate((residual_i, residual_r))
    
    # Return results
    return residual


# # Proposed approach
# I will present a step-by-step understanding approach:
# - **Data pre-processing**
#     - Here I imported my dataset and turned it more clean to work
# - **Exploratory Data Analysis** 
#     - Here I evaluated the most impacted region in Italy. This region will be used to present how exactly I made my model and got my results . Later the same methodology was extrapolated to other risk regions.
# - **Region Analysis**
#     - Apply SEIR Model to selected region
#     - Evaluate the results
# - **Extrapolating to risk regions**
#     - Apply SEIR Model now to other risk regions
# - **Overall results**
#     - Here I gave a conclusion to this notebook

# # Data pre-processing
# 
# ## Importing dataset and removing unused columns

# In[ ]:


# Importing dataset
df_ita_status = pd.read_csv('/kaggle/input/uncover/UNCOVER/github/covid-19-italy-situation-monitoring-by-region.csv')

# Removing columns
df_ita_status.drop(axis=1, inplace=True, columns=['stato', 'codice_regione', 'lat', 'long', 'totale_positivi', 'isolamento_domiciliare','variazione_totale_positivi', 'nuovi_positivi', 'tamponi', 'note_it', 'note_en'])

# Change column names (turn my life easier)
df_ita_status.columns = ['date', 'state', 'hospitalized_with_symptom', 'icu', 'hospitalized_total', 'recovered', 'deaths', 'infected']

# Create a list of Italy states
ita_states = list(df_ita_status['state'].unique())

# Visualize dataset
df_ita_status


# # Exploratory Data Analysis
# 
# ## Hospitalization per region through time

# In[ ]:


# Define figure and plot!
fig = px.line(df_ita_status, x='date', y='hospitalized_total', color='state')
pyo.iplot(fig)


# ## Death per region through time

# In[ ]:


# Define figure and plot!
fig2 = px.line(df_ita_status, x='date', y='deaths', color='state')
pyo.iplot(fig2)


# ## Infected per region through time

# In[ ]:


# Define figure and plot!
fig3 = px.line(df_ita_status, x='date', y='infected', color='state')
pyo.iplot(fig3)


# ## EDA - Conclusions
# Although my plots refer to different variables:
# - number of deaths
# - number of hospitalized total (includes ICU)
# - number of infected
# 
# The result is apparently similar: **Lombardia** is the most serious region in Italy followed by Emilia-Romagna, Veneto and Piemonte. The last three appear to be at the beginning of the dispersion, so the **recommendation would be immediate apply isolation/lockdown in those regions** to ensure that the country's efforts are aimed at the **Lombardia** region. That situation would remind how China faced COVID in one specific region: Wuhan. 
# 
# The data related to **Lombardia** show a situation well known to those who follow Italian scenario in the newspapers:
# - [Coronavirus Italy: Lombardy province at centre of outbreak offers glimmer of hope](https://www.theguardian.com/world/2020/apr/08/coronavirus-italy-lombardy-province-at-centre-of-outbreak-offers-glimmer-of-hope)
# - [Coronavirus patients line the corridors in footage from inside Italian hospital, as military trucks transport scores of victims' coffins to be cremated](https://www.dailymail.co.uk/news/article-8129959/Military-trucks-transport-Italian-coronavirus)
# 
# The last became an emblematic and sad scene during this this pandemic.
# 
# The map bellow, show all Italy regions:
# 
# ![italy-regions-map2](https://user-images.githubusercontent.com/32513366/80289533-56e2f980-8715-11ea-87b1-a3ffc4390210.png)
# 
# In red is **Lombardia** and in purple the others three risk regions pointed by me that shows an ascending infected cases. This clearly shows that COVID infection started there (Lombardia) and spread to other regions given borders proximity.
# 

# # Region Analysis
# 
# ## Set region dataframe
# 
# As I concluded in EDA, my selected region will be: **Lombardia**

# In[ ]:


# Get Lombardia region
rgn_status = df_ita_status[df_ita_status['state'] == 'Lombardia'][['date', 'recovered', 'infected']].reset_index().drop(axis=1, columns=['index'])


# ## Define region population
# 
# According to this [source](https://www.citypopulation.de/en/italy/admin/03__lombardia/) I defined Lombardia N population as:

# In[ ]:


N = 10060574


# ## Define model start conditions
# This is necessary for any ODE solving problem. For each dependent variable I need a start condition, that I present here:
# 
# #### r(0) = r(0)
# #### i(0) = i(0)
# #### e(0) = i(5) - i(4)
# #### s(0) = 1 - i(0) - e(0)
# 
# Special attention for **e(0)** and **s(0)**. Acording to WHO (World Health Organization), the [COVID average incubation period is **5 days**](https://www.who.int/news-room/q-a-detail/q-a-coronaviruses#:~:text=The%20%E2%80%9Cincubation%20period%E2%80%9D,more%20data%20become%20available.). I used this info to set **e(0)**. For **s(0)** I just remove the population fraction exposed and infected.

# In[ ]:


# Define Initial Condition (necessary for ODE solve)
R_start = rgn_status.loc[0, 'recovered']/N
I_start = rgn_status.loc[0, 'infected']/N
E_start = (rgn_status.loc[4, 'infected'] - rgn_status.loc[3, 'infected'])/N
S_start = 1 - E_start - I_start
# Set this values as a tuple
ic = (S_start, E_start, I_start, R_start)


# In[ ]:


print('Start condition:')
print('s(0): ', ic[0])
print('e(0): ', ic[1])
print('i(0): ', ic[2])
print('r(0): ', ic[3])


# ## Prepare useful information for optimization

# In[ ]:


# Define a time array measure in days, but with values
time_opt = range(0, len(rgn_status))

# Create a tuple with the true values in fraction for Infected/Recovered cases (necessary for error measurement)
i_r_true = (list(rgn_status['infected']/N), list(rgn_status['recovered']/N))

# Define a start guess for our parameters [infection_rate, recovered rate, exposed rate]
params_start_guess = [0.1, 0.01, 0.1]


# ## Optimization process

# In[ ]:


optimal_params, sucess = optimize.leastsq(fit_seir_model,
                                          x0=params_start_guess,
                                          args=(time_opt, ic, i_r_true),
                                          ftol=1.49012e-20)


# ## Lombardia parameters

# In[ ]:


print('## Lombardia')
print('Optimize infection rate: ', optimal_params[0])
print('Optimize recovered rate: ', optimal_params[1])
print('Optimize exposed rate: ', optimal_params[2])


# ## Compare the original curve (true) and predicted curve (pred)

# In[ ]:


# Get the optimal parameters
ir = optimal_params[0]
rr = optimal_params[1]
er = optimal_params[2]


# In[ ]:


# Calculate a curve based on those parameters
fit_result = calculate_seir_model((ir, rr, er), time_opt, ic)


# In[ ]:


# Plot the results for Infected/Recovered
## Define plot object
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20,10))
## Plot process
axes[0].plot(time_opt, i_r_true[0], 'g')
axes[0].plot(time_opt, fit_result[:,2], 'y')
axes[0].legend(['Ground truth', 'Predicted'],loc=2, fontsize=15)
axes[0].set_title('Infected cases - Lombardia',fontsize=20)
axes[1].plot(time_opt, i_r_true[1], 'g')
axes[1].plot(time_opt, fit_result[:,3], 'y')
axes[1].legend(['Ground truth', 'Predicted'],loc=2, fontsize=15)
axes[1].set_title('Recovered cases - Lombardia',fontsize=20);


# ### Commentaries
# As we can see the optimization found a good approximation. I said good because wasn't perfect for some reasons:
# - It's a epidemiological model, so will not be perfect.
# - I measure my residual value (error to minimize) as a LS (Least Square) over a residual for each point in both curves. This was made because I didn't found a better solution for this problem (minize two cost functions at the same time) using ```scipy``` module.
# 
# Besides that, the model shows the same dynamic as the original curves which is a good sign.
# 
# ## SEIR predictions until july for Lombardia
# Let's see this model predictions until July.

# In[ ]:


# Get prediction full period time in datetime object and the convert to string
datetime_pred = pd.date_range(start="2020-02-24",end="2020-07-31", freq='D')
time_pred = [x.strftime("%Y-%m-%d") for x in datetime_pred]

# Get a list from 01/April to 31/July 
time_pred_range = range(0, len(time_pred))


# In[ ]:


# Calculate a SEIR prediction 
future_pred = calculate_seir_model((ir, rr, er), time_pred_range, ic)


# In[ ]:


# Plot results
## Define Date axis to better visualization (only first/half/last day of every month)
time_axis = [time_pred[i] for i in [0,6,20,37,51,67,81,98,112,128,142,158]]
## Define plot object
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(20,10))
## Plot SEIR
sns.lineplot(x=time_pred, y=future_pred[:,0], ax=axes, color = 'blue')
sns.lineplot(x=time_pred, y=future_pred[:,1], ax=axes, color = 'red')
sns.lineplot(x=time_pred, y=future_pred[:,2], ax=axes, color = 'purple')
sns.lineplot(x=time_pred, y=future_pred[:,3], ax=axes, color = 'green')
plt.setp(axes.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
axes.legend(loc=1, labels=['Suscetible', 'Exposed', 'Infected','Recovered'], fontsize=10)
axes.set_title('LOMBARDIA - SEIR predictions', fontsize=20)
axes.set_xlabel('Date', fontsize=15)
axes.set_ylabel('Total cases', fontsize=15)
axes.set_xticks(time_axis);


# ### Commentaries:
# - One point to notice: my results are in 0 to 1 interval which means that I'm measuring in population % of N
# - Based in those results, Lombardia would achieve a spike in **infected curve** around 05/15 with a portion of ~60% infected!
# - You can see that **exposed curve** is pretty low. This is possible given the EDO formula and lower calculated parameter. This could mean that infection was already too strong (given the higher infection rate compared to exposed rate) and most of the population passed from exposed to infected **before 02/24**. Other possible reason is that my optimization is focused in recovery and infected curve.
# 
# ## Ventilators necessity
# 
# Based in **Study assumptions**, my ventilators necessity will be **13%** of my **infected curve**. Let's calculate this curve.

# In[ ]:


# Calculate ventilators curve based in SEIR infected curve
future_pred_vent = 0.13*future_pred[:,2]


# In[ ]:


# Plot results
## Define plot object
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(20,10))
## Plot SIR
sns.lineplot(x=time_pred, y=future_pred_vent, ax=axes, color = 'red')
sns.lineplot(x=time_pred, y=future_pred[:,2], ax=axes, color = 'purple')
plt.setp(axes.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
axes.legend(loc=1, labels=['Ventilators necessity curve','Infected curve'], fontsize=10)
axes.set_title('Lombardia', fontsize=20)
axes.set_xlabel('Date', fontsize=15)
axes.set_ylabel('Total count', fontsize=15)
axes.set_xticks(time_axis);


# ## Evaluate the maximum value in Ventilators necessity curve
# Given the fact that I'm working with a cumulative curve, the maximum value would show my higher ventilator necessity to that region:

# In[ ]:


# Get the maximum curve value and transform to absolute value multiplying by region population
max_vent_necessity = N*max(future_pred_vent)
# Show results
print('Lombardia would need: ', int(max_vent_necessity),'ventilators.')


# Well, in this scenario we have a strong number for only **one region**.

# # Extrapolating to risk regions
# Here I did the same process presented in the last topic for other risk regions:
# - Emilia-Romagna
# - Piemonte
# - Veneto
# 
# Through a magical function that does everything automatically! I evaluated each ventilator curve for each region until July to present an general overview for this country!
# 
# **PS**: Here I assume that my process is accurate given **Lombardia** analysis, so I didn't evaluate each comparison between predicted curve and original curve!

# ## Find N population for each state
# First I need to prepare a list of population for each risk state:
# 
# **PS**: My source was the [google result after placing 'state population'](https://www.google.com/)

# In[ ]:


# Italy risk regions
ita_risk_states = ['Veneto', 'Piemonte', 'Emilia-Romagna']
# Population for each risk region
pop_states = [4906000.0, 4356000.0, 4459000.0] 


# In[ ]:


# Magical function
def extrapolating_italy(population_in_states=pop_states,states_in_italy=ita_risk_states):
    '''
    Function to return a list of tuples: (state_name, infection_rate, recovery_rate, exposed_rate, state_ventilators_curve)
    '''
    # Create a return tuple
    result = list()
    # Loop for each state
    for state, population in zip(states_in_italy, population_in_states):
        # Get a region
        rgn_status = df_ita_status[df_ita_status['state'] == state][['date', 'recovered', 'infected']].reset_index(
        ).drop(axis=1, columns=['index'])
        # Set N population value
        N = population
        # Find a pandemic start where at least recovered or infected have one value!
        for i in range(0, len(rgn_status)):
            if rgn_status.loc[i, 'recovered'] != 0.0 and rgn_status.loc[i, 'infected'] != 0.0:
                index = i
                break
            else:
                index = 0
        # Define Initial Condition (necessary for ODE solve)
        R_start = rgn_status.loc[index, 'recovered']/N
        I_start = rgn_status.loc[index, 'infected']/N
        E_start = (rgn_status.loc[index + 4, 'infected'] - rgn_status.loc[index + 3, 'infected'])/N
        S_start = 1 - E_start - I_start
        # Set this values as a tuple for initial condition
        ic = (S_start, E_start, I_start, R_start)
        # Define a time array measure in days, but with values
        time_opt = range(index, len(rgn_status))
        # Create a tuple with the true values in fraction for Infected/Recovered cases (necessary for error measurement)
        i_r_true = (list(rgn_status.loc[index:, 'infected']/N), list(rgn_status.loc[index:, 'recovered']/N))
        # Define a start guess for our parameters [infection_rate, recovered rate, exposed rate]
        params_start_guess = [0.1, 0.01, 0.1]
        # Optimization
        optimal_params, sucess = optimize.leastsq(fit_seir_model,
                                                  x0=params_start_guess,
                                                  args=(time_opt, ic, i_r_true),
                                                  ftol=1.49012e-15,  maxfev=10000)
        # Get calculated parameters
        ir = optimal_params[0]
        rr = optimal_params[1]
        er = optimal_params[2]
        # Get prediction full period time in datetime object and the convert to string
        datetime_pred = pd.date_range(start="2020-02-24",end="2020-07-31", freq='D')
        time_pred = [x.strftime("%Y-%m-%d") for x in datetime_pred]
        # Get a list from 01/April to 31/July 
        time_pred_range = range(0, len(time_pred))
        # Calculate a SEIR prediction 
        future_pred = calculate_seir_model((ir, rr, er), time_pred_range, ic)
        # Generate tuple result for this state
        state_result = (state, ir, rr, er, 0.13*future_pred[:,2])
        # Append to my list
        result.append(state_result)
    # Return
    return result    


# In[ ]:


# Get results for each region
italy_general_results = extrapolating_italy()


# So now I have a list where each element is a tuple of: 
# 
# ```(state_risk_name, infection rate, recovery rate, exposed rate, infected curve, ventilators curve)```
# 
# ## Plot predicted Ventilators curves for each risk region

# In[ ]:


# Create legend
legend_state = ['Lombardia', 'Veneto', 'Piemonte', 'Emilia-Romagna']
# Plot results
## Define plot object
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(20,10))
## Plot first Lombardia curve
sns.lineplot(x=time_pred, y=future_pred_vent, ax=axes)
## Plot SEIR
for state in italy_general_results:
    state_name = state[0]
    vent_curve = state[4]
    sns.lineplot(x=time_pred, y=vent_curve, ax=axes)
plt.setp(axes.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
axes.legend(loc=0, labels=legend_state, fontsize=15)
axes.set_title('Predicted Ventilators curve for Risk regions in Italy', fontsize=25)
axes.set_xlabel('Date', fontsize=20)
axes.set_ylabel('Total count', fontsize=20)
axes.set_xticks(time_axis);


# ### Commentary:
# - The most meaningful observation that you can get from this plot is that each spike for 4 risk regions in Italy will occur between 05/15 and 06/01.
# - Piemonte present a more flatten curve.
# - Lombardia actual state would present the worst scenario.
# 
# ## Maximum ventilators necessity for each region

# In[ ]:


# Define a sum to count all ventilators for those regions
sum_italy_vents = 0
# Print each Ventilator necessity for each reagion
print('# Risk regions')
print('## Lombardia')
print('ventilators units: ', int(max_vent_necessity))
sum_italy_vents += int(max_vent_necessity)
for i, N in zip(italy_general_results, pop_states):
    print('## ', i[0])
    print('ventilators units: ', int(max(i[4]*N)))
    sum_italy_vents += int(max(i[4]*N))
    
# Print final result
print('\nIn total for this Scenario, Italy would need for only 4 regions: ', sum_italy_vents, ' ventilators.')


# ### Last commentary:
# In this scenario my recommendation would be to 04/01:
# - Lockdown Lombardia region as China made with Wuhan
# - Doing this would prevent Veneto, Piemonte and Emilia-Romagna to shows a worst spread
# - Move all resources (including ventilators) to Lombardia
# 
# **Summing up**: Prevent Lombardia to be a spread origin to the country!

# # Overall results
# 
# This analysis shows that keeping this scenario i.e. doing nothing to prevent covid spread, probably only in risk regions in Italy (4 of 20 regions) would need a ventilators quantity that probably doesn't exist in this world! It's important to see here the necessity to fight this virus!
