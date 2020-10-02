#!/usr/bin/env python
# coding: utf-8

# # COVID-19 "study" with epidemiology models (WIP)
# 
# * **READ HERE FIRST: This is not an official study! This is very unlike to represent real scenarios! I'm not specialist in Epidemiology! This is a very simple demonstration about how to handle the models!**
# 
# * **READ EVEN MORE: Real studies and predictions should be performed by teams with multiple specialities. COVID-19 is a real thing, don't propagate results and data as conclusive if YOU ARE NOT A SPECIALIST IN THE FIELD.**
# 
# ## Table of Contents
# 
# 1. [Importing libs](#importing)
# 
# 2. [Loading data](#loading)
# 
# 3. [Data cleaning](#cleaning)
# 
# 4. [(Very) Basic EDA](#eda)
# 
# 5. [Epidemiology models](#models)
# 
# 6. [Programming SIR/SEIR-based models in Python](#implementations)
# 
# 7. [Least-squares fitting](#least-squares)
# 
# 8. [Extrapolation/Predictions](#deterministic-predictions)
# 
# 9. [Bayesian Calibration](#bayes-calibration)
# 
#     * [SIR model](#bayes-sir)
#     * [Modified SEIR model](#bayes-seir2)
# 
# This notebook presents a simple "study", more like as an exercise, of how a data conciliation can be performed on Epidemiology models. This kind of approach can be adapted for other cases that have Dynamical Systems as mathematical model.
# 
# Classical models are analyzed:
# 
# * SIR (Susceptible-Infected-Recovered) model;
# * SEIR (Susceptible-Exposed-Infected-Recovered) model;
# 
# New modifications are proposed in order to represent better features that are present in COVID-19 spreading. The modified models are:
# 
# * SIRD (Susceptible-Infected-Recovered-Dead) model, which considers the disease mortality rate explicity;
# * SEIR-2 model, which is a modification of SEIR that take into account the fact that exposed individuals without symptoms can transmit the disease to susceptible individuals during incubation time;
# * SEIRD model. This model is just SEIR-2 model, but considering the sub-population of Dead individuals;
# * SEIRD-Q model. This is the most ambitious model here. This model try to model, in some sense, the effect in the population dynamics related to removal due to quarantine.
# 
# *New models will be devised in the next few days. Note that these models are "local" models, so it must be applied to territorial area for cities. It will be adjusted latter to take it into account.*
# 
# Before analyze the models, we begin having a look at the available data.
# 
# **P.S.: This is just a study related to computational aspects. I'm not an epidemiologist. COVID-19 is a REAL issue! This notebook can not be regarded as professional advices in any sense except for, maybe, as a computational analysis. Maybe these models, or some idea, can help someone else that can really face this issue.**

# <a id="importing"></a>
# ## Importing libs

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pymc3 as pm # for uncertainty quantification and model calibration
from scipy.integrate import solve_ivp # to solve ODE system
from scipy import optimize # to solve minimization problem from least-squares fitting
from numba import jit # to accelerate ODE system RHS evaluations
import theano # to control better pymc3 backend and write a wrapper
import theano.tensor as t # for the wrapper to a custom model to pymc3

# Plotting libs
import matplotlib.pyplot as plt
import altair as alt

seed = 12345 # for the sake of reproducibility :)
np.random.seed(seed)

plt.style.use('seaborn-talk') # beautify the plots!

THEANO_FLAGS='optimizer=fast_compile' # A theano trick


# <a id="loading"></a>
# ## Loading data

# In[ ]:


df_covid = pd.read_csv("../input/corona-virus-report/covid_19_clean_complete.csv", parse_dates=['Date'])

df_covid.info()


# In[ ]:


df_covid.head()


# In[ ]:


columns_to_filter_cases = ["Country/Region", "Date", "Confirmed", "Deaths"]
df_covid_cases = df_covid[columns_to_filter_cases]

df_covid_cases.head()


# <a id="cleaning"></a>
# ## Data cleaning
# 
# Let's do a data cleaning based in this [amazing notebook!](https://www.kaggle.com/abhinand05/covid-19-digging-a-bit-deeper)

# In[ ]:


print(f"First day entry:\t {df_covid['Date'].min()}")
print(f"Last day reported:\t {df_covid['Date'].max()}")
print(f"Total of tracked days:\t {df_covid['Date'].max() - df_covid['Date'].min()}")


# In[ ]:


df_covid.rename(
    columns={
        'Date': 'date', 
        'Province/State':'state',
        'Country/Region':'country',
        'Last Update':'last_updated',
        'Confirmed': 'confirmed',
        'Deaths':'deaths',
        'Recovered':'recovered'}, 
    inplace=True
)

df_covid


# Active Case = confirmed - deaths - recovered

# In[ ]:


df_covid['active'] = df_covid['confirmed'] - df_covid['deaths'] - df_covid['recovered']

df_covid


# Replacing Mainland china with just China:

# In[ ]:


df_covid['country'] = df_covid['country'].replace('Mainland China', 'China')

df_covid


# <a id="eda"></a>
# ## (Very) Basic EDA

# Worldwide scenario:

# In[ ]:


df_grouped = df_covid.groupby('date')['date', 'confirmed', 'deaths'].sum().reset_index()

df_grouped


# In[ ]:


confirmed_plot = alt.Chart(df_grouped).mark_circle(size=60, color='blue').encode(
    x=alt.X('date', axis=alt.Axis(title='Date')),
    y=alt.Y('confirmed', axis=alt.Axis(title='Cases'))
)

deaths_plot = alt.Chart(df_grouped).mark_circle(size=60, color='red').encode(
    x='date',
    y='deaths'
)

worldwide_plot = confirmed_plot + deaths_plot
worldwide_plot.interactive()


# Now, let's take a look at Brazil:

# In[ ]:


def get_df_country_cases(df: pd.DataFrame, country_name: str) -> pd.DataFrame:
    df_grouped_country = df[df['country'] == country_name].reset_index()
    df_grouped_country_date = df_grouped_country.groupby('date')['date', 'confirmed', 'deaths'].sum().reset_index()
    df_grouped_country_date["confirmed_marker"] = df_grouped_country_date.shape[0] * ['Confirmed']
    df_grouped_country_date["deaths_marker"] = df_grouped_country_date.shape[0] * ['Deaths']
    return df_grouped_country_date


# In[ ]:


def altair_plot_for_confirmed_and_deaths(df_grouped: pd.DataFrame, data_at_x_axis: str='date') -> alt.Chart:
    confirmed_plot = alt.Chart(df_grouped).mark_circle(size=60).encode(
        x=alt.X(data_at_x_axis, axis=alt.Axis(title='Date')),
        y=alt.Y('confirmed', axis=alt.Axis(title='Cases'), title='Confirmed'),
        color=alt.Color("confirmed_marker", title="Cases"),
    )

    deaths_plot = alt.Chart(df_grouped).mark_circle(size=60).encode(
        x=data_at_x_axis,
        y='deaths',
        color=alt.Color("deaths_marker"),
    )

    return confirmed_plot + deaths_plot


# In[ ]:


df_grouped_brazil = get_df_country_cases(df_covid, "Brazil")

df_grouped_brazil


# In[ ]:


altair_plot_for_confirmed_and_deaths(df_grouped_brazil).interactive()


# Let's have a look at China:

# In[ ]:


df_grouped_china = get_df_country_cases(df_covid, "China")

df_grouped_china


# In[ ]:


altair_plot_for_confirmed_and_deaths(df_grouped_china).interactive()


# Now a look at Italy:

# In[ ]:


df_grouped_italy = get_df_country_cases(df_covid, "Italy")

df_grouped_italy


# In[ ]:


altair_plot_for_confirmed_and_deaths(df_grouped_italy).interactive()


# ### Comparison between Brazil and Italy

# In[ ]:


df_brazil_cases_by_day = df_grouped_brazil[df_grouped_brazil.confirmed > 0]
df_brazil_cases_by_day = df_brazil_cases_by_day.reset_index(drop=True)
df_brazil_cases_by_day['day'] = df_brazil_cases_by_day.date.apply(lambda x: (x - df_brazil_cases_by_day.date.min()).days)

reordered_columns = ['date', 'day', 'confirmed', 'deaths', 'confirmed_marker', 'deaths_marker']
df_brazil_cases_by_day = df_brazil_cases_by_day[reordered_columns]

df_brazil_cases_by_day


# In[ ]:


df_italy_cases_by_day = df_grouped_italy[df_grouped_italy.confirmed > 0]
df_italy_cases_by_day = df_italy_cases_by_day.reset_index(drop=True)
df_italy_cases_by_day['day'] = df_italy_cases_by_day.date.apply(lambda x: (x - df_italy_cases_by_day.date.min()).days)

reordered_columns = ['date', 'day', 'confirmed', 'deaths', 'confirmed_marker', 'deaths_marker']
df_italy_cases_by_day = df_italy_cases_by_day[reordered_columns]

df_italy_cases_by_day


# In[ ]:


df_italy_cases_by_day_limited_by_br = df_italy_cases_by_day[df_italy_cases_by_day.day <= df_brazil_cases_by_day.day.max()]
days = df_brazil_cases_by_day.day

plt.figure(figsize=(9, 6))
plt.plot(days, df_brazil_cases_by_day.confirmed, marker='x', linestyle="", markersize=10, label='Confirmed (BR)')
plt.plot(days, df_italy_cases_by_day_limited_by_br.confirmed, marker='o', linestyle="", markersize=10, label='Confirmed (ITA)')
plt.plot(days, df_brazil_cases_by_day.deaths, marker='s', linestyle="", markersize=10, label='Deaths (BR)')
plt.plot(days, df_italy_cases_by_day_limited_by_br.deaths, marker='*', linestyle="", markersize=10, label='Deaths (ITA)')

plt.xlabel("Day(s)")
plt.ylabel("Cases")
plt.legend()
plt.grid()

plt.show()


# ### China scenario since first entry

# In[ ]:


df_china_cases_by_day = df_grouped_china[df_grouped_china.confirmed > 0]
df_china_cases_by_day = df_china_cases_by_day.reset_index(drop=True)
df_china_cases_by_day['day'] = df_china_cases_by_day.date.apply(lambda x: (x - df_china_cases_by_day.date.min()).days)

reordered_columns = ['date', 'day', 'confirmed', 'deaths', 'confirmed_marker', 'deaths_marker']
df_china_cases_by_day = df_china_cases_by_day[reordered_columns]

df_china_cases_by_day


# ### Spain since first recorded case

# In[ ]:


df_grouped_spain = get_df_country_cases(df_covid, "Spain")
df_spain_cases_by_day = df_grouped_spain[df_grouped_spain.confirmed > 0]
df_spain_cases_by_day = df_spain_cases_by_day.reset_index(drop=True)
df_spain_cases_by_day['day'] = df_spain_cases_by_day.date.apply(lambda x: (x - df_spain_cases_by_day.date.min()).days)

reordered_columns = ['date', 'day', 'confirmed', 'deaths', 'confirmed_marker', 'deaths_marker']
df_spain_cases_by_day = df_spain_cases_by_day[reordered_columns]

df_spain_cases_by_day


# ### Iran since first case

# In[ ]:


df_grouped_iran = get_df_country_cases(df_covid, "Iran")
df_iran_cases_by_day = df_grouped_iran[df_grouped_iran.confirmed > 0]
df_iran_cases_by_day = df_iran_cases_by_day.reset_index(drop=True)
df_iran_cases_by_day['day'] = df_iran_cases_by_day.date.apply(lambda x: (x - df_iran_cases_by_day.date.min()).days)

reordered_columns = ['date', 'day', 'confirmed', 'deaths', 'confirmed_marker', 'deaths_marker']
df_iran_cases_by_day = df_iran_cases_by_day[reordered_columns]

df_iran_cases_by_day


# ### USA since first case

# In[ ]:


df_grouped_usa = get_df_country_cases(df_covid, "US")
df_usa_cases_by_day = df_grouped_usa[df_grouped_usa.confirmed > 0]
df_usa_cases_by_day = df_usa_cases_by_day.reset_index(drop=True)
df_usa_cases_by_day['day'] = df_usa_cases_by_day.date.apply(lambda x: (x - df_usa_cases_by_day.date.min()).days)

reordered_columns = ['date', 'day', 'confirmed', 'deaths', 'confirmed_marker', 'deaths_marker']
df_usa_cases_by_day = df_usa_cases_by_day[reordered_columns]

df_usa_cases_by_day


# <a id="models"></a>
# ## Epidemiology models
# 
# Now, let me explore the data in order to calibrate an epidemiologic model in order to try to simulate and predict cases.
# 
# ### Classical models
# 
# Here I present a brief review of classical temporal models (space dependency is not considered). Then I proposed modifications for such models.
# 
# #### SIR model
# 
# The model represents an epidemic scenario, aiming to predict and control infectious diseases. It consists in a non-linear dynamical system, which considers populational sub-groups according to the state of the individuals. A simple model would be composed by 3 subgroups:
# 
# * Susceptible individuals (S);
# * Infected (I);
# * Recovered (R).
# 
# With such components, a classical dynamical system known as SIR model. The equations of such a system is written as:
# 
# \begin{align*}
#   \dot{S} &= - \beta S I \\ 
#   \dot{I} &= \beta S I - \zeta I \\ 
#   \dot{R} &= \zeta I
# \end{align*}
# 
# where $\dot{(\bullet)}$ stands for time-derivative.
# 
# Some biological explanation for parameters:
# 
# * $\beta$ is the conversion parameter due to interaction between a susceptible individual with an infected one;
# * $\zeta$ is the conversion parameter related to the recovery rate. In other words, the individuals that become immune;
# 
# #### SEIR model
# 
# Another classical model known as SEIR (Susceptible-Exposed-Infected-Recovered) is common applied in Computational Epidemiology literature (you can check it elsewhere). In this model, a new sub-group of individuals is considered: Exposed. Such individuals are those that are infected, but don't show any sympton. In the classical SEIR model, exposed individuals **do not transmit the disease**. The ODE system now becomes:
# 
# \begin{align*}
#     \dot{S} &= - \beta S  I \\
#     \dot{E} &= \beta S I - \alpha E \\
#     \dot{I} &= \alpha E - \zeta I \\
#     \dot{R} &= \zeta I \\
# \end{align*}
# 
# Brief biological interpretation for additional parameter:
# 
# * $\alpha$ is the conversion parameter for exposed individuals that transformed into infected ones.
# 
# ### Modified models
# 
# Here, I propose some simple modifications in order to improve model representability for COVID-19.
# 
# #### Modified SIR model (SIRD)
# 
# In this model, deaths due to the disease is considered explicitly. A new individuals sub-group is introduced: dead individuals. To consider such phenomenon, an additional equation is required, as well as a modification in the Infected equation balance. The ODE system is given below:
# 
# \begin{align*}
#   \dot{S} &= - \beta S I \\ 
#   \dot{I} &= \beta S I - \zeta I - \delta I \\ 
#   \dot{R} &= \zeta I \\
#   \dot{D} &= \delta I
# \end{align*}
# 
# Brief biological interpretation for additional parameter:
# 
# * $\delta$ is the mortality rate for the disease.
# 
# #### Modified SEIR model (SEIR-2)
# 
# This model aims to solve the lack of the original SEIR model, which does not consider disease transmission between exposed and susceptible individuals. In order to take it into account,
# we modified balance equations for S and E as follows:
# 
# \begin{align*}
#     \dot{S} &= - \beta S  I  - \gamma S E \\
#     \dot{E} &= \beta S I - \alpha E + \gamma S E \\
#     \dot{I} &= \alpha E - \zeta I \\
#     \dot{R} &= \zeta I \\
# \end{align*}
# 
# Brief biological interpretation for additional parameter:
# 
# * $\gamma$ is the conversion rate parameter for susceptible individuals that interact with exposed individuals and then become exposed.
# 
# #### Modified SEIR model with deaths (SEIRD)
# 
# Very similiar to the last one, but it considers a sub-population of dead individuals due to the disease. Thus, the model is written as:
# 
# \begin{align*}
#     \dot{S} &= - \beta S  I  - \gamma S E \\
#     \dot{E} &= \beta S I - \alpha E + \gamma S E \\
#     \dot{I} &= \alpha E - \zeta I - \delta I \\
#     \dot{R} &= \zeta I \\
#     \dot{D} &= \delta I
# \end{align*}
# 
# #### Modified SEIRD model considering quarantine lockdown (SEIRD-Q)
# 
# This is a modified model that take into account a removal rate from Susceptible, Exposed and Infected individuals to quarantine. The main hypothesis is that this conversion
# is under a constant removal parameter (by time, i.e., 1 / time), and after the conversion, the individual becomes "Recovered" and can not transmit the disease anymore. The new system is written as
# 
# \begin{align*}
#     \dot{S} &= - \beta S  I  - \gamma S E - \omega S\\
#     \dot{E} &= \beta S I - \alpha E + \gamma S E - \omega E \\
#     \dot{I} &= \alpha E - \zeta I - \delta I - \omega I \\
#     \dot{R} &= \zeta I + \omega (S + E + I) \\
#     \dot{D} &= \delta I
# \end{align*}
# 
# Brief biological interpretation for additional parameter:
# 
# * $\omega$ is the conversion rate parameter for Susceptible, Exposed and Infected individuals that becomes Recovered due to a removal to a quarantine.
# 
# ### Remarks for the models units
# 
# All sub-population variables (S, I, R, etc) are dimensionless. To obtain the variables, we have to consider that
# 
# \begin{align*}
#     &S := \frac{\mathcal{S}}{N} \\
#     &E := \frac{\mathcal{E}}{N} \\
#     &I := \frac{\mathcal{I}}{N} \\
#     &R := \frac{\mathcal{R}}{N} \\
#     &D := \frac{\mathcal{D}}{N} \\
# \end{align*}
# 
# with $N$ denoting the total population and $\mathcal{S}$, $\mathcal{E}$, $\mathcal{I}$, $\mathcal{R}$ and $\mathcal{D}$ as the absolute sub-population amounts. Therefore, S, E, I, R and D are given as fractions of the total population.

# <a id="implementations"></a>
# ## Programming SIR/SEIR-based models in Python

# In[ ]:


@jit(nopython=True)
def sir_model(t, X, beta=1, zeta=1/15):
    S, I, R = X
    S_prime = - beta * S * I
    I_prime = beta * S * I - zeta * I
    R_prime = zeta * I
    return S_prime, I_prime, R_prime


@jit(nopython=True)
def sird_model(t, X, beta=1, delta=0.02, zeta=1/15):
    """
    SIR model that takes into account the number of deaths.
    """
    S, I, R, D = X
    S_prime = - beta * S * I
    I_prime = beta * S * I - zeta * I - delta * I
    R_prime = zeta * I
    D_prime = delta * I
    return S_prime, I_prime, R_prime, D_prime


@jit(nopython=True)
def seir2_model(t, X, alpha=1/5, beta=1, gamma=0, zeta=1/15, delta=0.02):
    """
    This is a modified SEIR model in order to take into account incubation time in exposed individual.
    The exposed individuals can transmit the infection to susceptible individuals.
    """
    S, E, I, R = X
    S_prime = - beta * S * I - gamma * E * S
    E_prime = beta * S * I - alpha * E + gamma * E * S
    I_prime = alpha * E - zeta * I - delta * I
    R_prime = zeta * I
    return S_prime, E_prime, I_prime, R_prime


@jit(nopython=True)
def seird_model(t, X, alpha=1/5, beta=1, gamma=0, zeta=1/15, delta=0.02):
    """
    A modified SEIR model in order to take into account deaths.
    """
    S, E, I, R, D = X
    S_prime = - beta * S * I - gamma * E * S
    E_prime = beta * S * I - alpha * E + gamma * E * S
    I_prime = alpha * E - zeta * I - delta * I
    R_prime = zeta * I
    D_prime = delta * I
    return S_prime, E_prime, I_prime, R_prime, D_prime


@jit(nopython=True)
def seirdq_model(t, X, alpha=1/5, beta=1, gamma=0, omega=0, zeta=1/15, delta=0.02):
    """
    A modified SEIRD model in order to take into account quarantine.
    """
    S, E, I, R, D = X
    S_prime = - beta * S * I - gamma * E * S - omega * S
    E_prime = beta * S * I - alpha * E + gamma * E * S - omega * E
    I_prime = alpha * E - zeta * I - delta * I - omega * I
    R_prime = zeta * I + omega * (S + E + I)
    D_prime = delta * I
    return S_prime, E_prime, I_prime, R_prime, D_prime


# ODE solvers wrappers using `scipy.integrate.solve_ivp`:

# In[ ]:


def sir_ode_solver(y0, t_span, t_eval, beta=1, zeta=1/14):
    solution_ODE = solve_ivp(
        fun=lambda t, y: sir_model(t, y, beta=beta, zeta=zeta), 
        t_span=t_span, 
        y0=y0,
        t_eval=t_eval,
        method='LSODA'
    )
    
    return solution_ODE


def sird_ode_solver(y0, t_span, t_eval, beta=1, delta=0.02, zeta=1/14):
    solution_ODE = solve_ivp(
        fun=lambda t, y: sird_model(t, y, beta=beta, zeta=zeta, delta=delta), 
        t_span=t_span, 
        y0=y0,
        t_eval=t_eval,
        method='LSODA'
    )
    
    return solution_ODE


def seir_ode_solver(y0, t_span, t_eval, beta=1, gamma=0, alpha=1/4, zeta=1/14, delta=0.0):
    solution_ODE = solve_ivp(
        fun=lambda t, y: seir2_model(t, y, alpha=alpha, beta=beta, gamma=gamma, zeta=zeta, delta=delta), 
        t_span=t_span, 
        y0=y0,
        t_eval=t_eval,
        method='LSODA'
    )
    
    return solution_ODE


def seird_ode_solver(y0, t_span, t_eval, beta=1, gamma=0, delta=0.02, alpha=1/4, zeta=1/14):
    solution_ODE = solve_ivp(
        fun=lambda t, y: seird_model(t, y, alpha=alpha, beta=beta, gamma=gamma, zeta=zeta, delta=delta), 
        t_span=t_span, 
        y0=y0,
        t_eval=t_eval,
        method='LSODA'
    )
    
    return solution_ODE


def seirdq_ode_solver(y0, t_span, t_eval, beta=1, gamma=0, delta=0.02, omega=0, alpha=1/4, zeta=1/14):
    solution_ODE = solve_ivp(
        fun=lambda t, y: seirdq_model(t, y, alpha=alpha, beta=beta, gamma=gamma, omega=omega, zeta=zeta, delta=delta), 
        t_span=t_span, 
        y0=y0,
        t_eval=t_eval,
        method='LSODA'
    )
    
    return solution_ODE


# Getting population for each country:

# In[ ]:


df_population = pd.read_csv("../input/countries-of-the-world/countries of the world.csv")

df_population


# In[ ]:


brazil_population = float(df_population[df_population.Country == 'Brazil '].Population)
italy_population = float(df_population[df_population.Country == 'Italy '].Population)
china_population = float(df_population[df_population.Country == 'China '].Population)
spain_population = float(df_population[df_population.Country == 'Spain '].Population)
iran_population = float(df_population[df_population.Country == 'Iran '].Population)
us_population = float(df_population[df_population.Country == 'United States '].Population)

target_population = italy_population
target_population


# I'll assume that the whole Brazil's population is suscetible. So I can define the following initial conditions:

# In[ ]:


df_target_country = df_italy_cases_by_day
S0, E0, I0, R0, D0 = target_population, 5 * float(df_target_country.confirmed[0]), float(df_target_country.confirmed[0]), 0., 0.

y0_sir = S0 / target_population, I0 / target_population, R0  # SIR IC array
y0_sird = S0 / target_population, I0 / target_population, R0, D0  # SIRD IC array
y0_seir = S0 / target_population, E0 / target_population, I0 / target_population, R0  # SEIR IC array
y0_seird = S0 / target_population, E0 / target_population, I0 / target_population, R0, D0  # SEIRD IC array


# Select the models to run setting bool variables below:

# In[ ]:


has_to_run_sir = True
has_to_run_sird = False
has_to_run_seir = True
has_to_run_seird = False
has_to_run_seirdq = True


# <a id="least-squares"></a>
# ## Least-Squares fitting
# 
# Now, we can know how to solve the forward problem, so we can try to fit it with a non-linear Least-Squares method for parameter estimation. Let's begin with a generic Least-Square formulation:

# In[ ]:


def sir_least_squares_error_ode(par, time_exp, f_exp, fitting_model, initial_conditions):
    args = par
    time_span = (time_exp.min(), time_exp.max())
    
    y_model = fitting_model(initial_conditions, time_span, time_exp, *args)
    simulated_time = y_model.t
    simulated_ode_solution = y_model.y
    _, simulated_qoi, _ = simulated_ode_solution
    
    residual = f_exp - simulated_qoi

    return np.sum(residual ** 2.0)


def sird_least_squares_error_ode(par, time_exp, f_exp, fitting_model, initial_conditions):
    args = par
    f_exp1, f_exp2 = f_exp
    time_span = (time_exp.min(), time_exp.max())
    
    y_model = fitting_model(initial_conditions, time_span, time_exp, *args)
    simulated_time = y_model.t
    simulated_ode_solution = y_model.y
    _, simulated_qoi1, _, simulated_qoi2 = simulated_ode_solution
    
    residual1 = f_exp1 - simulated_qoi1
    residual2 = f_exp2 - simulated_qoi2

    weighting_for_exp1_constraints = 1e0
    weighting_for_exp2_constraints = 1e0
    return weighting_for_exp1_constraints * np.sum(residual1 ** 2.0) + weighting_for_exp2_constraints * np.sum(residual2 ** 2.0)


def seir_least_squares_error_ode(par, time_exp, f_exp, fitting_model, initial_conditions):
    args = par
    time_span = (time_exp.min(), time_exp.max())
    
    y_model = fitting_model(initial_conditions, time_span, time_exp, *args)
    simulated_time = y_model.t
    simulated_ode_solution = y_model.y
    _, _, simulated_qoi, _ = simulated_ode_solution
    
    residual = f_exp - simulated_qoi

    return np.sum(residual ** 2.0)


def seird_least_squares_error_ode(par, time_exp, f_exp, fitting_model, initial_conditions):
    args = par
    f_exp1, f_exp2 = f_exp
    time_span = (time_exp.min(), time_exp.max())
    
    y_model = fitting_model(initial_conditions, time_span, time_exp, *args)
    simulated_time = y_model.t
    simulated_ode_solution = y_model.y
    _, _, simulated_qoi1, _, simulated_qoi2 = simulated_ode_solution
    
    residual1 = f_exp1 - simulated_qoi1
    residual2 = f_exp2 - simulated_qoi2

    weighting_for_exp1_constraints = 1e0
    weighting_for_exp2_constraints = 1e0
    return weighting_for_exp1_constraints * np.sum(residual1 ** 2.0) + weighting_for_exp2_constraints * np.sum(residual2 ** 2.0)


def callback_de(xk, convergence):
    print(f'parameters = {xk}')


# Setting fitting domain (given time for each observation) and the observations (observed population at given time):

# In[ ]:


data_time = df_target_country.day.values.astype(np.float64)
infected_individuals = df_target_country.confirmed.values / target_population
dead_individuals = df_target_country.deaths.values / target_population


# To calibrate the model, we define an objective function, which is a Least-Squares function in the present case, and minimize it. To (*try to*) avoid local minima, we use Differential Evolution (DE) method (see this [nice presentation](https://www.maths.uq.edu.au/MASCOS/Multi-Agent04/Fleetwood.pdf) to get yourself introduced to this great subject). In summary, DE is a family of Evolutionary Algorithms that aims to solve Global Optimization problems. Moreover, DE is derivative-free and population-based method.
# 
# Below, calibration is performed for selected models:

# In[ ]:


if has_to_run_sir:
    num_of_parameters_to_fit_sir = 1
    bounds_sir = num_of_parameters_to_fit_sir * [(0, 1)]

    result_sir = optimize.differential_evolution(
        sir_least_squares_error_ode, 
        bounds=bounds_sir, 
        args=(data_time, infected_individuals, sir_ode_solver, y0_sir), 
        popsize=300,
        strategy='best1bin',
        tol=1e-2,
        recombination=0.5,
#         mutation=0.7,
        maxiter=100,
        disp=True,
        seed=seed,
        callback=callback_de,
        workers=-1
    )

    print(result_sir)


# In[ ]:


if has_to_run_sird:
    # num_of_parameters_to_fit_sir = 1
    # bounds_sir = num_of_parameters_to_fit_sir * [(0, 1)]
    bounds_sird = [(0, 1), (0, 0.2)]

    result_sird = optimize.differential_evolution(
        sird_least_squares_error_ode, 
        bounds=bounds_sird, 
        args=(data_time, [infected_individuals, dead_individuals], sird_ode_solver, y0_sird), 
        popsize=300,
        strategy='best1bin',
        tol=1e-2,
        recombination=0.5,
    #     mutation=0.7,
        maxiter=100,
        disp=True,
        seed=seed,
        callback=callback_de,
        workers=-1
    )

    print(result_sird)


# In[ ]:


if has_to_run_seird:
    num_of_parameters_to_fit_sir = 1
    # bounds_sir = num_of_parameters_to_fit_sir * [(0, 1)]
    bounds_seird = [(0, 1), (0, 1), (0, 0.2)]

    result_seird = optimize.differential_evolution(
        seird_least_squares_error_ode, 
        bounds=bounds_seird, 
        args=(data_time, [infected_individuals, dead_individuals], seird_ode_solver, y0_seird), 
        popsize=300,
        strategy='best1bin',
        tol=1e-2,
        recombination=0.7,
    #     mutation=0.7,
        maxiter=100,
        disp=True,
        seed=seed,
        callback=callback_de,
        workers=-1
    )

    print(result_seird)


# In[ ]:


if has_to_run_seirdq:
#     num_of_parameters_to_fit_sir = 1
    # bounds_sir = num_of_parameters_to_fit_sir * [(0, 1)]
    bounds_seird = [(0, 1), (0, 1), (0, 0.2), (0, 1)]

    result_seirdq = optimize.differential_evolution(
        seird_least_squares_error_ode, 
        bounds=bounds_seird, 
        args=(data_time, [infected_individuals, dead_individuals], seirdq_ode_solver, y0_seird), 
        popsize=200,
        strategy='best1bin',
        tol=1e-2,
        recombination=0.7,
    #     mutation=0.7,
        maxiter=200,
        disp=True,
        seed=seed,
        callback=callback_de,
        workers=-1
    )

    print(result_seirdq)


# In[ ]:


if has_to_run_seir:
    num_of_parameters_to_fit_seir = 2
    bounds_seir = num_of_parameters_to_fit_seir * [(0, 1)]

    result_seir = optimize.differential_evolution(
        seir_least_squares_error_ode, 
        bounds=bounds_seir, 
        args=(data_time, infected_individuals, seir_ode_solver, y0_seir), 
        popsize=300,
        strategy='best1bin',
        tol=1e-2,
        recombination=0.7,
#         mutation=0.7,
        maxiter=100,
        disp=True,
        seed=seed,
        callback=callback_de,
        workers=-1
    )

    print(result_seir)


# In[ ]:


zeta_fitted = 1/14  # recover rate... the inverse is equal to the amount of days needed to recover from the disease
if has_to_run_sir:
    beta_fitted_sir = result_sir.x  # SIR parameters
    
if has_to_run_sird:
    beta_fitted_sird, delta_fitted_sird = result_sird.x  # SIRD parameters
    
alpha_fitted = 1/4
if has_to_run_seird:
    beta_fitted_seird, gamma_fitted_seird, delta_fitted_seird = result_seird.x  # SEIRD parameters
    
if has_to_run_seirdq:
    beta_fitted_seirdq, gamma_fitted_seirdq, delta_fitted_seirdq, omega_fitted_seirdq = result_seirdq.x  # SEIRD parameters

if has_to_run_seir:
#     beta_fitted_seir, gamma_fitted_seir = result_seir.x  # SEIR parameters
#     gamma_fitted_seir = 0.0
    beta_fitted_seir, gamma_fitted_seir = result_seir.x  # SEIR parameters


# In[ ]:


t0 = data_time.min()
tf = data_time.max()

if has_to_run_sir:
    solution_ODE_sir = sir_ode_solver(y0_sir, (t0, tf), data_time, beta_fitted_sir, zeta_fitted)  # SIR
    t_computed_sir, y_computed_sir = solution_ODE_sir.t, solution_ODE_sir.y
    S_sir, I_sir, R_sir = y_computed_sir

if has_to_run_sird:
    solution_ODE_sird = sird_ode_solver(y0_sird, (t0, tf), data_time, beta_fitted_sird, delta_fitted_sird, zeta_fitted)  # SIRD
    t_computed_sird, y_computed_sird = solution_ODE_sird.t, solution_ODE_sird.y
    S_sird, I_sird, R_sird, D_sird = y_computed_sird

if has_to_run_seird:
    solution_ODE_seird = seird_ode_solver(y0_seird, (t0, tf), data_time, beta_fitted_seird, gamma_fitted_seird, delta_fitted_seird, alpha_fitted, zeta_fitted)  # SEIRD
    t_computed_seird, y_computed_seird = solution_ODE_seird.t, solution_ODE_seird.y
    S_seird, E_seird, I_seird, R_seird, D_seird = y_computed_seird

if has_to_run_seirdq:
    solution_ODE_seirdq = seirdq_ode_solver(
        y0_seird, 
        (t0, tf), 
        data_time, 
        beta_fitted_seirdq, 
        gamma_fitted_seirdq, 
        delta_fitted_seirdq, 
        omega_fitted_seirdq, 
        alpha_fitted, 
        zeta_fitted
    )
    t_computed_seirdq, y_computed_seirdq = solution_ODE_seirdq.t, solution_ODE_seirdq.y
    S_seirdq, E_seirdq, I_seirdq, R_seirdq, D_seirdq = y_computed_seirdq
    
if has_to_run_seir:
    solution_ODE_seir = seir_ode_solver(y0_seir, (t0, tf), data_time, beta_fitted_seir, gamma_fitted_seir, alpha_fitted,  zeta_fitted)  # SEIR
    t_computed_seir, y_computed_seir = solution_ODE_seir.t, solution_ODE_seir.y
    S_seir, E_seir, I_seir, R_seir = y_computed_seir


# In[ ]:


model_list = list()
alpha_list = list()
beta_list = list()
delta_list = list()
gamma_list = list()
omega_list = list()
zeta_list = list()

if has_to_run_sir:
    model_list.append("SIR")
    alpha_list.append("-")
    beta_list.append(np.float(beta_fitted_sir))
    delta_list.append("-")
    gamma_list.append("-")
    omega_list.append("-")
    zeta_list.append(zeta_fitted)

if has_to_run_sird:
    model_list.append("SIRD")
    alpha_list.append("-")
    beta_list.append(beta_fitted_sird)
    delta_list.append(delta_fitted_sird)
    gamma_list.append("-")
    omega_list.append("-")
    zeta_list.append(zeta_fitted)
    
if has_to_run_seir:
    model_list.append("SEIR")
    alpha_list.append(alpha_fitted)
    beta_list.append(beta_fitted_seir)
    delta_list.append("-")
    gamma_list.append(gamma_fitted_seir)
    omega_list.append("-")
    zeta_list.append(zeta_fitted)

if has_to_run_seird:
    model_list.append("SEIRD")
    alpha_list.append(alpha_fitted)
    beta_list.append(beta_fitted_seird)
    delta_list.append(delta_fitted_seird)
    gamma_list.append(gamma_fitted_seird)
    omega_list.append("-")
    zeta_list.append(zeta_fitted)

if has_to_run_seirdq:
    model_list.append("SEIRD-Q")
    alpha_list.append(alpha_fitted)
    beta_list.append(beta_fitted_seirdq)
    delta_list.append(delta_fitted_seirdq)
    gamma_list.append(gamma_fitted_seirdq)
    omega_list.append(omega_fitted_seirdq)
    zeta_list.append(zeta_fitted)
    
parameters_dict = {
    "Model": model_list,
    r"$\alpha$": alpha_list,
    r"$\beta$": beta_list,
    r"$\delta$": delta_list,
    r"$\gamma$": gamma_list,
    r"$\omega$": omega_list,
    r"$\zeta$": zeta_list,
}

df_parameters_calibrated = pd.DataFrame(parameters_dict)

df_parameters_calibrated


# In[ ]:


print(df_parameters_calibrated.to_latex(index=False))


# Show calibration result based on available data:

# In[ ]:


plt.figure(figsize=(9,7))

if has_to_run_sir:
    plt.plot(t_computed_sir, I_sir * target_population, label='Infected (SIR)', marker='v', linestyle="-", markersize=10)
    plt.plot(t_computed_sir, R_sir * target_population, label='Recovered (SIR)', marker='o', linestyle="-", markersize=10)
    
if has_to_run_sird:
    plt.plot(t_computed_sird, I_sird * target_population, label='Infected (SIRD)', marker='X', linestyle="-", markersize=10)
    plt.plot(t_computed_sird, R_sird * target_population, label='Recovered (SIRD)', marker='o', linestyle="-", markersize=10)
    plt.plot(t_computed_sird, D_sird * target_population, label='Deaths (SIRD)', marker='s', linestyle="-", markersize=10)
    
if has_to_run_seird:
    plt.plot(t_computed_seird, I_seird * target_population, label='Infected (SEIRD)', marker='X', linestyle="-", markersize=10)
    plt.plot(t_computed_seird, R_seird * target_population, label='Recovered (SEIRD)', marker='o', linestyle="-", markersize=10)
    plt.plot(t_computed_seird, D_seird * target_population, label='Deaths (SEIRD)', marker='s', linestyle="-", markersize=10)
    
if has_to_run_seirdq:
    plt.plot(t_computed_seirdq, I_seirdq * target_population, label='Infected (SEIRD-Q)', marker='X', linestyle="-", markersize=10)
#     plt.plot(t_computed_seirdq, R_seirdq * target_population, label='Recovered (SEIRD-Q)', marker='o', linestyle="-", markersize=10)
    plt.plot(t_computed_seirdq, D_seirdq * target_population, label='Deaths (SEIRD-Q)', marker='s', linestyle="-", markersize=10)

if has_to_run_seir:
    plt.plot(t_computed_seir, I_seir * target_population, label='Infected (SEIR)', marker='X', linestyle="-", markersize=10)
    plt.plot(t_computed_seir, R_seir * target_population, label='Recovered (SEIR)', marker='o', linestyle="-", markersize=10)
    
plt.plot(data_time, infected_individuals * target_population, label='Observed infected', marker='s', linestyle="", markersize=10)
plt.plot(data_time, dead_individuals * target_population, label='Recorded deaths', marker='v', linestyle="", markersize=10)
plt.legend()
plt.grid()
plt.xlabel('Time (days)')
plt.ylabel('Population')

plt.tight_layout()
plt.savefig("all_deterministic_calibration.png")
plt.show()


# In[ ]:


methods_list = list()
deaths_list = list()
if has_to_run_sird:
    methods_list.append("SIRD")
    deaths_list.append(int(D_sird.max() * target_population))
    print(f"Death estimate for today (SIRD):\t{int(D_sird.max() * target_population)}")
    
if has_to_run_seird:
    methods_list.append("SEIRD")
    deaths_list.append(int(D_seird.max() * target_population))
    print(f"Death estimate for today (SEIRD):\t{int(D_seird.max() * target_population)}")
    
if has_to_run_seirdq:
    methods_list.append("SEIRD-Q")
    deaths_list.append(int(D_seirdq.max() * target_population))
    print(f"Death estimate for today (SEIRD-Q):\t{int(D_seirdq.max() * target_population)}")

methods_list.append("Recorded")
deaths_list.append(int(dead_individuals[-1] * target_population))

death_estimates_dict = {"Method": methods_list, "Deaths estimate": deaths_list}
df_deaths_estimates = pd.DataFrame(death_estimates_dict)
print(f"Recorded deaths until today:\t{int(dead_individuals[-1] * target_population)}")


# In[ ]:


# df_deaths_estimates.set_index("Model", inplace=True)
print(df_deaths_estimates.to_latex(index=False))


# <a id="deterministic-predictions"></a>
# ## Extrapolation/Predictions
# 
# Now, let's extrapolate to next days.

# In[ ]:


t0 = float(data_time.min())
number_of_days_after_last_record = 90
tf = data_time.max() + number_of_days_after_last_record
time_range = np.linspace(0., tf, int(tf))

if has_to_run_sir:
    solution_ODE_predict_sir = sir_ode_solver(y0_sir, (t0, tf), time_range, beta_fitted_sir, zeta_fitted)  # SIR
    t_computed_predict_sir, y_computed_predict_sir = solution_ODE_predict_sir.t, solution_ODE_predict_sir.y
    S_predict_sir, I_predict_sir, R_predict_sir = y_computed_predict_sir

if has_to_run_sird:
    solution_ODE_predict_sird = sird_ode_solver(y0_sird, (t0, tf), time_range, beta_fitted_sird, delta_fitted_sird, zeta_fitted)  # SIR
    t_computed_predict_sird, y_computed_predict_sird = solution_ODE_predict_sird.t, solution_ODE_predict_sird.y
    S_predict_sird, I_predict_sird, R_predict_sird, D_predict_sird = y_computed_predict_sird

if has_to_run_seird:
    solution_ODE_predict_seird = seird_ode_solver(y0_seird, (t0, tf), time_range, beta_fitted_seird, gamma_fitted_seird, delta_fitted_seird, alpha_fitted, zeta_fitted)  # SEIRD
    t_computed_predict_seird, y_computed_predict_seird = solution_ODE_predict_seird.t, solution_ODE_predict_seird.y
    S_predict_seird, E_predict_seird, I_predict_seird, R_predict_seird, D_predict_seird = y_computed_predict_seird
    
if has_to_run_seirdq:
    solution_ODE_predict_seirdq = seirdq_ode_solver(y0_seird, (t0, tf), time_range, beta_fitted_seirdq, gamma_fitted_seirdq, delta_fitted_seirdq, omega_fitted_seirdq, alpha_fitted, zeta_fitted)  # SEIRD
    t_computed_predict_seirdq, y_computed_predict_seirdq = solution_ODE_predict_seirdq.t, solution_ODE_predict_seirdq.y
    S_predict_seirdq, E_predict_seirdq, I_predict_seirdq, R_predict_seirdq, D_predict_seirdq = y_computed_predict_seirdq

if has_to_run_seir:
    solution_ODE_predict_seir = seir_ode_solver(y0_seir, (t0, tf), time_range, beta_fitted_seir, gamma_fitted_seir, alpha_fitted, zeta_fitted)  # SEIR
    t_computed_predict_seir, y_computed_predict_seir = solution_ODE_predict_seir.t, solution_ODE_predict_seir.y
    S_predict_seir, E_predict_seir, I_predict_seir, R_predict_seir = y_computed_predict_seir


# Calculating the day when the number of infected individuals is max:

# In[ ]:


has_to_plot_infection_peak = True

if has_to_run_sir:
    crisis_day_sir = np.argmax(I_predict_sir)
    
if has_to_run_sird:
    crisis_day_sird = np.argmax(I_predict_sird)

if has_to_run_seir:
    crisis_day_seir = np.argmax(I_predict_seir)
    
if has_to_run_seird:
    crisis_day_seird = np.argmax(I_predict_seird)
    
if has_to_run_seirdq:
    crisis_day_seirdq = np.argmax(I_predict_seirdq)


# In[ ]:


plt.figure(figsize=(9,7))

if has_to_run_sir:
    plt.plot(t_computed_predict_sir, 100 * S_predict_sir, label='Susceptible (SIR)', marker='s', linestyle="-", markersize=10)
    plt.plot(t_computed_predict_sir, 100 * I_predict_sir, label='Infected (SIR)', marker='X', linestyle="-", markersize=10)
    plt.plot(t_computed_predict_sir, 100 * R_predict_sir, label='Recovered (SIR)', marker='o', linestyle="-", markersize=10)
    if has_to_plot_infection_peak:
        plt.axvline(x=crisis_day_sir + 1, color="red", linestyle="--", label="Infected peak (SIR)")
    
if has_to_run_sird:
    plt.plot(t_computed_predict_sird, 100 * S_predict_sird, label='Susceptible (SIRD)', marker='s', linestyle="-", markersize=10)
    plt.plot(t_computed_predict_sird, 100 * I_predict_sird, label='Infected (SIRD)', marker='X', linestyle="-", markersize=10)
    plt.plot(t_computed_predict_sird, 100 * R_predict_sird, label='Recovered (SIRD)', marker='o', linestyle="-", markersize=10)
    plt.plot(t_computed_predict_sird, 100 * D_predict_sird, label='Deaths (SIRD)', marker='v', linestyle="-", markersize=10)
    if has_to_plot_infection_peak:
        plt.axvline(x=crisis_day_sird + 1, color="red", linestyle="--", label="Infected peak (SIRD)")

if has_to_run_seird:
    plt.plot(t_computed_predict_seird, 100 * S_predict_seird, label='Susceptible (SEIRD)', marker='s', linestyle="-", markersize=10)
    plt.plot(t_computed_predict_seird, 100 * E_predict_seird, label='Exposed (SEIRD)', marker='*', linestyle="-", markersize=10)
    plt.plot(t_computed_predict_seird, 100 * I_predict_seird, label='Infected (SEIRD)', marker='X', linestyle="-", markersize=10)
    plt.plot(t_computed_predict_seird, 100 * R_predict_seird, label='Recovered (SEIRD)', marker='o', linestyle="-", markersize=10)
    plt.plot(t_computed_predict_seird, 100 * D_predict_seird, label='Deaths (SEIRD)', marker='v', linestyle="-", markersize=10)
    if has_to_plot_infection_peak:
        plt.axvline(x=crisis_day_seird + 1, color="red", label="Infected peak (SEIRD)")
    
if has_to_run_seirdq:
    plt.plot(t_computed_predict_seirdq, 100 * S_predict_seirdq, label='Susceptible (SEIRD-Q)', marker='s', linestyle="-", markersize=10)
    plt.plot(t_computed_predict_seirdq, 100 * E_predict_seirdq, label='Exposed (SEIRD-Q)', marker='*', linestyle="-", markersize=10)
    plt.plot(t_computed_predict_seirdq, 100 * I_predict_seirdq, label='Infected (SEIRD-Q)', marker='X', linestyle="-", markersize=10)
    plt.plot(t_computed_predict_seirdq, 100 * R_predict_seirdq, label='Recovered (SEIRD-Q)', marker='o', linestyle="-", markersize=10)
    plt.plot(t_computed_predict_seirdq, 100 * D_predict_seirdq, label='Deaths (SEIRD-Q)', marker='v', linestyle="-", markersize=10)
    if has_to_plot_infection_peak:
        plt.axvline(x=crisis_day_seirdq + 1, color="red", label="Infected peak (SEIRD-Q)")

if has_to_run_seir:
    plt.plot(t_computed_predict_seir, 100 * S_predict_seir, label='Susceptible (SEIR)', marker='s', linestyle="-", markersize=10)
    plt.plot(t_computed_predict_seir, 100 * E_predict_seir, label='Exposed (SEIR)', marker='*', linestyle="-", markersize=10)
    plt.plot(t_computed_predict_seir, 100 * I_predict_seir, label='Infected (SEIR)', marker='X', linestyle="-", markersize=10)
    plt.plot(t_computed_predict_seir, 100 * R_predict_seir, label='Recovered (SEIR)', marker='o', linestyle="-", markersize=10)
    if has_to_plot_infection_peak:
        plt.axvline(x=crisis_day_seir + 1, color="red", linestyle="--", label="Infected peak (SEIR)")

plt.xlabel('Time (days)')
plt.ylabel('Population %')
plt.legend()
plt.grid()

plt.tight_layout()
plt.savefig("seir_deterministic_predictions.png")
plt.show()


# In[ ]:


if has_to_run_sir:
    print(f"Max number of infected individuals (SIR model):\t {int(np.max(I_predict_sir) * target_population)}")
    print(f"Population percentage of max number of infected individuals (SIR model):\t {np.max(I_predict_sir) * 100:.2f}%")
    print(f"Day estimate for max number of infected individuals (SIR model):\t {crisis_day_sir + 1}")
    print("")

if has_to_run_sird:
    print(f"Max number of infected individuals (SIRD model):\t {int(np.max(I_predict_sird) * target_population)}")
    print(f"Population percentage of max number of infected individuals (SIRD model):\t {np.max(I_predict_sird) * 100:.2f}%")
    print(f"Day estimate for max number of infected individuals (SIRD model):\t {crisis_day_sird + 1}")
    print(f"Percentage of number of death estimate (SIRD model):\t {100 * D_predict_sird[-1]:.3f}%")
    print(f"Number of death estimate (SIRD model):\t {target_population * D_predict_sird[-1]:.3f}")
    print("")

if has_to_run_seir:
    print(f"Max number of infected individuals (SEIR model):\t {int(np.max(I_predict_seir) * target_population)}")
    print(f"Population percentage of max number of infected individuals (SEIR model):\t {np.max(I_predict_seir) * 100:.2f}%")
    print(f"Day estimate for max number of infected individuals (SEIR model):\t {crisis_day_seir + 1}")
    print("")
    
if has_to_run_seird:
    print(f"Max number of infected individuals (SEIRD model):\t {int(np.max(I_predict_seird) * target_population)}")
    print(f"Population percentage of max number of infected individuals (SEIRD model):\t {np.max(I_predict_seird) * 100:.2f}%")
    print(f"Day estimate for max number of infected individuals (SEIRD model):\t {crisis_day_seird + 1}")
    print(f"Percentage of number of death estimate (SEIRD model):\t {100 * D_predict_seird[-1]:.3f}%")
    print(f"Number of death estimate (SEIRD model):\t {target_population * D_predict_seird[-1]:.3f}")
    print("")
    
if has_to_run_seirdq:
    print(f"Max number of infected individuals (SEIRD-Q model):\t {int(np.max(I_predict_seirdq) * target_population)}")
    print(f"Population percentage of max number of infected individuals (SEIRD-Q model):\t {np.max(I_predict_seirdq) * 100:.2f}%")
    print(f"Day estimate for max number of infected individuals (SEIRD-Q model):\t {crisis_day_seirdq + 1}")
    print(f"Percentage of number of death estimate (SEIRD-Q model):\t {100 * D_predict_seirdq[-1]:.3f}%")
    print(f"Number of death estimate (SEIRD-Q model):\t {target_population * D_predict_seirdq[-1]:.3f}")
    print("")


# <a id="bayes-calibration"></a>
# ## Bayesian Calibration
# 
# Now that we have an idea of the values that we must expect (i.e., some prior knowledge), we can properly apply a Bayesian model calibration. This way, we can estimate uncertainties on our models parameters and outcomes/predictions. For such purpose, we'll use the great [PyMC3](https://docs.pymc.io/) package.

# <a id="bayes-sir"></a>
# ### SIR Model
# 
# Let's begin with easy :) Just one parameter!

# In[ ]:


@theano.compile.ops.as_op(itypes=[t.dvector, t.dvector, t.dvector, t.dscalar], otypes=[t.dvector])
def sir_ode_solver_wrapper(time_exp, f_observations, initial_conditions, beta):
    time_span = (time_exp.min(), time_exp.max())
    
    zeta = 1/14
    y_model = sir_ode_solver(initial_conditions, time_span, time_exp, beta, zeta)
    simulated_time = y_model.t
    simulated_ode_solution = y_model.y
    _, simulated_qoi, _ = simulated_ode_solution

    return simulated_qoi


# In[ ]:


with pm.Model() as model_mcmc:
    # Prior distributions for the model's parameters
    beta = pm.Uniform('beta', lower=0.2, upper=0.4)

    # Defining the deterministic formulation of the problem
    fitting_model = pm.Deterministic('sir_model', sir_ode_solver_wrapper(
        theano.shared(data_time), 
        theano.shared(infected_individuals), 
        theano.shared(np.array(y0_sir)),
        beta,
        )
    )

    # Variance related to population fraction amount! Let's assume a variance of 100 individuals, since there are cases that have been not tracked
    variance = (100 / target_population) * (100 / target_population)
    standard_deviation = np.sqrt(variance)
    likelihood_model = pm.Normal('likelihood_model', mu=fitting_model, sigma=standard_deviation, observed=infected_individuals)

    # The Monte Carlo procedure driver
    step = pm.step_methods.Metropolis()
    sir_trace = pm.sample(4500, chains=4, cores=4, step=step)


# In[ ]:


pm.traceplot(sir_trace, var_names=('beta'))
plt.savefig('sir_beta_traceplot.png')
plt.show()


# In[ ]:


pm.plot_posterior(sir_trace, var_names=('beta'), kind='hist', round_to=3)
plt.savefig('sir_beta_posterior.png')
plt.show()


# In[ ]:


percentile_cut = 2.5

y_min_sir = np.percentile(sir_trace['sir_model'], percentile_cut, axis=0)
y_max_sir = np.percentile(sir_trace['sir_model'], 100 - percentile_cut, axis=0)
y_fit_sir = np.percentile(sir_trace['sir_model'], 50, axis=0)


# In[ ]:


plt.figure(figsize=(9, 7))

plt.plot(data_time, y_fit_sir, 'b', label='Infected')
plt.fill_between(data_time, y_min_sir, y_max_sir, color='b', alpha=0.2)

plt.legend()
plt.xlabel('Time (day)')
plt.ylabel('Population %')
# plt.xlim(0, 10)

plt.savefig('sir_uncertainty.png')
plt.show()


# Improvements in this SIR model calibration are under investigations!

# <a id="bayes-seir2"></a>
# ### SEIR-2 Model
# 
# A more interesting model, with two parameters to calibrate.

# In[ ]:


@theano.compile.ops.as_op(itypes=[t.dvector, t.dvector, t.dvector, t.dscalar, t.dscalar], otypes=[t.dvector])
def seir_ode_solver_wrapper(time_exp, f_observations, initial_conditions, beta, gamma):
    time_span = (time_exp.min(), time_exp.max())
    
    y_model = seir_ode_solver(initial_conditions, time_span, time_exp, beta, gamma)
    simulated_time = y_model.t
    simulated_ode_solution = y_model.y
    _, _, simulated_qoi, _ = simulated_ode_solution

    return simulated_qoi


# In[ ]:


with pm.Model() as model_mcmc:
    # Prior distributions for the model's parameters
    beta = pm.Uniform('beta', lower=0, upper=0.001)
    gamma = pm.Uniform('gamma', lower=0, upper=0.5)

    # Defining the deterministic formulation of the problem
    fitting_model = pm.Deterministic('seir2_model', seir_ode_solver_wrapper(
        theano.shared(data_time), 
        theano.shared(infected_individuals), 
        theano.shared(np.array(y0_seir)),
        beta,
        gamma
        )
    )

    # Variance related to population fraction amount! Let's assume a variance of 100 individuals, since there are cases that have been not tracked
    variance = (100 / target_population) * (100 / target_population)
    standard_deviation = np.sqrt(variance)
    likelihood_model = pm.Normal('likelihood_model', mu=fitting_model, sigma=standard_deviation, observed=infected_individuals)

    # The Monte Carlo procedure driver
    step = pm.step_methods.Metropolis()
    seir_trace = pm.sample(8500, chains=4, cores=4, step=step)


# In[ ]:


pm.traceplot(seir_trace, var_names=('beta'))
plt.savefig('seir2_beta_traceplot.png')
plt.show()

pm.traceplot(seir_trace, var_names=('gamma'))
plt.savefig('seir2_gamma_traceplot.png')
plt.show()


# In[ ]:


pm.plot_posterior(seir_trace, var_names=('beta'), kind='hist', round_to=3)
plt.savefig('seir2_beta_posterior.png')
plt.show()

pm.plot_posterior(seir_trace, var_names=('gamma'), kind='hist', round_to=3)
plt.savefig('seir2_gamma_posterior.png')
plt.show()


# In[ ]:


percentile_cut = 2.5

y_min_seir = np.percentile(seir_trace['seir2_model'], percentile_cut, axis=0)
y_max_seir = np.percentile(seir_trace['seir2_model'], 100 - percentile_cut, axis=0)
y_fit_seir = np.percentile(seir_trace['seir2_model'], 50, axis=0)


# In[ ]:


plt.figure(figsize=(9, 7))

plt.plot(data_time, y_fit_seir, 'b', label='Infected')
plt.fill_between(data_time, y_min_seir, y_max_seir, color='b', alpha=0.2)

plt.legend()
plt.xlabel('Time (day)')
plt.ylabel('Population %')
# plt.xlim(0, 10)

plt.savefig('seir2_uncertainty.png')
plt.show()


# Note: Bayesian calibration is currently under investigations/improvements.
