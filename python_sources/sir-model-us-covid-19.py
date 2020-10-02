#!/usr/bin/env python
# coding: utf-8

# # SIR Modelling of COVID-19
# 
# What does a naive SIR model have to say about the likely course of COVID-19 through the US population?
# 
# See https://en.wikipedia.org/wiki/Compartmental_models_in_epidemiology#The_SIR_model for an explanation of the model and its underlying assumptions.
# 
# This model has no concept of uncertainty, super-spreader events, geographic distances, or anything like that.  It's just really straightforward math about rates of infection and recovery.  Given how little data we have to work with right now, that seems fine.

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


# From SciPython book: https://scipython.com/book/chapter-8-scipy/additional-examples/the-sir-epidemic-model/

# Plugging in estimates for Novel Coronavirus and US.

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# The SIR model differential equations.
def deriv(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt

def plot_sir_curves(title, N, I0, R0, beta, gamma):
    R_0 = beta / gamma

    # A grid of time points (in days)
    t = np.linspace(0, 365 * 2, 1000)

    # Initial conditions vector
    y0 = S0, I0, R0
    # Integrate the SIR equations over the time grid, t.
    ret = odeint(deriv, y0, t, args=(N, beta, gamma))
    S, I, R = ret.T

    # Plot the data on three separate curves for S(t), I(t) and R(t)
    fig = plt.figure(facecolor='w', figsize=(20, 10))

    ax = fig.add_subplot(111) #, axisbelow=True)
    ax.plot(t, S, 'b', alpha=0.5, lw=2, label='Susceptible')
    ax.plot(t, I, 'r', alpha=0.5, lw=2, label='Infected')
    ax.plot(t, R, 'g', alpha=0.5, lw=2, label='Recovered with immunity')
    ax.set_xlabel('Time /days')
    ax.set_ylabel('Number')
    ax.set_ylim(0, N * 1.1)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))

    legend = ax.legend()
    legend.get_frame().set_alpha(1.0)

    ax.minorticks_on()
    ax.grid(b=True, axis='y', which='minor', linestyle=':')
    ax.grid(b=True, axis='y', which='major', lw=2, ls='-')
    ax.grid(b=True, axis='x', which='major', lw=2, ls='-', alpha=0.5)

    plt.title("{:}\n(R_0: {:.3f}, beta: {:.3f})".format(title, R_0, beta))

    plt.show()


# # Pessimistic Scenario
# 
# Use an $R_0=4$.  This is among the higher estimates in https://sph.umich.edu/pursuit/2020posts/how-scientists-quantify-outbreaks.html.
# 
# Assumes no social distancing and virus is left free to spread at this "natural" rate.

# In[ ]:


# Total population, N.
N = 327000000

# Initial number of infected and recovered individuals, I0 and R0.
I0, R0 = 3500, 0
# Everyone else, S0, is susceptible to infection initially.
S0 = N - I0 - R0

# mean recovery rate, gamma, (in 1/days).
gamma = 1./14

# Contact rate, beta (percent of population touched+infected by an infected person daily).
# Also, R_0 = beta / gamma
R_0 = 4
beta = R_0 * gamma

plot_sir_curves("US - Worst Case Estimate, No Lockdown Scenario", N, I0, R0, beta, gamma)


# Everything from here on out starts with an $R_0=2.5$, which is closer to the middle of the estimate range.

# # Optimistic Scenario
# 
# Use an $R_0=1.4$.  This is among the lower estimates in https://sph.umich.edu/pursuit/2020posts/how-scientists-quantify-outbreaks.html.
# 
# Assumes no social distancing and virus is left free to spread at this "natural" rate.

# In[ ]:


# Total population, N.
N = 327000000

# Initial number of infected and recovered individuals, I0 and R0.
I0, R0 = 3500, 0
# Everyone else, S0, is susceptible to infection initially.
S0 = N - I0 - R0

# mean recovery rate, gamma, (in 1/days).
gamma = 1./14

# Contact rate, beta (percent of population touched+infected by an infected person daily).
# Also, R_0 = beta / gamma
R_0 = 1.4
beta = R_0 * gamma

plot_sir_curves("US - Optimistic Estimate, No Lockdown Scenario", N, I0, R0, beta, gamma)


# # Reasonable Estimates + Social Distancing
# 
# Assumes a more middle-of-the-road $R_0=2.5$.  Looks at "social distancing" in terms of its impact on the contact rate ($\beta$).

# In[ ]:


# Total population, N.
N = 327000000

# Initial number of infected and recovered individuals, I0 and R0.
I0, R0 = 3500, 0
# Everyone else, S0, is susceptible to infection initially.
S0 = N - I0 - R0

# mean recovery rate, gamma, (in 1/days).
gamma = 1./14

# Contact rate, beta (percent of population touched+infected by an infected person daily).
# Also, R_0 = beta / gamma
R_0 = 2.5
beta = R_0 * gamma

plot_sir_curves("US - Reasonable $R_0$ with no lockdown", N, I0, R0, beta, gamma)


# In[ ]:


# Total population, N.
N = 327000000

# Initial number of infected and recovered individuals, I0 and R0.
I0, R0 = 3500, 0
# Everyone else, S0, is susceptible to infection initially.
S0 = N - I0 - R0

# mean recovery rate, gamma, (in 1/days).
gamma = 1./14

# Contact rate, beta (percent of population touched+infected by an infected person daily).
# Also, R_0 = beta / gamma
R_0 = 2.5
beta = R_0 * gamma * (1 - 0.25)

plot_sir_curves("US - Lockdown 25% Effective Scenario", N, I0, R0, beta, gamma)


# In[ ]:


# Total population, N.
N = 327000000

# Initial number of infected and recovered individuals, I0 and R0.
I0, R0 = 3500, 0
# Everyone else, S0, is susceptible to infection initially.
S0 = N - I0 - R0

# mean recovery rate, gamma, (in 1/days).
gamma = 1./14

# Contact rate, beta (percent of population touched+infected by an infected person daily).
# Also, R_0 = beta / gamma
R_0 = 2.5
beta = R_0 * gamma * (1 - 0.4)

plot_sir_curves("US - Lockdown 40% Effective Scenario", N, I0, R0, beta, gamma)


# In[ ]:


# Total population, N.
N = 327000000

# Initial number of infected and recovered individuals, I0 and R0.
I0, R0 = 3500, 0
# Everyone else, S0, is susceptible to infection initially.
S0 = N - I0 - R0

# mean recovery rate, gamma, (in 1/days).
gamma = 1./14

# Contact rate, beta (percent of population touched+infected by an infected person daily).
# Also, R_0 = beta / gamma
R_0 = 2.5
beta = R_0 * gamma * (1 - 0.60)

plot_sir_curves("US - Lockdown 60% Effective Scenario", N, I0, R0, beta, gamma)

