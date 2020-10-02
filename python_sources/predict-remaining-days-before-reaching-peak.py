#!/usr/bin/env python
# coding: utf-8

# # Predict remaining days before reaching peak
# In this Notebook, I used SIR model to predict the remaining days needed before reaching the peak of infections in a given country.
# 
# This value can be used as a feature to train your ML model.
# 
# Reference for SIR model : [https://www.kaggle.com/saga21/covid-global-forecast-sir-model-ml-regressions/notebook](https://www.kaggle.com/saga21/covid-global-forecast-sir-model-ml-regressions/notebook)

# In[ ]:


# provide a country and its population here
country = 'France'
country_population = 67000000


# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from scipy import integrate, optimize

submission_example = pd.read_csv("../input/covid19-global-forecasting-week-2/submission.csv")
test = pd.read_csv("../input/covid19-global-forecasting-week-2/test.csv")
train = pd.read_csv("../input/covid19-global-forecasting-week-2/train.csv")


# In[ ]:


def sir_model(y, x, beta, gamma):
    sus = -beta * y[0] * y[1] / N
    rec = gamma * y[1]
    inf = -(sus + rec)
    return sus, inf, rec

def fit_odeint(x, beta, gamma):
    return integrate.odeint(sir_model, (sus0, inf0, rec0), x, args=(beta, gamma))[:,1]


# Expand to see code needed to make our SIR model fit with the real data of the given country :

# In[ ]:


def fit_sir_country(country, country_pop, initial_date, additional_sim_days):
    """
    Fit the SIR curve with real values of a country starting from the initial_date
    
    Returns:
    remaining_to_peak = remaining days before we reach the peak of infections
    ydata = real infection data
    fitted = simulated infection data
    xdata = days vector of real data
    xdata2 = extended vector of days taking into consideration future predictions
    """
    population = float(country_pop)
    confirmed_total_date_country = train[train['Country_Region']==country].groupby(['Date']).agg({'ConfirmedCases':['sum']})
    fatalities_total_date_country = train[train['Country_Region']==country].groupby(['Date']).agg({'Fatalities':['sum']})
    total_date_country = confirmed_total_date_country.join(fatalities_total_date_country)
    country_df = total_date_country[(initial_date+1):]
    country_df['day_count'] = list(range(1,len(country_df)+1))

    ydata = [i for i in country_df.ConfirmedCases['sum'].values]
    xdata = country_df.day_count
    ydata = np.array(ydata, dtype=float)
    xdata = np.array(xdata, dtype=float)

    N = population
    inf0 = ydata[0]
    sus0 = N - inf0
    rec0 = 0.0
    
    def sir_model(y, x, beta, gamma):
        sus = -beta * y[0] * y[1] / N
        rec = gamma * y[1]
        inf = -(sus + rec)
        return sus, inf, rec

    def fit_odeint(x, beta, gamma):
        return integrate.odeint(sir_model, (sus0, inf0, rec0), x, args=(beta, gamma))[:,1]

    sim_length = len(xdata) + additional_sim_days # Length of simulation
    xdata2 = np.arange(1,sim_length)
    popt, pcov = optimize.curve_fit(fit_odeint, xdata, ydata)
    fitted = fit_odeint(xdata2, *popt)
    print("Initial Start day : ", initial_date, " Optimal parameters: beta =", popt[0], " and gamma = ", popt[1])
    remaining_to_peak = np.argmax(fitted) - len(xdata)
    print("   Remaining days to reach global peak infected cases : ", remaining_to_peak)
    return remaining_to_peak, ydata, fitted, xdata, xdata2


# In[ ]:


# Fit SIR to the corresponding country and for the initial simulation start date
remaining_to_peak, ydata, fitted, xdata, xdata2 = fit_sir_country(country, country_population, 10, 40)
plt.plot(xdata, ydata, 'o', label='Real data')
plt.plot(xdata2, fitted, label='SIR prediction')
plt.title("Fit of SIR model to the country infected cases")
plt.ylabel("Population infected")
plt.xlabel("Days")
plt.legend(loc='best')
plt.show()


# In[ ]:


remaining_days = []
# Loop on different initial start date for the simulation
for i in range(2,20):
    remaining_to_peak, _, _, _, _ = fit_sir_country(country, country_population, i, 40)
    remaining_days.append(remaining_to_peak)
print(remaining_days)


# Here, we can see that -sometimes- we get negative duration values for some simulations. This is due to a bad fit between real data and SIR model. We can remove the negative values from that list and calculate the mean of the list to get an idea about the remaining days before the peak.

# In[ ]:


from scipy import stats

def rmNegative(L):
    index = len(L) - 1
    while index >= 0:
        if L[index] < 0:
            del L[index]
        index = index - 1

rmNegative(remaining_days)
stats.describe(remaining_days)
print("On average, the peak of infected cases in ", country, " is coming in : ", np.mean(remaining_days) ,"days")

