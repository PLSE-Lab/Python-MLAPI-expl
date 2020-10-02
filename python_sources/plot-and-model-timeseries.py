# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import csv
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
def plot_data(data, min_days=20, min_cases=100, label=''):
    for index, row in data.iterrows():
        row = row.to_numpy()
        datapoints = row[5:].astype(np.float)
        datapoints = datapoints[~np.isnan(datapoints)]
        # this should not be needed as data is cumulative -> maybe faulty raw data
        datapoints = datapoints[datapoints != 0]

        y = np.expand_dims(datapoints, axis=1)
        y_log = np.log10(y)
        M = np.ones((y.shape[0], 2))
        M[:, 0] = np.arange(y.shape[0])

        M_inv = np.linalg.pinv(M)

        [m, b] = M_inv @ y_log

        plt.figure()

        plt.subplot(121)
        plt.plot(y, label=label)
        plt.title(row[1])
        plt.xlabel('Days after first case')
        plt.ylabel('Number cases')
        plt.legend()

        plt.subplot(122)
        plt.plot(y_log, label= label + ' log')

        x = np.arange(y_log.shape[0])
        y_fit = m * x + b

        plt.plot(y_fit, label='linear fit')

        plt.title(row[1] + ', m: ' + '{0:.2f}'.format(m[0]))
        plt.xlabel('Days after first case')
        plt.ylabel('Number cases [log]')
        plt.legend()


def plot_timeseries(infections, fips=None, label=''):
    # if no fips list is given: plot all counties which fulfill min days and cases
    fips_rows = infections.loc[infections['FIPS'].isin(fips)]
    plot_data(fips_rows, label=label)
    plt.show()

## Extract the timeseries from a pandas dataframe by fips
def get_timeseries(infections, deaths, fips, population=None):
    i = infections.loc[infections['FIPS'] == fips]
    i = i.values[0][3:].astype(np.float)
    i = i[~np.isnan(i)]
    d = deaths.loc[deaths['FIPS'] == fips]
    d = d.values[0][3:].astype(np.float)
    d = d[~np.isnan(d)]
    d = np.pad(d, (len(i) - len(d),0), 'constant', constant_values=(0,0))
    t = np.linspace(1, len(i), num=len(i))

    if population:
        i = i/population
        d = d/population
    return t, i, d

## Fit an exponential model to the timeseries
def fit_exponential(x, a):
    y = np.exp(a*x)
    return y

## Define some error metric
def calc_error(a, b):
    sq_err = ((a-b)**2).mean()
    return np.sqrt(sq_err)

def print_fit(name, param, cov, err):
    print(f"{name}'s growth factor is estimated to be {param} ({cov}) with error {err}")

def plot(gt, pred, label='', county=''):
    plt.figure()
    plt.plot(gt, label='Measured')
    plt.plot(pred, label='Predicted')
    plt.xlabel('Days after first case')
    plt.ylabel('Number of ' + label)
    plt.title(county + ' ' + label)
    plt.legend()
    plt.show()

def fit_timeseries(model, time, data):
    param, param_cov = curve_fit(model, time, data)
    pred = model(time, param[0])
    error = calc_error(data, pred)
    return param, param_cov, pred, error

## First, we load the data and identify some counties we're interested in
## Let's go with King County, WA (where the first US case was identified), and WESTCHESTER
infections = pd.read_csv('/kaggle/input/covid19-us-countylevel-summaries/infections_timeseries.csv')
deaths = pd.read_csv('/kaggle/input/covid19-us-countylevel-summaries/deaths_timeseries.csv')
fips = [53033]

## To get a general overview of the data, we can first plot them
plot_timeseries(infections, fips=fips, label='Infections')
plot_timeseries(deaths, fips=fips, label='Deaths')

## Let's take a deeper look at the data and see how the growth in these two counties compare
## Read out the timeseries in each county and we can calculate the growth rate
king_time, king_infections, king_deaths = get_timeseries(infections, deaths, fips[0])
king_infections_param, king_infections_param_cov, king_infections_pred, king_infections_error = fit_timeseries(fit_exponential, king_time, king_infections)
king_deaths_param, king_deaths_param_cov, king_deaths_pred, king_deaths_error = fit_timeseries(fit_exponential, king_time, king_deaths)

print_fit('King County infections', king_infections_param, king_infections_param_cov, king_infections_error)
print_fit('King County death', king_deaths_param, king_deaths_param_cov, king_deaths_error)

plot(king_infections, king_infections_pred, 'Infections', 'King County')
plot(king_deaths, king_deaths_pred, 'Deaths', 'King County')

