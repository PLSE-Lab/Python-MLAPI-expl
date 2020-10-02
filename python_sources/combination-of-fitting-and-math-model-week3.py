#!/usr/bin/env python
# coding: utf-8

# # Introduction
# To predict the number of confirmed/fatal cases with COVID-19, we will take the following steps.
# 
# 1. Predict the values using Gamma CDF curve fitting
# 2. Predict the values using logistic curve fitting of recovered cases and SIR-F model
# 3. Calculate the mean value of 1. and 2.

# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


from collections import defaultdict
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import os
from pprint import pprint
import warnings
from fbprophet import Prophet
from fbprophet.plot import add_changepoints_to_plot
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
from matplotlib.ticker import ScalarFormatter
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import optuna
optuna.logging.disable_default_handler()
import pandas as pd
pd.plotting.register_matplotlib_converters()
import seaborn as sns
from scipy.optimize import curve_fit
from scipy.integrate import solve_ivp
import pystan.misc


# In[ ]:


plt.style.use("seaborn-ticks")
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["font.size"] = 11.0
plt.rcParams["figure.figsize"] = (9, 6)


# In[ ]:


warnings.simplefilter("ignore")


# ## Gamma Curve Fitting - Overview of This Approach
# 
# This notebook is a followup to some previous effort of Bill Holst (https://www.kaggle.com/wjholst), Daner Ferhadi (https://www.kaggle.com/dferhadi) and Dan Pearson (https://www.kaggle.com/dan89pearson).
# See:
# * https://www.kaggle.com/dferhadi/logistic-curve-fit-parameter-tuning
# * https://www.kaggle.com/wjholst/covid-19-growth-patterns-in-critical-countries
# * https://www.youtube.com/watch?v=Kas0tIxDvrg&t=35s 
# 
# 
# Both of us used a logistic model to both predict and to identify critial inflection points in the growth model. You can observer from early track of the virus growth, that the initial rate is exponential. Eventually the curve tends to flatten and turn down. That is when the curve begins to take on the sigmoid properties.
# 
# However, when you observe these events over time, the probability distribution does not look normal but rather skewed with a long right tail. That is why this model uses a gamma pdf, which can be tuned to more realistically fit the actual distributions. 
# 
# This approach is not a machine learning effort, but rather employs the Python curve_fit library to find the closest fit for each population group. I use Daner's code base and use the gamma function to formulate the predictions.
# 
# 
# 
# 

# In[ ]:


path = "/kaggle/input/covid19-global-forecasting-week-3/"


# In[ ]:


train_data = pd.read_csv(path+"train.csv")
#train_data = train_data[(train_data.Country_Region=="US") ]
train_df = train_data

train_df['area'] = [str(i)+str(' - ')+str(j) for i,j in zip(train_data['Country_Region'], train_data['Province_State'])]
train_df['Date'] = pd.to_datetime(train_df['Date'])
full_data = train_df


# In[ ]:


#train_df.tail(50)


# In[ ]:


today = full_data['Date'].max()+timedelta(days=1) 
print (today)
print ('Last update of this dataset was ' + str(train_df.loc[len(train_df)-1]['Date']))


# In[ ]:


def get_country_data(train_df, area, metric):
    country_data = train_df[train_df['area']==area]
    country_data = country_data.drop(['Id','Province_State', 'Country_Region'], axis=1)
    country_data = pd.pivot_table(country_data, values=['ConfirmedCases','Fatalities'], index=['Date'], aggfunc=np.sum) 
    country_data = country_data[country_data[metric]!=0]
    tmp = country_data.shift(periods=1,fill_value=0)
    country_data['prior_confirmed'] = tmp.ConfirmedCases
    country_data['prior_deaths'] = tmp.Fatalities
    country_data['DailyConfirmed'] = country_data.ConfirmedCases - country_data.prior_confirmed
    country_data['DailyFatalities'] = country_data.Fatalities - country_data.prior_deaths
    return country_data        


# In[ ]:


area_info = pd.DataFrame(columns=['area', 'cases_start_date', 'deaths_start_date', 'init_ConfirmedCases', 'init_Fatalities','init_DailyConfirmed','init_DailyFatalities'])
for i in range(len(train_df['area'].unique())):
    area = train_df['area'].unique()[i]
    area_cases_data = get_country_data(train_df, area, 'ConfirmedCases')
    #print (area_cases_data)
   
    area_deaths_data = get_country_data(train_df, area, 'Fatalities')
    cases_start_date = area_cases_data.index.min()
    deaths_start_date = area_deaths_data.index.min()
    if len(area_cases_data) > 0:
        confirmed_cases = max(area_cases_data['ConfirmedCases'])
        last = area_cases_data.tail(1)
        last_daily = np.float(last['DailyConfirmed'])

    else:
        confirmed_cases = 0
        last_daily = 0
    if len(area_deaths_data) > 0:
        fatalities = max(area_deaths_data['Fatalities'])
        last = area_deaths_data.tail(1)
        last_death = np.float(last['DailyFatalities'])
    else:
        fatalities = 0
        last_death = 0
    #print (last_daily)
    #print (last_death)
    area_info.loc[i] = [area, cases_start_date, deaths_start_date, confirmed_cases, fatalities,last_daily,last_death]
area_info = area_info.fillna(pd.to_datetime(today))
area_info['init_cases_day_no'] = pd.to_datetime(today)-area_info['cases_start_date']
area_info['init_cases_day_no'] = area_info['init_cases_day_no'].dt.days.fillna(0).astype(int)
area_info['init_deaths_day_no'] = pd.to_datetime(today)-area_info['deaths_start_date']
area_info['init_deaths_day_no'] = area_info['init_deaths_day_no'].dt.days.fillna(0).astype(int)
#area_info['init_DailyConfirmed'] = last_daily.astype(float)
#area_info['init_DailyFatalities'] = last_death.astype(float)
area_info.head()


# In[ ]:


def make_cdf (y):
    cdf = []
    for i in range(1,len(y)+1): 
        total = np.sum(y[:i])
        #print (total)
        cdf.append(total)
        #print (cdf)
    return cdf


# In[ ]:


from scipy.special import factorial
def gamma_pdf(x, k, lam, ymax):

    k = np.float(k)
    #print ('k is ' + str(k))
    
    num = ymax * (np.power(lam,k) * np.power(x,(k-1)) * np.exp(-lam*x))
    if k < 0.5:
        k = 1
    else:
         k = np.round(k)    
    den = (factorial (k-1))
    return num/den

    
def gamma_fit(train_df, area, metric,to_fit, est_count):
    area_data = get_country_data(train_df, area, metric)
    x_data = range(len(area_data.index))
    y_data = area_data[to_fit]
    x_data = np.array(x_data,dtype='float64')
    y_data = np.array(y_data,dtype='float64')
    #x_data = x_data.ravel()
    #y_data = y_data.ravel()
    #_data = np.asarray(x_data).ravel()
    #y_data = np.asarray(y_data).ravel()
    #print (y_data)
    if len(y_data) < 5:
        estimated_k = 6  
        estimated_lam = 0.1 
        ymax = np.float(est_count)
    elif max(y_data) == 0:
        estimated_k = 6  
        estimated_lam = 0.1 
        ymax = np.float(est_count)
    else:
        
        p0_est=[6.0 ,0.1,est_count]
        try:
            popt, pcov = curve_fit(gamma_pdf, x_data, y_data,bounds=([0,0,0],100000000),p0=p0_est, maxfev=1000000)
                                   #bounds=([0,0,0],100000000), p0=p0_est, maxfev=1000000)
            estimated_k, estimated_lam, ymax = popt
        except RuntimeError:
            print(area)
            print("Runtime Error - curve_fit failed") 
            estimated_k = 6  
            estimated_lam = 0.1 
            ymax = est_count
        #else:
        #    print(area)
        #    print("Catch all Error - curve_fit failed") 
        #    estimated_k = 5  
        #    estimated_lam = 0.1 
        #    ymax = est_count

    estimated_parameters = pd.DataFrame(np.array([[area, estimated_k, estimated_lam, ymax]]), columns=['area', 'k', 'lam', 'ymax'])
    return estimated_parameters


# In[ ]:


def get_parameters(metric, to_fit):
    parameters = pd.DataFrame(columns=['area', 'k', 'lam', 'ymax'], dtype=np.float)
    for area in train_df['area'].unique():
        #print ('Area fitting is ' + area)
        if metric == 'ConfirmedCases':
            init = area_info[area_info.area == area]['init_ConfirmedCases']
        else:
            init = area_info[area_info.area == area]['init_Fatalities']
        init = init.astype(float)
        #print (init)
        # establish an initial guess for maxy
        est_count = init * 4.0
        #print (est_count)
        estimated_parameters = gamma_fit(train_df, area, metric, to_fit, est_count)
        parameters = parameters.append(estimated_parameters)
    if True:
        try:
            parameters['k'] = pd.to_numeric(parameters['k'], downcast="float")
            parameters['lam'] = pd.to_numeric(parameters['lam'], downcast="float")
            parameters['ymax'] = pd.to_numeric(parameters['ymax'], downcast="float")
        except RuntimeError: 
            print ('run time error')
        except TypeError:
            print ('type error')
        #else:
        #    print ("error on parameter conversion")
        #parameters = parameters.replace({'k': {-1: parameters[parameters['ymax']>0].median()[0]}, 
        #                                 'lam': {-1: parameters[parameters['ymax']>0].median()[1]}, 
        #                                 'ymax': {-1: parameters[parameters['ymax']>0].median()[2]}})
    return parameters


# In[ ]:


cases_parameters = get_parameters('ConfirmedCases','DailyConfirmed')
cases_parameters.tail(20)


# In[ ]:


deaths_parameters = get_parameters('Fatalities','DailyFatalities')
deaths_parameters.tail(20)


# In[ ]:


fit_df = area_info.merge(cases_parameters, on='area', how='left')
fit_df = fit_df.rename(columns={"k": "cases_k", "lam": "cases_lam", "ymax": "cases_ymax"})
fit_df = fit_df.merge(deaths_parameters, on='area', how='left')
fit_df = fit_df.rename(columns={"k": "deaths_k", "lam": "deaths_lam", "ymax": "deaths_ymax"})

fit_df.head()


# In[ ]:


test_data = pd.read_csv(path+"test.csv")
test_df = test_data.copy()
#test_df = test_data[(test_data.Country_Region=="US") & (test_data.Province_State != 'x')].copy()
test_df['area'] = [str(i)+str(' - ')+str(j) for i,j in zip(test_df['Country_Region'], test_df['Province_State'])]

test_df = test_df.merge(fit_df, on='area', how='left')
test_df = test_df.merge(cases_parameters, on='area', how='left')
#print (len(test_df))

#test_df = test_df.rename(columns={"k": "cases_k", "lam": "cases_lam", "ymax": "cases_ymax"})
#test_df = test_df.merge(deaths_parameters, on='area', how='left')
#test_df = test_df.rename(columns={"k": "deaths_k", "lam": "deaths_lam", "ymax": "deaths_ymax"})
#test_df.cases_k = test_df.cases_k.astype(float)
#test_df.deaths_k = test_df.deaths_k.astype(float)
#for i,t in test_df.iterrows():
#    print (i)
#    print (t.area)
#    print (t.cases_k)
test_df['Date'] = pd.to_datetime(test_df['Date'])
test_df['cases_start_date'] = pd.to_datetime(test_df['cases_start_date'])
test_df['deaths_start_date'] = pd.to_datetime(test_df['deaths_start_date'])

test_df['cases_day_no'] = test_df['Date']-test_df['cases_start_date']
test_df['cases_day_no'] = test_df['cases_day_no'].dt.days.fillna(0).astype(int)
test_df['deaths_day_no'] = test_df['Date']-test_df['deaths_start_date']
test_df['deaths_day_no'] = test_df['deaths_day_no'].dt.days.fillna(0).astype(int)
test_df['DailyFatalities_fit'] = 0


# In[ ]:


fit_df[(fit_df.area>'US') & (fit_df.area < 'UT')]


# In[ ]:


#y = gamma_pdf(d, ['cases_k']), t['cases_lam'], t['cases_ymax'])
pred_yd = []
pred_yc = []
for (idx, df) in test_df.iterrows():
    #print('for death day ' + str(df['deaths_day_no']))
    y  = gamma_pdf(df['deaths_day_no'], df['deaths_k'], df['deaths_lam'], df['deaths_ymax'])
    #print (y)
    pred_yd.append([df.area,y])
    #print ('for confirmed day ' + str(df['cases_day_no']))
    yc = gamma_pdf(df['cases_day_no'], df['cases_k'], df['cases_lam'], df['cases_ymax'])
    pred_yc.append([df.area,yc])
    #test_df['DailyCases_pred'] = round(test_df['DailyConfirmed_fit']+test_df['DailyConfirmed_error'])

    #test_df['DailyFatalities_pred'] = round(test_df['DailyFatalities_fit']+test_df['DailyFatalities_error'])


# In[ ]:


yd_df = pd.DataFrame( pred_yd)
yc_df = pd.DataFrame( pred_yc)


# In[ ]:


yc_df.columns = ['Area','Predicted']
yd_df.columns = ['Area','Predicted']


# In[ ]:


def make_pred(df):
    cdf_all = pd.DataFrame()
    for a in df['Area'].unique():
        tmp = df[df.Area==a]
        cdf = make_cdf (tmp.Predicted)
        cdf = pd.DataFrame(cdf)
        cdf_all = pd.concat([cdf_all, cdf])
    return cdf_all

cdfc = make_pred(yc_df)
cdfd = make_pred(yd_df)
cdfc.columns =['Pred']
cdfd.columns =['Pred']


# In[ ]:


test_df['DailyFatalities_fit'] = np.round(cdfd.Pred.values)
test_df['DailyCases_fit'] = np.round(cdfc.Pred.values)


# In[ ]:


# generate submission
submission = pd.DataFrame(data={'ForecastId': test_df['ForecastId'], 'ConfirmedCases': test_df['DailyCases_fit'], 'Fatalities': test_df['DailyFatalities_fit']}).fillna(0.5)
# submission.to_csv("/kaggle/working/submission.csv", index=False)


# In[ ]:


submission.head()


# # Logistic curve fitting and SIR-F model
# Using curve fitting method and SIR-F model, we will predict the number of confirmed cases and fatal cases with COVID-19 global data. SIR-F model was created in another notebook of an auther. Please refer to the references.  
# 
# Contents:
# * Preparation
# * Prediction of the number of recovered cases with logistic curve
# * Prameter estimation with SIR-F model
# * Prediction of global data
# * Data submission
# 
# References:
# * [COVID-19 - Growth of Virus in Specific Countries](https://www.kaggle.com/wjholst/covid-19-growth-of-virus-in-specific-countries) by Bill Holst
# * [COVID-19 data with SIR model](https://www.kaggle.com/lisphilar/covid-19-data-with-sir-model) by Lisphilar

# ## Tool

# In[ ]:


def line_plot(df, title, ylabel="Cases", h=None, v=None,
              xlim=(None, None), ylim=(0, None), math_scale=True, y_logscale=False, y_integer=False):
    """
    Show chlonological change of the data.
    """
    ax = df.plot()
    if math_scale:
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style="sci",  axis="y",scilimits=(0, 0))
    if y_logscale:
        ax.set_yscale("log")
    if y_integer:
        fmt = matplotlib.ticker.ScalarFormatter(useOffset=False)
        fmt.set_scientific(False)
        ax.yaxis.set_major_formatter(fmt)
    ax.set_title(title)
    ax.set_xlabel(None)
    ax.set_ylabel(ylabel)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.legend(bbox_to_anchor=(1.02, 0), loc="lower left", borderaxespad=0)
    if h is not None:
        ax.axhline(y=h, color="black", linestyle="--")
    if v is not None:
        if not isinstance(v, list):
            v = [v]
        for value in v:
            ax.axvline(x=value, color="black", linestyle="--")
    plt.tight_layout()
    plt.show()


# In[ ]:


def select_area(ncov_df, group="Date", places=None, areas=None, excluded_places=None,
                start_date=None, end_date=None, date_format="%d%b%Y"):
    """
    Select the records of the palces.
    @ncov_df <pd.DataFrame>: the clean data
    @group <str or None>: group-by the group, or not perform (None)
    @area or @places:
        if ncov_df has Country and Province column,
            @places <list[tuple(<str/None>, <str/None>)]: the list of places
                - if the list is None, all data will be used
                - (str, str): both of country and province are specified
                - (str, None): only country is specified
                - (None, str) or (None, None): Error
        if ncov_df has Area column,
            @areas <list[str]>: the list of area names
                - if the list is None, all data will be used
                - eg. Japan
                - eg. US/California
    @excluded_places <list[tuple(<str/None>, <str/None>)]: the list of excluded places
        - if the list is None, all data in the "places" will be used
        - (str, str): both of country and province are specified
        - (str, None): only country is specified
        - (None, str) or (None, None): Error
    @start_date <str>: the start date or None
    @end_date <str>: the start date or None
    @date_format <str>: format of @start_date and @end_date
    @return <pd.DataFrame>: index and columns are as same as @ncov_df
    """
    # Select the target records
    df = ncov_df.copy()
    if (places is not None) or (excluded_places is not None):
        c_series = df["Country"]
        p_series = df["Province"]
        if places is not None:
            df = pd.DataFrame(columns=ncov_df.columns)
            for (c, p) in places:
                if c is None:
                    raise Exception("places: Country must be specified!")
                if p is None:
                    new_df = ncov_df.loc[c_series == c, :]
                else:
                    new_df = ncov_df.loc[(c_series == c) & (p_series == p), :]
                df = pd.concat([df, new_df], axis=0)
        if excluded_places is not None:
            for (c, p) in excluded_places:
                if c is None:
                    raise Exception("excluded_places: Country must be specified!")
                if p is None:
                    df = df.loc[c_series != c, :]
                else:
                    c_df = df.loc[(c_series == c) & (p_series != p), :]
                    other_df = df.loc[c_series != c, :]
                    df = pd.concat([c_df, other_df], axis=0)
    if areas is not None:
        df = df.loc[df["Area"].isin(areas), :]
    if group is not None:
        df = df.groupby(group).sum().reset_index()
    # Range of date
    if start_date is not None:
        df = df.loc[df["Date"] >= datetime.strptime(start_date, date_format), :]
    if end_date is not None:
        df = df.loc[df["Date"] <= datetime.strptime(end_date, date_format), :]
    # Only use the records with Confirmed > 0
    try:
        df = df.loc[df["Confirmed"] > 0, :]
    except KeyError:
        pass
    # Aleart empty
    if df.empty:
        raise Exception("The output dataframe is empty!")
    return df


# In[ ]:


def show_trend(ncov_df, name=None, variable="Confirmed", n_changepoints=2, **kwargs):
    """
    Show trend of log10(@variable) using fbprophet package.
    @ncov_df <pd.DataFrame>: the clean data
    @variable <str>: variable name to analyse
        - if Confirmed, use Infected + Recovered + Deaths
    @n_changepoints <int>: max number of change points
    @kwargs: keword arguments of select_area()
    """
    # Data arrangement
    df = select_area(ncov_df, **kwargs)
    df = df.loc[:, ["Date", variable]]
    df.columns = ["ds", "y"]
    # Log10(x)
    warnings.resetwarnings()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df["y"] = np.log10(df["y"]).replace([np.inf, -np.inf], 0)
    # fbprophet
    model = Prophet(growth="linear", daily_seasonality=False, n_changepoints=n_changepoints)
    model.fit(df)
    future = model.make_future_dataframe(periods=0)
    forecast = model.predict(future)
    # Create figure
    fig = model.plot(forecast)
    _ = add_changepoints_to_plot(fig.gca(), model, forecast)
    if name is None:
        try:
            name = f"{kwargs['places'][0][0]}: "
        except Exception:
            name = str()
    else:
        name = f"{name}: "
    plt.title(f"{name}log10({variable}) over time and chainge points")
    plt.ylabel(f"log10(the number of cases)")
    plt.xlabel("")


# In[ ]:


def create_target_df(ncov_df, total_population,
                     confirmed="Confirmed", recovered="Recovered", fatal="Deaths", **kwargs):
    """
    Select the records of the places, calculate the number of susceptible people,
     and calculate the elapsed time [day] from the start date of the target dataframe.
    @ncov_df <pd.DataFrame>: the clean data
    @total_population <int>: total population in the places
    column names in @ncov_df:
        @confirmed <str>: column name of the number of confirmed cases
        @recovered <str>: column name of the number of recovered cases
        @fatal <str>: column name of the number of fatal cases
    @kwargs: keword arguments of select_area()
    @return <tuple(2 objects)>:
        - 1. first_date <pd.Timestamp>: the first date of the selected records
        - 2. target_df <pd.DataFrame>:
            - column T: elapsed time [min] from the start date of the dataset
            - column Susceptible: the number of patients who are in the palces but not infected/recovered/died
            - column Infected: the number of infected cases
            - column Recovered: the number of recovered cases
            - column Deaths: the number of death cases
    """
    # Select the target records
    df = select_area(ncov_df, **kwargs)
    first_date = df.loc[df.index[0], "Date"]
    # column T
    df["T"] = ((df["Date"] - first_date).dt.total_seconds() / 60).astype(int)
    # coluns except T
    cols = [confirmed, recovered, fatal]
    if not set(cols).issubset(set(df.columns)):
        raise KeyError(f"ncov_df must have {', '.join(cols)} column!")
    df["Susceptible"] = total_population - df[confirmed]
    df["Infected"] = df[confirmed] - df[recovered] - df[fatal]
    df["Recovered"] = df[recovered]
    df["Fatal"] = df.loc[:, fatal]
    response_variables = ["Susceptible", "Infected", "Recovered", "Fatal"]
    # Return
    target_df = df.loc[:, ["T", *response_variables]]
    return (first_date, target_df)


# In[ ]:


def simulation(model, initials, step_n, **params):
    """
    Solve ODE of the model.
    @model <ModelBase>: the model
    @initials <tuple[float]>: the initial values
    @step_n <int>: the number of steps
    @params: the paramerters of the model
    """
    tstart, dt, tend = 0, 1, step_n
    sol = solve_ivp(
        fun=model(**params),
        t_span=[tstart, tend],
        y0=np.array(initials, dtype=np.float64),
        t_eval=np.arange(tstart, tend + dt, dt),
        dense_output=False  # True
    )
    t_df = pd.Series(data=sol["t"], name="t")
    y_df = pd.DataFrame(data=sol["y"].T.copy(), columns=model.VARIABLES)
    sim_df = pd.concat([t_df, y_df], axis=1)
    return sim_df


# In[ ]:


class ModelBase(object):
    NAME = "Model"
    VARIABLES = ["x"]
    PRIORITIES = np.array([1])
    QUANTILE_RANGE = [0.3, 0.7]
    MONOTONIC = ["x"]

    @classmethod
    def param_dict(cls, train_df_divided=None, q_range=None):
        """
        Define parameters without tau. This function should be overwritten.
        @train_df_divided <pd.DataFrame>:
            - column: t and non-dimensional variables
        @q_range <list[float, float]>: quantile rage of the parameters calculated by the data
        @return <dict[name]=(min, max):
            @min <float>: min value
            @max <float>: max value
        """
        param_dict = dict()
        return param_dict

    @staticmethod
    def calc_variables(df):
        """
        Calculate the variables of the model.
        This function should be overwritten.
        @df <pd.DataFrame>
        @return <pd.DataFrame>
        """
        return df

    @staticmethod
    def calc_variables_reverse(df):
        """
        Calculate measurable variables using the variables of the model.
        This function should be overwritten.
        @df <pd.DataFrame>
        @return <pd.DataFrame>
        """
        return df

    @classmethod
    def create_dataset(cls, ncov_df, total_population, **kwargs):
        """
        Create dataset with the model-specific varibles.
        The variables will be divided by total population.
        The column names (not include T) will be lower letters.
        **kwargs: See the function named create_target_df()
        @return <tuple(objects)>:
            - start_date <pd.Timestamp>
            - initials <tuple(float)>: the initial values
            - Tend <int>: the last value of T
            - df <pd.DataFrame>: the dataset
        """
        start_date, target_df = create_target_df(ncov_df, total_population, **kwargs)
        df = cls.calc_variables(target_df).set_index("T") / total_population
        df.columns = [n.lower() for n in df.columns]
        initials = df.iloc[0, :].values
        df = df.reset_index()
        Tend = df.iloc[-1, 0]
        return (start_date, initials, Tend, df)

    def calc_r0(self):
        """
        Calculate R0. This function should be overwritten.
        """
        return None

    def calc_days_dict(self, tau):
        """
        Calculate 1/beta [day] etc.
        This function should be overwritten.
        @param tau <int>: tau value [hour]
        """
        return dict()


# In[ ]:


class SIRF(ModelBase):
    NAME = "SIR-F"
    VARIABLES = ["x", "y", "z", "w"]
    PRIORITIES = np.array([1, 10, 10, 2])
    MONOTONIC = ["z", "w"]

    def __init__(self, theta, kappa, rho, sigma):
        super().__init__()
        self.theta = theta
        self.kappa = kappa
        self.rho = rho
        self.sigma = sigma

    def __call__(self, t, X):
        # x, y, z, w = [X[i] for i in range(len(self.VARIABLES))]
        # dxdt = - self.rho * x * y
        # dydt = self.rho * (1 - self.theta) * x * y - (self.sigma + self.kappa) * y
        # dzdt = self.sigma * y
        # dwdt = self.rho * self.theta * x * y + self.kappa * y
        dxdt = - self.rho * X[0] * X[1]
        dydt = self.rho * (1 - self.theta) * X[0] * X[1] - (self.sigma + self.kappa) * X[1]
        dzdt = self.sigma * X[1]
        dwdt = self.rho * self.theta * X[0] * X[1] + self.kappa * X[1]
        return np.array([dxdt, dydt, dzdt, dwdt])

    @classmethod
    def param_dict(cls, train_df_divided=None, q_range=None):
        param_dict = super().param_dict()
        q_range = super().QUANTILE_RANGE[:] if q_range is None else q_range
        param_dict["theta"] = (0, 1)
        param_dict["kappa"] = (0, 1)
        if train_df_divided is not None:
            df = train_df_divided.copy()
            # rho = - (dx/dt) / x / y
            rho_series = 0 - df["x"].diff() / df["t"].diff() / df["x"] / df["y"]
            param_dict["rho"] = rho_series.quantile(q_range)
            # sigma = (dz/dt) / y
            sigma_series = df["z"].diff() / df["t"].diff() / df["y"]
            param_dict["sigma"] = sigma_series.quantile(q_range)
            return param_dict
        param_dict["rho"] = (0, 1)
        param_dict["sigma"] = (0, 1)
        return param_dict

    @staticmethod
    def calc_variables(df):
        df["X"] = df["Susceptible"]
        df["Y"] = df["Infected"]
        df["Z"] = df["Recovered"]
        df["W"] = df["Fatal"]
        return df.loc[:, ["T", "X", "Y", "Z", "W"]]

    @staticmethod
    def calc_variables_reverse(df):
        df["Susceptible"] = df["X"]
        df["Infected"] = df["Y"]
        df["Recovered"] = df["Z"]
        df["Fatal"] = df["W"]
        return df

    def calc_r0(self):
        try:
            r0 = self.rho * (1 - self.theta) / (self.sigma + self.kappa)
        except ZeroDivisionError:
            return np.nan
        return round(r0, 2)

    def calc_days_dict(self, tau):
        _dict = dict()
        _dict["alpha1 [-]"] = round(self.theta, 3)
        if self.kappa == 0:
            _dict["1/alpha2 [day]"] = 0
        else:
            _dict["1/alpha2 [day]"] = int(tau / 24 / 60 / self.kappa)
        _dict["1/beta [day]"] = int(tau / 24 / 60 / self.rho)
        if self.sigma == 0:
            _dict["1/gamma [day]"] = 0
        else:
            _dict["1/gamma [day]"] = int(tau / 24 / 60 / self.sigma)
        return _dict


# In[ ]:


class Estimator(object):
    def __init__(self, model, ncov_df, total_population, name=None, places=None, areas=None,
                 excluded_places=None, start_date=None, end_date=None, date_format="%d%b%Y", **params):
        """
        Set training data.
        @model <ModelBase>: the model
        @name <str>: name of the area
        @params: fixed parameter of the model
        @the other params: See the function named create_target_df()
        """
        # Fixed parameters
        self.fixed_param_dict = params.copy()
        if None in params.values():
            self.fixed_param_dict = {
                k: v for (k, v) in params.items() if v is not None
            }
        # Register the dataset arranged for the model
        dataset = model.create_dataset(
            ncov_df, total_population, places=places, areas=areas,
            excluded_places=excluded_places,
            start_date=start_date, end_date=end_date, date_format=date_format
        )
        self.start_time, self.initials, self.Tend, self.train_df = dataset
        self.total_population = total_population
        self.name = name
        self.model = model
        self.param_dict = dict()
        self.study = None
        self.optimize_df = None

    def run(self, n_trials=500):
        """
        Try estimation (optimization of parameters and tau).
        @n_trials <int>: the number of trials
        """
        if self.study is None:
            self.study = optuna.create_study(direction="minimize")
        self.study.optimize(
            lambda x: self.objective(x),
            n_trials=n_trials,
            n_jobs=-1
        )
        param_dict = self.study.best_params.copy()
        param_dict.update(self.fixed_param_dict)
        param_dict["R0"] = self.calc_r0()
        param_dict["score"] = self.score()
        param_dict.update(self.calc_days_dict())
        self.param_dict = param_dict.copy()
        return param_dict

    def history_df(self):
        """
        Return the hsitory of optimization.
        @return <pd.DataFrame>
        """
        optimize_df = self.study.trials_dataframe()
        optimize_df["time[s]"] = optimize_df["datetime_complete"] -             optimize_df["datetime_start"]
        optimize_df["time[s]"] = optimize_df["time[s]"].dt.total_seconds()
        self.optimize_df = optimize_df.drop(
            ["datetime_complete", "datetime_start", "system_attrs__number"], axis=1)
        return self.optimize_df.sort_values("value", ascending=True)

    def history_graph(self):
        """
        Show the history of parameter search using pair-plot.
        """
        if self.optimize_df is None:
            self.history_df()
        df = self.optimize_df.copy()
        sns.pairplot(df.loc[:, df.columns.str.startswith(
            "params_")], diag_kind="kde", markers="+")
        plt.show()

    def objective(self, trial):
        # Time
        try:
            tau = self.fixed_param_dict["tau"]
        except KeyError:
            tau = trial.suggest_int("tau", 1, 1440)
        train_df_divided = self.train_df.copy()
        train_df_divided["t"] = (train_df_divided["T"] / tau).astype(np.int64)
        # Parameters
        param_dict = self.model.param_dict(train_df_divided)
        p_dict = {"tau": None}
        p_dict.update(
            {
                k: trial.suggest_uniform(k, *v)
                for (k, v) in param_dict.items()
            }
        )
        p_dict.update(self.fixed_param_dict)
        p_dict.pop("tau")
        # Simulation
        t_end = train_df_divided.loc[train_df_divided.index[-1], "t"]
        sim_df = simulation(self.model, self.initials, step_n=t_end, **p_dict)
        return self.error_f(train_df_divided, sim_df)

    def error_f(self, train_df_divided, sim_df):
        """
        We need to minimize the difference of the observed values and estimated values.
        This function calculate the difference of the estimated value and obsereved value.
        """
        n = self.total_population
        df = pd.merge(train_df_divided, sim_df, on="t", suffixes=("_observed", "_estimated"))
        diffs = [
            # Weighted Average: the recent data is more important
            p * np.average(
                abs(df[f"{v}_observed"] - df[f"{v}_estimated"]) / (df[f"{v}_observed"] * n + 1),
                weights=df["t"]
            )
            for (p, v) in zip(self.model.PRIORITIES, self.model.VARIABLES)
        ]
        return sum(diffs) * n

    def compare_df(self):
        """
        Show the taining data and simulated data in one dataframe.

        """
        est_dict = self.study.best_params.copy()
        est_dict.update(self.fixed_param_dict)
        tau = est_dict["tau"]
        est_dict.pop("tau")
        observed_df = self.train_df.drop("T", axis=1)
        observed_df["t"] = (self.train_df["T"] / tau).astype(int)
        t_end = observed_df.loc[observed_df.index[-1], "t"]
        sim_df = simulation(self.model, self.initials, step_n=t_end, **est_dict)
        df = pd.merge(observed_df, sim_df, on="t", suffixes=("_observed", "_estimated"))
        df = df.set_index("t")
        return df

    def compare_graph(self):
        """
        Compare obsereved and estimated values in graphs.
        """
        df = self.compare_df()
        use_variables = [
            v for (i, (p, v)) in enumerate(zip(self.model.PRIORITIES, self.model.VARIABLES))
            if p != 0 and i != 0
        ]
        val_len = len(use_variables) + 1
        fig, axes = plt.subplots(
            ncols=1, nrows=val_len, figsize=(9, 6 * val_len / 2))
        for (ax, v) in zip(axes.ravel()[1:], use_variables):
            df[[f"{v}_observed", f"{v}_estimated"]].plot.line(
                ax=ax, ylim=(0, None), sharex=True,
                title=f"{self.model.NAME}: Comparison of observed/estimated {v}(t)"
            )
            ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
            ax.ticklabel_format(style="sci",  axis="y", scilimits=(0, 0))
            ax.legend(bbox_to_anchor=(1.02, 0),
                      loc="lower left", borderaxespad=0)
        for v in use_variables:
            df[f"{v}_diff"] = df[f"{v}_observed"] - df[f"{v}_estimated"]
            df[f"{v}_diff"].plot.line(
                ax=axes.ravel()[0], sharex=True,
                title=f"{self.model.NAME}: observed - estimated"
            )
        axes.ravel()[0].axhline(y=0, color="black", linestyle="--")
        axes.ravel()[0].yaxis.set_major_formatter(
            ScalarFormatter(useMathText=True))
        axes.ravel()[0].ticklabel_format(
            style="sci",  axis="y", scilimits=(0, 0))
        axes.ravel()[0].legend(bbox_to_anchor=(1.02, 0),
                               loc="lower left", borderaxespad=0)
        fig.tight_layout()
        fig.show()

    def calc_r0(self):
        """
        Calculate R0.
        """
        est_dict = self.study.best_params.copy()
        est_dict.update(self.fixed_param_dict)
        est_dict.pop("tau")
        model_instance = self.model(**est_dict)
        return model_instance.calc_r0()

    def calc_days_dict(self):
        """
        Calculate 1/beta etc.
        """
        est_dict = self.study.best_params.copy()
        est_dict.update(self.fixed_param_dict)
        tau = est_dict["tau"]
        est_dict.pop("tau")
        model_instance = self.model(**est_dict)
        return model_instance.calc_days_dict(tau)

    def predict_df(self, step_n):
        """
        Predict the values in the future.
        @step_n <int>: the number of steps
        @return <pd.DataFrame>: predicted data for measurable variables.
        """
        est_dict = self.study.best_params.copy()
        est_dict.update(self.fixed_param_dict)
        tau = est_dict["tau"]
        est_dict.pop("tau")
        df = simulation(self.model, self.initials, step_n=step_n, **est_dict)
        df["Time"] = (
            df["t"] * tau).apply(lambda x: timedelta(minutes=x)) + self.start_time
        df = df.set_index("Time").drop("t", axis=1)
        df = (df * self.total_population).astype(np.int64)
        upper_cols = [n.upper() for n in df.columns]
        df.columns = upper_cols
        df = self.model.calc_variables_reverse(df).drop(upper_cols, axis=1)
        return df

    def predict_graph(self, step_n, name=None, excluded_cols=None):
        """
        Predict the values in the future and create a figure.
        @step_n <int>: the number of steps
        @name <str>: name of the area
        @excluded_cols <list[str]>: the excluded columns in the figure
        """
        if self.name is not None:
            name = self.name
        else:
            name = str() if name is None else name
        df = self.predict_df(step_n=step_n)
        if excluded_cols is not None:
            df = df.drop(excluded_cols, axis=1)
        r0 = self.param_dict["R0"]
        title = f"Prediction in {name} with {self.model.NAME} model: R0 = {r0}"
        line_plot(df, title, v=datetime.today(), h=self.total_population)

    def rmsle(self, compare_df):
        """
        Return the value of RMSLE.
        @param compare_df <pd.DataFrame>
        """
        df = compare_df.set_index("t") * self.total_population
        score = 0
        for (priority, v) in zip(self.model.PRIORITIES, self.model.VARIABLES):
            if priority == 0:
                continue
            observed, estimated = df[f"{v}_observed"], df[f"{v}_estimated"]
            diff = (np.log(observed + 1) - np.log(estimated + 1))
            score += (diff ** 2).sum()
        rmsle = np.sqrt(score / len(df))
        return rmsle

    def score(self):
        """
        Return the value of RMSLE.
        """
        rmsle = self.rmsle(self.compare_df().reset_index("t"))
        return rmsle

    def info(self):
        """
        Return Estimater information.
        @return <tupple[object]>:
            - <ModelBase>: model
            - <dict[str]=str>: name, total_population, start_time, tau
            - <dict[str]=float>: values of parameters of model
        """
        param_dict = self.study.best_params.copy()
        param_dict.update(self.fixed_param_dict)
        info_dict = {
            "name": self.name,
            "total_population": self.total_population,
            "start_time": self.start_time,
            "tau": param_dict["tau"],
            "initials": self.initials
        }
        param_dict.pop("tau")
        return (self.model, info_dict, param_dict)


# In[ ]:


class Predicter(object):
    """
    Predict the future using models.
    """
    def __init__(self, name, total_population, start_time, tau, initials, date_format="%d%b%Y"):
        """
        @name <str>: place name
        @total_population <int>: total population
        @start_time <datatime>: the start time
        @tau <int>: tau value (time step)
        @initials <list/tupple/np.array[float]>: initial values of the first model
        @date_format <str>: date format to display in figures
        """
        self.name = name
        self.total_population = total_population
        self.start_time = start_time
        self.tau = tau
        self.date_format = date_format
        # Un-fixed
        self.last_time = start_time
        self.axvlines = list()
        self.initials = initials
        self.df = pd.DataFrame()
        self.title_list = list()
        self.reverse_f = lambda x: x

    def add(self, model, end_day_n=None, count_from_last=False, vline=True, **param_dict):
        """
        @model <ModelBase>: the epidemic model
        @end_day_n <int/None>: day number of the end date (0, 1, 2,...), or None (now)
            - if @count_from_last <bool> is True, start point will be the last date registered to Predicter
        @vline <bool>: if True, vertical line will be shown at the end date
        @**param_dict <dict>: keyword arguments of the model
        """
        # Validate day nubber, and calculate step number
        if end_day_n is None:
            end_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        else:
            if count_from_last:
                end_time = self.last_time + timedelta(days=end_day_n)
            else:
                end_time = self.start_time + timedelta(days=end_day_n)
        if end_time <= self.last_time:
            raise Exception(f"Model on {end_time.strftime(self.date_format)} has been registered!")
        step_n = int((end_time - self.last_time).total_seconds() / 60 / self.tau)
        self.last_time = end_time
        # Perform simulation
        new_df = simulation(model, self.initials, step_n=step_n, **param_dict)
        new_df["t"] = new_df["t"] + len(self.df)
        self.df = pd.concat([self.df, new_df.iloc[1:, :]], axis=0).fillna(0)
        self.initials = new_df.set_index("t").iloc[-1, :]
        # For title
        if vline:
            self.axvlines.append(end_time)
            r0 = model(**param_dict).calc_r0()
            if len(self.axvlines) == 1:
                self.title_list.append(f"{model.NAME}(R0={r0}, -{end_time.strftime(self.date_format)})")
            else:
                self.title_list.append(f"{model.NAME}({r0}, -{end_time.strftime(self.date_format)})")
        # Update reverse function (X, Y,.. to Susceptible, Infected,...)
        self.reverse_f = model.calc_variables_reverse
        return self

    def restore_df(self, min_infected=1):
        """
        Return the dimentional simulated data.
        @min_infected <int>: if Infected < min_infected, the records will not be used
        @return <pd.DataFrame>
        """
        df = self.df.copy()
        df["Time"] = self.start_time + df["t"].apply(lambda x: timedelta(minutes=x * self.tau))
        df = df.drop("t", axis=1).set_index("Time") * self.total_population
        df = df.astype(np.int64)
        upper_cols = [n.upper() for n in df.columns]
        df.columns = upper_cols
        df = self.reverse_f(df).drop(upper_cols, axis=1)
        df = df.loc[df["Infected"] >= min_infected, :]
        return df

    def restore_graph(self, drop_cols=None, min_infected=1, **kwargs):
        """
        Show the dimentional simulate data as a figure.
        @drop_cols <list[str]>: the columns not to be shown
        @min_infected <int>: if Infected < min_infected, the records will not be used
        @kwargs: keyword arguments of line_plot() function
        """
        df = self.restore_df(min_infected=min_infected)
        if drop_cols is not None:
            df = df.drop(drop_cols, axis=1)
        axvlines = [datetime.now(), *self.axvlines] if len(self.axvlines) == 1 else self.axvlines[:]
        line_plot(
            df,
            title=f"{self.name}: {', '.join(self.title_list)}",
            v=axvlines[:-1],
            h=self.total_population,
            **kwargs
        )


# In[ ]:


class Scenario(object):
    """
    Class for scenario analysis.
    """
    SUFFIX_DICT = defaultdict(lambda: "th")
    SUFFIX_DICT.update({1: "st", 2: "nd", 3: "rd"})

    def __init__(self, ncov_df, name, date_format="%d%b%Y", **kwargs):
        """
        @ncov_df <pd.DataFrame>: the cleaned data
        @name <str>: name of the country/area
        @date_format <str>: string format of date
        @kwargs: keyword arguments of select_area() function
        """
        record_df = select_area(ncov_df, **kwargs)
        record_df = record_df.set_index("Date").resample("D").last()
        record_df = record_df.interpolate(method="linear")
        record_df = record_df.loc[:, ["Confirmed", "Infected", "Deaths", "Recovered"]]
        self.record_df = record_df.reset_index()
        self.name = name
        self.date_format = date_format
        self.phase_dict = dict()
        self.estimator_dict = dict()
        self.param_df = pd.DataFrame()
        self.future_phase_dict = dict()
        self.future_param_dict = dict()

    def show_record(self):
        """
        Show the records.
        """
        line_plot(
            self.record_df.drop("Confirmed", axis=1).set_index("Date"),
            f"{self.name}: Cases over time",
            y_integer=True
        )
        return self.record_df

    def growth_factor(self):
        """
        Return growth factor group and the history of growth factor values.
        """
        # Calculate growth factor
        records = self.record_df.set_index("Date")["Confirmed"]
        growth = records.diff() / records.diff().shift(freq="D")
        growth = growth.replace(np.inf, np.nan).fillna(1.0)
        growth = growth.rolling(7).mean()[6:-1].round(2)
        # Group
        more_n = (growth > 1)[::-1].cumprod().sum()
        less_n = (growth < 1)[::-1].cumprod().sum()
        group = "Outbreaking" if more_n >= 7 else "Stopping" if less_n >= 7 else "Crossroad"
        group_df = pd.DataFrame(
            {
                "Group": group,
                "GF > 1 [straight days]": more_n,
                "GF < 1 [straight days]": less_n
            },
            index=[self.name]
        )
        # Growth factor over time
        growth.plot(title=f"{self.name}: Growth factor over time")
        plt.axhline(1.0, color="black", linestyle="--")
        plt.xlabel(None)
        plt.show()
        return group_df
        
    def trend(self, variables=["Confirmed", "Deaths", "Recovered"], **kwargs):
        """
        Perform trend analysis.
        @variables <list[str]>: list of variables
        @kwargs: keyword arguments of show_trend() function
        """
        if "variable" in kwargs.keys():
            raise KeyError("Please use variables argument rather than variable arugument.")
        for val in variables:
            show_trend(self.record_df, name=self.name, variable=val, **kwargs)
        return None

    def set_phase(self, start_dates, population):
        """
        Set phase for hyperparameter estimation.
        @start_dates <list[str]>: list of start dates of the phases
        @population <int or list[int]>: total population or list of total population
        """
        end_dates = [
            (datetime.strptime(s, self.date_format) - timedelta(days=1)).strftime(self.date_format)
            for s in start_dates[1:]
        ]
        end_dates.append(None)
        if isinstance(population, int):
            population_values = [population for _ in range(len(start_dates))]
        elif len(population) == len(start_dates):
            population_values = population[:]
        else:
            raise Exception("start_date and population must have the same length!")
        self.phase_dict = {
            self._num2str(n): {"start_date": s, "end_date": e, "population": p}
            for (n, (s, e, p)) in enumerate(zip(start_dates, end_dates, population_values), 1)
        }
        return pd.DataFrame.from_dict(self.phase_dict, orient="index").fillna("-")

    def estimate(self, model, n_trials=100, same_tau=True):
        """
        Perform hyperparameter estimation.
        @model <ModelBase>: math model
        @n_trials <int>: the number of trials
        @same_tau <bool>:
            whether apply the tau value of first phase to the following phases or not.
        """
        if not self.phase_dict:
            raise Exception("Please use Scenario.set_phase() at first.")
        tau = None
        est_start_time = datetime.now()
        for num in self.phase_dict.keys():
            print(f"Hyperparameter estimation of {num} phase.")
            target_dict = self.phase_dict[num]
            while True:
                # Create estimator
                self.estimator_dict[num] = Estimator(
                    model, self.record_df, target_dict["population"],
                    name=self.name,
                    start_date=target_dict["start_date"],
                    end_date=target_dict["end_date"],
                    date_format=self.date_format,
                    tau=tau
                )
                print("\tEstimator was created.")
                # Run trials
                while True:
                    print(f"\t\t{n_trials} trials", end=" ")
                    est_start_time_run = datetime.now()
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        _ = self.estimator_dict[num].run(n_trials=n_trials)
                    minutes, seconds = divmod(int((datetime.now() - est_start_time_run).total_seconds()), 60)
                    print(f"finished in {minutes} min {seconds} sec.")
                    # Check if estimated in (observed * 0.8, observed * 1.2)
                    compare_df = self.estimator_dict[num].compare_df()
                    targets = [
                        (compare_df[f"{val}_estimated"], compare_df[f"{val}_observed"])
                        for val in model.MONOTONIC
                    ]
                    max_ok = [obs.max() * 0.8 <= est.max() <= obs.max() * 1.2 for (est, obs) in targets]
                    monotonic_ok = [target[0].is_monotonic for target in targets]
                    if all(max_ok) or not all(monotonic_ok):
                        break
                if all(monotonic_ok):
                    print("\tSuccessfully estimated.")
                    break
                vals = [val for (val, ok) in zip(model.MONOTONIC, monotonic_ok) if not ok]
                print(f"\tEstimator will be replaced because estimated {vals[0]} is non-monotonic.")
            tau = self.estimator_dict[num].param_dict["tau"]
        minutes, seconds = divmod(int((datetime.now() - est_start_time).total_seconds()), 60)
        print(f"Total: {minutes} min {seconds} sec.")
        self.show_parameters()

    def accuracy_graph(self, phase_n=1):
        """
        Show observed - estimated graph.
        @phase_n <int>: phase number
        """
        phase_numbers = self.estimator_dict.keys()
        phase = self._num2str(phase_n)
        if phase not in phase_numbers:
            raise KeyError(f"phase_n must be in {list(phase_numbers)[0]} - {list(phase_numbers)[-1]}")
        self.estimator_dict[phase].compare_graph()

    def _num2str(self, num):
        """
        Convert numbers to 1st, 2nd etc.
        @num <int>: number
        @return <str>
        """
        q, mod = divmod(num, 10)
        suffix = "th" if q == 1 else self.SUFFIX_DICT[mod]
        return f"{num}{suffix}"

    def show_parameters(self):
        """
        Show the parameter values.
        @retunr <pd.DataFrame>
        """
        df1 = pd.DataFrame.from_dict(self.phase_dict, orient="index")
        _dict = {
            k: estimator.param_dict
            for (k, estimator) in self.estimator_dict.items()
        }
        df2 = pd.DataFrame.from_dict(_dict, orient="index")
        # Rename R0 to Rt
        df2 = df2.rename({"R0": "Rt"}, axis=1)
        self.param_df = pd.concat([df1, df2], axis=1).fillna("-")
        return self.param_df

    def param(self, phase, param_name):
        """
        Return parameter value.
        """
        try:
            estimator = self.estimator_dict[phase]
        except KeyError:
            raise KeyError("Please revise phase name. e.g. 1st, 2nd, 3rd, 4th")
        try:
            param_name = "R0" if param_name == "Rt" else param_name
            return estimator.param_dict[param_name]
        except KeyError:
            raise KeyError("Please revise parameter name. e.g. rho, gamma, R0 or R0")

    def param_history(self, targets=None):
        """
        Show the ratio to 1st parameters as a figure (bar plot).
        @targets <list[str]>: parameters to show (including Rt etc.)
        """
        targets = self.param_df.columns if targets is None else targets
        df = self.param_df.loc[:, targets]
        df.index = self.param_df[["start_date", "end_date"]].apply(
            lambda x: f"{x[0]}-{x[1].replace('-', 'today')}",
            axis=1
        )
        df = df / df.iloc[0]
        df.plot.bar(title="Ratio to 1st parameters")
        plt.xticks(rotation=0)
        plt.show()

    def compare_estimated_numbers(self):
        """
        Compare the number of confimred cases estimated with the parameters and show graph.
        @variable <str>: variable to compare
        """
        # Observed
        df = pd.DataFrame(self.record_df.set_index("Date")["Confirmed"])
        # Estimated
        for (num, estimator) in self.estimator_dict.items():
            model, info_dict, param_dict = estimator.info()
            day_n = int((datetime.today() - info_dict["start_time"]).total_seconds() / 60 / 60 / 24 + 1)
            predicter = Predicter(**info_dict)
            predicter.add(model, end_day_n=day_n, **param_dict)
            # Calculate the number of confirmed cases
            new_df = predicter.restore_df().drop("Susceptible", axis=1).sum(axis=1)
            new_df = new_df.resample("D").last()
            df = pd.concat([df, new_df], axis=1)
        # Show graph
        df = df.fillna(0).astype(np.int64)
        df.columns = ["Observed"] + [f"{phase}_param" for phase in self.phase_dict.keys()]
        df = df.loc[self.phase_dict["1st"]["start_date"]: self.record_df["Date"].max(), :]
        for col in df.columns[1:]:
            line_plot(
                df.replace(0, np.nan)[["Observed", col]],
                f"Confirmed cases over time: Actual and predicted with {col}",
                y_integer=True
            )

    def add_future_param(self, start_date, **kwargs):
        """
        Add parameters of the future.
        @start_date <str>: the start date of the phase
        @kwargs: keword argument of parameters to change
        """
        last_phase = list(self.phase_dict.items())[-1][0]
        param_dict = self.estimator_dict[last_phase].info()[2]
        param_dict.update(**kwargs)
        new_phase = self._num2str(len(self.phase_dict) + 1)
        self.future_param_dict[new_phase] = param_dict
        return pd.DataFrame.from_dict(self.future_param_dict, orient="index")

    def predict(self, days=1000, min_infected=1):
        """
        Predict the future.
        @days <int or None>: how many days to predict from today
        @min_infected <int>: if Infected < min_infected, the records will not be used
        """
        if not isinstance(days, int):
            raise TypeError("days_to_predict must be integer!")
        # Create parameter dictionary
        predict_param_dict = {
            phase: self.estimator_dict[phase].info()[2]
            for (phase, _) in self.phase_dict.items()
        }
        predict_param_dict.update(self.future_param_dict)
        # Predict
        model, info_dict, _ = self.estimator_dict["1st"].info()
        predicter = Predicter(**info_dict)
        phase_dict = self.phase_dict.copy()
        phase_dict.update(self.future_phase_dict)
        for (phase, date_dict) in phase_dict.items():
            start = pd.to_datetime(date_dict["start_date"])
            end = pd.to_datetime(date_dict["end_date"])
            if end is None:
                day_n = int((datetime.now() - start).total_seconds() / 60 / 60 / 24) + days
            else:
                day_n = int((end - start).total_seconds() / 60 / 60 / 24)
            param_dict = predict_param_dict[phase]
            predicter.add(model, end_day_n=day_n, count_from_last=True, **param_dict)
        # Restore
        df = predicter.restore_df(min_infected=min_infected)
        # Graph: If max(other variables) < min(Susceptible), not show Susceptible
        without_s = df.drop("Susceptible", axis=1).sum(axis=1).max()
        drop_cols = ["Susceptible"] if without_s < df["Susceptible"].min() else None
        predicter.restore_graph(drop_cols=drop_cols, min_infected=min_infected, y_integer=True)
        return df


# In[ ]:


def log_curve(x, k, x_0, ymax):
    return ymax / (1 + np.exp(-k*(x-x_0)))


# In[ ]:


def log_fit(train_df, area, metric):
    area_data = select_area(train_df, areas=[area])
    area_data = area_data.loc[area_data[metric] > 0, :]
    x_data = range(len(area_data.index))
    y_data = area_data[metric]
    if len(y_data) < 5:
        estimated_k = -1  
        estimated_x_0 = -1 
        ymax = -1
    elif max(y_data) == 0:
        estimated_k = -1  
        estimated_x_0 = -1 
        ymax = -1
    else:
        try:
            popt, pcov = curve_fit(
                log_curve, x_data, y_data, bounds=([0,0,0],np.inf),
                p0=[0.3,100,10000], maxfev=1000000
            )
            estimated_k, estimated_x_0, ymax = popt
        except RuntimeError:
            print(area)
            print("Error - curve_fit failed") 
            estimated_k = -1  
            estimated_x_0 = -1 
            ymax = -1
    estimated_parameters = pd.DataFrame(
        np.array([[area, estimated_k, estimated_x_0, ymax]]), columns=['Area', 'k', 'x_0', 'ymax']
    )
    return estimated_parameters


# In[ ]:


def get_parameters(metric):
    parameters = pd.DataFrame(columns=['Area', 'k', 'x_0', 'ymax'], dtype=np.float)
    for area in train_df['Area'].unique():
        estimated_parameters = log_fit(train_df, area, metric)
        parameters = parameters.append(estimated_parameters)
    parameters['k'] = pd.to_numeric(parameters['k'], downcast="float")
    parameters['x_0'] = pd.to_numeric(parameters['x_0'], downcast="float")
    parameters['ymax'] = pd.to_numeric(parameters['ymax'], downcast="float")
    parameters = parameters.replace({'k': {-1: parameters[parameters['ymax']>0].median()[0]}, 
                                     'x_0': {-1: parameters[parameters['ymax']>0].median()[1]}, 
                                     'ymax': {-1: parameters[parameters['ymax']>0].median()[2]}})
    return parameters


# ## Data

# In[ ]:


train_raw = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-3/train.csv")
test_raw = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-3/test.csv")
submission_sample_raw = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-3/submission.csv")
# Population
population_raw = pd.read_csv(
    "/kaggle/input/covid19-global-forecasting-locations-population/locations_population.csv"
)


# In[ ]:


submission_sample_raw.head()


# In[ ]:


df = pd.DataFrame(
    {
        "Nunique_train": train_raw.nunique(),
        "Nunique_test": test_raw.nunique(),
        "Null_Train": train_raw.isnull().sum(),
        "Null_Test": test_raw.isnull().sum(),
    }
)
df.fillna("-").T


# In[ ]:


population_raw.head()


# In[ ]:


df = population_raw.rename({"Province.State": "Province", "Country.Region": "Country"}, axis=1)
df["Country/Province"] = df[["Country", "Province"]].apply(
    lambda x: f"{x[0]}/{x[0]}" if x[1] is np.nan else f"{x[0]}/{x[1]}",
    axis=1
)
df = df.loc[:, ["Country/Province", "Population"]]
# Culculate total value of each country/province
df = df.groupby("Country/Province").sum()
# Global population
df.loc["Global", "Population"] = df["Population"].sum()
# DataFrame to dictionary
population_dict = df.astype(np.int64).to_dict()["Population"]
population_dict


# In[ ]:


df = pd.merge(
    train_raw.rename({"Province_State": "Province", "Country_Region": "Country"}, axis=1),
    population_raw.rename({"Province.State": "Province", "Country.Region": "Country"}, axis=1),
    on=["Country", "Province"],
    how="left"
)
# Area: Country or Country/Province
df["Area"] = df[["Country", "Province"]].apply(
    lambda x: f"{x[0]}" if x[1] is np.nan else f"{x[0]}/{x[1]}",
    axis=1
)
# Date
df["Date"] = pd.to_datetime(df["Date"])
# The number of cases
df = df.rename({"ConfirmedCases": "Confirmed", "Fatalities": "Fatal"}, axis=1)
df[["Confirmed", "Fatal"]] = df[["Confirmed", "Fatal"]].astype(np.int64)
# Show data
df = df.loc[:, ["Date", "Area", "Population", "Confirmed", "Fatal"]]
train_df = df.copy()
train_df.tail()


# In[ ]:


df = pd.merge(
    test_raw.rename({"Province_State": "Province", "Country_Region": "Country"}, axis=1),
    population_raw.rename({"Province.State": "Province", "Country.Region": "Country"}, axis=1),
    on=["Country", "Province"],
    how="left"
)
df["Area"] = df[["Country", "Province"]].apply(
    lambda x: f"{x[0]}" if x[1] is np.nan else f"{x[0]}/{x[1]}",
    axis=1
)
df["Date"] = pd.to_datetime(df["Date"])
df = df.loc[:, ["ForecastId", "Date", "Area", "Population"]]
test_df = df.copy()
test_df.tail()


# ## Prediction of the number of recovered cases with logistic curve

# ### Curve fitting: Confirmed, Fatal

# In[ ]:


get_ipython().run_cell_magic('time', '', 'confirmed_param_df = get_parameters("Confirmed")\nconfirmed_param_df.head()')


# In[ ]:


df = train_df.loc[train_df["Confirmed"] > 0, ["Date", "Area"]].groupby("Area").first()
df = df.rename({"Date": "First_date"}, axis=1).reset_index()
df = pd.merge(confirmed_param_df, df)
confirmed_df = df.copy()
confirmed_df.head()


# In[ ]:


fatal_param_df = get_parameters("Fatal")
fatal_param_df.head()


# In[ ]:


df = train_df.loc[train_df["Fatal"] > 0, ["Date", "Area"]].groupby("Area").first()
df = df.rename({"Date": "First_date"}, axis=1).reset_index()
df = pd.merge(fatal_param_df, df)
fatal_df = df.copy()
fatal_df.head()


# ### Predict the number of recovered cases

# In[ ]:


df = pd.merge(confirmed_df, fatal_df, on="Area", suffixes=["_confirmed", "_fatal"])
# k
df["k"] = df["k_confirmed"]
# x_0
df["First_date_recovered"] = df["First_date_fatal"]
df["x_0"] = df[["x_0_confirmed", "x_0_fatal"]].max(axis=1)
# ymax
df["ymax"] = df["ymax_confirmed"] - df["ymax_fatal"]
# save
df = df.loc[:, ["Area", "First_date_recovered", "k", "x_0", "ymax"]]
recovered_df = df.copy()
recovered_df.head()


# In[ ]:


df = train_df.loc[train_df["Confirmed"] > 0, :]
df = pd.merge(df, recovered_df, on="Area", how="left")
df["date_diff"] = (df["Date"] - df["First_date_recovered"]).dt.total_seconds() / 60 / 60 / 24
df.loc[(df["date_diff"] < 0) | (df["date_diff"].isnull()), "date_diff"] = -1
df["date_diff"] = df["date_diff"].astype(np.int64)
df["Recovered"] = df[["date_diff", "k", "x_0", "ymax"]].apply(
    lambda x: 0 if x[0] < 0 else log_curve(x[0], x[1], x[2], x[3]),
    axis=1
).astype(np.int64)
df = df.rename({"Fatal": "Deaths"}, axis=1)
df["Infected"] = df["Confirmed"] - df["Recovered"] - df["Deaths"]
ncov_df = df.copy()
ncov_df.head()


# ## Prepare for SIR-F model hyperparameter optumization

# In[ ]:


scenario = Scenario(ncov_df, name="Global")


# In[ ]:


scenario.show_record().tail()


# In[ ]:


scenario.growth_factor()


# ## Trend Analysis

# In[ ]:


scenario.trend(variables=["Confirmed"])


# In[ ]:


scenario.trend(variables=["Confirmed"], start_date="13Mar2020")


# In[ ]:


scenario.set_phase(
    start_dates=["13Mar2020"],
    population=population_dict["Global"]
)


# In[ ]:


scenario.estimate(SIRF)


# In[ ]:


scenario.accuracy_graph(phase_n=1)


# In[ ]:


scenario.compare_estimated_numbers()


# In[ ]:


scenario.param_df


# In[ ]:


days_to_predict = int((test_df["Date"].max() - datetime.today()).total_seconds() / 3600 / 24 + 10)
days_to_predict


# In[ ]:


global_predict = scenario.predict(days=days_to_predict)
global_predict.tail(7).style.background_gradient(axis=0)


# ## Area lebel

# In[ ]:


# Current record
df = ncov_df.copy()
df = df.loc[df["Date"] == df["Date"].max(), ["Area", "Confirmed", "Deaths"]]
df["Confirmed"] = df["Confirmed"] / df["Confirmed"].sum()
df["Deaths"] = df["Deaths"] / df["Deaths"].sum()
current_df = df.rename({"Deaths": "Fatal"}, axis=1)
current_df.tail()


# In[ ]:


df = global_predict.copy()
df["Date"] = df.index.date
df["Confirmed"] = df["Infected"] + df["Recovered"] + df["Fatal"]
df = df.groupby("Date").last().reset_index()[["Date", "Confirmed", "Fatal"]]
global_df = df.copy()
global_df.tail()


# In[ ]:


record_df = pd.DataFrame()

for i in range(len(global_df)):
    date, confirmed, fatal = global_df.iloc[i, :].tolist()
    df = current_df.copy()
    df["Date"] = date
    df["Confirmed"] = (confirmed * df["Confirmed"]).astype(np.int64)
    df["Fatal"] = (fatal * df["Fatal"]).astype(np.int64)
    record_df = pd.concat([record_df, df], axis=0)

record_df["Date"] = pd.to_datetime(record_df["Date"])
record_df = record_df.loc[:, ["Date", "Area", "Confirmed", "Fatal"]].reset_index(drop=True)
record_df


# ## Submission data

# In[ ]:


submission_sample_raw.shape


# In[ ]:


df = pd.merge(record_df, test_df, on=["Date", "Area"], how="right")
df = df.sort_values("ForecastId").reset_index()
df = df.loc[:, ["ForecastId", "Confirmed", "Fatal"]]
df = df.rename({"Confirmed": "ConfirmedCases", "Fatal": "Fatalities"}, axis=1)
submission_df = df.copy()
submission_df


# In[ ]:


submission_df.shape


# In[ ]:


len(submission_df) == len(submission_sample_raw)


# # Calculate the mean value of 1. and 2.

# In[ ]:


submission


# In[ ]:


submission_df


# In[ ]:


merged = pd.merge(submission, submission_df, on="ForecastId")
merged["ConfirmedCases"] = (merged["ConfirmedCases_x"] + merged["ConfirmedCases_y"]) / 2
merged["Fatalities"] = (merged["Fatalities_x"] +  merged["Fatalities_y"]) / 2
merged = merged.loc[:, ["ForecastId", "ConfirmedCases", "Fatalities"]].astype(np.int64)
merged


# In[ ]:


merged.to_csv("/kaggle/working/submission.csv", index=False)

