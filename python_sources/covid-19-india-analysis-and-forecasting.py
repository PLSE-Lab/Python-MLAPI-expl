#!/usr/bin/env python
# coding: utf-8

# # Covid 19 : India Analysis and Forecasting

# ## Note: 
# <p> The author doesn't claim to have any domain knowledge and isn't an epidemiologist. The analysis in this notebook is just an attempt to learn new techniques and gather potential insights from the available data but shouldn't be taken seriously. </p>

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.dates import AutoDateFormatter, AutoDateLocator
from matplotlib import dates as dt
import datetime

plt.style.use('seaborn')


# In[ ]:


path_train = 'covid19-global-forecasting-week-4/train.csv'
path_test = 'covid19-global-forecasting-week-4/test.csv'
path_sbumit = 'covid19-global-forecasting-week-2/submission.csv'
path_india = '../input/COVID_India_Updated_from_API.csv'
path_test = '../input/COVID_India_Updated_Test_data.csv'

train_kaggle = '../input/train.csv'
test_kaggle = '../input/test.csv'
#submit_kaggle = '/kaggle/input/covid19-global-forecasting-week-2/submission.csv'

path_graphs = ''


# In[ ]:


df_train_count = pd.read_csv(train_kaggle)
df_test_count = pd.read_csv(test_kaggle)
df_train_count.rename(columns = {'Country_Region': 'Country/Region', 'Province_State':'Province/State'}, inplace = True)
df_test_count.rename(columns = {'Country_Region': 'Country/Region', 'Province_State':'Province/State'}, inplace = True)


# In[ ]:


def country_df(country, df):
    """Filters a Dataframe according to Country.
    Args: 
    country: String. Name of country for which dataframe is to be filtered for.
    df: Dataframe. The Dataframe that is to be filtered.
    Returns: 
    df_cases: Dataframe. Filtered dataframe containing fields with confirmed Covid cases for the country.
    df_fatal: Dataframe. Filtered dataframe containing fileds with Covid fatalities for the country.
    """
    if country != 'World':
        country_filt = (df['Country/Region'] == country)
        df_cases = df.loc[country_filt].groupby(['Date'])['ConfirmedCases'].sum()
        df_fatal = df.loc[country_filt].groupby(['Date'])['Fatalities'].sum()
    else:
        df_cases = df.groupby(['Date'])['ConfirmedCases'].sum()
        df_fatal = df.groupby(['Date'])['Fatalities'].sum()
    
    return df_cases, df_fatal


# In[ ]:


def prepare_train_data(df_cases, df_fatal):
    """Splits ConfirmedCases and Fatalities dataframe into training features and labels.
    Args:
    df_cases: Dataframe. Filtered dataframe containing fields with confirmed Covid cases for the country.
    df_fatal: Dataframe. Filtered dataframe containing fileds with Covid fatalities for the country.
    Returns:
    x_train: np array. Array of integers denoting days since firstday of original df for model training features.
    y_cases: List. List of Confirmed Cases as labels for model from first day of original df.
    y_fatal: List. List of Fatalities as labels for model from first day of original df.
    """
    x_train = np.arange(0, len(df_cases)).reshape(-1,1)
    y_cases = df_cases.to_list()
    y_fatal = df_fatal.to_list()
   
    
    return x_train, y_cases, y_fatal


# In[ ]:


def plot_actual_predicted(country,label,  y_cases, y_pred, show_lockdown = False):
    """Plots the Actual and Predicted ConfirmedCases/Fatalities for a country.
    Args:
    country(string) country name.
    y_cases, y_pred (array/list) - Actual And Predicted Metrics.
    label(string) - Which metric is being passed.
    Returns:
    """
        
    xtick_locator = AutoDateLocator()
    xtick_formatter = AutoDateFormatter(xtick_locator)
    
    # generating dates using pandas, can be used with date_plot.
    train_times = pd.date_range(start = '2020-01-22', periods=len(y_cases))
    test_times = pd.date_range(start = '2020-01-22', periods=len(y_pred))
    dates = train_times.to_series().dt.date
    dates_test = test_times.to_series().dt.date

    # converting to Series, list 
    d_train = dates.tolist()
    d_test  = dates_test.tolist()
    
    # getting date_format variable for matplotlib and lockdown date.
    date_format = dt.DateFormatter('%b, %d')
    
    # converting lockdown date string to datetime.
    lockdown_date_str = '2020-03-25'
    lockdown_date = datetime.datetime.strptime(lockdown_date_str, '%Y-%m-%d')
    
    plt.style.use('seaborn')
    fig, ax = plt.subplots()
    fig.set_figheight(8)
    fig.set_figwidth(17)
    fig.autofmt_xdate()
    ax.plot_date(d_train , y_cases, label = 'Actual Values', linestyle = 'solid', marker = None)
    ax.plot_date(d_test , y_pred , label = 'Predicted Values', linestyle = 'solid', marker = None)
    ax.set_title(f'{country} : {label} - Polynomial Regression (Degree = 6)')
    ax.set_ylabel(f'No of {label}')
    if show_lockdown:
        ax.axvline(lockdown_date, color = 'r', label='Lockdown')
    ax.xaxis.set_major_formatter(date_format)
    ax.legend()
    fig.savefig( path_graphs + f'{country}_conf_case.jpg')


# In[ ]:


#from india_API_data.Covid19_india_org_api import make_dataframe
#from india_API_data.Covid19_india_org_api import get_test_dataframe


# In[ ]:


def plot_daily(df, columns, save = False, log = False):
    """ Helper Function to Plot Current Metrics from API data.
    Args: 
    df (Dataframe) - Dataframe of API data create Using Make_dataframe.
    columns (List of column names)
    save(Bool) - Whether to save fig.
    Returns:
    """
    fig, ax = plt.subplots()
    fig.set_figheight(10)
    fig.set_figwidth(20)
    plt.style.use('seaborn')
    plt.title('Daily Trends: India (From API)')
    plt.ylabel('No. Of Individuals')
    plt.xlabel('Date')
    fig.autofmt_xdate()
    for item in columns:
        plt.plot_date(x = df.index, y= df[item], label = item, linestyle = 'solid', marker = None)
    plt.legend()
    if save!= False:
        fig.savefig(path_graphs + save + 'India_Cumulative_stats.png')
    #plt.yscale('log')
    plt.show()


# # Visualizations & Analysis
# 

# ## Logistic Curve or S- shaped Curve : Comparing Epidemic Behaviour
# <p>From the 3blue1brown video, epidemics are observed to follow a logistic curve. The no. of infected rises exponentially and reaches an inflection point before gradually decreasing. This can be somewhat seen in China's Confirmed Cases plot.</p>
# ![Image of Sigmoid](https://wikimedia.org/api/rest_v1/media/math/render/svg/9e26947596d387d045be3baeb72c11270a065665)
# Here x0 = the x value of the sigmoids midpoint. <br>
# L = the curve's maximum value. <br>
# k = the logistic growth rate or steepness of the curve.

# In[ ]:


# Plotting a simple logistic curve using numpy and matplotlib.
# x = (-6,6), L =1, k = 1, x0 =0
x = np.arange(-6,7)
power = -1*x
y = 1 / (1 + np.exp(power))

plt.figure(figsize=(10,10))
plt.title('Simple Logistic Curve')
plt.grid(True)
plt.plot(x, y)
plt.show()


# ## This behaviour can be somewhat seen in the case of China

# In[ ]:


china_cases, china_fatal = country_df('China', df_train_count)
china_cases.plot(figsize = (10,10), title = 'China Confirmed Cases', grid = True)
plt.show()


# In[ ]:


china_cases[:44].plot(figsize = (10,10), title = 'China Confirmed Cases (Log Scale)', grid = True, logy=True)
plt.show()


# ## Worlwide Cases

# In[ ]:


world_cases, world_fatal = country_df('World', df_train_count)
world_cases.plot(figsize = (10,10), title = 'Worldwide Confirmed Cases', grid = True)
plt.show()


# In[ ]:


world_cases.plot(figsize = (10,10), title = 'Worldwide Confirmed Cases (Log)', grid = True, logy = True)
#ax = plt.gca()
#plt.plot([ax.get_xlim()[0],ax.get_xlim()[1]],[ax.get_ylim()[0],ax.get_ylim()[1]])
plt.show()


# 1. The Number of People getting infected is still growing, but tobetter visualise whether it is growing exponentially we can use the log scale.
# 2. The data plotted on the log Scale shows that the growth seems to have slowed down compared to the orginal straight line representing exponential growth before a smaller second wave. This could be a temporary Dip Before another wave but is an encouraging sign that shows that the expoenetial growth of the virus might finally be slowing down.This also does not guarantee that the inflection point of the growth of the virus has been reached.

# ## India - Current Statistics
# Looking at the trend on the logarithmic scale, the exponential growth of the virus seems to be slowing down, this is an encouraging trend. We also need to look at the amount of testing done to get a better sense of whether this is representative or just duw to a lack of testing, data.

# In[ ]:


#india_data = make_dataframe()
india_data = pd.read_csv(path_india, index_col = 0)
plot_daily(india_data.iloc[40:], india_data.columns.tolist(), save='All_stats_')


# In[ ]:


india_data.plot(y = 'DailyConfirmed', title  = 'India Cases (Log Scale)', logy = True, figsize = (18,9))
plt.show()


# ### India Testing Statistics
# The comparison of the graphs of the amount of Testing Done and the confirmed Cases each day can be a good indicator of whether enough testing is being done. If the curves are very similar it can indicate that not enough people are getting tested.<br>
# Looking at the comparison and observing the sharp increase in samples collected, we can say that testing is finally being done on a larger scale. (Testing Data from 03-13. Sharper increase in April).

# In[ ]:


#india_test_df = get_test_dataframe()
india_test_df = pd.read_csv(path_test, index_col = 0)


# In[ ]:


india_test_df.plot(title = 'No. of Testing Samples Collected per day by ICMR', figsize = (16,8))
plt.show()


# In[ ]:


# Combining with Statistics data
india_combined_data = india_test_df.join(india_data, how = 'right')


# In[ ]:


plot_daily(india_combined_data, ['TotalConfirmed', 'Testing Samples'], save='Cofirmed_testing_samples_')


# ### Growth Factor Of Virus
# <p>
#     The growth factor on day N is the number of confirmed cases on day N minus confirmed cases on day N-1 divided by the number of confirmed cases on day N-1 minus confirmed cases on day N-2.
#      <br>Measure of whether the disease is growing or not. </p>
# 1. A value of greater than 1 = growth. <br>
# 2. Less than 1 = decline. <br>
# 3. A growth factor of 1 is the inflection point and at this point the disease is not increasing.
# 
# <p>
# It can be seen that the growth of the virus <b>hasn't stabilised at the inflection point</b>.<br>
# The Linear Regression on the Growth Factor seems to point to a downward trend in the Growth Factor but the margin of error is high.

# In[ ]:


def growth_factor(confirmed):
    confirmed_nminus1 = confirmed.shift(1, axis = 0)
    confirmed_nminus2 = confirmed.shift(2, axis = 0)
    return ((confirmed - confirmed_nminus1)/(confirmed_nminus1 - confirmed_nminus2))


# In[ ]:


india_growth_factor = growth_factor(india_data.TotalConfirmed[41:])
print(f'Mean Growth Factor : {india_growth_factor.mean()}')


# In[ ]:


india_growth_factor.plot(grid = True, title = 'India Growth Factor Since 2020-03-11 (Widespread Testing) ', figsize = (18,6), ylim = (0,5))

plt.axhline(india_growth_factor.mean(), color = 'r', label='Mean Growth Factor')
plt.legend()
plt.savefig(path_graphs + 'India_Growth_Factor')
plt.show()


# #### Linear Regression for Growth Factor.
# This points to downwards trend, however due to not enough early testing there are outliers with very high growth rate. The confidence interval shows that this isn't really a meaningful result. 

# In[ ]:


plt.figure(figsize = (18,6))
ax = sns.regplot(x = np.arange(len(india_growth_factor.index.tolist())) , y =india_growth_factor.to_list())
ax.set_title('Linear reegression on Growth Factor')
ax.set_xlabel(' Days Since 2020-03-11')
plt.savefig(path_graphs +'India_Growth_Factor_Pred')
plt.show()
#ax.set_ybound(upper = 10)
# there is an outlier, causing problems (can just remove.)
# This is just a visualisation library, won't give you the equation, use plain linear reg to get line and plot it.


# ### Growth Ratio of Virus
# The growth ratio on day N is the number of confirmed cases on day N divided by the number of confirmed cases on day N-1.</p>
# Percentage Increase in total cumulative Cases from one day to the next.

# In[ ]:


growth_ratio_india = india_data.TotalConfirmed[41:]/india_data.TotalConfirmed[41:].shift(1)
print(f' Mean growth Ratio of Cases In India : {growth_ratio_india.mean()}')
growth_ratio_india.plot(grid = True, title = 'India Growth Ratio', figsize = (18,6))
plt.axhline(growth_ratio_india.mean(), color = 'r', label='Mean Growth Ratio')
plt.legend()
plt.savefig(path_graphs + 'India_Growth_Ratio')
plt.show()


# # 3. Modeling - Polynomial Regression
# <p>This won't model the eventual inflection and decline of the growth. Due to a scarcity of features, the results should be taken with a grain of salt and might just be predicting based on the days since the outbreak feature.</p> <p>The results have been left in as it may be possible to get an idea of the initial growth using this approach. 
# Found the best validation set score using gridsearchCV and a polynomial of degree 4.
# Regularized methods like ridge regression don't seem to predict further than the last known confirmedcases value.</p>

# In[ ]:


from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split


# ####  Using Growth Factor as a feature

# In[ ]:


# Imputing Missing values with mean of growth factor from 11-03 to present day.
# df_test_count.rename(columns = {'Country_Region': 'Country/Region', 'Province_State':'Province/State'}, inplace = True)

india_growth_factor_1 = growth_factor(india_data.TotalConfirmed[:])
india_growth_factor_1.replace(np.nan, india_growth_factor.mean(), inplace=True)
india_growth_factor_1.replace(np.inf, india_growth_factor.mean(), inplace = True)
india_growth_factor_1 = india_growth_factor_1.to_frame()


# In[ ]:


india_growth_factor_1.rename(columns = {'TotalConfirmed' : 'GrowthFactor'} , inplace=True)


# In[ ]:


# Merging the two dataframes
india_data = pd.concat([india_data, india_growth_factor_1], axis = 1)


# ####  Using Growth Ratio as a feature

# In[ ]:


growth_ratio_india_1 = india_data.TotalConfirmed[:]/india_data.TotalConfirmed[:].shift(1)


# In[ ]:


growth_ratio_india_1.replace(np.nan, 1, inplace = True)


# In[ ]:


growth_ratio_india_1 = growth_ratio_india_1.to_frame()
growth_ratio_india_1.rename(columns = {'TotalConfirmed' : 'GrowthRatio'} , inplace=True)


# In[ ]:


# Merging the two dataframes
india_data = pd.concat([india_data, growth_ratio_india_1], axis = 1)


# In[ ]:


# Days since First case/data Availibity.
days_outb = np.arange(len(india_data))

india_data['Days'] = days_outb


# In[ ]:


x = india_data[['Days','GrowthFactor', 'GrowthRatio']]
y = india_data['TotalConfirmed']


# In[ ]:


# Not Shuffling time-Series Data.
x_train, x_test, y_train, y_test = train_test_split(x, y , test_size = 0.1, shuffle = False)


# In[ ]:


Input = [('poly', PolynomialFeatures(degree= 4)), ('lin_reg', LinearRegression())]

pipe = Pipeline(Input)


# In[ ]:


pipe.fit(x_train, y_train)


# In[ ]:


pipe.score(x_test, y_test)


# In[ ]:


predictions = pipe.predict(x)


# In[ ]:


plot_actual_predicted('India', 'Confirmed Cases', y, predictions, show_lockdown= True)

