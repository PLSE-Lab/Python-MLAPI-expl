#!/usr/bin/env python
# coding: utf-8

# 
#     
#     
# # <center>Using Testing Data to Estimate The True Number of Infected People</center>
# ### <center>Tarek Ayed</center>
# 
# ![kaggleviz.png](attachment:kaggleviz.png)
# 
# 
# # Abstract
# 
# This notebook examines the possibility of **extrapolating the proportion of infected people in a given country or area**, by using testing data and fitting a model that predicts the probability of testing positive, given how much a country tests. This is achieved by fitting a model that predicts the ratio $\frac{confirmed\ cases}{samples\ tested}$ as a function of $\frac{deaths\ at\ d+4}{samples\ tested}$, in the form of $f(x)=m\times x^\alpha$. Pearson **correlation coefficients ($R^2$) range from $0.92$ to $0.99$** depending on how much data is used to fit the model. This model estimates that, before April 20, **$14.48\%$ of NYC** had been infected, **$5.70\%$ of France** and **$7.78\%$ of Italy**. The model's predictions are also **close underestimations of all available serological survey results**. The estimated death rates given by the model are within the range of expert estimates.
# 

# # Introduction
# 
# The starting point is the publication by *Our World in Data* of **testing data** for several countries, and the fact that the proportion of tests that are positive decreases when the number of tests increases. The fact is that **even countries which test the most don't know what the true number of infected people is**. This is why I chose to examine the possibility of extrapolating the evolution of this proportion and extract its value *if the whole population was tested*.
# 
# In order to know if a country is *testing as much as another*, the absolute number of tests is not sufficient, as it needs to be treated relatively to Covid-19's penetration in the country. To assess the prevalence of the virus, I used the number of deaths at day $d+N$ as a proxy. The first part of this notebook is dedicated to find the best $N$ to choose by observing the time series of daily confirmed cases and daily deaths in different countries, and measuring the delay between peaks in countries like Italy or China.

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

os.chdir('/kaggle/input/covid19')

conf_raw = pd.read_csv('confirmed_cases.csv') # 'total-confirmed-cases-of-covid-19-per-million-people.csv' from Our World in Data
tests_raw = pd.read_csv('tests.csv') # 'full-list-cumulative-total-tests-per-thousand.csv' from Our World in Data
deaths_raw = pd.read_csv('deaths.csv') # 'total-daily-covid-deaths-per-million.csv' from Our World in Data
daily_conf_raw = pd.read_csv('daily_confirmed_cases.csv') # 'total-and-daily-cases-covid-19.csv' from Our World in Data

tests_raw['Entity'] = tests_raw['Entity'].apply(lambda s: s.split('-')[0][:-1])


# # Finding best number N of days for offset

# Comparing the evolution of daily confirmed cases and daily confirmed deaths, using a moving average of $6$ days, for different countries:

# In[ ]:


correl = deaths_raw.merge(daily_conf_raw, on=['Entity','Date']).rename(
    columns={'Daily confirmed deaths per million (deaths per million)':'Daily deaths/million',
             'Daily new confirmed cases (cases)':'Daily cases'}
)[['Entity','Date','Daily deaths/million','Daily cases']]
correl['Date'] = pd.to_datetime(correl['Date'])
correl = correl[correl['Daily cases']>50] # Reducing noisy data
countries_correl = correl['Entity'].value_counts()
countries_correl = countries_correl[countries_correl > 10].index.to_list()
correl = correl[correl['Entity'].apply(lambda i: i in countries_correl)]


# In[ ]:


def plot_correl(country):    
    df = correl[correl['Entity']==country]
    df = df.sort_values('Date', ascending=True)
    plt.figure(figsize=(20,7))
    plt.plot(df['Date'], df['Daily deaths/million'].rolling(window=6).mean()/df['Daily deaths/million'].max(), label='Daily deaths (scaled)')
    plt.plot(df['Date'], df['Daily cases'].rolling(window=6).mean()/df['Daily cases'].max(), label='Daily cases (scaled)')
    plt.xticks(ticks=df['Date'],rotation=45)
    plt.title(country)
    plt.legend()
    plt.show()
plot_correl('France')
plot_correl('Italy')
plot_correl('Spain')
plot_correl('South Korea')
plot_correl('United Kingdom')


# This heuristic justifies the use of a value in the range $4-6$. I will keep the value of $4$, as changing it to $5$ or $6$ does not fundamentally change the results, and a lower $N$ allows to have more data.

# # Building a model

# In[ ]:


from pandas.tseries.offsets import DateOffset
N = 4

offset_deaths = deaths_raw[['Entity','Date','Total confirmed deaths per million (deaths per million)']]
offset_deaths['Date'] = pd.to_datetime(offset_deaths['Date'])
offset_deaths['Date'] = offset_deaths['Date'].apply(lambda t : t-DateOffset(days=N))
conf_raw['Entity-1'] = conf_raw['Entity'].apply(lambda s: s[:-1])
tests_raw['Entity-1'] = tests_raw['Entity']
full = tests_raw.merge(conf_raw, on=['Entity-1','Date']).rename(columns={'Entity_y':'Entity'})
full['Date'] = pd.to_datetime(full['Date'])
full = full.merge(offset_deaths, on=['Entity','Date'])

data = full[['Entity', 
             'Date', 
             'Total tests per thousand',
             'Total confirmed cases of COVID-19 per million people (cases per million)',
             'Total confirmed deaths per million (deaths per million)']]
data = data.rename(columns={'Total tests per thousand':'Tests/thousand',
                            'Total confirmed cases of COVID-19 per million people (cases per million)':'Cases/million',
                            'Total confirmed deaths per million (deaths per million)':f'Deaths/million d+{N}',
                            })
data['Date'] = pd.to_datetime(data['Date'])
eps = 0.0
data = data[data[f'Deaths/million d+{N}'] > eps]


# Some preprocessing and corrections are done to the dataset, to improve coherence and homogeneity.

# In[ ]:


# Will only consider data from after this date, in order to have a more homogenous dataset
start_date = pd.to_datetime('2020-03-15')
data = data[data['Date']>start_date]

# Some countries are excluded from the dataset because of reliability issues
countries_to_exclude = ['Malaysia', 'Philippines', 'Australia', 'Bahrain', 'Indonesia', 'India',
                        'Pakistan', 'Costa Rica', 'Ecuador', 'Uruguay', 'Thailand', 'Lithuania', 
                        'Tunisia', 'Senegal', 'Turkey', 'Serbia', 'Panama', 'Peru', 'Paraguay',
                        'Mexico', 'Bangladesh', 'Bolivia', 'Chile', 'Ethiopia', 'Argentina',
                        'Ghana', 'Colombia', 'El Salvador', 'Hungary']
data = data[data['Entity'].apply(lambda i: i not in countries_to_exclude)]

# Computing the relevant ratios
data['Confirmed/test'] = data['Cases/million']/(1000*data['Tests/thousand']) # Converted to Tests/million
data[f'Death d+{N}/test'] = data[f'Deaths/million d+{N}']/(1000*data['Tests/thousand'])

# Final list of countries to be considered
countries = data['Entity'].value_counts().sort_values(ascending=False).index.to_list()
print('List of countries to be considered (sorted by descending number of data points):\n', countries)
print(f'\nNumber of countries: {len(countries)}')
print(f'Number of data points: {len(data)}')


# In[ ]:


# Some corrections need to be made to further homogenize our data
correction = dict(zip(countries, [1.0]*len(countries))) # The value in this dict will be applied 
                                                        # as a multiplier of the value of
                                                        # Deaths d+N / tests for each country

# Correcting for differences in mortality rates due to age distribution and possibly genetic and cutural factors
# Source: https://twitter.com/TrevorSutcliffe/status/1246944321107976192
correction['Italy'] = 0.5
correction['Austria'] = 1.5
correction['Germany'] = 1.5

# Converting number of people tested into number of samples, assuming 2 samples/person on average
# (Some countries report the number of people tested, others report the number of samples)
correction['South Korea'] = 0.5
correction['United Kingdom'] = 0.5
correction['Norway'] = 0.5
correction['Netherlands'] = 0.5
correction['Sweden'] = 0.5

# Correcting for potential relative undercount in deaths (these are assumptions)
correction['Belgium'] *= 0.5
correction['France'] *= 0.8
correction['Italy'] *= 1.2


for i in data.index:
    line = data.loc[i]
    data.loc[i,f'Death d+{N}/test'] = line[f'Death d+{N}/test'] * correction[line['Entity']]

# Removing outliers from dataset    
from scipy import stats
data = data[(np.abs(stats.zscore(data[['Confirmed/test',f'Death d+{N}/test']])) < 3).all(axis=1)]


# In[ ]:


import matplotlib.pyplot as plt
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.tab20.colors)
def scatter_countries(countries,m=None,alpha=1,scale='linear', n_countries=19):
    """
    This function plots a scatter plot using Matplotlib.
        * countries: list of Strings corresponding to values of 'Entity' in data
        * m: first parameter of the regression (optional)
        * alpha: second paramter of the regression (optional)
        * scale: String (optional) is the scale used for x values
    """
    plt.figure(figsize=(10,10))
    for country in countries[:n_countries]:
        df = data[data['Entity']==country]
        plt.scatter(df[f'Death d+{N}/test'], df['Confirmed/test'],marker='+', label=country)
    df = data[data['Entity'].apply(lambda i: i in countries[n_countries:])]
    plt.scatter(df[f'Death d+{N}/test'], df['Confirmed/test'],marker='+', label='Other')
    if m is not None:
        eps = 0.00001
        x = np.linspace(eps, data[f'Death d+{N}/test'].max(), 1000)
        plt.plot(x,m*np.exp(alpha*np.log(x)))
    plt.xscale(scale)
    plt.xlabel(f'Deaths $d+{N}$ per test')
    plt.ylabel('Positives per test')
    plt.legend()
    plt.show()

countries_plot = countries[:19]
scatter_countries(countries, n_countries=19) # Limiting number of countries for readability


# The profile of this scatter plot is concave. Furthermore, for coherence purposes, the model needs to be equal to $0$ on $0$. As a first model, I fitted a linear regression between $y$ and $\sqrt{x}$. 

# In[ ]:


import statsmodels.formula.api as sfa
from scipy.stats import pearsonr

# Linear regression without intercept between y and sqrt(x)
x = data[f'Death d+{N}/test'].values
y = data['Confirmed/test'].values
df = pd.DataFrame({'x': np.sqrt(x), 'y': y})
r = sfa.ols('y ~ x + 0', data=df).fit()

fig, ax = plt.subplots(figsize=(10, 10))
plt.xlabel("$\sqrt{x}$")
plt.ylabel("$y$")
ax.scatter(x=np.sqrt(x), y=y)
ax.plot(np.sqrt(x), r.fittedvalues)

corr, _ = pearsonr(np.sqrt(x), y)
print(f'Pearson Correlation - R^2: {corr:.4f}')


# #### As you can see, this model fits relatively well, with a Pearson coefficient of $0.92$.
# 
# In order to generalize this heuristic and optimize a bit further, a model in the form of $$y=m\times x^\alpha$$ is adopted, with $\alpha \in ]0,1]$.

# In[ ]:


from scipy.optimize import curve_fit

def f(x,m,alpha):
    return(m*np.power(x,alpha))

popt, pcov = curve_fit(f, x, y,[1.5,0.5], bounds=(0.2,2))
m, alpha = popt
print(f"Optimal parameters found:\n\tm = {m:.4f}\n\talpha = {alpha:.4f}")
corr, _ = pearsonr(np.power(x,alpha), y)
print(f'Pearson Correlation - R^2: {corr:.4f}')

scatter_countries(countries,*popt, n_countries=19)


# #### The $R^2$ coefficient here is almost the same as previously but the value of $\alpha$ is different from $\frac{1}{2}$.
# 
# In order to use this model to predict the proportion of infected people, I use the predicted value while inputting the value of the $x$ ratio when the entire population is tested. This means replacing $n_{tests}$ by $n_{pop}$.
# 
# One needs to assume something about the undercount in the number of deaths, as that has a big impact on the result. Also, the prediction gives the proportion of *positives if the entire country had been tested exactly once*. To get the proportion of *infected*, we need an assumption on the efficiency of Covid-19 tests, precisely, the probability of being tested positive if infected (*test sensitivity*). One study suggests that this probability could be around $0.5$ for both mild and severe cases (Source: https://www.medrxiv.org/content/10.1101/2020.02.11.20021493v2).

# # Main Results

# In[ ]:


def pred(d,pop,popt,test_sensitivity=0.50):
    """
    This function computes a prediction of the proportion of infected people in country with:
        * d: number of deaths in the country
        * pop: population of the country
        * popt: parameters to be used in the model
        * test_precision: assumption on the probability of being tested positive, if infected.
    """
    frac = d/pop
    return(f(frac,*popt)/test_sensitivity)

# April 24 data --> predictions about April 20
off_deaths_Italy = 25549 * 1.2
pop_Italy = 60480000
print(f"In Italy: {100*pred(off_deaths_Italy,pop_Italy,popt):.2f}% infected")
off_deaths_France = 21889 * 0.8
pop_France = 66990000
print(f"In France: {100*pred(off_deaths_France,pop_France,popt):.2f}% infected")
off_deaths_NYC = 16388
pop_NYC = 8623000
print(f"In NYC: {100*pred(off_deaths_NYC,pop_NYC,popt):.2f}% infected")

print(f"\nEstimated current IFR in France: {100*(off_deaths_France/(pred(off_deaths_France,pop_France,popt)*pop_France)):.3f}%")
print(f"Estimated current IFR in NYC: {100*(off_deaths_NYC/(pred(off_deaths_NYC,pop_NYC,popt)*pop_NYC)):.3f}%")
print(f"Estimated current IFR in Italy: {100*(off_deaths_Italy/(pred(off_deaths_Italy,pop_Italy,popt)*pop_Italy)):.3f}%")


# Although it is technically possible to compute estimations for the entire world, it is unclear how reliable such estimations are and how they should be interpreted.

# In[ ]:


off_deaths_world = 191962
pop_world = 7794799000
print(f"World: \n\n{100*pred(off_deaths_world,pop_world,popt):.2f}% infected")
print(f"{100*(off_deaths_world/(pred(off_deaths_world,pop_world,popt)*pop_world)):.3f}% estimated current IFR")


# ## Comparison/Validation
# 
# As of today, several **serological surveys** have been conducted, not always following the same sampling methodology nor using the same tests. I will be reporting these results in this section for validation purposes and comparing them with my model's estimations.
# 
# **Reported results are:**
# 
# Santa Clara - end of March: **1.80 to 5.70 %** according to https://www.medrxiv.org/content/10.1101/2020.04.14.20062463v1.full.pdf (preprint)
# 
# Netherlands - end of March: **3%** according to https://www.reddit.com/r/COVID19/comments/g2ec30/3_of_dutch_blood_donors_have_covid19_antibodies/ 
# 
# Sweden (Stockholm area) - April 14: **11%** according to https://www.reddit.com/r/COVID19/comments/g4znbg/at_least_11_of_tested_blood_donors_in_stockholm/
# 
# Los Angeles county - early April: **2.8% to 5.6%** accordint to http://publichealth.lacounty.gov/phcommon/public/media/mediapubhpdetail.cfm?prid=2328
# 
# Geneva, Switzerland - April 17: **3,3% to 7,7%** according to https://www.hug-ge.ch/medias/communique-presse/seroprevalence-covid-19-premiere-estimation
# 
# Geneva, Switzerland - April 10: **1,6% to 5,4%%** according to https://www.hug-ge.ch/medias/communique-presse/seroprevalence-covid-19-premiere-estimation
# 
# New York state - April 19-24: **13.9%** according to https://eu.usatoday.com/story/news/nation/2020/04/23/coronavirus-new-york-millions-residents-may-have-been-infected-antibody-test/3012920001/
# 
# New York City - April 19-24: **21.2%** according to https://eu.usatoday.com/story/news/nation/2020/04/23/coronavirus-new-york-millions-residents-may-have-been-infected-antibody-test/3012920001/

# In[ ]:


print('At the end of March:')

off_deaths_Netherlands = 540 # At the end of March
pop_Netherlands = 17280000
print(f"\nIn the Netherlands: {100*pred(off_deaths_Netherlands,pop_Netherlands,popt):.2f}% infected")

off_deaths_SantaClara = 25 # At the end of March
pop_SantaClara = 1928000
print(f"In Santa Clara county: {100*pred(off_deaths_SantaClara,pop_SantaClara,popt):.2f}% infected")

off_deaths_LA = 617 * 7994/13816 # early April data
pop_LA = 10040000
print(f"In Los Angeles county: {100*pred(off_deaths_LA,pop_LA,popt):.2f}% infected")

print("\nAs of April 14:")
off_deaths_Stockholm = 1400 * 944/1580
pop_Stockholm = 2377081
print(f"\nIn Stockholm county: {100*pred(off_deaths_Stockholm,pop_Stockholm,popt):.2f}% infected")

print("\nIn Geneva county:")
off_deaths_Geneva_10 = 858 * 4438/27856
off_deaths_Geneva_17 = 1141 * 4438/27856
pop_geneva = 499480
print(f"\nBy April 10: {100*pred(off_deaths_Geneva_10,pop_geneva,popt):.2f}% infected")
print(f"By April 17: {100*pred(off_deaths_Geneva_17,pop_geneva,popt):.2f}% infected")

print("\nAs of April 20:")
off_deaths_NYS = 20982 # April 24 data
off_deaths_NYC = 16388
pop_NYS = 19450000
pop_NYC = 8623000
print(f"\nIn New York State: {100*pred(off_deaths_NYS,pop_NYS,popt):.2f}% infected")
print(f"In New York City: {100*pred(off_deaths_NYC,pop_NYC,popt):.2f}% infected")


# ### In addition to being close to these sero-survey results, the model's predictions are all underestimations, except for Geneva which is expected due to Switzerland's aggressive testing policy that makes it an outlier country in our dataset.

# # On single countries

# The idea here is to reduce our dataset to one single country and compute predictions for that country alone. 

# In[ ]:


belgium = data[data['Entity']=='Belgium']
x = belgium[f'Death d+{N}/test'].values
y = belgium['Confirmed/test'].values

def f(x,m,alpha):
    return(m*np.power(x,alpha))

popt_belgium, pcov_belgium = curve_fit(f, x, y,[1.5,0.5], bounds=(0.2,2))
scatter_countries(["Belgium"],*popt_belgium)
m, alpha = popt_belgium
corr, _ = pearsonr(np.power(x,alpha), y)
print(f'Pearson Correlation - R^2: {corr:.4f}')

off_deaths_Belgium = 5453 * 0.5
pop_Belgium = 11400000
print(f"\nIn Belgium: {100*pred(off_deaths_Belgium,pop_Belgium,popt_belgium):.2f}% infected")
print(f"Estimated death rate in Belgium: {100*(off_deaths_Belgium/(pred(off_deaths_Belgium,pop_Belgium,popt_belgium)*pop_Belgium)):.3f}%")


# The following cell can be used to visualize results for any combination of countries.

# In[ ]:


countries_to_include = ['Netherlands']
cp_data = data[data['Entity'].apply(lambda i: i in countries_to_include)]
x = cp_data[f'Death d+{N}/test'].values
y = cp_data['Confirmed/test'].values

def f(x,m,alpha):
    return(m*np.power(x,alpha))

popt_cp_data, pcov_cp_data = curve_fit(f, x, y,[1.5,0.5], bounds=(0.2,2))
scatter_countries(countries_to_include,*popt_cp_data)
m, alpha = popt_cp_data
corr, _ = pearsonr(np.power(x,alpha), y)
print(f'Pearson Correlation - R^2: {corr:.4f}')
off_deaths_Netherlands = 3601
pop_Netherlands = 17280000
print(f"\nIn the Netherlands, April 14: {100*pred(off_deaths_Netherlands,pop_Netherlands,popt_cp_data):.2f}% infected")
print(f"Estimated death rate in the Netherlands: {100*(off_deaths_Netherlands/(pred(off_deaths_Netherlands,pop_Netherlands,popt_cp_data)*pop_Netherlands)):.3f}%")
off_deaths_Netherlands = 540 # At the end of March
pop_Netherlands = 17280000
print(f"\nIn the Netherlands, end of March: {100*pred(off_deaths_Netherlands,pop_Netherlands,popt_cp_data):.2f}% infected")
print(f"Estimated death rate in the Netherlands: {100*(off_deaths_Netherlands/(pred(off_deaths_Netherlands,pop_Netherlands,popt_cp_data)*pop_Netherlands)):.3f}%")


# As you can see, $R^2$ coefficients are stronger when using data from fewer countries, but then model we get is more biased and less generalizable to unseen countries.

# # Known issues
# 
# * There is no prior justification for the use of a model in the form of $y = m\times x^\alpha$ and it is unclear wether the $R^2$ coefficients we get are a strong enough validation of the model.
# 
# * There is some fair amount of variance in the dataset, which could be explained by inconsistencies in testing data, intrinsic differences in mortality between countries, differences of performance in tests used by different countries, or other factors.
# 
# * The conclusions are very dependent on Covid-19's RT-PCR test sensitivity (How many infected people would have tested negative? There is a lack of evidence about this)

# # Final words
# 
# I hope this notebook allowed you to have a better understanding of the Covid-19 pandemic. I did all this simply out of curiosity and, probably, looking for some good news. Unfortunately, even if the mortality rates found may strike you as being low, in reality, they are not lower than what most experts believe and the fact is that, even in countries as hardly hit as France or Italy, the prevalence of the virus is still well under $10\%$ nationally, which means that this is nowhere near the end of this crisis. Nonetheless, the proportion of infected found for areas like NYC is good news because it means that herd immunity could kick in sooner that expected, even if the mortality rate there is not as low as what could be hoped for.
