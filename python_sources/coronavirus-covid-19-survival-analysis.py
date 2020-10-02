#!/usr/bin/env python
# coding: utf-8

# As someone in the Data Sciences, I have been fascinated by how news agencies have covered data on global COVID-19 pandemic. 
# 
# What I think is challenging for people to grasp from these numbers is how to accurately account for risk factors driving one's chances of survival.  If we look at the popular datasets on Kaggle and news media, we typically see country-wide aggregates on Confirmed Cases and Fatalities.  The obvious idea would be to take the fatalities and divide through by the Confirmed Cases to get some heuristic for the chances people will die of the disease.  This can be very misleading as members of the current infected population may also die.
# 
# Another approach may be then to divide the Fatalities by the number of recovered patients.  This too can be misleading as it may take a long time before medical professionals can confirm that patients have fully recovered, and patients may die within hours of finally being tested.  This is more of an issue early on in the pandemic as there may exist medical guidelines on how long a person must be asymptomatic for before we can deem them fully recovered.  
# 
# The challenge this presents for people in the Data Sciences is how best to align the outcomes of patients from the same infection cohort who may have been exposed to the virus at about the same time.  Doing this between countries is very hard, as health systems can vary greatly and data recorded and released to the public can make harmonizing the data impossible.  
# 
# Luckily, looking at the Data Science for COVID-19 (DS4C) dataset, we can get a relatively detailed picture of patient cohorts which require little data harmonization as they fall in the same or similar healthcare system and should have been recorded similarly.  Additionally looking at this data, we benefit from South Korea's and Singapore's extensive and robust testing response which means we should have reasonably good estimates of the true proportion of Confirmed Cases to Fatalities, which also should be more robust under an analysis of patient records compared to population aggregates.  
# 
# Survival analysis is a branch of statistics for analyzing the expected duration of time until one or more events occur. These events can be the time until fall ill, make an insurance claim, respond to a treatment intervention or die.  Using this branch of statistics, we can more accurately account for artefacts in our data like the fact that we have not observed the outcome of sick patients as of yet, and that new patients may present themselves continuously throughout the pandemic which needs to be included in the estimation of our model. 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from statsmodels.duration.hazard_regression import PHReg
from statsmodels.graphics import regressionplots
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# The Data Science for COVID-19 (DS4C) dataset offers a rich set of information on patient demographics, locations and weather conditions, as well as data on cities and regions where these infections are observed.  This provides a fascinating treasure-trove of information to include in our analysis and explore.  
#   
# After looking through all the datasets, I chose to focus on two sources on patient information and on region information.  Using these two datasets, I hoped to control for region information, which may give insight into the healthcare system in that city or country, while estimating the impacts of age and gender on patient outcomes.  

# In[ ]:


ages = ['0s','10s','20s','30s','40s','50s','60s','70s','80s','90s','100s']
patientinfo = pd.read_csv('/kaggle/input/coronavirusdataset/PatientInfo.csv',
            index_col='patient_id',
            dtype={'patient_id': pd.Int64Dtype(),
                   'global_num': pd.Int64Dtype(),  'sex': pd.CategoricalDtype(), 'birth_year': pd.Int64Dtype(), 
                   'age': pd.CategoricalDtype(categories=ages, ordered=True),
                   'country': pd.CategoricalDtype(), 'province': pd.CategoricalDtype(), 'city': pd.CategoricalDtype(),
                   'disease': pd.CategoricalDtype(), 'infection_case':  pd.CategoricalDtype(),'infected_by': pd.Int64Dtype(),
                   'contact_number': pd.Int64Dtype(), 
                   'state': pd.CategoricalDtype()},
            parse_dates =['symptom_onset_date', 'confirmed_date', 'released_date', 'deceased_date'])

patientinfo


# In[ ]:


region = pd.read_csv('/kaggle/input/coronavirusdataset/Region.csv',
                     dtype={'province': pd.CategoricalDtype(), 'city': pd.CategoricalDtype()})
region


# In[ ]:


data = patientinfo.merge(region, on=['province', 'city'])
data


# The final design matrix for my experiment, included data on 'sex', 'birth_year', 'country' and 'elderly_population_ratio', as well as a squared truncated power basis for age with knot at the median age for patients in the dataset. I this matrix, I decided to exclude data points on Mongolian and Thai Patients as they form such a minority in the data and make estimation challenging.  
# 
# One challenge in applying Survival Analysis to this dataset was in deciding how best to compute the number of days from the patient presenting symptoms or confirmation of the illness until death.  This is challenging as patients get released and may die long in the future, far outside the study. In this case, I took the approach to estimate these patients days-to-fatality as the number of days until they reach the South Korean life expectancy, as we expect these patients, once released, to have recovered and to have a life expectancy in-line with the national average.  For patients older than the national life expectancy, I set their days-to-fatality as the length of the trial as I deemed this more faithful than allowing their days-to-fatality to take on a negative number.  
# 
# One challenge in performing this analysis is decided where to start the study for each patient.  While we may consider the study to have begun when their case was confirmed or when they first presented symptoms, we know for COVID-19 people can be latently infected long before they start exhibiting symptoms. As we are unable to estimate how long this time has been, I opted to start all patients when the first case of confirmed and to deem all other cases as left-censored. I believe this approach allows us to model more explicitly our uncertainty around when they contracted the disease. 

# In[ ]:


LIFE_EXPECTANCY = 83 * 365
X, y, died, started = (pd.get_dummies(data
                         .loc[:,['sex', 'birth_year', 'country',
                                 'elderly_population_ratio'
                                 ]]
                         .assign(country = lambda df: df.country.cat.remove_categories(['Mongolia', 'Thailand']))
                         .assign(age = lambda df: 2020 - df.birth_year)
                         .drop(columns='birth_year'), columns = ['sex', 'country'], drop_first=True)
                         .assign(age_squared = lambda df: (df.age - df.age.median()).clip(lower=0).pow(2))
                         .astype(np.float)
                         .pipe(lambda x: x.fillna(x.mean())),
                       
                       (data
                        .loc[:, ['symptom_onset_date', 'confirmed_date', 'released_date', 'deceased_date', 'birth_year']]
                        .assign(age = lambda df: 2020 - df.birth_year)
                        .assign(days_to_exit_study = lambda df: (df.loc[:, ['symptom_onset_date', 'confirmed_date', 'released_date', 'deceased_date']].max(1)
                                                           - df.loc[:, ['symptom_onset_date', 'confirmed_date', 'released_date', 'deceased_date']].min(1).min()).dt.days)
                        .assign(days_life_expectancy = lambda df: df.days_to_exit_study.where(df.released_date.isna(), df.days_to_exit_study + (LIFE_EXPECTANCY - df.age.where(lambda x: x < LIFE_EXPECTANCY,  LIFE_EXPECTANCY))* 365))
                        .loc[:, 'days_life_expectancy']
                        .astype(np.float)),
                       
                         (~data.deceased_date.isna().to_numpy()),
                       
                         (data.confirmed_date - data.confirmed_date.min()).dt.days.to_numpy())

# design matrix
X.corr()


# One concern for this study is the apparent correlation between variables in our design matrix.  While we expectedly observe strong correlation between our power basis on age and age, we also appear to observed a high Variable Inflation Factor for South Korean cases (as opposed to Singaporean cases) due to its aparent correlation with the elderly population ratio of particular regions.  

# In[ ]:


vif = pd.DataFrame([variance_inflation_factor(X.to_numpy(), i) for i in range(X.shape[1])], columns=['VIF'], index = X.columns)
vif


# One interesting aspect to observe is any latent structure within our data.  To uncover this latent structure, we have opted to visualize the Principle Components of our dataset against our patient outcomes.  While we observe some interesting outliers in the top-left of our plot, we also observe two interesting clusters in our dataset which appear to mainly correspond to the gender of our patients- with the bottom cluster reflecting male patients with higher levels of observed fatalities. 

# In[ ]:


from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA

pipeline = make_pipeline(StandardScaler(), PCA(2))
Z = pipeline.fit_transform(X.fillna(X.mean()))

evr = pipeline.named_steps['pca'].explained_variance_ratio_
columns = [f'Component {i} ({round(e*100)}%)' for i, e in enumerate(evr)]

died  = pd.Series(died)
released = ~data.released_date.isna()

components = pd.DataFrame(Z, columns=columns)

ax = (components
 .where(~released & ~died).dropna()
 .plot.scatter(x=columns[0], y=columns[1], c='LightGrey',
               label='Hospital', title='Principle Components',
               alpha=0.25))
ax = (components
      .where(died).dropna()
      .plot.scatter(x=columns[0], y=columns[1], c='Red', label='Dead', ax=ax))
plot = (components
      .where(released).dropna()
      .plot.scatter(x=columns[0], y=columns[1], c='LightGreen', label='Released', ax=ax))
plot


# Finally, we will estimate our Proportional Hazard Model. For those not accustomed to Proportional Hazard Models, the interpretation of Hazard Ratios (HR) can be challenging.  If we look at the computed Hazard Ratios, we can conclude that a Hazard Ratio of 2 means that a group has twice the chance of dying than a comparison group.  
# 
# Looking at our summary statistics, we can see the apparent statistical significance of the variables included in our model.  What appears interesting is the log Hazard Ratio on our 'age_squared' truncated power basis, which suggests that the Proportional Hazard for age flattens as patients get much older.  What also appears fascinating about this model, is how significant 'elderly_population_ratio' appears in the model, with patients in cities with 'elderly_population_ratios' of 1 having a 1.1970 time more likely chance of dying when controlling for the patient's age. This may suggest that in regions where there are high proportions of older people, hospitals may be overwhelmed by the extra care they require or choose to focus their attention on these at-risk patients.  

# In[ ]:


model = PHReg(endog=y, exog=X, status=died.astype(np.int), entry=started, ties="efron")
model_data = pd.DataFrame(model.exog, columns=model.exog_names)

results = model.fit()
results.summary()


# In[ ]:


(pd.DataFrame(np.exp(results.params), index=model.exog_names, columns=['Hazard Ratio'])
 .plot.bar(title='Times the Chance of Fatality after Case Confirmed - Not controlling for preexisting condition'))


# If we look at our proportional Hazard the days of the virus, we can see the proportional hazard increasing the longer one has exposure.  

# In[ ]:


bch = results.baseline_cumulative_hazard
bch = bch[0] # Only one stratum here
time, cumhaz, surv = tuple(bch)
plt.clf()
plt.plot(time, cumhaz, '-o', alpha=0.6)
plt.grid(True)
plt.xlabel("Days")
plt.ylabel("Cumulative hazard")
plt.title("Cumulative hazard against Days")


# We can also look at the Martingale residuals of our estimated model against age. Looking at these estimated, which we do see large variances around of median age, we do not see other clear signs of heteroskadasticity in the data. These variances, may provide insight into how faithful our estimated are for middle-ages people and may point to variables currently not-included in the model- like pre-existing conditions. 

# In[ ]:


plt.plot(model_data.age, results.martingale_residuals, 'o', alpha=0.5)
plt.xlabel("age")
plt.ylabel("Martingale residual")
plt.title('Martingale residual against age')


# Plots of the Schoenfeld residuals against time can be informative about whether the hazards are reasonably proportional, as modeled. Note that Schoenfeld residuals are not defined for censored cases. There is a slight increase in the Schoenfeld residuals with respect to age.

# In[ ]:


sr = results.schoenfeld_residuals
col = np.argwhere(np.array(model.exog_names) == 'age').item()
ii = np.flatnonzero(pd.notnull(sr[:,col]))

plt.plot(model.endog[ii], sr[ii,col], 'o')
plt.xlabel("Days")
plt.ylabel("Age Schoenfeld residual")


# I would love to get peoples feedback to my approach.  I am new to methods in Survival Analaysis and have really enjoyed the excercise in broadening my pallete.  I would be really interested to see similar approached people have taken on other datasets. 
