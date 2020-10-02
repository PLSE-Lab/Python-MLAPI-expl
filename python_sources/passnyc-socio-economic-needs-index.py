#!/usr/bin/env python
# coding: utf-8

# # Abstract
# 
# In this notebook, we attempt to find predictors of higher percentage of SHSAT taking at the kind request of the PASSNYC organization. We identify the relationship between academic outcomes and willingness of students to sit the SHSAT. We then enrich PASSNYC's data with four additional, public data sources. These data sources contain information on: housing, crime, income and transportation. We utilize the enriched school explorer dataset in order to derive a socio-economic need index that predicts the academic outcomes of each school. Finally, we verify our model and list several conclusions and recommendations for next best actions, based on the model's results.

# In[ ]:


#numeric
import numpy as np
import pandas as pd
import scipy
from collections import defaultdict

#visualization
import matplotlib.pyplot as plt
import seaborn as sns
import folium

from IPython.display import display

plt.style.use('bmh')
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['axes.titlepad'] = 25
sns.set_color_codes('pastel')

#Pandas warnings
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 300)
pd.set_option('display.max_rows', 100)

#system
import os
import gc
import datetime
import re
#print(os.listdir('../input'))

#Machine learning
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, normalize, RobustScaler
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, KFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDRegressor
from sklearn import tree
from sklearn.metrics import make_scorer, mean_squared_error, r2_score, mean_squared_log_error, explained_variance_score, mean_absolute_error
from sklearn.dummy import DummyRegressor


# # Table Of Contents
# 
# 1. [Definition](#Definition)
#     1. [Project Overview](#Project-Overview)
#     2. [Problem Statemet](#Problem-Statemet)
#     3. [Datasets and Input](#Datasets-and-Input)
#     4. [Solution Statement](#Solution-Statement)
#     5. [Benchmark Model](#Benchmark-Model)
#     6. [Evaluation Metrics](#Evaluation-Metrics)
# 2. [Analysis](#Analysis)
#     1. [Identifying The Relationship Between SHSAT Registration And Taking](#Identifying-The-Relationship-Between-SHSAT-Registration-And-Taking)
#     2. [Investigating The Relationship Between Academic Proficiencies and Socio-Economic Factors](#Investigating-The-Relationship-Between-Academic-Proficiencies-and-Socio-Economic-Factors)
#     3. [Data Enrichment](#Data-Enrichment)
#         1. [NY Crime Dataset](#NY-Crime-Dataset)
#         2. [US Household Income Data](#US-Household-Income-Data)
#         3. [NY Housing Data](#NY-Housing-Data)
#         4. [School Directory Dataset](#School-Directory-Dataset)
# 3. [Modeling](#Modeling)
# 4. [Socio-Economic Needs Visualization](#Socio-Economic-Needs-Visualization)
# 5. [Conclusion](#Conclusion)
# 6. [Improvement Suggestions](#Improvement-Suggestions)
# 

# # Definition
# ## Project Overview
# New York City is home to nine specialized schools [[1]](https://en.wikipedia.org/wiki/Specialized_high_schools_in_New_York_City 'Wikipedia article'). These schools present enhanced developmental opportunities for academically gifted students. However, admission to all but one of these schools is granted based on the results of a Specialized High Schools Admissions Test (SHSAT) [[2]](https://en.wikipedia.org/wiki/Specialized_High_Schools_Admissions_Test 'Wikipedia article'). The test has two general categories of student assessment:
# 
# * English Language Arts and Literacy (ELA)
# * Mathematics
# 
# Due to the fact that students must actively pursue admission themselves, the following situation could occur [[3]](http://www.passnyc.org/opportunity/ 'PASSNYC's webpage'):
# 
# > "First, a non-trivial share of high-achieving students does not sit for the SHSAT at all. This may reflect a lack of interest, a lack of resources for test preparation, or a poor understanding of their odds of admission."
# 
# That is, academically talented individuals may not have access to the informational, motivational or financial resources necessary to prepare and take the SHSAT. Furthermore, PASSNYC focuses on promoting diversity and encouraging students from areas that are historically underrepresented in SHSAT registration.
# 
# To achieve their goal, PASSNYC relies on publicly-available data in order to locate such areas as well as provide (through their partners) services such as [[4]](http://www.passnyc.org/our-partners 'PASSNYC's webpage'):
# 
# * Test prep and tutoring
# * Afterschool programs and activities
# * Resources for parents
# * Community groups
# 
# ## Problem Statement
# With this Kaggle challenge, the PASSNYC organization aims to improve its outreach services which in turn will enable more students to take the SHSAT. To that end, the requirements for this project are as follows:
# 
# 1. To identify publicly-available data sources containing predictors of outreach services necessity.
# 2. To quantify the socio-economic challenges that students face at a granular level.
# 3. To assist PASSNYC in identifying schools where minority as well as economically disadvantaged students would benefit from the outreach services.
# 
# ## Datasets and Input
# In this analysis, we will utilize the following datasets:
# 
# * PASSNYC's school explorer dataset. This dataset contains records on more than 1200 schools in New York. The records list information on:
#     * The schools average Math and ELA proficiencies
#     * Every school's corresponding ethnicity composition
#     * The school's economic need index
#     * Custom measures (e.g. trust percentage, economic need index, supportive environment percentage)
# * District 5 SHSAT registration data. This dataset is provided by the NYC Department of Education and informs about the SHSAT registration and participation  from 2013 to 2016 for the district of Central Harlem.
# * [NYC Crime Data](https://www.kaggle.com/adamschroeder/crimes-new-york-city#Population_by_Borough_NYC.csv 'Kaggle'). This dataset contains 1.05 million complaints with a crime date format of (month/day/year). The latest crime date for this dataset is from December 31, 2015. The dataset contains, in total, 468 thousand crimes reported throughout 2015.
# * [US Household Income Data](https://www.kaggle.com/goldenoakresearch/us-household-income-stats-geo-locations 'Kaggle'). This dataset contains the mean income, median income, standard deviation of the income as well as detailed location features for more than 32 thousand locations in the US. For the state of New York, there are 2160 entries.
# * [NY Housing Data](https://www.kaggle.com/new-york-city/housing-new-york-units 'Kaggle'). The NY Housing dataset contains information about 2900 construction projects within 116 ZIP codes from New York having start date between January 2014 and April 2018. The dataset contains information about (not exhaustive list):
#     * Project start and completion dates
#     * Project location
#     * Construction type (new construction or preservation)
#     * Number of units (rental, low income, medium income, other income, ownership, number of bedrooms, etc.)
# * [NYC High School Directory](https://www.kaggle.com/new-york-city/nyc-high-school-directory#2016-doe-high-school-directory.csv 'Kaggle'). This dataset contains the prospecti for 437 schools in New York City. Since this dataset contains information about the available bus and subway lines as means of transport to each school, an inference can be made on the transportation needs of the neighborhoods hosting each school.
# 
# ## Solution Statement
# In order to address the requirements laid out in the [Problem Statement](#Problem-Statement) chapter, this project will:
# 
# 1. Demonstrate that there is a relationship between a school's Math and ELA proficiencies and the percentage of students taking the test out of all the students registering for the test. That is the willingness of students to sit the SHSAT is linked to their academic proficiency. The academic proficiency is in turn linked to the socio-economic environment of each school (neighborhood income, housing options, transportation infrastructure, safety and diversity).
# 2. Import, prepare and refine public datasets that provide information about the above-mentioned socio-economic environments of each school in PASSNYC's school explorer dataset.
# 3. Build a model that predicts a school's normalized academic proficiencies based on socio-economic factors derived from the public datasets.
# 4. Provide visualization of the challenges schools face and next best action for each school in PASSNYC's school explorer dataset.
# 
# ## Benchmark Model
# Since the task in point 3 from the [Solution Statement](#Solution-Statement) is a regression task for the prediction of a normalized target variable ranging from 0 to 1, we will utilize 3 dummy predictors to evaluate the performance of our actual model against:
# 
# * A dummy predictor that always predicts 0
# * A dummy predictor that always predicts 0.5
# * A dummy predictor that always predicts 1
# 
# We will evaluate these regressors to our actual candidate models based on the [Evaluation Metrics](#Evaluation-Metrics).
# 
# ## Evaluation Metrics
# We will utilize the following evaluation metrics for our candidate models:
# 
# * Mean squared error: $MSE(y, \hat{y}) = \frac{1}{n_{samples}}\sum_{i=0}^{n_{samples} - 1}(y_i - \hat{y}_i)^2$
# * R2 score: $R^2(y, \hat{y}) = 1 - \frac{\sum_{i=0}^{n_{samples} - 1}(y_i - \hat{y}_i)^2}{\sum_{i=0}^{n_{samples} - 1}(y_i - \bar{y}_i)^2}$
# * Explained variance score: $EV(y, \hat{y}) = 1 - \frac{Var\{y - \hat{y}\}}{Var\{y\}}$
# 

# # Analysis
# ## Identifying The Relationship Between SHSAT Registration And Taking
# We will first import the `D5 SHSAT Registrations and Testers` dataset. The dataset contains the features:
# 
# * `Number of students who registered for the SHSAT` (we will rename this feature to num `num registered`  for conciseness)
# * `Number of students who took the SHSAT` : (we will rename this feature to num `num took`  for conciseness)
# 
# Using these two features, we will derive the percentage of students who took the SHSAT out of all registered as: `num took / num registered`. The new feature will be called ``num took pct registered`.

# In[ ]:


#import the dataset
shsat_res = pd.read_csv('../input/data-science-for-good/D5 SHSAT Registrations and Testers.csv')

#rename columns to short, query-friendly names
shsat_res.rename(columns = {'School name' : 'school',
                            'Year of SHST' : 'year',
                            'Grade level' : 'grade',
                            'Enrollment on 10/31' : 'enrollment',
                            'Number of students who registered for the SHSAT' : 'num_registered',
                            'Number of students who took the SHSAT' : 'num_took'},
                 inplace = True)

#convert school name to lower caps string (used when joining other data sources)
shsat_res.school = shsat_res.school.str.lower()

#derive additional features
shsat_res['took_pct_registered'] = shsat_res['num_took'] / shsat_res['num_registered']
shsat_res['took_pct_registered'] = shsat_res['took_pct_registered'].fillna(0)

shsat_res['registered_pct_enrolled'] = shsat_res['num_registered'] / shsat_res['enrollment']
shsat_res['registered_pct_enrolled'] = shsat_res['registered_pct_enrolled'].fillna(0)


# We will now create a scatter plot for the number of students who registered for SHAT and the number of students who took the SHSAT for all available years and grades in the dataset:

# In[ ]:


#plt.title('Number Of Students Who Registered / Took The SHSAT')
sns.jointplot(x = 'num_took', y = 'num_registered', size = 9, data = shsat_res)


# There seems to be a group of schools that form a linear trend between the number of students registered and number of students taking the test. On this trendline, about half of the students that register, take the test. However, there is also a group of schools where this proportion is greatly reduced. At the top left quadrant, we can clearly observe that there are schools where only 1 out of 10 registered for the SHSAT takes it. Since our dataset contains only 140 points, it could be easy to label this occurrence as noise, but we should also remember that this is an actual school. This could mean that this school is in a neighborhood with a high degree of social distress or that some sort of educational intervention is required. In any case, we should break down this data by year and grade. This will allow us to understand whether these issues are related to certain point in time or they are contingent on the number of years of schooling, or none of the above:

# In[ ]:


sns.lmplot(x = 'num_took', y = 'num_registered', col = 'year', hue = 'grade', data = shsat_res)


# As we can see, there are very few 9-graders registering for or taking the SHSAT (most of the points corresponding to 9th grade above are clustered around 0). The trend formed by the points corresponding to 8th grade seems to have a relatively stable slope, however there are outliers.
# 
# To further understand the time series, we will plot the distributions of the number of students taking the SHSAT by year and grade:

# In[ ]:


fig = plt.figure(figsize = (20, 8))

ax1 = fig.add_subplot(1, 2, 1)
ax1.set_title('Number Of Students Who Took The SHSAT by Year and Grade')
sns.boxplot(x = 'year', y = 'num_took', hue = 'grade', data = shsat_res)
ax2 = fig.add_subplot(1, 2, 2)
ax2.set_title('Number Of Students Who Registered For The SHSAT by Year and Grade')
sns.boxplot(x = 'year', y = 'num_registered', hue = 'grade', data = shsat_res)


# As we saw previously, there are very few 9-graders who sit the SHSAT. For the 8-graders, however, it seems like the distribution is spreading. This could be interpreted as a case of divergence between schools in terms of grades, student confidence or socio-economic conditions. We should also note that the median of the distribution is shifting downwards.
# 
# In order to better understand these trends, we will use the `took pct registered` ratio we derived earlier. We will split the entire range of the ratio into 3 equal bins, which we will label `low`, `medium` and `high`.

# In[ ]:


shsat_res['took_pct_registered_label'] = pd.cut(x = shsat_res.took_pct_registered, bins = 3, labels = ['low', 'medium', 'high'])


# In[ ]:


sns.lmplot(x = 'num_took', y = 'num_registered', col = 'took_pct_registered_label', hue = 'took_pct_registered_label', data = shsat_res)
#sns.lmplot(x = 'num_took', y = 'num_registered', col = 'year', hue = 'took_pct_registered_label', data = shsat_res)


# The grouping we did seems to separate well the group that initially looked as outliers to the main trend. In the case of `high` percentage of students who take the SHSAT, there is almost no scatter and all the points seem to sit on the same line. We will attempt to explain these trends by using the `School Explorer` dataset. This dataset contains social and academic information about the schools considered for outreach services by the PASSNYC organization. 

# In[ ]:


#import the data
school_explorer = pd.read_csv('../input/data-science-for-good/2016 School Explorer.csv')

#rename 1 column with inconsistent capitalization
school_explorer.rename(columns = {'Grade 3 Math - All Students tested' : 'Grade 3 Math - All Students Tested'}, inplace = True)

#rename school column (to match the name of the corresponding column in the SHSAT results dataset)
school_explorer.rename(columns = {'School Name' : 'school'}, inplace = True)

#convert school name to lower caps string (used when joining other data sources)
school_explorer['school'] = school_explorer['school'].str.lower()

#convert the community school string to flag
school_explorer['Community School?'] = school_explorer['Community School?'].map({'Yes' : 1, 'No' : 0})

#convert the income estimate to numeric
school_explorer['School Income Estimate'] = school_explorer['School Income Estimate'].str.replace(',', '')
school_explorer['School Income Estimate'] = school_explorer['School Income Estimate'].str.replace('$', '')
school_explorer['School Income Estimate'] = pd.to_numeric(school_explorer['School Income Estimate'])

#convert percentage columns to numeric format
prc = re.compile(r'%', re.I)
prc_columns = pd.Series(school_explorer.loc[0].apply(lambda x: True if prc.search(str(x)) else False))
prc_columns = prc_columns[prc_columns == True].index.tolist()

for c in prc_columns:
    school_explorer[c] = school_explorer[c].str.replace('%', '')
    school_explorer[c] = pd.to_numeric(school_explorer[c])
    school_explorer[c] = school_explorer[c] / 100

#derive charter school flag
charter_school = re.compile(r'charter school' , re.I)
school_explorer['charter_school'] = np.vectorize(lambda x : 0 if charter_school.search(x) is None else 1)(school_explorer['school'].values)

#derive diversity percentage
school_explorer['diversity_prc'] = 1 - school_explorer['Percent White']

#discard grade columns
grade_cols = ['Grade 8 ELA 4s - American Indian or Alaska Native',
              'Grade 8 ELA 4s - Black or African American',
              'Grade 8 ELA 4s - Hispanic or Latino',
              'Grade 8 ELA 4s - Asian or Pacific Islander',
              'Grade 8 ELA 4s - White',
              'Grade 8 ELA 4s - Multiracial',
              'Grade 8 ELA 4s - Limited English Proficient',
              'Grade 8 ELA 4s - Economically Disadvantaged',
              'Grade 8 ELA 4s - All Students',
              'Grade 8 ELA - All Students Tested']

all_grade_cols = []
for g in range(3, 9):
    for c in grade_cols:
        for s in ['ELA', 'Math']:
            all_grade_cols.append(c.replace('8', '%s' % g).replace('ELA', '%s' % s))

school_explorer.drop(all_grade_cols, axis = 1, inplace = True)

#discrad other irrelevant columns
school_explorer.drop(['Adjusted Grade', 'Other Location Code in LCGMS', 'Grades',
                      'Grade Low', 'Grade High', 'Percent Black / Hispanic', 'Percent Asian',
                      'Percent Black', 'Percent Hispanic', 'Rigorous Instruction Rating',
                      'Collaborative Teachers Rating', 'Supportive Environment Rating',
                      'Effective School Leadership Rating',
                      'Strong Family-Community Ties Rating', 'Trust Rating',
                      'Student Achievement Rating'], axis = 1, inplace = True)


# In[ ]:


shsat_res = shsat_res.merge(school_explorer, how = 'left', left_on = 'school', right_on = 'school')


# We will start by tabulating some aggregated values of the mean Math and ELA proficiency for our `took percentage registered` bins. We will do that by taking the results for the latest possible year from the time series (2016) and for the grade with highest number of students who took the test (8th grade).

# In[ ]:


pd.pivot_table(data = shsat_res[(shsat_res.year == 2016) & (shsat_res.grade == 8)],
               index = 'took_pct_registered_label',
               values = ['Average ELA Proficiency', 'Average Math Proficiency'],
               aggfunc = ['mean', 'median']).style.format('{:.2}')


# In addition, we will visualize the distribution of the Math and ELA proficiencies by `took percentage registered` bin.

# In[ ]:


fig = plt.figure(figsize = (20, 18))

to_plot = shsat_res[(shsat_res.year == 2016) & (shsat_res.grade == 8)]

ax1 = fig.add_subplot(2, 2, 1)
sns.boxplot(x = 'took_pct_registered_label', y = 'Average Math Proficiency', data = to_plot)
ax2 = fig.add_subplot(2, 2, 2)
sns.boxplot(x = 'took_pct_registered_label', y = 'Average ELA Proficiency', data = to_plot)
ax3 = fig.add_subplot(2, 2, 3)
sns.violinplot(x = 'took_pct_registered_label', y = 'Average Math Proficiency', data = to_plot)
ax4 = fig.add_subplot(2, 2, 4)
sns.violinplot(x = 'took_pct_registered_label', y = 'Average ELA Proficiency', data = to_plot)


# As we can see, both the tabular format as well as the distribution charts show that the Math and ELA proficiencies (which we will hereafter refer to as 'academic proficiencies') are linked to the proportion of students who took the SHSAT out of all registered students. Let's provide additional confirmation of this link by investigating:
# 
# * The correlation between the `took pct registered` and the rest of the socio-academic indicators of the `school explorer` dataset.
# * The correlation between the academic proficiencies and the rest of the socio-academic indicators of the `school explorer` dataset.
# 
# In the three bar charts below, we will inspect the correlation fingerprint of the `took pct registered`, `Average Math Proficiency` and `Average ELA Proficiency` variables.

# In[ ]:


school_explorer_ind = ['Percent of Students Chronically Absent', 'Rigorous Instruction %',
                       'Economic Need Index', 'Student Attendance Rate', 'diversity_prc',
                       'Community School?', 'charter_school', 'Trust %',
                       'Collaborative Teachers %', 'Supportive Environment %',
                       'Effective School Leadership %', 'Strong Family-Community Ties %',
                       'School Income Estimate']

target_ind = ['took_pct_registered', 'Average Math Proficiency', 'Average ELA Proficiency']


# In[ ]:


to_plot = shsat_res[(shsat_res.year == 2016) & (shsat_res.grade == 8)]

shsat_res_corr = to_plot[target_ind + school_explorer_ind].corr()

box_style = dict(boxstyle = 'round', fc = (1.0, 0.7, 0.7), ec = 'none')
arrowprp = dict(arrowstyle = 'wedge,tail_width=1.',
                fc = (1.0, 0.7, 0.7),
                ec = 'none',
                patchA = None,
                patchB = None,
                relpos = (0.2, 0.5))

fig = plt.figure(figsize = (16, 26))
plt.subplots_adjust(hspace = 0.6)

for c, i in zip(target_ind, range(1, len(target_ind) + 1)):
    ax = fig.add_subplot(3, 1, i)
    to_plot = shsat_res_corr.loc[c, :].drop(target_ind)
    x_ticks = range(len(to_plot.index))
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(to_plot.index, rotation = 45)
    ax.set_title('Correlation Between %s And The Socio-Academic Features In The School Explorer Dataset' % c)
    if c == 'took_pct_registered':
        ax.annotate(s = '''Diversity Prc is less influential
        for Took Pct Registered''',
              xy = (6, 0.3),
              xycoords = 'data',
              xytext = (55, 0),
              textcoords = 'offset points',
              size = 20,
              va = 'center',
              bbox = box_style,
              arrowprops = arrowprp)
        
        ax.annotate(s = '''Community School is more influential
        for Took Pct Registered''',
              xy = (5, -0.3),
              xycoords = 'data',
              xytext = (55, 0),
              textcoords = 'offset points',
              size = 20,
              va = 'center',
              bbox = box_style,
              arrowprops = arrowprp)
    
    ax.bar(x_ticks, to_plot)
#fig.autofmt_xdate()


# As we can see, the correlation between the three target variables is very similar safe for the diversity percentage and community school indicator. This means that we can use the Math and ELA proficiencies of every school as proxies for the took as percentage of registered labels. This is a relief as with the limited data from NYC's department of education, we had only 21 data points for 2016, grade 8. In the school explorer (where we have Math and ELA proficiency data for all schools, we have more than 1200 data points).
# 
# In the next chapter we will analyze the relationship between the academic proficiencies and various socio-economic factors. We will then prove that academic proficiency is highly-contingent on the socio-economic environment of each school.

# ## Investigating The Relationship Between Academic Proficiencies and Socio-Economic Factors
# 
# As we pointed out in the [Datasets and Inputs](#Datasets-and-Inputs) chapter, we will use several public datasets in order to obtain more data on the socio-economic environment of each school. We will then calculate the correlation of such data with the academic proficiencies.
# 
# We will first look at the correlation between a selection of school-based socio-academic features and the target features (`took pct registered`, `Average Math Proficiency` and `Average ELA Proficiency`) within the `D5 SHSAT Registrations and Testers` dataset (note that this comparison is for our limited 21-point selection) and the `School Explorer` dataset:

# In[ ]:


fig = plt.figure(figsize = (30, 12))
ax1 = fig.add_subplot(1, 2, 1)
ax1.set_title('Correlation Between The Features Of The D5 Dataset')
sns.heatmap(shsat_res_corr, annot = True, cmap = 'Spectral')
ax2 = fig.add_subplot(1, 2, 2)
ax2.set_title('Correlation Between The Features Of The School Explorer Dataset')

target_ind.pop(target_ind.index('took_pct_registered'))
school_explorer_corr = school_explorer[target_ind + school_explorer_ind].corr()

sns.heatmap(school_explorer_corr, annot = True, cmap = 'Spectral')


# From the comparison between the correlation within the two datasets, we can establish some important points:
# 
# 1. **We observe good levels of correlation between the average Math and ELA proficiencies and the took pct registered ratio. Therefore we hypothesize that we could use the average Math and ELA proficiencies of the schools as proxies for the took pct registered ratio. Later on in the project, we can test whether the modeling results will explain the took pct registered ratio.**
# 2. **There is a strong correlation between the average Math and ELA proficiency of a school in both datasets. This means that good school provide strong training in both subject areas and there could be very few if any exceptions where a school provides only good training in Math and not ELA. This means that we could use either of those characteristics in order to measure what percentage of students from a given school will take the SHSAT . Furthermore, we can find schools with lower Math proficiencies and identify the social issues they face. This will allow PASSNYC to provide the required outreach services and increase that school's percentage of students who sit the SHSAT test.**
# 3. **There is a strong inverse correlation between the academic proficiencies and the economic need index in both datasets. This could indicate that better funding leads to better academic outcomes.**
# 4. **There is an inverse correlation between the academic proficiencies and the diversity percentage in both datasets. This could mean that more ethnically-diverse schools could be located in neighborhoods facing more serious socio-economic issues than less diverse schools. This, in turn, translates to worse academic outcomes. Such issues could be related to: transportation, safety, income, housing, etc. We will investigate this relationship further on in the project in order to allow PASSNYC to better target such schools.**
# 5. There is a correlation between the average Math proficiency and the type of schools. It seems like charter schools generally have higher proficiencies based on the D5 dataset. However, in the school explorer dataset, there is a weaker correlation between the math proficiency and charter schools, but a stronger inverse correlation with the community schools. We need to further investigate:
#     * Whether the community school inverse correlation is related to the social conditions of neighborhoods having community schools.
#     * Whether community schools are more present in certain neighborhoods than other.
#     * Whether the reduced correlation between math proficiency and charter schools is due to a low count of charter schools in the school explorer dataset.
# 6. There is a correlation between the academic proficiencies and the following custom school measures in both datasets:
#     * Collaborative teachers
#     * Effective school leadership
#     * Strong family-community ties
#     * Supportive environment
# 7. There is a correlation between the rigorous instruction percentage custom measure and the following custom measures:
#     * Collaborative teachers
#     * Supportive environment
#     * Effective school leadership
#     * Strong family-community ties
# 8. There is a correlation between the academic proficiencies and the percentage of chronically absent students in the school explorer dataset.
# 
# Points 6 and 7 could generally mean that all the mentioned custom measures are more or less indicators of a well operating school with good social integration.
# 
# Quickly returning to point 5, let's see the number of community and charter schools:

# In[ ]:


pd.pivot_table(data = school_explorer, index = 'Community School?', columns = 'charter_school', values = 'school', aggfunc = 'count')


# There are even more charter schools (155) than community schools (76). Together the two types of schools constitute about 18% of all the schools, so we could use those characteristics as predictors for certain academic outcomes.
# 
# We are now ready to enrich our school explorer dataset with new publicly-available data.

# ## Data Enrichment
# ### NY Crime Dataset
# As mentioned in the [Datasets and Input](#Datasets-and-Input) chapter, the New York City crime dataset contains 1.05 million complaints with a crime date format of (month/day/year). The dataset's key features are:
# 
# * Offense code
# * Offense category (violation / misdemeanor / felony)
# * New York borough where the crime was committed
# * Precinct where the crime was committed
# * Latitude and longitude of the crime's location
# * Date the crime was committed on
# 
# There are two main issues with this dataset:
# 
# * The date format is inconsistent throughout the dataset and a small percentage of the crimes have a missing date. To address this issue, we will create a date conversion function. The function will parse as dates all entries where the date is in the format month/day/year. Missing values and implicitly-missing values (e.g. '01/01/1007') will be discarded.
# * The most straightforward way to aggregate the crimes by location is by pivoting the sum of crimes around the precinct variable. This will group and sum all crimes by their precinct, however we need to find a way to map these precincts to their corresponding schools from the `school explorer` dataset. We can address this challenge by mapping the precincts to their corresponding ZIP codes and aggregating the crimes by ZIP code. The ZIP code feature is already present in the `school explorer` dataset and we can use it as a join key.
# 
# Let's import the dataset, make the two transformations above and show 5 rows from the result:

# In[ ]:


ny_crimes = pd.read_csv('../input/crimes-new-york-city/NYPD_Complaint_Data_Historic.csv', index_col = 0)
ny_precinct = pd.read_csv('../input/nyc-precinct-zip-codes/precinct_zip.csv')

#dates cannot be parsed because of inconsistent formatting
#we will derive a function that parses the dates where the format is month/day/year
#the rest of the rows will return not-any-date (nad)
def conv_date(date):
    try:
        return datetime.datetime.strptime(date, '%m/%d/%Y').date()
    except ValueError:
        return 'nad'
    except TypeError:
        return 'nad'

ny_crimes['date'] = ny_crimes.CMPLNT_FR_DT.apply(conv_date)

ny_crimes.drop(['CMPLNT_FR_TM', 'CMPLNT_TO_DT', 'CMPLNT_TO_TM',
                'RPT_DT', 'PD_CD', 'PD_DESC', 'LOC_OF_OCCUR_DESC',
                'PREM_TYP_DESC', 'PARKS_NM', 'HADEVELOPT',
                'X_COORD_CD', 'Y_COORD_CD', 'CMPLNT_FR_DT'],
               axis = 1,
               inplace = True)

ny_crimes = ny_crimes[ny_crimes.date != 'nad']

ny_crimes = ny_crimes.merge(ny_precinct, how = 'left', left_on = 'ADDR_PCT_CD', right_on = 'precinct')


# In[ ]:


ny_crimes.head()


# Let's sample 500 random offenses committed in 2015 and plot their location on a map to get a glimpse of their geographical distribution (violations are annotated in blue, misdemeanors are annotated in orange and felonies are annotated in red):

# In[ ]:


m = folium.Map(location = [40.75, -73.85], tiles = 'cartodbpositron', zoom_start = 11.25)

to_plot = ny_crimes[(ny_crimes.date >= datetime.date(2015, 1, 1)) &
                    (ny_crimes.date <= datetime.date(2015, 12, 31)) &
                    (ny_crimes.Latitude.notna())].sample(500)

for i in range(0, len(to_plot)):
    if to_plot.iloc[i].LAW_CAT_CD == 'VIOLATION':
        c = 'blue'
    elif to_plot.iloc[i].LAW_CAT_CD == 'MISDEMEANOR':
        c = 'orange'
    else:
        c = 'red'
    
    folium.Circle(location = [to_plot.iloc[i].Latitude ,
                              to_plot.iloc[i].Longitude],
                  radius = 150,
                  color = c,
                  fill = True,
                  stroke = True,
                  fillOpacity = 0.2
   ).add_to(m)

mapWidth, mapHeight = (400, 500)
m


# We will now aggregate the crimes committed in 2015 by precinct (and its corresponding zip code).

# In[ ]:


ny_crimes_by_zip = pd.pivot_table(data = ny_crimes[(ny_crimes.date >= datetime.date(2015, 1, 1)) & (ny_crimes.date <= datetime.date(2015, 12, 31))],
                                  index = 'zip',
                                  columns = 'LAW_CAT_CD',
                                  values = 'date',
                                  aggfunc = 'count',
                                  fill_value = 0)

ny_crimes_by_zip.head()


# Let's now see a brief description of the resulting dataset:

# In[ ]:


ny_crimes_by_zip.describe().style.format('{:.2}')


# In order to assert that the resulting variables have a good variance (i.e. can be used as predictors of crime given an area), we can inspect the variance of each column as a percentage of the range of the column, defined so: $VarPct = \frac{std}{max - min}$

# In[ ]:


to_plot = ny_crimes_by_zip.describe().loc['std'] / (ny_crimes_by_zip.describe().loc['max'] - ny_crimes_by_zip.describe().loc['min'])
for c in to_plot.index:
    print('The standard deviation as percentage of the range for {} is: {:.2%}'.format(c, to_plot[c]))


# One slight complication is the following:

# In[ ]:


print('''The number of unique ZIP codes in the NY Crimes dataset is: {}. \n
The number of unique ZIP codes in the school explorer dataset is: {}'''.format(len(ny_crimes_by_zip.index.unique()),
                                                                               len(school_explorer.Zip.unique())))


# To solve this issue, we will impute the median number of crimes for all ZIP codes missing from the NY crime dataset. Finally, we will produce a percentage of all felonies for every unique ZIP code.

# In[ ]:


school_explorer = school_explorer.merge(ny_crimes_by_zip, how = 'left', left_on = 'Zip', right_index = True)

imp = SimpleImputer(strategy = 'median', copy = False)

columns_to_impute = ny_crimes_by_zip.columns.tolist() + ['School Income Estimate']

imputed_columns = pd.DataFrame(imp.fit_transform(school_explorer[columns_to_impute]), columns = columns_to_impute)
school_explorer.drop(columns_to_impute, axis = 1, inplace = True)
school_explorer = school_explorer.join(imputed_columns)


# Finally we will engineer a feature - we will calculate the percentage of all crimes in a ZIP code out of all crimes committed in New York City. This will give us a measure of the security challenges faced by this area.

# In[ ]:


for c in columns_to_impute:
    school_explorer['pct_%s' % c] = school_explorer[c] / school_explorer[c].sum()


# We can now join the result to the school explorer dataset and inspect the correlation of the new feature.

# In[ ]:


crime_ind = ['pct_FELONY']


# In[ ]:


school_explorer_corr = school_explorer[target_ind + crime_ind + school_explorer_ind].corr()
to_plot = school_explorer_corr.loc['pct_FELONY', :].sort_values(ascending = False).drop(crime_ind)#.transpose().iloc[:, 1:]

#fig = plt.figure(figsize = (14, 6))
x_ticks = range(len(to_plot.index))
plt.xticks(x_ticks, to_plot.index, rotation = 90)
plt.title('Correlation Between Percentage of Felonies Committed In a Certain ZIP Code \nAnd The Socio-Academic Features In The School Explorer Dataset')
plt.bar(x_ticks, to_plot)


# As we can see, the percentage of all felonies within a certain neighborhood has:
# 
# * Its strongest correlation with the economic need index and the diversity percentage
# * Certain levels of correlation with the percent of students chronically absent
# * Certain levels of inverse correlation with the academic proficiencies
# 
# We can now proceed with the US household income data.

# ### US Household Income Data
# The US Household Income dataset contains 2160 entries for the state of New York. All but 5 of the zip codes in the school explorer dataset have corresponding entries in the US Household Income dataset. The features of interest in those entries are the mean, median incomes and their standard deviation:

# In[ ]:


us_household_income = pd.read_csv('../input/us-household-income-stats-geo-locations/kaggle_income.csv', encoding = 'ISO-8859-1')
us_household_income.rename(columns = {'Mean' : 'income_mean', 'Median' : 'income_median', 'Stdev' : 'income_stdev'}, inplace = True)


# In[ ]:


print('The number of ZIP codes from the school explorer dataset, not present in the US Household Income dataset is: {}'      .format(len(school_explorer[~(school_explorer.Zip.isin(us_household_income.Zip_Code))])))


# Let's look at 5 rows from the dataset (for the state og Ney York):

# In[ ]:


us_household_income[us_household_income.State_ab == 'NY'].head()


# There are multiple entries per ZIP code, so we should aggregate the values. We can use the median value per ZIP code for our final income estimate.

# In[ ]:


us_household_income_med = pd.pivot_table(data = us_household_income[(us_household_income.State_ab == 'NY') & (us_household_income.income_mean != 0)],
                                         index = 'Zip_Code',
                                         values = ['income_mean', 'income_median', 'income_stdev'],
                                         aggfunc = 'mean')


# Thanks to the detailed location features, we can directly join the income features to the school explorer dataset (using the ZIP code as a join key). We will subsequently impute the missing values and inspect the correlation with our existing features.

# In[ ]:


columns_to_impute = ['income_mean', 'income_median', 'income_stdev']

school_explorer = school_explorer.merge(us_household_income_med, how = 'left', left_on = 'Zip', right_index = True)

imputed_columns = pd.DataFrame(imp.fit_transform(school_explorer[columns_to_impute]), columns = columns_to_impute)

school_explorer.drop(columns_to_impute, axis = 1, inplace = True)
school_explorer = school_explorer.join(imputed_columns)


# In[ ]:


income_ind = ['income_mean', 'income_median', 'income_stdev']


# In[ ]:


school_explorer_corr = school_explorer[target_ind + income_ind + crime_ind + school_explorer_ind].corr()

fig = plt.figure(figsize = (22, 9))
plt.subplots_adjust(wspace = 0.75)

for c, i in zip(income_ind, range(1, len(income_ind) + 1)):
    ax = fig.add_subplot(1, 3, i)
    to_plot = school_explorer_corr.loc[c, :].sort_values(ascending = False).drop(income_ind)
    x_ticks = range(len(to_plot.index))
    ax.set_yticks(x_ticks)
    ax.set_yticklabels(to_plot.index, rotation = 0)
    ax.set_title('Correlation Between %s \n And The Socio-Academic Features \n In The School Explorer Dataset' % c)
    ax.barh(x_ticks, to_plot)


# All three features have a very similar correlation profile. The intensity of the relation between income and the rest of the features is most pronounced in the mean income. The mean income has:
# 
# * Its strongest (inverse) correlation with the economic need index of each school
# * Inverse correlation with the diversity percentage
# * Correlation with the academic proficiencies
# * Certain levels of correlation with the percentage of felonies, percentage of chronically absent students and the supportive environment percentage
# 
# We can now proceed with the New York housing data.

# ### NY Housing Data
# The NY Housing dataset contains information about 2900 construction projects within 116 ZIP codes from New York having start date between January 2014 and April 2018. The dataset contains information about (not exhaustive list):
# 
# * Project start and completion dates
# * Project location
# * Construction type (new construction or preservation)
# * Number of units (rental, low income, medium income, other income, ownership, number of bedrooms, etc.)
# 
# We can use the information from the last category in order to add an new socio-economic dimension to the school explorer dataset. We should mention that some projects are confidential and their details are unlisted. We will exclude these projects from our statistics. Let's see 5 rows from the dataset itself:

# In[ ]:


ny_housing = pd.read_csv('../input/housing-new-york-units/housing-new-york-units-by-building.csv', parse_dates = ['Project Start Date', 'Project Completion Date'])


# In[ ]:


ny_housing[(ny_housing.Postcode.notna())].head()


# As we can see, detailed information is provided on the construction location as well as the type of construction (preservation / new building) and the type of units. We will group the constructions in each ZIP code by their type and sum the:
# 
# * number of total units
# * number of rental units
# * number of low income units
# * number of very low income units
# 
# We will join the above-defined pivots to the school explorer dataset using the ZIP code as join key.

# In[ ]:


ny_housing_by_zip_unit_type = pd.pivot_table(data = ny_housing[(ny_housing.Postcode.notna())],
                                             index = 'Postcode',
                                             values = ['Very Low Income Units', 'Low Income Units',
                                                       'Counted Rental Units', 'Counted Homeownership Units'],
                                             aggfunc = 'sum',
                                             fill_value = 0)

ny_housing_by_zip_constr_type = pd.pivot_table(data = ny_housing[ny_housing.Postcode.notna()],
                                               index = 'Postcode',
                                               columns = 'Reporting Construction Type',
                                               values = 'Total Units',
                                               aggfunc = 'sum',
                                               fill_value = 0)
ny_housing_by_zip_constr_type['total_units'] = ny_housing_by_zip_constr_type['New Construction'] + ny_housing_by_zip_constr_type['Preservation']

school_explorer = school_explorer.merge(ny_housing_by_zip_unit_type, how = 'left', left_on = 'Zip', right_index = True)
school_explorer = school_explorer.merge(ny_housing_by_zip_constr_type, how = 'left', left_on = 'Zip', right_index = True)


# We will infer the missing values following the previously used imputation strategy (median value for all unknown ZIP codes). We will then engineer the following features per ZIP code:
# 
# * Percentage new construction
# * Percentage rental units
# * Percentage low income units
# * Percentage very low income units

# In[ ]:


columns_to_impute = ['Counted Homeownership Units', 'Counted Rental Units',
                     'Low Income Units', 'Very Low Income Units',
                     'New Construction', 'Preservation', 'total_units']

imputed_columns = pd.DataFrame(imp.fit_transform(school_explorer[columns_to_impute]), columns = columns_to_impute)

school_explorer.drop(columns_to_impute, axis = 1, inplace = True)
school_explorer = school_explorer.join(imputed_columns)


# In[ ]:


school_explorer['pct_new_construction'] = school_explorer['New Construction'] / school_explorer['total_units']
school_explorer['pct_rental_units'] = school_explorer['Counted Rental Units'] / school_explorer['total_units']
school_explorer['pct_low_income_units'] = school_explorer['Low Income Units'] / school_explorer['total_units']
school_explorer['pct_very_low_income_units'] = school_explorer['Very Low Income Units'] / school_explorer['total_units']


# As before, we will inspect the correlation of our new housing features to the rest of the school explorer dataset:

# In[ ]:


housing_ind = ['pct_new_construction', 'pct_rental_units', 'pct_low_income_units', 'pct_very_low_income_units']


# In[ ]:


school_explorer_corr = school_explorer[target_ind +
                                       housing_ind +
                                       income_ind +
                                       crime_ind +
                                       school_explorer_ind].corr()

fig = plt.figure(figsize = (24, 8))
plt.subplots_adjust(wspace = 0.9)

for c, i in zip(housing_ind, range(1, len(housing_ind) + 1)):
    ax = fig.add_subplot(1, 4, i)
    to_plot = school_explorer_corr.loc[c, :].sort_values(ascending = False).drop(housing_ind)
    x_ticks = range(len(to_plot.index))
    ax.set_yticks(x_ticks)
    ax.set_yticklabels(to_plot.index, rotation = 0)
    ax.set_title('Correlation Between %s \n And The Socio-Academic Features \n In The School Explorer Dataset' % c)
    ax.barh(x_ticks, to_plot)

#for f in housing_features:
#    display(school_explorer_corr.sort_values(by = f, ascending = False)[[f]].transpose().iloc[:, 1:])


# There are three features that really stand out (from the four initially selected) in terms of their correlation magnitude: the percentage of rental units, the percentage of low income units and the percentage of very low income units.
# 
# The percentage of rental units feature has:
# 
# * Correlation with the: economic need index, diversity percentage, percentage of felonies and the percent of students chronically absent
# * Inverse correlation with the: academic proficiencies and the mean income of the area
# 
# The percentage of low income units feature has:
# 
# * Correlation with the: diversity percentage and the percentage of felonies
# * Inverse correlation with the academic proficiencies
# 
# The percentage of very low income units feature has:
# 
# * Correlation with the economic need index
# * Inverse correlation with the mean income of the area
# 
# The next dataset we will look into is the school directory dataset.

# ### School Directory Dataset
# The New York School Directory dataset contains 437 school prospecti structured in tabular format. Since this dataset contains information about the available bus and subway lines as means of transport to each school, an inference can be made on the transportation needs of the neighborhoods hosting each school. Let's import the dataset and look at 5 entries:

# In[ ]:


highschool_dir = pd.read_csv('../input/nyc-high-school-directory/2016-doe-high-school-directory.csv')

#rename school column (to match the name of the corresponding column in the school explorer results dataset)
highschool_dir.rename(columns = {'school_name' : 'school'}, inplace = True)

#convert school name to lower caps string (used when joining other data sources)
highschool_dir['school'] = highschool_dir['school'].str.lower()


# In[ ]:


highschool_dir.head()


# As we can see, the available bus and subway lines are listed in a common format (comma-separated for busses and semicolon-separated for subway lines). This means that we can split the strings for each school by their corresponding separators and count the number of busses and subway lines. This will give us a measure of how well a given school is connected to the city's infrastructure. Connectedness is supposedly a measure of social integration.
# 
# We will derive the following new features:
# 
# * Number of busses routed to each school: `num bus`
# * Number of subway lines routed to each school: `num subway`
# 
# We will also convert the `shared space` 'Yes' / 'No' strings to a 1 / 0 flag:

# In[ ]:


#convert the shared space string to flag
highschool_dir['shared_space'] = highschool_dir['shared_space'].map({'Yes' : 1, 'No' : 0})

#derive number of transport methods for every school
highschool_dir['num_bus'] = highschool_dir['bus'].fillna('').apply(lambda x : len(x.split(',')))
highschool_dir['num_subway'] = highschool_dir['subway'].fillna('').apply(lambda x : len(x.split(';')))


# We are now ready to join the three new features to the school explorer dataset using the school name as the join key. We will then impute the missing values for the rest of the schools using the median imputation strategy.

# In[ ]:


school_explorer = school_explorer.merge(highschool_dir[['school', 'borough','shared_space', 'num_bus', 'num_subway']], how = 'left', on = 'school')


# In[ ]:


columns_to_impute = ['shared_space', 'num_bus', 'num_subway']

imputed_columns = pd.DataFrame(imp.fit_transform(school_explorer[columns_to_impute]), columns = columns_to_impute)

school_explorer.drop(columns_to_impute, axis = 1, inplace = True)
school_explorer = school_explorer.join(imputed_columns)


# In[ ]:


school_dir_ind = ['shared_space', 'num_bus', 'num_subway']


# Let's now inspect the correlation between the features from the New York Highschool Directory and the rest of the selected features from the school explorer dataset:

# In[ ]:


school_explorer_corr = school_explorer[target_ind +
                                       school_dir_ind +
                                       housing_ind +
                                       income_ind +
                                       crime_ind +
                                       school_explorer_ind].corr()

fig = plt.figure(figsize = (24, 10))
plt.subplots_adjust(wspace = 0.9)

for c, i in zip(school_dir_ind, range(1, len(school_dir_ind) + 1)):
    ax = fig.add_subplot(1, 4, i)
    to_plot = school_explorer_corr.loc[c, :].sort_values(ascending = False).drop(school_dir_ind)
    x_ticks = range(len(to_plot.index))
    ax.set_yticks(x_ticks)
    ax.set_yticklabels(to_plot.index, rotation = 0)
    ax.set_title('Correlation Between %s \n And The Socio-Academic Features \n In The School Explorer Dataset' % c)
    ax.barh(x_ticks, to_plot)


#for f in school_dir_features:
#    display(school_explorer_corr.sort_values(by = f, ascending = False)[[f]].transpose().iloc[:, 1:])


# We can observe:
# 
# * Modest inverse correlation between the shared space feature and the academic proficiencies.
# * Modest correlation between the shared space and the strong family-community ties features.
# * Modest correlation between the number of subway lines and the strong family-community ties features.
# 
# It seems like the transportation features explain one of the custom measures (strong family-community ties). This could mean that it actually contains some information about the transport infrastructure of each neighborhood. In any case, we will keep the features for now and see their ranking in the modeling phase.

# ## Modeling
# 
# Since the school explorer dataset contains only 1200 rows, it would be preferable to recycle the data than to set aside valuable rows only for validation or only for testing. To this end, we will use k-fold cross validation. This is a technique where the data is split into k number of groups and the modeling process in done k number of times. The modeling process is repeated k number of times and every time a different group is used as the validation set. The final model is an average of all k folds.
# 
# We will use several different candidate models, pass them through a k-fold cross validation, evaluate their results through the selected evaluation metrics and compare their performance. We will then select the best model and use it to predict the Math proficiency of each school.
# 
# Since the prediction is based on socio-economic indicators, the inverse result of the normalized prediction will be an indicator of the school's socio-economic needs. Let's explain this concept in greater detail. The purpose of PASSNYC is to increase the proportion of students that take the SHSAT out of all registered students. Since this proportion is strongly linked to the average academic proficiency of each school, the actual goal is then to increase that proficiency. PASSNYC has stated that their work consists of providing outreach services to schools, such as [[4]](http://www.passnyc.org/our-partners 'PASSNYC's webpage'):
# 
# * Test prep and tutoring
# * Afterschool programs and activities
# * Resources for parents
# * Community groups
# 
# These services are indeed aimed at increasing the academic proficiency of the schools. Since we discovered that the academic proficiencies are actually linked to socio-economic issues, we can attempt to predict the Math or ELA proficiency of a given school based on such factors, namely the:
# 
# * Mean income of the school's neighborhood
# * Percentage of rental units in the school's neighborhood
# * Percentage of felonies on the school's neighborhood
# * Transportation density of each school's surrounding area
# 
# We can also use the custom measures (e.g. supportive environment, rigorous instruction), the academic measures (e.g. chronically absent students percentage) and the social measures (e.g. ethnic diversity) already provided by PASSNYC. A model built on all of those features will predict the academic proficiency of school based on socio-economic input. If we norm this prediction to 1 (the values of the regressor will range from 0 to 1) and invert it (take 1 - predicted value), we will essentially get a socio-economic need in order to obtain the maximum academic proficiency. This in turn will allow us to split the schools by their socio-economic needs and assign them a recommended action. Such actions could be:
# 
# | Socio-Economic Need Index Range | Color Code | Recommended Action  |
# |:-------------------- |:-------------- |:----------------------------------------- |
# | 0 - 20%                | Blue             | No outreach services are necessary. |
# | 20% - 40%           | Green          | No outreach services are necessary , but school should be monitored. |
# | 40% - 60%           | Yellow         | The school should be monitored and the need for outreach services should be investigated further. |
# | 60% - 80%           | Orange        | The school needs assistance. One type of outreach service is recommended (e.g. Test prep and tutoring). |
# | 80% - 100%         | Red              | The school needs assistance. Two or more types of outreach service are recommended (e.g. Test prep and tutoring and Afterschool programs and activities). |

# We will now scale and normalize our selected predictive features as well as our target (the Average Math Proficiency). We will do that because the target variable is positively-skewed (as seen on the figure below):

# In[ ]:


plt.title('Distribution Of The Average Math Proficiency')
sns.distplot(school_explorer[school_explorer['Average Math Proficiency'].notna()]['Average Math Proficiency'])


# In[ ]:


pred_cols = school_dir_ind + housing_ind + income_ind + crime_ind + school_explorer_ind

target_col1 = 'Average Math Proficiency'
target_col2 = 'Average ELA Proficiency'


# In[ ]:


scaler = MinMaxScaler((0, 1))

school_explorer = school_explorer[school_explorer[target_col1].notna()]

final_data = school_explorer[[target_col1] + pred_cols]

final_data = pd.DataFrame(scaler.fit_transform(final_data), columns = final_data.columns)
final_data = pd.DataFrame(normalize(final_data), columns = final_data.columns)

final_data.head()


# Let's now see a brief statistical description of our final dataset:

# In[ ]:


final_data.describe()


# We will now:
# 
# * Define our model candidates
# * Split the data into a training and testing sets
# * Perform k-fold cross validation on the training set only
# * Display the cross validation results for each model

# In[ ]:


models = {'ada_boost' : AdaBoostRegressor(),
          'linear_regression' : LinearRegression(),
          'lgbm' : LGBMRegressor(),
          'sgd_regressor' : SGDRegressor(),
          'decision_tree' : tree.DecisionTreeRegressor(),
          'random_forest' : RandomForestRegressor(),
          'dummy_0' : DummyRegressor(strategy = 'constant', constant = 0.),
          'dummy_0.5' : DummyRegressor(strategy = 'constant', constant = 0.5),
          'dummy_1' : DummyRegressor(strategy = 'constant', constant = 1.)}


# In[ ]:


metrics = {'mean_squared_error' : mean_squared_error,
           'r2_score' : r2_score,
           'mean_abs_error' : mean_absolute_error,
           'explained_variance_score' : explained_variance_score}


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(final_data[pred_cols],
                                                    final_data[target_col1],
                                                    test_size = 0.25,
                                                    random_state = 42)


# In[ ]:


n_folds = 3

scoring = ['neg_mean_absolute_error',
           'neg_mean_squared_error',
           'neg_median_absolute_error',
           'r2',
           'completeness_score',
           'explained_variance']

model_eval = defaultdict(list)
score_vis = defaultdict(list)

for m_name, reg in models.items():
    scores = cross_validate(reg, X_train, y_train, scoring = scoring, cv = n_folds, return_train_score = True)
    
    for k, v in scores.items():
        model_eval[k].append(v.mean())
        score_vis[k].extend(v)
    
    score_vis['model'].extend([m_name] * 3)

model_eval_res = pd.DataFrame(model_eval, index = models.keys()).transpose()
score_vis_res = pd.DataFrame(score_vis)

model_eval_res.style.format('{:.3f}')


# As we can see from the validation results table, the dummy models perform way worse than the rest of the models. The best performing models in terms of R2 score, mean absolute error and mean squared error are:
# 
# 1. LightGBM regressor
# 2. Linear regression
# 3. AdaBoost regressor
# 
# However, in terms of training time, the linear regression performs 3.5 times better than the AdaBoost regressor and 25 times better than the LightGBM regressor. Let's see that on the graphics below:

# In[ ]:


fig = plt.figure(figsize = (20, 8))

to_plot = score_vis_res[~score_vis_res.model.isin(['dummy_0', 'dummy_0.5', 'dummy_1'])]

ax1 = fig.add_subplot(1, 2, 1)
ax1.set_title('R2 Score For All Non-Dummy Models')
sns.boxplot(y = 'test_r2',
            x = 'model',
            data = to_plot,
            color = 'b')

ax2 = fig.add_subplot(1, 2, 2)
ax2.set_title('Fit Time For All Models')
sns.boxplot(y = 'fit_time',
            x = 'model',
            data = to_plot,
            color = 'b')


# A good compromise between R2 score and fit time is the linear regression. We will select it as a final model and perform hyperparameter tuning. In this exercise we will consider a range of learning rates from 0.0001 to 100. We will then print out the best learning rate.

# In[ ]:


reg = LinearRegression()

if isinstance(reg, SGDRegressor):
    hyperparam = {'average' : [True, False],
                  'alpha' : np.geomspace(1e-4, 1e2, num = 7)}
if isinstance(reg, AdaBoostRegressor):
    hyperparam = {'learning_rate' : np.geomspace(1e-4, 1e2, num = 7)}
if isinstance(reg, LinearRegression):
    hyperparam = {'normalize' : [True, False]}
if isinstance(reg, (RandomForestRegressor, LGBMRegressor)):
    hyperparam = {'max_depth' : [10, 50, 100, 500, 1000]}

final_reg = GridSearchCV(reg,
                         hyperparam,
                         scoring = scoring,
                         refit = 'r2')

final_reg.fit(X_train, y_train)


# In[ ]:


best_model = final_reg.best_estimator_
final_reg.best_params_


# We will now print out the performance of our final, tuned model on our selected [Evaluation Metrics](#Evaluation-Metrics), for the testing set:

# In[ ]:


final_eval = defaultdict(list)

for metric, scorer in metrics.items():
    for set_type, set_data in zip(('train', 'test'), ([y_train, X_train], [y_test, X_test])):
        final_eval['metric'].append(metric + '_' + set_type) 
        final_eval['value'].append(scorer(set_data[0], best_model.predict(set_data[1])))

final_eval_res = pd.DataFrame(final_eval)
final_eval_res


# We will now display the feature importances of our final model:

# In[ ]:


#n_best = 15

if isinstance(best_model, (SGDRegressor, LinearRegression)):
    feature_importances = pd.Series(best_model.coef_, index = X_train.columns)
else:
    feature_importances = pd.Series(best_model.feature_importances_, index = X_train.columns)

feature_importances.sort_values(ascending = False)#[:n_best]


# In[ ]:


to_plot = feature_importances.sort_values(ascending = False)#[:n_best]

if isinstance(best_model, (SGDRegressor, LinearRegression)):
    plt.title('Feature Coefficients')
    plt.xlabel('feature coefficient')
else:
    plt.title('Most Important Features')
    plt.xlabel('feature importance')

y_ticks = range(len(to_plot))
plt.yticks(y_ticks, to_plot.index)
plt.barh(y_ticks, to_plot)


# ## Socio-Economic Needs Visualization
# 
# We will start this part by deriving the socio-economic need label. As we mentioned previously, it is basically a split in 5 equal parts of the socio-economic need index.

# In[ ]:


school_explorer['scioecon_need'] = 1 - best_model.predict(final_data[pred_cols])
school_explorer['scioecon_need_label'] = pd.cut(x = school_explorer.scioecon_need,
                                                bins = 5,
                                                labels = ['blue', 'green', 'yellow', 'orange', 'red'])

#[-1, 0.40, 0.55, 0.7, 0.85, 1]
#[-1, 0.20, 0.55, 0.8, 0.9, 1]


# Now, it's time to individually test, sense-check and verify our model's predictions. We will do that by taking three entries from each label and checking the results against the results of [niche.com](https://www.niche.com/?ref=k12 'External website') - a website that provides ratings for thousands of schools in the USA.
# 
# We will now display 3 examples of schools for each label:

# In[ ]:


columns_to_plot = ['school', 'scioecon_need_label']
to_plot = pd.DataFrame(columns = columns_to_plot)
for l in ['red', 'orange', 'yellow', 'green', 'blue']:
    to_plot = to_plot.append(school_explorer[school_explorer['scioecon_need_label'] == l][columns_to_plot].head(3))
to_plot


# Here is a comparison between the results from the socio-economic needs index label and the ratings by Niche for the examples above:
# 
# | School                                     | Socio-Economic Code  | Overall Niche Grade | Niche Academics Grade | Niche Diversity Grade | Niche Teachers Grade |
# |:--------------------------------- |:--------------------------- |:------------ ------------ |:------------- |:------------ |:------------ |
# | orchard collegiate academy | red                                  | Not found                    | Not found  | Not found | Not found  |
# | technology, arts, and sciences studio | red                  | C+                                | C                 | A-               | B                 |
# | p.s. 149 sojourner truth          | red                                 | C+                                | C                 | B+              | C+              |
# | p.s. 15 roberto clemente       | orange                            | C+                                | C                 | B                | B                 |
# | p.s. 19 asher levy                    | orange                           | A-                                 | B+               | A-               | A                 |
# | p.s. 20 anna silver                  | orange                            | B                                  | B-                | A-               | A-               |
# | the children's workshop school | yellow                       | A-                                 | B+               | A                 | A                |
# | neighborhood school              | yellow                            | B                                  | B-                | A                 | A-              |
# | east side community school  | yellow                            | B-                                 | C                 | A-                | A-              |
# | p.s. 110 florence nightingale  | green                             | A-                                 | A-                | A                 | A                |
# | p.s. 184m shuang wen           | green                              | A                                  | A                  | B                | A                |
# | the east village community school | green                    | A                                  | A-                | A-                | A+             |
# | new explorations into science, technology and ... | blue | A+                            | A+                 | A                | A               |
# | p.s. 3 charrette school          | blue                                 | A                                   | A                  | B                 | A+             |                             
# | p.s. 6 lillie d. blake                 | blue                                 | A                                  | A                   | B-               | A+             |

# Let's summarize the results of the comparison:
# 
# * For the red label, the overall Niche grades are only C+.
# * For the orange label, the overall Niche grades are: one C+, one A- and one B.
# * For the yellow label, the overall Niche grades are: one A-, one B and one B-.
# * For the green label, the overall Niche grades are: one A- and two As.
# * For the blue label, the overall Niche grades are: one A+ and two As.
# 
# As we can see from the table above, the labels match well the independent results from [niche.com](https://www.niche.com/?ref=k12 'External website'). We will now plot the labels on a map of New York and see the distribution by neighborhood:

# In[ ]:


m = folium.Map(location = [40.75, -73.85], tiles = 'cartodbpositron', zoom_start = 11.25)

to_plot = school_explorer

for i in range(0, len(to_plot)):
    c = to_plot.iloc[i].scioecon_need_label
    
    folium.Circle(location = [to_plot.iloc[i].Latitude ,
                              to_plot.iloc[i].Longitude],
                  radius = 150,
                  color = c,
                  fill = True,
                  stroke = True,
                  fillOpacity = 0.2
   ).add_to(m)

mapWidth, mapHeight = (400, 500)
m


# As we can see from the map above, the results make sense geographically and economically - some of the richest areas of NYC (e.g. Upper Eastside) are saturated with green and blue schools. On the other hand, some of the more financially-troubled areas such as. known industrial areas like Ridgewood, Bushwick or known areas with high crime rates like Claremont [[5]](https://www.trulia.com/real_estate/Claremont-Bronx/5065/crime/ 'External website') have a high concentration of orange and red dots.
# 
# There is one final verification step to perform - to join the socio-economic needs index to the District 5 SHSAT registration data. In the initial stages of this analysis, we split the `took pct registered` variable into 3 intervals (low, medium and high). We now want to see if a high socio-economic need score corresponds to low percentage of students taking the SHSAT (as we initially set out to do). Let's see the dependency for our original selection of schools:

# In[ ]:


shsat_res = shsat_res.merge(school_explorer[['school', 'scioecon_need', 'scioecon_need_label']], how = 'left', left_on = 'school', right_on = 'school')


# In[ ]:


to_plot = shsat_res[(shsat_res.year == 2016) & (shsat_res.grade == 8)]

plt.title('Economic Need Index Distribution Per Take As Percentage Of Registered Label')
sns.boxplot(y = 'scioecon_need', x = 'took_pct_registered_label', data = to_plot, color = 'b')


# As we can see, schools where a small percentage of students who register for the SHSAT actually sit the test have a very high socio-economic need score, while the opposite is true for schools where the `taken pct registered` ratio is low.
# 
# Having verified the socio-economic need index, we can now save the results in the output of this notebook.
# 
# We will now start investigating and quantifying the challenges that students face in taking the SHSAT. We will first explore the dependency between socio-economic needs and several social indicators:

# In[ ]:


#export the results
school_explorer.to_csv('school_explorer_seni_labels.csv')


# In[ ]:


fig = plt.figure(figsize = (20, 22))

cols_to_plot = ['diversity_prc', 'Economic Need Index', 'income_mean', 'pct_low_income_units', 'pct_FELONY', 'income_stdev']

for c, i in zip(cols_to_plot, range(1, len(cols_to_plot) + 1) ):
    ax = fig.add_subplot(3, 2, i)
    ax.set_title('Distribution Of %s By Socio-Economic Label' % c)
    sns.boxplot(y = c, x = 'scioecon_need_label', data = school_explorer, color = 'b')


# From the graphics above we can clearly identify some trends, namely:
# 
# * More diverse schools are in higher need of outreach by PASSNYC. A substantial amount of those schools is located in diverse and simultaneously - low income neighborhoods. We can verify that by looking at the `Economic Need Index` variable broken down by socio-economic need label.
# * The economic need index of each school is a clear indicator of how confident its students will be in sitting the SHSAT.
# * The academic outcomes of schools are related to the income in the area where said schools are located. Low income communities are in need of more resources in order to achieve better academic outcomes. This is also evident from the distribution of the low income housing percentage by socio-economic need label.
# * Communities require security in order to produce better academic outcomes as evident from the distribution of the percentage of felonies by socio-economic need label.
# 
# However, our story has a silver lining. It is that better academic outcomes and consequently - higher amount of students taking the SHSAT can be achieved by the social outreach services that PASSNYC provides:

# In[ ]:


fig = plt.figure(figsize = (20, 18))

cols_to_plot = ['Percent of Students Chronically Absent', 'Supportive Environment %',
                'Rigorous Instruction %', 'Strong Family-Community Ties %']

for c, i in zip(cols_to_plot, range(1, len(cols_to_plot) + 1) ):
    ax = fig.add_subplot(2, 2, i)
    ax.set_title('Distribution Of %s By Socio-Economic Label' % c)
    sns.boxplot(y = c, x = 'scioecon_need_label', data = school_explorer, color = 'b')


# As we can see:
# 
# * Reducing the number of chronically-absent students (e.g. by providing 'resources for parents') firmly places a school in the green and blue labels.
# * Improving the school environment (as seen in the `Supportive Environment` distribution) leads to better academic outcomes. This is probably within the reach of the 'afterschool programs and activities' provided by PASSNYC.
# * Increasing the intensity of school instruction (as seen in the `Rigorous Instruction` distribution) is linked to better academic outcomes. The 'test prep and tutoring' program by PASSNYC has the potential to address this challenge.
# 
# # Conclusion
# 
# We started our analysis with a simple issue: 'how can we increase the number of students taking the SHSAT'. In order to answer this question we had to find the relationship between the number of students taking the SHSAT, the number of students registering for the SHSAT and the characteristics of each school. We saw that high academic proficiency correlates strongly with the proportion of the students sitting the SHSAT. Therefore, we decided to find various socio-economic factors that can predict academic proficiency.
# 
# We imported four new datasets in order to enrich PASSNYC's school explorer with additional features. The new, public datasets that we imported were:
# 
# * [NYC Crime Data](https://www.kaggle.com/adamschroeder/crimes-new-york-city#Population_by_Borough_NYC.csv 'Kaggle')
# * [US Household Income Data](https://www.kaggle.com/goldenoakresearch/us-household-income-stats-geo-locations 'Kaggle')
# * [NY Housing Data](https://www.kaggle.com/new-york-city/housing-new-york-units 'Kaggle')
# * [NYC Highschool Directory](https://www.kaggle.com/new-york-city/nyc-high-school-directory#2016-doe-high-school-directory.csv 'Kaggle')
# 
# We aggregated various features from the above-mentioned datasets either on ZIP code or individual school level. We imputed the rows where there was no match and enriched the school explorer dataset. We investigated the correlation between our new features and our target variables (the academic proficiencies). We then scaled and normalized the data.
# 
# We derived a model that predicts normalized and scaled academic outcomes based on various socio-economic factors. We called the inverse of this prediction: 'socio-economic index'. We split the socio-economic index into 5 equal intervals with recommended actions for each interval. We named these intervals: 'socio-economic labels'. We then validated the model's results against:
# 
# * An independent source of school ranking (the [niche.com](https://www.niche.com/?ref=k12 'External website') website).
# * The geographical distribution of the socio-economic labels.
# * The dataset on district 5 (Central Harlem), provided by the Department of Education.
# 
# Finally we investigated the relationship between the socioeconomic labels and multiple features from the enriched school explorer dataset. In conclusion, we found out that:
# 
# * Socio-economic factors have a high influence on the academic outcomes of schools. This is observed in the following trends:
#     * More diverse schools are in higher need of outreach by PASSNYC. A substantial amount of those schools is located in diverse and simultaneously - low income neighborhoods. We can verify that by looking at the `Economic Need Index` variable broken down by socio-economic need label.
#     * The economic need index of each school is a clear indicator of how confident its students will be in sitting the SHSAT.
#     * The academic outcomes of schools are related to the income in the area where said schools are located. Low income communities are in need of more resources in order to achieve better academic outcomes. This is also evident from the distribution of the low income housing percentage by socio-economic need label.
#     * Communities require security in order to produce better academic outcomes as evident from the distribution of the percentage of felonies by socio-economic need label.
# * PASSNYC's outreach has the potential to positively-influence New York's school and improving their academic outcome. This can be achieved by:
#     * Reducing the number of chronically-absent students (e.g. by providing 'resources for parents') firmly places a school in the green and blue labels.
#     * Improving the school environment (as seen in the `Supportive Environment` distribution) leads to better academic outcomes. This is probably within the reach of the 'afterschool programs and activities' provided by PASSNYC.
#     * Increasing the intensity of school instruction (as seen in the `Rigorous Instruction` distribution) is linked to better academic outcomes. The 'test prep and tutoring' program by PASSNYC has the potential to address this challenge.
# 
# # Improvement Suggestions
# 
# Given more time and resources, our work can be improved both by adding new datasets to our enriched school explorer and by deriving new features from our existing datasets. Some examples of new features are:
# 
# * Create a function for calculating geographical radius - this will allow for even more granular approach (e.g. see crimes, houses, etc. within 1 mile radius of a given school).
# * Use natural language processing on the school prospecti in order to derive academic proficiencies. The highschool directory dataset had multiple text columns. The texts from the prospecti can be processed for certain phrases that could be predictors of the school's actual academic proficiency.
# 
# Some examples of new datasets that could be utilized are:
# 
# * [NYC Wifi Hotspot Locations](https://www.kaggle.com/new-york-city/nyc-wi-fi-hotspot-locations 'Kaggle'). Wifi connectivity could be a good measure of social integration. This could set apart schools in neighborhood with poor internet connection and could indicate where additional resources are necessary.
# * [NYC Payroll Data](https://www.kaggle.com/new-york-city/ny-citywide-payroll-data-fiscal-year 'Kaggle'). Similarly to the US household income data, this could give us a better estimate of the incomes in NYC.
# 
# Thank you for your attention!

# In[ ]:




