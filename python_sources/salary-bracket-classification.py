#!/usr/bin/env python
# coding: utf-8

# # Which (USA) developers earn the most?

# * Objectives: Separate the salary into tree groups, test different models and find out what answers are the most relevant.
# * Kaggle link: https://www.kaggle.com/stackoverflow/stack-overflow-2018-developer-survey
# 
# Stackoverflow is the biggest hub for developers in the world with about 50 million visitors a month. It is used on a daily by professionals and hobysts alike, mainly as a tool (a forum, if you will) for discussing programming problems. 
# 
# Ever since 2015, Stackoverflow has hosted a developer survey to learn more about its user-base, with questions ranging from programming related questions, salary and ethics. The new installment of the survey reached about 100.000 people, almost 4 times as much as the first one held 3 years ago.
# 
# There's a wide range of information available, and no obligation to answer all questions. The main focus of this study is to find out if there's a direct correlation between how much someone earns and several programming and career related topics. In order to have a meaningful analysis, questions that have little relation with salary, or that have too many missing values, will be removed.
# 
# Since people from all over the world respond to this survey and it's impossible to have a good global salary analysis in a single model (without build extremely complex ensembles) we will limit our analysis to USA residents.

# # Index

# This kernel is organized in the following way:
# 
# 1. [Data pre-processing](#1.-Data-pre-processing)
# 2. [Data analysis and feature selection](#2.-Data-analysis-and-feature-selection)
# 3. [Feature engineering](#3.-Feature-engineering)
# 4. [Model building](#4.-Model-building)
# 5. [Statistically testing our models](#5.-Statistically-testing-our-models)

# # 1. Data pre-processing

# We start off by importing all libraries we're gonna use (remove unused imports later):

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import model_selection

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from statsmodels.stats.outliers_influence import variance_inflation_factor   
from scipy.stats import friedmanchisquare
from scipy.stats import wilcoxon


# ### Importing the data

# The next step is to import the csv using the pandas library, and take a look at quick peek at the data: 

# In[ ]:


pd.options.display.max_columns = None
warnings.filterwarnings('ignore')
dataset = pd.read_csv('../input/survey_results_public.csv')
dataset.head()


# In[ ]:


# number of entries
print("Number of entries: " + str(len(dataset.index)))


# ### Checking for empty values

# With that done, let's take a quick look at what columns have empty values:

# In[ ]:


pd.options.display.max_rows = None
dataset.isnull().sum()


# Only the first three columns have no empty values: respondent, hobby and open source. Respondent is obviously needed and probably registered automatically by StackOverflow. As for the other values, it's pretty interesting to note that only two fields have no empty replies: hobby and open source. This probably says more about people than about the questions itself, that is, lazyness probably took over when people noticed the size of the survey. Let's see what the top offenders are:

# In[ ]:


all_data_na = (dataset.isnull().sum() / len(dataset)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
missing_data.head(20)


# It's interesting to note that the general trend does not suggest that people got lazy midway through the survey, as some of the questions at the end had the similar number of empty replies as some at the beginning. The biggest offenders with low reply rate are:
# 1. TimeAfterBootcamp;
# 2. MilitaryUS;
# 3. HackatonReasons;
# 4. EngonomicDevices;
# 5. AdBlockReasons;
# 6. StackOverflowJobRecommend.
# 
# Half of these can be attributed to being dependent on previous questions, which have a high chance of being empty (either due to previous question being skipped or reponded negativelly): time after bootcamp, hackaton reasons and adblock reasons. The reamining three are probably due to not being very popular questions.
# 
# Due to the ammount of empty replies, it's pretty clear we will have to remove some attributes (that is, some questions won't be used in our analysis):
# 1. The top 6 offenders: too many empty values;  
# 2. SurveyTooLong and SurveyTooEasy: unless there's an odd correlation between salary and people finding surveys easy, we can remove these questions;
# 3. All advertizing related questions, which is a survey in itself;
# 4. All currency/salary related questions, except by ConvertedSalary. We have too many options to say the same thing, having a single source of information for income should be enough. However, before removing this information, it will be used a filter to keep only entries that have similar earnings (pound, euros and dollars).
# 5. Hypothetical tools, which is a survey for what tools people would like to see on StackOverflow. Although it could be indirectly related to salary, as in, people who want certain tools are either happy or unhappy in their jobs, it's too specific for us to consider as part of a general study;
# 6. Job contact and email priorities: it tells a similar story to job assessment questions;
# 7. AI related questions, which should tell little about salary;
# 8. Questions with too many possible answers, such as framework, languages, platforms and so on, both on the worked with and desired next year categories;
# 9. Respondent number, which is only an identifier without any meaning for this study.
# 
# Some of the other attributes may seen uncorrelated to salary at first sight (such as if you consider being a member of stackoverflow community), but they may end up showing a relation in the form of "how much does community engagement helps you being successfull in your career, and thus earn more".

# In[ ]:


features = dataset[dataset.columns.difference(['Respondent','TimeAfterBootcamp', 'MilitaryUS', 'HackatonReasons', 'EngonomicDevices', 'AdBlockReasons', 'StackOverflowJobRecommend', 'SurveyTooLong', 'SurveyTooEasy', 'AdBlocker', 'AdBlockerDisable', 'AdBlockerReasons', 'AdsAgreeDisagree1', 'AdsAgreeDisagree2', 'AdsAgreeDisagree3', 'AdsActions', 'AdsPriorities1', 'AdsPriorities2', 'AdsPriorities3', 'AdsPriorities4', 'AdsPriorities5', 'AdsPriorities6', 'AdsPriorities7', 'HypotheticalTools1', 'HypotheticalTools2', 'HypotheticalTools3', 'HypotheticalTools4', 'HypotheticalTools5', 'CurrencySymbol', 'Salary', 'SalaryType'])]
features = features[features.columns.difference(['ErgonomicDevices', 'HackathonReasons', 'SurveyEasy', 'JobEmailPriorities1', 'JobEmailPriorities2', 'JobEmailPriorities3', 'JobEmailPriorities4', 'JobEmailPriorities5', 'JobEmailPriorities6', 'JobEmailPriorities7', 'JobContactPriorities1', 'JobContactPriorities2', 'JobContactPriorities3', 'JobContactPriorities4', 'JobContactPriorities5', 'AIDangerous', 'AIInteresting', 'AIResponsible', 'AIFuture', 'IDE', 'LanguageDesireNextYear', 'LanguageWorkedWith', 'PlatformDesireNextYear', 'PlatformWorkedWith', 'DatabaseDesireNextYear', 'DatabaseWorkedWith', 'FrameworkDesireNextYear', 'FrameworkWorkedWith', 'CommunicationTools', 'CheckInCode', 'VersionControl', 'UpdateCV',  'StackOverflowVisit', 'StackOverflowDevStory', 'StackOverflowHasAccount', 'StackOverflowParticipate', 'StackOverflowRecommend', 'StackOverflowJobs', 'StackOverflowJobsRecommend'])]
#features = dataset[dataset.columns.difference(['UpdateCV',  'StackOverflowVisit', 'StackOverflowDevStory', 'StackOverflowHasAccount', 'StackOverflowParticipate', 'StackOverflowRecommend', 'StackOverflowJobs', 'StackOverflowJobsRecommend'])]


# We should also remove features that have way too many options, as we will end-up with too many features. Let's see what are the worst offenders:

# In[ ]:


features.nunique()


# There are some pretty bad features out there: Methodology, SelfTaughtTypes, DevType and EducationTypes. It's tempting to not use country due to that much variation, however, it will likely play a big role on salary, so it can't be ignored. Converted salary does have a lot of possibilities, but that's because it's a user-inputed number. This will be corrected by creating salary ranges. As for the rest, they should be removed:

# In[ ]:


features = features[features.columns.difference(['Methodology', 'SelfTaughtTypes', 'DevType', 'EducationTypes'])]


# # 2. Data analysis and feature selection

# ## 2.1 Countries and currencies

# In this analysis it's important to see what country the respondent is from, as it greatly affect how much he earns. A software developer in Brazil, for example, earns about 4 times less than his counterpart in USA. Therefore, we lose a bit of meaning without doing any operations to address this. We have two possibilities:
#    1) Processing the currencies and converting the salary to the estimated value, if this person was a USA citzen.
#    2) Ignoring all non-dollar, euro or pound based countries (which usually pay better).
#    
# In order to decide what aproach to take, we need to take a look at the country and currencies distribution:

# In[ ]:


f, ax = plt.subplots(figsize=(30, 7))
plt.xticks(rotation='90')
sns.countplot(features['Country'])
plt.title('Country distribution on the Stackoverflow 2018 survey', fontsize=12)


# In[ ]:


f, ax = plt.subplots(figsize=(30, 7))
plt.xticks(rotation='45')
sns.countplot(features['Currency'])
plt.title('Currency  distribution on the Stackoverflow 2018 survey', fontsize=15)


# We can see that most of the respondents are either earning in dollars, followed by euros and pounds. Although all three currencies have similar values, it's better if we only keep developers from the USA in this study. The reason for this is that, in averagem a developer in the USA earns more than others: https://stackoverflow.blog/2017/09/19/much-developers-earn-find-stack-overflow-salary-calculator/, therefore, it seems like we can't mix this data without getting too much noise.

# In[ ]:


features = features[(features.Country == 'United States')]
features = features[features.columns.difference(['Currency', 'Country'])];


# We can now finally drop all empty rows:

# In[ ]:


features = features.dropna();
print("Number of entries: " + str(len(features.index)))


# At the end, we ended up retaining 8216 responses in full, which is more than enough to run a meaningful analysis. In the end, we kept the following attributes:

# In[ ]:


features.columns


# ## 2.2 Salary

# Lets take a look at how salary is distributed:

# In[ ]:


f, ax = plt.subplots(figsize=(14, 7))
plt.xticks(rotation='45')
sns.distplot(features['ConvertedSalary']);
plt.xlabel('Annual salary in US dollars', fontsize=15)


# We can see that salary is a left skewed normal distribution, with some outliers, as seen on the 1 and 2 million mark. These do not represent the vast majority of people that participated in this survey, so we should strip it out least it affect our models. Before doing that, let's take a look at the distribution:

# In[ ]:


features['ConvertedSalary'].describe()


# According to the payscale website (https://www.payscale.com/research/US/Job=Software_Developer/Salary), 90% of USA developers salary range from 46,797 to 107,228 U$. However, to go through the bigger picture, we can use 15000 as a baseline, as that is the minimum wage in USA (https://poverty.ucdavis.edu/faq/what-are-annual-earnings-full-time-minimum-wage-worker). 
# 
# We also need a cap, as the original dataset has salaries up to 2M, which are way out of the ordinary and should be treated separately on a different study. We'll cap it at 300k, as it seems to be what most successful developers can hope to achieve (https://www.networkworld.com/article/3167569/careers/13-tech-jobs-that-pay-200k-salaries.html) 

# In[ ]:


features = features[features['ConvertedSalary'] < 300000]


# In[ ]:


features = features[features['ConvertedSalary'] > 15000]


# With that in mind, our model can now only predict the average developer, which should be over 80% of the population.

# Lets check the new distribution and plot:

# In[ ]:


features['ConvertedSalary'].describe()


# In[ ]:


f, ax = plt.subplots(figsize=(14, 7))
plt.xticks(rotation='45')
sns.distplot(features['ConvertedSalary']);
plt.xlabel('Annual salary in US dollars', fontsize=15)


# There's little feature engineering to be done on this dataset: almost all questions were either multiple choice or yes/no, with no room for input error. The only special one is salary, which should be converted into salary ranges to better categorize people. We know that the lower 10% is at 49k, the top is at 102k and the medium is 69k dollars. That is, the majority should be somewhere in between 49k and 102k, which seems about right for developer salaries in most parts of USA, except by cities with high cost of living. 
# 
# However, we can see on our distribution above that most developers seems to be on the 50k to 150k range! With that in mind, we can't blindly follow the average USA salary statistics, it seems more appropriate to use the following salary brackets:
#  1. 15k to 60k
#  2. 60k to 120k
#  3. 120k to 300k
# 

# In[ ]:


first_bracket = "From 15k to 60k"
second_bracket =  "From 60k to 110k"
third_bracket = "From 110k to 300k"

features['SalaryRange'] = pd.cut(features['ConvertedSalary'], bins=[15000, 60000, 110000, 300000], labels=[first_bracket, second_bracket, third_bracket])
features = features[features.columns.difference(['ConvertedSalary'])];


# Let's take a look at the distribution:

# In[ ]:


f, ax = plt.subplots(figsize=(14, 7))
plt.xticks(rotation='45')
sns.countplot(features['SalaryRange'], order=[first_bracket, second_bracket, third_bracket])
plt.title('Salary distribution', fontsize=15)


# The distribution indicates that people who use StackOverflow and participate on its events (such as this survey) are, on average, much better paid.

# In[ ]:


features = features.dropna();
print("Number of entries: " + str(len(features.index)))


# Let's take a look at the distribution: 

# In[ ]:


features['SalaryRange'].describe()


# ## 2.3 Career satisfaction

# Lets take a look at the career satisfaction distribution:

# In[ ]:


f, ax = plt.subplots(figsize=(14, 7))
plt.xticks(rotation='45')
sns.countplot(features['CareerSatisfaction'], order=['Extremely dissatisfied', 'Slightly dissatisfied', 'Moderately dissatisfied', 'Neither satisfied nor dissatisfied', 'Slightly satisfied', 'Moderately satisfied', 'Extremely satisfied'])
plt.xlabel('Career satisfaction', fontsize=15)
plt.title('How people feel about their career', fontsize=15)


# Almost half the respondents are moderately satisfied with their careers. In second place we have people who are "extremely satisfied", closely followed by slightly satisfied. In general, we're seeing a big indication that people who took this survey are in general more satisfied than not with their careers. 

# Let's take a closer look at the distribution:

# In[ ]:


features['CareerSatisfaction'].describe()


# Now we can see how career satisfaction (from 0 to 6) compare to salary:

# In[ ]:


labelencoder = LabelEncoder()
features['carrer_label'] = labelencoder.fit_transform(features['CareerSatisfaction'])

f, ax = plt.subplots(figsize=(14, 7))
sns.boxplot(x="SalaryRange", y="carrer_label", data=features)
plt.xlabel('Salary bracket', fontsize=15)
plt.ylabel('Career satisfaction', fontsize=15)
plt.title('Career satisfaction across salary brackets', fontsize=15)

features = features[features.columns.difference(['carrer_label'])]


# "Does money bring career satisfaction" is the modern adaptation of "does money bring happiness", and the answer (at least to USA developers) seems to be no. That is understandable, as it can mean several things:
#  1. The developer is already burned out due to working too many years to get where he is;
#  2. Works for a company or in an industry which he does not care much about only to get a high pay-check;
#  3. There's little sense of going forward, since you probably already achieved most of your professional goals.

# ## 2.4 Job preferences

# This survey has many in-depth questions that explore how someone assess a potential job oportunity in the following categories:
# 1. The opportunity itself: industry, department you will be working with, overall benefits and so on;
# 2. Benefits: explores a little bit more in depth which benefits are the most interesting;
# 3. Contact preference: how do you prefer being contacted (telephone, email, etc.);
# 4. Information you would like to see on a job opportunity sent by email.
# 
# Only the first and second question groups were kept for this study, as the last too are too specific and are already covered to some extent  by other questions. 

# First, lets take a look at how salary relates to job opportunity assessment. There's one graphic per question, with the y axis being how the respondent rates the job aspected from 1 to 11 (the lowest, the most important) and the x axis being the salary bracket. The lenght of the vertical lines indicate the variance and the dot indicates the average rating for that particular group.

# In[ ]:


# Define a set of graphs, 3 by 5, usin the matplotlib library
f, axes = plt.subplots(4, 3, figsize=(25, 25), sharex=False, sharey=False)
#f.subplots_adjust(hspace=0.4)
plt.subplots_adjust(left=0.2, wspace=0.3, top=0.95)
plt.suptitle('Salary bracket versus importance given to job opportunities aspects', fontsize=16)
axes[-1, -1].axis('off')
axes[-1, -2].axis('off')

sns.pointplot(x="SalaryRange", y="AssessJob1", data=features, ax=axes[0,0])
axes[0,0].set(ylabel='Importance given to industry you will be working in')

sns.pointplot(x="SalaryRange", y="AssessJob2", data=features, ax=axes[0,1])
axes[0,1].set(ylabel='Importance given to financial performance of the company/organization')

sns.pointplot(x="SalaryRange", y="AssessJob3", data=features, ax=axes[0,2])
axes[0,2].set(ylabel='Importance given to deparment or team you will be working on')

sns.pointplot(x="SalaryRange", y="AssessJob4", data=features, ax=axes[1,0])
axes[1,0].set(ylabel='Importance given to technologies you will be working with')

sns.pointplot(x="SalaryRange", y="AssessJob5", data=features, ax=axes[1,1])
axes[1,1].set(ylabel='Importance given to compensation and benefits offered')

sns.pointplot(x="SalaryRange", y="AssessJob6", data=features, ax=axes[1,2])
axes[1,2].set(ylabel='Importance given to the company culture')

sns.pointplot(x="SalaryRange", y="AssessJob7", data=features, ax=axes[2,0])
axes[2,0].set(ylabel='Importance given to the opportunity to work from home/remotely')

sns.pointplot(x="SalaryRange", y="AssessJob8", data=features, ax=axes[2,1])
axes[2,1].set(ylabel='Importance given to opportunities for professioal development')

sns.pointplot(x="SalaryRange", y="AssessJob9", data=features, ax=axes[2,2])
axes[2,2].set(ylabel='Importance given to the diversity of the company or organization')

sns.pointplot(x="SalaryRange", y="AssessJob10", data=features, ax=axes[3,0])
axes[3,0].set(ylabel='Importance given to the impact of the product/software you will be working with has')


# The graphs above tell us the following story:
# 1. Without exception, people who earn the least had the biggest variance on how they rate the aspects of a job offer. This could indicate several things; for instance, it could mean that our bracket system is not perfect, or that people who earn less don't have a clear goal.
# 2. On a similar note, people who are in the middle salary bracket had the smallest variance on how they rate job opportunities.
# 3. Everyone seems to rate the importance given to the department/team they will be working with the same.
# 4. On average, people who earn more prioritize: financial performance of the company, compensation and benefits offered, diversity of the company and the impact of the product/software he works with has. 
# 5. On the other hand, people who are earn the least give more importance to opportunities for professional development, what industry they work with, company culture and technologies used.
# 
# In general, we can that people who earn more are not really worried anymore about career development; they want to earn well, have stability and work with products that have an impact on the world. On the other hand, people who ear less want to develop their careers, work with technologies and industries they like.
# 
# A few of this topics had too little variance to matter, so we're going to remove it: compensation and benefits, department or team will be working with and home-office.

# In[ ]:


features = features[features.columns.difference(['AssessJob3', 'AssessJob5', 'AssessJob7'])]


# With that in mind, we can now go more in depth and take a look at the benefits are rated:

# In[ ]:


# Define a set of graphs, 3 by 5, usin the matplotlib library
f, axes = plt.subplots(4, 3, figsize=(25, 25), sharex=False, sharey=False)
#f.subplots_adjust(hspace=0.4)
plt.subplots_adjust(left=0.2, wspace=0.3, top=0.95)
plt.suptitle('Salary brackets versus importance given to benefits', fontsize=16)
axes[-1, -1].axis('off')

sns.pointplot(x="SalaryRange", y="AssessBenefits1", data=features, ax=axes[0,0])
axes[0,0].set(ylabel='Importance given to salary/bonuses')

sns.pointplot(x="SalaryRange", y="AssessBenefits2", data=features, ax=axes[0,1])
axes[0,1].set(ylabel='Importance given to stock options/shares')

sns.pointplot(x="SalaryRange", y="AssessBenefits3", data=features, ax=axes[0,2])
axes[0,2].set(ylabel='Importance given to healthcare')

sns.pointplot(x="SalaryRange", y="AssessBenefits4", data=features, ax=axes[1,0])
axes[1,0].set(ylabel='Importance given to parental leave')

sns.pointplot(x="SalaryRange", y="AssessBenefits5", data=features, ax=axes[1,1])
axes[1,1].set(ylabel='Importance given to fitness or wellness benefits')

sns.pointplot(x="SalaryRange", y="AssessBenefits6", data=features, ax=axes[1,2])
axes[1,2].set(ylabel='Importance given to retirement or pension savings matching')

sns.pointplot(x="SalaryRange", y="AssessBenefits7", data=features, ax=axes[2,0])
axes[2,0].set(ylabel='Importance given to company provided meals or snacks')

sns.pointplot(x="SalaryRange", y="AssessBenefits8", data=features, ax=axes[2,1])
axes[2,1].set(ylabel='Importance given to computer/office equipment allowance')

sns.pointplot(x="SalaryRange", y="AssessBenefits9", data=features, ax=axes[2,2])
axes[2,2].set(ylabel='Importance given to childcare benefit')

sns.pointplot(x="SalaryRange", y="AssessBenefits10", data=features, ax=axes[3,0])
axes[3,0].set(ylabel='Importance given to transportation benefit')

sns.pointplot(x="SalaryRange", y="AssessBenefits11", data=features, ax=axes[3,1])
axes[3,1].set(ylabel='Importance given to conference or education budget')


# Here are the general trends for how people evaluate a job's benefits package:
# 
# 1. In a similar fashion to the job offer aspects analysis, people who earn the least have the biggest variance on how they rate the benefit package of a job offer. 
# 2. In general, people who earn more care more about having stock options/shares participation and bonuses more than people who earn less.
# 3. People who earn less give care more about education budget, fitness/wellbeing offered and equipment allowance than people who earn more.
# 4. Overall, salary/bonuses are by far the most important aspect of a job's benefits package, and childcare the least.
# 
# The other benefits are too similar across all brackets and can be removed: health care,  parental leave, pension savings matching, company provided meals/snacks,  childcare benefits and transportation.

# In[ ]:


features = features[features.columns.difference(['AssessBenefits3', 'AssessBenefits4', 'AssessBenefits6', 'AssessBenefits9', 'AssessBenefits10'])]


# # 3. Feature engineering

# First, let's split the output from the features:

# In[ ]:


output = features['SalaryRange']
features = features[features.columns.difference(['SalaryRange'])]


# Now we can separate the categorical variances from the numerical ones:

# In[ ]:


categorical = []
for col, value in features.iteritems():
    if value.dtype == 'object':
        categorical.append(col)

# Store the numerical columns in a list numerical
numerical = features.columns.difference(categorical)

print(categorical)


# To simplify things, we'll one hot encode all categorical variables and remove one for each to avoid the dummy variable trap:

# In[ ]:


# get the categorical dataframe
features_categorical = features[categorical]


# In[ ]:


# one hot encode it
features_categorical = pd.get_dummies(features_categorical, drop_first=True)


# In[ ]:


# get the numerical dataframe
features_numerical = features[numerical]


# In[ ]:


# concatenate the features
features = pd.concat([features_numerical, features_categorical], axis=1)


# And the salary is encoded:

# In[ ]:


labelencoder = LabelEncoder()
output = labelencoder.fit_transform(output)


# Remove collinear variables:

# In[ ]:


def calculate_vif_(X, thresh=100):
    cols = X.columns
    variables = np.arange(X.shape[1])
    dropped=True
    while dropped:
        dropped=False
        c = X[cols[variables]].values
        vif = [variance_inflation_factor(c, ix) for ix in np.arange(c.shape[1])]
    
        maxloc = vif.index(max(vif))
        if max(vif) > thresh:
            print('dropping \'' + X[cols[variables]].columns[maxloc] + '\' at index: ' + str(maxloc))
            variables = np.delete(variables, maxloc)
            dropped=True

    print('Remaining variables:')
    print(X.columns[variables])
    return X[cols[variables]]


# To save up on computing time, I'm not going to execute the method above. I'll simply drop the variables it indicated on the first time I ran it: 
# 
# > dropping 'WakeTime_Between 6:01 - 7:00 AM' at index: 190
# 
# > dropping 'OperatingSystem_Windows' at index: 108
# 
# > dropping 'SexualOrientation_Straight or heterosexual' at index: 156

# In[ ]:


features = features[features.columns.difference(['WakeTime_Between 6:01 - 7:00 AM','SexualOrientation_Straight or heterosexual','OperatingSystem_Windows'])]


# And lastly, we split it into train and test sets:

# In[ ]:


features_train, features_test, salary_train, salary_test = train_test_split(features, output, test_size = 0.3, random_state = 0)


# # 4. Model building

# With some preliminary feature selection and engineering being done, we can test a few models and see how they perform without further changes. 
# 
# At first, we will not do any cross-validation and just get a feel of what we can achieve with each model. The next chapter will wrap it up by cross-validating each model and comparing the results.
# 
# The following two methods will help us along the way:

# In[ ]:


def plot_confusion_matrix(cm, title):
    # building a graph to show the confusion matrix results
    cm_plot = pd.DataFrame(cm, index = [i for i in {first_bracket, second_bracket, third_bracket}],
                  columns = [i for i in {first_bracket, second_bracket, third_bracket}])
    plt.figure(figsize = (6,5))
    sns.heatmap(cm_plot, annot=True, vmin=5, vmax=90.5, cbar=False, fmt='g')


# In[ ]:


# Fit the classifier, get the prediction array and print the accuracy
def fit_and_pred_classifier(classifier, X_train, X_test, y_train, y_test):
    # Fit the classifier to the training data
    classifier.fit(X_train, y_train)

    # Get the prediction array
    y_pred = classifier.predict(X_test)
    
    # Get the accuracy %
    print("Accuracy: " + str(accuracy_score(y_test, y_pred) * 100) + "%") 
    
    return y_pred


# ## 4.1 Random Forest

# No problem is too big or too small for a random forest model. In fact, I tend to always start my analysis with it as it gives me a quick overview of what I can expect about a problem:
# 1. Are there a clear set of rules that govern the dependent variable being studied?
# 2. If so, what are the most important ones?
# 3. If not, go back to the drawing board.

# Without further ado, here is our first model:

# In[ ]:


# Build and fit the model
rf = RandomForestClassifier(n_estimators = 800, random_state = 0)
rf_salary_pred = fit_and_pred_classifier(rf, features_train, features_test, salary_train, salary_test)


# In[ ]:


cm = confusion_matrix(salary_test, rf_salary_pred)
plot_confusion_matrix(cm, 'Random Forest Confusion Matrix')


# ## 4.2 Linear SVC

# Lastly, we can try a simple linear support vector classification, because sometimes less is more:

# In[ ]:


# Build and fit the model
svc = SVC(kernel = 'linear', probability=True, random_state = 0)
svc_salary_pred = fit_and_pred_classifier(svc, features_train, features_test, salary_train, salary_test)


# In[ ]:


cm = confusion_matrix(salary_test, svc_salary_pred)
plot_confusion_matrix(cm, 'Linear SVC Classifier Confusion Matrix')


# ## 4.3 Linear discriminant analysis

# In[ ]:


lda = LinearDiscriminantAnalysis()
lda_salary_pred = fit_and_pred_classifier(lda, features_train, features_test, salary_train, salary_test)


# In[ ]:


cm = confusion_matrix(salary_test, lda_salary_pred)
plot_confusion_matrix(cm, 'OneVsRest Linear SVC Confusion Matrix')


# ## 4.5 Ensemble

# Our models did not perform terribly well. Perhaps there's a limitation with what we can achieve with the given dataset, perhaps they were not properly modeled. In theory, an ensemble should perform better, as even though the fitted classifiers are not that different, the small difference in them should help boost up the overall accuracy:

# In[ ]:


ensemble = VotingClassifier(estimators=[('rf', rf), ('svc', svc), ('lda', lda)], voting='soft', weights=[2,3,2], flatten_transform=True)
ensemble_salary_pred = fit_and_pred_classifier(ensemble, features_train, features_test, salary_train, salary_test)


# In[ ]:


cm = confusion_matrix(salary_test, ensemble_salary_pred)
plot_confusion_matrix(cm, 'Ensemble Confusion Matrix')


# Our ensemble improved our situation a little bit with roughly 68.28% precision. It's far from ideal, but it seems to be about as far as we'll go with the current pool of data.

# # 5. Statistically testing our models

# ## 5.1 Friedmann Test

# With several models built, it's now time to evaluate our results. Looking at the precision % gives us a good idea of what model went better, however, we can't blindly trust that. For that, we can use the Friedmann test.

# The Friedman test is the nonparametric version of the repeated measures analysis of variance test, or repeated measures ANOVA. The test can be thought of as a generalization of the Kruskal-Wallis H Test to more than two samples.
# 
# The default assumption, or null hypothesis, is that the multiple paired samples have the same distribution. A rejection of the null hypothesis indicates that one or more of the paired samples has a different distribution.
# 
# * Fail to Reject H0: Paired sample distributions are equal.
# * Reject H0: Paired sample distributions are not equal.
#     
# Our h0 is: there's no significant difference between our models. In theory, the ensemble model should be supperior, as it incorporates the strong points of random forest, linear svc and linear discriminant analysis. However, since the models performed very similarly, and made very similar mistakes, it could be a close call. Sometimes the ensemble may actually perform worse if there are not enough differences to go by.

# In[ ]:


# Friedman test
stat, p = friedmanchisquare(svc_salary_pred, lda_salary_pred, ensemble_salary_pred)
print('p=%.10f' % (p))


# In[ ]:


# interpret
def h0_test(p):
    alpha = 0.05
    if p > alpha:
        print('Same distributions (fail to reject H0)')
    else:
        print('Different distributions (reject H0)')

h0_test(p)


# In the end, we can reject the then null hypothesis: the methods are different, which means that we must go ahead and do pair-wise comparison between the models to identify which is the best. Wilcoxon Signed-Rank test should help us with that:

# ## 5.2 ANOVA

# In[ ]:


from scipy.stats import f_oneway
# compare samples
stat, p = f_oneway(svc_salary_pred, lda_salary_pred, ensemble_salary_pred)
print('p=%.3f' % (p))

# interpret
h0_test(p)


# With p = 0.099, we can't reject H0. It seems like our models are not that different in their predictions, at least for a single train/test split. However, the same ain't true if we include the random forest model on the pool:

# In[ ]:


# compare samples
stat, p = f_oneway(svc_salary_pred, lda_salary_pred, ensemble_salary_pred, rf_salary_pred)
print('p=%.3f' % (p))

# interpret
h0_test(p)


# This is due to the fact that the RF model performed worst than the others enough for it to significantly different.

# ## 5.2 Cross validating our models

# We already know that LDA, Linear SVC and our Ensemble are fairly similar. To make sure our analysis is correct, we can't trust our analysis on top of a single train/test split, we need to cross-validate our models in 10 folds and see how they perform under different circumstances:

# In[ ]:


# evaluate each model in turn
models = [['Linear SVC', svc], ['Random Forest', rf], ['Linear Discriminant Analysis', lda],['Ensemble', ensemble]]
results = []
names = []
scoring = 'accuracy'
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=7)
    cv_results = model_selection.cross_val_score(model, features_train, salary_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    print("%s accuracy: %0.4f (+/- %0.4f)" % (name, cv_results.mean(), cv_results.std() * 2))


# In[ ]:


# boxplot algorithm comparison
f, ax = plt.subplots(figsize=(15, 7))
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()


# In the end, Linear SVC, Linear Discriminant Analysis and our Ensemble performed very similarly. LDA had the highest mean, and the second best variation (Q1 and Q3). Our ensemble performed very similarly, didn't had a single outlier performance (as it was the case for Linear SVC and LDA) and had the highest maximum. We can discard the Random Forest model and keep the other three, to perform one final test.
# 
# It's hard to pick a winner, no model in particular achieved a consistent performance above 70%, which means that our features are not enough in their current form. 
# 
# That being said, the ensemble model was able to make use of the Linear SVC's good min/max values while eliminating the bad outlier performance and having a higher mean, which means that we can (barely) pick it as our model of choice.
