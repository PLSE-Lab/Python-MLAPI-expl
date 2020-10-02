#!/usr/bin/env python
# coding: utf-8

# # COMP 683 - Group B: Projects 1 & 2
# 
# ## Group Members
# Uilani Ballay, Yasin Dahi, John Yayros

# ## 1.0 Topic and Questions
# Introduction to our topic (i.e. Career/Job satisfaction for IT professionals, why it was chosen, expected outcome, etc.
# 
# We chose the topic of 'predictors' of job satisfaction for software developers for the following reasons:
# 1.   There is good value in trying to analyze non-straightforward questions using the datasets chosen and using Machine Learning in Kaggle. It is a good way for us to better learn the tools     and test our comprehension of topics learned in the course to date.
# 2.  It is a relevant field to our current area of study and we could potentially come out of the assessment with information that helps drive future decisions about our own career paths.
# 3. The assignment looks for a 'complex' topic, so we are looking into breaking our rather broad "Job Satisfaction in IT Professionals down to Software Developers and then down to more 
#     specific questions that hope to gain insight from the data. For example, we could look at what exactly "Job Satisfaction" factors there are and cluster interrogations towards the subject.
#                 

# ## 2.0 Data Source
# 
# #### 2.1 Stack Overflow Developer Survey
# Each year, [Stack Overflow ](https://stackoverflow.com/) (hereby referred to as SO) releases a survey to the develop community with questions ranging from their favourite technologies to their job preferences. This year marks the [eight year](https://insights.stackoverflow.com/survey/) that SO has published their Annual Developer Survey results. 
# 
# #### 2.2 Data Contents
# There are two tables of data:
# * **survey_results_public.csv**: contains the main survey results, one respondent per row and one column per question. This dataset will pose as the primary subject to be interrogated.
# * **survey_results_schema.csv**: contains each column name from the main results along with the question text corresponding to that column.
# 
# The dataset consists of 98,855 "qualified" (based on completion, time spent, PII content) responses.  Approximately 20,000 responses were started but not included in the public dataset because respondents did not answer enough questions, or only answered questions with personally identifying information. Of the qualified responses, 67,441 completed the entire survey.[1] 
# 
# #### 2.3 Data from Previous Surveys
# As stated above, this is the eighth year of the survey. There is potential to analyse the datasets from previous years in subsequent research initiatives, to compare findings and look for patterns.
# 
# #### 2.4Relation to Topic and Questions
# This is a fairly large bank of responsens from which we can garner some insight into not only the general question of where job satisfaction lays for Software Developers, but also provide potential insight into our subtopic questions and in our desire to extrapolate insight over time for a future prediction to be made. 
# 

# ## 3.0 Methodology
# Our team explored 2 approaches:
# 
# 1. Association rule mining: extract positive correlations between job satisfaction and other responses. 
# 1. Random forest classification: based on a set of responses for a given respondant with a 'job satisfaction' label, build a classification model which predicts job satisfaction level for a set of inputs (responses).
# 
# The team will also conduct research into the area of software developer job satisfaction with the goal of validating results.

# #### 3.1 Association Rule Mining

# In[ ]:


import numpy as np
import pandas as pd 

from mlxtend.frequent_patterns import apriori, association_rules
from IPython.display import Markdown, display

def write_markdown(filename, df):
    fmt = ['---' for i in range(len(df.columns))]
    df_fmt = pd.DataFrame([fmt], columns=df.columns)
    df_formatted = df
    
     # lose the frozensets
    df_formatted = pd.DataFrame(df)
    for column in df_formatted:
        if column in ['itemsets', 'antecedants', 'consequents']:
            df_formatted[column] = df_formatted[column].apply(lambda x: list(x))

    df_formatted = pd.concat([df_fmt, df_formatted])
    output = df_formatted.to_csv(sep="|", index=False)
    display(Markdown(output))
    with open(filename, 'w') as outfile:
        outfile.write(output)
        
dataset = '../input/survey_results_public.csv'

columns = ['Employment', 'FormalEducation', 'UndergradMajor', 'JobSatisfaction',  
            'HopeFiveYears', 'YearsCodingProf', 'CompanySize', 'YearsCoding',
            'LastNewJob', 'ConvertedSalary', 'EducationTypes', 'LanguageWorkedWith',    
            'IDE', 'OperatingSystem', 'Methodology', 'CheckInCode', 
            'EthicsChoice', 'EthicsReport', 'EthicsResponsible', 'WakeTime', 'SkipMeals',
            'Exercise', 'Gender', 'Age', 'Dependents']

df = pd.read_csv(dataset,
                 dtype='str',
                 na_values=['NA'],
                 usecols=columns)

# Remove any rows without a JobSatisfaction value
df = df[df['JobSatisfaction'].isnull() == False]

df_encoded = pd.get_dummies(df)

frequent_itemsets = apriori(df_encoded, min_support=0.05, use_colnames=True)

rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.3)

# pull the results in a new dataframe to display
results = pd.DataFrame()
result_cols = ['antecedants', 'consequents', 'lift', 'support', 'confidence']
results = rules[result_cols]

# filter the results on rules only containing consquents with 'JobSatisfaction'
results = results[results['consequents'].apply(str).str.contains('JobSatisfaction')]

# sort the results by lift
results = results.sort_values(by='lift', ascending=False)

# filter the results on rules that don't contain 'JobSatisfaction' as an antecedant
results = results[results['antecedants'].apply(str).str.contains('JobSatisfaction') == False]

print('done')


# Now let's further filter the results to only include rules with 'extremely satisfied' consequents.

# In[ ]:


es_results = results[results['consequents'].apply(str).str.contains('JobSatisfaction_Extremely satisfied')]
es_results = es_results.sort_values(by='lift', ascending=False)

# output the top 10 rows
write_markdown('es_rules.md', es_results.head(n=10))


# #### 3.2 Relevant column definitions
# 
# * EthicsChoice_No: The survey question was "Imagine that you were asked to write code for a purpose or product that you consider extremely unethical. Do you write the code anyway?".
# * Hope_Five_Years_Doing the same work: "Which of the following best describes what you hope to be doing in five years?". 7 options, including "doing the same work".
# * CheckInCode_Multiple times per day: "Over the last year, how often have you checked-in or committed code?". 6 options, ranging from none to multiple times per day'
# * SkipMeals_Never: "In a typical week, how many times do you skip a meal in order to be more productive?". 4 options, ranging from daily to never. 
# * Employment_Employed full-time: 4 options - prefer not say, no, part time, full time.
# * Gender_Male: The respondant's gender

# ##### 3.2.1 Ethics
# 
# Below we have respondant's answers to the ethics question, grouped by job satisfaction level. 66% of respondants that indicated the highest level of job satisfacton also indicated they would not do work that contradicts their ethical beliefs. 

# In[ ]:


# breakdown of ethics choice per job satisfaction group

data = df[['EthicsChoice', 'JobSatisfaction']].dropna()

# sort the job satisfaction levels from highest to lowest
repl_values = {
    'Extremely satisfied': '1. Extremely satisfied',
    'Moderately satisfied': '2. Moderately satisfied',
    'Slightly satisfied': '3. Slightly satisfied',
    'Neither satisfied nor dissatisfied': '4. Neither satisfied nor dissatisfied',
    'Slightly dissatisfied': '5. Slightly dissatisfied',
    'Moderately dissatisfied': '6. Moderately dissatisfied',
    'Extremely dissatisfied': '7. Extremely dissatisfied'
} 
data['JobSatisfaction'].replace(repl_values, inplace=True)

data_grouped = data.groupby(['JobSatisfaction', 'EthicsChoice']).size().reset_index(name='count')

pivot = data_grouped.pivot(index='JobSatisfaction', columns='EthicsChoice')

g = pivot.plot(kind='bar', width=.8)
g.legend(['Depends', 'No', 'Yes'])

pivot['count', 'total'] = pivot.sum(level=0, axis=1)
pivot['count', 'pct_no'] = round((pivot['count', 'No'] / pivot['count', 'total']) * 100, 2)
pivot['count', 'pct_yes'] = round((pivot['count', 'Yes'] / pivot['count', 'total']) * 100, 2)
pivot['count', 'pct_dep'] = round((pivot['count', 'Depends on what it is'] / pivot['count', 'total']) * 100, 2)

pivot


# ##### 3.2.2 Code check-in frequency
# 
# Below we see that 71.33% respondants that indicated the highest level of job satisfaction also indicated they checked in multiple times per day. Again we see the middle (neither satisfied or dissatisfied) group with lowest number respondants indicating multiple check-ins a day (59.59%), followed by the extremely dissatisfied group (60.22%).

# In[ ]:


#Checking in code multiple times per day
data = df[['CheckInCode', 'JobSatisfaction']].dropna()

# sort the job satisfaction levels from highest to lowest
data['JobSatisfaction'].replace(repl_values, inplace=True)

data_grouped = data.groupby(['JobSatisfaction', 'CheckInCode']).size().reset_index(name='count')

pivot = data_grouped.pivot(index='JobSatisfaction', columns='CheckInCode')

g = pivot.plot(kind='bar', width=.8)
g.legend(['A few times per week', 'Less than once per month', 'Multiple times per day', 
              'Never', 'Once a day', 'Weekly or a few times per month'])

pivot['count', 'total'] = pivot.sum(level=0, axis=1)
pivot['count', 'pct_few_wk'] = round((pivot['count', 'A few times per week'] / pivot['count', 'total']) * 100, 2)
pivot['count', 'pct_monthly'] = round((pivot['count', 'Less than once per month'] / pivot['count', 'total']) * 100, 2)
pivot['count', 'pct_mtl_day'] = round((pivot['count', 'Multiple times per day'] / pivot['count', 'total']) * 100, 2)
pivot['count', 'pct_never'] = round((pivot['count', 'Never'] / pivot['count', 'total']) * 100, 2)
pivot['count', 'pct_daily'] = round((pivot['count', 'Once a day'] / pivot['count', 'total']) * 100, 2)
pivot['count', 'pct_wkly'] = round((pivot['count', 'Weekly or a few times per month'] / pivot['count', 'total']) * 100, 2)

pivot


# ##### 3.2.3 Gender
# The resulting rules seem to indicate a strong correlation between being male and greater job satisfaction. Given the small percentage of female respondents in this survey, it is possible that many of the rules containg gender = female extracted did not meet our 5% support threshold.
# 
# There is some indication, within this dataset, of a potential correlation between job **dissatisfaction** and female respondents. Below we see that more females indicated they were 'extremely dissatisfied' than any other rating. 

# In[ ]:


data = df[['Gender', 'JobSatisfaction']].dropna()
data['JobSatisfaction'].replace(repl_values, inplace=True)
data['Gender'] = data['Gender'].apply(lambda x: 'Female' if 'Female' in x else ('Male' if 'Male' in x else 'Other'))
data_grouped = data.groupby(['JobSatisfaction', 'Gender']).size().reset_index(name='count')

pivot = data_grouped.pivot(index='JobSatisfaction', columns='Gender')

g = pivot.plot(kind='bar', width=.8)
g.legend(['Female', 'Male', 'Other'])

pivot['count', 'total'] = pivot.sum(level=0, axis=1)
pivot['count', 'pct_Female'] = round((pivot['count', 'Female'] / pivot['count', 'total']) * 100, 2)
pivot['count', 'pct_Male'] = round((pivot['count', 'Male'] / pivot['count', 'total']) * 100, 2)
pivot['count', 'pct_Other'] = round((pivot['count', 'Other'] / pivot['count', 'total']) * 100, 2)

pivot


# #### 3.3 Inferences on respondants that indicated the highest level of job satisfaction
# 
# * They want to be in the same role in 5 years.
# * They check-in code many times per day.
# * They hold a strong ethical stand as it relates to the purpose of their work.
# * They can be productive without the having to skip meals. 
# * Its unclear how much of a role gender plays in job satisfaction.

# ## Important Links
# * https://www.kaggle.com/stackoverflow/stack-overflow-2018-developer-survey
# * https://insights.stackoverflow.com/survey/
# * https://www.kaggle.com/jonathonv/career-satisfaction-analysis/notebook
# * https://www.kaggle.com/ranjeetjain3/deep-analysis-of-stackoverflow-survey-2018-v2
# 
# ## Refences
# 1. https://insights.stackoverflow.com/survey/2018/#methodology
# 
