#!/usr/bin/env python
# coding: utf-8

# ## Hacker Men V/s Hacker Women
# This notebook emphasises on finding difference in Women in Technology and Men in technology. HackerRank has provided their Survey data which they have collected from many different countries for analysing the Developers interest and taste of technology. <br>
# **The work has been organised in following way:**<br>
# 1. Loading And Reading files<br>
# 2. How many Male and Female Respondents?<br>
#     2.1 Difference in Total participation<br>
#     2.2 Country Wise participation<br>
# 3. Time Taken to complete survey Form<br>
# 4. Start of Technical life<br>
#     4.1 When people started coding in their life?<br>
#     4.2 What is the highest qualification of respondents?<br>
#     4.3 Degree focus of Respondents<br>
#     4.4 How people learnt coding?<br>
#     4.5 Employment level of Respondents and their Current Roles<br>
#     4.6 Which industry best describe the Respondents?<br>
# 5. Job Funting !!<br>
#     5.1 What people look in a company, Male V/s Female?<br>
#     5.2 How employers measure skills in candidates?<br>
#     5.3 Challenges for Hiring Managers while selecting candidates<br>
#     5.4 How many Hirings to be placed in the next year?<br>
#     5.5 How Hiring Managers filter Candidates?<br>
#     5.6  Core Competancies looked by Hiring Managers<br>
# 6. Technical Taste of People<br>
#     6.1 What Frameworks People like?<br>
#     6.2 Programming languages people know or will learn<br>
#     6.3 Frameworks people know or will learn<br>
#     6.4 Emerging Technical Skills<br>
#     6.5 Languages People Love or Hate<br>
#     6.6 Frameworks People Love or Hate<br>
# 7. Sources from where people learnt coding<br>
# 8. HackerRank as Part of interview<br>
# 9. Summary<br>
# 
# Hope this notebook will help in understanding the People in Technology and their taste for technology.<br><br>
# If this notebook helps, **Do Upvote**<br>
# **HAPPY CODING!!**
# 

# ## 1. Loading and Reading Files

# In[1]:


import numpy as np
import pandas as pd 
import copy
import datetime as dt
import os
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
from plotly import tools
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.offline as offline
offline.init_notebook_mode()

import warnings
warnings.filterwarnings('ignore')
print(os.listdir("../input"))


# In[2]:


country_code_mapping = pd.read_csv('../input/Country-Code-Mapping.csv')
survey_codebook = pd.read_csv('../input/HackerRank-Developer-Survey-2018-Codebook.csv')
survey_numeric_mapping = pd.read_csv('../input/HackerRank-Developer-Survey-2018-Numeric-Mapping.csv')
survey_country_numeric = pd.read_csv('../input/HackerRank-Developer-Survey-2018-Numeric.csv')
survey_country_values = pd.read_csv('../input/HackerRank-Developer-Survey-2018-Values.csv')


# ## 2. How many Male and Female Respondents?

# In[79]:


male_female_count = survey_country_values['q3Gender'].value_counts()[:3]                    .plot.bar(width=0.8)
    
plt.xticks(rotation=0)
plt.xlabel('Gender', fontsize=13)
plt.ylabel('Count', fontsize=13)
plt.title('Count of Gender Participation', fontsize=15)
plt.show()


# Most of the Respondents who has participated in HackerRank are Male while Females are very less almost 1/4th of male participants. Hope this gap will get reduced.

# In[6]:


def find_country_part(countries_count, gender, ax):
    plt.figure(figsize=(12,12))
    sns.barplot(countries_count.values, countries_count.index, ax=ax)
    for index, value in enumerate(countries_count.values):
        ax.text(0.8, index, value, color='k', fontsize=12)
    ax.set_title('Count of {} Participation in different countries'.format(gender), fontsize=15)
    ax.set_yticklabels(countries_count.index, fontsize=13)
    return ax


# In[78]:


countries_female_count = survey_country_values[survey_country_values['q3Gender'] == 'Female']                            ['CountryNumeric2'].value_counts().head(20)
countries = survey_country_values[survey_country_values['q3Gender'] == 'Male']            ['CountryNumeric2'].value_counts().head(20)
    
f, ax = plt.subplots(1, 2, figsize=(25, 20))
a1 = find_country_part(countries_female_count, 'Female', ax[0])
a2 = find_country_part(countries, 'Male', ax[1])
plt.show()


# **Indian and United States** has more participants for both male and female but the difference in male and female respondents in respective countries is very large.

# In[8]:


all_countries = survey_country_values['CountryNumeric2'].value_counts()
data = [ dict(
        type = 'choropleth',
        locations = all_countries.index,
        locationmode = 'country names',
        z = all_countries.values,
        text = all_countries.values,
        colorscale = [[0.0, 'rgb(242,240,247)'],[0.2, 'rgb(218,218,235)'],[0.4, 'rgb(188,189,220)'],\
            [0.6, 'rgb(158,154,200)'],[0.8, 'rgb(117,107,177)'],[1.0, 'rgb(84,39,143)']],
        autocolorscale = False,
        reversescale = False,
        marker = dict(
            line = dict (
                color = 'rgb(190,190,190)',
                width = 0.5
            ) ),
        colorbar = dict(
            autotick = False,
            tickprefix = '',
            title = 'Survey Participation'),
      ) ]

layout = dict(
    title = 'Participation in survey from different countries',
    geo = dict(
        showframe = False,
        showcoastlines = False,
        projection = dict(
            type = 'Mercator'
        )
    )
)

fig = dict( data=data, layout=layout )
py.iplot( fig, validate=False, filename='Survey participation' )


# As earlier analysed, **India and United stated** has far more repondents than any other country in the world. So we can say that these country's People are most active on HackerRank and there may be more opportunities there. We will see in the further notebook which country is going to have more opportunities.

# ## 3. Time taken to complete the survey form

# In[76]:


survey_response_time = copy.deepcopy(survey_country_values[['StartDate', 'EndDate']])
survey_response_time['StartDate'] = pd.to_datetime(survey_response_time['StartDate'])
survey_response_time['EndDate'] = pd.to_datetime(survey_response_time['EndDate'])

survey_response_time['Difference'] = survey_response_time.apply                                    (lambda df: pd.Timedelta(df['EndDate'] - df['StartDate'])                                     .seconds/60, axis=1)


# In[77]:


max_survey_time = survey_response_time['Difference'].max()
min_survey_time = survey_response_time['Difference'].min()
mean_survey_time = survey_response_time['Difference'].mean()
print('Maximum time taken to complete survey is {} minutes'.format(max_survey_time))
print('Minimum time taken to complete survey is {} minutes'.format(min_survey_time))
print('Mean time taken to complete survey is {} minutes'.format(mean_survey_time))


# From the above output, the average time taken by respondents to fill survey form is 55 minutes while the highest time taken is 1436 minutes which is far more than the time taken to fill any survey form.  And the minimum time taken to fill the form was 2 minutes which is very less to complete the form so the person has filled the form just by  **skipping and skipping and Done**. 

# #### Let's see how many individuals has taken far more time to fill the form

# In[11]:


print('Number of individual taken more than 55 minutes are {}'      .format(survey_response_time[survey_response_time['Difference'] > 55]['Difference'].count()))


# So as expected many people, almost 1976 respondents, has taken far more time to complete the form. So we will skip these people while finding the distribution of time Taken.

# ### Distribution of time taken by all respondents

# In[12]:


fig = plt.figure(figsize=(8,8))
sns.distplot(survey_response_time[survey_response_time['Difference'] <= 55]['Difference'])
plt.xlabel('Time in minutes', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.title('Distribution of Time Taken')
plt.show()


# Most of the respondents has filled the survey form in ** 8 to 15 minutes** which is the most expected time to fill any survey form.

# ## 4. Start of Technical life!!
# Here we will see how people came into technical life and what are their preferences.
# ### 4.1 When People start Coding in their life ?

# In[13]:


f, ax = plt.subplots(1,2, figsize=(27, 14))
sns.countplot(y='q1AgeBeginCoding', hue='q3Gender',            data=survey_country_values, ax=ax[0])
ax[0].set_ylabel('Age Groups', fontsize=12)
ax[0].set_xlabel('Count', fontsize=12)
ax[0].set_yticklabels(ax[0].get_yticklabels(), fontsize=13)
ax[0].set_title('Age When Started Coding')

sns.countplot(y='q1AgeBeginCoding', hue='q3Gender',            data=survey_country_values, ax=ax[1])
ax[1].set_ylabel('Age Groups', fontsize=12)
ax[1].set_xlabel('Count', fontsize=12)
ax[1].set_yticklabels(ax[1].get_yticklabels(), fontsize=13)
ax[1].set_title('Current Age of Respondents')
plt.show()


# Most of the people started coding between age group 15 to 20 years and also the current age of most of the respondents lie between 16 to 20 years. So we can say that **younger generation is more interested in coding** in comparison to people of age greater than 35 years.

# ### 4.2 What is the Highest Qualification of Respondent?

# In[75]:


qualification = survey_country_values['q4Education'].value_counts()                .sort_values(ascending=True).drop('#NULL!').plot.barh()
    
plt.xlabel('Count', fontsize=13)
plt.ylabel('Qualification', fontsize=13)
plt.yticks(fontsize=12)
plt.title('Highest Qualification of Respondents', fontsize=15)
plt.show()


# More people are **college graduates** who has participated in the HackerRank Survey, as expected because most of the respondents has their current age between 16 to 24 years. After college graduates, post graduates also participated in large number.

# ### 4.3 What is the Degree Focus of Respondents?

# In[74]:


f, ax = plt.subplots(1,2, figsize=(20,8))
degree_focus = survey_country_values['q5DegreeFocus'].value_counts()                .sort_values(ascending=False)                .drop('#NULL!').plot.bar(ax=ax[0])
        
ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=0)
ax[0].set_xlabel('Count', fontsize=12)
ax[0].set_ylabel('Degree', fontsize=12)
ax[0].set_title('Degree Focus of Respondents', fontsize=14)

other_focus = survey_country_values['q0005_other'].value_counts()[:10]
ax[1].pie(other_focus.values, labels=other_focus.index, autopct='%1.1f%%')
empty_circle = plt.Circle((0,0), 0.7, color='white')
p = plt.gcf()
p.gca().add_artist(empty_circle)
ax[1].set_title('Other Degree Focus of Respondents', fontsize=14)
plt.show()


# **Very large number of respondents has Computer science Degree** while some has Other STEM. So more population is interested in Technical life which is a good sign of increasing interest in computer science. So we can say that, in future there will be many software engineers who can help in transforming the world. Other respondents has Business, Economic, Psychology, etc preferences.

# ### 4.4 How Respondents learnt Coding?

# In[73]:


columns = survey_country_values.columns[survey_country_values.columns.str                                        .startswith('q6')]                                        .drop('q6LearnCodeOther')
how_learn_code = []
labels = []

for col in columns:
    labels.append(str(survey_numeric_mapping[survey_numeric_mapping['Data Field'] == col]                      ['Label'].values).strip("[]''").strip('""'))
    how_learn_code.append(survey_country_values[col].value_counts()[0])

    
fig, ax = plt.subplots(1, 2, figsize=(20,8))
ax[0].pie(how_learn_code, labels=labels, autopct='%1.1f%%', shadow=True)
ax[0].set_title('How learned Coding ?', fontsize=20)


wc = WordCloud(background_color="white", max_words=1000, 
               stopwords=STOPWORDS, width=1000, height=1000)
wc.generate(" ".join(survey_country_values['q0006_other'].dropna()))
ax[1].imshow(wc)
ax[1].axis('off')
ax[1].set_title('How Other learned Coding ?', fontsize=20)
plt.show()


# Most of Respondents say that they learnt coding **on their own or at their School or Universities**. Other people say that they have learnt coding **during work, Job, from friend, from HackerRank, during training, etc.** 
# <br>
# <br>
# <h3>Now lets see who are the people who say that they have learnt coding in work or Job ?</h3>

# In[72]:


survey_country_values[survey_country_values['q0006_other']                      .isin(['Work', 'Job'])]['q9CurrentRole'].dropna()                      .value_counts().to_frame()                      .style.background_gradient(cmap='summer_r')


# As Expected, we can see that all the respondents who say they learnt coding at work or job are working individuals as Developers.

# ### 4.5 What is the Employment level of Respondents?

# In[68]:


other_focus = survey_country_values['q8JobLevel'].value_counts().sort_values()                .plot.barh()
plt.xlabel('Count', fontsize=13)
plt.ylabel('Employment Status', fontsize=13)
plt.yticks(fontsize=12)
plt.title('Employment Level of Respondents', fontsize=15)
plt.show()


# As expected, more respondents are **students, Senior Developers and Level 1Junor Developers** who has participated in the survey because most of the respondents fall in the age group 16 to 24 years. 
# <br>
# ### Now let's see what are their current roles?

# In[67]:


f, ax = plt.subplots(1, 2, figsize=(20, 10))
sns.countplot(y = 'q9CurrentRole', hue='q3Gender',              data=survey_country_values[['q9CurrentRole', 'q3Gender']], ax=ax[0])

ax[0].set_xlabel('Count', fontsize=17)
ax[0].set_ylabel('Current Roles', fontsize=17)
ax[0].set_yticklabels(ax[0].get_yticklabels(), fontsize=13)
ax[0].set_title('Current Roles of Male and Female Respondent', fontsize=20)

current_role = survey_country_values['q9CurrentRole'].value_counts()[:10]
ax[1].pie(current_role.values, labels=current_role.index, autopct='%1.1f%%')
empty_circle = plt.Circle((0,0), 0.7, color='white')
p = plt.gcf()
p.gca().add_artist(empty_circle)
ax[1].set_title('Current Roles of ALL Respondents', fontsize=20)
plt.show()


# 1) Survey respondents (male and female) are mostly **Students** but the proportion of female respondents is very less in comparison to male respondents.<br>
# 2) We can see that more male population is in Technology like Software Engineering, Full Stack Developer, Backend Developer, etc. in comparison to female and this difference is very large. **<u>Hope this difference will become less and less.</u>**<br>
# 

# ### 4.6 Which industry best describe the Respondents?
# We can see that most of the population is in Technology from above grpah. Now let's see what are their industry roles?

# In[66]:


fig = plt.figure(figsize=(12,12))
industry_data = survey_country_values.groupby(['CountryNumeric2','q10Industry'])                                            ['q3Gender'].count().reset_index()
industry_data = industry_data[industry_data['q10Industry']!='#NULL!']
industry_data = industry_data.pivot('CountryNumeric2','q10Industry','q3Gender')                .dropna(thresh=15)
sns.heatmap(industry_data, cmap='RdYlGn', fmt='2.0f', annot=True)
plt.ylabel('Countries', fontsize=14)
plt.xlabel('Industry', fontsize=14)
plt.title('Industry Specification across different countries', fontsize=18)
plt.show()


# From the Heatmap, **it can be seen that most of the respondents are from Technical Industry with more numbers from India (3687) followed by United States (1850).** Other Industrial preference includes Aerospace, Transportation, Government, etc.

# ## 5. Now Comes Job Hunting !!
# ### 5.1 What people look in a company when looking for Job Opportunites, Male Vs Female?

# In[64]:


columns = survey_country_values.columns[survey_country_values.columns.                                        str.startswith('q12')]                                        .drop('q12JobCritOther')
columns = list(columns)
# columns.insert(len(columns), 'q3Gender')

def find_choices(columns, gender):
    choices = []
    choices_count = []
    for col in columns:
        value = survey_country_values[survey_country_values['q3Gender'] == gender][col]                .value_counts()
        choices.append(str(survey_numeric_mapping[survey_numeric_mapping['Data Field']                                                  == col]['Label'].values).strip("[]''")                                                  .strip('""'))
        choices_count.append(int(value.values))
    return choices, choices_count

females_choices, females_choices_count = find_choices(columns, 'Female')
males_choices, males_choices_count = find_choices(columns, 'Male')


# In[65]:


f, ax = plt.subplots(1, 2, sharey=True, figsize=(15,8))
sns.barplot(females_choices_count, females_choices, ax=ax[0])
ax[0].set_xlabel('Count', fontsize=15)
for index, value in enumerate(females_choices_count):
        ax[0].text(0.8, index, str(value).strip("[]"), color='k', fontsize=12)
ax[0].set_ylabel('Features in Company', fontsize=15)
ax[0].set_title('What Females look in a company?', fontsize=18)
ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=45)


sns.barplot(males_choices_count, males_choices, ax=ax[1])
for index, value in enumerate(males_choices_count):
        ax[1].text(0.8, index, str(value).strip("[]"), color='k', fontsize=12)
ax[1].set_xlabel('Count', fontsize=15)
ax[1].set_title('What Males look in a company?', fontsize=18)
ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=45)
plt.show()


# Mostly people including male and female look for **Professional growth and learning and Good Work life Balance** in the company. People, today, are more interested in the growth of their personal skills while working in company which is a good sign. Other interests include Smart Team, Interesting Problems to solve and Good Company Culture which is very much expected outcome.<br><br>
# **So it can be said that people are more keen to learn and solve problems while working in a company.**

# ### 5.2 How employers Measure Skills in candidates?

# In[63]:


columns = survey_country_values.columns[survey_country_values.columns.str.                                        startswith('q13')].drop('q13EmpMeasOther')
emp_measure = []
labels = []

for col in columns:
    labels.append(str(survey_numeric_mapping[survey_numeric_mapping['Data Field']                                             == col]['Label'].values).strip("[]''").strip('""'))
    emp_measure.append(survey_country_values[col].value_counts()[0])

fig1= plt.figure(figsize=(8,8))
plt.pie(emp_measure, labels=labels, autopct='%1.1f%%', shadow=True)
empty_circle = plt.Circle((0,0), 0.7, color='white')
p = plt.gcf()
p.gca().add_artist(empty_circle)
plt.title('Skills Measurement ways', fontsize=20)
plt.show()


# **Resume, Telephonic Interviews, Past Work, White Board interview** are the methods which recruiters prefer to measure the skills of the candidates while Hackerrank Challenges, Home Projects and Past Work can help to boost up the profile of the candidate and also can help to land the candidate in front of thousands.

# ### 5.3 Challenges for Hiring Manager !!
# - **How many Hiring Managers ?**<br>
# - **What Challenges they face ?**

# In[61]:


def find_num_managers(gender):
    num_managers = survey_country_values[survey_country_values['q3Gender'] == gender]                    ['q16HiringManager'].value_counts()
    return num_managers

def find_challenges_data(gender):
    columns = survey_country_values.columns[survey_country_values.columns.str                                            .startswith('q17')].drop('q17HirChaOther')
   
    challenge_measure = []
    labels = []
    for col in columns:
        label = str(survey_numeric_mapping[survey_numeric_mapping['Data Field'] == col]                    ['Label'].values).strip("[]''").strip('""')
        labels.append(label)
        challenge_measure.append(survey_country_values[survey_country_values['q3Gender'] ==                                                       gender][col].value_counts()[label])
    return labels, challenge_measure


# In[62]:


female_manager = find_num_managers('Female')
male_manager = find_num_managers('Male')

female_labels, chall_faced_female = find_challenges_data('Female')
male_labels, chall_faced_male = find_challenges_data('Male')

trace0 = go.Bar(
    x = ['Female', 'Male'],
    y = [female_manager['Yes'], male_manager['Yes']]
)
trace1 = go.Bar(
    x = chall_faced_female,
    y = female_labels,
    orientation = 'h',
    yaxis = 'y2'
)
trace2 = go.Bar(
    x = chall_faced_male,
    y = male_labels,
    orientation = 'h',
    yaxis = 'y3'
)

fig = tools.make_subplots(rows=2, cols=2, specs=[[{}, {'rowspan': 2}], [{}, None]],
                          subplot_titles=('Female Manager Facing Challenges',\
                                          'Hiring Managers',\
                                          'Male Manager Facing Challenges'))

fig.append_trace(trace0, 1, 2)
fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 2, 1)


fig['layout'].update(showlegend=False, title='Hiring Managers Analysis', margin=go.Margin(l=200))
py.iplot(fig)


# There is difference between male and female population in Hiring Managers as well.<br>
# 1) There are **far more Male Hiring Managers than Female Hiring Managers approximately 7 times**.<br>
# 2) Both Female and Male Hiring Managers say that it is hard for them to assess skills before onsite. Other Challenges which are most common for both male and female are the time consumed in the interview processes while selecting the candidates and not enough talent found in the candidates.<br>
# 

# ### 5.4 How many Hiring in the next Year ?
# We will find the number of hiring as per countries so as to see which country is going to produce more jobs!!

# In[60]:


plt.figure(figsize=(8,8))
coun=survey_country_values.groupby(['CountryNumeric2','q18NumDevelopHireWithinNextYear'])                                    ['q3Gender'].count().reset_index()
coun=coun[coun['q18NumDevelopHireWithinNextYear']!='#NULL!']
coun=coun.pivot('CountryNumeric2','q18NumDevelopHireWithinNextYear','q3Gender')                .dropna(thresh=5)
sns.heatmap(coun,cmap='RdYlGn',fmt='2.0f',annot=True)
plt.ylabel('Country Name', fontsize=13)
plt.xlabel('Number of Hirings in the next year', fontsize=13)
plt.title('Number of Hiring in the next year', fontsize=15)
plt.show()


# **India and United States** are going to offer more jobs in the next year so people has more scope for growth in these countries. Let's see how many jobs each of these two countries are going to offer.

# In[59]:


all_countries=survey_country_values[survey_country_values['q18NumDevelopHireWithinNextYear']                                    != '#NULL!']['CountryNumeric2'].value_counts()
data = [ dict(
        type = 'choropleth',
        locations = all_countries.index,
        locationmode = 'country names',
        z = all_countries.values,
        text = all_countries.values,
        colorscale = [[0.0, 'rgb(242,240,247)'],[0.2, 'rgb(218,218,235)'],[0.4, 'rgb(188,189,220)'],\
            [0.6, 'rgb(158,154,200)'],[0.8, 'rgb(117,107,177)'],[1.0, 'rgb(84,39,143)']],
        autocolorscale = False,
        reversescale = False,
        marker = dict(
            line = dict (
                color = 'rgb(190,190,190)',
                width = 0.5
            ) ),
        colorbar = dict(
            autotick = False,
            tickprefix = '',
            title = 'Number of Hirings'),
      ) ]

layout = dict(
    title = 'Number of Hirings in the next year',
    geo = dict(
        showframe = False,
        showcoastlines = False,
        projection = dict(
            type = 'Mercator'
        )
    )
)

fig = dict( data=data, layout=layout )
py.iplot( fig, validate=False, filename='Survey participation' )


# **India is going to hire multiples of 1736 Developers Next year while United states will hire multiples of 1643 Developers. There are more chances for growth in these countries.**

# ### 5.5 How Hiring Managers filter Candidates??
# <u>We will catch two methods here</u><br>
# 1) Filtering of candidates before first step of interview process.<br>
# 2) Top Qualities before Onsite.

# In[57]:


def filter_candidate(col_start):
    columns = survey_country_values.columns[survey_country_values.columns.str.                                            startswith(col_start)]
    if col_start == 'q19':
        columns = columns.drop('q19TalToolOther')
    elif col_start == 'q20':
        columns = columns.drop(['q20Other', 'q20CandGithubPersProj'])
    elif col_start == 'q30':
        columns = columns.drop('q30LearnCodeOther')
   
    challenge_measure = []
    labels = []
    for col in columns:
        label = np.asscalar(survey_numeric_mapping[survey_numeric_mapping['Data Field']==col]                            ['Label'].values)
        labels.append(label)
        challenge_measure.append(survey_country_values[col].value_counts()[label])
    return labels, challenge_measure


# In[58]:


pre_interview_labels, pre_interview_count = filter_candidate('q19')
before_onsite_labels, before_onsite_count = filter_candidate('q20')

fig, ax = plt.subplots(2, 1, figsize=(30,20))

# Plotting Pre interview filtering
sns.barplot(pre_interview_count, pre_interview_labels, ax=ax[0])
for index, value in enumerate(pre_interview_count):
        ax[0].text(0.8, index, str(value).strip("[]"), color='k', fontsize=12)
        
ax[0].set_xlabel('Count', fontsize=15)
ax[0].set_yticklabels(ax[0].get_yticklabels(), fontsize=15)
ax[0].set_title('Pre Interview Filtering?', fontsize=18)


# Plotting Before onsite Filtering
sns.barplot(before_onsite_count, before_onsite_labels, ax=ax[1])
for index, value in enumerate(before_onsite_count):
        ax[1].text(0.8, index, str(value).strip("[]"), color='k', fontsize=12)
        
ax[1].set_xlabel('Count', fontsize=15)
ax[1].set_yticklabels(ax[1].get_yticklabels(), fontsize=15)
ax[1].set_title('Before Onsite Filtering', fontsize=18)
plt.show()


# 
# 1) **Before First Step of interview: ** Most of the Hiring Managers filter candidates from
#     - Resume
#     - participation in problem solving challenges
#     - Referrals
# 2) **Before Onsite Interview: ** Most of the Hiring Managers filter candidates by-
#     - Years of Experience of Candidates
#     - Previous Work Experience
#     - Github or Personal Projects

# ### 5.6 Core Competancies in Candidates looked by Hiring Team!!

# In[55]:


def filter_data(col_start):
    columns = survey_country_values.columns[survey_country_values.columns.str                                            .startswith(col_start)]
    if col_start == 'q22':
        columns = columns.drop('q22LangProfOther')
    elif col_start == 'q23':
        columns = columns.drop('q23FrameOther')
   
    measure = []
    labels = []
    for col in columns:
        label = np.asscalar(survey_numeric_mapping[survey_numeric_mapping['Data Field']==col]                            ['Label'].values)
        labels.append(label)
        measure.append(survey_country_values[col].value_counts()[label])
    return labels, measure


# In[56]:


general_skills_labels, general_skills_count = filter_data('q21')
language_labels, language_count = filter_data('q22')

fig, ax = plt.subplots(1, 2, figsize=(30,12))


sns.barplot(general_skills_labels, general_skills_count, ax=ax[0])
ax[0].set_ylabel('Count', fontsize=15)
ax[0].set_title('Other Skills', fontsize=18)
ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=45, fontsize=14)



sns.barplot(language_labels, language_count, ax=ax[1])
ax[1].set_ylabel('Count', fontsize=15)
ax[1].set_title('Language Skills', fontsize=18)
ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=45, fontsize=14)
plt.show()


# 1) **Problem Solving skills** are mostly preferred by Hiring Team. So candidate should focus more on Data Structures and Algorithms in order to prepare for the interview.<br>
# 2) **Java and Javascript** are the two languages which Hiring Team look into the candidates. So if you have these skills then that will be a boost to your profile.

# ## 6. Technical Taste of People!!
# We will analyse which technology people like, which one they are ooking to learn in future, etc

# ### 6.1 What are the Frameworks liked by Respondents??

# In[34]:


framework_labels, framework_count = filter_data('q23')

fig = plt.figure(figsize=(20,8))
sns.barplot(framework_labels, framework_count)
plt.ylabel('Count', fontsize=15)
plt.title('Frameworks liked', fontsize=20)
plt.xticks(rotation=45)
plt.show()


# **AngularJS, React and Node.js** are the most liked frameworks by Respondents.

# ### 6.2 Which Programming Languages Respondents know or will learn ??

# In[38]:


columns=['q25LangC','q25LangCPlusPlus','q25LangJava','q25LangPython','q25LangJavascript',         'q25LangCSharp','q25LangGo','q25Scala','q25LangPHP','q25LangR']
col_mapping = {'q25LangC': 'C',
               'q25LangCPlusPlus': 'C++',
               'q25LangJava': 'Java',
               'q25LangPython': 'Python',
               'q25LangJavascript': 'Javascript',
               'q25LangCSharp': 'CSharp',
               'q25LangGo': 'Go',
               'q25Scala': 'Scala',
               'q25LangPHP': 'PHP',
               'q25LangR': 'R'
              }
knows = []
will_learn =[]
cols = []
for col in columns:
    cols.append(col_mapping[col])
    knows.append(survey_country_values[survey_country_values[col] == 'Know'][col].count())
    will_learn.append(survey_country_values[survey_country_values[col] == 'Will Learn'][col]                      .count())
    
trace1 = go.Bar(
    x = cols,
    y = knows,
    name = 'Know'
)
trace2 = go.Bar(
    x = cols,
    y = will_learn,
    name = 'Will Learn'
)

data = [trace1, trace2]
layout = go.Layout(
    barmode = 'stack',
    title = 'Which Programming Language people Know or Will Learn'
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)


# We can say that **C, C++, Java, Javascript** are the languages which most of the people know while **CSharp, LangGo, Scala, R** are the languages which people are willing to learn which is a good sign that people are keen to learn more languages.

# ### 6.3 Which frameworks people know or will learn?

# In[40]:


columns = survey_country_values.columns[survey_country_values.columns.str.startswith('q26')]            .drop(['q26FrameLearnPadrino2', 'q26FrameLearnDjango2', 'q26FrameLearnPyramid2'])
col_mapping = {'q26FrameLearnAngularJS': 'AngularJS',
               'q26FrameLearnReact': 'React',
               'q26FrameLearnVueDotjs': 'Vue.js',
               'q26FrameLearnEmber': 'Ember',
               'q26FrameLearnBackboneDotjs': 'Backbone.js',
               'q26FrameLearnSpring': 'Spring',
               'q26FrameLearnJSF': 'JSF',
               'q26FrameLearnStruts': 'Struts',
               'q26FrameLearnDjango': 'Django',
               'q26FrameLearnPyramid': 'Pyramid',
               'q26FrameLearnRubyonRails': 'Ruby on Rails',
               'q26FrameLearnPadrino': 'Padrino',
               'q26FrameLearnASP': 'ASP',
               'q26FrameLearnNetCore': 'Net Core',
               'q26FrameLearnNodeDotjs': 'Node.js',
               'q26FrameLearnExpressJS': 'Express.js',
               'q26FrameLearnMetero': 'Metero',
               'q26FrameLearnCocoa': 'Cocoa',
               'q26FrameLearnReactNative': 'React Native',
               'q26FrameLearnRubyMotion': 'Ruby Motion',
               'q26FrameLearnPadrino2': 'Padrino 2',
               'q26FrameLearnDjango2': 'Django 2',
               'q26FrameLearnPyramid2': 'Pyramid 2'
              }
knows = []
will_learn =[]
col_name = []
for col in columns:
    col_name.append(col_mapping[col])
    knows.append(survey_country_values[survey_country_values[col] == 'Know'][col].count())
    will_learn.append(survey_country_values[survey_country_values[col] == 'Will Learn'][col]                      .count())
    
trace1 = go.Bar(
    x = col_name,
    y = knows,
    name = 'Know'
)
trace2 = go.Bar(
    x = col_name,
    y = will_learn,
    name = 'Will Learn'
)

data = [trace1, trace2]
layout = go.Layout(
    barmode = 'stack',
    title = 'Which Frameworks people Know or Will Learn'
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)


# Most of the people are familiar with **Angular and Node.js** but more of the respondent don't know about these framework but are willing to learn in future. This behavious is expected as most of the people liked AngularJS and Node.js framework as seen in the above analysis.

# ### 6.4 What are the Emerging Technical Skills people are learning or looking to learn in future?

# In[41]:


plt.figure(figsize=(8,8))
wc = WordCloud(background_color="green", max_words=1000, 
               stopwords=STOPWORDS, width=1000, height=1000)
wc.generate(" ".join(survey_country_values['q27EmergingTechSkill'].dropna()))
plt.imshow(wc)
plt.axis('off')
plt.title('What are emerging Tech skills', fontsize=20)
plt.show()


# Artificial Intellignece is creatinf more enthusiasm among people of different countires as **Machine Learning and Deep Learning** are the skills which most of the people are learning or looking to learn in next year.

# ### 6.5 What Languages Love or Hate ?

# In[43]:


columns = survey_country_values.columns[survey_country_values.columns.str.startswith('q28')]            .drop('q28LoveOther')
    
col_mapping = {'q28LoveC': 'C',
            'q28LoveCPlusPlus': 'C++',
            'q28LoveJava': 'Java',
            'q28LovePython': 'Python',
            'q28LoveRuby': 'Ruby',
            'q28LoveJavascript': 'Javascript',
            'q28LoveCSharp': 'C#',
            'q28LoveGo': 'Go',
            'q28LoveScala': 'Scala',
            'q28LovePerl': 'Perl',
            'q28LoveSwift': 'Swift',
            'q28LovePascal': 'Pascal',
            'q28LoveClojure': 'Clojure',
            'q28LovePHP': 'PHP',
            'q28LoveHaskell': 'Haskell',
            'q28LoveLua': 'Lua',
            'q28LoveR': 'R',
            'q28LoveRust': 'Rust',
            'q28LoveKotlin': 'Kotlin',
            'q28LoveTypescript': 'Typescript',
            'q28LoveErlang': 'Erlang',
            'q28LoveJulia': 'Julia',
            'q28LoveOCaml': 'OCaml'
           }
love = []
hate =[]
col_name
for col in columns:
    col_name.append(col_mapping[col])
    love.append(survey_country_values[survey_country_values[col] == 'Love'][col].count())
    hate.append(survey_country_values[survey_country_values[col] == 'Hate'][col].count())
    
trace1 = go.Bar(
    x = col_name,
    y = love,
    name = 'Love'
)
trace2 = go.Bar(
    x = col_name,
    y = hate,
    name = 'Hate'
)

data = [trace1, trace2]
layout = go.Layout(
    barmode = 'stack',
    title = 'Languages Love or Hate?'
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)


# **C, C++, Python, Javascript and Java** are the most loved by respondents, as expected because same languages people want to learn or are learning as we have seen previous.

# ### 6.6 Frameworks Love or Hate?

# In[46]:


columns = survey_country_values.columns[survey_country_values.columns.str.startswith('q29')]
col_mapping = {'q29FrameLoveAngularJS': 'AngularJS',
               'q29FrameLoveReact': 'React',
               'q29FrameLoveVuedotjs': 'Vue.js',
               'q29FrameLoveEmber': 'Ember',
               'q29FrameLoveBackboneDotjs': 'Backbone.js',
               'q29FrameLoveSpring': 'Spring',
               'q29FrameLoveJSF': 'JSF',
               'q29FrameLoveStruts': 'Structs',
               'q29FrameLoveDjango': 'Django',
               'q29FrameLovePyramid': 'Pyramid',
               'q29FrameLoveRubyonRails': 'Ruby on Rails',
               'q29FrameLovePadrino': 'Padrino',
               'q29FrameLoveASP': 'ASP',
               'q29FrameLoveNetCore': 'Net Core',
               'q29FrameLoveNodeDotjs': 'Node.js',
               'q29FrameLoveExpressJS': 'Express.js',
               'q29FrameLoveMeteor': 'Meteor',
               'q29FrameLoveCocoa': 'Cocoa',
               'q29FrameLoveReactNative': 'React Native',
               'q29FrameLoveRubyMotion': 'Ruby Motion'
              }
love = []
hate =[]
col_name = []
for col in columns:
    col_name.append(col_mapping[col])
    love.append(survey_country_values[survey_country_values[col] == 'Love'][col].count())
    hate.append(survey_country_values[survey_country_values[col] == 'Hate'][col].count())
    
trace1 = go.Bar(
    x = col_name,
    y = love,
    name = 'Love'
)
trace2 = go.Bar(
    x = col_name,
    y = hate,
    name = 'Hate'
)

data = [trace1, trace2]
layout = go.Layout(
    barmode = 'stack',
    title = 'Frameworks Love or Hate?'
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)


# As Expected ** AngularJS, React, Node.JS and Django** are the frameworks which ar emost loved by respondent while **Pyramid and Padrino** are frameworks which most of the people hate.

# ## 7. How Learnt Coding? HackerRank is Recommended or Not?

# In[49]:


f, ax = plt.subplots(1, 2, figsize=(20, 10))
learn_coding_labels, learn_coding_count = filter_candidate('q30')
sns.barplot(learn_coding_count, learn_coding_labels, ax=ax[0])
for index, value in enumerate(learn_coding_count):
        ax[0].text(0.8, index, str(value).strip("[]"), color='k', fontsize=12)
        
ax[0].set_xlabel('Count', fontsize=15)
ax[0].set_xticklabels(ax[0].get_xticklabels(), fontsize=13)
ax[0].set_title('How Leanred Coding?', fontsize=18)

current_role = survey_country_values['q32RecommendHackerRank'].value_counts().drop('#NULL!')
ax[1].pie(current_role.values, labels=current_role.index, autopct='%1.1f%%')
empty_circle = plt.Circle((0,0), 0.7, color='white')
p = plt.gcf()
p.gca().add_artist(empty_circle)
ax[1].set_title('Recommend HackerRank ?', fontsize=20)
plt.show()


# It can be  inferred that:<br>
# 1) **HackerRank Recommendation: ** 97% of Respondents recommend HackerRank. So HackerRank is a good platform for building Skills.<br>
# 2) **How Learn Coding?: ** People learn coding from various sources like StackOverflow, YouTube, Books and MOOC, etc. Stackoverflow proide answers to almost every questions and is most voted answer by respondents.

# ## 8. HackerRank Challenge As Part of Interview!!

# In[54]:


Hackerrank_for_interview = survey_country_values['q33HackerRankChallforJob']                            .value_counts().drop('#NULL!')
hackerrank_exp = survey_country_values['q34PositiveExp'].value_counts()

f, ax = plt.subplots(1, 2, figsize=(15,8))
ax[0].pie(Hackerrank_for_interview.values, labels=Hackerrank_for_interview.index,          autopct='%1.1f%%')
ax[0].set_title('Participated in HackerRank as Interview Part?', fontsize=15)

ax[1].pie(hackerrank_exp.values, labels=hackerrank_exp.index, autopct='%1.1f%%')
ax[1].set_title('Rank Interview Experience using HackerRank?', fontsize=15)
empty_circle = plt.Circle((0,0), 0.7, color='white')
p = plt.gcf()
p.gca().add_artist(empty_circle)
plt.show()


# It can be inferred that:<br>
# 1) **60% of respondents** say that they have been given challenges on hackerrank as part of their interview process.<br>
# 2) **How People feel about interview using Hackerrank?:** Most of the People have positive experienc eon giving interviews on HackerRank.

# In[53]:


len_hackerrank_interview = survey_country_values['q34IdealLengHackerRankTest']                            .value_counts().drop('#NULL!')
plt.pie(len_hackerrank_interview.values, labels=len_hackerrank_interview.index,        autopct='%1.1f%%')
plt.title('Lenght of Test using HackerRank as part of interview?', fontsize=15)

empty_circle = plt.Circle((0,0), 0.7, color='white')
p = plt.gcf()
p.gca().add_artist(empty_circle)
plt.show()


# Test on HackerRank for any interview process various in length. Some has given 1-2 hours test and some has 46-60 minutes and some 30-45 minutes test. 

# ## 9. Summary
# 1. There is a large difference between women in technology and men in technology but there is a growing sign of interest among women for technology which is a good sign.<br>
# 2. India and United States are going to have more hiring next years.<br>
# 3. People in technology prefer frameworks like AngularJS, Node.JS and languages like Java, Javascript and Python.<br>
# 4. Online profile can help to increase the chances of getting interviewed by interviewers. So people should focus on creating an impressive online profile to showcase in fromt of Hiring Managers.
# 
# **Note:**
# Always Love Coding. Happy Coding!!
