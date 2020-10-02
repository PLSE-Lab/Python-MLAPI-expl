#!/usr/bin/env python
# coding: utf-8

# # Import Packages
# 
# Import packages, numpy, maplotlib and seaborn.

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# # Import Stack Overflow Developer Survey 2017
# 
# Explore Stack Overflow Developer Survey 2017,  with respondent number as index column.
# 
# The survey dataset contained results from 51391 respondents.

# In[2]:


surveyresults = pd.read_csv('../input/survey_results_public.csv', index_col='Respondent')

print (surveyresults.shape, surveyresults.columns)


# # Let's Explore Web Developers #
# 
# I want to explore the subset of surveyed Web Developers from the Stack Overflow 2017 Survey
# 
# Web Developers are broadly classified into - Full stack Developers, Back End Developers and Front End Developers. 
# See a udacity blog post by Michael Wales (https://blog.udacity.com/2014/12/front-end-vs-back-end-vs-full-stack-web-developers.html) for more details
# 
# In the dataset, Stack Overflow surveyed:
# 
# 6816 Full stack Web developers   
# 2610 Back-end Web developers     
# 1270 Front-end Web developers  
# 

# In[3]:


surveyresults['WebDeveloperType'].unique()


# In[4]:


surveyresults['WebDeveloperType'].value_counts()


# In[5]:


get_ipython().run_line_magic('matplotlib', 'inline')

webdevcount = pd.value_counts(surveyresults['WebDeveloperType'].values, sort=True)

webdevcount.plot(kind='bar', title='Web Developer Types')
plt.ylabel('Count')
plt.show()


# # There is a lot of missing Data #
# 
# The stack overflow survey dataset is great to work with as it has many data points (respondents). Unfortunately, it also comes with a lot of missing data points and it is not clear why exactly each of these data points are missing.
# 
# 
# For instance, if I were to look at the data regarding languages that developers have worked with, it is not clear if NaN represents :
# 
# 1) a null value because respondents were lazy and did not fill this in, 
# 
# or 
# 
# 2) if respondents did not work with any languages in the past year
# 
# Because of this ambiguity, I have elected to drop data entries with null values throughout the process of exploratory data analysis. Of course, noting that this is a highly imperfect means to an end.
# 
# I will look at the subset of data points where respondents have indicated that they are a web developer (be it front-end, back-end or full stack)
# 
# Of the 10696 web developers surveyed, there is a considerable amount of missing datapoints for:
# - Salary (6703 missing)
# - Frameworks worked with in the past year (5618 missing)
# - Platforms worked with in the past year (5229 missing)
# - Database worked with in the past year (3638)
# - Gender (3169)
# - Languages worked with in the past year (2762)

# In[6]:


surveyresults['WebDeveloperType'].notnull().sum()


# In[7]:


surveyresults[['WebDeveloperType','HaveWorkedLanguage', 'HaveWorkedFramework', 'HaveWorkedDatabase', 'HaveWorkedPlatform', 'YearsCodedJob', 'YearsProgram', 'FormalEducation', 'Gender', 'MajorUndergrad', 'Salary']][surveyresults.WebDeveloperType.notnull()].isnull().sum().sort_values(ascending=False)


# # Programming Languages that Web Developers have Worked With in Past Year#
# 
# Stack Overflow has also surveyed regarding technologies that developers have used in the past year, such as languages, frameworks, databases, platforms & IDES.
# 
# I will start by finding out which  programming languages web developers have been working with in the past year, and how this differs between front-end, back-end and full-stack developers.

# # Number of Languages by Developer Type #
# 
# On average, full stack developers have worked with more langauges over the past year (3.5), compared to back-end (3.0) and front-end developers (2.6).

# In[8]:


webdeveloperlanguages = surveyresults[['WebDeveloperType','HaveWorkedLanguage']].dropna(axis=0, how='any')

webdeveloperlanguages['NumberofLanguages'] = webdeveloperlanguages['HaveWorkedLanguage'].apply(lambda x: len(str(x).split(';')))

webdeveloperlanguages.head()


# In[9]:


webdeveloperlanguages.groupby('WebDeveloperType')['NumberofLanguages'].describe()


# In[10]:


plt.hist(webdeveloperlanguages['NumberofLanguages'], bins=range(1,18), normed=True)
plt.title('Number of Langauges Web Developers (All Types) have worked with in Past Year')
plt.ylabel('Proportion of Web Developers')
plt.xlabel('Number of Languages')
plt.show()


# In[11]:


fig, axes = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True, figsize=(10, 5))

webdeveloperlanguages['NumberofLanguages'].hist(by=webdeveloperlanguages['WebDeveloperType'], bins=range(1,18), normed=True, ax=axes)
plt.suptitle('Number of Langauges Web Developers have worked with in Past Year', x=0.5, y=1.05, ha='center', fontsize='xx-large')
fig.text(0.5, 0.0, 'Number of Languages', ha='center')
fig.text(0.0, 0.5, 'Proportion of Web Developers', va='center', rotation='vertical')

plt.tight_layout()
plt.show()


# In[12]:


sns.boxplot(x='WebDeveloperType', y='NumberofLanguages', data=webdeveloperlanguages)
plt.title('Number of Languages Web Developers have Worked with in Past Year')
plt.xlabel('Web Developer Type')
plt.ylabel('Number of Languages')
plt.xticks([0, 1, 2], ['Full stack', 'Back-end', 'Front-end'], rotation=40)
plt.show()


# # Writing functions for Repetitive Tasks #
# 
# For the columns regarding web developer's preferred technologies (such as language, platform, IDE), the data was in a string format separated by a semicolon (;).
# 
# e.g.: 
# Java; JavaScript; Ruby; SQL
# 
# I will write 3 functions to simplify the analysis of these columns.
# 
# The function webDevExpandedTable wll return cross tablulation of the total number of developers who have engaged with a particular technology, by developer type.

# In[13]:


def webDevExpandedTable(column):
    df = surveyresults[['WebDeveloperType', column]].dropna(axis=0, how='any')
    expanded = df[column].str.replace(' ', '').str.get_dummies(';')
    fulldf = pd.concat([df['WebDeveloperType'], expanded], axis=1).groupby('WebDeveloperType').sum().T.rename_axis(column)
    return fulldf


# However, the raw numbers returned by webDevExpandedTable are difficult to interpret as there as a non-even distribution between Front-end, Back-end and Full-stack developers.
# 
# The function webDevExpandedProportionTable takes into accont the total number of each developer type, and returns the proportion of each developer type who has engaged with a particular technology.
# 
# As developers can enage in more than one technology, the propportions in each column will not sum to 1.

# In[14]:


def webDevExpandedProportionTable(column):
    df = surveyresults[['WebDeveloperType', column]].dropna(axis=0, how='any')
    counts = {'Back-end Web developer': df.WebDeveloperType.value_counts()['Back-end Web developer'],
              'Front-end Web developer': df.WebDeveloperType.value_counts()['Front-end Web developer'],
              'Full stack Web developer': df.WebDeveloperType.value_counts()['Full stack Web developer']
             }  
    
    
    expanded = df[column].str.replace(' ', '').str.get_dummies(';')
    fulldf = pd.concat([df['WebDeveloperType'], expanded], axis=1).groupby('WebDeveloperType').sum().T.rename_axis(column)
   
    for column in list(fulldf.columns):
        fulldf[column] =fulldf[column].apply(lambda x: round((x/counts[column]), 3))
    
    return fulldf


# The function top will display the most popular technologyies by developer type

# In[15]:


def top(df, number):
    top = {}
    rank = np.arange(1, number+1)
    for column in list(df.columns):
        topdf = df[column].sort_values(ascending=False).head(number)
        series = []
        for index, row in topdf.iteritems():
            series.append(str(index) + ' ( ' + str(row) + ' ) ')
        top[column] = series
        
        
    return pd.DataFrame(top, index=rank)


# # Web Developer's Most Popular Languages #

# In[16]:


webDevExpandedTable('HaveWorkedLanguage')


# In[17]:


languages = webDevExpandedProportionTable('HaveWorkedLanguage')
languages


# In[18]:


top(languages, 10)


# In[19]:


selected = list(languages.mean(axis=1).sort_values(ascending=False).head(10).index)

languages.loc[selected].plot(kind='bar')
plt.title('Languages Web Developers Have Worked With in Past Year')
plt.xlabel('Programming Language')
plt.ylabel('Proportion')
plt.legend(title='Web Developer Type')


# Across the board, javascript is by far the most popular language.
# 
# Being a client-side script, Javascript is the bread and butter of front-end. As such nearly all front-end & full-stack developers would have encountered working with javascript.
# 
# The back end's choice of language is more varied. 

# # Web Developer's preferred Frameworks #
# 
# Inspection of this frameworks data reveals some flaws in this dataset.
# 
# Of the frameworks listed in the dataset
# 
# 1) Common web development frameworks are missing.
# - Obvious omissions include: Ruby on Rails, Laravel, Django, Flask, Vue.js, Meteor
# 
# 2) Not all frameworks are web development frameworks
# - Firebase & Cordova are for mobile development
# - Hadoop & Spark are for data

# In[20]:


webDevExpandedTable('HaveWorkedFramework')


# In[21]:


frameworks = webDevExpandedProportionTable('HaveWorkedFramework')
frameworks


# In[22]:


top(frameworks, 5)


# Given how flawed the results in this section appear to be, there is little point in doing further analysis in this section.
# 
# There is the obvious conclusion that frameworks for web development are more popular with web developers. Unfortunately, we are missing important information regarding Ruby on Rails, Django, Flask, Laravel, and thus cannot compare which was the most popular web development framework.

# # Databases Web Developers have Worked with #

# In[23]:


webDevExpandedTable('HaveWorkedDatabase')


# In[24]:


database = webDevExpandedProportionTable('HaveWorkedDatabase')
database


# In[25]:


top(database, 5)


# In[26]:


topdatabase = list(database.mean(axis=1).sort_values(ascending=False).head(5).index)

database.loc[topdatabase].plot(kind='bar')
plt.title('Databases Web Developers Have Worked With in Past Year')
plt.xlabel('Database')
plt.ylabel('Proportion')
plt.legend(title='Web Developer Type')


# It is important to note that with this data, in calculating the proportions, we are only taking data from respondents who have listed at least one Database they had worked with, due to the ambiguity of NaN represents none or if respondents were too lazy to fill it in.
# 
# As such, the proportions reflect that of respondents who have worked with at least one database, rather than the respondent group as a whole.
# 
# Across the board, MySQL appears to be the most popular database type.

# # Platforms Web Developers have Worked with #
# 
# This is another section with results that are difficult to interpret.
# 
# In stack overflow's scheme. The results from this column correspond to the question - 
# Which of the following platforms have you done extensive development work for over the past year, and which do you want to work on over the next year?

# In[27]:


webDevExpandedTable('HaveWorkedPlatform')


# In[28]:


platforms = webDevExpandedProportionTable('HaveWorkedPlatform')
platforms


# In[29]:


top(platforms, 5)


# # IDEs Web Developers have Worked with #

# In[30]:


ide = webDevExpandedProportionTable('IDE')
ide


# In[31]:


top(ide, 10)


# Looking at the top IDEs by web developer type:
# 
# The most popular amongst the front-end developers are text editors & electron based editors: SublimeText, Atom, VS Code. As these tools are sufficient & efficient at accomplishing front-end tasks.
# 
# Apart from the text editors, VisualStudio, Vim & IntelliJ were also popular amongst Full-stack & Back-end Developers
# 

# In[32]:


topide = list(ide.mean(axis=1).sort_values(ascending=False).head(10).index)

ide.loc[topide].plot(kind='bar')
plt.title('IDEs Developers Use')
plt.xlabel('IDE')
plt.ylabel('Proportion')
plt.legend(title='Web Developer Type')


# # Number of Years Web Developer's have Coded on a Job for #
# 
# Interestingly, the stack overflow survey has elected to find out the amount of years developers have been coding / coding on jobs for using 1 year bins.
# 
# For analysis, I decided to map these categorical values into numerical values, to find the average number of years web developers have been coding for.
# 
# Of note is that there was a category '20 or more years', which could represent a broad range of values. However, as the number of values in this range is small, I arbitrarily have mapped this to 20.5.

# In[33]:


def mapyears(column):
    yeardict = {
    'Less than a year': 0.5,
    '1 to 2 years': 1.5,
    '2 to 3 years': 2.5,
    '3 to 4 years': 3.5,
    '4 to 5 years': 4.5,
    '6 to 7 years': 6.5, 
    '7 to 8 years': 7.5,
    '8 to 9 years': 8.5,
    '9 to 10 years': 9.5,
    '10 to 11 years': 10.5,
    '11 to 12 years': 11.5,
    '12 to 13 years': 12.5,
    '13 to 14 years': 13.5,
    '14 to 15 years': 14.5,
    '15 to 16 years': 15.5,
    '16 to 17 years': 16.5,
    '17 to 18 years': 17.5,
    '18 to 19 years': 18.5,
    '19 to 20 years': 19.5,
    '20 or more years': 20.5
    }
    df = surveyresults[['WebDeveloperType', column]].dropna(axis=0, how='any')
    df[column] = df[column].map(yeardict)
    return df


# In[34]:


webDevYearsCode = mapyears('YearsCodedJob')

webDevYearsCode.groupby('WebDeveloperType').describe()


# Front-end developers on average have spent 2 years less coding jobs, compared to full-stack and back end developers.
# 
# In all distributions, there is a left skew (larger proportion of developers who have less than 5 years experience)

# In[35]:


fig, axes = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True, figsize=(10, 5))

webDevYearsCode.hist(by=webdeveloperlanguages['WebDeveloperType'], bins=8, normed=True, ax=axes)
plt.suptitle('Number of Years Web Developers have Coded on Jobs for', x=0.5, y=1.05, ha='center', fontsize='xx-large')
fig.text(0.5, 0.0, 'Number of Years', ha='center')
fig.text(0.0, 0.5, 'Proportion of Web Developers', va='center', rotation='vertical')

plt.tight_layout()
plt.show()


# In[36]:


sns.boxplot(x='WebDeveloperType', y='YearsCodedJob', data=webDevYearsCode)
plt.title('Number of Years Web Developers have Coded on Jobs for')
plt.xlabel('Web Developer Type')
plt.ylabel('Number of Years')
plt.xticks([0, 1, 2], ['Full stack', 'Back-end', 'Front-end'], rotation=40)
plt.show()


# # Number of Years Web Developers Have Coded For #
# The analysis and results of this section are similar to the previous section (Number of Years Web Developers have Coded on Jobs For).

# In[37]:


webDevYearsProgram = mapyears('YearsProgram')

webDevYearsProgram.groupby('WebDeveloperType').describe()


# Again, on average, front-end developers have 2 years less coding experience than back-end & full-stack developers.

# In[38]:


fig, axes = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True, figsize=(10, 5))

webDevYearsProgram.hist(by=webdeveloperlanguages['WebDeveloperType'], bins=8, normed=True, ax=axes)
plt.suptitle('Number of Years Web Developers have Coded for', x=0.5, y=1.05, ha='center', fontsize='xx-large')
fig.text(0.5, 0.0, 'Number of Years', ha='center')
fig.text(0.0, 0.5, 'Proportion of Web Developers', va='center', rotation='vertical')

plt.tight_layout()
plt.show()


# In[39]:


sns.boxplot(x='WebDeveloperType', y='YearsProgram', data=webDevYearsProgram)
plt.title('Number of Years Web Developers have Coded for')
plt.xlabel('Web Developer Type')
plt.ylabel('Number of Years')
plt.xticks([0, 1, 2], ['Full stack', 'Back-end', 'Front-end'], rotation=40)
plt.show()


# # Writing Functions for Categorical Variables #
# I will look at some categorical variables (highest level of formal education & bachelor degree).
# 
# Unlike the previous variables for preferred technology, respondents were only allowed to select one option for highest level of formal education or bachelor degree.
# 
# I will perform cross tabulations on these variables & web developer type.
# 
# The functions
# - crosscategory will show a cross tabulation of the  number of respondents & web developer type (raw number)
# - crossProp will show a cross tabulation of the proportion of respondents by web developer type
# 

# In[40]:


def crossCategory(column):
    df = surveyresults[['WebDeveloperType', column]].dropna(axis=0, how='any')
    cross = pd.crosstab(index=df['WebDeveloperType'],  columns=df[column], margins=True)
    return cross.T


# In[41]:


def crossProp(column):
    df = surveyresults[['WebDeveloperType', column]].dropna(axis=0, how='any')
    cross = pd.crosstab(index=df['WebDeveloperType'],  columns=df[column], margins=True)
    return cross.div(cross['All'],axis=0).round(3).T.sort_values(by='All', ascending=False).drop('All', 0)


# In[42]:


def crossCategorybyCat(column):
    df = surveyresults[['WebDeveloperType', column]].dropna(axis=0, how='any')
    cross = pd.crosstab(index=df['WebDeveloperType'],  columns=df[column], margins=True)
    cross = cross/cross.loc['All']
    return cross.T


# # Highest Level Of Formal Education Achieved by Web Developers #

# In[43]:


crossCategory('FormalEducation')


# In[44]:


crossProp('FormalEducation')


# About 52% of Web Developers in all categories have a Bachelor's degree as their highest level of education.
# 
# The next most common qualification was a Master's degree. With a slightly higher proportion in back-end (26.2%) and full-stack (21.5%), compared to front-end (17.7%)

# In[45]:


education = crossProp('FormalEducation').drop('All', 1)
education.rename(columns={}, inplace=True)

education.plot(kind='bar')
plt.title('Highest Level of Formal Education Web Developers Have Achieved')
plt.ylabel('Proportion')
plt.xlabel('Education Level')
edlabels = list(education.index)
edlabels[2] = 'College/University without Degree'
plt.xticks(np.arange(0,9), edlabels)
plt.legend(title='Web Developer Type')


# # Majors of Web Developers who have an Undergraduate Degree #

# In[46]:


crossCategory('MajorUndergrad')


# In[47]:


crossProp('MajorUndergrad')


# Amongst all web developers, a computer science or software engineering degree was the most common degree.
# 
# Of developers who had a degree. A higher proportion of full-stack (52.7%) & backend (56.0%) developers had a  computer science or software engineering degree compared to front-end developers (37.2%).
# 
# The most common degrees were Computer/IT related:
# - Computer Science / software engineering
# - Computer programming / Web development
# - Computer engineering / Electronics engineering
# - IT / networking / system administration
# 
# 
# 
# 

# In[48]:


undergrad = crossProp('MajorUndergrad').drop('All', 1).head(10)

undergrad.plot(kind='bar')
plt.title('Undergraduate Degrees attained by Web Developers')
plt.ylabel('Proportion')
plt.xlabel('Undergraduate Degree')
plt.legend(title='Web Developer Type')


# Given that the 4 most popular degrees were Computer/IT related, I decided to examine of those who had a bachelor's degree, what was the total proportion of those who had a IT/Computer related major.
# 
# - 80.6% of back-end developers had a Computer/IT related major
# - 77.3% of full-stack developers Computer/IT related major
# - 68.2% of front-end develoeprs had a Computer/IT related major
# 

# In[49]:


pd.DataFrame(crossProp('MajorUndergrad').loc['Computer science or software engineering':'Information technology, networking, or system administration', :].sum())


# # Web Developer's Career Satisfaction #

# In[50]:


webDevSatisfaction = surveyresults[['WebDeveloperType','CareerSatisfaction']].dropna(axis=0, how='any')

webDevSatisfaction.groupby('WebDeveloperType').describe()


# There was high career satisfaction amongst all web developers.
# 
# With a mean satisfaction of 7.3 for back-end developers, 7.4 for front-end & full-stack developers.

# In[51]:


sns.boxplot(x='WebDeveloperType', y='CareerSatisfaction', data=webDevSatisfaction)
plt.title('Web Developer Career Satisfcation')
plt.xlabel('Web Developer Type')
plt.ylabel('Career Satisfaction')
plt.xticks([0, 1, 2], ['Full stack', 'Back-end', 'Front-end'], rotation=40)
plt.show()


# In[52]:


fig, axes = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True, figsize=(10, 5))

webDevSatisfaction.hist(by=webDevSatisfaction['WebDeveloperType'], bins=range(0,10), normed=True, ax=axes)
plt.suptitle('Career Satisfaction of Web Developers', x=0.5, y=1.05, ha='center', fontsize='xx-large')
fig.text(0.5, 0.0, 'Career Satisfaction', ha='center')
fig.text(0.0, 0.5, 'Proportion of Web Developers', va='center', rotation='vertical')

plt.tight_layout()
plt.show()


# # Web Developer's Salary #

# In[53]:


webDevSalary = surveyresults[['WebDeveloperType','Salary']][surveyresults['Salary'] > 0].dropna(axis=0, how='any')

webDevSalary.groupby('WebDeveloperType').describe()


# In[54]:


print(len(webDevSalary[webDevSalary.Salary < 1]), len(webDevSalary[webDevSalary.Salary < 1000]), len(webDevSalary))


# Web Developer's salaries had a highly suspicious amount of what appears to be dirty data, with
# - 10 entries reporting to earn less than \$10 a year
# - 119 entries reporting to earn less than \$1000 year
# 
# out of a total 3991 entries
# 
# Noting that our summary statistics are sullied by a large amount of left-shifted data, which is difficult to mark as outliers --
# On average, full stack developers had a higher annual income (\$55260), compared to fornt-end (\$51338) and back-end developers (\$51103)

# In[55]:


sns.boxplot(x='WebDeveloperType', y='Salary', data=webDevSalary)
plt.title('Web Developer Salary')
plt.xlabel('Web Developer Type')
plt.ylabel('Salary')
plt.xticks([0, 1, 2], ['Full stack', 'Back-end', 'Front-end'], rotation=40)
plt.show()


# In[56]:


fig, axes = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True, figsize=(10, 5))

webDevSalary.hist(by=webDevSalary['WebDeveloperType'], bins=10, normed=True, ax=axes)
plt.suptitle('Web Developer Salary', x=0.5, y=1.05, ha='center', fontsize='xx-large')
fig.text(0.5, 0.0, 'Salary', ha='center')
fig.text(0.0, 0.5, 'Proportion of Web Developers', va='center', rotation='vertical')

plt.tight_layout()
plt.show()


# I wanted to look at the 'Salary' & 'Years Coded for Job' variables, with the intention of plotting a regression line through it, however there was too much dirty data at this point to accurately plot our regression line. Though it is obvious, that there is a correlation between years of coding & salary.

# In[57]:


multidata = surveyresults[['WebDeveloperType','Salary', 'YearsCodedJob']].dropna(axis=0, how='any')

multidata = multidata[multidata.Salary > 0]

multidata['YearsCodedJob'] = multidata['YearsCodedJob'].map({
    'Less than a year': 0.5,
    '1 to 2 years': 1.5,
    '2 to 3 years': 2.5,
    '3 to 4 years': 3.5,
    '4 to 5 years': 4.5,
    '6 to 7 years': 6.5, 
    '7 to 8 years': 7.5,
    '8 to 9 years': 8.5,
    '9 to 10 years': 9.5,
    '10 to 11 years': 10.5,
    '11 to 12 years': 11.5,
    '12 to 13 years': 12.5,
    '13 to 14 years': 13.5,
    '14 to 15 years': 14.5,
    '15 to 16 years': 15.5,
    '16 to 17 years': 16.5,
    '17 to 18 years': 17.5,
    '18 to 19 years': 18.5,
    '19 to 20 years': 19.5,
    '20 or more years': 20
    })


sns.lmplot(y='Salary', x='YearsCodedJob', data=multidata, hue='WebDeveloperType', x_jitter=0.7, fit_reg=False)
plt.xlabel('Number of Years Developer has Coded for Job')
plt.title('Relationship between Salary and Years of Coding for Job')


# What a dirty looking graph.

# # Gender Representation amongst Web Developers #

# It appears that in the Stack Overflow survey, respondents were able to check multiple options of:
# - 'Male'
# - 'Female'
# - 'Other'
# - 'Transgender'
# - 'Gender non-conforming'
# 
# such that I could not easily cross Tabulate Gender as a categorical variable, as some had identified as multiple categories.
# 
# - Male; Female; Transgender; Gender non-conforming; Other (all categories)
# - Male; Female (Both Male and Female, but not Transgender)
# 

# In[58]:


crossProp('Gender')


# I decided to map options of those who gave single entries ('Male', 'Female', 'Other', 'Transgender', 'Gender non-conforming') to their corresponding entry, whilst grouping those who ticked multiple options into 'Multiple'

# In[59]:


webDevByGender = surveyresults[['WebDeveloperType', 'Gender']][pd.notnull(surveyresults['WebDeveloperType'])]

gender = ['Male', 'Female', 'Other', 'Transgender', 'Gender non-conforming', np.NaN]

webDevByGender['Gender'] = webDevByGender['Gender'].apply(lambda i: i if i in gender else 'Multiple')


webDevGenderFullTab = pd.crosstab(index=webDevByGender['WebDeveloperType'],  columns=webDevByGender['Gender'], margins=True)

webDevGenderFullTab.T.sort_values(by='All', ascending=False)


# In[60]:


allgender = webDevGenderFullTab.div(webDevGenderFullTab["All"],axis=0).T.sort_values(by='All', ascending=False).drop('All', 0)
allgender


# Across all web developer types, Males formed the majoirty.
# - 90% in back-end & full-stack
# - 80% in front-end
# 
# Front-end developers had the largest proportion of females (17.6%), compared to full-stack (8.4%) and back-end (8.3%)

# In[61]:


allgenders = allgender.drop('All', 1)

allgenders.plot(kind='bar')
plt.title('Gender Ratio by Web Developer Type')
plt.ylabel('Proportion')
plt.xlabel('Gender')
plt.legend(title='Web Developer Type')


# Further analysis of the two largest gender groups (male & female), to get a better picture of male to female ratios.

# In[62]:


webDevMaleFemale = webDevByGender[(webDevByGender.Gender=='Male') | (webDevByGender.Gender=='Female')]

webDevMaleFemaleCrossTab = pd.crosstab(index=webDevMaleFemale['WebDeveloperType'],  columns=webDevMaleFemale['Gender'], margins=True)

webDevMaleFemaleCrossTab


# In[63]:


webDevMaleFemaleProp = webDevMaleFemaleCrossTab.div(webDevMaleFemaleCrossTab["All"],axis=0)
webDevMaleFemaleProp 


# In[64]:


gender = webDevMaleFemaleProp.sort_values(by='All', ascending=False).drop('All', 0).drop('All', 1)

gender.plot(kind='bar')
plt.title('Male / Female Ratio by Web Developer Type')
plt.ylabel('Proportion')
plt.xlabel('Web Developer Type')
plt.legend(title='Web Developer Type')


# Within each gender group, a larger proportion of females are front-end Developers (22.2%) compared to males (10.8%).
# 
# Full-stack developers still make up the largest proportion, regardless of gender.

# In[65]:


webDevMaleFemaleCrossTab/webDevMaleFemaleCrossTab.loc['All']


# In[66]:


genderDivison = webDevMaleFemaleCrossTab/webDevMaleFemaleCrossTab.loc['All']

genderDivison.drop('All', 0).T.plot(kind='bar')
plt.title('Proportion of Web Developer Type by Gender')
plt.ylabel('Proportion')
plt.legend(title='Web Developer Type')

