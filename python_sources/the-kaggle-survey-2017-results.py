#!/usr/bin/env python
# coding: utf-8

# # Welcome to Kaggle Survey 2017 results.
# > ** Description of the Survey: ** For the first time, Kaggle conducted an industry-wide survey to establish a comprehensive view of the state of data science and machine learning. The survey received over 16,000 responses and we learned a ton about who is working with data, what's happening at the cutting edge of machine learning across industries, and how new data scientists can best break into the field.
# 
# > **About this notebook: **In this notebook, I will be analysing Kaggle Survey for finding meaningful information in the data scientist and machine learning industries.
# 
# ** Table of Contents **
# > 1. [Import necessary dependencies and Load data](#load_data)
# > 2. [Employment](#employment_section)
# > 3. [Learning](#learning_section)
# > 4. [Daily spent time](#data_sci_daily_life)
# > 5. [Salary](#salary_section)
# > 6. [Data Collect source](#data_co_section)
# > 7. [Trending Tools, MLMethod in Next Year](#trending_section)
# > 8. [Job](#job_section)
# > 9. [At work, Data Scientist's tools, database model, team, etc.](#at_work_section)
# > 10. [Knowledge](#know_section)
# > 11. [Language](#language_section)

# <a id='load_data'></a>
# ## 1. Import necessary dependencies and Load data

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import seaborn as sns
from IPython.display import display, Image
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df_schema = pd.read_csv('../input/schema.csv')
display(df_schema.head())


# In[ ]:


df_conversion_rates = pd.read_csv('../input/conversionRates.csv')
display(df_conversion_rates.head(10))


# In[ ]:


df_multi_choice = pd.read_csv('../input/multipleChoiceResponses.csv',encoding='latin-1', low_memory=False)
df_multi_choice.head()


# In[ ]:


print(df_multi_choice.shape)


# In[ ]:


popular_country = list(df_multi_choice['Country'].value_counts().head(5).index)
print(popular_country)


# <a id='employment_section'></a>
# ## 2. Employment
# > In this section we will be analysing the employment section in data science and machine learning industries.
# > 1. [Employment Status](#employment_status)
# > 2. [Employment Status in different country](#employment_status_different_country)
# > 3. [Popular Job](#popular_job)
# > 4. [Who are currently thinking for switching career into Data Science?](#career_switcher)
# > 5. [How often data scientist work remotely?](#remote_work)

# <a id='employment_status'></a>
# > ### 1. Employment Status
# > Question asked during survey: What's your current employment status?

# In[ ]:


color = sns.color_palette()
# get value counts of EmploymentStatus column
employment_status = df_multi_choice.EmploymentStatus.value_counts()
# delare plt figure for plotting
plt.figure(figsize=(10, 8))
# seaborn barplot 
sns.barplot(y=employment_status.index, 
            x=employment_status.values,
            color=color[1])
# add a suptitle
plt.suptitle("Employment Status in Data science and Machine Learning Industries", fontsize=14)
# add xlabel
plt.xlabel('Employment Status Count', fontsize=12)
# add ylabel
plt.ylabel('Employment Status', fontsize=12)
# finally show the plot
plt.show()


# > **Takeaway: ** Most of the participant of the survey are Employed Full time and many of them looking for work.

# <a id='employment_status_different_country'></a>
# > ### 2. Employment Status in different country
# > Let's compare employment status with different country.

# In[ ]:


# get data of popular country
df_employment_popular_country = df_multi_choice[df_multi_choice.Country.isin(popular_country)]
# declare plt figure for plotting
plt.figure(figsize=(10, 8))
plt.title('Employment Status in different country', fontsize=14)
# plot countplot
sns.countplot(x="Country", hue="EmploymentStatus", data=df_employment_popular_country)
plt.ylabel('Employment Status Count')
plt.show()


# > **Takeaway: **It seems like the percentage of Employed full-time is also high in the different country. *** And the percentage of "Not employed, but looking for work" is higher in India compare to other countries. ***

# <a id='popular_job'></a>
# > ### 3. Popular Job
# > Let's see the popular job among participant.
# 
# > Question asked during survey: ***Select the option that's most similar to your current job/professional title***

# In[ ]:


# get the value_counts of CurrentJobTitleSelect column
df_job_title = df_multi_choice[df_multi_choice.CurrentJobTitleSelect.notnull()]["CurrentJobTitleSelect"].value_counts()
plt.figure(figsize=(8, 6))
plt.title('Popular Job')
sns.barplot(y=df_job_title.index, x= df_job_title)
plt.xlabel('Current Job Title Count')
plt.show()


# > **Takeaway:** Data Scientist and Software Developer are the most popular job title.

# <a id='career_switcher'></a>
# > ### 4. Who are currently thinking for switching career into Data Science?
# > Question asked during survey: ***Are you actively looking to switch careers to data science?***
#  
# > Let's find out who want switch career.

# In[ ]:


df_career_switcher = df_multi_choice[df_multi_choice.CareerSwitcher.notnull()]
df_career_switcher = df_career_switcher.loc[df_career_switcher['CareerSwitcher'] == 'Yes']
plt.figure(figsize=(10, 6))
plt.title('Career switcher in different Job field', fontsize=14)
sns.countplot(y="CurrentJobTitleSelect", hue="CareerSwitcher", data=df_career_switcher)
plt.xticks(rotation=90)
plt.xlabel('Career switcher Count')
plt.ylabel('Current Job Title')
plt.show()


# > **Takeaway: ** Wow! Software Developer, Business Analyst, Programmer and Engineer are more likly to switch their career into data science.

# <a id='remote_work'></a>
# > ### 5. How often data scientist work remotely?
# > Question asked during survey: ***How often do you work remotely?***
# 
# > Let's find out how often data scientist work remotely

# In[ ]:


df_remote_work = df_multi_choice['RemoteWork'].value_counts()
plt.figure(figsize=(8, 6))
plt.title('Data Scientist work remotely', fontsize=14)
sns.barplot(y=df_remote_work.index, x= df_remote_work)
plt.xlabel('')
plt.show()


# > **Takeway:** Hmm! It looks like many data scientist work remotely sometimes.

# <a id='learning_section'></a>
# > ## 3. Learning
# > In this section we will be analysing how people are learning data science what is the most popular platform, etc.
# > 1. [Which country's people are learning data science most](#learning_most)
# > 2. [Which platform is popular for learning data science](#popular_learning_platform)
# > 3. [Usefulness of different learning platform](#usefulness_learning_platform)
# > 4. [Popular Blogs, Podcast and Newsletter](#popular_blog_podcast_newsletter)
# > 5. [University Importance](#university_importance)

# <a id='learning_most'></a>
# > ### 1. Which country's people are learning data science most
# > Question asked during survey: ***Are you currently focused on learning data science skills either formally or informally?***
# 
# > Let's find out which country's people are learning data science most

# In[ ]:


df_learning_popular_country = df_multi_choice[df_multi_choice.Country.isin(popular_country)]
df_learning_popular_country = df_learning_popular_country[df_learning_popular_country.LearningDataScience.notnull()]
plt.figure(figsize=(13, 8))
plt.title('Focused on learning Data Science in different country', fontsize=14)
sns.countplot(x="Country", hue="LearningDataScience", data=df_learning_popular_country)
plt.ylabel('Learning Data science')
plt.show()


# > **Takeaway: **Wow! It looks like India's people are learning data science most.

# <a id='popular_learning_platform'></a>
# > ### 2. Which platform is popular for learning data science
# > Question asked during survey: *** What platforms & resources have you used to continue learning data science skills?***
# 
# > Let's find out what is the most popular learning platform

# In[ ]:


# learning platform select
df_learning_platform = df_multi_choice[df_multi_choice.LearningPlatformSelect.notnull()]['LearningPlatformSelect']
df_learning_platform = df_learning_platform.astype('str').apply(lambda x: x.split(','), 1)
flat_list = [item for sublist in df_learning_platform for item in sublist]
d = pd.DataFrame(flat_list)[0].value_counts()

plt.figure(figsize=(10, 6))
plt.title('Popular learning platform', fontsize=14)
sns.barplot(x=d, y=d.index)
plt.xlabel('Learning Platform Popularity')
plt.ylabel('Learning Platform Name')
plt.show()


# > **Takeaway: **Wow! Kaggle, Online Courses, Stackoverflow and Youtube are the most popular learning platform. 

# <a id='usefulness_learning_platform'></a>
# > ### 3. Usefulness of different learning platform
# > Question asked during survey: ***How useful did you find these platforms & resources for learning data science skills?***
# 
# > Let's find out the usefulness of different learning platform

# In[ ]:


filter_col = [col for col in df_multi_choice if col.startswith('LearningPlatformUsefulness')]
df_learning_platform_usefulness = df_multi_choice[filter_col]
df_learning_platform_usefulness = df_learning_platform_usefulness.rename(columns=lambda x: x.replace('LearningPlatformUsefulness', ''))
plt.figure(figsize=(12, 10))
plt.title('Usefulness of Different learning platform', fontsize=14)
sns.countplot(x="variable", hue="value", data=pd.melt(df_learning_platform_usefulness))
plt.xticks(rotation=90)
plt.xlabel('Learning Platform Name')
plt.ylabel('Learning Platform Usefulness')
plt.show()


# > **Takeaway: **It looks like Kaggle and Courses are very useful for learning data science.

# <a id='popular_blog_podcast_newsletter'></a>
# > ### 4. Popular Blogs, Podcast and Newsletter
# > Question asked during survey: ***What are your top 3 favorite data science blogs/podcasts/newsletters?***
# 
# > Let's find out which is the most popular Blogs/Podcasts/Newsletters

# In[ ]:


df_popular_bpn = df_multi_choice[df_multi_choice.BlogsPodcastsNewslettersSelect.notnull()]['BlogsPodcastsNewslettersSelect']
df_popular_bpn = df_popular_bpn.astype('str').apply(lambda x: x.split(','), 1)
flat_list = [item for sublist in df_popular_bpn for item in sublist]
d = pd.DataFrame(flat_list)[0].value_counts()

plt.figure(figsize=(8, 6))
plt.suptitle('Popular Blogs/Podcasts/Newsletters', fontsize=14)
sns.barplot(x=d, y=d.index)
plt.xlabel('')
plt.ylabel('Blogs/Podcasts/Newsletters Name')
plt.show()


# > **Takeaway: ** Wow! KDnuggets Blog and R Bloggers Blog Aggregator are popular among data scientists

# <a id='university_importance'></a>
# >### 5. University Importance
# > Question asked during survey: ***How important was your formal education or degree to your career success analyzing data?***

# In[ ]:


df_uni_imp = df_multi_choice['UniversityImportance'].value_counts()
plt.figure(figsize=(8, 6))
plt.title("University Importance", fontsize=14)
sns.barplot(y=df_uni_imp.index, x= df_uni_imp)
plt.xlabel('')
plt.show()


# > **Takeaway:** It looks like University is very important

# <a id='data_sci_daily_life'></a>
# >##  4. Data Scientists Daily Life
# > In this section we will be analysing data scientists daily life.
# > 1. [Spent time for Job Hunting](#spent_time_job_hunting)
# > 2. [How much time Data scientists spent for Job Hunting in different country](#spent_time_job_hunting_different_country)
# > 3. [Daily Work Challenges](#daily_work_challenges) 
# > 4. [Daily Work Challenges Frequency](#daily_work_challenges_frequency) 

# <a id='spent_time_job_hunting'></a>
# > ### 1. Spent time for Job Hunting
# > ***Question: How many hours per week have you typically spend looking for a data science job?***

# In[ ]:


# get the 'JobHuntTime' column from our data without null values
df_hunt_time = df_multi_choice[df_multi_choice.JobHuntTime.notnull()]["JobHuntTime"]
# let's count each unique time and their occurrences. 
df_hunt_time = df_hunt_time.value_counts()
# let's build a matplotlib figure for plotting
plt.figure(figsize=(8, 6))
plt.title("Time spent per week for job hunting")
# bar plot using seaborn barplot function given y the title and x to value
sns.barplot(y=df_hunt_time.index, x = df_hunt_time)
# rotate the x axis label to 90 degrees 
plt.ylabel('Job Hunting Time Frame')
plt.xlabel('Number of occurrences of each time frame')
plt.xticks(rotation=90)

# finally show the bar plot
plt.show()


# > It seems like the majority of data scientists doesn't spend time on Job Hunting and a large group spent 1-2 hours per week for Job Hunting. Le'ts see how data scientists from different country spent their time for Job Hunting.

# <a id='spent_time_job_hunting_different_country'></a>
# > ### 2. How much time Data scientists spent for Job Hunting in different country

# In[ ]:


con_df = pd.DataFrame(df_multi_choice['Country'].value_counts())
top_country = con_df.head(5).index
df_top_country = df_multi_choice.loc[df_multi_choice['Country'].isin(top_country)]

plt.figure(figsize=(8, 6))
plt.title('Time spent per week for job hunting in different country', fontsize=14)
sns.countplot(x='JobHuntTime', hue='Country', data=df_top_country)
plt.xlabel('Job hunting time frame')
plt.ylabel('Popular time frame in different country')
plt.show()


# > It looks like India's data scientists spent 1-2 hours per week for job hunting and United States data scientist spent 6-10 hours per week for job hunting

# <a id='daily_work_challenges'></a>
# > ### 3. Daily Work Challenges
# > ***Question: At work, which barriers or challenges have you faced this past year?***

# In[ ]:


# get WorkChallengesSelect column without null values
df_working_challenges = df_multi_choice[df_multi_choice.WorkChallengesSelect.notnull()]['WorkChallengesSelect']
# split at ","
df_working_challenges = df_working_challenges.astype('str').apply(lambda x: x.split(','), 1)
# keep only the unique item
flat_list = [item for sublist in df_working_challenges for item in sublist]
# value counts
d = pd.DataFrame(flat_list)[0].value_counts()
# declare the figure
plt.figure(figsize=(10, 8))
plt.title("Challenges faced during work", fontsize=14)
# plot the graph
sns.barplot(x=d, y=d.index)
plt.ylabel('Challenges')
plt.xlabel('')
plt.show()


# > It's look data scientist spent most of their time handling dirty data

# <a id='daily_work_challenges_frequency'></a>
# > ### 4. Daily Work Challenges Frequency

# In[ ]:


# work challenges frequency
filter_col = [col for col in df_multi_choice if col.startswith('WorkChallengeFrequency')]
df_work_challenge_frequency = df_multi_choice[filter_col]
df_work_challenge_frequency = df_work_challenge_frequency.rename(columns=lambda x: x.replace('WorkChallengeFrequency', ''))
plt.figure(figsize=(12, 10))
plt.title('Frequency of different challenges at work')
sns.countplot(x="variable", hue="value", data=pd.melt(df_work_challenge_frequency))
plt.ylabel('Frequency')
plt.xlabel('Daily work challenges')
plt.xticks(rotation=90)
plt.show()


# <a id='salary_section'></a>
# > ## 5. Salary
# > #### In this section, we will be analyzing data scientists salary like what is the average salary, gender diversity in salary etc. 
# > 1. [What is the common salary range among data scientist?](#common_salary_range) 
# > 2. [Gender diversity in salary](#gender_diversity_salary) 
# > 3. [Country diversity in salary](#country_diversity_salary)

# In[ ]:


# replace CompensationCurrency column value with actual exchangeRate of df_conversion_rates dataframe
df_multi_choice['CompensationCurrency'] = df_multi_choice['CompensationCurrency'].map(df_conversion_rates.set_index('originCountry')['exchangeRate'])
# drop row where df_multi_choice['CompensationCurrency'] column is null
df_compension = df_multi_choice.dropna(axis=0, how='all', subset=['CompensationCurrency'])
# some people put "-" sign before their salary, so we drop it
df_compension = df_compension[df_compension['CompensationAmount'].str.contains("-")==False]
# replace "," with ""
df_compension['CompensationAmount'] = df_compension['CompensationAmount'].str.replace(',', '')
# change df_multi_choice['CompensationAmount'] column datatype to float
df_compension['CompensationAmount'] = df_compension['CompensationAmount'].astype(float)

# multiply CompensationAmount column and CompensationCurrency column and create a new column to hold that value names "SalaryAmount"
df_compension['SalaryAmount'] = df_compension['CompensationAmount']*df_compension['CompensationCurrency']
# convert SalaryAmount column to 'int' datatype
df_compension['SalaryAmount'] = df_compension['SalaryAmount'].astype(int)


# * <a id='common_salary_range'></a>
# > ### 1. What is the common salary range among data scientist? 
# > ***Question: What is your current total yearly compensation (salary + bonus)? ***

# In[ ]:


# create salary range for plotting
bins = [0, 1000,2500, 5000, 10000, 50000, 100000, 150000, 200000, 250000, 300000]
df_salary_amount_value_counts = pd.cut(df_compension['SalaryAmount'], bins).value_counts()
# let's build a matplotlib figure for plotting
plt.figure(figsize=(8, 6))
plt.title("Salary of Data Scientists", fontsize=14)
# bar plot using seaborn barplot function given y the title and x to value
sns.barplot(y=df_salary_amount_value_counts.index, x = df_salary_amount_value_counts)
# rotate the x axis label to 90 degrees 
plt.ylabel('Salary range in dollar')
plt.xlabel('Number of occurrences of each salary range')
plt.xticks(rotation=90)

# finally show the bar plot
plt.show()


# > It looks like most of the data scientists salary range is 100000 to 500000

# <a id='gender_diversity_salary'></a>
# > ### 2. Gender diversity in salary 

# In[ ]:


# get salary corresponding to gender and keep only Female and Male
df_gender_salary = df_compension['SalaryAmount'].groupby(df_compension['GenderSelect']).mean()[['Female', 'Male']]
# pyplot figure for plotting
plt.figure(figsize=(8, 8))
plt.title("Gender diversity in Salary")
# plot the diversity
df_gender_salary.plot(kind='bar', color=[['darkorange', 'red']])
plt.xlabel('Gender Name')
plt.ylabel('Salary')
plt.show()


# > It looks like the salary of a male is twice than female!

# <a id='country_diversity_salary'></a>
# > ### 3. Country diversity in salary

# In[ ]:


# get salary corresponding to country and keep popular country
df_country_salary = df_compension['SalaryAmount'].groupby(df_compension['Country']).mean()[['United States', 'India', 'Other', 'Russia', "People 's Republic of China", 'Brazil', 'Germany', 'France', 'Canada']]
plt.figure(figsize=(8, 6))
plt.title("Country diversity in salary")
# plot the diversity
df_country_salary.plot(kind='bar', color=[['r', 'g', 'blue', 'purple', 'hotpink', 'orange', 'chocolate', 'skyblue', 'tomato']])
plt.xlabel('Country Name')
plt.ylabel('Salary')
plt.show()


# > It looks like US and China's data scientists salary is higher compare to other coutry

# <a id='data_co_section'></a>
# > ## 6. Data Collect Source
# > In this section we will be analysing how data scientist collect their data

# In[ ]:


df_dataset = df_multi_choice[df_multi_choice.PublicDatasetsSelect.notnull()]['PublicDatasetsSelect']
df_dataset = df_dataset.astype('str').apply(lambda x: x.split(','), 1)
flat_list = [item for sublist in df_dataset for item in sublist]
d = pd.DataFrame(flat_list)[0].value_counts()
plt.figure(figsize=(6, 6))
plt.title('Popular places to find public datasets to practice data science skills', fontsize=14)
sns.barplot(x=d, y=d.index)
plt.xlabel('')
plt.show()


# > Wow! It looks like most of data scientist collect data from Dataset aggregator/platform

# <a id='trending_section'></a>
# > ## 7. Trending Tools, MLMethod in Next Year
# > 1. [Popular Machine Learning Tools Next Year](#popular_machine_learning_tools_next_year)
# > 2. [Next Year Popular tools in different profession](#next_year_pop_tools_diff_profession)
# > 3. [Popular Machine Learning Method Next Year](#popular_machine_learning_method_next_year)

# <a id='popular_machine_learning_tools_next_year'></a>
# > ### 1. Popular Machine Learning Tools Next Year
# > Question asked during survey: ***Which tool or technology are you most excited about learning in the next year?***

# In[ ]:


# what is most popular mltoolsnextyear
df_ml_tools_next_year = df_multi_choice[df_multi_choice.MLToolNextYearSelect
                               .notnull()]["MLToolNextYearSelect"].value_counts().head(20)
plt.figure(figsize=(8, 6))
plt.title('Popular ML tools in next year', fontsize=14)
sns.barplot(y=df_ml_tools_next_year.index, x= df_ml_tools_next_year)
plt.ylabel('ML tools name')
plt.xlabel('ML tools popularity')
plt.show()


# > It looks like Tensorflow is the most popular tool for next year

# <a id='next_year_pop_tools_diff_profession'></a>
# > ### 2. Next Year Popular tools in different profession

# In[ ]:


df_ml_tools_next_year = df_multi_choice[df_multi_choice.MLToolNextYearSelect
                               .notnull()]
df_ml_tools_next_yearrr = df_ml_tools_next_year['MLToolNextYearSelect'].value_counts().head(5).index
pop = df_ml_tools_next_yearrr.get_values().tolist()
# pop = ', '.join("'{0}'".format(w) for w in pop)

df_ml_tools_next_year = df_ml_tools_next_year[df_ml_tools_next_year['MLToolNextYearSelect'].isin(pop)]
plt.figure(figsize=(10, 6))
plt.title('Next year popular ML tools in different job field', fontsize=14)
sns.countplot(x="CurrentJobTitleSelect", hue='MLToolNextYearSelect', data=df_ml_tools_next_year)
plt.xticks(rotation=90)
plt.xlabel('Job field name')
plt.ylabel('Popularity of ML Tools')
plt.show()


# > It's look like everybody is talking about Tensorflow.

# <a id='popular_machine_learning_method_next_year'></a>
# > ### 3. Popular Machine Learning Method Next Year
# > Question asked during survey: ***Which ML/DS method are you most excited about learning in the next year?***

# In[ ]:


# ml method next year
df_ml_method_next_year = df_multi_choice['MLMethodNextYearSelect'].value_counts()
plt.figure(figsize=(8, 6))
plt.title('Popular ML method in next year', fontsize=14)
sns.barplot(y=df_ml_method_next_year.index, x= df_ml_method_next_year)
plt.ylabel('ML methods name')
plt.xlabel('ML methods popularity')
plt.show()


# > Wow! Deep learning!

# <a id='job_section'></a>
# >## 8. Job
# > 1. [Different Job Factor and their Importance](#different_job_factor)
# > 2. [What medium people use for finding job?](#medium_finding_jo)
# > 3. [Job Satisfaction](#job_satisfaction) 
# > 4. [Salary Change](#salary_change)

# <a id='different_job_factor'></a>
# > ### 1. Different Job Factor and their Importance

# In[ ]:


filter_col = [col for col in df_multi_choice if col.startswith('JobFactor')]
df_factor = df_multi_choice[filter_col]
df_factor = df_factor.rename(columns=lambda x: x.replace('JobFactor', ''))
plt.figure(figsize=(12, 10))
plt.title('Importance of different job factor', fontsize=14)
sns.countplot(x="variable", hue="value", data=pd.melt(df_factor))
plt.xlabel('JobFactor')
plt.ylabel('Job Factor Importance')
plt.xticks(rotation=90)
plt.show()


# <a id='medium_finding_jo'></a>
# > ### 2. What medium people use for finding job?

# In[ ]:


df_job_find_resource = df_multi_choice['JobSearchResource'].value_counts()
plt.figure(figsize=(8, 6))
plt.title('Popular platform for finding job', fontsize=14)
sns.barplot(y=df_job_find_resource.index, x= df_job_find_resource)
plt.ylabel('Platform name')
plt.xlabel('Platform popularity')
plt.show()


# > Many people use company's website or job listing page for Job search

# <a id='job_satisfaction'></a>
# > ### 3. Job Satisfaction 

# In[ ]:


df_job_satisfaction = df_multi_choice['JobSatisfaction'].value_counts()
plt.figure(figsize=(8, 6))
plt.title('Job satisfaction', fontsize=14)
sns.barplot(y=df_job_satisfaction.index, x= df_job_satisfaction)
plt.xlabel('')
plt.show()


# > Yeah! It looks like many people are satisfy with their job

# <a id='salary_change'></a>
# > ### 4. Salary Change
# > ***Question: How has your salary/compensation changed in the past 3 years?***

# In[ ]:


df_salary_change = df_multi_choice['SalaryChange'].value_counts()
plt.figure(figsize=(7, 6))
plt.title('Salary changed in the past 3 years', fontsize=14)
sns.barplot(y=df_salary_change.index, x= df_salary_change)
plt.show()


# > It looks like many data scientists salary increased 20% in past 3 years.

# <a id='at_work_section'></a>
# > ## 9. At work, Data Scientist's tools, database model, team, etc.
# > 1. [What tools data scientist use for Code Sharing?](#tools_code_sharing)
# > 2. [What tools data scientist use for DataSourcing?](#tools_data_sourcing)
# > 3. [What data storage models data scientist use most?](#data_storage_models)
# > 4. [In organization, where data scientist team sit?](#data_sci_team_sit)
# > 5. [Internal vs External resources used in data scientist team](#inte_exte_resou_data_team)
# > 6. [What proportion of your analytics projects incorporate data visualization?](#propor_visual)
# > 7. [Time spent on project in different task](#time_spent_dif_task)
# > 8. [Data science method at work](#data_scien_method)
# > 9. [Data scientist analytics tools, technology, languages for work](#data_tools_tech_lang)
# > 10. [Popular Machine Learning Algorithom](#popu_ml_algo)
# > 11. [Typical size of data set used to training model](#data_size)
# > 12. [Models go into production](#model_production)
# > 12. [Work Data Type](#work_data_type)
# > 13. [Hardware for working](#hardware)
# > 14. [Primary role in Job](#primary_fun_role)

# <a id='tools_code_sharing'></a>
# > ### 1. What tools data scientist use for Code Sharing?
# > ***Question: At work, which tools do you use to share code?***

# In[ ]:


df_work_tool_code_sharing = df_multi_choice[df_multi_choice.WorkCodeSharing.notnull()]['WorkCodeSharing']
df_work_tool_code_sharing = df_work_tool_code_sharing.astype('str').apply(lambda x: x.split(','), 1)
flat_list = [item for sublist in df_work_tool_code_sharing for item in sublist]
df_work_tool_code_sharing = pd.DataFrame(flat_list)[0].value_counts()
plt.figure(figsize=(8, 6))
plt.title('Popular tools for sharing code', fontsize=14)
sns.barplot(x=df_work_tool_code_sharing, y=df_work_tool_code_sharing.index)
plt.ylabel('Tools name')
plt.xlabel('Tools popularity')
plt.show()


# > As expected, Git is most popular tools for code sharing

# * <a id='tools_data_sourcing'></a>
# > ### 2. What tools data scientist use for DataSourcing?
# > ***Question: At work, which tools do you use to share source data?***

# In[ ]:


df_work_tool_data_sourcing = df_multi_choice[df_multi_choice.WorkDataSourcing.notnull()]['WorkDataSourcing']
df_work_tool_data_sourcing = df_work_tool_data_sourcing.astype('str').apply(lambda x: x.split(','), 1)
flat_list = [item for sublist in df_work_tool_data_sourcing for item in sublist]
df_work_tool_data_sourcing = pd.DataFrame(flat_list)[0].value_counts().head(10)
plt.figure(figsize=(8, 6))
plt.title('Popular tools for sharing source data', fontsize=14)
sns.barplot(x=df_work_tool_data_sourcing, y=df_work_tool_data_sourcing.index)
plt.ylabel('Tools name')
plt.xlabel('Tools popularity')
plt.show()


# > It looks like S3, Dropbox, Google Drive and Slack are popular for Share Source data

# <a id='data_storage_models'></a>
# > ### 3. What data storage models data scientist use most?
# > ***Question: At work, which of these data storage models do you typically use?***

# In[ ]:


df_work_data_storage = df_multi_choice[df_multi_choice.WorkDataStorage.notnull()]['WorkDataStorage']
df_work_data_storage = df_work_data_storage.astype('str').apply(lambda x: x.replace('),', ')//').split('//'), 1)
flat_list = [item for sublist in df_work_data_storage for item in sublist]
df_work_data_storage = pd.DataFrame(flat_list)[0].value_counts()
plt.figure(figsize=(7, 6))
plt.title('Popular data storage models', fontsize=14)
sns.barplot(x=df_work_data_storage, y=df_work_data_storage.index)
plt.ylabel('Data storage models name')
plt.xlabel('Data storage models popularity')
plt.show()


# > It looks like most of the people use CSV, JSON, XML, MySQL for their data storage models. 

# <a id='data_sci_team_sit'></a>
# > ### 4. In organization, where data scientist team sit?
# > ***Question: At work, where does the data scientist team sit within the organization?***

# In[ ]:


df_ds_team_sit = df_multi_choice['WorkMLTeamSeatSelect'].value_counts()
plt.figure(figsize=(8, 6))
plt.title('Data scientist sit within the organization', fontsize=14)
sns.barplot(y=df_ds_team_sit.index, x= df_ds_team_sit)
plt.xlabel('')
plt.show()


# > Wow! In most of the organization, data scientist team is Standalone.

# <a id='inte_exte_resou_data_team'></a>
# > ### 5. Internal vs External resources used in data scientist team
# > ***Question: At work, to what degree does your team use internal versus external resources for data science projects?***

# In[ ]:


df_internal_external_tools = df_multi_choice['WorkInternalVsExternalTools'].value_counts()
plt.figure(figsize=(8, 6))
plt.title('Internal vs external resources used in data science projects', fontsize=14)
sns.barplot(y=df_internal_external_tools.index, x= df_internal_external_tools)
plt.xlabel('')
plt.show()


# > It looks like data science team use internal resources more than external resources

# <a id='propor_visual'></a>
# > ### 6. What proportion of your analytics projects incorporate data visualization?
# ***Question: At work, what proportion of your analytics projects incorporate data visualization?***

# In[ ]:


df_work_data_vis = df_multi_choice['WorkDataVisualizations'].value_counts()
plt.figure(figsize=(8, 6))
plt.title('Percentage of data visualization in analytics projects', fontsize=14)
sns.barplot(y=df_work_data_vis.index, x= df_work_data_vis)
plt.xlabel('')
plt.show()


# > It looks like data scientist's 10-25% of projects are about visualization

# <a id='time_spent_dif_task'></a>
# > ### 7. Time spent on project in different task 
# > ***Question: At work, on average, what percentage of your time is devoted to: ***

# In[ ]:


filter_col = [col for col in df_multi_choice if col.startswith('Time')]
df_time_in_project = df_multi_choice[filter_col]
df_time_in_project = df_time_in_project.rename(columns=lambda x: x.replace('Time', ''))
plt.figure(figsize=(12, 10))
plt.title('Percentage of time in different task in a project', fontsize=14)
df_time_in_project.mean().plot(kind='bar', color=[['red', 'green', 'blue', 'purple', 'hotpink', 'orange']])
plt.xlabel('Task name')
plt.show()


# > Hmm! It looks like data scientist spent most of his time on Gathering data in work place.

# <a id='data_scien_method'></a>
# >### 8. Data science method at work

# In[ ]:


df_work_ml_model = df_multi_choice[df_multi_choice.WorkMethodsSelect.notnull()]['WorkMethodsSelect']
df_work_ml_model = df_work_ml_model.astype('str').apply(lambda x: x.split(','), 1)
flat_list = [item for sublist in df_work_ml_model for item in sublist]
df_work_ml_model = pd.DataFrame(flat_list)[0].value_counts().head(10)
plt.figure(figsize=(8, 6))
plt.title('Popular work method', fontsize=14)
sns.barplot(x=df_work_ml_model, y=df_work_ml_model.index)
plt.xlabel('')
plt.show()


# <a id='data_tools_tech_lang'></a>
# > ### 9. Data scientist analytics tools, technology, languages for work
# > ***Question: For work, which data science/analytics tools, technologies, and languages have you used in the past year?***

# In[ ]:


df_work_tool_tech = df_multi_choice[df_multi_choice.WorkToolsSelect.notnull()]['WorkToolsSelect']
df_work_tool_tech = df_work_tool_tech.astype('str').apply(lambda x: x.split(','), 1)
flat_list = [item for sublist in df_work_tool_tech for item in sublist]
df_work_tool_tech = pd.DataFrame(flat_list)[0].value_counts().head(10)
plt.figure(figsize=(8, 6))
plt.title('Popular analytics tools, technologies and languages', fontsize=14)
sns.barplot(x=df_work_tool_tech, y=df_work_tool_tech.index)
plt.xlabel('')
plt.show()


# <a id='popu_ml_algo'></a>
# > ### 10. Popular Machine Learning Algorithom
# > ***Question: At work, which algorithms/analytic methods do you typically use?***

# In[ ]:


df_work_algorithom = df_multi_choice[df_multi_choice.WorkAlgorithmsSelect.notnull()]['WorkAlgorithmsSelect']
df_work_algorithom = df_work_algorithom.astype('str').apply(lambda x: x.split(','), 1)
flat_list = [item for sublist in df_work_algorithom for item in sublist]
df_work_algorithom = pd.DataFrame(flat_list)[0].value_counts()
plt.figure(figsize=(8, 6))
plt.suptitle('Popular analytic methods')
sns.barplot(x=df_work_algorithom, y=df_work_algorithom.index)
plt.xlabel('')
plt.show()


# > Hmm! Logistic Regression is popular among data scientist

# <a id='data_size'></a>
# > ### 11. Typical size of data set used to training model 
# > ***Question: Of the models you've trained at work, what is the typical size of datasets used?***

# In[ ]:


df_data_size = df_multi_choice['WorkDatasetSize'].value_counts()
plt.figure(figsize=(8, 6))
plt.suptitle("Typical datasets size for training model")
sns.barplot(y=df_data_size.index, x= df_data_size)
plt.xlabel('')
plt.show()


# <a id='model_production'></a>
# >### 12. Models go into production
# > ***Question: At work, how often do the models you build get put into production?***

# In[ ]:


df_work_production_frequ = df_multi_choice['WorkProductionFrequency'].value_counts()
plt.figure(figsize=(8, 6))
plt.title("Frequency of model get put into production at work", fontsize=14)
sns.barplot(y=df_work_production_frequ.index, x= df_work_production_frequ)
plt.show()


# <a id='work_data_type'></a>
# >### 13. Work Data Type
# >***Question: At work, which kind of data do you typically work with?***

# In[ ]:


df_work_data_type = df_multi_choice[df_multi_choice.WorkDataTypeSelect.notnull()]['WorkDataTypeSelect']
df_work_data_type = df_work_data_type.astype('str').apply(lambda x: x.split(','), 1)
flat_list = [item for sublist in df_work_data_type for item in sublist]
df_work_data_type = pd.DataFrame(flat_list)[0].value_counts()
plt.figure(figsize=(8, 6))
plt.title('Popular data type', fontsize=14)
sns.barplot(x=df_work_data_type, y=df_work_data_type.index)
plt.xlabel('')
plt.show()


# <a id='hardware'></a>
# >### 14. Hardware for working
# > ***Question: At work, which computing hardware do you use for ML/DS projects?***

# In[ ]:


df_work_hardware = df_multi_choice[df_multi_choice.WorkHardwareSelect.notnull()]['WorkHardwareSelect']
df_work_hardware = df_work_hardware.astype('str').apply(lambda x: x.replace('),', ')//').split('//'), 1)
flat_list = [item for sublist in df_work_hardware for item in sublist]
df_work_hardware = pd.DataFrame(flat_list)[0].value_counts().head(15)
plt.figure(figsize=(6, 8))
plt.suptitle('Hardware used by Data Scientist')
sns.barplot(x=df_work_hardware, y=df_work_hardware.index)
plt.xlabel('')
plt.show()


# <a id='primary_fun_role'></a>
# >### 15. Primary role in Job
# >***Question: What is the primary function of your role?***

# In[ ]:


df_work_role = df_multi_choice[df_multi_choice.JobFunctionSelect.notnull()]['JobFunctionSelect']
df_work_role = df_work_role.astype('str').apply(lambda x: x.split(','), 1)
flat_list = [item for sublist in df_work_role for item in sublist]
df_work_role = pd.DataFrame(flat_list)[0].value_counts()
plt.figure(figsize=(8, 6))
plt.title('Primary role in job', fontsize=14)
sns.barplot(x=df_work_role, y=df_work_role.index)
plt.xlabel('')
plt.show()


# <a id='know_section'></a>
# > ## 10. Knowledge
# > 1. Algorithom Understanding Label
# 
# >***Question: At which level do you understand the mathematics behind the algorithms you use at work?***

# In[ ]:


df_algo_under_label = df_multi_choice['AlgorithmUnderstandingLevel'].value_counts()
plt.figure(figsize=(8, 6))
plt.title('Understanding of mathematics behind the algorithom', fontsize=14)
sns.barplot(y=df_algo_under_label.index, x= df_algo_under_label)
plt.xlabel('')
plt.show()


# > It looks like most of the people can explain the math behin an algorithom

# <a id='language_section'></a>
# > ## 11. Language

# In[ ]:


# language recommendation
df_language = df_multi_choice['LanguageRecommendationSelect'].value_counts()
plt.figure(figsize=(8, 6))
plt.title('Popular language', fontsize=14)
sns.barplot(y=df_language.index, x= df_language)
plt.xlabel('Language popularity')
plt.show()


# In[ ]:


# language recommendation by profession
df_language = df_multi_choice[df_multi_choice.LanguageRecommendationSelect
                               .notnull()]
df_ml_tools_next_yearrr = df_ml_tools_next_year['LanguageRecommendationSelect'].value_counts().head(3).index
pop = df_ml_tools_next_yearrr.get_values().tolist()
# pop = ', '.join("'{0}'".format(w) for w in pop)

df_language = df_language[df_language['LanguageRecommendationSelect'].isin(pop)]
plt.figure(figsize=(10, 6))
plt.title('Popular language in different profession', fontsize=14)
sns.countplot(x="CurrentJobTitleSelect", hue='LanguageRecommendationSelect', data=df_language)
plt.xticks(rotation=90)
plt.xlabel('Job title')
plt.ylabel('Language popularity')
plt.show()


# > ## Conclusion
# > Thanks for reading the notebook. Hope you enjoy it. Feel free to comment for any mistake or suggestions. Also don't forget to Upvote and share.
