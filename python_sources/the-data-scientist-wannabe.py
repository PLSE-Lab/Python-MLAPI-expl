#!/usr/bin/env python
# coding: utf-8

# In this notebook I'll explore the Kaggle survey results trying to tell the story of the data engineer wannabe data scientist. I will compare data scientists and data engineers jobs, tools they use, and opinions and biases they have to raise awareness of the non-sexy profession of data engineering.

# Let's first import the relevant libraries for the analysis.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings 
warnings.simplefilter('ignore')


# Now let's read the subset of data that we'll be looking at. I scoped the analysis to users that responded to be full-time employees from these professions: Data Scientists, software developers, data analysts, statisticians, software developers, programmers and bussiness analysts.

# In[ ]:


# data engineers vs data scientists
mc_responses = pd.read_csv('../input/multipleChoiceResponses.csv', encoding = "ISO-8859-1")
mc_responses_subset = mc_responses[(mc_responses['CurrentJobTitleSelect'].isin(['DBA/Database Engineer', 'Data Scientist','Software Developer/Software Engineer','Data Analyst','Business Analyst','Programmer','Computer Scientist','Statistician']))
                                    & (mc_responses['EmploymentStatus'] == 'Employed full-time')]


# To start with the analysis, let's get familiarized to what data engineers do at their jobs, what tools they use, what challenges they face and how they compare against data scientists.

# In[ ]:


jobfunction = mc_responses_subset[['CurrentJobTitleSelect','JobFunctionSelect']].groupby(['CurrentJobTitleSelect','JobFunctionSelect'])['CurrentJobTitleSelect'].count()
jobfunction_ds = jobfunction.loc['Data Scientist'].sort_values(ascending = False)[:5]
jobfunction_de = jobfunction.loc['DBA/Database Engineer'].sort_values(ascending = False)[:5]

fig, axs = plt.subplots(1,2, figsize=(25,10))
axs[0].barh(range(len(jobfunction_de)),jobfunction_de.values,tick_label = jobfunction_de.index)
axs[0].invert_yaxis()
axs[0].set_title('Data Engineer Function')
axs[1].barh(range(len(jobfunction_ds)),jobfunction_ds.values,tick_label = jobfunction_ds.index)
axs[1].invert_yaxis()
axs[1].set_title('Data Scientist Function')
plt.show()


# We can observe that the main focus of the data engineer relies on Building and running the data infrastructure that the business uses for storing, analysing and operationalizing the data. Leaving analysis as a second priority and machine learning barely touched. On the other hand the data scientists have analysis as a first thing in their plate, and machine learning second top priority. One would say that the data engineer jobs and data scientist are complementary, where data engineers focus on the dirty job of keeping the data flowing in an organized manner and the data scientists on making that data useful.

# In[ ]:


worktools = pd.DataFrame(mc_responses_subset['WorkToolsSelect'].str.split(',', expand = True).stack()                          .reset_index(level = 1, drop = True)).rename(columns = {0:'Tools'})
worktools = pd.DataFrame(mc_responses['CurrentJobTitleSelect']).join(worktools, how = 'inner')
worktools = worktools.groupby(['CurrentJobTitleSelect', 'Tools'])['CurrentJobTitleSelect'].count()
worktools_ds = worktools.loc['Data Scientist'].sort_values(ascending = False)[:5]
worktools_de = worktools.loc['DBA/Database Engineer'].sort_values(ascending = False)[:5]

fig, axs = plt.subplots(1,2, figsize=(25,10))
axs[1].barh(range(len(worktools_ds)),worktools_ds.values,tick_label = worktools_ds.index)
axs[1].invert_yaxis()
axs[1].set_title('Data Scientist tools')
axs[0].barh(range(len(worktools_de)),worktools_de.values,tick_label = worktools_de.index)
axs[0].invert_yaxis()
axs[0].set_title('Data Engineer tools')
plt.show()


# In terms of tools the 2 proffessions use for their daily jobs, SQL, Python and R come first for both, with SQL being the most important tool for the data engineers while Python and R for the data scientists.

# In[ ]:


time_spent = mc_responses_subset[['CurrentJobTitleSelect','TimeGatheringData','TimeModelBuilding', 'TimeProduction', 'TimeVisualizing', 'TimeFindingInsights', 'TimeOtherSelect']]
time_spent = time_spent.groupby(['CurrentJobTitleSelect']).mean()
timespent_de = time_spent.loc['DBA/Database Engineer']
timespent_ds = time_spent.loc['Data Scientist']

plt.figure(figsize=(12, 5))
plt.bar(range(len(timespent_ds)), timespent_ds, tick_label = timespent_ds.index, alpha = 0.5, label = 'Data Scientist')
plt.bar(range(len(timespent_de)), timespent_de, tick_label = timespent_de.index, alpha = 0.5, label = 'Data Engineer')
plt.title('Percentage of time spent on each stage of the data-pipeline')
plt.legend()
plt.show()


# As to how data engineers spend their time in comparison to data scientists, both seem to spend most of their work on gathering the data, thing which makes sense for the data engineer, but probably it shouldn't be in the scope of the data scientist. Next in the stage of model building, data scientists spend logically more time in comparison, though I don't understand why the data engineers have a high score here as well. Something that surprises is that Data engineers spend more time visualizing in comparison than the data scientist, but less finding insights, probably they are building visualizations so that the business analyst can figure out the insight.

# In[ ]:


challenges = pd.DataFrame(mc_responses_subset['WorkChallengesSelect'].str.split(',', expand = True).stack()                          .reset_index(level = 1, drop = True)).rename(columns = {0:'Challenge'})
challenges = pd.DataFrame(mc_responses['CurrentJobTitleSelect']).join(challenges, how = 'inner')
challenges = challenges.groupby(['CurrentJobTitleSelect', 'Challenge'])['CurrentJobTitleSelect'].count()
challenges_ds = challenges.loc['Data Scientist'].sort_values(ascending = False)[:5]
challenges_de = challenges.loc['DBA/Database Engineer'].sort_values(ascending = False)[:5]

fig, axs = plt.subplots(1,2, figsize=(25,10))
axs[1].barh(range(len(challenges_ds)),challenges_ds.values,tick_label = challenges_ds.index)
axs[1].invert_yaxis()
axs[1].set_title('Data Scientists Challenges')
axs[0].barh(range(len(challenges_de)),challenges_de.values,tick_label = challenges_de.index)
axs[0].invert_yaxis()
axs[0].set_title('Data Engineers Challenges')
plt.show()


# When it comes to challenges at work, data scientists number one concern is dirty data (we can relate this data point with the previously shown amount of time they spend gathering the data), but when you look at the data engineer it revolves around company politics, lack of management and lack of financial support to set up a data science team. 
# 
# The fact that data scientists number one challenge may it mean that data engineers are not doing their job correctly? Or maybe the companies where data scientists work don't have a proper data engineering team? 
# 
# As to the data engineers main concern, the lack of investment in data science could mean they want to become the data scientists their company are not supporting?

# To check if data engineers who answered this survey are indeed trying to move careers, we can check job satisfaction and career switch intention.

# In[ ]:


mc_responses_subset['JobSatisfaction'][mc_responses_subset['JobSatisfaction'] == '1 - Highly Dissatisfied'] = 1
mc_responses_subset['JobSatisfaction'][mc_responses_subset['JobSatisfaction'] == '10 - Highly Satisfied'] = 10
mc_responses_subset['JobSatisfaction'][mc_responses_subset['JobSatisfaction'] == 'I prefer not to share'] = np.nan
mc_responses_subset['JobSatisfaction'] = mc_responses_subset['JobSatisfaction'].astype(float)

job_satisfaction = mc_responses_subset.groupby(['CurrentJobTitleSelect'])['JobSatisfaction'].mean().sort_values()

ax = plt.subplot()
ax.barh(range(len(job_satisfaction)), job_satisfaction.values,tick_label = job_satisfaction.index)
ax.set_title('Job Satisfaction by profession')
plt.show()


# Job satisfaction would validate the hypothesis that data engineers that answered this survey are not happy about their jobs in comparison to other roles. Programmers are the only ones less happy with an average of less than 6 in satisfaction. Who can blame the poor data engineer, who struggles making order out of chaotic data while seeing how data scientists and business analysts have the fun of extracting the insight?

# In[ ]:


n = mc_responses_subset['CurrentJobTitleSelect'].value_counts()
career_switch = mc_responses_subset['CurrentJobTitleSelect'][(mc_responses_subset['CareerSwitcher'] == 'Yes')].value_counts()/n*100
career_switch = career_switch.sort_values()

ax = plt.subplot()
ax.barh(range(len(career_switch)), career_switch.values,tick_label = career_switch.index)
ax.set_title('Career switch into data science intention')
plt.show()


# When checking which are the professionals that are trying to switch into data science, we can see that there is a good correlation with those unhappy about their jobs, with programmers on top of the list with 35% of switching intention. The list goes on not precisely in the same order, but clearly those who are happiest have the least intention to switch. More than 20% of data engineers are looking for an opportunity in the data science field.

# But do data engineers have what it takes to become data scientists? Let's start figuring out what they think about it and then what their opinions reflect.

# In[ ]:


n_datascience_belief = mc_responses_subset['CurrentJobTitleSelect'][mc_responses_subset['DataScienceIdentitySelect'].isin(['Yes','No'])].value_counts()
datascience_belief = mc_responses_subset['CurrentJobTitleSelect'][mc_responses_subset['DataScienceIdentitySelect'] == 'Yes'].value_counts()/n_datascience_belief*100
datascience_belief = datascience_belief.sort_values()
ax = plt.subplot()
ax.barh(range(len(datascience_belief)), datascience_belief.values,tick_label = datascience_belief.index)
ax.set_title('How much each profession believes they are data scientists')
plt.show()


# When compared to some professions, Data engineers have average humbleness towards thinking of themselves as data scientists. Around 35% of them think they are data scientists already, humble in comparision with statistician, of whom, more than 70% think of themselves as data scientists, but ahead of the unhappy programmers.

# In[ ]:


n_stats_importance = mc_responses_subset['CurrentJobTitleSelect'][~mc_responses_subset['JobSkillImportanceStats'].isnull()].value_counts()
stats_importance = mc_responses_subset['CurrentJobTitleSelect'][mc_responses_subset['JobSkillImportanceStats'] == 'Necessary'].value_counts()/n_stats_importance*100
stats_importance = stats_importance.sort_values()
ax = plt.subplot()
ax.barh(range(len(stats_importance)), stats_importance.values,tick_label = stats_importance.index)
ax.set_title('How important is Advanced Statistics skill to land a job in data science')
plt.show()


# When we evaluate their opinion, the data engineers don't seem to rank Advanced Statistics as necessary as much as the data scientists do. This may be related to the fact that they don't know much about stats? Statisticians of course rank their own knowledge as very necessary, but surprisingly data scientists rank advanced statistics even more important than stasticians themselves. Maybe because they came into the field with not enough knowledge and had to struggle?

# In[ ]:


n_sql_importance = mc_responses_subset['CurrentJobTitleSelect'][~mc_responses_subset['JobSkillImportanceSQL'].isnull()].value_counts()
sql_importance = mc_responses_subset['CurrentJobTitleSelect'][mc_responses_subset['JobSkillImportanceSQL'] == 'Necessary'].value_counts()/n_sql_importance*100
sql_importance = sql_importance.sort_values()
ax = plt.subplot()
ax.barh(range(len(sql_importance)), sql_importance.values,tick_label = sql_importance.index)
ax.set_title('How important is SQL skill to land a job in data science')
plt.show()


# When looking at SQL, the primary tool of a Data Engineer, naturally data engineers rank it high in necessary knowledge to land a data science position, but surprisingly again, data scientists rank it even higher in importance. 

# In[ ]:


lang = mc_responses_subset[['CurrentJobTitleSelect','LanguageRecommendationSelect']].groupby(['CurrentJobTitleSelect','LanguageRecommendationSelect'])['CurrentJobTitleSelect'].count()
lang_ds = lang.loc['Data Scientist'].sort_values(ascending = False)[:5]
lang_de = lang.loc['DBA/Database Engineer'].sort_values(ascending = False)[:5]

fig, axs = plt.subplots(1,2, figsize=(25,10))
axs[0].barh(range(len(lang_ds)),lang_ds.values,tick_label = lang_ds.index)
axs[0].invert_yaxis()
axs[0].set_title('Recommended language for data science by Data Scientists')
axs[1].barh(range(len(lang_de)),lang_de.values,tick_label = lang_de.index)
axs[1].invert_yaxis()
axs[1].set_title('Recommended language for data science by Data Engineers')
plt.show()


# Regarding which language data engineers recommend vs data scientists, Python, R and SQL lead the pack, with SQL having more important of course for the SQL wizards data engineers.

# To conclude, let's analyse what Data Engineers and Data Scientist are planning to do next year.

# In[ ]:


ml_tool_next = mc_responses_subset[['CurrentJobTitleSelect','MLToolNextYearSelect']].groupby(['CurrentJobTitleSelect','MLToolNextYearSelect'])['CurrentJobTitleSelect'].count()
ml_tool_next_ds = ml_tool_next.loc['Data Scientist'].sort_values(ascending = False)[:5]
ml_tool_next_de = ml_tool_next.loc['DBA/Database Engineer'].sort_values(ascending = False)[:5]

fig, axs = plt.subplots(1,2, figsize=(25,10))
axs[0].barh(range(len(ml_tool_next_ds)),ml_tool_next_ds.values,tick_label = ml_tool_next_ds.index)
axs[0].invert_yaxis()
axs[0].set_title('Tool data scientists are planning to learn next year')
axs[1].barh(range(len(ml_tool_next_de)),ml_tool_next_de.values,tick_label = ml_tool_next_de.index)
axs[1].invert_yaxis()
axs[1].set_title('Tool data engineers are planning to learn next year')
plt.show()


# Tensor Flow is the trendy tool for everyone it seems, Python is there for both, and we can see that Data Engineers want to catch up in R probably to improve the statistics knowledge they don't currently have.
