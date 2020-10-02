#!/usr/bin/env python
# coding: utf-8

# # Human Resources analysis
# 
# 
#     The goal of this notebook is to investigate the factors that make an employee leave his/her company.
# 
# 
# ### Main question to be answered
#     - Which variables contribute the most to an employee leaving the company? 
#     
# ### Outline
#     - Reading data and importing libaries
#     - Basic exploration
#     - Features
#     - Cleaning the Data
#     - General overview by basic plots
#     - Probabilistic analysis 

# ## Reading the Data

# In[ ]:


# Importing libraries

get_ipython().run_line_magic('matplotlib', 'inline')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


sns.set_style('whitegrid')
sns.despine()


# In[ ]:


# hr = pd.read_csv("HR_comma_sep.csv")

hr = pd.read_csv("../input/HR_comma_sep.csv")


# In[ ]:


HR_copy=hr.copy() # A copy of the data set is made for probability analysis


# ## Basic Exploration - Is there any problem with our dataset?

# In[ ]:


hr.head(3)


# In[ ]:


hr.tail(3)


# In[ ]:


hr.info()


# In[ ]:


print('Percent of employees who left: {}'.format(round(hr.left.sum()/hr.shape[0]*100)))


# In[ ]:


hr.describe()


# In[ ]:


hr.isnull().values.any()


# **Findings:**   
# So far, the dataset seems to be clean and not much is needed to make sure all parts are easily analysed. There are no missing values and only sales and salary contain strings. Salary seems like a strange name for what seems to be departments, so that will be changed later. About 1/4th of the employees left, which at first seems like a substantial number. However, it is not known why those employees left the company which makes analyses somewhat more difficult. Next, we will take a more in depth look at the features.

# ## What are the features?
# According to website below, the following are the features:
# - **satisfaction_level**: Level of satisfaction between 0 and 1
# - **last_evaluation**: Their last evaluation between 0 and 1
# - **number_project**: Number of projects (ranging from 2 to 7)
# - **average_montly_hours**: The average work hours per month at workplace (ranging from 96 to 310)
# - **time_spend_company**: The time spend in the company (in years ranging from 2 to 10)
# - **Work_accident**: Whether they have had a work accident
# - **left**: If they left the company (0 = no, 1 = yes)
# - **promotion_last_5years**: Whether they have had a promotion in the last 5 years
# - **sales**: Department (sales, accounting, hr, technical, support, management, IT, product_mng, marketing, RandD)
# - **salary**: The salary of employees (low, medium, high)
# 
# Source: https://www.kaggle.com/ludobenistant/hr-analytics

# Next, we will look at the values and whether some of them need to be changed.

# In[ ]:


print('Unique values of:\n')
print('number_project \n{}\n'.format(hr["number_project"].unique()))
print('salary \n{}\n'.format(hr["salary"].unique()))
print('Work_accident \n{}\n'.format(hr["Work_accident"].unique()))
print('time_spend_company \n{}\n'.format(hr["time_spend_company"].unique()))
print('sales \n{}\n'.format(hr["sales"].unique()))
print('promotion_last_5years \n{}\n'.format(hr["promotion_last_5years"].unique()))
print('average_montly_hours: \n{}\n'.format(hr["average_montly_hours"].unique()))
print('last_evaluation: \n{}\n'.format(hr["last_evaluation"].unique()))
print('satisfaction_level: \n{}\n'.format(hr["satisfaction_level"].unique()))


# **Findings**:   
# 
# A quick inspection of the unique values in each column shows that there are no weird values that should be considered when analysing the data. This does not regard outliers, merely NaN (i.e., missing data), strings etc.  

# ## Cleaning Data
# 
# There are some features that have strings as values. We will change them so they can be more easily viewed and analysed later. Furthermore, the labels of departments and salary is kept so they can be used in plotting.

# In[ ]:


salary_labels = {'low':0,'medium':1,'high':2}
hr['salary'] = hr['salary'].map(salary_labels)


# In[ ]:


hr.rename(columns={'sales': 'department'}, inplace=True)

department_labels = {'sales':0,'accounting':1,'hr':2, 'technical':3, 'support':4,
                     'management':5, 'IT':6, 'product_mng':7, 'marketing':8, 'RandD':9}
hr['department'] = hr['department'].map(department_labels)

hr.loc[hr['satisfaction_level'] < .2, 'satisfaction_bins'] = 0
hr.loc[(hr['satisfaction_level'] >= .2) & (hr['satisfaction_level'] < .4), 'satisfaction_bins'] = 1
hr.loc[(hr['satisfaction_level'] >= .4) & (hr['satisfaction_level'] < .6), 'satisfaction_bins'] = 2
hr.loc[(hr['satisfaction_level'] >= .6) & (hr['satisfaction_level'] < .8), 'satisfaction_bins'] = 3
hr.loc[hr['satisfaction_level'] >= .8, 'satisfaction_bins'] = 4

hr.loc[hr['last_evaluation'] < .2, 'evaluation_bins'] = 0
hr.loc[(hr['last_evaluation'] >= .2) & (hr['last_evaluation'] < .4), 'evaluation_bins'] = 1
hr.loc[(hr['last_evaluation'] >= .4) & (hr['last_evaluation'] < .6), 'evaluation_bins'] = 2
hr.loc[(hr['last_evaluation'] >= .6) & (hr['last_evaluation'] < .8), 'evaluation_bins'] = 3
hr.loc[hr['last_evaluation'] >= .8, 'evaluation_bins'] = 4


# In[ ]:


hr.head()


# ## Basic Plots
# We will start with some basic plots to simply see the distribution of the features. How does everything look? Are there any outliers we should be worried about? etc. 

# In[ ]:


hr.hist(figsize =(14,10.0))
plt.show()


# **Findings:**  
# So far, the most surprising distribution is the very low percentage of employees that were promoted in the last 5 years. Thus, let's explore that first before we go into the feature 'left'. 

# In[ ]:


sns.countplot(x="department", data=hr,palette="muted")
plt.title('Distribution of employess across departments')


# In[ ]:


sns.factorplot(y="promotion_last_5years", x="department", data=hr, kind = "bar", 
               hue = "salary", size = 5, aspect = 2, palette='muted', ci=None)
plt.ylim(0,0.20)
plt.title('Promotion within departments')
plt.xlabel('Department')
plt.ylabel('% of promotion in last 5 years')
plt.xticks(list(department_labels.values()),department_labels.keys(),rotation=75)
plt.show()


# **Findings:**  
# Apparently, in most departments, employees with at least a medium salary are more likely to get promoted. This effect is even greater for employees with a high salary in the accounting and management departments. This suggests that the company is not a place where you'll want to be if you start off with a low salary. However, getting promoted is easier if you already have a high salary.

# ## Correlations

# In[ ]:


corr = hr.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values, annot=True, fmt='.2g')
plt.show()


# **Findings:**   
# It seems that the number of projects and the average monthly hours is reasonably correlated. Furthermore, last evaluation and average monthly hours, and last evaluation and number of projects seems to be correlated. Thus, let's plot those correlations before going into the leaving of employees. 

# In[ ]:


palette = ["#e41a1c", "#377eb8", '#4daf4a','#984ea3','#ff7f00','#a65628','#f781bf']
sns.factorplot(x='number_project',y='average_montly_hours',data=hr,kind='bar', palette=palette)
plt.title('Monthly hours per number of projects')
plt.xlabel('Number of projects')
plt.ylabel('Average monthly hours')
plt.show()


# The correlation that we have seen in the correlation matrix above suggested a correlation between number of projects and average monthly hours. We can clearly see that as the number of projects increases, the number of average monthly hours also increases. This makes a lot of sense, seeing that having more projects results in more work and therefore, more hours spent on that work.  

# In[ ]:


palette = ["#e41a1c", "#377eb8", '#4daf4a','#984ea3','#ff7f00','#a65628','#f781bf']
sns.factorplot(x='evaluation_bins',y='average_montly_hours',data=hr,kind='bar', palette=palette)
plt.title('last evaluation for average monthly hours')
plt.xlabel('Last evaluation')
plt.ylabel('Average monthly hours')
plt.show()


# Similarly, we noticed a correlation between evaluation and average monthly hours. As seen before, it seems that the better the last evaluation, the more hours employees have spent on their work. It may be that because they worked more hours, their evaluation would increase, but that is only a hypothesis.

# In[ ]:


palette = ["#e41a1c", "#377eb8", '#4daf4a','#984ea3','#ff7f00','#a65628','#f781bf']
sns.factorplot(x='evaluation_bins',y='number_project',data=hr,kind='bar', palette=palette)
plt.title('last evaluation per number of projects')
plt.xlabel('Last evaluation')
plt.ylabel('Number of projects')
plt.show()


# For an evaluation of .4 and higher we see that the higher the evaluation the more likely an employee will have more projects to work on. Seeing as how the number of projects and average monthly hours were correlated, this seems to be in line with the previous plots.  

# In[ ]:


sns.factorplot(y="promotion_last_5years", x="number_project", data=hr, kind = "bar", 
               hue = "evaluation_bins", size = 5, aspect = 2, palette='muted', ci=None, legend= True)
plt.xticks([0,1,2,3,4],['very low', 'low', 'medium', 'high', 'very high'],rotation=75)
plt.title('Percentage of promotion for number of projects and last evaluation')
plt.xlabel('Number of projects')
plt.ylabel('% of promotion in last 5 years')
plt.show()


# **Findings:**   
# Interestingly, we see that the last evaluation and number of projects is not dependent on the chances of getting a promotion. Thus, although the number of projects are more likely to get employees a higher evaluation, it seems to not influence the chances of them getting a promotion.

# ##  Which variables contribute the most to an employee leaving the company?  
# 
# Now, for the main question.  Which variables contribute the most to an employee leaving the company?  We have seen that evaluation seem to be based on the number of projects and hours worked, but that there is little to do with getting a promotion.

# In[ ]:


sns.pairplot(hr, hue="left", palette = "husl")


# Although a pair plot creates a large pair of plots, we can immediately see some interesting clusters. If we mainly look at last evaluation, satisfaction level, average monthly hours and time spend at company, we can see that some values are clusters very closely together. However, that is difficult to see in the plot above, so we go into a narrower view using those four features.  

# In[ ]:


g = sns.PairGrid(hr, vars=["last_evaluation", "satisfaction_level","time_spend_company", "average_montly_hours"], 
                 hue='left', palette = "husl")
g = g.map_diag(plt.hist)
g = g.map_upper(plt.scatter)
g = g.map_lower(sns.kdeplot, cmap="Blues_d")

g = g.add_legend()


# In the pair grid above we can see the suggested relations more clearly. Especially for average monthly hours, satisfaction level and last evaluation we can see some clusters that indicate that employees are almost certain to leave. Thus, let's plot those features in more detail.  

# ### Heatmaps

# In[ ]:


hr_pivot = hr.pivot_table(index='last_evaluation', columns='satisfaction_level', 
                          values='left')
sns.heatmap(hr_pivot, xticklabels=20, yticklabels=20, linecolor='white')
plt.title('Satisfaction and last evaluation on leaving')
plt.xlabel('Satisfaction level')
plt.ylabel('Last evalution')
plt.show()


# In[ ]:


hr_pivot = hr.pivot_table(index='average_montly_hours', columns='satisfaction_level', 
                          values='left')
sns.heatmap(hr_pivot, xticklabels=20, yticklabels=20, linecolor='white')
plt.title('Satisfaction and average monthly hours on leaving')
plt.xlabel('Satisfaction level')
plt.ylabel('Average monthly hours')
plt.show()


# In[ ]:


hr_pivot = hr.pivot_table(index='time_spend_company', columns='satisfaction_level', 
                          values='left')
sns.heatmap(hr_pivot, xticklabels=20, yticklabels=20, linecolor='white')
plt.xlabel('Satisfaction level')
plt.ylabel('Time spend at company')
plt.show()


# **Findings:**   
# 
# It seems that the features that we have selected for further exploration are sufficient to warrant some more in-depth plots. Next, we will look at each of the features. 

# ### Time spend at company
# Time spend at the company showed some clusters in the pair plots, so we looked at it more in depth by plotting it versus left, number of projects and department. 

# In[ ]:


sns.barplot(x = 'time_spend_company', y = 'left', data = hr)
plt.xlabel('Time spend at company in years')
plt.ylabel('Mean of left')
plt.title('Time spend at company on whether employees left')
plt.show()


# In[ ]:


sns.factorplot(y="time_spend_company", x="number_project", data=hr, kind = "bar", hue = "left", size = 5, aspect = 2)
plt.xlabel('Number of projects')
plt.ylabel('Time spend at company (years)')
plt.title('Influence of number of projects and time at company on leaving')
plt.show()


# In[ ]:


sns.factorplot(x='department', y='time_spend_company', hue='left',data=hr, ci=None)
plt.xlabel('Department')
plt.ylabel('Time spend at company (years)')
labels = ['sales','accounting','hr', 'technical', 'support',
                     'management', 'IT', 'product_mng', 'marketing', 'RandD']
plt.xticks([0,1,2,3,4,5,6,7,8,9],labels,rotation=75)
plt.title('Influence of department and time spend at company on leaving')
plt.show()


# What we can clearly see is that management is more likely to stay if they spend more time at the company, which is in contrast with the other departments. It might be that employees in the management department make more money and have a higher chance to receive a promotion.  

# ### Satisfaction level and number of projects

# In[ ]:


sns.barplot(x = 'number_project', y = 'left', data = hr)
plt.xlabel('Number of projects')
plt.title('Number of projects on leaving')
plt.ylabel('Chances of leaving')
plt.show()


# In[ ]:


sns.pointplot(hr['satisfaction_bins'],hr['left'])
plt.title('Satisfaction on leaving')
plt.xlabel('Satisfaction level in converted bins')
plt.ylabel('Chances of leaving')
plt.show()


# In[ ]:


sns.pointplot(hr['number_project'],hr['satisfaction_level'], hue=hr['left'])
plt.title('Number of projects and satisfaction on leaving')
plt.xlabel('Number of projects')
plt.ylabel('Mean satisfaction level')
plt.show()


# From these plots, we can see that when an employee must work on more projects, the chances of leaving are gets higher, except for when an employee is assigned for only two projects. In addition, counterintuitive, on average, someone that gave their job satisfaction level a 4 has a higher chance of leaving than those who ranged their job satisfaction level a 3.
# 
# From the last plot, we see how the number of projects is correlated with satisfaction rate for the employees how stayed or left.

# ### General plots on leaving

# In[ ]:


sns.barplot(x = 'salary', y = 'left', data = hr)
plt.xlabel('Salary')
plt.ylabel('Chance of leaving')
plt.title('Salary on leaving')
plt.xticks([0,1,2],['low','medium','high'])
plt.show()


# In[ ]:


sns.barplot(x = 'promotion_last_5years', y = 'left', data = hr)
plt.xlabel('Promotion in last 5 years')
plt.ylabel('Chances of leaving')
plt.title('Promotion on leaving')
plt.show()


# In[ ]:


sns.factorplot(y="evaluation_bins", x="satisfaction_bins", data=hr, kind = "bar", hue = "left", 
               size = 5, aspect = 2, palette='muted', ci=None)
plt.xlabel('Satisfaction level in bins')
plt.ylabel('Mean evaluation')
plt.title('Satisfaction vs. evaluation on leaving')
plt.show()


# From these plots, we see that the chance on leaving gets lower if the salary is higher and when there was a promotion in the past 5 years.

# ## Probability analysis

# Before starting the probability analysis some manipulation on the dataset is necessary, since we will be using, for example, the 'department' column.

# In[ ]:


HR_copy.rename(columns={"sales":"department", "Work_accident":"work_accident", "left":"left_job"},inplace=True)
HR_copy.head(3)


# ### How many employees distributed by department?

# In[ ]:


ax = sns.countplot(x="department", data=HR_copy,palette="muted")


# ### Percentage of employees that left per department

# In[ ]:


g = sns.factorplot(x="department", hue="left_job", kind="count",
                   data=HR_copy,size=10,palette="muted")
g.set_ylabels("Employees/department ")


# In[ ]:


HR_copy['left_job'][HR_copy['department']=='sales'].value_counts(normalize = True)


# In[ ]:


HR_copy.head()


# In[ ]:


# Creating a crosstab view to examine department relationship with the leaves
department_df = pd.crosstab(HR_copy.department, HR_copy.left_job, margins=True, 
            normalize = 'index', rownames = ['Department Names'], colnames = ['Left Job Status'])
department_df = pd.DataFrame(department_df.loc[['accounting', 'hr', 'sales', 'technical', 'support', 'management', 'IT', 'product_mng', 'marketing','RandD'],:])
department_df.columns = ['Still Works', 'Left Job']


# In[ ]:


department_df.round(2)


# The HR department was the one with highest number of employees leaving, 29.09%. And management the one with least, 14.44%.

# ### Satisfaction level X leaving job

# In[ ]:


HR_copy['satisfaction_level'].describe()


# Taking in account the information obtained above (quartiles), we will calculate the probabilities of an employee leaving the job in relation to his/her level of satisfaction.

# In[ ]:


# Creating a new column to see satisfaction levels in categories
HR_copy['satisfaction_level_categorical'] = None
HR_copy.loc[HR_copy['satisfaction_level']<=0.44, 'satisfaction_level_categorical'] = 'Less Than 44%'
HR_copy.loc[(HR_copy['satisfaction_level']>0.44) & (HR_copy['satisfaction_level']<=0.64), 'satisfaction_level_categorical'] = '44% - 64%'
HR_copy.loc[(HR_copy['satisfaction_level']>0.64) & (HR_copy['satisfaction_level']<=0.82), 'satisfaction_level_categorical'] = '65% - 82%'
HR_copy.loc[(HR_copy['satisfaction_level']>0.82), 'satisfaction_level_categorical'] = 'More Than 82%'

# Satisfaction level crosstab view
satisfaction_level_df = pd.crosstab(HR_copy.satisfaction_level_categorical, HR_copy.left_job,  
            normalize = 'index', rownames = ['Satisfaction Levels'],
                                   colnames = ['Left Job Status'])
satisfaction_level_df.columns = ['Still Works', 'Left Job']

# Sorting the rows
satisfaction_level_df = satisfaction_level_df.loc[['Less Than 44%', '44% - 64%', '65% - 82%', 'More Than 82%'], :]


# In[ ]:


satisfaction_level_df.round(2)


# Confirming what was observed at other sections, the employee with lower satisfaction level has greater probability to leave the job. However, it is interesting to observe that it doesn't need to be the highest satisfaction level to have the highest probability to stay. For example, having an evaluation between 0.44 and 0.64 gives a probability of 91.52% while an evaluation above 0.82 gives a probability of staying of 86.83%.
# 
# To go further with the investigation let's consider other factors that should influence the employee in leaving his/her job, as pointed in previous sections.
# 
# Let's consider, for example the last evaluation.

# ### How satisfaction level and last evaluation influences leaving the job.

# #### Last evaluation influence

# In[ ]:


HR_copy['last_evaluation'].describe()


# In[ ]:


# Creating a new column for last evaluation to be able to view that in categories
HR_copy['last_evaluation_categorical'] = None
HR_copy.loc[HR_copy['last_evaluation']<=0.56, 'last_evaluation_categorical'] = 'Less Than 57%'
HR_copy.loc[(HR_copy['last_evaluation']>0.56) & (HR_copy['last_evaluation']<=0.72), 'last_evaluation_categorical'] = '57% - 72%'
HR_copy.loc[(HR_copy['last_evaluation']>0.72) & (HR_copy['last_evaluation']<=0.87), 'last_evaluation_categorical'] = '73% - 87%'
HR_copy.loc[(HR_copy['last_evaluation']>0.87), 'last_evaluation_categorical'] = 'More Than 87%'

# Creating a crosstab view 
last_evaluation_df = pd.crosstab(HR_copy.last_evaluation_categorical, HR_copy.left_job, margins=True, 
            normalize = 'index', rownames = ['Last Evaluation Grades'], colnames = ['Left Job Status'])
last_evaluation_df = pd.DataFrame(last_evaluation_df.loc[['Less Than 57%', '57% - 72%', '73% - 87%', 'More Than 87%', 'All'],:])
last_evaluation_df.columns = ['Still Works', 'Left Job']


# In[ ]:


last_evaluation_df.round(2)


# It seems that last evaluation alone doesn't play a role in someone leaving the job. We can observe for example, that someone with low evaluation (under 0.56) has almost the same probability of leaving the job as someone with a high evaluation above 0.87, respectively, 37,44% and 32.01%.
# 
# Let's see what happens when satisfaction and last evaluation are put together.

# In[ ]:


# Creating a crosstab view
satisfaction_evaluation_df = pd.crosstab([HR_copy.last_evaluation_categorical, HR_copy.satisfaction_level_categorical], HR_copy.left_job,  
            normalize = 'index', rownames = ['Last Evaluation Grades', 'Satisfaction Levels'],
                                   colnames = ['Left Job Status'])

# Sorting the rows
satisfaction_evaluation_df = pd.DataFrame(satisfaction_evaluation_df.loc[[
            ('Less Than 57%','Less Than 44%'),
            ('Less Than 57%','44% - 64%'),
            ('Less Than 57%','65% - 82%'),
            ('Less Than 57%','More Than 82%'),
            ('57% - 72%','Less Than 44%'),
            ('57% - 72%','44% - 64%'),
            ('57% - 72%','65% - 82%'),
            ('57% - 72%','More Than 82%'),
            ('73% - 87%','Less Than 44%'),
            ('73% - 87%','44% - 64%'),
            ('73% - 87%','65% - 82%'),
            ('73% - 87%','More Than 82%'),
            ('More Than 87%','Less Than 44%'),
            ('More Than 87%','44% - 64%'),
            ('More Than 87%','65% - 82%'),
            ('More Than 87%','More Than 82%')],:])
satisfaction_evaluation_df.columns = ['Still Works', 'Left Job']


# In[ ]:


satisfaction_evaluation_df.round(2)


# It is interesting to observe that when putting together satisfaction level and last evaluation, the ones that are in the range 0.44 to 0.64 of satisfaction level have greater probability to stay as high is the evaluation. Above 0.64 in the satisfaction level it seems that high the evaluation, higher the probability of leaving the job.

# ### What is the influence of the time spent in the company?

# In[ ]:


HR_copy['time_spend_company'].describe()


# In[ ]:


# Creating categorical column for time spent in the company
HR_copy['time_spend_company_categorical'] = None
HR_copy.loc[HR_copy['time_spend_company'] < 3, 'time_spend_company_categorical'] = 'Less Than 3'
HR_copy.loc[(HR_copy['time_spend_company'] >= 3) & (HR_copy['time_spend_company'] < 4), 'time_spend_company_categorical'] = '3 - 4'
HR_copy.loc[(HR_copy['time_spend_company'] >= 4) & (HR_copy['time_spend_company'] < 5), 'time_spend_company_categorical'] = '4 - 5'
HR_copy.loc[(HR_copy['time_spend_company'] >= 5) & (HR_copy['time_spend_company'] < 6), 'time_spend_company_categorical'] = '5 - 6'
HR_copy.loc[HR_copy['time_spend_company'] >= 6, 'time_spend_company_categorical'] = 'More Than 6'

# Creating crosstab and sorting the rows
time_spend_company_df = pd.crosstab(HR_copy.time_spend_company_categorical, HR_copy.left_job, margins=True, 
            normalize = 'index', rownames = ['Time Spent in Company'], colnames = ['Left Job Status'])
time_spend_company_df = pd.DataFrame(time_spend_company_df.loc[['Less Than 3', '3 - 4', '4 - 5', '5 - 6', 'More Than 6'],:])
time_spend_company_df.columns = ['Still Works', 'Left Job']


# In[ ]:


time_spend_company_df.round(2)


# More time the employee stays, higher the probability of leaving, at least this is true until 5 years, after that the probability go slightly down.
# 
# Does satisfaction level influence in this probability?

# ### Satisfaction level and years staying in the company? Higher satisfaction will make an employee stay longer?

# In[ ]:


# Creating a crosstab with time spend in company and satisfaction levels
time_spent_satisfaction_df = pd.crosstab([HR_copy.time_spend_company_categorical, HR_copy.satisfaction_level_categorical], HR_copy.left_job,  
            normalize = 'index', rownames = ['Time Spent in Company', 'Satisfaction Levels'],
                                   colnames = ['Left Job Status'])

# Sorting the rows
time_spent_satisfaction_df = time_spent_satisfaction_df.loc[[('Less Than 3','Less Than 44%'),
                                                            ('Less Than 3','44% - 64%'),
                                                            ('Less Than 3','65% - 82%'),
                                                            ('Less Than 3','More Than 82%'),
                                                            ('3 - 4','Less Than 44%'),
                                                            ('3 - 4','44% - 64%'),
                                                            ('3 - 4','65% - 82%'),
                                                            ('3 - 4','More Than 82%'),
                                                            ('4 - 5','Less Than 44%'),
                                                            ('4 - 5','44% - 64%'),
                                                            ('4 - 5','65% - 82%'),
                                                            ('4 - 5','More Than 82%'),
                                                            ('5 - 6','Less Than 44%'),
                                                            ('5 - 6','44% - 64%'),
                                                            ('5 - 6','65% - 82%'),
                                                            ('5 - 6','More Than 82%'),
                                                            ('More Than 6','Less Than 44%'),
                                                            ('More Than 6','44% - 64%'),
                                                            ('More Than 6','65% - 82%'),
                                                            ('More Than 6','More Than 82%')
                                                        ], :]
time_spent_satisfaction_df.columns = ['Still Works', 'Left Job']


# In[ ]:


time_spent_satisfaction_df.round(2)


# Satisfaction level seems to play a role, specially when it is high, but only until 5 years at work, after that, there is not much influnce when considering leaving the job.

# ## Summary: Findings 
# Note: All findings below do not indicate any causation, merely correlation and probabilities. 
# 
# - 23.81 % of the employees left their jobs. The HR department was the one with highest number of employees leaving, 29.09%. And management the one with least, 14.44%.
# - In addition to being the department that has the least number of employees leaving, management seems to be the department where the employees work at the company the longest. 
# - Last evaluation, number of projects and average monthly hours do not seem to influence the chances of getting a promotion
# - Employees with a satisfaction level between 0.44 and 0.64 seem to be more influenced by the last evaluation in relation to stay or leave. The higher the evaluation, the more likely it will be that an employee will stay at the company. 
# - On the other side, employees with a satisfaction level above 0.64 have a higher chance of leaving the job if they have a higher last evaluation. 
# - Employees that have been at the company for 5 years have a high chance of leaving the company. 
# - When analysing satisfaction level and years worked in the company, satisfaction level can have a positive influence on preventing employees from leaving until 5 years spend at the company, but after that it will hot have much influence on preventing the employees from leaving.  
# - The more money an employee makes, the less likely he/she will leave the company
# - Employees are highly unlikely to get a promotion, unless they work in management and make a lot of money
# - Employees who were promoted in the last 5 years have a lower chance of leaving the company
# - **In sum**, there are four features that seem to be most related to employees leaving and are interesting to analyse further:
#     - Time spend at company
#     - Satisfaction level
#     - Last evaluation
#     - Average monthly hours

# ### Difficulties with the data
# 
# The problem with the data is that not much information is available about the employee. Age, contract (fulltime vs. part-time), sex, salary is operated as high, medium or low, no information regarding the company and their product, etc. So, it is difficult to really understand the data seeing as there will be problems generalizing the data for the next assignment. This isn't necessarily a problem, but it should be noted when making generalized predictions about our data in the following analyses.
