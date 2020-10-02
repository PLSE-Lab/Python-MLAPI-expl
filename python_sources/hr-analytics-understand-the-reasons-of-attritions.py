#!/usr/bin/env python
# coding: utf-8

# ## IBM HR Analytics Employee Attrition & Performance
# The objective is here to figure out major factors which are impacting the attrition of employees. These factors can be the distance from home to office, Salary, Worklife balance etc.
# 
# Data is available for 1600 employees. This Explanatory Data Analysis will figure out the actual reasons behind attrition.

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns
# the commonly used alias for seaborn is sns

# set a seaborn style of your taste
sns.set_style("whitegrid")


# ### Import and Understand Dataset

# In[ ]:


df=pd.read_csv('../input/WA_Fn-UseC_-HR-Employee-Attrition.csv')


# In[ ]:


df.head()


# In[ ]:


df.tail()


# In[ ]:


df.shape


# In[ ]:


print(df.info())


# ### Data Quality Checks

# In[ ]:


# Check Missing Values
round(100*(df.isnull().sum()/len(df.index)), 2)


# - No missing value in data

# In[ ]:


# Employee Number is Unique field Check for Duplicates
print(any(df['EmployeeNumber'].duplicated())) 


# - No duplicates values

# ### Check the correlation

# In[ ]:


df.loc[df['Attrition']=="Yes",'attr']=1
df.loc[df['Attrition']=="No",'attr']=0
df.attr.value_counts()


# ### Feature Scaling to check the correlation

# In[ ]:


df.columns


# In[ ]:


# Importing matplotlib and seaborn
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
# Let's see the correlation matrix 
plt.figure(figsize = (16,10))     # Size of the figure
sns.heatmap(df.corr(),annot = True)


# - This Heat Map help to understand how all these variables correlated
# - age is correlated with MonthlyIncome, Joblevel, Totalworkingyears
# - Job level and Monthly income has similar kind of correlation with other variables, only one variable can be use for analysis
# - YearsAtcompany, YearsinCurrentrole, YearsSincelastpromotion, YearswithcurrManager--out of these vars only one or 2 vars can be use for analysis

# ### Univariate and Mulivariate Analysis

# In[ ]:


#Assign Label 
#Education 1 'Below College' 2 'College' 3 'Bachelor' 4 'Master' 5 'Doctor'
#EnvironmentSatisfaction 1 'Low' 2 'Medium' 3 'High' 4 'Very High'
#JobInvolvement 1 'Low' 2 'Medium' 3 'High' 4 'Very High'
#JobSatisfaction 1 'Low' 2 'Medium' 3 'High' 4 'Very High'
#PerformanceRating 1 'Low' 2 'Good' 3 'Excellent' 4 'Outstanding'
#RelationshipSatisfaction 1 'Low' 2 'Medium' 3 'High' 4 'Very High'
#WorkLifeBalance 1 'Bad' 2 'Good' 3 'Better' 4 'Best'

#Education
df.loc[df['Education'] == 1,'Education_type'] = "Below College"
df.loc[df['Education'] == 2,'Education_type'] = "College"
df.loc[df['Education'] == 3,'Education_type'] = "Bachelor"
df.loc[df['Education'] == 4,'Education_type'] = "Master"
df.loc[df['Education'] == 5,'Education_type'] = "Doctor"
#EnvironmentSatisfaction
df.loc[df['EnvironmentSatisfaction'] == 1,'Envnt_Satisfctn_type'] = "Low"
df.loc[df['EnvironmentSatisfaction'] == 2,'Envnt_Satisfctn_type'] = "Medium"
df.loc[df['EnvironmentSatisfaction'] == 3,'Envnt_Satisfctn_type'] = "High"
df.loc[df['EnvironmentSatisfaction'] == 4,'Envnt_Satisfctn_type'] = "Very High"

#JobInvolvement

df.loc[df['JobInvolvement'] == 1,'JobInvolvement_type'] = "Low"
df.loc[df['JobInvolvement'] == 2,'JobInvolvement_type'] = "Medium"
df.loc[df['JobInvolvement'] == 3,'JobInvolvement_type'] = "High"
df.loc[df['JobInvolvement'] == 4,'JobInvolvement_type'] = "Very High"

#JobSatisfaction

df.loc[df['JobSatisfaction'] == 1,'JobSatisfaction_type'] = "Low"
df.loc[df['JobSatisfaction'] == 2,'JobSatisfaction_type'] = "Medium"
df.loc[df['JobSatisfaction'] == 3,'JobSatisfaction_type'] = "High"
df.loc[df['JobSatisfaction'] == 4,'JobSatisfaction_type'] = "Very High"

#PerformanceRating
df.loc[df['PerformanceRating']==1,'Perfrm_rating_type']="Low"
df.loc[df['PerformanceRating'] == 2,'Perfrm_rating_type'] = "Medium"
df.loc[df['PerformanceRating'] == 3,'Perfrm_rating_type'] = "High"
df.loc[df['PerformanceRating'] == 4,'Perfrm_rating_type'] = "Very High"
#RelationshipSatisfaction

df.loc[df['RelationshipSatisfaction']==1,'Rel_satisfctn_type']="Low"
df.loc[df['RelationshipSatisfaction'] == 2,'Rel_satisfctn_type'] = "Medium"
df.loc[df['RelationshipSatisfaction'] == 3,'Rel_satisfctn_type'] = "High"
df.loc[df['RelationshipSatisfaction'] == 4,'Rel_satisfctn_type'] = "Very High"

#WorkLifeBalance
df.loc[df['WorkLifeBalance']==1,'WorkLifeBal_type']="Bad"
df.loc[df['WorkLifeBalance'] == 2,'WorkLifeBal_type'] = "Good"
df.loc[df['WorkLifeBalance'] == 3,'WorkLifeBal_type'] = "Better"
df.loc[df['WorkLifeBalance'] == 4,'WorkLifeBal_type'] = "Best"

df['WorkLifeBal_type'].value_counts()

#Create age band
df.Age.max() #60
df.Age.min() #18
df.Age.mean() #36

df.loc[((df['Age'] >= 18) & (df['Age'] <= 25)),'Age_band']='18_ge_to_le_25'
df.loc[((df['Age'] > 25) & (df['Age'] <= 35)),'Age_band']='25_gr_to_le_35'
df.loc[((df['Age'] > 35) & (df['Age'] <= 45)),'Age_band']='35_gr_to_le_45'
df.loc[((df['Age'] > 45) & (df['Age'] <= 55)),'Age_band']='45_gr_to_le_55'
df.loc[(df['Age'] > 55),'Age_band']='gr_55'
df['Age_band'].value_counts()


# ### Age, Gender, Salary, Job satisfaction

# In[ ]:


plt.figure(num=None, figsize=(12, 8), dpi=80, facecolor='w', edgecolor='k')
sns.boxplot(x='Gender',y="Age", hue="Attrition", data=df)
plt.show()


# - Not a huge diff in the avg. age of attritied employee of male and female employees

# #### Let's see how is pay diffirence gender wise

# In[ ]:


plt.figure(figsize=(20, 6))
sns.boxplot(x='Gender', y='MonthlyIncome', hue="Attrition", data=df)
plt.title('Gender wise avg. Salaries')
plt.show()


# -  There is no significance diff pay diff in male and female

# #### Let's see gender wise job satisfaction

# In[ ]:


plt.figure(figsize=(20, 4))
sns.boxplot(x='Gender',y='JobSatisfaction', data=df, hue='Attrition')
plt.title("Genderwise Job Satisfaction")
plt.ylabel("Jobsatisfaction Unit")
plt.show()


# - For attrited employees male has lower jobs satisfaction

# In[ ]:


plt.figure(figsize=(20, 4))
sns.countplot(x='Age_band', data=df, hue='Attrition')
plt.title("Attrition Age Bandwise")
plt.ylabel("Number of Clients")
plt.show()


# In[ ]:


# Check Attrition Percentage for each band

group_by = df.groupby('Age_band')

df_t1=df.groupby('Age_band')['EmployeeNumber'].count().reset_index()
df_t2=df[df['Attrition'] == "Yes"].groupby('Age_band')['EmployeeNumber'].count().reset_index()
df_t1['tot_count']=df_t1['EmployeeNumber']
df_t1=df_t1.drop(columns = ['EmployeeNumber'])
df_t2['attr_cnt']=df_t2['EmployeeNumber']
df_t2=df_t2.drop(columns = ['EmployeeNumber'])
df_t1
df_t2
#Merge these two datasets
final = pd.merge(df_t1, df_t2, how='inner', on = 'Age_band')
final['attrition_per_total']=round(final['attr_cnt']/(final['tot_count'])*100)
final
#Plot on bar chart
plt.figure(figsize=(15, 4))
sns.barplot(x='Age_band', y='attrition_per_total', data=final)
plt.title("No of Clients Age Bandwise")
plt.ylabel("Attrition(%) wrt. total")
plt.show()


# - Maximun attrition happens between age 18 to 25 (36%) followed by age band 25 to 35 -(19%)
# - Minimum attrition happens between age range of 35 to 45(9%)

# ### DistanceFromHome and Jobrole

# In[ ]:


# subplot 1
#distplot for not attrited clients
plt.figure(figsize=(24, 6))
df_dis1=df.loc[df['Attrition']=="Yes"]
plt.subplot(2, 2, 1)
plt.title('DistanceFromHome for Attrited Employees')
sns.distplot(df_dis1['DistanceFromHome'])

# subplot 2
plt.figure(figsize=(24, 6))
df_dis2=df.loc[df['Attrition']=="No"]
plt.subplot(2, 2, 2)
plt.title('DistanceFromHome for Not attrited Employees')
sns.distplot(df_dis2['DistanceFromHome'])


# - From dist plot, office distance more than 10KM has higher attrition

# In[ ]:


plt.figure(figsize=(20, 6))
sns.boxplot(x='JobRole', y='DistanceFromHome', data=df, hue="Attrition")


# - Distancefromhome has more impact on Job role of "Health Represntative" and "Human Resource", WFH option can not be
#   use in such kind of roles

# ### Monthly Salary, Education ,Jobtitle, department, job staisfaction

# In[ ]:


# subplot 1
#distplot for not attrited clients
plt.figure(figsize=(24, 6))
df_dis1=df.loc[df['Attrition']=="Yes"]
plt.subplot(2, 2, 1)
plt.title('MonthlyIncome for Attrited Employees')
sns.distplot(df_dis1['MonthlyIncome'])

# subplot 2
plt.figure(figsize=(24, 6))
df_dis2=df.loc[df['Attrition']=="No"]
plt.subplot(2, 2, 2)
plt.title('MonthlyIncome for Not attrited Employees')
sns.distplot(df_dis2['MonthlyIncome'])


# - Attited employee has less monthly income in compare to not attrited employees

# In[ ]:


plt.figure(figsize=(20, 6))
sns.boxplot(x='Department', y='MonthlyIncome', hue="Attrition", data=df)
plt.title('Department  wise Salaries')
plt.show()


# -  In department RND and HR there is a huge diff in salaries and people with low salaries are quitting

# #### Let's check that department with high salary gaps are having more attriton or not

# In[ ]:


group_by = df.groupby('Department')

df_t1=df.groupby('Department')['EmployeeNumber'].count().reset_index()
df_t2=df[df['Attrition'] == "Yes"].groupby('Department')['EmployeeNumber'].count().reset_index()
df_t1['tot_count']=df_t1['EmployeeNumber']
df_t1=df_t1.drop(columns = ['EmployeeNumber'])
df_t2['attr_cnt']=df_t2['EmployeeNumber']
df_t2=df_t2.drop(columns = ['EmployeeNumber'])
df_t1
df_t2
#Merge these two datasets
final = pd.merge(df_t1, df_t2, how='inner', on = 'Department')
final['attrition_per_total']=round(final['attr_cnt']/(final['tot_count'])*100)
final
#Plot on bar chart
plt.figure(figsize=(15, 4))
sns.barplot(x='Department', y='attrition_per_total', data=final)
plt.title("No of Clients Department wise")
plt.ylabel("Attrition(%) wrt. total")
plt.show()


#  - Sales has 21% attrition rate and HR has 19% which very high in compare to RND (14%)

# #### Let's check what is impact of monthly salary and job satisfaction  together

# In[ ]:


plt.figure(figsize=(20, 6))
sns.boxplot(x='JobSatisfaction_type', y='MonthlyIncome', hue="Attrition", data=df)
plt.title('Jobsatisfaction Level and Salaries')
plt.show()


# - Salary has an impact on all kind of job satisfaction level, less salary is big reason for job change

# In[ ]:


plt.figure(figsize=(20, 6))
sns.boxplot(x='Education_type', y='MonthlyIncome', hue="Attrition", data=df)
plt.show()


# - People are quitting because at same education their salaris are less in compare to who don't quit

# In[ ]:


plt.figure(figsize=(20, 4))
sns.boxplot(x='JobLevel', y='MonthlyIncome', hue="Attrition", data=df)
plt.show()


# - People who are having less monthly income at joblevel 4 are quitting
# - At job level 5, people who are having more monthly income are quitting

# #### Lets check what is the impact of Dailyrate and jobrole on attrition

# In[ ]:


plt.figure(figsize=(20, 4))
sns.boxplot(x='JobRole', y='DailyRate', hue="Attrition", data=df)
plt.show()


# ### PercentSalaryHike, PerformanceRating, EnvironmentSatisfaction, WorkLifeBalance, and OverTime

# In[ ]:


plt.figure(figsize=(15, 4))
sns.boxplot(x='Perfrm_rating_type', y='PercentSalaryHike', hue="Attrition", data=df)
plt.title("Avg. Salary Hike by performance ratingwise")
plt.show()


# - Salary Hike and Performance ratings are in sink

# #### Let's check Environment Satisfaction

# In[ ]:


plt.figure(figsize=(20, 4))
sns.boxplot(x='JobRole', y='EnvironmentSatisfaction', hue="Attrition", data=df)
plt.title("Enivironment Satisfaction JobRole wise")
plt.show()


# - Environment Satisfaction is low for the poeple who are quitting

# In[ ]:


plt.figure(figsize=(20, 4))
sns.boxplot(x='Department', y='WorkLifeBalance', hue="Attrition", data=df)
plt.title("Work life balance department wise")
plt.show()


# -  Worklife balance seems ok in all the departments

# In[ ]:


plt.figure(figsize=(20, 4))
sns.countplot(x='OverTime', hue="Attrition", data=df)
plt.title("Attriton Overtime wise")
plt.show()


# -  Attrition is high due to overtime

# In[ ]:


#Overtime vs Jobrole
df.loc[df['OverTime']=="Yes",'OverTime_1']=1
df.loc[df['OverTime'] =="No",'OverTime_1'] =0

plt.figure(figsize=(20, 4))
sns.boxplot(x='JobRole', y='OverTime_1', hue="Attrition", data=df)
plt.title("OverTime JobRole wise")
plt.show()


# -  Some job roles such as Sales Executive, Laboratory Technician and HR are impacted by ovetime

# ### YearsAtCompany,YearsInCurrentRole,YearsSinceLastPromotion,TotalWorkingYears and YearsWithCurrManager

# In[ ]:


plt.figure(figsize=(20, 4))
sns.boxplot(x='JobRole', y='YearsAtCompany', hue="Attrition", data=df)
plt.title("YearsAtCompany JobRole wise")
plt.show()


# -  Employees who have attrited in some roles like Manager, Research Director has significant higher number of years
#    in company in compare to employees who stayed back

# In[ ]:


plt.figure(figsize=(20, 4))
sns.barplot(x='JobRole', y='YearsInCurrentRole', hue="Attrition", data=df)
plt.title("YearsInCurrentRole JobRole wise")
plt.show()


# -  In roles such as Healthcare, Manager and Research Director attrired employees have spend more time in compare to
#    employess who stayed back
# -  In others roles attrites employees have spend less time in compare to non- attrited employees in same role

# In[ ]:


plt.figure(figsize=(20, 4))
sns.barplot(x='JobLevel', y='YearsSinceLastPromotion', hue="Attrition", data=df)
plt.title("YearsSinceLastPromotion JobLevel wise")
plt.show()


# -  In Job level 3,4,5 attrited employees have spend significant more time without promotion in compare to employees
#    who stayed back

# ### RelationshipSatisfaction,MaritalStatus

# In[ ]:


group_by = df.groupby('Rel_satisfctn_type')

df_t1=df.groupby('Rel_satisfctn_type')['EmployeeNumber'].count().reset_index()
df_t2=df[df['Attrition'] == "Yes"].groupby('Rel_satisfctn_type')['EmployeeNumber'].count().reset_index()
df_t1['tot_count']=df_t1['EmployeeNumber']
df_t1=df_t1.drop(columns = ['EmployeeNumber'])
df_t2['attr_cnt']=df_t2['EmployeeNumber']
df_t2=df_t2.drop(columns = ['EmployeeNumber'])
df_t1
df_t2
#Merge these two datasets
final = pd.merge(df_t1, df_t2, how='inner', on = 'Rel_satisfctn_type')
final['attrition_per_total']=round(final['attr_cnt']/(final['tot_count'])*100)
final
#Plot on bar chart
plt.figure(figsize=(15, 4))
sns.barplot(x='Rel_satisfctn_type', y='attrition_per_total', data=final)
plt.title("No of Employees Rel_satisfctn_type Bandwise")
plt.ylabel("Attrition(%) wrt. total")
plt.show()


# - Relationship Satisfaction Status low has the highest attrition Rate(21%), rest all have same attr rate

# In[ ]:


group_by = df.groupby(['MaritalStatus','Gender'])
df_t1=group_by['EmployeeNumber'].count().reset_index()
df_t2=df[df['Attrition'] == "Yes"].groupby(['MaritalStatus','Gender'])['EmployeeNumber'].count().reset_index()
df_t1['tot_count']=df_t1['EmployeeNumber']
df_t1=df_t1.drop(columns = ['EmployeeNumber'])
df_t2['attr_cnt']=df_t2['EmployeeNumber']
df_t2=df_t2.drop(columns = ['EmployeeNumber'])
df_t1
final = pd.merge(df_t1, df_t2, how='inner', on = ['MaritalStatus','Gender'])
final['attrition_per_total']=round(final['attr_cnt']/(final['tot_count'])*100)
final
#Plot on bar chart
plt.figure(figsize=(15, 4))
sns.barplot(x='MaritalStatus', y='attrition_per_total',hue='Gender', data=final)
plt.title("No of Employees MaritalStatus Bandwise")
plt.ylabel("Attrition(%) wrt. total")
plt.show()


# In[ ]:


final


# - Single employess has the highest attrition rate(25%)
# - Male perchantage of attrition is more in compare to female

# ### BusinessTravel,MaritalStatus

# In[ ]:


df['BusinessTravel'].value_counts()


# In[ ]:


group_by = df.groupby(['MaritalStatus','BusinessTravel'])
df_t1=group_by['EmployeeNumber'].count().reset_index()
df_t2=df[df['Attrition'] == "Yes"].groupby(['MaritalStatus','BusinessTravel'])['EmployeeNumber'].count().reset_index()
df_t1['tot_count']=df_t1['EmployeeNumber']
df_t1=df_t1.drop(columns = ['EmployeeNumber'])
df_t2['attr_cnt']=df_t2['EmployeeNumber']
df_t2=df_t2.drop(columns = ['EmployeeNumber'])
df_t1
final = pd.merge(df_t1, df_t2, how='inner', on = ['MaritalStatus','BusinessTravel'])
final['attrition_per_total']=round(final['attr_cnt']/(final['tot_count'])*100)
final
#Plot on bar chart
plt.figure(figsize=(15, 4))
sns.barplot(x='MaritalStatus', y='attrition_per_total',hue='BusinessTravel', data=final)
plt.title("")
plt.ylabel("Attrition(%) wrt. total")
plt.show()


# In[ ]:


final


# - Travel Frequently leads to higher attrition, specially incase of single it is 39%, for married and divorced it 
#   is less than 21%
