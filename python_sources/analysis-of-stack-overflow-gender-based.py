#!/usr/bin/env python
# coding: utf-8

# # This is my first ever Kaggle submission
# 
# I have been a silent observer of the Kaggle community for years but now I feel its time I should start contributing and publishing content.
# 
# **I would highly appreciate your feedback in the form of comments and your support (preferably) in the form of likes. 
# **
# 
# 

# # Introduction
# 
# Most of us are very familiar with Stack Overflow and I feel that whenever I got stuck with any of my coding problems I was able to find the best possible solution on the website.
# 
# I have attempted to briefly analyse the demographics, Salary, Job Satisfaction and highlighted the differences based on the gender
# 

# # Importing the dataset & libraries

# In[ ]:


import plotly as pycred
import plotly.offline  as py
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt 
import seaborn as sns
import pycountry
from sklearn import preprocessing
survey_data = pd.read_csv('../input/survey_results_public.csv')
survey_schema = pd.read_csv('../input/survey_results_schema.csv')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
import warnings
warnings.filterwarnings("ignore")
survey_data.head()


# # Percentage of People who code as a hobby.

# In[ ]:


labels = survey_data['Hobby'].value_counts().index
values = survey_data['Hobby'].value_counts(1).values 
colors = ['yellowgreen', 'lightcoral']
explode = (0.2, 0) 
plt.pie(values, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=40)
plt.axis('equal')
plt.title('Percentage of people who code a hobby?')


# **It was very interesting to see that 80.8% people code as a hobby.**

# # How many people contribute to open source projects ?
# 
# Its interesting to see that 80.8% of the people code as a hobby but only 43.6% choose to contribute to the open source community

# In[ ]:


labels1 = survey_data['OpenSource'].value_counts().index
values1 = survey_data['OpenSource'].value_counts(1).values 
plt.title('Do you contribute to open source projects?')
colors = [ 'lightcoral','yellowgreen']
explode = (0.1, 0) 
plt.pie(values1, explode=explode, labels=labels1, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=260)
plt.axis('equal')

plt.show()


# 

# # In which country do you currently reside (On Map) ?

# In[ ]:


input_countries = survey_data['Country']
countries = {}
for country in pycountry.countries:
    countries[country.name] = country.alpha_3

survey_data['CountryCode'] = [countries.get(country, 'Unknown code') for country in survey_data['Country']]


data = [ dict(
        type = 'choropleth',
        locations = survey_data['CountryCode'].value_counts().index ,
        z = survey_data['CountryCode'].value_counts().values ,
        autocolorscale = True,
        marker = dict(
            line = dict (
                color = 'rgb(180,180,180)',
                width = 0.5
            ) ),
        colorbar = dict(
            autotick = False,
            title = 'In which country do you currently reside?'),
      ) ]

layout = dict(
    title = 'In which country do you currently reside?',
    geo = dict(
        showframe = False,
        showcoastlines = False,
        projection = dict(
            type = 'Mercator'
        )
    )
)
fig = dict( data=data, layout=layout )
py.iplot( fig, validate=False, filename='In which country do you currently reside' )


# # In which country do you currently reside?
# * It is evident that United States of America has the maximum number of respondents at 20,309 followed by India at 13,721.
# * I was expecting china to be in the list somewhere next to india amongst the other asian countries.
# 

# In[ ]:


Top10 = survey_data["Country"].dropna().value_counts().head(10)

df = pd.DataFrame({'Country': Top10.index,
                   'Number of Participant': Top10.values},index=Top10.index,columns=['Number of Participant'])
ax = sns.barplot(x=Top10.index, y="Number of Participant", data=df)
plt.xticks(rotation=90)
plt.show()


# # Plotting the Salary of Male/Female survey partipants.

# In[ ]:


Filtered_Salary = survey_data[survey_data['ConvertedSalary'].notnull()]
Filtered_Salary = Filtered_Salary[Filtered_Salary.ConvertedSalary != 0]
malefemale=Filtered_Salary[Filtered_Salary['Gender'].isin(["Male","Female"])]
ax= sns.boxplot(x='ConvertedSalary',y='Gender',data=malefemale)
plt.xticks(rotation=90)
plt.show()


# Looking at the plot it gives an impression that the salary is pretty close but slightly higher to men. It would be interesting to see how the salary varies with age group.

# # Plotting the Salary of Male/Female based on the age group 
# 
# 
# This chart gives several insights like:<br>
# 
# 1) Women usually start off at a pretty low salary range in comparison to men ( Looking at participants under 18)<br>
# 2) Contrary to popular belief women in the age group 18 to 44 are making equal if not more money than men based on the survey.<br>
# 3) We can see a steep decline the in salary in the age group of 45-54 for women wherein the salary for men continues to grow to the age of 64.

# In[ ]:


sns.set(style="ticks")
Sal_age=malefemale[['ConvertedSalary','Gender','Age','YearsCoding','FormalEducation','HoursComputer']]
g=sns.factorplot(x="ConvertedSalary", y="Age", hue="Gender", col="Gender", data=Sal_age,
                   capsize=.2, palette="muted", size=8, aspect=.75,order=[ 'Under 18 years old','18 - 24 years old', '25 - 34 years old',
                                                                             '35 - 44 years old',
       '45 - 54 years old', '55 - 64 years old', '65 years or older'])
g.despine(left=True)
plt.show()


# 
# # Plotting the Salary of Male/Female based on the education level

# * This is a very interesting observation  that women who never completed a formal education earn the least amount of money whereas men earn the maximum amount of money if they did not go to college.
# *

# In[ ]:


g1=sns.factorplot(x="FormalEducation", y="ConvertedSalary", hue="Gender", data=Sal_age,x_estimator=np.mean
                  , palette="muted", size=8, label="Men Vs. Women Salary based on education")
plt.xticks(rotation=90)

plt.show()


# # Plotting the Salary  and the time Men/Women spend in from the of the computer
# * Surprisingly the women who spend less than one hour make more money than men
# * We can clearly observe that men who work between 5-8 hours a day make most amount of money ( which makes sense because they are regular office hours)

# In[ ]:


g1=sns.factorplot(x="HoursComputer", y="ConvertedSalary", hue="Gender", data=Sal_age, palette="muted", size=8)

plt.show()


# 

# # Job satisfaction and salary for men/women  
# 
# In order to get a clear picture. I  collapsed the observations in  discrete bins to plot an estimate of central tendency along with a confidence interval. 
# 
# The scale is :
# 
# **0: Extremely dissatisfied------>   6:Extremely satisfied**
# 
# 
# * The salary for men show a linear relationship wherein the job satisfaction increases with the salary . However, for women its not exactly an uphill 

# In[ ]:


#Define a generic function using Pandas replace function
def coding(col, codeDict):
  colCoded = pd.Series(col, copy=True)
  for key, value in codeDict.items():
    colCoded.replace(key, value, inplace=True)
  return colCoded

malefemale['CodedJobSatisfaction']= coding(malefemale["JobSatisfaction"], {'Extremely dissatisfied':0,'Moderately dissatisfied':1,'Slightly dissatisfied':2,'Neither satisfied nor dissatisfied':3,'Slightly satisfied':4,'Moderately satisfied':5,'Extremely satisfied':6})
sns.lmplot(x="CodedJobSatisfaction", y="ConvertedSalary", hue="Gender", data=malefemale,x_estimator=np.mean,
           markers=["o", "x"], palette="Set1", size=8);
plt.show()


# # Job satisfaction and Age for men/women  
# 
# *  Most men get happier with age with their job and happiest when they are over 65 followed by when they are under 18.
# *  Exactly opposite to men women are least satisfied with their job when they are under 18 or over 65

# In[ ]:


sns.pointplot(x='Age', y="CodedJobSatisfaction",hue="Gender", data=malefemale, markers=["o", "x"], linestyles=["-", "--"], aspect=2)
plt.xticks(rotation=90)
plt.show()


# # To be continued.... 
# 
# **This is my first draft. I would like to constantly contribute to the analysis.  This is my first ever kaggle post and would always remain special to me.
# Thank you for taking time to go through it and I appreciate your feedback and support.
# **
# 
# 
# 
