#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# Data science is a growing field which cuts across many industries. Kaggle has published the 2019 survey results of the Data Scientists who use Kaggle around the world. I am fairly new on Kaggle and would like to use this dataset to answer some questions that are important to me from the Canadian perspective.
# 
# # Providing Answers Using the CRISP-DM (Cross Industry Process for Data Mining) Approach
# 1. Context or Problem or Business Understanding
# 2. Data Understanding
# 3. Data Preparation
# 4. Modeling the Data
# 5. Evaluate the Results
# 5. Deploy

# In[ ]:


from IPython.display import Image
import os
Image("../input/photos/data_plot.jpg")


# Photo by Carlos Muza on Unsplash

# # 1. Context Understanding
# Below are the questions I want to answer from the dataset.
# 
# * What is the demography of the participants
# * What is the age, gender and salary distribution of these Data Scientists in the entire dataset and specifically in Canada?
# * What percentage of Canadian survey participants are women
# * How does Education affect Salary in Canada?
# * How does Education affect Job title in Canada?
# * Which Job Titles earn the most in Canada?

# # 2. Data Understanding
# What is the nature of the data used in this analysis? I will be using python and some python libraries to dissect the data in order to gain more insight.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="ticks", color_codes=True)
from plotly.offline import init_notebook_mode, iplot 
import plotly.graph_objs as go
import plotly.offline as py
import pycountry
py.init_notebook_mode(connected=True)

from IPython import display
get_ipython().run_line_magic('matplotlib', 'inline')

get_ipython().run_line_magic('config', 'IPCompleter.greedy=True')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# ## Load and Prepare the Data

# In[ ]:


dirname = '/kaggle/input/kaggle-survey-2019'
dirname


# In[ ]:


#Read the csv files with Pandas

questions = pd.read_csv(dirname + '/questions_only.csv')
survey_res = pd.read_csv(dirname + '/multiple_choice_responses.csv')
survey_schema = pd.read_csv(dirname + '/survey_schema.csv')
text_res = pd.read_csv(dirname + '/other_text_responses.csv')
print(f'Reading the shape of the data')
print(f"Questions: {questions.shape}")
print(f"Survey Responses: {survey_res.shape}")
print(f"Survey Schema: {survey_schema.shape}")
print(f"Text responses: {text_res.shape}")


# There are four csv files available for this analysis. I will be using only the file containing the survey responses from the participants for this analysis since the questions are also form the columns description in multiple-choice responses file.

# In[ ]:


questions.head(10)


# In[ ]:


survey_res.head()


# In[ ]:


survey_res.describe()


# From the shape of the questions data file, there are 35 questions asked in the survey.
# The multiple choice response table reveals that there were 19,718 responses.
# The describe function shows the top numbers from the dataset. There are many more Indian males between the age of 25 and 29, with a masters' degree whose job titles are Data Scientist than any other group in the dataset.

# # Canadian Analysis of the Kaggle Data Science/ML Survey 2019
# 
# Let's narrow down the analysis to Canada

# In[ ]:


Image("../input/photos/canada.jpg")


# [Photo by Maxime Vermeil (@max.vrm) on Unsplash](https://unsplash.com/photos/3fyF1XhZDf4)

# # 3. Data Preparation
# 
# I am interested in some parts of the dataset, so I will drop the columns that are not important in the process of finding answers to my questions. This will also help reduce the memory size used in my analysis.
# 
# ## Rename the Columns
# First, the columns important for this analysis have to be renamed to have a clearer label. 

# In[ ]:


survey_res.rename(columns={'Q1' : 'Age',
                           'Q2' : 'Gender',
                           'Q3' : 'Country',
                           'Q4' : 'Education Level',
                           'Q5' : 'Job Title',
                           'Q6' : 'Company Size',
                           'Q10' : 'Salary'},
                  inplace=True)
print(survey_res.columns)


# ## Drop some columns
# Since I will be using only 7 out of the 246 columns in the survey_res data, the irrelevant columns will be dropped.

# In[ ]:


# Keep relevant columns
col_to_keep = ['Age',
               'Gender',
               'Country',
               'Education Level',
               'Job Title',
               'Company Size',
               'Salary',]
col_to_keep


# In[ ]:


#use only the kept columns as the new survey result data
survey_res = survey_res[col_to_keep]
survey_res.head()


# In[ ]:


Canada = survey_res[survey_res['Country'].str.contains('Canada')]
print('{} people from Canada participated in the survey'.format(Canada.shape[0]))


# In[ ]:


Canada.describe()


# In[ ]:


#Check current data type:
Canada.dtypes


# ### Check for missing values and handle them

# In[ ]:


Canada.isnull().sum()


# 149 respondents in Canada did not enter salary information. Could it be that they are still students(not working), unemployed or they just decided not to put any value.
# 12 did not enter a job title while 5 left Education empty. We have to take care of those missing values.

# ### Manage Missing Values
# *** Forward-fill and back-fill up the missing Salary values***
# 

# In[ ]:


#Comment out the code in order not to refill multiple times

# Canada['Salary'] = Canada['Salary'].fillna(method='ffill', limit=1, inplace=True).fillna(method='bfill', limit=1, inplace=True)
# Canada


# In[ ]:


Canada.describe()


# ### Age Distribution

# In[ ]:


age = Canada.Age.value_counts()
age


# In[ ]:


percent_age = (age/(Canada.shape[0]))*100
percent_age


# In[ ]:


import seaborn as sns


stan_color = sns.color_palette()[0]
sal_order = age.index
sns.countplot(data=Canada, y="Age", color=stan_color, order=sal_order)


# * Majority of the respondents in the data science and machine learning field in Canada are aged between 25-29 years old which is 19% of the Canadian survey participants.
# * The age of the oldest group is 70+. The youngest age range is 18-21 accounting for 6% of the Canadian respondents (pretty impressive for the youngest age range)

# ### How about gender distribution in the survey?

# In[ ]:


gender = Canada.Gender.value_counts()
gender


# Taking a deeper insight into the female respondents

# In[ ]:


Canada_wom = Canada[Canada['Gender']=='Female']
Canada_wom.describe()


# What percentage of the Canadian respondents are women?

# In[ ]:


percent_women = (Canada_wom.count()/Canada.shape[0])*100
percent_women


# Approximately, 22% of the respondents identified as Female. Oops! We still have a lot of work to do. We need to increase the volume of the STEM "evangelism loudspeakers". A whopping 74% are Male.
# 
# Majority of Canadian women in the survey are aged between 30-34 years, have Master's degree with Job title as Data Scientist. They earn mostly between 60,000 to 69,999 US dollars and work in a company of 0-49 employees. This is typical of the Canadian distribution except for the age range which is older than the top age in the entire dataset across other countries.

# #### What about the men?

# In[ ]:


Canada_men = Canada[Canada['Gender']=='Male']
Canada_men.describe()


# The main difference between the men and the women is in the age range, at least at a peripheral look at the data summary. The top age range for the women (30-34) is older than the top age of the Canadian men which is 25-29.

# ## What is the Education level of Canadian Data Scientists

# In[ ]:


Canada_Ed = Canada['Education Level'].value_counts(sort=True)
Canada_Ed


# In Canada, similar to the entire dataset, majority of respondents have Master's degree.

# In[ ]:


labels_edu = Canada_Ed.index
values_edu = Canada_Ed.values

edu_colors= ['#A7FFEB','#43A047', '#1B5E20', '#76FF03', '#C6FF00', '#DCEDC8', '#E8F5E9'] 

pie = go.Pie(labels=labels_edu, values=values_edu, marker=dict(colors=edu_colors,line=dict(color='#000000', width=1)))
layout = go.Layout(title='Education Level')

fig = go.Figure(data=[pie], layout=layout)
py.iplot(fig)


# ## Money Matters (Salary)

# In[ ]:


Canada_Salary = Canada.Salary.value_counts()
Canada_Salary


# In[ ]:


import seaborn as sns

stan_color = sns.color_palette()[0]
#Order the plot by count values
sal_order = Canada_Salary.index
sns.countplot(data=Canada, y='Salary', color=stan_color, order=sal_order)


# 30 respondents entered their salary values as 0-999. This is below Canadian average salary. These people may be unemployed. Let's look at the Job title to see if the number of unemployed people will match or be close to the number that declared such a low amount in salary.

# ## What are job titles of the Canadian survey participants

# In[ ]:


Canada_Job = Canada['Job Title'].value_counts()
Canada_Job


# As assumed in the salary details, the number of respondents (31) with the **lowest salary** value is close to the number of respondents (30) that declared **unemployed**.
# 
# Data Scientist appears top on the job titles.

# # 4. Comparing the Features in the Data

# Using [ranjeetjain3](https://www.kaggle.com/ranjeetjain3/aws-vs-gcp-vs-azure) for bivariate plot code:

# In[ ]:


def compute_percentage(df,col):
    """
    The compute_percentage object computs the percentage of the value counts for the column.
    
    Args:
        df (dataframe): The dataset for the analysis
        col: The specific column (feature) of interest within the dataframe
        
    Returns:
            percentage of the frequency of the feature
    """
    return df[col].value_counts(normalize=True) * 100

def bi_variant_chart(col1,col2,x_title,y_title):
    """
    Args:
        col1 (str): col1 is the first feature for plotting the bar chart
        col2 (str): col2 is the second feature for plotting the bar chart
        x_title (str): Title for the x-axis
        y_title: Title for the y-axis
    
    """
    
    index = Canada[col1].dropna().unique()
    vals = Canada[col2].unique()
    layout = go.Layout()
    trace = []
    for j,y_axis in enumerate(vals):
        trace.append(go.Bar(x = Canada[Canada[col2] == y_axis][col1].value_counts().index,
                            y = Canada[Canada[col2] == y_axis][col1].sort_values().value_counts().values,
                opacity = 0.6, name = vals[j]))
    fig = go.Figure(data = trace, layout = layout)
    fig.update_layout(
        title = x_title,
        yaxis = dict(title = y_title),
        legend = dict( bgcolor = 'rgba(255, 255, 255, 0)', bordercolor = 'rgba(255, 255, 255, 0)'),
        bargap = 0.15, bargroupgap = 0.1,legend_orientation="h")
    fig.show()


# ## How Does Education Level Affect Salary?

# In[ ]:


bi_variant_chart("Salary","Education Level","Salary VS Education Level","Count")


# ### Interpretation

# Some of the participants with Doctoral degree earned within $0-999$ while some earned as high as $299,999$. 
# 
# The low value is probably those who are unemployed because this Salary does not correlate with the Canada average wage.
# The participants who earned over "$100k"$ had at least some college degree and mostly Master's or Doctoral degree.
# The outlier salary within range of 300,000-500,000 is a fellow with Bachelor's degree. What kind of job do they do? I'm curious to find out.

# ## Salary Vs Job Title

# In[ ]:


bi_variant_chart("Salary","Job Title","Salary VS Job Title","Count")


# To get a closer look at each of the Job Titles vs Salary, double click on the Job Title legend below the bar chart plot. 
# See the maximum salary for each job title below:
# 
# * Research Scientist: 125,000 to 149,999
# * Data Engineer: 250,000 to 299,999
# * Business Analyst: 125,00-149,999
# * Not Employed: 250,000 to 299,999. Wait! How can someone who claimed to be unemployed still declare a salary value up to 250,000? What is going on here? Does that mean Salary is not a piece of reliable information at all
# * Data Scientist: 200,000-249,999
# * Other: 0-999 to >500,000. Why did they choose other? It could be that their job title is not in the options listed in the survey questions. 
# * Software Engineer: 150,000-199,999
# * Statisticians: 300,000-500,000. Very few people have this job title and earn quite high.
# * Data Analyst: 90,000-99,999
# * DBA/Database Engineer: 100,000-124,999
# * Student: 150,000-199,000. Do these students really earn this much or they entered their "Wish List Salary" in the survey?
# * Product/Project Manager: 100,000-124,999
# 
# The salary values for the different job title raise some questions and seems unreliable especially from those who said they are Students, Unemployed and Other.
# The Statistician is the one I was curious to know what his/her job title is. We'll find out if this person is a man or woman soon.

# ## Company Size Vs Salary

# In[ ]:


bi_variant_chart("Company Size","Salary","Company Size VS Salary","Count")


# #### Interpretation

# * Those earning from 300,000 to 500,000 dollars and above work for companies with 0-49 employee size
# * Majority those earning 200,000 to 299,999 work for companies with over 1000 to 10,000 employees (large corporations).
# * The 60,000 to 69,999 earners which is the mode salary for this data work every where. They work at companies with employee size 0-49 (Start-Ups) to company size over 10,000 employees.
# 

# ## Gender VS Salary

# In[ ]:


bi_variant_chart("Gender","Salary","Gender VS Salary","Count")


# **Interpretation**
# 
# * The second-highest salaries 300,000 to 500,000...goes to a man who happens to be a Statistician. The other man with the salary over 500,000 did not tell us his job title. We'll leave it at that. At least we have found answer to the curious money question.
# * Only one woman declared salary of 250,000-299,999 which is the highest for the Female participants
# * 6 women were in the range of 150,000-199,999
# 

# This kernel is still work in progress. I will come back from time to time to update it. Thanks for reading.
