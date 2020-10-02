#!/usr/bin/env python
# coding: utf-8

# # H1-B VISA 

# A COMPREHENSIVE ANALYSIS ON H1-B VISAS - Saiteja Nakka
# - To know about the data visit https://storage.googleapis.com/kaggle-datasets/11361/15737/h1b-2017-metadata.pdf?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1539822929&Signature=sd7i4%2BNuZyrh5E19BShsF8wGJB08CNYsf%2BvX4%2FAufvXe57w9dK7uVIV0uwqZI6ZKRNp56amXAAZsKciXmABlbuZ4rmRanmNjhM6JP3Iuv4UyIa7IvKJKnVaSq%2F8ppYYZMVveVivH3yVJn3eWfNhvhrBdShRC%2BcG4j8KuvCdpO70wzUX7bNvfA2McsSp5zMIbYA9mtu0Sqk6gtAuSmgfY9Q7e5U1wRSnKKEJzINXIVUtShZPZ3WBJyf8Yypxoj7QfXNzXMUXtUSMSTY45UYIvVtoRTs4jucgpMptXA5pvlGmIDShct5qgw%2Fn83%2Bv3Gm8J%2BgXNxMkv5AQw6kavgmYQmA%3D%3D

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
import matplotlib.pyplot as plt
init_notebook_mode(connected=True)
from plotly import tools
import warnings
warnings.filterwarnings("ignore")


# In[ ]:


df = pd.read_csv("../input/H-1B_Disclosure_Data_FY17.csv")


# In[ ]:


df.head()


# ## We focus on H1-B visa and Employers from USA only

# In[ ]:


df.VISA_CLASS.value_counts()


# In[ ]:


df.EMPLOYER_COUNTRY.value_counts()


# In[ ]:


df = df[df.VISA_CLASS == 'H-1B']
df= df[df.EMPLOYER_COUNTRY == 'UNITED STATES OF AMERICA']


# #### Number of unique values each column has

# In[ ]:


df.apply(lambda x:len(x.unique()))


# #### Columns which have missing values in it

# In[ ]:


df.isnull().sum()[df.isnull().sum() > 0]


# ### DROP the useless columns.
# - By examining the above two columns we can remove all the useless columns.
# - Since we have a lot of those columns to drop, I instead selected the ones I need

# In[ ]:


to_select = ['CASE_STATUS', 'EMPLOYMENT_START_DATE','EMPLOYER_NAME', 'EMPLOYER_STATE','JOB_TITLE', 'SOC_NAME','FULL_TIME_POSITION',
            'PREVAILING_WAGE','PW_UNIT_OF_PAY','WORKSITE_STATE']


# In[ ]:


df = df[to_select]


# In[ ]:


df.isnull().sum()[df.isnull().sum() > 0]


# ## Dealing with missing values
# - The following method is same as dropna.

# In[ ]:


df = df[df['EMPLOYMENT_START_DATE'].notnull()]
df = df[df['JOB_TITLE'].notnull()]
df = df[df['SOC_NAME'].notnull()]
df = df[df['FULL_TIME_POSITION'].notnull()]
df = df[df['PW_UNIT_OF_PAY'].notnull()]
df = df[df['WORKSITE_STATE'].notnull()]
df = df[df['EMPLOYER_NAME'].notnull()]


# ###### We got rid of null values

# In[ ]:


df.isnull().sum()[df.isnull().sum() > 0]


# In[ ]:


df.head()


# ### Convert the EMPLOYMENT_START_DATE to pandas date time format

# In[ ]:


df['EMPLOYMENT_START_DATE'] = pd.to_datetime(df['EMPLOYMENT_START_DATE'])


# #### The following cell shows us how the wage is varied.
# - You might notice some abornmal values like the max hourly pay as 2017,143166 etc etc.
# - These are called the outliers and our data has a lot of them

# In[ ]:


df.groupby(['FULL_TIME_POSITION','PW_UNIT_OF_PAY']).describe()['PREVAILING_WAGE']


# ### To make our analysis easy lets first convert the Monthly, Weekly and Bi-weekly pay to Annual pay
# - The hourly pay conversion takes a lot of time. So, i have done in following stages after outliers were removed.
# - Montly pay is multiplied by 12. ( As we have 12 months in a year)
# - Weekly pay is multiplied by 48. ( Even though we have 52 weeks per year, I felt this might be better)
# - Bi-Weekly pay is multiplied by 24. ( As we have two bi-weeks in a month)

# In[ ]:


for i in df.index:   
        if df.loc[i,'PW_UNIT_OF_PAY'] == 'Month':
            df.loc[i,'PREVAILING_WAGE'] = df.loc[i,'PREVAILING_WAGE'] * 12
        if df.loc[i,'PW_UNIT_OF_PAY'] == 'Week':
            df.loc[i,'PREVAILING_WAGE'] = df.loc[i,'PREVAILING_WAGE'] * 48
        if df.loc[i,'PW_UNIT_OF_PAY'] == 'Bi-Weekly':
            df.loc[i,'PREVAILING_WAGE'] = df.loc[i,'PREVAILING_WAGE'] * 24


# #### Replace the names bi-weekly, month and week by year.

# In[ ]:


df.PW_UNIT_OF_PAY.replace(['Bi-Weekly','Month','Week'],['Year','Year','Year'], inplace=True)


# #### Checking again.
# - Now you that we have unit of pay in hour and year only.

# In[ ]:


df.groupby(['FULL_TIME_POSITION','PW_UNIT_OF_PAY']).describe()['PREVAILING_WAGE']


# #### Creating a new dummy column
# - As, we are going to deal with a lot of groupby methods below, it will be easy for us if we have a count column.

# In[ ]:


df['countvar'] = 1


# ### Top Employers sponsoring H1-B's
# - The plots are made using plotly which are interactive. So, you can hover over the plot to know more details

# In[ ]:


dftop = df.groupby('EMPLOYER_NAME',as_index=False).count()
dftop = dftop.sort_values('countvar',ascending= False)[['EMPLOYER_NAME','countvar']][0:30]


# In[ ]:


t1 = go.Bar(x=dftop.EMPLOYER_NAME.values,y=dftop.countvar.values,name='top30')
layout = go.Layout(dict(title= "TOP EMPLOYERS SPONSORING",yaxis=dict(title="Num of applications")))
data = [t1]
fig =go.Figure(data,layout)
iplot(fig)


# ### Top Employers and its Case status bar.

# In[ ]:


dftop1 = df.groupby(['EMPLOYER_NAME','CASE_STATUS'],as_index=False).count()
dftop1=dftop1[dftop1.EMPLOYER_NAME.isin(dftop.EMPLOYER_NAME)]


# In[ ]:


t1 = go.Bar(x=dftop1[dftop1.CASE_STATUS == 'CERTIFIED'].sort_values('countvar',ascending= False)['EMPLOYER_NAME'].values,y=dftop1[dftop1.CASE_STATUS == 'CERTIFIED'].sort_values('countvar',ascending= False)['countvar'].values,name='CERTIFIED')
t2 = go.Bar(x=dftop1[dftop1.CASE_STATUS == 'CERTIFIED-WITHDRAWN'].sort_values('countvar',ascending= False)['EMPLOYER_NAME'].values,y=dftop1[dftop1.CASE_STATUS == 'CERTIFIED-WITHDRAWN'].sort_values('countvar',ascending= False)['countvar'].values,name='CERTIFIED-WITHDRAWN')
t3 = go.Bar(x=dftop1[dftop1.CASE_STATUS == 'DENIED'].sort_values('countvar',ascending= False)['EMPLOYER_NAME'].values,y=dftop1[dftop1.CASE_STATUS == 'DENIED'].sort_values('countvar',ascending= False)['countvar'].values,name='DENIED')
t4 = go.Bar(x=dftop1[dftop1.CASE_STATUS == 'WITHDRAWN'].sort_values('countvar',ascending= False)['EMPLOYER_NAME'].values,y=dftop1[dftop1.CASE_STATUS == 'WITHDRAWN'].sort_values('countvar',ascending= False)['countvar'].values,name='WITHDRAWN')

data = [t1,t2,t3,t4]
layout = go.Layout(
    barmode='stack'
)

fig =go.Figure(data,layout)
iplot(fig)


# # Number of Applications per State.
# - Barplot and Choropleth graph

# In[ ]:


dfempst = df.groupby('EMPLOYER_STATE',as_index=False).count()[['EMPLOYER_STATE','countvar']].sort_values('countvar',ascending=False)


# In[ ]:


t1 = go.Bar(x=dfempst.EMPLOYER_STATE.values,y=dfempst.countvar.values,name='Employerstate')
layout = go.Layout(dict(title= "NUMBER OF APPLICATIONS PER STATE",xaxis=dict(title="STATES"),yaxis=dict(title="Num of applications")))
data = [t1]
fig =go.Figure(data,layout)
iplot(fig)


# In[ ]:


data=[dict(
    type='choropleth',
    locations = dfempst.EMPLOYER_STATE,
    z = dfempst.countvar,
    locationmode = 'USA-states',marker = dict(
            line = dict (
                color = 'rgb(255,255,255)',
                width = 2
            ) ),
        colorbar = dict(
            title = "Number of applications")
)]
layout= dict(title="2011-2018 H1B VISA APPLICATIONS ( EMPLOYER STATE)",geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)'),
             )
fig = dict( data=data, layout=layout )
iplot(fig)


# ## Top 20 Job titles

# In[ ]:


dfjob = df.groupby('JOB_TITLE',as_index=False).count()[['JOB_TITLE','countvar']].sort_values('countvar',ascending=False)[0:20]


# In[ ]:


t1 = go.Bar(x=dfjob.JOB_TITLE.values,y=dfjob.countvar.values,name='jobtitle')
layout = go.Layout(dict(title= "TOP 20 JOBS",yaxis=dict(title="Num of applications")))
data = [t1]
fig =go.Figure(data,layout)
iplot(fig)


# ### Extracting the YEAR from the EMPLOYMENT_START_DATE.

# In[ ]:


df['year'] = df.EMPLOYMENT_START_DATE.apply(lambda x: x.year)


# #### Number of applications per year
# - As this the data upto 2017, the employment_start_date has only very few 2018 dates.
# - And of those few, some are removed during the process of dealing with missing values

# In[ ]:


dfyear = df.groupby('year',as_index=False).count()[['year','countvar']]


# In[ ]:


t1 = go.Scatter(
    x=dfyear.year,
    y=dfyear.countvar
)
layout = go.Layout(dict(title= " NUMBER OF APPLICATIONS PER YEAR",xaxis=dict(title="YEARS"),yaxis=dict(title="Num of applications")))
data = [t1]
fig = go.Figure(data=data,layout=layout)
iplot(fig)


# ### Distribution of Case_Status column
# - A lot of them were certified. (Hope's alive :) )

# In[ ]:


t1 = go.Bar(x=df.groupby('CASE_STATUS').count().index,y=df.groupby('CASE_STATUS').count()['countvar'],name='CASESTATUSWISE')
data = [t1]
iplot(data)


# ### Distribution of Case_Status column/ Full_Time position

# In[ ]:


t1 = go.Bar(x=df[df.FULL_TIME_POSITION == 'Y'].groupby('CASE_STATUS').count().index,y=df[df.FULL_TIME_POSITION == 'Y'].groupby('CASE_STATUS').count()['countvar'],name='FULL-TIME ')
t2 = go.Bar(x=df[df.FULL_TIME_POSITION == 'N'].groupby('CASE_STATUS').count().index,y=df[df.FULL_TIME_POSITION == 'N'].groupby('CASE_STATUS').count()['countvar'],name='PART-TIME ')
data = [t1,t2]
layout = go.Layout(barmode='stack')
fig = go.Figure(data =data,layout =layout)
iplot(fig)


# ## Dealing with outliers in Pay scale.
# - In the below code, see the difference between the 75th percentile and the max value, that huge difference clearly indicates the presence of outliers.
# - The min has value of 0, which is obviously false. ( No one works for free)

# In[ ]:


df.PREVAILING_WAGE.describe()


# In[ ]:


df.PW_UNIT_OF_PAY.value_counts()


# ### My knowledge about H1B
# - The full time H1B employees have a minimum salary of 65k now, as we have data fromm 2011 I'm setting the minimum salary to be 40K.
# - There is no limit for maximum salary. But, any package above 200K is skeptical. But, considering the Doctors in mind. I will chose the cutoff for 270K.

# In[ ]:


dum = df[(df.FULL_TIME_POSITION == 'Y') & (df.PW_UNIT_OF_PAY == 'Year')]
ind1 = dum[(dum.PREVAILING_WAGE > 270000) | (dum.PREVAILING_WAGE < 40000)].index
df = df.drop(ind1,axis=0)


# - The partime employees might have a minimum salary of 30K-32K(in 2011) and maximum couldnt be more than 150K

# In[ ]:


dum = df[(df.FULL_TIME_POSITION == 'N') & (df.PW_UNIT_OF_PAY == 'Year')]
ind1 = dum[(dum.PREVAILING_WAGE > 150000) | (dum.PREVAILING_WAGE < 32000)].index
df = df.drop(ind1,axis=0)


# #### Hourly Pay
# - The minimum hourly salary should atleast 15. I cant imagine less than 15 cuz gas stations pay 10-12/hour
# - the maximum salary may be around 110/hour.(Purely my guess)

# In[ ]:


dum = df[(df.PW_UNIT_OF_PAY == 'Hour')]
ind1 = dum[(dum.PREVAILING_WAGE > 110) | (dum.PREVAILING_WAGE < 15)].index
df = df.drop(ind1,axis=0)


# ### Converting the hourly pay to annual pay.
# - FULL_TIMERS: As they work 40hours a week and we have 48 weeks in a year, we multiply with 40*48
# - PART_TIMERS: As they work something around 25-35 hours a week, let me take the average of 30 hours a week,So we multiply with 30*48.

# In[ ]:


k = df[(df.PW_UNIT_OF_PAY == 'Hour') & (df.FULL_TIME_POSITION == 'Y')].index
df.loc[k,'PREVAILING_WAGE'] = df.loc[k,'PREVAILING_WAGE'] * 1920


# In[ ]:


k = df[(df.PW_UNIT_OF_PAY == 'Hour') & (df.FULL_TIME_POSITION == 'N')].index
df.loc[k,'PREVAILING_WAGE'] = df.loc[k,'PREVAILING_WAGE'] * 1440


# #### As, we now have all data in year pay scale. We go ahead and remove PW_UNIT_PAY

# In[ ]:


df=df.drop(['PW_UNIT_OF_PAY'],axis=1)


# ### Average Annual pay of each year from 2011-2018

# In[ ]:


t1 = go.Scatter(
    x=df.groupby('year').mean().index,
    y=df.groupby('year').mean().PREVAILING_WAGE
)

layout = go.Layout(dict(title= " AVERAGE ANNUAL PAY vs YEAR",xaxis=dict(title="YEARS"),yaxis=dict(title="AVERAGE ANNUAL PAY")))
data = [t1]
fig = go.Figure(data=data,layout=layout)
iplot(fig)


# ### HOTTEST JOB IN EVERY STATE
# #### - In the plot below hover around the states in map to know more.

# In[ ]:


dum = df[["EMPLOYER_STATE","JOB_TITLE"]]
dum = dum.groupby(["EMPLOYER_STATE","JOB_TITLE"]).size().reset_index()
dum.columns = ['EMPLOYER_STATE', 'JOB_TITLE', "COUNT"]
dum = dum.groupby(['EMPLOYER_STATE', 'JOB_TITLE']).agg({'COUNT':sum})
dum = dum['COUNT'].groupby(level=0, group_keys=False)
dum = dum.apply(lambda x: x.sort_values(ascending=False).head(1))
dum = pd.DataFrame(dum).reset_index()


# In[ ]:


data=[dict(
    type='choropleth',
    locations = dum.EMPLOYER_STATE,
    z = dum.COUNT,
    locationmode = 'USA-states',
    text = dum.JOB_TITLE,
    marker = dict(
            line = dict (
                color = 'rgb(255,255,255)',
                width = 2
            ) ),
        colorbar = dict(
            title = "Number of application")
)]
layout= dict(title="Top job title in the state",geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)'),
             )
fig = dict( data=data, layout=layout )
iplot(fig)


# ### Average salary of H1B employee in each state.
# - As expected, california pays more

# In[ ]:


dum = df.groupby('EMPLOYER_STATE',as_index=False).mean()[['EMPLOYER_STATE','PREVAILING_WAGE']]


# In[ ]:


data=[dict(
    type='choropleth',
    locations = dum.EMPLOYER_STATE,
    z = dum.PREVAILING_WAGE,
    locationmode = 'USA-states',
    marker = dict(
            line = dict (
                color = 'rgb(255,255,255)',
                width = 2
            ) ),
        colorbar = dict(
            title = "Avg salary in USD")
)]
layout= dict(title="Average salaries per state",geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)'),
             )
fig = dict( data=data, layout=layout )
iplot(fig)


# ## IT and TECH Analysis.
# - The following was taken from @DhrumilVora.(Kaggle)
# - It creates a new column occupation based on the key words from the SOC_NAME column.

# In[ ]:


df['OCCUPATION'] = np.nan
df['SOC_NAME'] = df['SOC_NAME'].str.lower()
df.OCCUPATION[df['SOC_NAME'].str.contains('computer','programmer')] = 'Computer Occupations'
df.OCCUPATION[df['SOC_NAME'].str.contains('data scientist','data analyst')] = 'Computer Occupations'
df.OCCUPATION[df['SOC_NAME'].str.contains('data engineer','data base')] = 'Computer Occupations'
df.OCCUPATION[df['SOC_NAME'].str.contains('machine learning','artifical intelligence')] = 'Computer Occupations'
df.OCCUPATION[df['SOC_NAME'].str.contains('spark','apache')] = 'Computer Occupations'
df.OCCUPATION[df['SOC_NAME'].str.contains('hadoop','big data')] = 'Computer Occupations'
df.OCCUPATION[df['SOC_NAME'].str.contains('sql','cyber')] = 'Computer Occupations'
df.OCCUPATION[df['SOC_NAME'].str.contains('developer','full stack')] = 'Computer Occupations'
df.OCCUPATION[df['SOC_NAME'].str.contains('fullstack','etl')] = 'Computer Occupations'
df.OCCUPATION[df['SOC_NAME'].str.contains('data','network')] = 'Computer Occupations'
df.OCCUPATION[df['SOC_NAME'].str.contains('software tester','cloud')] = 'Computer Occupations'
df.OCCUPATION[df['SOC_NAME'].str.contains('information','informatica')] = 'Computer Occupations'
df.OCCUPATION[df['SOC_NAME'].str.contains('jira','programmer')] = 'Computer Occupations'
df.OCCUPATION[df['SOC_NAME'].str.contains('software','web developer')] = 'Computer Occupations'
df.OCCUPATION[df['SOC_NAME'].str.contains('database')] = 'Computer Occupations'
df.OCCUPATION[df['SOC_NAME'].str.contains('math','statistic')] = 'Mathematical Occupations'
df.OCCUPATION[df['SOC_NAME'].str.contains('predictive model','stats')] = 'Mathematical Occupations'
df.OCCUPATION[df['SOC_NAME'].str.contains('teacher','linguist')] = 'Education Occupations'
df.OCCUPATION[df['SOC_NAME'].str.contains('professor','Teach')] = 'Education Occupations'
df.OCCUPATION[df['SOC_NAME'].str.contains('school principal')] = 'Education Occupations'
df.OCCUPATION[df['SOC_NAME'].str.contains('medical','doctor')] = 'Medical Occupations'
df.OCCUPATION[df['SOC_NAME'].str.contains('physician','dentist')] = 'Medical Occupations'
df.OCCUPATION[df['SOC_NAME'].str.contains('Health','Physical Therapists')] = 'Medical Occupations'
df.OCCUPATION[df['SOC_NAME'].str.contains('surgeon','nurse')] = 'Medical Occupations'
df.OCCUPATION[df['SOC_NAME'].str.contains('psychiatr')] = 'Medical Occupations'
df.OCCUPATION[df['SOC_NAME'].str.contains('chemist','physicist')] = 'Advance Sciences'
df.OCCUPATION[df['SOC_NAME'].str.contains('biology','scientist')] = 'Advance Sciences'
df.OCCUPATION[df['SOC_NAME'].str.contains('biologi','clinical research')] = 'Advance Sciences'
df.OCCUPATION[df['SOC_NAME'].str.contains('public relation','manage')] = 'Management Occupation'
df.OCCUPATION[df['SOC_NAME'].str.contains('management','operation')] = 'Management Occupation'
df.OCCUPATION[df['SOC_NAME'].str.contains('chief','plan')] = 'Management Occupation'
df.OCCUPATION[df['SOC_NAME'].str.contains('executive')] = 'Management Occupation'
df.OCCUPATION[df['SOC_NAME'].str.contains('advertis','marketing')] = 'Marketing Occupation'
df.OCCUPATION[df['SOC_NAME'].str.contains('promotion','market research')] = 'Marketing Occupation'
df.OCCUPATION[df['SOC_NAME'].str.contains('business','business analyst')] = 'Business Occupation'
df.OCCUPATION[df['SOC_NAME'].str.contains('business systems analyst')] = 'Business Occupation'
df.OCCUPATION[df['SOC_NAME'].str.contains('accountant','finance')] = 'Financial Occupation'
df.OCCUPATION[df['SOC_NAME'].str.contains('financial')] = 'Financial Occupation'
df.OCCUPATION[df['SOC_NAME'].str.contains('engineer','architect')] = 'Architecture & Engineering'
df.OCCUPATION[df['SOC_NAME'].str.contains('surveyor','carto')] = 'Architecture & Engineering'
df.OCCUPATION[df['SOC_NAME'].str.contains('technician','drafter')] = 'Architecture & Engineering'
df.OCCUPATION[df['SOC_NAME'].str.contains('information security','information tech')] = 'Architecture & Engineering'
df['OCCUPATION']= df.OCCUPATION.replace(np.nan, 'Others', regex=True)
df['SOC_NAME'] = df['SOC_NAME'].str.upper()


# In[ ]:


df.head()


# ## The newly created column contents

# In[ ]:


df.OCCUPATION.value_counts()


# ### The Average annual salaries of the newly created departments.

# In[ ]:


dum = df.groupby('OCCUPATION',as_index = False).mean()[['OCCUPATION','PREVAILING_WAGE']]
t1 =go.Bar(x=dum.OCCUPATION,y=dum.PREVAILING_WAGE,name='wageperoccuaption')
layout = go.Layout(dict(title= " AVERAGE ANNUAL PAY vs OCCUPATION",yaxis=dict(title="AVERAGE ANNUAL PAY")))
data = [t1]
fig = go.Figure(data=data,layout=layout)
iplot(fig)


# ### Working only on the TECH/IT data.

# In[ ]:


dfcomp = df[df.OCCUPATION == 'Computer Occupations']


# ### Average salary of IT H1B employee in each state.
# - As expected, california pays more

# In[ ]:


dum = dfcomp.groupby('EMPLOYER_STATE',as_index=False).mean()[['EMPLOYER_STATE','PREVAILING_WAGE']]


# In[ ]:


data=[dict(
    type='choropleth',
    locations = dum.EMPLOYER_STATE,
    z = dum.PREVAILING_WAGE,
    locationmode = 'USA-states',
    marker = dict(
            line = dict (
                color = 'rgb(255,255,255)',
                width = 2
            ) ),
        colorbar = dict(
            title = "Avg salary in USD")
)]
layout= dict(title="Average salaries of TECH(IT) per state",geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)'),
             )
fig = dict( data=data, layout=layout )
iplot(fig)


# ## TOP 20 MOST PAID FIELDS.
# - Medical field domination 

# In[ ]:


dum = df.groupby('SOC_NAME',as_index=False).mean()[['SOC_NAME','PREVAILING_WAGE']]
dum.sort_values('PREVAILING_WAGE',ascending= False).head(20)


# ## TOP 20 MOST PAID TECH FIELDS.

# In[ ]:


dum = dfcomp.groupby('SOC_NAME',as_index=False).mean()[['SOC_NAME','PREVAILING_WAGE']]
dum.sort_values('PREVAILING_WAGE',ascending= False).head(20)


# ## DATA SCIENCE: 
# - I'm going to do some analysis on our domain.
# - Created a new column DS as shown below. 

# In[ ]:


df['DS'] = np.nan
df.DS[df['JOB_TITLE'].str.contains('DATA SCIENTIST')] = 'DATA SCIENTIST'
df.DS[df['JOB_TITLE'].str.contains('DATA ANALYST')] = 'DATA ANALYST'
df.DS[df['JOB_TITLE'].str.contains('MACHINE LEARNING')] = 'MACHINE LEARNING'
df.DS[df['JOB_TITLE'].str.contains('BUSINESS ANALYST')] = 'BUSINESS ANALYST'
df.DS[df['JOB_TITLE'].str.contains('DEEP LEARNING')] = 'DEEP LEARNING'
df.DS[df['JOB_TITLE'].str.contains('ARTIFICIAL INTELLIGENCE')] = 'ARTIFICIAL INTELLIGENCE'
df.DS[df['JOB_TITLE'].str.contains('BIG DATA')] = 'BIG DATA'
df.DS[df['JOB_TITLE'].str.contains('HADOOP')] = 'HADOOP'
df.DS[df['JOB_TITLE'].str.contains('DATA ENGINEER')] = 'DATA ENGINEER'
df['DS']= df.DS.replace(np.nan, 'Others', regex=True)


# #### Examine the new column we created.

# In[ ]:


df.DS.value_counts()


# In[ ]:


dum = df.groupby('DS',as_index=False).mean()[['DS','PREVAILING_WAGE']]


# In[ ]:


t1 =go.Bar(x=dum.DS,y=dum.PREVAILING_WAGE,name='DataScience')
data = [t1]
iplot(data)


# ### GROWTH IN DATA SCIENCE FROM YEARS

# In[ ]:


dum = df.groupby(['year','DS']).count().reset_index()[['year','DS','countvar']]


# In[ ]:


data = []
for i in dum.DS.unique():
    if i != 'Others':
        data.append(go.Scatter(x = dum[dum.DS == i].year,y= dum[dum.DS == i].countvar,name=i))

layout = go.Layout(dict(title= "GROWTH IN DATA SCIENCE",xaxis=dict(title="YEARS"),yaxis=dict(title="Number of applications")))
        
fig = go.Figure(data,layout)    
iplot(fig)    


# ### DATA SCIENCE JOB AND THE STATE WHICH IT TOPS
# - Means, the respective data science job and the state which has the most number of respective job.
# - To explain, the business analysts are more in New Jersey while the data analysts are more in california.

# In[ ]:


dum = df[["DS","EMPLOYER_STATE"]]
dum = dum.groupby(["DS","EMPLOYER_STATE"]).size().reset_index()
dum.columns = ["DS","EMPLOYER_STATE", "COUNT"]
dum = dum.groupby(["DS","EMPLOYER_STATE"]).agg({'COUNT':sum})
dum = dum['COUNT'].groupby(level=0, group_keys=False)
dum = dum.apply(lambda x: x.sort_values(ascending=False).head(1))
dum = pd.DataFrame(dum).reset_index()
dum[0:-1]


# ### DATA SCIENCE JOB AND THE EMPLOYER WHICH IT TOPS
# - Means, the respective data science job and the Employer which has the most number of respective job.

# In[ ]:


dum = df[["DS","EMPLOYER_NAME"]]
dum = dum.groupby(["DS","EMPLOYER_NAME"]).size().reset_index()
dum.columns = ["DS","EMPLOYER_NAME", "COUNT"]
dum = dum.groupby(["DS","EMPLOYER_NAME"]).agg({'COUNT':sum})
dum = dum['COUNT'].groupby(level=0, group_keys=False)
dum = dum.apply(lambda x: x.sort_values(ascending=False).head(1))
dum = pd.DataFrame(dum).reset_index()
dum[0:-1]


# ### VIRGINIA and DC

# In[ ]:


dfvadc = df[(df.EMPLOYER_STATE == 'VA') | (df.EMPLOYER_STATE == 'DC')]


# In[ ]:


dfvadc = dfvadc[dfvadc.DS != 'Others']


# In[ ]:


dum = dfvadc[["DS","EMPLOYER_NAME"]]
dum = dum.groupby(["DS","EMPLOYER_NAME"]).size().reset_index()
dum.columns = ["DS","EMPLOYER_NAME", "COUNT"]
dum = dum.groupby(["DS","EMPLOYER_NAME"]).agg({'COUNT':sum})
dum = dum['COUNT'].groupby(level=0, group_keys=False)
newdum = dum.apply(lambda x: x.sort_values(ascending=False).head(1))
newdum = pd.DataFrame(newdum).reset_index()
newdum[0:-1]


# ## Companies to FOCUS (for me)

# In[ ]:


pd.DataFrame(dum.apply(lambda x: x.sort_values(ascending=False).head(15))).reset_index()['EMPLOYER_NAME']


# ## END
# - SAITEJA NAKKA
