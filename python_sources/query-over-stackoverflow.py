#!/usr/bin/env python
# coding: utf-8

# # Stack Overflow Developers Survey 

# Hi there !!!! This is my second analysis with survey data.The earlier one was with the super famous **Kaggle ML and DataScience Survey** which provided an awesome platform for data enthusiasts to play around with data and learn a lot .Heres my [link](https://www.kaggle.com/gsdeepakkumar/decoding-the-sexiest-job-of-21st-century) to that kernel.Hope you find that exciting and useful .Now lets dive straight into this dataset.I feel obliged to mention is that **this dataset, being a product of an online survey, is extremely prone to bias.** We will, however, not look too much into this unless absolutely required
# 
# **Credits** - I would like to thank [sban](https://www.kaggle.com/shivamb) ,[Leonardo Ferreira](https://www.kaggle.com/kabure), [SRK](https://www.kaggle.com/sudalairajkumar), [Anisotropic](https://www.kaggle.com/arthurtok) , [Icoder](https://www.kaggle.com/ash316) whose works I have referred to for creating this kernel.As a novice in python language,I have learnt a lot from these people.A big thank you...

# # Loading Data and Libraries

# In[ ]:


#Render Matplotlib Plots Inline
get_ipython().run_line_magic('matplotlib', 'inline')

#Import the standard Python Scientific Libraries
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

#Suppress Deprecation and Incorrect Usage Warnings 
import warnings
warnings.filterwarnings('ignore')
plt.style.use('fivethirtyeight')


# In[ ]:


# Load the dataset into pandas dataframe
result = pd.read_csv('../input/survey_results_public.csv',low_memory=False)
result.shape


# We have data over 98855 people.According to the data description,there were 20,000 respondents whose responses were not captured since they did not answer enough questions.Of the qualified responses,only 67,441 completed the entire survey.Therefore we may anticipate lot of the columns to have missing values.

# In[ ]:


result.head()


# Let us analyse the geographical information first.But before that,lets understand about how the survey takers found the survey.

# #### Survey Summary:

# In[ ]:


plt.figure(figsize=(8,8))
g=sns.countplot(x='SurveyTooLong',data=result,palette=sns.color_palette(palette="viridis"),order=result['SurveyTooLong'].dropna().value_counts().index)
g.set_xticklabels(g.get_xticklabels(),rotation=90)
g.set_xlabel("Response")
g.set_ylabel("Count")
g.set_title("How do the respondents feel about the survey ?")


# We find that the response was mixed - since the difference between the two bars are not too far.Let us compare them with gender.

# In[ ]:


gend=result.set_index('SurveyTooLong').Gender.str.split(";",expand=True).stack().reset_index('SurveyTooLong').dropna()
gend.columns=['SurveyTooLong','Gender']
plt.figure(figsize=(12,10))
g=sns.countplot(x='SurveyTooLong',data=gend,hue=gend['Gender'],palette=sns.color_palette(palette="Set3"),order=gend['SurveyTooLong'].dropna().value_counts().index)
g.set_xticklabels(g.get_xticklabels(),rotation=90)
g.set_xlabel("Response")
g.set_ylabel("Count")
g.set_title("How do the respondents feel about the survey ? - Gender SplitUp")


# We find that the survey has overrepresented one particular gender.This means that most of the responses will be a sample representation coming from one gender.Let us see the difficulty level experienced by the respondents.

# In[ ]:


plt.figure(figsize=(8,8))
g=sns.countplot(result['SurveyEasy'],data=result,palette=sns.color_palette(palette="PiYG"),order=result['SurveyEasy'].value_counts().index)
g.set_xticklabels(g.get_xticklabels(),rotation=90)
g.set_xlabel('Response')
g.set_ylabel('Count')
g.set_title('How did you find the survey ?')


# We find that more than 20,000 of the repondents have felt that the survey was only somewhat easy.A quick skimming of the columns indicates the reason why people have said this.Fear of AI,Ethics related questions,lot of questions dealing with tools might be some of the reasons why people have responded this.A fraction of people have also indicated that the survey was very difficult.

# # Demographic Information

# In[ ]:


print('Total Number of Countries with respondents:',result.Country.nunique())
print('Country with highest respondents:',result.Country.value_counts().index[0],'number of respondents:',result.Country.value_counts().values[0])


# ## Gender

# In[ ]:


gender=result['Gender'].str.split(';')
gend=[]
for i in gender.dropna():
    gend.extend(i)
pd.Series(gend).value_counts().sort_values(ascending=True).plot.barh(width=0.9,color=sns.color_palette('gnuplot2'))


# The plot clearly shows that there were more male respondents while there were low respondents from other gender.

# ## Country

# Since there are 183 unique contries,we visualise the top 10 .

# In[ ]:


plt.figure(figsize=(12,8))
result['Country'].value_counts()[:10].plot.barh(width=0.9,color=sns.color_palette('viridis_r',10))
plt.title('Respondents Country',size=15)
for i,v in enumerate(result['Country'].value_counts()[:10].values):
    plt.text(0.5,i,v,fontsize=10,color='white',weight='bold')
    
    


# We find that large number of respondents are from US followed by India and Germany.The difference between each of the countries is about ~7000 respondents.Only after Canada we find that there is a difference in terms of hundreds.

# # Salary

# In[ ]:


print(" % of respondents who prefer not to share their salary:",round((result['Salary'].isnull().sum()/result.shape[0])*100,2),"%")


# We find that almost half the number of respondents prefer not to mention their salary.

# In[ ]:


result['Salary']=result['Salary'].astype(str)
result['Salary']=result['Salary'].apply(lambda x: np.nan  if (pd.isnull(x)) or (x=='-') or (x==0) else float(x.replace(",","")))
result['ConvertedSalary']=result['ConvertedSalary'].astype(str)
result['ConvertedSalary']=result['ConvertedSalary'].apply(lambda x: np.nan  if (pd.isnull(x)) or (x=='-') or (x==0) else float(x.replace(",","")))


# Now that we have cleaned the column,lets visualise the salary statistics for top countries - US ,India and Germany.

# In[ ]:


salary=result[(result['Salary'].notnull())]
salary[salary['Country']=='United States']['Salary'].describe()


# In[ ]:


salary[salary['Country']=='India']['Salary'].describe()


# In[ ]:


salary[salary['Country']=='Germany']['Salary'].describe()


# Going by the three countries salary report,prima face we find that the data is affected by large number of outlier that takes the max somewhere in crores...Looking at the median value and pronouncing our verdict can mislead us since we are not sure whether the currency is converted to USD or people have given the salary values in their home currency.Let us log transform the salary variable and visualize.

# In[ ]:


plt.figure(figsize=(10,10))

g=sns.distplot(np.log(result['Salary'].dropna()+1))
g.set_xlabel('Log of Salary',fontsize=16)
g.set_ylabel('Frequency',fontsize=16)
g.set_title('Salary Vs Frequency',fontsize=18)


# The data seems to be skewed towards right.

# Let us see if there is any significant difference between the salaries with the countries.For this we take converted salary and visualise the top 10 countries alone.

# In[ ]:


country=result['Country'].value_counts().sort_values(ascending=False).head(10).reset_index()
country.columns=['Country','Count']
temp=result[result.Country.isin(country['Country'])]
temp.head()


# In[ ]:


plt.figure(figsize=(10,10))
g=sns.boxplot(x='ConvertedSalary',y='Country',data=temp,palette=sns.color_palette(palette='Set1'),linewidth=1.2,saturation=0.8)
g.set_xlabel('Salary',fontsize=10)
g.set_ylabel('Country',fontsize=10)
g.set_title('Country and Salary',fontsize=16)


# We find that there is a difference between the salary levels for each country.US seems to have a higher median value of salary and is also affected by large number of outliers.Lets test this statistically.

# In[ ]:


import statsmodels.api as sm
from statsmodels.formula.api import ols
 
mod = ols('ConvertedSalary ~ Country',
                data=temp).fit()
                
aov_table = sm.stats.anova_lm(mod, typ=2)
print (aov_table)

#https://www.marsja.se/four-ways-to-conduct-one-way-anovas-using-python/


# We find that the p value is very low and <0.05 which means that test is highly significant and there is a significant difference between the compensation levels between the countries.

# ## Hobby Vs Career Satisfaction Vs Employment Type

# Let us try to analyse these three categorical variables and mine some insights.

# In[ ]:


#list(result)
cols = ['Employment','CareerSatisfaction']
hob_yes = result[result['Hobby']=='Yes']
col = sns.light_palette(color='green',as_cmap=True)
pd.crosstab(hob_yes[cols[0]],hob_yes[cols[1]]).style.background_gradient(cmap=col,axis=0,high=0.8)


# The crosstab compares the data for people who have taken up coding as hobby and their employment type and job satisfaction.We find that across the employment type,those who are employed fulltime have expressed interest in coding and stack over flow compared to other employment types.Most of the users with high coding interest are moderately satisfied about their job.While freelancer's have shown interest,retired people have least interest in the hobby ..But I wonder how come they have given their rating for each career satisfaction types !!! Maybe this could have been a reflection of their last job I guess !!!

# In[ ]:


hob_no = result[result['Hobby']=='No']
col = sns.light_palette(color='green',as_cmap=True)
pd.crosstab(hob_no[cols[0]],hob_no[cols[1]]).style.background_gradient(cmap=col,axis=0,high=0.8)


# Similar results can be found for people who have not taken up coding as hobby.

# ## Years of Coding and Career Satisfaction

# According to me,the years of coding and career satisfaction should be correlated.Let us find out if this is the case.

# In[ ]:



#result.YearsCoding.astype('category')
#pd.Categorical(result[cols[0]],categories=['0-2 years','3-5 years','6-8 years','9-11 years','12-14 years','15-17 years','18-20 years'
                                         # '21-23 years','24-26 years','27-29 years','30 or more years'],ordered=True)
#cols = ['YearsCoding','CareerSatisfaction']
#relation =pd.crosstab(result[cols[0]],result[cols[1]])
#relation.plot(kind="bar",figsize=(15,8),width=0.7,title="Career Satisfaction Vs Years of Coding")


# In[ ]:


plt.figure(figsize=(15,8))
g=sns.countplot(x=result['YearsCoding'],hue=result['CareerSatisfaction'],order=['0-2 years','3-5 years','6-8 years','9-11 years','12-14 years','15-17 years','18-20 years','21-23 years','24-26 years','27-29 years','30 or more years'])
g.set_xticklabels(g.get_xticklabels(),rotation=90)
g.set_title("Exploring the Genders by Repayment Interval", fontsize=15)
g.set_xlabel("")
g.set_ylabel("Count Distribuition", fontsize=12)

plt.show()


# Across all the years ,we find that moderately satisfied finds a higher population of people.Representation of people from 3-5 years,6-8 years is high.Among them we find that section of people are moderatly satisfied follwed by extremely satisfied numbers.But the difference between the two numbers is huge.

# ## How many are students ?

# Let us now visualize how many of the community members are students.

# In[ ]:


plt.figure(figsize=(8,8))
g=sns.countplot(x='Student',data=result)
g.set_xticklabels(g.get_xticklabels(),rotation=90)
g.set_xlabel('Student Type')
g.set_ylabel('Count')
g.set_title('Distribution of Students in the Survey',fontsize=16)


# We find that about 70,000 of the community members are not students whereas about ~19000 are full time students and the rest are part time students.This gives an interesting insight that SOF is popular among professions who face lot of real world problems that they can search and get it solved in this community.

# ## Student Vs Country:

# Let us find out from which country are most students from.

# In[ ]:


students = result.loc[result['Student']!='No']
plt.figure(figsize=(8,8))
g=sns.countplot(x='Country',data=students,order=students.Country.value_counts().iloc[:10].index)
g.set_xticklabels(g.get_xticklabels(),rotation=90)
g.set_xlabel('Country')
g.set_ylabel('Count')
g.set_title('Top 10 Countries where Students are from',fontsize=16)


# ## Aspects of Job Opportunity 

# The survey has also incuded some likert scale type of questions which needs to be visualized and understood.Let us consider assessing the job type first.We see that the users are required to assess the importance of the job based on a 10 point scale for the following aspects 
# 
# 1. The industry that I'd be working in 
# 2. The financial performance or funding status of the company or organization
# 3. The specific department or team I'd be working on
# 4. The languages, frameworks, and other technologies I'd be working with
# 5. The compensation and benefits offered
# 6. The office environment or company culture
# 7. The opportunity to work from home/remotely
# 8. Opportunities for professional development
# 9. The diversity of the company or organization
# 10. How widely used or impactful the product or service I'd be working on is
# 
# Let us visualise the responses and try to understand the scenario.

# In[ ]:


cols =['AssessJob1','AssessJob2','AssessJob3','AssessJob4','AssessJob5','AssessJob6','AssessJob7','AssessJob8','AssessJob9','AssessJob10']
assessjob = result[cols].dropna(how='all')
colname=["The industry that I'd be working in","The financial performance or funding status of the company or organization",
         "The specific department or team I'd be working on","The languages, frameworks, and other technologies I'd be working with","The compensation and benefits offered","The office environment or company culture","The opportunity to work from home/remotely","Opportunities for professional development","The diversity of the company or organization","How widely used or impactful the product or service I'd be working on is"]
assessjob.columns=colname
assessjobmlt=pd.melt(assessjob,value_vars=colname)
assessjobmlt.columns = ['Question','Value']
plt.figure(figsize=(8,8))
g=sns.boxplot(x=assessjobmlt['Question'],y=assessjobmlt['Value'],order=colname,palette='Set3',linewidth=1.8)
g.set_xticklabels(g.get_xticklabels(),rotation=90)
g.set_xlabel('')
g.set_yticks([1,2,3,4,5,6,7,8,9,10])
g.set_ylabel('Order of Importance')
g.set_title('Survey Response for Job Type Assessment',fontsize=16)


# I see that visualising the likert type of questions as the best method though there are better ways to visualize.Pls let me know through comments if there are other ways to visualize.
# 
# Coming to the interpretation part we see that the most important aspect which potential candidates are looking for is the compensation and benefits ( which is not surprising !!!!) followed by language& technology ,office culture and opportunities for improvement.

# ## Benefits of the Job 

# Now let us use the same methodology used before to understand the assessment of job benefits package.The survery deals with 11 aspects as given below
# 
# 1. Salary and/or bonuses
# 2. Stock options or shares
# 3. Health insurance
# 4. Parental leave
# 5. Fitness or wellness benefit (ex. gym membership, nutritionist)
# 6. Retirement or pension savings matching
# 7. Company-provided meals or snacks
# 8. Computer/office equipment allowance
# 9. Childcare benefit
# 10. Transportation benefit (ex. company-provided transportation, public transit allowance)
# 11. Conference or education budget
# 
# For a change,I will use the interactive plotly visuals.

# In[ ]:


cols = [result.AssessBenefits1.values,result.AssessBenefits2.values,result.AssessBenefits3.values,result.AssessBenefits4.values,result.AssessBenefits5.values,result.AssessBenefits6.values,result.AssessBenefits7.values,result.AssessBenefits8.values,result.AssessBenefits9.values,result.AssessBenefits10.values,result.AssessBenefits11.values]
colnames=[" Salary and/or bonuses","Stock options or shares","Health insurance","Parental leave","Fitness or wellness benefit (ex. gym membership, nutritionist)",
         "Retirement or pension savings matching"," Company-provided meals or snacks","Computer/office equipment allowance","Childcare benefit","Transportation benefit (ex. company-provided transportation, public transit allowance)"," Conference or education budget"]
trace =[]

for i in range(11):
    trace.append(
        go.Box(
        y=cols[i],
        name=colnames[i],
        
        )
                )
    layout=go.Layout(title="Assessing the job benefits",hovermode='closest',showlegend=False,xaxis=dict(showticklabels=True,tickangle=90))
fig = go.Figure(data=trace,layout=layout)
py.iplot(fig,filename='jobbenefits')
    


# From the plot , it is seen that salary and bonus , Health insurance,retirement/pension saving , computer /office equipment are considered the most important when assesing a job benefits package.

# ## Communication Tools :

# Let us understand the type of communication tools which people use to communicate, coordinate, or share knowledge with coworkers.

# In[ ]:


result['CommunicationTools'].head(10)


# In[ ]:


commtools=result['CommunicationTools'].str.split(';')
tools =[]
for i in commtools.dropna():
    tools.extend(i)
    
tools_series =pd.Series(tools)

plts=tools_series.value_counts().sort_values(ascending=False).to_frame()
#plt.figure(figsize=(8,8))
g=sns.barplot(plts[0],plts.index,palette=sns.color_palette('inferno_r',10),order=plts.iloc[:10].index)
g.set_xlabel('Communication Tool')
g.set_ylabel('Number of respondents')
g.set_title('Top 10 Most used communication tools',fontsize=16)


# ## Education Types:

# We find that as technology evolves,one need to be ahead of the curve inorder to sustain in the market and grow along with the technology.This means that one needs to constantly update with latest technology which can be done through non-degree educations too.With the advent of MOOC platforms like edx and coursera etc this has become much easier.The following question tries to probe this aspect of the community.

# In[ ]:


result['EducationTypes'].head(10)


# In[ ]:


edu=result['EducationTypes'].str.split(';')
edutype=[]
for i in edu.dropna():
    edutype.extend(i)
plt2=pd.Series(edutype).value_counts()[1:10].sort_values(ascending=False).to_frame()
plt2.reset_index(level=0,inplace=True)
plt2.columns=['EduType','Count']
plt2['Percent']=round(plt2['Count']/sum(plt2['Count']) *100,2)

plt.figure(figsize=(8,10))
g=sns.barplot(x='EduType',y='Percent',data=plt2,palette=sns.color_palette('Set2',10))
g.set_xticklabels(g.get_xticklabels(),rotation=90)
g.set_xlabel('Type of Education')
g.set_ylabel('%')
g.set_title('Top 10 Non Degree Education types',fontsize=16)


# We find that more than 23 % of the respondents to this question have taken up MOOC (which mentioned earlier is gaining popularity) as the number 1 form of self learning .The rest of the pack is interesting - ~19 % of them have contributed to open source software and ~17% of them have received on the job training on software development.

# ## Education

# Let us explore more on education related variables.

# In[ ]:


plt.figure(figsize=(7,7))
g=sns.countplot(y=result['UndergradMajor'],order=result['UndergradMajor'].value_counts().index,palette=sns.color_palette('Set2'))
g.set_title('Formal Undergraduate Education',fontsize=16)
g.set_xlabel('Education')
g.set_ylabel('Count')


# It is not a surpise to know that a majority of users of the survey come from an engineering background (CS,or non-CS).

# Let us compare their salaries.Here I will be using the ConvertedSalary column.

# In[ ]:


plt.figure(figsize=(12,7))
g=sns.boxplot(y=result['ConvertedSalary'],x=result['UndergradMajor'],data=result.dropna(),order=result['UndergradMajor'].value_counts().index,palette=sns.color_palette('Set1'))
g.set_xticklabels(g.get_xticklabels(),rotation=90)
g.set_title('Formal Undergraduate Education vs Salary',fontsize=16)
g.set_xlabel('Education')
g.set_ylabel('Salary')


# We find that the data is highly dominated by outliers.Median salary for all grads hovers below 0.2Ml.

# Let us take out CS grads alone and see the education of their parents to find some interesting insight.

# In[ ]:


cs=result[result['UndergradMajor']=='Computer science, computer engineering, or software engineering']
plt.figure(figsize=(7,7))
g=sns.countplot(y=cs['EducationParents'],order=cs['EducationParents'].value_counts().index,palette=sns.color_palette('Set3'))
g.set_title('Formal Education of parents',fontsize=16)
g.set_xlabel('Education')
g.set_ylabel('Count')


# If the survey takers had mentioned exact education type then we could have answered questions like whether they have got inspired from their parents to take up CS.But here the education is generic and we cant conclude anything.We can assume that since masters ,bachelors degree are in large numbers ,a majority of the study could possibly be computer science and hence they could have possibly got inspired by their parents to take up CS.

# ### Developer Type:

# Let us understand the scenario of various developer's present and see if we can mine some good insights comparing it with other variables like job satisfaction,salary ,5 yrs from now etc..

# In[ ]:


dev=result['DevType'].str.split(";")
devtype=[]
for i in dev.dropna():
    devtype.extend(i)
plt2=pd.Series(devtype).value_counts()[1:10].sort_values(ascending=False).to_frame()
plt2.reset_index(level=0,inplace=True)
plt2.columns=['devType','Count']
plt2


# We see that people have given **Student** also as an response.We will remove this and visualise.

# In[ ]:


plt.figure(figsize=(8,8))
sns.barplot(x='devType',y='Count',data=plt2[~plt2['devType'].str.contains('Student')],palette=sns.color_palette(palette="dark"))
plt.title("Dev Type identified through the survey")
plt.xlabel('Dev Type')
plt.xticks(rotation=90)
plt.ylabel("Count")


# We see that there are many full stack developer,front end developer's who have taken the survey.Lets see how their career satisfaction is.

# In[ ]:


dev_sat=result.set_index('CareerSatisfaction').DevType.str.split(';',expand=True).stack().reset_index('CareerSatisfaction').dropna()
dev_sat.columns=['CareerSatisfaction','Job']
dev_sat.head()


# In[ ]:


cm = sns.light_palette("yellow", as_cmap=True)
pd.crosstab(dev_sat['Job'], dev_sat['CareerSatisfaction'],normalize='index').style.background_gradient(cmap = cm)


# We can see that  Backend developers are only slightly satisfied about their job.Full stack developer ,the population which is majorly represented in the survey have moderate satisfaction of their job.0.28 % of the total Csuite executives and 25 % of the total population of engineering manager are extremely satisfied about the job.

# ## Developers Vs Career Aspirations:

# In[ ]:


dev_asp=result.set_index('HopeFiveYears').DevType.str.split(';',expand=True).stack().reset_index('HopeFiveYears').dropna()
dev_asp.columns=['HopeFiveYears','Job']
dev_asp.head()


# In[ ]:


cm = sns.light_palette("green", as_cmap=True)
pd.crosstab(dev_asp['Job'], dev_asp['HopeFiveYears'],normalize='index').style.background_gradient(cmap = cm,axis=1)


# Pretty interesting findings. 
# 
# 1.Almost all of them have aspirations for working in a different specialisation than the current one after 5 years.
# 
# 2.Also second in the rank is that people expect starting their own venture after 5 years which looks like an ideal 5 year growth plan for all irrespective of the role's they are currently in.
# 
# 3.C Suite executives opting for their own startups or retirement is expected.
# 
# 4.Less than 2 % of the people have mentioned that they would be working in a career totally unrelated to software development.

# In[ ]:




