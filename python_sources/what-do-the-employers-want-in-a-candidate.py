#!/usr/bin/env python
# coding: utf-8

# ## In this notebook we will try to find out what do the employeers look for in a candidate and what do the job seekers look for in a company
# 
# 
# 

# In[35]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import plotly
import plotly.offline as py
from collections import Counter
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
plt.style.use('fivethirtyeight')
py.init_notebook_mode(connected=True)
import os
def Questions_finder(Question_number):
    qlist=[]
    for x in df.columns:
        if x.find(Question_number)!=-1:
            qlist.append(x)
    return qlist

def Create_dictionary(data_f):
    dict1={}
    for i,columns in data_f.iterrows():
        for x in columns:
            if str(x)=="nan" or str(x)=="#NULL!" :
                continue
            if str(x) not in dict1:
                dict1[str(x)]=0
            dict1[str(x)]+=1
    return dict1
df_coded=pd.read_csv("../input/HackerRank-Developer-Survey-2018-Numeric.csv")
df_codebook=pd.read_csv("../input/HackerRank-Developer-Survey-2018-Codebook.csv")
df_mapping=pd.read_csv("../input/HackerRank-Developer-Survey-2018-Numeric-Mapping.csv")
df=pd.read_csv("../input/HackerRank-Developer-Survey-2018-Values.csv")


# ### Before analyzing lets look at the and size of data.

# In[36]:


df.shape


# #### So there are exactly 250 QUESTIONS and 25K respondendants.
# #### Lets look at some of the rows

# In[37]:


df.columns


# Although there are many columns but let us try to find some basic analysis on Gender, Age and Education first
# 
# Lets us look at how many male and female responded

# In[38]:


x=df["q3Gender"].groupby(df["q3Gender"]).count().drop("Non-Binary").drop("#NULL!")
plt.figure(figsize=(12,8))
g = sns.barplot( x=list((x.index)), y=x.values, palette="winter")
plt.title('Male Female Count')
plt.ylabel("Count")
plt.xlabel("Gender")
plt.savefig('Gender Count.png')


# ### it seems that most of the respondents were male
# 
# 
# #### Lets try to find something important, Lets find out the ratio of female respondents with respect to male respondents for the country.
# ### Please let me explain, we see that if lets say country A have 100 female respondents and 900 male respondents 
# ### On the other hadn country B have just 10 female respondents and 1 male respondents.
# ### So the female to male ratio for Country A is 100/900= 100/900=11%. 
# ### Similarly for Country B it will be 10/1= 10/1=10*100=1000%
# ### We are trying to find the number of female respondents in comparision with the male respondents. Just like this we will find the female to male ratio for all countries and lets find out which country have the most female to male ratio
# 

# In[39]:


countries_list=list(df['CountryNumeric'].unique())
female_male_ratio_country={}
for x in countries_list:
    data=df[df.CountryNumeric==x]
    xx=data["q3Gender"].groupby(data["q3Gender"]).count()
    if "Female" not in xx:
        avg=0
    else:
        avg=float(xx["Female"])/float((xx["Male"]))
        avg=avg*100
    avg=int(avg)
    
    female_male_ratio_country[x]=avg
d=Counter(female_male_ratio_country)
plt.figure(figsize=(12,8))
in_tuple=d.most_common()
g = sns.barplot( x=[xx[1] for xx in in_tuple][:5], y=[xx[0] for xx in in_tuple][:5], palette="winter")
plt.title('Female to male ratio')
plt.ylabel("Country")
plt.xlabel("Ratio in Percentage")
plt.savefig('female to male ratio.png')


# ## This implies that in Papua New Guinea there are 3 times more female respondents than male,, BUT WAIT dont let this graph mislead you
# 

# In[40]:


x=df[df.CountryNumeric=="Papua New Guinea"]["q3Gender"].groupby(df["q3Gender"]).count()
print (x)


# ### As we can see that the number of total respondents from Papua New Guinea were just 8 and 6 of them were females hence our calculation made Papua New Guinea to be the country with more female to male ration
# ### At this point I could not think of some other way to scale the female to male ratio to get the idea of which country have more female respondents comparatively while keeping into mind the total number of participants.
# ### Please suggests your methodology/suggesstion for this analysis in comments.

# ### Lets find out which age group were most found amongst the respondents

# In[41]:


x=df["q2Age"].groupby(df["q2Age"]).count().drop("#NULL!")
plt.figure(figsize=(12,8))
g = sns.barplot( y=list((x.index)), x=x.values, palette="winter")
plt.title('Bar Chart showing the AGE group')
plt.ylabel("Count")
plt.xlabel("Age")
plt.savefig('Age_chart.png')


# ### From a job seekers perspective which thing is required most in the company?
# ### Lets find out

# In[42]:


Q12_df=df[Questions_finder("q12")]
Top_in_company=Create_dictionary(Q12_df)
for_plot=Counter(Top_in_company)
to_plot=for_plot.most_common()
plt.figure(figsize=(12,8))
g = sns.barplot( y=[x[0] for x in to_plot], x=[x[1] for x in to_plot], palette="winter")
plt.title('Bar Chart Showing most required things in a company from a job seeker perspective')
plt.ylabel("Things Required")
plt.xlabel("number of people requiring it")
plt.savefig('things_required_in_a_company.png')


# ### It seems that most of the employees require a professional growth and learning in a job
# ### Work Life Balance comes second.
# ### We can clearly see that Stability is amongst the least thing required in a company from a job seeker.

# In[43]:


#what is required in a job by country and age:
Q12_df=df[df.CountryNumeric=="India"][Questions_finder("q12")]
Top_in_company=Create_dictionary(Q12_df)
for_plot=Counter(Top_in_company)
to_plot=for_plot.most_common()
plt.figure(figsize=(12,8))
g = sns.barplot( y=[x[0] for x in to_plot], x=[x[1] for x in to_plot], palette="winter")
plt.title('Bar Chart Showing most required things in a company from a job seeker perspective in India')
plt.ylabel("Things Required")
plt.xlabel("number of people requiring it")
plt.savefig('things_required_in_a_company.png')


# In[44]:


#what is required in a job by country and age:
Q12_df=df[df.CountryNumeric=="United States"][Questions_finder("q12")]
Top_in_company=Create_dictionary(Q12_df)
for_plot=Counter(Top_in_company)
to_plot=for_plot.most_common()
plt.figure(figsize=(12,8))
g = sns.barplot( y=[x[0] for x in to_plot], x=[x[1] for x in to_plot], palette="winter")
plt.title('Bar Chart Showing most required things in a company from a job seeker perspective in US')
plt.ylabel("Things Required")
plt.xlabel("number of people requiring it")
plt.savefig('things_required_in_a_company.png')


# # Based on your last job hunting experience, how did employers measure your skills?Check all that apply
# ### Now that we have seen what a job seeker wants in a company
# ### Lets find out HOW do employers check/measure/quantize the skill of a person
#    

# In[45]:


q13_df=Questions_finder("q13")
q13_df=df[q13_df]
q13_df
evaluation_technique=Create_dictionary(q13_df)
for_plot=Counter(evaluation_technique)

to_plot=for_plot.most_common()
plt.figure(figsize=(12,8))
g = sns.barplot( y=[x[0] for x in to_plot], x=[x[1] for x in to_plot], palette="winter")
plt.title('How a skill is quantized ')
plt.ylabel("Method")
plt.xlabel("Count")
plt.savefig('skill_measure.png')


# ### Clearly Resume + technical phone interview remains amongst the top methodology
# ### So I should certainly upgrade my resume now.
# 
# 
# 

# In[46]:


q13_df=Questions_finder("q13")
q13_df=df[df.CountryNumeric=="India"][q13_df]
q13_df
evaluation_technique=Create_dictionary(q13_df)
for_plot=Counter(evaluation_technique)

to_plot=for_plot.most_common()
plt.figure(figsize=(12,8))
g = sns.barplot( y=[x[0] for x in to_plot], x=[x[1] for x in to_plot], palette="winter")
plt.title('How a skill is quantized in India')
plt.ylabel("Method")
plt.xlabel("Count")
plt.savefig('skill_measure_india.png')


# In[47]:


q13_df=Questions_finder("q13")
q13_df=df[df.CountryNumeric=="United States"][q13_df]
q13_df
evaluation_technique=Create_dictionary(q13_df)
for_plot=Counter(evaluation_technique)

to_plot=for_plot.most_common()
plt.figure(figsize=(12,8))
g = sns.barplot( y=[x[0] for x in to_plot], x=[x[1] for x in to_plot], palette="winter")
plt.title('How a skill is quantized in United States of America')
plt.ylabel("Method")
plt.xlabel("Count")
plt.savefig('skill_measure_america.png')


# In[48]:


q13_df=Questions_finder("q13")
q13_df=df[df.CountryNumeric=="Ghana"][q13_df]
q13_df
evaluation_technique=Create_dictionary(q13_df)
for_plot=Counter(evaluation_technique)

to_plot=for_plot.most_common()
plt.figure(figsize=(12,8))
g = sns.barplot( y=[x[0] for x in to_plot], x=[x[1] for x in to_plot], palette="winter")
plt.title('How a skill is quantized in Ghana')
plt.ylabel("Method")
plt.xlabel("Count")
plt.savefig('skill_measure_ghana.png')


# In[49]:


q13_df=Questions_finder("q13")
q13_df=df[df.CountryNumeric=="Pakistan"][q13_df]
q13_df
evaluation_technique=Create_dictionary(q13_df)
for_plot=Counter(evaluation_technique)

to_plot=for_plot.most_common()
plt.figure(figsize=(12,8))
g = sns.barplot( y=[x[0] for x in to_plot], x=[x[1] for x in to_plot], palette="winter")
plt.title('How a skill is quantized in Pakistan')
plt.ylabel("Method")
plt.xlabel("Count")
plt.savefig('skill_measure_pakistan.png')


# ### Lets find out how many hiring managers participated in the survey

# In[50]:


# lets find out how many hiring managers participated in a survey
# Do you interview people as part of your company's hiring process? q16
q16_df=Questions_finder("q16")
q16_df=df[q16_df]
x=q16_df.q16HiringManager.groupby(q16_df.q16HiringManager).count()
plt.figure(figsize=(12,8))
g = sns.barplot( x=list((x.index)), y=x.values, palette="winter")
plt.title('Hiring Manager?')
plt.ylabel("Count")
plt.xlabel("Response")
plt.savefig('How_many_hiring Managers.png')


# ## There were approximately 7500 respondents who were Hiring Managers, Lets look at where do the manager respondents who participated in hackerrank survey come from:
# Acknowledgement: the code for plotting the world map is taken  from I,Coder Kernel: https://www.kaggle.com/ash316/coders-not-hackers-hackerrank

# In[51]:


## Lets find out where in the world do most of the managers come from.
py.init_notebook_mode(connected=True)
newdf=df[df.q16HiringManager=="Yes"]
countries=newdf['CountryNumeric'].value_counts().to_frame()
data = [ dict(
        type = 'choropleth',
        locations = countries.index,
        locationmode = 'country names',
        z = countries['CountryNumeric'],
        text = countries['CountryNumeric'],
        colorscale ='Viridis',
        autocolorscale = False,
        reversescale = False,
        marker = dict(
            line = dict (
                color = 'rgb(180,180,180)',
                width = 0.5
            ) ),
        colorbar = dict(
            autotick = False,
            tickprefix = '',
            title = 'Survey Respondents who are Managers'),
      ) ]

layout = dict(
    title = 'Hiring Managers By Nationality',
    geo = dict(
        showframe = False,
        showcoastlines = True,
        projection = dict(
            type = 'Mercator'
        )
    )
)
fig = dict( data=data, layout=layout )
py.iplot( fig, validate=False, filename='survey-world-map')


# ### It seems that most of the managers who participated in the competition are from India, followed by US.

# In[58]:


# Female Managers per country
py.init_notebook_mode(connected=True)

newdf=df[df.q3Gender=="Female"]
newdf=newdf[df.q16HiringManager=="Yes"]

countries=newdf['CountryNumeric'].value_counts().to_frame()
data = [ dict(
        type = 'choropleth',
        locations = countries.index,
        locationmode = 'country names',
        z = countries['CountryNumeric'],
        text = countries['CountryNumeric'],
        colorscale ='Viridis',
        autocolorscale = False,
        reversescale = False,
        marker = dict(
            line = dict (
                color = 'rgb(180,180,180)',
                width = 0.5
            ) ),
        colorbar = dict(
            autotick = False,
            tickprefix = '',
            title = 'Survey Respondents who are Female Managers'),
      ) ]

layout = dict(
    title = 'Female Hiring Managers By Nationality',
    geo = dict(
        showframe = False,
        showcoastlines = True,
        projection = dict(
            type = 'Mercator'
        )
    )
)
fig = dict( data=data, layout=layout )
py.iplot( fig, validate=False, filename='Female Managers by country')


# ### It can be seen that US have more female managers than India. 
# ### The difference is not that much as US have 226 Female Managers and India have 197

# In[53]:


## Now we are going to find out what these 
challenges_faced_by_managers=Questions_finder("q17")
challenges_faced_by_managers=df[challenges_faced_by_managers]
challenges_count=Create_dictionary(challenges_faced_by_managers)
print (challenges_count)
for_plot=Counter(challenges_count)
to_plot=for_plot.most_common()
plt.figure(figsize=(12,8))
g = sns.barplot( y=[x[0] for x in to_plot], x=[x[1] for x in to_plot], palette="winter")
plt.title('Challenges Faced By Managers')
plt.ylabel("Challenges")
plt.xlabel("")
plt.savefig('Challenges_faced.png')


# ### Difficulty to assess skills  seems to be the most prominent challenge faced by a hiring manager
# 
# 

# ### What do they look for in a candidate?

# In[54]:


#WHAT TO THE MANAGERS LOOK FOR IN A CANDIDATE
ideal_candidate=df[Questions_finder("q20")]
x=Create_dictionary(ideal_candidate)
for_plot=Counter(x)
to_plot=for_plot.most_common()
plt.figure(figsize=(12,8))
g = sns.barplot( y=[x[0] for x in to_plot], x=[x[1] for x in to_plot], palette="winter")
plt.title('Ideal Candidate')
plt.ylabel("")
plt.xlabel("")
plt.savefig('idea_Candidate.png')


# ### In case you dont have an experience then Github or personal projects portfolio can be your best bet.
# 
# 
# ### Lets come down to the most important part (in my opinion) LANGUAGE WARS..
# ### From the perspective of the hiring manager which language is most preferrable?

# In[ ]:


#language wars
#	Which of these core competencies do you look for in software developer candidates? Check all that apply.
ideal_language=df[Questions_finder("q22")]
x=Create_dictionary(ideal_language)
for_plot=Counter(x)
to_plot=for_plot.most_common()
plt.figure(figsize=(12,8))
g = sns.barplot( y=[x[0] for x in to_plot], x=[x[1] for x in to_plot], palette="winter")
plt.title('Core Language for a software developer all round the world')
plt.ylabel("")
plt.xlabel("")
plt.savefig('languages_required.png')


# ### Javascript takes the throne
# ### My favourite language python comes amongst the top 3. 

# In[ ]:


#language wars
#	Which of these core competencies do you look for in software developer candidates? Check all that apply.
ideal_language=df[df.CountryNumeric=="India"][Questions_finder("q22")]
x=Create_dictionary(ideal_language)
for_plot=Counter(x)
to_plot=for_plot.most_common()
plt.figure(figsize=(12,8))
g = sns.barplot( y=[x[0] for x in to_plot], x=[x[1] for x in to_plot], palette="winter")
plt.title('Core Language for a software developer in India')
plt.ylabel("")
plt.xlabel("")
plt.savefig('languages_required.png')


# In[ ]:


#language wars
#	Which of these core competencies do you look for in software developer candidates? Check all that apply.
ideal_language=df[df.CountryNumeric=="Pakistan"][Questions_finder("q22")]
x=Create_dictionary(ideal_language)
for_plot=Counter(x)
to_plot=for_plot.most_common()
plt.figure(figsize=(12,8))
g = sns.barplot( y=[x[0] for x in to_plot], x=[x[1] for x in to_plot], palette="winter")
plt.title('Core Language for a software developer in Pakistan')
plt.ylabel("")
plt.xlabel("")
plt.savefig('languages_required.png')


# ### TO BE CONTINUED, I will update it with more information in the future.
# #### Please upvote it if you find it useful and do give your feedback in comments
