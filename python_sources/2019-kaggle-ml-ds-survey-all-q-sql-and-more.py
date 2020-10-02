#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


print('--------------------------------------------------------')
print('------+--+--------------+------+++--+++-+-+++-----------')
print('------+-+---------------+--------+--+-+-+-+-+-----------')
print('------++---+++--+++-+++-+-+++----+--+-+-+-+++-----------')
print('------+-+--+-+--+-+-+-+-+-+-+----+--+-+-+---+-----------')
print('------+--+-++++-+++-+++-+-+++---+++++++-+-+++-----------')
print('------------------+---+---------------------------------')
print('------------------+---+---------------------------------')
print('------------------+---+---------------------------------')
print('--------------------------------------------------------')


# # Introduction
# 
# I am very excited to analyze Kaggle's annual survey data. Not just because there it is plenty of graphs to make. But it is my own curiosity to know which programming languages are most popular among other developers, which tools do they use for data analysis and more... You can divide my analysis in two main parts. [First](#first) plot all the questions and for a sake of my own curiosity compare them with SQL users data. [Second](#second) part is just questions that I was interested to ask after analyzing all general questions. After every graph I leave a short comment. If you find something interesting leave a comment below. I would be happy to start a discussion ! Or may be make further inquiries! I hope you get some new information about other developers from my data or atleast something new about SQL users. Have fun!

# In[ ]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import plotly.graph_objects as go
import numpy as np


get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


multiple_choice_responses_2019 = pd.read_csv('/kaggle/input/kaggle-survey-2019/multiple_choice_responses.csv')


# In[ ]:


multiple_choice_responses_2018 = pd.read_csv('/kaggle/input/kaggle-survey-2018/multipleChoiceResponses.csv')


# In[ ]:


multiple_choice_responses_2017 = pd.read_csv('/kaggle/input/kaggle-survey-2017/multipleChoiceResponses.csv', encoding='ISO-8859-1')


# # Content 
# 
#    [First Part.](#first)<br>
# 1. [Respondents age](#Q1)<br>
# 2. [Gender of respondents](#Q2)<br>
#     2.1. [The gender of SQL users](#Q2-SQL)<br>
# 3. [The country of residence](#Q3)<br>
#     3.1. [The country of residence for SQL users](#Q3-SQL)<br>
# 4. [The highest level of formal education](#Q4)<br>
#     4.1. [The highest level of formal education of SQL users](#Q4-SQL)<br>
# 5. [The title most similar to your current](#Q5)<br>
#     5.1. [The title most similar to your current of SQL users](#Q5-SQL)<br>
# 6. [The size of the company where you work](#Q6)<br>
# 7. [ Approximate number of individuals are responsible for data science workloads at your place of business](#Q7)<br>
# 8. [Employer incorporate machine learning methods into their business](#Q8)<br>
# 9. [Important part of your role at work](#Q9)<br>
#     9.1. [Important part of your role at work for SQL users](#Q9-SQL)<br>
# 10. [Yearly compensation](#Q10)<br>
# 11. [Approximately money have you spent on machine learning and/or cloud computing products at your work in the past 5 years](#Q11)<br>
# 12. [Your favorite media sources that report on data science topics](#Q12)<br>
# 13. [Platforms that you have begun or completed data science courses](#Q13)<br>
# 14. [The primary tool that you use at work or school to analyze data](#Q14)<br>
# 15. [Time span you have been writing code to analyze data](#Q15)<br>
# 16. [IDE you are using for regural basis](#Q16)<br>
# 17. [Following hosted notebook products do you use on a regular basis](#Q17)<br>
# 18. [Language popularity](#Q18)<br>
# 19. [Programming language would you recommend an aspiring data scientist to learn first](#Q19)<br>
# 20. [Visualization libraries or tools do you use on a regular basis](#Q20)<br>
# 21. [Types of specialized hardware do you use on a regular basis](#Q21)<br>
# 22. [Use of a TPU (tensor processing unit)](#Q22)<br>
# 23. [Use of machine learning methods](#Q23)<br>
# 24. [ML algorithms do you use on a regular basis](#Q24)<br>
# 25. [Categories of ML tools do you use on a regular basis](#Q25)<br>
# 26. [Categories of computer vision methods do you use on a regular basis](#Q26)<br>
# 27. [The following natural language processing (NLP) methods do you use on a regular basis](#Q27)<br>
# 28. [Machine learning frameworks do you use on a regular basis](#Q28)<br>
# 29. [Computer platforms you use on regular basis](#Q29)<br>
# 30. [Specific cloud computing products do you use on a regular basis](#Q30)<br>
# 31. [Specific big data / analytics products do you use on a regular basis](#Q31)<br>
# 32. [Following machine learning products do you use on a regular basis](#Q32)<br>
# 33. [Automated machine learning tools (or partial AutoML tools) do you use on a regular basis](#Q33)<br>
# 34. [The following relational database products do you use on a regular basis](#Q34)<br>
# [Second part. Questions and answers.](#second)<br>
# 35. [Countries of junior, middle-level and senior developers](#Q35)<br>
# 36. [Programming language popular among junior, middle-level and senior developers](#Q36)<br>
# 37. [IDE popular among junior, middle-level and senior developers](#Q37)<br>
# 38. [Relational database popular among junior, middle-level and senior developers](#Q38)<br>
# 39. [Yearly compensation among junior, middle-level and senior developers](#Q39)<br>
# 40. [Yearly compensation depending of which progamming language you use](#Q40)<br>
# 41. [Countries where highly paid Pythonistas come from](#Q41)<br>
# 42. [Pythonistas favorite media sources that report on data science topics](#Q42)<br>
# 43. [Pythonistas favorite ML algorithms](#Q43)<br>
# 44. [Platforms that Pythonistas have begun or completed data science courses](#Q44)<br>

# #########################################################
# # First part <a class="anchor" id="first"></a>#########
# #########################################################

# In[ ]:


# Building a dictionary for age of paricipants in 2017 survey. 

age_dic = {}

for age_range in multiple_choice_responses_2018.Q2.value_counts().index[:-2]:

    
    age_range_list = list(range(int(age_range.split('-')[0]),int(age_range.split('-')[1]) +1))
    
    age_count_list = []
    
   
    for age in age_range_list:
        
        age_count = sum(multiple_choice_responses_2017.Age == age)
        
        age_count_list.append(age_count)
        
    age_dic[age_range] = sum(age_count_list)
age_dic['80+'] = sum(multiple_choice_responses_2017.Age >= 80)
age_df_2017 = pd.DataFrame(age_dic.values(),age_dic.keys()) # Corrected df for age of respondens.


# ## Q1-Q2
# ### Respondents age<a class="anchor" id="Q1"></a>
# 

# In[ ]:



age_order_2017 = ['18-21','22-24','25-29','30-34','35-39','40-44','45-49','50-54','55-59','60-69','70-79','80+']
age_values_2017 = age_df_2017.T[age_order_2017].values[0]

age_order_2018 = ['18-21','22-24','25-29','30-34','35-39','40-44','45-49','50-54','55-59','60-69','70-79','80+']
age_values_2018 = multiple_choice_responses_2018.Q2.value_counts()[age_order_2018].values

age_order_2019 = ['18-21','22-24','25-29','30-34','35-39','40-44','45-49','50-54','55-59','60-69','70+']
age_values_2019 = multiple_choice_responses_2019.Q1.value_counts()[age_order_2019].values

age_order_2019_SQL = ['18-21','22-24','25-29','30-34','35-39','40-44','45-49','50-54','55-59','60-69','70+']
age_values_2019_SQL = multiple_choice_responses_2019.Q1[multiple_choice_responses_2019.Q18_Part_3 == "SQL"].value_counts()[age_order_2019].values

age_vals_2017_prc = age_values_2017/sum(age_values_2017)
age_vals_2018_prc = age_values_2018/sum(age_values_2018)
age_vals_2019_prc = age_values_2019/sum(age_values_2019)
age_vals_2019_prc_SQL = age_values_2019_SQL/sum(age_values_2019_SQL)

fig = go.Figure(data=[
    go.Bar(name='2017 data', x=age_order_2017, y=age_vals_2017_prc,),
    go.Bar(name='2018 data', x=age_order_2018, y=age_vals_2018_prc),
    go.Bar(name='2019 data', x=age_order_2019, y=age_vals_2019_prc),
    go.Bar(name='2019 data SQL', x=age_order_2019_SQL, y=age_vals_2019_prc_SQL),
    
])
# Change the bar mode
fig.update_layout(barmode='group')
fig.update_layout(title_text='Respodents age',yaxis=dict(
        title='Percent',
        titlefont_size=16,
        tickfont_size=14,))
fig.show()


# To make a bar graph comparable with SQL users data I converted counts of age to percent. You can clearly see that most of the population up to 25% percent is 25-29 years age group. You can also see that the SQL population is increased in older groups 30-59. The cause of such distribution might be that the SQL is not the first language you learn for a programming.

# ## Q1-Q2
# ### Gender of respondents<a class="anchor" id="Q2"></a>

# In[ ]:


# Respondents gender.

plt.figure(3, figsize=(20,15))
the_grid = GridSpec(1, 3)

# Q GennderSelect.
# Year 2017.

gender_values_2017 = multiple_choice_responses_2017.GenderSelect.value_counts().values
gender_2017 = multiple_choice_responses_2017.GenderSelect.value_counts().index

plt.subplot(the_grid[0, 0])

my_circle=plt.Circle((0,0), 0.9, color='white')
plt.pie(gender_values_2017,  autopct='%1.1f%%', labels=gender_2017, colors=['skyblue','pink','green','brown'])
p=plt.gcf()
plt.title("Gender distribution in 2017.")
p.gca().add_artist(my_circle)

# Q1.
# Year 2018.

gender_values_2018 = multiple_choice_responses_2018.Q1.value_counts().values[:4]
gender_2018 = multiple_choice_responses_2018.Q1.value_counts().index[:4]

plt.subplot(the_grid[0, 1])

my_circle=plt.Circle((0,0), 0.9, color='white')
plt.pie(gender_values_2018, autopct='%1.1f%%', labels=gender_2018, colors=['skyblue','pink','green','brown'])
p=plt.gcf()
plt.title("Gender distribution in 2018.")
p.gca().add_artist(my_circle)

# Q2.
# Year 2019.

plt.subplot(the_grid[0, 2])

gender_values_2019 = multiple_choice_responses_2019.Q2.value_counts().values[:4]
gender_2019 = multiple_choice_responses_2019.Q2.value_counts().index[:4]


my_circle=plt.Circle((0,0), 0.9, color='white')
plt.pie(gender_values_2019,  autopct='%1.1f%%',labels=gender_2019, colors=['skyblue','pink','green','brown'])
p=plt.gcf()
plt.title("Gender distribution in 2019.")
p.gca().add_artist(my_circle)

plt.show()


# ## Q1-Q2
# ### Gender of respondents of SQL users<a class="anchor" id="Q2-SQL"></a>

# In[ ]:


gender_values_2019 = multiple_choice_responses_2019.Q2[multiple_choice_responses_2019.Q18_Part_3 == "SQL"].value_counts().values[:4]
gender_2019 = multiple_choice_responses_2019.Q2[multiple_choice_responses_2019.Q18_Part_3 == "SQL"].value_counts().index[:4]


my_circle=plt.Circle((0,0), 0.9, color='white')
plt.pie(gender_values_2019,  autopct='%1.1f%%',labels=gender_2019, colors=['skyblue','pink','green','brown'])
p=plt.gcf()
plt.title("Gender distribution in 2019.")
p.gca().add_artist(my_circle)

plt.show()


# As you can see the most of the respondents are male ~82%. The same tendency is also for SQL users.

# ## Q3
# ### The country of residence <a class="anchor" id="Q3"></a>

# In[ ]:


import pycountry

country = multiple_choice_responses_2019.Q3[1:].value_counts().index
country = pd.Series(country)
country = country.replace('United Kingdom of Great Britain and Northern Ireland','United Kingdom')
country = country.replace('United States of America','United States')
country = country.replace('Iran, Islamic Republic of...','Iran')
country = country.replace('Republic of Korea','Other')
country = country.replace('Hong Kong (S.A.R.)','Hong Kong')
country = country.replace('South Korea', 'Korea')

country_values = multiple_choice_responses_2019.Q3[1:].value_counts().values


countries_2019 = []
iso_alpha_2019 = []
countries_vals_2019 = []

for c,v in zip(country,multiple_choice_responses_2019.Q3[1:].value_counts().values):
    
    iso = pycountry.countries.search_fuzzy(c)[0].alpha_3
    pop = v*232009
 
    if c !="Other":
        countries_2019.append(c)
        iso_alpha_2019.append(iso)
        countries_vals_2019.append(pop)


# In[ ]:


df_countries_2019 = pd.DataFrame()
df_countries_2019['country'] = countries_2019
df_countries_2019['iso_alpha'] = iso_alpha_2019
df_countries_2019['pop'] = countries_vals_2019
df_countries_2019['year'] = '2019'


# In[ ]:


country_2017 = multiple_choice_responses_2017.Country.value_counts().index
country_2017 = pd.DataFrame(country_2017, columns=['Country'])
country_2017 = country_2017.replace('People \'s Republic of China', 'China')
country_2017 = country_2017.replace('Republic of China', 'China')
country_2017 = country_2017.replace('South Korea', 'Korea')

country_2017_values = multiple_choice_responses_2017.Country.value_counts().values

countries_2017 = []
iso_alpha_2017 = []
countries_vals_2017 = []

for c,v in zip(country_2017['Country'].values,country_2017_values):

    iso = pycountry.countries.search_fuzzy(c)[0].alpha_3
    pop = v*232009

    if c !="Other":
        countries_2017.append(c)
        iso_alpha_2017.append(iso)
        countries_vals_2017.append(pop)

    
df_countries_2017 = pd.DataFrame()
df_countries_2017['country'] = countries_2017
df_countries_2017['iso_alpha'] = iso_alpha_2017
df_countries_2017['pop'] = countries_vals_2017
df_countries_2017['year'] = '2017'


# In[ ]:


country_2018 = multiple_choice_responses_2018.Q3.value_counts().index
country_2018 = pd.DataFrame(country_2018, columns=['Country'])
country_2018 = country_2018.replace('United Kingdom of Great Britain and Northern Ireland','United Kingdom')
country_2018 = country_2018.replace('I do not wish to disclose my location', 'Other')
country_2018 = country_2018.replace('South Korea', 'Korea')
country_2018 = country_2018.replace('United States of America','United States')
country_2018 = country_2018.replace('Iran, Islamic Republic of...','Iran')
country_2018 = country_2018.replace('Hong Kong (S.A.R.)','Hong Kong')
country_2018 = country_2018.replace('Republic of Korea','Other')


country_2018_values = multiple_choice_responses_2018.Q3.value_counts().values

countries_2018 = []
iso_alpha_2018 = []
countries_vals_2018 = []

for c,v in zip(country_2018['Country'][:-1].values,country_2018_values):
    
    iso = pycountry.countries.search_fuzzy(c)[0].alpha_3
    pop = v*232009
    
    if c !="Other":
        countries_2018.append(c)
        iso_alpha_2018.append(iso)
        countries_vals_2018.append(pop)

    
df_countries_2018 = pd.DataFrame()
df_countries_2018['country'] = countries_2018
df_countries_2018['iso_alpha'] = iso_alpha_2018
df_countries_2018['pop'] = countries_vals_2018
df_countries_2018['year'] = '2018'


# In[ ]:


# Combining 2017, 2018, 2019 years in one dataframe.

frames = [df_countries_2017, df_countries_2018, df_countries_2019]

df_countries_2017_2018_2019 = pd.concat(frames)


# In[ ]:


import plotly.express as px
fig = px.scatter_geo(df_countries_2017_2018_2019, locations="iso_alpha",
                     hover_name="country", size="pop",
                     animation_frame="year",
                     projection="natural earth")
fig.show()


# In[ ]:


# Q3.
# Country do you currently reside.

country = multiple_choice_responses_2019.Q3[1:].value_counts().index
country = pd.Series(country)
country = country.replace('United Kingdom of Great Britain and Northern Ireland','UK')
country = country.replace('United States of America','USA')
country_values = multiple_choice_responses_2019.Q3[1:].value_counts().values

plt.figure(figsize=(16, 6))
g = sns.barplot(x=country, y=country_values)
g.set_xticklabels(g.get_xticklabels(), rotation=90)
g.set_ylabel("Number of respondens.")

for ix, x in zip(range(len(country_values)+1),country_values):
    g.text(ix,x,x, horizontalalignment='center')
    
plt.title("Country distribution in 2019 survey.")
plt.show()


# ## Q3
# ### The country of residence for SQL users <a class="anchor" id="Q3-SQL"></a>

# In[ ]:


# Q3.
# Country do you currently reside.

country =  multiple_choice_responses_2019.Q3[multiple_choice_responses_2019.Q18_Part_3 == "SQL"].value_counts().index
country = pd.Series(country)
country = country.replace('United Kingdom of Great Britain and Northern Ireland','UK')
country = country.replace('United States of America','USA')
country_values = multiple_choice_responses_2019.Q3[multiple_choice_responses_2019.Q18_Part_3 == "SQL"].value_counts().values



plt.figure(figsize=(16, 6))
g = sns.barplot(x=country, y=country_values)
g.set_xticklabels(g.get_xticklabels(), rotation=90)
g.set_ylabel("Number of respondens.")

for ix, x in zip(range(len(country_values)+1),country_values):
    g.text(ix,x,x, horizontalalignment='center')
    
plt.title("Country distribution in 2019 survey.")
plt.show()


# As you can see in the map that highest respondents rate changes from USA to India. And we also see big incease of users from Nigeria. For SQL users is similar story we see higest respondent rate from USA and India almost equal in count. So we can conclude that nowdays the main players in the data science field are USA and India(If we counting on numbers:)). 

# ## Q4
# ### The highest level of formal education <a class="anchor" id="Q4"></a>

# In[ ]:


education = multiple_choice_responses_2019.Q4.value_counts()[:-1].index
education_values = multiple_choice_responses_2019.Q4.value_counts()[:-1].values

sns.set({'figure.figsize':(6,6)})
my_circle=plt.Circle( (0,0), 0.9, color='white')
plt.pie(education_values, autopct='%1.1f%%', labels=education)
p=plt.gcf()
plt.title("Education distribution in 2019.")
p.gca().add_artist(my_circle)
plt.show()


# ## Q4
# ### The highest level of formal education of SQL users <a class="anchor" id="Q4-SQL"></a>

# In[ ]:


education = multiple_choice_responses_2019.Q4[multiple_choice_responses_2019.Q18_Part_3 == "SQL"].value_counts()[:-1].index
education_values = multiple_choice_responses_2019.Q4[multiple_choice_responses_2019.Q18_Part_3 == "SQL"].value_counts()[:-1].values

sns.set({'figure.figsize':(6,6)})
my_circle=plt.Circle( (0,0), 0.9, color='white')
plt.pie(education_values, autopct='%1.1f%%', labels=education)
p=plt.gcf()
plt.title("Education distribution in 2019.")
p.gca().add_artist(my_circle)
plt.show()


# From the graphs you can see that main respondents about half of respondents have a Master degree and about one third of respondents have a Bachelor's degree. About 14% of respondents have a doctoral degree. There are some minor differences in respondents that are SQL users but trends are the same.

# ## Q5
# ### The title most similar to your current <a class="anchor" id="Q5"></a>

# In[ ]:


degree = multiple_choice_responses_2019.Q5.value_counts()[:-1].index
degree_values = multiple_choice_responses_2019.Q5.value_counts()[:-1].values

sns.set({'figure.figsize':(6,6)})
my_circle=plt.Circle( (0,0), 0.9, color='white')
plt.pie(degree_values, autopct='%1.1f%%', labels=degree)
p=plt.gcf()
plt.title("Degree distribution in 2019.")
p.gca().add_artist(my_circle)
plt.show()


# ## Q5
# ### The title most similar to your current of SQL users <a class="anchor" id="Q5-SQL"></a>

# In[ ]:


degree = multiple_choice_responses_2019.Q5[multiple_choice_responses_2019.Q18_Part_3 == "SQL"].value_counts()[:-1].index
degree_values = multiple_choice_responses_2019.Q5[multiple_choice_responses_2019.Q18_Part_3 == "SQL"].value_counts()[:-1].values

sns.set({'figure.figsize':(6,6)})
my_circle=plt.Circle( (0,0), 0.9, color='white')
plt.pie(degree_values, autopct='%1.1f%%', labels=degree)
p=plt.gcf()
plt.title("Degree distribution in 2019.")
p.gca().add_artist(my_circle)
plt.show()


# About 21% of the respondents in 2019 survey identify themselves as a data scientist. This number is higher in SQL user population where about 30% of the respondents identify themselves as a data scientist. About the same amount(21%) of respondents identify themselves as a Students. This number is lower in SQL users 13%. There could be explanation that would suggest that SQL is used by more experienced users. So in this case these experienced users are data scientists and Data analysts.

# ## Q6
# ### The size of the company where you work <a class="anchor" id="Q6"></a>

# In[ ]:


company_order = ['0-49 employees','50-249 employees','250-999 employees','1000-9,999 employees','> 10,000 employees']
company_index = multiple_choice_responses_2019.Q6.value_counts()[company_order].index
company_values = multiple_choice_responses_2019.Q6.value_counts()[company_order].values
company_values_SQL = multiple_choice_responses_2019.Q6[multiple_choice_responses_2019.Q18_Part_3 == "SQL"].value_counts()[company_order].values

company_vals_prc = company_values/sum(company_values)
company_vals_SQL_prc = company_values_SQL/sum(company_values_SQL)

fig = go.Figure(data=[
    go.Bar(name='2019 data', x=company_order, y=company_vals_prc),
    go.Bar(name='2019 data SQL', x=company_order, y=company_vals_SQL_prc),
    
])
# Change the bar mode
fig.update_layout(barmode='group')
fig.update_layout(title_text='The size of the company where you work',yaxis=dict(
        title='Percent',
        titlefont_size=16,
        tickfont_size=14,))
fig.show()


# More than 25% of respondents work in companies where employee numbers are low 0-49. There are no dramatic differences between general respondents values and SQL users values. But we see slight increase of numbers of SQL users in larger companies.

# ## Q7
# ###  Approximate number of individuals are responsible for data science workloads at your place of business <a class="anchor" id="Q7"></a>

# In[ ]:


numb_of_people_order = ['0','1-2','3-4','5-9','10-14','15-19','20+']
numb_of_people_index = multiple_choice_responses_2019.Q7.value_counts()[numb_of_people_order].index
numb_of_people_values = multiple_choice_responses_2019.Q7.value_counts()[numb_of_people_order].values
numb_of_people_values_SQL = multiple_choice_responses_2019.Q7[multiple_choice_responses_2019.Q18_Part_3 == "SQL"].value_counts()[numb_of_people_order].values

numb_of_people_vals_prc = numb_of_people_values/sum(numb_of_people_values)
numb_of_people_vals_SQL_prc = numb_of_people_values_SQL/sum(numb_of_people_values_SQL)

fig = go.Figure(data=[
    go.Bar(name='2019 data', x=numb_of_people_order, y=numb_of_people_vals_prc),
    go.Bar(name='2019 data SQL', x=numb_of_people_order, y=numb_of_people_vals_SQL_prc),
    
])
# Change the bar mode
fig.update_layout(barmode='group')
fig.update_layout(title_text='Aproximate number of individuals are responsible for data science workloads',yaxis=dict(
        title='Percent',
        titlefont_size=16,
        tickfont_size=14,))
fig.show()


# In the companies where more people(>2) are responsible for data science workloads SQL users tend to get this responsibilities more often than general population.

# ## Q8
# ### Employer incorporate machine learning methods into their business  <a class="anchor" id="Q8"></a>

# In[ ]:


labeling_current_ML_incorporation = multiple_choice_responses_2019.Q8.value_counts()[:-1].index
values_current_ML_incorporation = multiple_choice_responses_2019.Q8.value_counts().values
values_current_ML_incorporation_SQL =  multiple_choice_responses_2019.Q8[multiple_choice_responses_2019.Q18_Part_3 == "SQL"].value_counts().values
values_current_ML_incorporation_prc = values_current_ML_incorporation/sum(values_current_ML_incorporation)
values_current_ML_incorporation_SQL_prc = values_current_ML_incorporation_SQL/sum(values_current_ML_incorporation_SQL)

fig = go.Figure(data=[
    go.Bar(name='2019 data', x=labeling_current_ML_incorporation, y=values_current_ML_incorporation_prc),
    go.Bar(name='2019 data SQL', x=labeling_current_ML_incorporation, y=values_current_ML_incorporation_SQL_prc),
    
])
# Change the bar mode
fig.update_layout(barmode='group')
fig.update_layout(title_text='Employer incorporate machine learning methods into their business',yaxis=dict(
        title='Percent',
        titlefont_size=16,
        tickfont_size=14,))
fig.show()


# All positive answers to question of employer incorporating ML methods in their current acitivities where higher in respondents using SQL. Consequently negative aswerst to the question where in general respondents group.

# ## Q9
# ### Important part of your role at work  <a class="anchor" id="Q9"></a>

# In[ ]:


Q9cols = ['Q9_Part_1','Q9_Part_2','Q9_Part_3','Q9_Part_4','Q9_Part_5','Q9_Part_6','Q9_Part_7','Q9_Part_8']

list_number_important_activities = []
list_important_activities = []

for col in Q9cols:
    important_activities = multiple_choice_responses_2019[col].value_counts().index[0] 
    number_important_activities = multiple_choice_responses_2019[col].value_counts()[0] 
    list_number_important_activities.append(number_important_activities)
    list_important_activities.append(important_activities)

df = pd.DataFrame(list_number_important_activities)
df = df.T
df.columns = list_important_activities

labels = list_important_activities
sizes = list_number_important_activities
fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%',shadow=False, startangle=140)
ax1.axis('equal')
plt.title("Your role at work in 2019 survey.")
plt.show()


# ## Q9
# ### Important part of your role at work for SQL users  <a class="anchor" id="Q9-SQL"></a>

# In[ ]:


Q9cols = ['Q9_Part_1','Q9_Part_2','Q9_Part_3','Q9_Part_4','Q9_Part_5','Q9_Part_6','Q9_Part_7','Q9_Part_8']

list_number_important_activities_SQL = []
list_important_activities = []

for col in Q9cols:
    important_activities = multiple_choice_responses_2019[col].value_counts().index[0] 
    number_important_activities_SQL = multiple_choice_responses_2019[col][multiple_choice_responses_2019.Q18_Part_3 == "SQL"].value_counts()[0] 
    list_number_important_activities_SQL.append(number_important_activities_SQL)
    list_important_activities.append(important_activities)

df = pd.DataFrame(list_number_important_activities_SQL)
df = df.T
df.columns = list_important_activities

labels = list_important_activities
sizes = list_number_important_activities_SQL
fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%',shadow=False, startangle=140)
ax1.axis('equal')
plt.title("Your role at for SQL user")
plt.show()


# As you can see from the pie chart there are not that many differences in roles at work for SQL users and general respondents. 

# ## Q10
# ### Yearly compensation (approximate $USD) <a class="anchor" id="Q10"></a>

# In[ ]:



multiple_choice_responses_2019.Q10.value_counts().index

compensation_order = ['$0-999', '1,000-1,999', '2,000-2,999', '3,000-3,999','4,000-4,999',
                      '5,000-7,499', '7,500-9,999','10,000-14,999','15,000-19,999',  
                      '20,000-24,999','25,000-29,999','30,000-39,999','40,000-49,999', 
                      '50,000-59,999','60,000-69,999','70,000-79,999','80,000-89,999',
                      '90,000-99,999','100,000-124,999','125,000-149,999', 
                      '150,000-199,999','200,000-249,999','250,000-299,999', '300,000-500,000', '> $500,000']

values_compensation = multiple_choice_responses_2019.Q10.value_counts()[compensation_order].values
values_compensation_SQL = multiple_choice_responses_2019.Q10[multiple_choice_responses_2019.Q18_Part_3 == "SQL"].value_counts()[compensation_order].values


values_compensation_prc = values_compensation/sum(values_compensation)
values_compensation_SQL_prc = values_compensation_SQL/sum(values_compensation_SQL)

fig = go.Figure(data=[
    go.Bar(name='2019 data', x=compensation_order, y=values_compensation_prc),
    go.Bar(name='2019 data SQL', x=compensation_order, y=values_compensation_SQL_prc),
    
])
# Change the bar mode
fig.update_layout(barmode='group')
fig.update_layout(title_text="Yearly compensation (approximate $USD) in 2019",yaxis=dict(
        title='Percent',
        titlefont_size=16,
        tickfont_size=14,))
fig.show()


# From the graph we can see that the highest number of respondents obtain the lowest yearly compensation(0-999 USD). It is also interesting to notice that SQL users obtain slightly higher compensation in a salary range from 50,000 USD to 150,000 USD. 

# ## Q11
# ### Approximately money have you spent on machine learning and/or cloud computing products at your work in the past 5 years <a class="anchor" id="Q11"></a>

# In[ ]:



money_spent_order = ['$0 (USD)', '$1-$99', '$100-$999', '$1000-$9,999', '$10,000-$99,999','> $100,000 ($USD)']
money_spent_order_formated_x = ['$0 (USD)', '$1-99', '$100-999', '$1000-9,999', '$10,000-99,999','> $100,000']
values_money_spent = multiple_choice_responses_2019.Q11.value_counts()[money_spent_order].values
values_money_spent_SQL = multiple_choice_responses_2019.Q11[multiple_choice_responses_2019.Q18_Part_3 == "SQL"].value_counts()[money_spent_order].values

values_money_spent_prc = values_money_spent/sum(values_money_spent)
values_money_spent_SQL_prc = values_money_spent_SQL/sum(values_money_spent_SQL)

fig = go.Figure(data=[
    go.Bar(name='2019 data', x=money_spent_order_formated_x, y=values_money_spent_prc),
    go.Bar(name='2019 data SQL', x=money_spent_order_formated_x, y=values_money_spent_SQL_prc),
    
])
# Change the bar mode
fig.update_layout(barmode='group')
fig.update_layout(title_text=" Approximately money have you spent on machine learning and/or cloud computing products at your work",yaxis=dict(
        title='Percent',
        titlefont_size=16,
        tickfont_size=14,))
fig.show()


# We can see that approximately money spent on machine learning and/or cloud computing products at your work in the past 5 years highest percentage (about 33%) is 0$(USD). We can also see the trend that SQL users spend more on machine learning and/or cloud computing products.   

# ## Q12
# ### Your favorite media sources that report on data science topics <a class="anchor" id="Q12"></a>

# In[ ]:


Q12cols = ['Q12_Part_1','Q12_Part_2','Q12_Part_3','Q12_Part_4','Q12_Part_5','Q12_Part_6','Q12_Part_7','Q12_Part_8','Q12_Part_9','Q12_Part_10','Q12_Part_11','Q12_Part_12']

list_number_of_media_sources = []
list_of_media_sources = []
list_number_of_media_sources_SQL = []

for col in Q12cols:
    media_source = multiple_choice_responses_2019[col].value_counts().index[0] 
    number_media_sources = multiple_choice_responses_2019[col].value_counts()[0]
    number_media_sources_SQL = multiple_choice_responses_2019[col][multiple_choice_responses_2019.Q18_Part_3 == "SQL"].value_counts()[0]
    
    list_number_of_media_sources.append(number_media_sources)
    list_of_media_sources.append(media_source)
    list_number_of_media_sources_SQL.append(number_media_sources_SQL)
    
    
list_number_of_media_sources_prc = list_number_of_media_sources/sum(list_number_of_media_sources)
list_number_of_media_sources_SQL_prc =list_number_of_media_sources_SQL/sum(list_number_of_media_sources_SQL)

fig = go.Figure(data=[
    go.Bar(name='2019 data', x=list_of_media_sources, y=list_number_of_media_sources_prc),
    go.Bar(name='2019 data SQL', x=list_of_media_sources, y=list_number_of_media_sources_SQL_prc),
])
# Change the bar mode
fig.update_layout(barmode='group')
fig.update_layout(title_text="Your favorite media sources that report on data science topics",yaxis=dict(
        title='Percent',
        titlefont_size=16,
        tickfont_size=14,)), 
fig.show()


# As you can see from the chart the most popular media sources for respondents are Kaggle, Blogs and YouTube.

# ## Q13
# ### Platforms that you have begun or completed data science courses <a class="anchor" id="Q13"></a>

# In[ ]:


Q13cols = ['Q13_Part_1','Q13_Part_2','Q13_Part_3','Q13_Part_4','Q13_Part_5','Q13_Part_6','Q13_Part_7','Q13_Part_8','Q13_Part_9','Q13_Part_10','Q13_Part_11','Q13_Part_12']

list_science_courses = []
list_number_science_courses = []
list_number_science_courses_SQL = []

for col in Q13cols:
    science_course = multiple_choice_responses_2019[col].value_counts().index[0] 
    number_science_courses = multiple_choice_responses_2019[col].value_counts()[0] 
    number_science_courses_SQL = multiple_choice_responses_2019[col][multiple_choice_responses_2019.Q18_Part_3 == "SQL"].value_counts()[0]
    list_number_science_courses.append(number_science_courses)
    list_science_courses.append(science_course)
    list_number_science_courses_SQL.append(number_science_courses_SQL)
    
    
list_number_science_courses_prc = list_number_science_courses/sum(list_number_science_courses)
list_number_science_courses_SQL_prc = list_number_science_courses_SQL/sum(list_number_science_courses_SQL)

fig = go.Figure(data=[
    go.Bar(name='2019 data', x=list_science_courses, y=list_number_science_courses_prc),
    go.Bar(name='2019 data SQL', x=list_science_courses, y=list_number_science_courses_SQL_prc),
])
# Change the bar mode
fig.update_layout(barmode='group')
fig.update_layout(title_text="Platforms that you have begun or completed data science courses",yaxis=dict(
        title='Percent',
        titlefont_size=16,
        tickfont_size=14,)), 
fig.show()


# ## Q14
# ### The primary tool that you use at work or school to analyze data <a class="anchor" id="Q14"></a>

# In[ ]:


tools = multiple_choice_responses_2019.Q14.value_counts().index[:-1]
tools_number = multiple_choice_responses_2019.Q14.value_counts()[tools].values
tools_number_sql = multiple_choice_responses_2019.Q14[multiple_choice_responses_2019.Q18_Part_3 == "SQL"].value_counts()[tools].values

tools_number_prc = tools_number/sum(tools_number)
tools_number_SQL_prc = tools_number_sql/sum(tools_number_sql)

fig = go.Figure(data=[
    go.Bar(name='2019 data', x=tools, y=tools_number_prc),
    go.Bar(name='2019 data SQL', x=tools, y=tools_number_SQL_prc),
])
# Change the bar mode
fig.update_layout(barmode='group')
fig.update_layout(title_text="The primary tool that you use at work or school to analyze data",yaxis=dict(
        title='Percent',
        titlefont_size=16,
        tickfont_size=14,)), 
fig.show()


# As we can see from the graph the main tools for developent for respondents and SQL users are local development environment 54%. There are slight increases of SQL users in Cloud-based software and Business intelligence software. 

# ## Q15
# ### Time span you have been writing code to analyze data. <a class="anchor" id="Q14"></a>

# In[ ]:


code_writing_time = ['I have never written code','< 1 years', '1-2 years', '3-5 years', '5-10 years', '10-20 years', '20+ years']
numb_code_writing_time = multiple_choice_responses_2019.Q15.value_counts()[code_writing_time].values
numb_code_writing_time_SQL = multiple_choice_responses_2019.Q15[multiple_choice_responses_2019.Q18_Part_3 == "SQL"].value_counts()[code_writing_time].values

#print(numb_code_writing_time_SQL[1:])
code_writing_number_prc = numb_code_writing_time/sum(numb_code_writing_time)
code_writing_number_SQL_prc = numb_code_writing_time_SQL[1:]/sum(numb_code_writing_time_SQL[1:])

code_corrected_number_SQL = np.insert(code_writing_number_SQL_prc,0,0)

fig = go.Figure(data=[
    go.Bar(name='2019 data', x=code_writing_time, y=code_writing_number_prc),
    go.Bar(name='2019 data SQL', x=code_writing_time, y=code_corrected_number_SQL),
])
# Change the bar mode
fig.update_layout(barmode='group')
fig.update_layout(title_text="Time spent writing code to analyze data",yaxis=dict(
        title='Percent',
        titlefont_size=16,
        tickfont_size=14,)), 
fig.show()


# As you see from the chart most respondents are with 1-2 year experience. From the chart you also see that most of the respondents that use SQL programming language have more experience in programming. 

# ## Q16
# ### IDE you are using for regural basis <a class="anchor" id="Q16"></a>

# In[ ]:


Q16cols = ['Q16_Part_1','Q16_Part_2','Q16_Part_3','Q16_Part_4','Q16_Part_5','Q16_Part_6','Q16_Part_7','Q16_Part_8','Q16_Part_9','Q16_Part_10','Q16_Part_11','Q16_Part_12']

list_of_IDE_numbers = []
list_of_IDE = []
list_of_IDE_numbers_SQL = []


for col in Q16cols:
    IDE = multiple_choice_responses_2019[col].value_counts().index[0]
    number_of_IDE = multiple_choice_responses_2019[col].value_counts()[0]
    number_of_IDE_SQL = multiple_choice_responses_2019[col][multiple_choice_responses_2019.Q18_Part_3 == "SQL"].value_counts()[0]
  
    
    list_of_IDE_numbers.append(number_of_IDE)
    list_of_IDE.append(IDE)
    list_of_IDE_numbers_SQL.append(number_of_IDE_SQL)
    
list_of_IDE_numbers_prc = list_of_IDE_numbers/sum(list_of_IDE_numbers)
list_of_IDE_numbers_SQL_prc = list_of_IDE_numbers_SQL / sum(list_of_IDE_numbers_SQL)

fig = go.Figure(data=[
    go.Bar(name='2019 data', x=list_of_IDE, y=list_of_IDE_numbers_prc),
    go.Bar(name='2019 data SQL', x=list_of_IDE, y=list_of_IDE_numbers_SQL_prc),
])
# Change the bar mode
fig.update_layout(barmode='group')
fig.update_layout(title_text="IDE you are using for regural basis",yaxis=dict(
        title='Percent',
        titlefont_size=16,
        tickfont_size=14,)), 
fig.show()


# As we can see from the chart most popular IDE is Jupyter notebook. There is not much difference between general programmers and once that use SQL programming languages. Their preference for IDEs are almost identical. Just slight increase in RStudio and Notepad++.

# ## Q17
# ### Following hosted notebook products do you use on a regular basis <a class="anchor" id="Q17"></a>

# In[ ]:


Q17cols = ['Q17_Part_1','Q17_Part_2','Q17_Part_3','Q17_Part_4','Q17_Part_5','Q17_Part_6','Q17_Part_7','Q17_Part_8','Q17_Part_9','Q17_Part_10','Q17_Part_11','Q17_Part_12']

list_of_hnotebooks_numbers = []
list_of_hnotebooks = []
list_of_hnotebooks_numbers_SQL = []

for col in Q17cols:
    hnotebooks = multiple_choice_responses_2019[col].value_counts().index[0] 
    number_hnotebooks = multiple_choice_responses_2019[col].value_counts()[0] 
    number_hnotebooks_SQL = multiple_choice_responses_2019[col][multiple_choice_responses_2019.Q18_Part_3 == "SQL"].value_counts()[0] 
    
    list_of_hnotebooks_numbers.append(number_hnotebooks)
    list_of_hnotebooks.append(hnotebooks)
    list_of_hnotebooks_numbers_SQL.append(number_hnotebooks_SQL)
    
list_of_hnotebooks_numbers_prc = list_of_hnotebooks_numbers/sum(list_of_hnotebooks_numbers)
list_of_hnotebooks_numbers_SQL_prc = list_of_hnotebooks_numbers_SQL / sum(list_of_hnotebooks_numbers_SQL)

fig = go.Figure(data=[
    go.Bar(name='2019 data', x=list_of_hnotebooks, y=list_of_hnotebooks_numbers_prc),
    go.Bar(name='2019 data SQL', x=list_of_hnotebooks, y=list_of_hnotebooks_numbers_SQL_prc),
])
# Change the bar mode
fig.update_layout(barmode='group')
fig.update_layout(title_text="Following hosted notebook products do you use on a regular basis ",yaxis=dict(
        title='Percent',
        titlefont_size=16,
        tickfont_size=14,)), 
fig.show()


# Some of the most used hosted notebooks are Kaggle Notebooks and Google Colab.

# ## Q18
# ### Language popularity <a class="anchor" id="Q18"></a>

# In[ ]:


Q18cols = ['Q18_Part_1','Q18_Part_2','Q18_Part_3','Q18_Part_4','Q18_Part_5','Q18_Part_6','Q18_Part_7','Q18_Part_8','Q18_Part_9','Q18_Part_10','Q18_Part_11','Q18_Part_12']

list_number_of_users = []
list_of_programming_languages = []
for col in Q18cols:
    programming_language = multiple_choice_responses_2019[col].value_counts().index[0] 
    number_of_users = multiple_choice_responses_2019[col].value_counts()[0] 
    list_number_of_users.append(number_of_users)
    list_of_programming_languages.append(programming_language)
    
g = sns.barplot(x=list_of_programming_languages, y=list_number_of_users)
g.set_xticklabels(g.get_xticklabels(), rotation=90)
plt.title("Popularity of programming languages in 2019 survey")
g.set_ylabel("Number of respondents.")

for ix, x in zip(range(len(list_number_of_users)+1),list_number_of_users):
    g.text(ix,x,x, horizontalalignment='center')

plt.show()


# ## Q19
# ### Programming language would you recommend an aspiring data scientist to learn first <a class="anchor" id="Q19"></a>

# In[ ]:


languages = multiple_choice_responses_2019.Q19.value_counts().index[:-1]
languages_numbers = multiple_choice_responses_2019.Q19.value_counts().values[:-1]

g = sns.barplot(x=languages, y=languages_numbers)
g.set_xticklabels(g.get_xticklabels(), rotation=45)
plt.title("Programming language would you recommend an aspiring data scientist to learn first in 2019.")
g.set_ylabel("Number of respondents.")



for ix, x in zip(range(len(languages_numbers)+1),languages_numbers):
    g.text(ix,x,x, horizontalalignment='center')

plt.show()


# ## Q20
# ### Visualization libraries or tools do you use on a regular basis <a class="anchor" id="Q20"></a>

# In[ ]:


Q20cols = ['Q20_Part_1','Q20_Part_2','Q20_Part_3','Q20_Part_4','Q20_Part_5','Q20_Part_6','Q20_Part_7','Q20_Part_8','Q20_Part_9','Q20_Part_10','Q20_Part_11','Q20_Part_12']

list_viz_tools = []
list_viz_tools_numbers = []
list_viz_tools_numbers_SQL = []

for col in Q20cols:
    viz_tools = multiple_choice_responses_2019[col].value_counts().index[0] 
    viz_tools_numbers = multiple_choice_responses_2019[col].value_counts()[0] 
    viz_tools_numbers_SQL = multiple_choice_responses_2019[col][multiple_choice_responses_2019.Q18_Part_3 == "SQL"].value_counts()[0] 
    list_viz_tools.append(viz_tools)
    list_viz_tools_numbers.append(viz_tools_numbers)
    list_viz_tools_numbers_SQL.append(viz_tools_numbers_SQL)
    
    
list_of_viz_tools_numbers_prc = list_viz_tools_numbers/sum(list_viz_tools_numbers)
list_of_viz_tools_numbers_SQL_prc = list_viz_tools_numbers_SQL / sum(list_viz_tools_numbers_SQL)

fig = go.Figure(data=[
    go.Bar(name='2019 data', x=list_viz_tools, y=list_of_viz_tools_numbers_prc),
    go.Bar(name='2019 data SQL', x=list_viz_tools, y=list_of_viz_tools_numbers_SQL_prc),
])
# Change the bar mode
fig.update_layout(barmode='group')
fig.update_layout(title_text="Visualization libraries or tools do you use on a regular basis ",yaxis=dict(
        title='Percent',
        titlefont_size=16,
        tickfont_size=14,)), 
fig.show()
    


# The graph show that python library Matplotlib is the most popular among developers that use SQL lanuguage and ones that do not use it. Seaborn is second among the most popular vizualization libraries. This shows that most of the SQL developers also use an python programming language. We can test that using following code.

# In[ ]:


Q20cols = ['Q20_Part_1','Q20_Part_2','Q20_Part_3','Q20_Part_4','Q20_Part_5','Q20_Part_6','Q20_Part_7','Q20_Part_8','Q20_Part_9','Q20_Part_10','Q20_Part_11','Q20_Part_12']

list_viz_tools = []
list_viz_tools_numbers = []
list_viz_tools_numbers_SQL = []

for col in Q20cols:
    viz_tools = multiple_choice_responses_2019[col].value_counts().index[0] 
    viz_tools_numbers = multiple_choice_responses_2019[col].value_counts()[0] 
    viz_tools_numbers_SQL = multiple_choice_responses_2019[col][(multiple_choice_responses_2019.Q18_Part_3 == "SQL") & (multiple_choice_responses_2019.Q18_Part_1 != "Python")].value_counts()[0] 
    list_viz_tools.append(viz_tools)
    list_viz_tools_numbers.append(viz_tools_numbers)
    list_viz_tools_numbers_SQL.append(viz_tools_numbers_SQL)
    
    
list_of_viz_tools_numbers_prc = list_viz_tools_numbers/sum(list_viz_tools_numbers)
list_of_viz_tools_numbers_SQL_prc = list_viz_tools_numbers_SQL / sum(list_viz_tools_numbers_SQL)

fig = go.Figure(data=[
    go.Bar(name='2019 data', x=list_viz_tools, y=list_of_viz_tools_numbers_prc),
    go.Bar(name='2019 data SQL', x=list_viz_tools, y=list_of_viz_tools_numbers_SQL_prc),
])
# Change the bar mode
fig.update_layout(barmode='group')
fig.update_layout(title_text="Visualization libraries or tools do you use on a regular basis ",yaxis=dict(
        title='Percent',
        titlefont_size=16,
        tickfont_size=14,)), 
fig.show()


# Now we see huge increase of Ggplot / ggplot2 , Shiny libraries and None values. Meaning that that users who use SQL and do not use Python mostly use R or only use SQL.

# ## Q21
# ### Types of specialized hardware do you use on a regular basis <a class="anchor" id="Q21"></a>

# In[ ]:


Q21cols = ['Q21_Part_1','Q21_Part_2','Q21_Part_3','Q21_Part_4','Q21_Part_5']

list_hardware_tools = []
list_hardware_tools_numbers = []
list_hardware_tools_numbers_SQL = []


for col in Q21cols:
    
    hardware_tools = multiple_choice_responses_2019[col].value_counts().index[0] 
    hardware_tools_numbers = multiple_choice_responses_2019[col].value_counts()[0] 
    hardware_tools_numbers_SQL = multiple_choice_responses_2019[col][multiple_choice_responses_2019.Q18_Part_3 == "SQL"].value_counts()[0] 
    
    list_hardware_tools.append(hardware_tools)
    list_hardware_tools_numbers.append(hardware_tools_numbers)
    list_hardware_tools_numbers_SQL.append(hardware_tools_numbers_SQL)

list_hardware_tools_numbers_prc = list_hardware_tools_numbers/sum(list_hardware_tools_numbers)
list_hardware_tools_numbers_SQL_prc = list_hardware_tools_numbers_SQL / sum(list_hardware_tools_numbers_SQL)

fig = go.Figure(data=[
    go.Bar(name='2019 data', x=list_hardware_tools, y=list_hardware_tools_numbers_prc),
    go.Bar(name='2019 data SQL', x=list_hardware_tools, y=list_hardware_tools_numbers_SQL_prc),
])
# Change the bar mode
fig.update_layout(barmode='group')
fig.update_layout(title_text="Types of specialized hardware do you use on a regular basis",yaxis=dict(
        title='Percent',
        titlefont_size=16,
        tickfont_size=14,)), 
fig.show()


# As you can see from the chart the most used hardware tool in DS is CPU in the second place GPU.

# ## Q22
# ### Use of a TPU (tensor processing unit) <a class="anchor" id="Q22"></a>

# In[ ]:


TPU_usage =  ['Never', 'Once', '2-5 times', '6-24 times', '> 25 times']
TPU_usage_numbers = multiple_choice_responses_2019.Q22.value_counts()[TPU_usage].values
TPU_usage_numbers_SQL = multiple_choice_responses_2019.Q22[multiple_choice_responses_2019.Q18_Part_3 == "SQL"].value_counts()[TPU_usage].values


TPU_usage_numbers_prc = TPU_usage_numbers / sum(TPU_usage_numbers)
TPU_usage_numbers_SQL_prc = TPU_usage_numbers_SQL / sum(TPU_usage_numbers_SQL)

fig = go.Figure(data=[
    go.Bar(name='2019 data', x=TPU_usage, y=TPU_usage_numbers_prc),
    go.Bar(name='2019 data SQL', x=TPU_usage, y=TPU_usage_numbers_SQL_prc),
])
# Change the bar mode
fig.update_layout(barmode='group')
fig.update_layout(title_text="Use of a TPU (tensor processing unit)",yaxis=dict(
        title='Percent',
        titlefont_size=16,
        tickfont_size=14,)), 
fig.show()


# As you can see usage of TPU for general and SQL users are almost the same. Most of the time respondends do not use TPU.

# ## Q23
# ### Use of machine learning methods <a class="anchor" id="Q23"></a>

# In[ ]:


Users_years =  ['< 1 years', '1-2 years', '2-3 years', '3-4 years', '4-5 years', '5-10 years', '10-15 years', '20+ years']
Users_years_counts = multiple_choice_responses_2019.Q23.value_counts()[Users_years].values
Users_years_counts_SQL = multiple_choice_responses_2019.Q23[multiple_choice_responses_2019.Q18_Part_3 == "SQL"].value_counts()[Users_years].values

Users_years_counts_prc = Users_years_counts / sum(Users_years_counts)
Users_years_counts_SQL_prc = Users_years_counts_SQL / sum(Users_years_counts_SQL)

fig = go.Figure(data=[
    go.Bar(name='2019 data', x=Users_years, y=Users_years_counts_prc),
    go.Bar(name='2019 data SQL', x=Users_years, y=Users_years_counts_SQL_prc),
])
# Change the bar mode
fig.update_layout(barmode='group')
fig.update_layout(title_text="Use of machine learning methods",yaxis=dict(
        title='Percent',
        titlefont_size=16,
        tickfont_size=14,)), 
fig.show()


# You can see from the chart that SQL user have more experience using machine learning methods compared with general respondents.

# ## Q24
# ### ML algorithms do you use on a regular basis <a class="anchor" id="Q24"></a>

# In[ ]:


Q24cols = ['Q24_Part_1','Q24_Part_2','Q24_Part_3','Q24_Part_4','Q24_Part_5','Q24_Part_6','Q24_Part_7','Q24_Part_8','Q24_Part_9','Q24_Part_10','Q24_Part_11','Q24_Part_12']

list_ml_tools = []
list_ml_tools_numbers = []
list_ml_tools_numbers_SQL = []

for col in Q24cols:
    
    ml_tools = multiple_choice_responses_2019[col].value_counts().index[0] 
    ml_tools_numbers = multiple_choice_responses_2019[col].value_counts()[0] 
    ml_tools_numbers_SQL = multiple_choice_responses_2019[col][multiple_choice_responses_2019.Q18_Part_3 == "SQL"].value_counts()[0]
    
    list_ml_tools.append(ml_tools)
    list_ml_tools_numbers.append(ml_tools_numbers)
    list_ml_tools_numbers_SQL.append(ml_tools_numbers_SQL)

list_ml_tools_prc = list_ml_tools_numbers / sum(list_ml_tools_numbers)
list_ml_tools_SQL_prc = list_ml_tools_numbers_SQL / sum(list_ml_tools_numbers_SQL)

fig = go.Figure(data=[
    go.Bar(name='2019 data', x=list_ml_tools, y=list_ml_tools_prc),
    go.Bar(name='2019 data SQL', x=list_ml_tools, y=list_ml_tools_SQL_prc),
])
# Change the bar mode
fig.update_layout(barmode='group')
fig.update_layout(title_text="ML algorithms do you use on a regular basis",yaxis=dict(
        title='Percent',
        titlefont_size=16,
        tickfont_size=14,)), 
fig.show()


# We can se that SQL user more often user Linear or Logistic regression models, Decision tree models, Gradient boosting and Bayesian aproachees. While general users more often use Neural network models.
# 

# ## Q25
# ### Categories of ML tools do you use on a regular basis <a class="anchor" id="Q25"></a>

# In[ ]:


Q25cols = ['Q25_Part_1','Q25_Part_2','Q25_Part_3','Q25_Part_4','Q25_Part_5','Q25_Part_6','Q25_Part_7','Q25_Part_8']

list_ml_tools_reg = []
list_ml_tools_numbers_reg = []
list_ml_tools_numbers_reg_SQL = []

for col in Q25cols:
    
    ml_tools_reg = multiple_choice_responses_2019[col].value_counts().index[0] 
    ml_tools_numbers_reg = multiple_choice_responses_2019[col].value_counts()[0] 
    ml_tools_numbers_reg_SQL = multiple_choice_responses_2019[col][multiple_choice_responses_2019.Q18_Part_3 == "SQL"].value_counts()[0] 
    
    
    list_ml_tools_reg.append(ml_tools_reg)
    list_ml_tools_numbers_reg.append(ml_tools_numbers_reg)
    list_ml_tools_numbers_reg_SQL.append(ml_tools_numbers_reg_SQL)
    

list_ml_tools_prc = list_ml_tools_numbers_reg / sum(list_ml_tools_numbers_reg)
list_ml_tools_SQL_prc = list_ml_tools_numbers_reg_SQL / sum(list_ml_tools_numbers_reg_SQL)

fig = go.Figure(data=[
    go.Bar(name='2019 data', x=list_ml_tools_reg, y=list_ml_tools_prc),
    go.Bar(name='2019 data SQL', x=list_ml_tools_reg, y=list_ml_tools_SQL_prc),
])
# Change the bar mode
fig.update_layout(barmode='group')
fig.update_layout(title_text="ML algorithms do you use on a regular basis",yaxis=dict(
        title='Percent',
        titlefont_size=16,
        tickfont_size=14,)), 
fig.show()


# ## Q26
# ### Categories of computer vision methods do you use on a regular basis <a class="anchor" id="Q26"></a>

# In[ ]:


Q26cols = ['Q26_Part_1','Q26_Part_2','Q26_Part_3','Q26_Part_4','Q26_Part_5','Q26_Part_6','Q26_Part_7']

list_cv_tools = []
list_cv_tools_numbers = []
list_cv_tools_numbers_SQL = []

for col in Q26cols:
    cv_tools= multiple_choice_responses_2019[col].value_counts().index[0] 
    cv_tools_numbers = multiple_choice_responses_2019[col].value_counts()[0] 
    cv_tools_numbers_SQL = multiple_choice_responses_2019[col][multiple_choice_responses_2019.Q18_Part_3 == "SQL"].value_counts()[0] 
    
    list_cv_tools.append(cv_tools)
    list_cv_tools_numbers.append(cv_tools_numbers)
    list_cv_tools_numbers_SQL.append(cv_tools_numbers_SQL)

list_cv_tools_prc = list_cv_tools_numbers / sum(list_cv_tools_numbers)
list_cv_tools_SQL_prc = list_cv_tools_numbers_SQL / sum(list_cv_tools_numbers_SQL)

fig = go.Figure(data=[
    go.Bar(name='2019 data', x=list_cv_tools, y=list_cv_tools_prc),
    go.Bar(name='2019 data SQL', x=list_cv_tools, y=list_cv_tools_SQL_prc),
])
# Change the bar mode
fig.update_layout(barmode='group')
fig.update_layout(title_text="Categories of computer vision methods do you use on a regular basis",
        yaxis=dict(
        title='Percent',
        titlefont_size=16,
        tickfont_size=14,)), 
fig.show()


# As you can see Image classification tool is the most used among both groups of respondents. There is slight increase in None category of SQL users. It might make sense since SQL users are less likely to use computer vision tools.

# ## Q27
# ### The following natural language processing (NLP) methods do you use on a regular basis <a class="anchor" id="Q27"></a>

# In[ ]:


Q27cols = ['Q27_Part_1','Q27_Part_2','Q27_Part_3','Q27_Part_4','Q27_Part_5','Q27_Part_6']

list_nlp_tools = []
list_nlp_tools_numbers = []
list_nlp_tools_numbers_SQL = []

for col in Q27cols:
    
    nlp_tools= multiple_choice_responses_2019[col].value_counts().index[0] 
    nlp_tools_numbers = multiple_choice_responses_2019[col].value_counts()[0] 
    nlp_tools_numbers_SQL = multiple_choice_responses_2019[col][multiple_choice_responses_2019.Q18_Part_3 == "SQL"].value_counts()[0] 
    
    list_nlp_tools.append(nlp_tools)
    list_nlp_tools_numbers.append(nlp_tools_numbers)
    list_nlp_tools_numbers_SQL.append(nlp_tools_numbers_SQL)


list_nlp_tools_prc = list_nlp_tools_numbers / sum(list_nlp_tools_numbers)
list_nlp_tools_SQL_prc = list_nlp_tools_numbers_SQL / sum(list_nlp_tools_numbers_SQL)

fig = go.Figure(data=[
    go.Bar(name='2019 data', x=list_nlp_tools, y=list_nlp_tools_prc),
    go.Bar(name='2019 data SQL', x=list_nlp_tools, y=list_nlp_tools_SQL_prc),
])
# Change the bar mode
fig.update_layout(barmode='group')
fig.update_layout(title_text="The following natural language processing (NLP) methods do you use on a regular basis",
        yaxis=dict(
        title='Percent',
        titlefont_size=16,
        tickfont_size=14,)), 
fig.show()


# The NLP tools are almost the same for both groups general and SQL user group.

# ## Q28
# ### Machine learning frameworks do you use on a regular basis <a class="anchor" id="Q28"></a>

# In[ ]:


Q28cols = ['Q28_Part_1','Q28_Part_2','Q28_Part_3','Q28_Part_4','Q28_Part_5','Q28_Part_6','Q28_Part_7','Q28_Part_8','Q28_Part_9','Q28_Part_10','Q28_Part_11','Q28_Part_12']

list_ml_frs = []
list_ml_frs_numbers = []
list_ml_frs_numbers_SQL = []


for col in Q28cols:
    
    ml_frs = multiple_choice_responses_2019[col].value_counts().index[0] 
    ml_frs_numbers = multiple_choice_responses_2019[col].value_counts()[0] 
    ml_frs_numbers_SQL = multiple_choice_responses_2019[col][multiple_choice_responses_2019.Q18_Part_3 == "SQL"].value_counts()[0]
    
    list_ml_frs.append(ml_frs)
    list_ml_frs_numbers.append(ml_frs_numbers)
    list_ml_frs_numbers_SQL.append(ml_frs_numbers_SQL)
    
list_frs_numbers_prc = list_ml_frs_numbers / sum(list_ml_frs_numbers)
list_frs_numbers_SQL_prc = list_ml_frs_numbers_SQL / sum(list_ml_frs_numbers_SQL)

fig = go.Figure(data=[
    go.Bar(name='2019 data', x=list_ml_frs, y=list_frs_numbers_prc),
    go.Bar(name='2019 data SQL', x=list_ml_frs, y=list_frs_numbers_SQL_prc),
])
# Change the bar mode
fig.update_layout(barmode='group')
fig.update_layout(title_text="Machine learning frameworks do you use on a regular basis",
        yaxis=dict(
        title='Percent',
        titlefont_size=16,
        tickfont_size=14,)), 
fig.show()


# As in graph Q24 we can see icreased usage of RandomForest and Xgboost ML platforms in SQL users compared to general users.

# ## Q29
# ### Computer platforms you use on regular basis <a class="anchor" id="Q29"></a>

# In[ ]:


Q29cols = ['Q29_Part_1','Q29_Part_2','Q29_Part_3','Q29_Part_4','Q29_Part_5','Q29_Part_6','Q29_Part_7','Q29_Part_8','Q29_Part_9','Q29_Part_10','Q29_Part_11','Q29_Part_12']

list_cps = []
list_cps_numbers = []
list_cps_numbers_SQL = []

for col in Q29cols:
    
    cps = multiple_choice_responses_2019[col].value_counts().index[0] 
    cps_numbers = multiple_choice_responses_2019[col].value_counts()[0] 
    cps_numbers_SQL = multiple_choice_responses_2019[col][multiple_choice_responses_2019.Q18_Part_3 == "SQL"].value_counts()[0] 
    
    list_cps.append(cps)
    list_cps_numbers.append(cps_numbers)
    list_cps_numbers_SQL.append(cps_numbers_SQL)
    
list_cps_numbers_prc = list_cps_numbers / sum(list_cps_numbers)
list_cps_numbers_SQL_prc = list_cps_numbers_SQL / sum(list_cps_numbers_SQL)

fig = go.Figure(data=[
    go.Bar(name='2019 data', x=list_cps, y=list_cps_numbers_prc),
    go.Bar(name='2019 data SQL', x=list_cps, y=list_cps_numbers_SQL_prc),
])
# Change the bar mode
fig.update_layout(barmode='group')
fig.update_layout(title_text="Computer platforms you use on regular basis ",
        yaxis=dict(
        title='Percent',
        titlefont_size=16,
        tickfont_size=14,)), 
fig.show()
    


# From the graph you can see that SQL users tend to use AWS, Microsoft Azure, Oracle Cloud and WMware Cloud computer platforms more than general respondents.

# ## Q30
# ### Specific cloud computing products do you use on a regular basis <a class="anchor" id="Q30"></a>

# In[ ]:


Q30cols = ['Q30_Part_1','Q30_Part_2','Q30_Part_3','Q30_Part_4','Q30_Part_5','Q30_Part_6','Q30_Part_7','Q30_Part_8','Q30_Part_9','Q30_Part_10','Q30_Part_11','Q30_Part_12']

list_ccps = []
list_ccps_numbers = []
list_ccps_numbers_SQL = []


for col in Q30cols:
    
    ccps = multiple_choice_responses_2019[col].value_counts().index[0] 
    ccps_numbers = multiple_choice_responses_2019[col].value_counts()[0] 
    ccps_numbers_SQL = multiple_choice_responses_2019[col][multiple_choice_responses_2019.Q18_Part_3 == "SQL"].value_counts()[0] 

    
    list_ccps.append(ccps)
    list_ccps_numbers.append(ccps_numbers)
    list_ccps_numbers_SQL.append(ccps_numbers_SQL)
    
list_ccps_numbers_prc = list_ccps_numbers / sum(list_ccps_numbers)
list_ccps_numbers_SQL_prc = list_ccps_numbers_SQL / sum(list_ccps_numbers_SQL)

fig = go.Figure(data=[
    go.Bar(name='2019 data', x=list_ccps, y=list_ccps_numbers_prc),
    go.Bar(name='2019 data SQL', x=list_ccps, y=list_ccps_numbers_SQL_prc),
])
# Change the bar mode
fig.update_layout(barmode='group')
fig.update_layout(title_text="Specific cloud computing products do you use on a regular basis",
        yaxis=dict(
        title='Percent',
        titlefont_size=16,
        tickfont_size=14,)), 
fig.show()    


# We can see from the graph that SQL users use more of almost all specific cloud computer products compared to general respondents. This also confirms that we see lower percentage of column None for SQL users.

# ## Q31
# ### Specific big data / analytics products do you use on a regular basis <a class="anchor" id="Q31"></a>

# In[ ]:



Q31cols = ['Q31_Part_1','Q31_Part_2','Q31_Part_3','Q31_Part_4','Q31_Part_5','Q31_Part_6','Q31_Part_7','Q31_Part_8','Q31_Part_9','Q31_Part_10','Q31_Part_11','Q31_Part_12']

list_bds = []
list_bds_numbers = []
list_bds_numbers_SQL = []


for col in Q31cols:
    
    bds = multiple_choice_responses_2019[col].value_counts().index[0] 
    bds_numbers = multiple_choice_responses_2019[col].value_counts()[0] 
    bds_numbers_SQL = multiple_choice_responses_2019[col][multiple_choice_responses_2019.Q18_Part_3 == "SQL"].value_counts()[0] 
    
    
    list_bds.append(bds)
    list_bds_numbers.append(bds_numbers)
    list_bds_numbers_SQL.append(bds_numbers_SQL)
    
list_bds_numbers_prc = list_bds_numbers / sum(list_bds_numbers)
list_bds_numbers_SQL_prc = list_bds_numbers_SQL / sum(list_bds_numbers_SQL)

fig = go.Figure(data=[
    go.Bar(name='2019 data', x=list_bds, y=list_bds_numbers_prc),
    go.Bar(name='2019 data SQL', x=list_bds, y=list_bds_numbers_SQL_prc),
])
# Change the bar mode
fig.update_layout(barmode='group')
fig.update_layout(title_text="Specific big data / analytics products do you use on a regular basis",
        yaxis=dict(
        title='Percent',
        titlefont_size=16,
        tickfont_size=14,)), 
fig.show()    


# We can see from the graph that SQL users use more of big data / analytics products compared to general respondents. This also confirms that we see lower percentage of column None for SQL users.

# ## Q32
# ### Following machine learning products do you use on a regular basis <a class="anchor" id="Q32"></a>

# In[ ]:


Q32cols = ['Q32_Part_1','Q32_Part_2','Q32_Part_3','Q32_Part_4','Q32_Part_5','Q32_Part_6','Q32_Part_7','Q32_Part_8','Q32_Part_9','Q32_Part_10','Q32_Part_11','Q32_Part_12']

list_mlps = []
list_mlps_numbers = []
list_mlps_numbers_SQL = []

for col in Q32cols:
    mlps = multiple_choice_responses_2019[col].value_counts().index[0] 
    mlps_numbers = multiple_choice_responses_2019[col].value_counts()[0] 
    mlps_numbers_SQL = multiple_choice_responses_2019[col][multiple_choice_responses_2019.Q18_Part_3 == "SQL"].value_counts()[0] 
    
    list_mlps.append(mlps)
    list_mlps_numbers.append(mlps_numbers)
    list_mlps_numbers_SQL.append(mlps_numbers_SQL)

list_mlps_numbers_prc = list_mlps_numbers / sum(list_mlps_numbers)
list_mlps_numbers_SQL_prc = list_mlps_numbers_SQL / sum(list_mlps_numbers_SQL)

fig = go.Figure(data=[
    go.Bar(name='2019 data', x=list_mlps, y=list_mlps_numbers_prc),
    go.Bar(name='2019 data SQL', x=list_mlps, y=list_mlps_numbers_SQL_prc),
])
# Change the bar mode
fig.update_layout(barmode='group')
fig.update_layout(title_text="Following machine learning products do you use on a regular basis ",
        yaxis=dict(
        title='Percent',
        titlefont_size=16,
        tickfont_size=14,)), 
fig.show() 
    


# We can see from the graph that SQL users use more of machine learning products compared to general respondents. This also confirms that we see lower percentage of column None for SQL users.

# ## Q33
# ### Automated machine learning tools (or partial AutoML tools) do you use on a regular basis <a class="anchor" id="Q33"></a>

# In[ ]:


Q33cols = ['Q33_Part_1','Q33_Part_2','Q33_Part_3','Q33_Part_4','Q33_Part_5','Q33_Part_6','Q33_Part_7','Q33_Part_8','Q33_Part_9','Q33_Part_10','Q33_Part_11','Q33_Part_12']

list_amlts = []
list_amlts_numbers = []
list_amlts_numbers_SQL = []


for col in Q33cols:
    
    amlts = multiple_choice_responses_2019[col].value_counts().index[0] 
    amlts_numbers = multiple_choice_responses_2019[col].value_counts()[0] 
    amlts_numbers_SQL = multiple_choice_responses_2019[col][multiple_choice_responses_2019.Q18_Part_3 == "SQL"].value_counts()[0] 
    
    list_amlts.append(amlts)
    list_amlts_numbers.append(amlts_numbers)
    list_amlts_numbers_SQL.append(amlts_numbers_SQL)

list_amlts_numbers_prc = list_amlts_numbers / sum(list_amlts_numbers)
list_amlts_numbers_SQL_prc = list_amlts_numbers_SQL / sum(list_amlts_numbers_SQL)

fig = go.Figure(data=[
    go.Bar(name='2019 data', x=list_amlts, y=list_amlts_numbers_prc),
    go.Bar(name='2019 data SQL', x=list_amlts, y=list_amlts_numbers_SQL_prc),
])
# Change the bar mode
fig.update_layout(barmode='group')
fig.update_layout(title_text="Automated machine learning tools (or partial AutoML tools) do you use on a regular basis",
        yaxis=dict(
        title='Percent',
        titlefont_size=16,
        tickfont_size=14,)), 
fig.show() 


# We can see from the graph that SQL users use more of automated machine learning tools compared to general respondents. This also confirms that we see lower percentage of column None for SQL users.

# ## Q34
# ### The following relational database products do you use on a regular basis <a class="anchor" id="Q34"></a>

# In[ ]:


multiple_choice_responses_2019.Q34_Part_1.value_counts()

Q34cols = ['Q34_Part_1','Q34_Part_2','Q34_Part_3','Q34_Part_4','Q34_Part_5','Q34_Part_6','Q34_Part_7','Q34_Part_8','Q34_Part_9','Q34_Part_10','Q34_Part_11','Q34_Part_12']

list_dbs = []
list_dbs_numbers = []

for col in Q34cols:
    
    dbs = multiple_choice_responses_2019[col].value_counts().index[0] 
    dbs_numbers = multiple_choice_responses_2019[col].value_counts()[0] 
    
    list_dbs.append(dbs)
    list_dbs_numbers.append(dbs_numbers)
    
g = sns.barplot(x=list_dbs, y=list_dbs_numbers)
g.set_xticklabels(g.get_xticklabels(), rotation=80)
plt.title("The following relational database products do you use on a regular basis in 2019")
g.set_ylabel('Number of respondents.')

for ix, x in zip(range(len(list_dbs_numbers)+1),list_dbs_numbers):
    g.text(ix,x,x, horizontalalignment='center')

plt.show()


# #########################################################
# # Second part. Question and answers. <a class="anchor" id="second"></a>#########
# #########################################################

# Lets separate our ML developer in 3 groups according to experience in ML. And look at top 5 countries where these developers come from: junior developer: <1-3, middle-level developer: 3-5, senior developer: 5-20+

# ### Countries of junior, middle-level and senior developers <a class="anchor" id="Q35"></a>

# In[ ]:


junior_ml_devs = multiple_choice_responses_2019.Q3[(multiple_choice_responses_2019.Q23 == '< 1 years') | (multiple_choice_responses_2019.Q23 == '1-2 years')| (multiple_choice_responses_2019.Q23 ==  '2-3 years')].value_counts()[:5]
mid_ml_devs = multiple_choice_responses_2019.Q3[(multiple_choice_responses_2019.Q23 == '3-4 years') | (multiple_choice_responses_2019.Q23 == '4-5 years')].value_counts()[:5]
senior_ml_devs = multiple_choice_responses_2019.Q3[(multiple_choice_responses_2019.Q23 == '5-10 years') | (multiple_choice_responses_2019.Q23 == '10-15 years') | (multiple_choice_responses_2019.Q23 == '20+ years') ].value_counts()[:5]

plt.figure(3, figsize=(20,5))
the_grid = GridSpec(1, 3)

# Junior Developers are from these top 5 countries.

plt.subplot(the_grid[0, 0])
g = sns.barplot(x=junior_ml_devs.index, y=junior_ml_devs.values)
g.set_xticklabels(junior_ml_devs.index, rotation=80)
g.set_ylabel("Number of respondents")
plt.title("Countries where junior ML developers come from in 2019 survey.")


# Middle developers are from these top 5 countries.

plt.subplot(the_grid[0, 1])
g = sns.barplot(x=mid_ml_devs.index, y=mid_ml_devs.values)
g.set_xticklabels(mid_ml_devs.index, rotation=80)
g.set_ylabel("Number of respondents")
plt.title("Countries where middle-level ML developers come from in 2019 survey.")


# Senior developers are from these top 5 countries.

plt.subplot(the_grid[0, 2])

g = sns.barplot(x=senior_ml_devs.index, y=senior_ml_devs.values)
g.set_xticklabels(senior_ml_devs.index, rotation=80)
g.set_ylabel("Number of respondents")
plt.title("Countries where senior ML developer come from in 2019 survey.")

plt.show()


# We can see that a lot of junior and middle developers come from India. Whereas senior developers come from USA, UK and Germany. 

# Let's find our which programming language is popular among junior, middle-level, senior ML developers.

# ### Programming language popular among junior, middle-level and senior developers <a class="anchor" id="Q36"></a>

# In[ ]:


Q18cols = ['Q18_Part_1','Q18_Part_2','Q18_Part_3','Q18_Part_4','Q18_Part_5','Q18_Part_6','Q18_Part_7','Q18_Part_8','Q18_Part_9','Q18_Part_10','Q18_Part_11','Q18_Part_12']

l_junior_ml_dev_lang = []
l_junior_ml_dev_lang_num = [] 

for col in Q18cols:
 
    junior_ml_devs = multiple_choice_responses_2019[col][(multiple_choice_responses_2019.Q23 == '< 1 years') | (multiple_choice_responses_2019.Q23 == '1-2 years')| (multiple_choice_responses_2019.Q23 ==  '2-3 years')].value_counts()
    junior_ml_dev_lang = junior_ml_devs.index[0]
    junior_ml_dev_lang_num = junior_ml_devs.values[0]
    
    l_junior_ml_dev_lang.append(junior_ml_dev_lang)
    l_junior_ml_dev_lang_num.append(junior_ml_dev_lang_num)

    
l_mid_ml_dev_lang = []
l_mid_ml_dev_lang_num = [] 

for col in Q18cols:
    
    mid_ml_devs = multiple_choice_responses_2019[col][(multiple_choice_responses_2019.Q23 == '3-4 years') | (multiple_choice_responses_2019.Q23 == '4-5 years')].value_counts()

    mid_ml_dev_lang = mid_ml_devs.index[0]  
    mid_ml_dev_lang_num = mid_ml_devs.values[0]
    
    l_mid_ml_dev_lang.append(mid_ml_dev_lang)
    l_mid_ml_dev_lang_num.append(mid_ml_dev_lang_num)

l_senior_ml_dev_lang = []
l_senior_ml_dev_lang_num = [] 


for col in Q18cols:

    senior_ml_devs = multiple_choice_responses_2019[col][(multiple_choice_responses_2019.Q23 == '5-10 years') | (multiple_choice_responses_2019.Q23 == '10-15 years') | (multiple_choice_responses_2019.Q23 == '20+ years') ].value_counts()

    senior_ml_dev_lang = senior_ml_devs.index[0]  
    senior_ml_dev_lang_num = senior_ml_devs.values[0]
    
    l_senior_ml_dev_lang.append(senior_ml_dev_lang)
    l_senior_ml_dev_lang_num.append(senior_ml_dev_lang_num)

    
plt.figure(3, figsize=(20,5))
the_grid = GridSpec(1, 3)

# Junior Developers.

plt.subplot(the_grid[0, 0])
g = sns.barplot(x=l_junior_ml_dev_lang, y=l_junior_ml_dev_lang_num)
g.set_xticklabels(l_junior_ml_dev_lang, rotation=80)
g.set_ylabel("Number of respondents")
plt.title("Programming languages junior ML developers use in 2019 survey.")


# Middle-level developers.

plt.subplot(the_grid[0, 1])
g = sns.barplot(x=l_mid_ml_dev_lang, y=l_mid_ml_dev_lang_num)
g.set_xticklabels(l_mid_ml_dev_lang, rotation=80)
g.set_ylabel("Number of respondents")
plt.title("Programming languages middle-level ML developers use in 2019 survey")


# Senior developers.

plt.subplot(the_grid[0, 2])

g = sns.barplot(x=l_senior_ml_dev_lang, y=l_senior_ml_dev_lang_num)
g.set_xticklabels(l_senior_ml_dev_lang, rotation=80)
g.set_ylabel("Number of respondents")
plt.title("Programming languages senior ML developers use in 2019 survey")

plt.show()


# As you can see Python is the most used language. R, SQL and C++ are slightly more popular among senior level ML developer than a junior level.

# Let's find our which IDE is popular among junior, middle-level, senior ML developers.

# ### IDE popular among junior, middle-level and senior developers <a class="anchor" id="Q37"></a>

# In[ ]:


Q16cols = ['Q16_Part_1','Q16_Part_2','Q16_Part_3','Q16_Part_4','Q16_Part_5','Q16_Part_6','Q16_Part_7','Q16_Part_8','Q16_Part_9','Q16_Part_10','Q16_Part_11','Q16_Part_12']

l_junior_ml_dev_ide = []
l_junior_ml_dev_ide_num = [] 

for col in Q16cols:
 
    junior_ml_devs = multiple_choice_responses_2019[col][(multiple_choice_responses_2019.Q23 == '< 1 years') | (multiple_choice_responses_2019.Q23 == '1-2 years')| (multiple_choice_responses_2019.Q23 ==  '2-3 years')].value_counts()
    junior_ml_dev_ide = junior_ml_devs.index[0]
    junior_ml_dev_ide_num = junior_ml_devs.values[0]
    
    l_junior_ml_dev_ide.append(junior_ml_dev_ide)
    l_junior_ml_dev_ide_num.append(junior_ml_dev_ide_num)

    
l_mid_ml_dev_ide = []
l_mid_ml_dev_ide_num = [] 

for col in Q16cols:
    
    mid_ml_devs = multiple_choice_responses_2019[col][(multiple_choice_responses_2019.Q23 == '3-4 years') | (multiple_choice_responses_2019.Q23 == '4-5 years')].value_counts()

    mid_ml_dev_ide = mid_ml_devs.index[0]  
    mid_ml_dev_ide_num = mid_ml_devs.values[0]
    
    l_mid_ml_dev_ide.append(mid_ml_dev_ide)
    l_mid_ml_dev_ide_num.append(mid_ml_dev_ide_num)

l_senior_ml_dev_ide = []
l_senior_ml_dev_ide_num = [] 


for col in Q16cols:

    senior_ml_devs = multiple_choice_responses_2019[col][(multiple_choice_responses_2019.Q23 == '5-10 years') | (multiple_choice_responses_2019.Q23 == '10-15 years') | (multiple_choice_responses_2019.Q23 == '20+ years') ].value_counts()

    senior_ml_dev_ide = senior_ml_devs.index[0]  
    senior_ml_dev_ide_num = senior_ml_devs.values[0]
    
    l_senior_ml_dev_ide.append(senior_ml_dev_ide)
    l_senior_ml_dev_ide_num.append(senior_ml_dev_ide_num)

    
plt.figure(3, figsize=(20,5))
the_grid = GridSpec(1, 3)

# Junior Developers.

plt.subplot(the_grid[0, 0])
g = sns.barplot(x=l_junior_ml_dev_ide, y=l_junior_ml_dev_ide_num)
g.set_xticklabels(l_junior_ml_dev_ide, rotation=80)
g.set_ylabel("Number of respondents")
plt.title("IDEs junior ML developers use in 2019 survey.")


# Middle-level developers.

plt.subplot(the_grid[0, 1])
g = sns.barplot(x=l_mid_ml_dev_ide, y=l_mid_ml_dev_ide_num)
g.set_xticklabels(l_mid_ml_dev_ide, rotation=80)
g.set_ylabel("Number of respondents")
plt.title("IDEs middle-level ML developers use in 2019 survey")


# Senior developers.

plt.subplot(the_grid[0, 2])

g = sns.barplot(x=l_senior_ml_dev_ide, y=l_senior_ml_dev_ide_num)
g.set_xticklabels(l_senior_ml_dev_ide, rotation=80)
g.set_ylabel("Number of respondents")
plt.title("IDEs senior ML developers use in 2019 survey")

plt.show()


# We see that the most popular IDE is a Jupyter notebook. However for senior ML developers RStudio and Vim is reletively more popular than for junior ML developers.

# Let's find out which relational database product is popular among junior, middle-level, senior ML developers.

# ### Relational database popular among junior, middle-level and senior developers <a class="anchor" id="Q38"></a>

# In[ ]:


Q34cols = ['Q34_Part_1','Q34_Part_2','Q34_Part_3','Q34_Part_4','Q34_Part_5','Q34_Part_6','Q34_Part_7','Q34_Part_8','Q34_Part_9','Q34_Part_10','Q34_Part_11','Q34_Part_12']

l_junior_ml_dev_db = []
l_junior_ml_dev_db_num = [] 

for col in Q34cols:
 
    junior_ml_devs = multiple_choice_responses_2019[col][(multiple_choice_responses_2019.Q23 == '< 1 years') | (multiple_choice_responses_2019.Q23 == '1-2 years')| (multiple_choice_responses_2019.Q23 ==  '2-3 years')].value_counts()
    junior_ml_dev_db = junior_ml_devs.index[0]
    junior_ml_dev_db_num = junior_ml_devs.values[0]
    
    l_junior_ml_dev_db.append(junior_ml_dev_db)
    l_junior_ml_dev_db_num.append(junior_ml_dev_db_num)

    
l_mid_ml_dev_db = []
l_mid_ml_dev_db_num = [] 

for col in Q34cols:
    
    mid_ml_devs = multiple_choice_responses_2019[col][(multiple_choice_responses_2019.Q23 == '3-4 years') | (multiple_choice_responses_2019.Q23 == '4-5 years')].value_counts()

    mid_ml_dev_db = mid_ml_devs.index[0]  
    mid_ml_dev_db_num = mid_ml_devs.values[0]
    
    l_mid_ml_dev_db.append(mid_ml_dev_db)
    l_mid_ml_dev_db_num.append(mid_ml_dev_db_num)

l_senior_ml_dev_db = []
l_senior_ml_dev_db_num = [] 


for col in Q34cols:

    senior_ml_devs = multiple_choice_responses_2019[col][(multiple_choice_responses_2019.Q23 == '5-10 years') | (multiple_choice_responses_2019.Q23 == '10-15 years') | (multiple_choice_responses_2019.Q23 == '20+ years') ].value_counts()

    senior_ml_dev_db = senior_ml_devs.index[0]  
    senior_ml_dev_db_num = senior_ml_devs.values[0]
    
    l_senior_ml_dev_db.append(senior_ml_dev_db)
    l_senior_ml_dev_db_num.append(senior_ml_dev_db_num)

    
plt.figure(3, figsize=(20,5))
the_grid = GridSpec(1, 3)

# Junior Developers.

plt.subplot(the_grid[0, 0])
g = sns.barplot(x=l_junior_ml_dev_db, y=l_junior_ml_dev_db_num)
g.set_xticklabels(l_junior_ml_dev_db, rotation=80)
g.set_ylabel("Number of respondents")
plt.title("Relational database popular among junior ML developers")


# Middle-level developers.

plt.subplot(the_grid[0, 1])
g = sns.barplot(x=l_mid_ml_dev_db, y=l_mid_ml_dev_db_num)
g.set_xticklabels(l_mid_ml_dev_db, rotation=80)
g.set_ylabel("Number of respondents")
plt.title("Relational database popular among middle-level ML developers")


# Senior developers.

plt.subplot(the_grid[0, 2])

g = sns.barplot(x=l_senior_ml_dev_db, y=l_senior_ml_dev_db_num)
g.set_xticklabels(l_senior_ml_dev_db, rotation=80)
g.set_ylabel("Number of respondents")
plt.title("Relational database popular among senior ML developers")

plt.show()


# From diagrams we can see that most popular relational database is MySQL. It seems that PostgresSQL, SQLite and Microsoft SQL Server are more popular among mid and senior level developers than of junior level.

# Let's find out what yearly compensation junior, middle-level, senior ML developers get.

# ### Yearly compensation among junior, middle-level and senior developers <a class="anchor" id="Q39"></a>

# In[ ]:


compensation_order = ['$0-999', '1,000-1,999', '2,000-2,999', '3,000-3,999','4,000-4,999',
                      '5,000-7,499', '7,500-9,999','10,000-14,999','15,000-19,999',  
                      '20,000-24,999','25,000-29,999','30,000-39,999','40,000-49,999', 
                      '50,000-59,999','60,000-69,999','70,000-79,999','80,000-89,999',
                      '90,000-99,999','100,000-124,999','125,000-149,999', 
                      '150,000-199,999','200,000-249,999','250,000-299,999', '300,000-500,000', '> $500,000']
values_compensation = multiple_choice_responses_2019.Q10.value_counts()[compensation_order].values

 
junior_ml_devs = multiple_choice_responses_2019.Q10[(multiple_choice_responses_2019.Q23 == '< 1 years') | (multiple_choice_responses_2019.Q23 == '1-2 years')| (multiple_choice_responses_2019.Q23 ==  '2-3 years')].value_counts()[compensation_order].values
mid_ml_devs = multiple_choice_responses_2019.Q10[(multiple_choice_responses_2019.Q23 == '3-4 years') | (multiple_choice_responses_2019.Q23 == '4-5 years')].value_counts()[compensation_order].values
senior_ml_devs = multiple_choice_responses_2019.Q10[(multiple_choice_responses_2019.Q23 == '5-10 years') | (multiple_choice_responses_2019.Q23 == '10-15 years') | (multiple_choice_responses_2019.Q23 == '20+ years') ].value_counts()[compensation_order].values

    
plt.figure(3, figsize=(20,5))
the_grid = GridSpec(1, 3)

# Junior Developers.

plt.subplot(the_grid[0, 0])
g = sns.barplot(x=compensation_order, y=junior_ml_devs)
g.set_xticklabels(compensation_order, rotation=80)
g.set_ylabel("Number of respondents")
plt.title("Yearly compensation for junior developers in 2019 survey.")


# Middle-level developers.

plt.subplot(the_grid[0, 1])
g = sns.barplot(x=compensation_order, y=mid_ml_devs)
g.set_xticklabels(compensation_order, rotation=80)
g.set_ylabel("Number of respondents")
plt.title("Yearly compensation for middle-level developers in 2019 survey")


# Senior developers.

plt.subplot(the_grid[0, 2])

g = sns.barplot(x=compensation_order, y=senior_ml_devs)
g.set_xticklabels(compensation_order, rotation=80)
g.set_ylabel("Number of respondents")
plt.title("Yearly compensation for senior developers in 2019 survey")

plt.show()


# You can clearly see how increasing in experience increases your annual compensation. Keep in mind that number of low paid compensation 0-999 decreases and you see exaggeration of highly paid ML developer but still you see an increase in highly paid positions coresponds to their experience in the field.

#  Which programming language is paying more.

# ### Yearly compensation depending of which progamming language you use <a class="anchor" id="Q40"></a>

# In[ ]:


compensation_order = ['$0-999', '1,000-1,999', '2,000-2,999', '3,000-3,999','4,000-4,999',
                      '5,000-7,499', '7,500-9,999','10,000-14,999','15,000-19,999',  
                      '20,000-24,999','25,000-29,999','30,000-39,999','40,000-49,999', 
                      '50,000-59,999','60,000-69,999','70,000-79,999','80,000-89,999',
                      '90,000-99,999','100,000-124,999','125,000-149,999', 
                      '150,000-199,999','200,000-249,999','250,000-299,999', '300,000-500,000', '> $500,000']

python_devs = multiple_choice_responses_2019.Q10[multiple_choice_responses_2019.Q18_Part_1 == 'Python'].value_counts()[compensation_order].values
SQL_devs = multiple_choice_responses_2019.Q10[multiple_choice_responses_2019.Q18_Part_3 == 'SQL'].value_counts()[compensation_order].values
R_devs = multiple_choice_responses_2019.Q10[multiple_choice_responses_2019.Q18_Part_2 == 'R' ].value_counts()[compensation_order].values
Java_devs = multiple_choice_responses_2019.Q10[multiple_choice_responses_2019.Q18_Part_6 == 'Java' ].value_counts()[compensation_order].values
Cpp_devs = multiple_choice_responses_2019.Q10[multiple_choice_responses_2019.Q18_Part_5 == 'C++' ].value_counts()[compensation_order].values

plt.figure(2, figsize=(12,20))
the_grid = GridSpec(3, 2, hspace=0.5)

# Python Developers.

plt.subplot(the_grid[0, 0])
g = sns.barplot(x=compensation_order, y=python_devs)
g.set_xticklabels(compensation_order, rotation=80)
g.set_ylabel("Number of respondents")
plt.title("Yearly compensation for Python developers in 2019 survey.")


# SQL developers.

plt.subplot(the_grid[0, 1])
g = sns.barplot(x=compensation_order, y=SQL_devs)
g.set_xticklabels(compensation_order, rotation=80)
g.set_ylabel("Number of respondents")
plt.title("Yearly compensation for SQL developers in 2019 survey")


# R developers.

plt.subplot(the_grid[1, 0])

g = sns.barplot(x=compensation_order, y=R_devs)
g.set_xticklabels(compensation_order, rotation=80)
g.set_ylabel("Number of respondents")
plt.title("Yearly compensation for R developers in 2019 survey")

# Java developers.

plt.subplot(the_grid[1, 1])
g = sns.barplot(x=compensation_order, y=Java_devs)
g.set_xticklabels(compensation_order, rotation=80)
g.set_ylabel("Number of respondents")
plt.title("Yearly compensation for Java developers in 2019 survey")


# C++ developers.

plt.subplot(the_grid[2, 0])

g = sns.barplot(x=compensation_order, y=Cpp_devs)
g.set_xticklabels(compensation_order, rotation=80)
g.set_ylabel("Number of respondents")
plt.title("Yearly compensation for C++ developers in 2019 survey")


plt.show()


# It seems that relatively there are more respondents that earn more that use Python, SQL, R languages rather than C++ or Java. But that might be different in other surveys where Java and C++ are more popular languages.

# It would also be interesting to investigate from which country low and high paid Pythonistas come from.  

# ### Countries where highly paid Pythonistas come from <a class="anchor" id="Q41"></a>

# In[ ]:


low_python_devs = multiple_choice_responses_2019.Q3[(multiple_choice_responses_2019.Q18_Part_1 == 'Python') &
                                                   (multiple_choice_responses_2019.Q10 == '$0-999') | (multiple_choice_responses_2019.Q10 == '1000-1999')].value_counts()

high_python_devs = multiple_choice_responses_2019.Q3[(multiple_choice_responses_2019.Q18_Part_1 == 'Python') &
                                                   (multiple_choice_responses_2019.Q10 == '100,000-124,999') | (multiple_choice_responses_2019.Q10 == '125,000-149,999')].value_counts()
 

country = low_python_devs.index
country = pd.Series(country)
country = country.replace('United Kingdom of Great Britain and Northern Ireland','UK')
country = country.replace('United States of America','USA')
country_values = low_python_devs.values

plt.figure(figsize=(16, 6))
g = sns.barplot(x=country, y=country_values)
g.set_xticklabels(g.get_xticklabels(), rotation=90)
g.set_ylabel("Number of respondens.")

for ix, x in zip(range(len(country_values)+1),country_values):
    g.text(ix,x,x, horizontalalignment='center')
    
plt.title("Low pay '$0-999' or '1000-1999' country distribution in 2019 survey.")
plt.show()


# In[ ]:


country = high_python_devs.index
country = pd.Series(country)
country = country.replace('United Kingdom of Great Britain and Northern Ireland','UK')
country = country.replace('United States of America','USA')
country_values = high_python_devs.values

plt.figure(figsize=(16, 6))
g = sns.barplot(x=country, y=country_values)
g.set_xticklabels(g.get_xticklabels(), rotation=90)
g.set_ylabel("Number of respondens.")

for ix, x in zip(range(len(country_values)+1),country_values):
    g.text(ix,x,x, horizontalalignment='center')
    
plt.title("High pay '100,000-124,999' or '125,000-149,999' country distribution in 2019 survey.")
plt.show()


# ### Pythonistas favorite media sources that report on data science topics <a class="anchor" id="Q42"></a>

# In[ ]:


Q12cols = ['Q12_Part_1','Q12_Part_2','Q12_Part_3','Q12_Part_4','Q12_Part_5','Q12_Part_6','Q12_Part_7','Q12_Part_8','Q12_Part_9','Q12_Part_10','Q12_Part_11','Q12_Part_12']

list_number_of_media_sources = []
list_of_media_sources = []

for col in Q12cols:
    media_source = multiple_choice_responses_2019[col][(multiple_choice_responses_2019.Q18_Part_1 == 'Python')].value_counts().index[0] 
    number_media_sources = multiple_choice_responses_2019[col][(multiple_choice_responses_2019.Q18_Part_1 == 'Python')].value_counts()[0] 
    list_number_of_media_sources.append(number_media_sources)
    list_of_media_sources.append(media_source)

fig, ax = plt.subplots() 
    
ax.barh(list_of_media_sources, list_number_of_media_sources, align='center', color=(0.6, 0.4, 0.6, 0.6))
ax.set_yticklabels(list_of_media_sources)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Number of respondents.')
ax.set_title('Pythonistas favorite media sources that report on data science topics.')
for i, v in enumerate(list_number_of_media_sources):
    ax.text(v + 3, i + .25, str(v), color='black', fontweight='bold')

plt.show()


# ### Pythonistas favorite ML algorithms <a class="anchor" id="Q43"></a>

# In[ ]:


Q24cols = ['Q24_Part_1','Q24_Part_2','Q24_Part_3','Q24_Part_4','Q24_Part_5','Q24_Part_6','Q24_Part_7','Q24_Part_8','Q24_Part_9','Q24_Part_10','Q24_Part_11','Q24_Part_12']

list_ml_tools = []
list_ml_tools_numbers = []

for col in Q24cols:
    ml_tools = multiple_choice_responses_2019[col][(multiple_choice_responses_2019.Q18_Part_1 == 'Python')].value_counts().index[0] 
    ml_tools_numbers = multiple_choice_responses_2019[col][(multiple_choice_responses_2019.Q18_Part_1 == 'Python')].value_counts()[0] 
    list_ml_tools.append(ml_tools)
    list_ml_tools_numbers.append(ml_tools_numbers)

fig, ax = plt.subplots()

ax.barh(list_ml_tools, list_ml_tools_numbers, align='center', color=(0.6, 0.4, 0.6, 0.6))
ax.set_yticklabels(list_ml_tools)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Number of respondents.')
ax.set_title('ML algorithms Pythonistas use on a regular basis in 2019.')
for i, v in enumerate(list_ml_tools_numbers):
    ax.text(v + 3, i + .25, str(v), color='black', fontweight='bold')

plt.show()


# ### Platforms that Pythonistas have begun or completed data science courses <a class="anchor" id="Q44"></a>

# In[ ]:


Q13cols = ['Q13_Part_1','Q13_Part_2','Q13_Part_3','Q13_Part_4','Q13_Part_5','Q13_Part_6','Q13_Part_7','Q13_Part_8','Q13_Part_9','Q13_Part_10','Q13_Part_11','Q13_Part_12']

list_science_courses = []
list_number_science_courses = []

for col in Q13cols:
    science_course = multiple_choice_responses_2019[col][(multiple_choice_responses_2019.Q18_Part_1 == 'Python')].value_counts().index[0] 
    number_science_courses = multiple_choice_responses_2019[col][(multiple_choice_responses_2019.Q18_Part_1 == 'Python')].value_counts()[0] 
    list_number_science_courses.append(number_science_courses)
    list_science_courses.append(science_course)

fig, ax = plt.subplots()

ax.barh(list_science_courses, list_number_science_courses, align='center', color=(0.6, 0.4, 0.6, 0.6))
ax.set_yticklabels(list_science_courses)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Number of respondents.')
ax.set_title('Platforms that Pythonistas have begun or completed data science courses in 2019 survey.')
for i, v in enumerate(list_number_science_courses):
    ax.text(v + 3, i + .25, str(v), color='black', fontweight='bold')

plt.show()

