#!/usr/bin/env python
# coding: utf-8

# # Contents
# * [General Explanation of the Notebook](#"General_ideas")
# 
# * [Set up the data](#Set_up_the_data)
# 
# * [Questions that were asked to participants](#list_of_questions)
# 
# * [Use Pandas to store data](#set_up_dataFrames)
# 
# * [Gender and Age ](#Gender_and_age)
#     
#     * [Convert age ranges to categories](#age_categories)
#     * [Age Distribution of Female  Data Scientists](#female_age_distribution)
#     * [Age distribution of Male Data Scientists](#male_age_distribution)
#     * [Number of Men and Women in Data Scicence](#male_female_total)
# 
# * [Educational Backgrounds of Data Scientists](#start_edu)
#     * [Undergrad majors in Data Science](#all_majors)
#     * [Observations about undergraduate majors working in Data Science](#majors_summary)
#     
#     
#  * [Highest Education Level ](#highest_education)
#      * [HIghest degrees-a visual representation](#hist_degrees)
#      * [Observations about highest Education levels for Data Scientists](#degree_observ)
#     
#  * [Education levels and Salaries](#edu_salary)
#      * [Salary Scales](#salaries_range)
#      * [Observations about salaries and education levels](#observ_salary)
#      * [Salary outliers-upper threshold](#salary_outliers)
#      * [People with degrees making over $200k a year](#rich_money)-
#      * [Quick Observation about degrees and salaries](#rich_observ)
#      
# * [Machine Learning and Data Scientists getting paid](#ml)
#      * [Range of Machine Learning being used by companies](#ml_responses)
#      * [People making over $ 100k working at places which use ML](#ml_salary)   
#      * [People making over $$100k at places that DON'T USE ML](#noml_salary)
#      * [Observations about Machine learning and Data Scientists' salary](#observ_ml)
# 
# * [FINAL THOUGHTS](#final_thoughts)
#     
#           
#                 

# 
# <a id = "General_ideas"></a>
# # <font color = 'purple'>In this Notebook, I am going to try to satiate some of own curiosities. </font>
# 
# ## <font color = 'green'> The questions that I will be exploring in this notebook are: </font>
# 
# **1. What is the gender and age distribution of Data Scientists?**
# 
# **2. Educational backgrounds of Data Scientists?**
# 
# **3. Link between educational degrees and salaries?**
# 
# **4. Outliers in terms of people who earn a LOT of money?**
# 
# **5. Effect that the use of machine learning has on the salaries of the employees at a company.**
# 
# # So....Let's do this!!
# 
# ![https://media.giphy.com/media/3oz8xHY5TPG9CAW0xi/giphy.gif](https://media.giphy.com/media/3oz8xHY5TPG9CAW0xi/giphy.gif)

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#Visualization libraries 
import seaborn as sns
import matplotlib.pyplot as plt


import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# <a id="Set_up_the_data"></a>
# # <font color = 'purple'> LET'S SET UP THE DATA </font>

# In[ ]:



survey = '../input/SurveySchema.csv'
response = '../input/freeFormResponses.csv'
mcq = '../input/multipleChoiceResponses.csv'


# In[ ]:


df_survey = pd.read_csv(survey)
df_survey.head()
df_survey.columns
print('Below are all the questions that were asked:\n')

# df_survey[['Q1', 'Q10','Q11', 'Q12', 'Q13', 'Q14']]
# df_survey[[]]
# df_survey[['Q15', 'Q16', 'Q17', 'Q18', 'Q19', 'Q2']]
# for i in df_survey.columns:
#     print(i,':',df_survey[i][0])


# <a id = "list_of_questions"> </a>
# ## <font color ='purple'> This is the list of all the questions which were asked:</font>
# 
# ['Q1', 'Q2', 'Q3','Q4', 'Q5', 'Q7', 'Q8', 'Q9', 'Q10', 'Q15', 'Q16', 'Q18', 'Q23','Q36', 'Q37', 'Q4', 'Q40', 'Q50']
# 
# Q1 :Gender 
# 
# Q2: Age
# 
# Q3 : In which country do you currently reside?
# 
# Q4: Highest level of formal education?
# 
# Q5 : Which best describes your undergraduate major? - Selected Choice
# 
# Q7 : In what industry is your current employer/contract (or your most recent employer if retired)? - Selected Choice
# 
# Q8 : How many years of experience do you have in your current role?
# 
# Q9 : What is your current yearly compensation (approximate $USD)?
# 
# Q10: Does you employee incorporate machine learning in their business.
# 
# Q15: Which cloud computing service have you used in the last 5 years. 
# 
# Q16: What programming language do you use on a regular basis.
# 
# Q18: What programming language will you recommend an aspiring data scientist to learn first?
# 
# Q23 : Approximately what percent of your time at work or school is spent actively coding?
# 
# Q24 : How long have you been writing code to analyze data?
# 
# Q25 : For how many years have you used machine learning methods (at work or in school)?
# 
# Q26 : Do you consider yourself to be a data scientist?
# 
# Q36 : On which online platforms have you begun or completed data science courses? (Select all that apply) - Selected Choice
# 
# Q37 : On which online platform have you spent the most amount of time? - Selected Choice
# 
# Q40 : Which better demonstrates expertise in data science: academic achievements or independent projects? - Your views:
# 
# 
# Q50 : What barriers prevent you from making your work even easier to reuse and reproduce? (Select all that apply) - Selected Choice

# <a id = "set_up_dataFrames"></a>
# # <font color ='purple'>STORE ALL THE DATA IN DATAFRAMES</font>

# In[ ]:


df_response = pd.read_csv(response)
df_response.isna().values.any()
# df_response.isna().sum()
# df_response.describe()
# df_response.head()


# In[ ]:


df_mcq = pd.read_csv(mcq)
# df_mcq.head(5)


# <a id = "Gender_and_age"></a>
# # <font color = 'purple'>LET'S LOOK AT THE DATA FROM THE VIEW POINT OF GENDER AND AGE</font>
# 

# In[ ]:


gender_age_orig = df_mcq[['Q1','Q2']]
gender_age_orig.columns = ['gender', 'age_range']

# gender_age_orig.sample(10)
gender_age_orig.head()
# gender_age['gender'].describe()
# gender_age.age_range.unique()


# #  <font color = 'purple'> AGE AND GENDER </font>
# 
# * You can see that the first row in the data frame is the text of the question which isn't really helping us anyway, so I am going to drop it. 
# 
# 

# In[ ]:


gender_age= gender_age_orig
gender_age = gender_age.drop(gender_age.index[[0]])
gender_age.head()


# <a id = "age_and_gender_numerics"></a>
# ## <font color = 'purple'>As you might notice in this dataFrame that the gender column only has two  responses and the age_range column has a set range of responses. So I am going to handle it as follow: </font>
# 
# ** * Convert gender into a binary variable. Where Male = 1 and Female = 0**
# 
# ** * The values in the age_range column has the answers as range that the respondent fell in. **
# 
# I will be converting these values into categorical values that are easier to deal with when doing Data Analysis using pandas built in method.
# 
# So I will be using categorical representations of age instead of ranges.
# 

# In[ ]:


age_unique = gender_age['age_range'].unique()
print('The respondents fell in these age ranges: ')
for age in age_unique:
    print(age)
# gender_age.head()
# gender_age.dtypes


# <a id="age_categories"></a>
# ## <font color = 'green'> Below, you can see how each age range is being assigned a category</font>

# In[ ]:


age_code = {'18-21':1, '22-24':2, '25-29':3,'30-34':4,'35-39':5, '40-44':6,'45-49':7, '50-54':8,'55-59':9, '60-69':10, '70-79':11, '80+':12  }
for k, v in age_code.items():
    print('The code for the age group', k, 'is :', age_code[k])


# In[ ]:


gender_age['gender']= gender_age['gender'].astype('category').cat.codes
# gender_age['age_range'] = gender_age['age_range'].astype('category')
gender_age['age_range'] = gender_age['age_range'].map({'18-21':1, '22-24':2, '25-29':3,'30-34':4,'35-39':5, '40-44':6,'45-49':7, '50-54':8,'55-59':9, '60-69':10, '70-79':11, '80+':12  })#.astype(int)


# In[ ]:


df_ga = gender_age 


# In[ ]:


sns.set(style = "darkgrid")
# sns.set(rc={'figure.figsize':(10,10)})
ax_sns = sns.countplot(x = 'age_range', data = df_ga).set_title('Age range distribution for Data Scientists')


# ## <font color = 'green'> Let's take a look at the age distribution for male and females separately. </font>

# In[ ]:


df_ga = df_ga[df_ga['gender']<2]
df_ga_m = df_ga.loc[df_ga['gender'] == 1]
df_ga_f = df_ga.loc[df_ga['gender'] == 0]


# <a id = "female_age_distribution"></a>
# ## <font color = 'green'> Age distribution of Female Data Scientists</font>

# In[ ]:


# Female
sns.set(style= "darkgrid", rc={'figure.figsize':(8,8)})
ax_f = sns.countplot(x = "age_range", data = df_ga_f).set_title('Age distribution of female Data Scientists')


# <a id = "male_age_distribution"></a>
# ## <font color = 'green'> Age distribution of Male Data Scientists</font>

# In[ ]:


# Male 
sns.set(style= "darkgrid", rc={'figure.figsize':(8,8)})
ax_b = sns.countplot(x = "age_range", data = df_ga_m).set_title('Age distribution of male Data Scientists')


# ## <font color='green'>As we can see in the graphs above that of all the respondents the majority seem to be in the following age ranges: </font>
# 
# * **22-24: which is represented by the value of 2 in the graph. **
# * **25-29: which is represented by the value of 3 in the graph.**
# * **30-34: which is represented by the value of 4 in the graph**
# 
# So, it seems that there are a lot of young data scientists. The hump of this bell curve is centered around people from their early 20s to early 30s

# # <font color = 'purple' > Data Science and Gender Distribution </font>
# **  * In the following section, we are going to examine the age and gender distribtution of Data Scientist  **

# In[ ]:


# df_ga = df_ga[df_ga['gender']<2]
# df_ga_m = df_ga.loc[df_ga['gender'] == 1]
# df_ga_f = df_ga.loc[df_ga['gender'] == 0]


# In[ ]:


males = df_ga_m['gender'].value_counts()
# print(males)
female = df_ga_f['gender'].value_counts()
# print(female)

gender_total = {'gender':[1,0] ,'total': [males.values[0],female.values[0]]}
print(gender_total)
pd_gender_total = pd.DataFrame.from_dict(gender_total)

pd_gender_total


# <a id ="male_female_total"></a>
# ## <font color = 'green'>As shown above, the gender breakdown of people who identify as Data Scientist in the survey is as follows: </font>
#    **  * Male: 19430 **
#    
#    ** * Female: 4010**
#    
#    ## <font color = 'purple'> Below, we can see a histrogram showing the difference in the number of Male vs Female Data Scientists</font>    
#  ** *Female = 0 **
#  
#  ** *Male = 1 **
# 

# 
# 

# In[ ]:


# df_ga[df_ga['gender']==2]
# df_ga['gender'].hist()
sns.set(style="darkgrid", rc={'figure.figsize':(5,5)})
ax_g = sns.countplot(x='gender', data = df_ga).set_title('Total Females(0) vs Males(1) Data Scientists')


# ## <font color ='red'> Observation: While the age distribution seems uniform across both the sexes, there is a wide gap in how many women are in the field comapred to men, based on this dataset.  There are about 5 men for 1 woman working in the field</font>
# 
# ![https://media.giphy.com/media/a69ZfTaapff7W/giphy.gif](https://media.giphy.com/media/a69ZfTaapff7W/giphy.gif)
# 

# 

# <a id = "start_edu"></a>
# # <font color = 'purple'> LET'S LOOK AT THE EDUCATIONAL BACKGROUND OF DATA SCIENTISTS </font>

# In[ ]:


#Q1: Q4, Q5,  
df_education = df_mcq[['Q1','Q4','Q5']]
df_education.columns  = ['gender', 'highest education', 'undergrad major']
df_education = df_education.drop(df_education.index[[0]])
# df_education.sample(5)
df_education.head(6)


# <a id = "all_majors"></a>
# ## <font color = 'green'> Let's see the range of majors that are working as Data Sceintists </font>

# In[ ]:


# df_education.columns
majors =pd.DataFrame(df_education['undergrad major'].value_counts()).reset_index()
# majors = df_education['undergrad major'].unique()
majors.columns = ['undergrad major', 'count']
print('The total number of people answering these questions are:',majors['count'].sum())
# majors.describe() #13 total
majors


# In[ ]:


plt.figure(figsize=(10,5))
# ax = sns.barplot(majors['undergrad major'].head(5), majors['count'].head(5), alpha = 0.8)
ax = sns.barplot(majors['undergrad major'], majors['count'], alpha = 0.8)
ax.set_title('Undergraduate majors of Data Scientists')
ax.set_xlabel('Majors')
ax.set_ylabel('Number of people with the major.')
ax.set_xticklabels(majors['undergrad major'], rotation='vertical',fontsize= 10)
# ax.set_xticklabels(label_text['labels'], rotation='vertical', fontsize=10)
plt.show()                   


# <a id ="majors_summary"></a>
# # <font color = 'red'>Observations:</font>
# * There are 13 categories of answers listed here as response to majors in undergrad.
# * There is one major listed as ** Other** and ** I  never declared a major**. 
# * This would leave 11 declared and clearly defined areas of backfground knowledge. 
# *  As shown in the above plotted histogram, we can see how these 13 responses to undergraduate majors are represented in the field of Data Science. 
# 
# ## The top 5 undergraduate majors who went into Data Science are as follows:
# 1. Computer Science
# 2. Engineering.
# 3.Mathematics or Statistics.
# 4.A business related major.
# 5. Physics or Astronmy. 
# 
# ##  <font color = 'green'>These majors shouldn't be a surprise to anyone, since most of them deal with data in one regard or another.  </font>
# 
# ** However, **  I do find it interesting that business backgrounds come in at 4th out of 13 majors. 
# 
# * I come from a medical background, so I fall even lower on that scale. 
# 
# * I find it interesting that there are people from Humanities and Literature background represented in the mix. I, honesltly, would not have expected them to be in the top 10. 
# 
# 
# 
# 
# 
# 

# <a id = "highest_education"></a>
# # <font color = 'purple'> Now, I want to see what is the highest education level of all these respondents.</font>

# In[ ]:


high_edu = df_education[['gender', 'highest education']]
high_edu.head(5)


# In[ ]:


edu_count = pd.DataFrame(high_edu['highest education'].value_counts()).reset_index()
edu_count.columns = ['degree', 'count']
edu_count


# <a id  = "hist_degrees"></a>
# ## <font color = 'green'> The following histogram shows the share of different education levels held by Data Scientists</font>

# In[ ]:


plt.figure(figsize= (10,5))
ax = sns.barplot(edu_count['degree'], edu_count['count'])
ax.set_title('Degrees held by Data Scientists')
ax.set_xlabel('Degree')
ax.set_ylabel('Number of people')
ax.set_xticklabels(edu_count['degree'], rotation='vertical', fontsize=10)
plt.show()


# <a id = "degree_observ"></a>
# # <font color = 'red'>Observations on degrees held by Data Scientists: </font>
# * You can see that there seems to be a continuum of different educational backgrounds amongst Data Scientists.
# * The most common degree held by Data Scientists is **Masters, then Bachelors, then a PhD.**
# * You can also see that at # 4 are people who seem to have dropped out after some college level education. Ofcourse, you could speculate about the reasoning here.
# * Then at 232 people out of 22947 people didn't do any schooling past high school. They form 1.01% of the respondents
# 
# 

# <a id = "edu_salary"></a>
# # <font color ='purple'> LET'S SEE HOW THE YEARLY SALARY BREAKS DOWN BASED ON THEIR DEGREE</font>

# In[ ]:


df_edu_money = df_mcq[['Q4','Q9']]
df_edu_money = df_edu_money.drop(df_edu_money.index[[0]])
df_edu_money.columns = ['highest education', 'salary']
df_edu_money= df_edu_money.dropna()
# I am going to drop all the rows where the participants didn't disclose their salary
df_edu_money= df_edu_money[df_edu_money['salary'] !='I do not wish to disclose my approximate yearly compensation']
# df_edu_money.head(7)
df_edu_money.sample(10)
# df_edu_money['salary'].unique()

# df_edu_money['salary'].head(7).unique()


# <a id = "salaries_range"></a>
# ## <font color = 'green'>Since there are a range of salaries for people, I am going to create a dictionary where I will map these ranges to numbers and then handle them</font>

# In[ ]:


salary_scales = {'0-10,000':0, '10-20,000':1,'20-30,000':2, '30-40,000':3,
                 '40-50,000':4,'50-60,000':5, '60-70,000':6,
                 '70-80,000':7, '80-90,000':8,'90-100,000':9,'100-125,000':10,
                 '125-150,000':11,'150-200,000':12 ,'200-250,000':13, '250-300,000':14 ,
                 '300-400,000':15,'400-500,000':16 , '500,000+':17}

for k, v in salary_scales.items():
    print('For anyone earning $',k,'their pay scale is:', v)


# In[ ]:


# gender_age['gender']= gender_age['gender'].astype('category').cat.codes
# # gender_age['age_range'] = gender_age['age_range'].astype('category')
# gender_age['age_range'] = gender_age['age_range'].map({'18-21':1, '22-24':2, '25-29':3,'30-34':4,'35-39':5, '40-44':6,'45-49':7, '50-54':8,'55-59':9, '60-69':10, '70-79':11, '80+':12  })#.astype(int)

# df_edu_money['salary'] =df_edu_money['salary'].astype('category').cat.codes
df_edu_money['salary'] = df_edu_money['salary'].map({k:v for k, v in salary_scales.items()})


# In[ ]:


# df_test = pd.DataFrame(df_edu_money.groupby('highest education'))
# df_test = df_edu_money.groupby(['highest education'])['salary'].aggregate()
# df_test
df_money_avg= df_edu_money.groupby('highest education',as_index= False)['salary'].mean().round()
df_money_avg


# <a id = "observ_salary"></a>
# # <font color = 'red'> Observations about average salaries based on highest educational degree. </font >
# ## These observations are based on the mean calculations done in the above data frame
# 
# * Bachelors = 3rd scale which is $30-40,000/year
# 
# * Masters = 4th scale which is $40-50,000/year
# 
# * Doctorate = 5th scale which is $50-60,000/year
# 
# ## <font color = 'red'>Here is what I find interesting:  </font>
# * The average salary of people with no formal education past high school is also on the 4th scale which is the same as a Masters Degree, being 
#     $40-50,000/year. 
#      Ofcourse, we must remember that there are far more people with Masters degress than people with no higher education past high school.  
#   **(10855 for Masters degree vs 232 for people with no formal education past high school) **

# <a  id="salary_outliers"></a>
# # <font color = 'purple'>YEARLY SALARY OUTLIERS AND THEIR EDUCATIONAL BACKGROUND </font>
# 
# ## I will be looking at people making over$200,000 or more a year.
# 
# I will be creating separate data frames for people making between $$200,000 to $300,00. Then, a second one for people making between $300,000 and $400,000 and so on until $500,000+

# In[ ]:


#For people making between $200,000 and $300,000
df_2k = df_edu_money[df_edu_money['salary'].isin([13,14])]
# df_2k['salary'].value_counts()
df_2k.sample(10)
df_2k_people = pd.DataFrame(df_2k['highest education'].value_counts()).reset_index()
df_2k_people.columns = ['degree', '200-300']
df_2k_people


# In[ ]:


# For people making between $300,00 and $400,000
df_3k = df_edu_money[df_edu_money['salary']==15]
# df_3k.sample(10)
# df_3k['highest education'].value_counts()
df_3k_people = pd.DataFrame(df_3k['highest education'].value_counts()).reset_index()
df_3k_people.columns = ['degree', '300-400']
df_3k_people


# In[ ]:


# For people making between $400,000 and $500,000
df_4k = df_edu_money[df_edu_money['salary']==16]
df_4k.sample(10)
df_4k_people = pd.DataFrame(df_4k['highest education'].value_counts()).reset_index()
df_4k_people.columns = ['degree', '400-500']
df_4k_people


# In[ ]:


# For people making over $500,000+ 
df_5k = df_edu_money[df_edu_money['salary']==17]
df_5k.sample(10)
df_5k_people = pd.DataFrame(df_5k['highest education'].value_counts().reset_index())
df_5k_people.columns =  ['degree', '500+']
df_5k_people


# ## <font color = 'green'> Now, I am going to create one giant dataframe containg the degree information for people making over $200,000 +</font>

# In[ ]:


df_rich =pd.merge(pd.merge(pd.merge(df_2k_people, df_3k_people, on ='degree'), df_4k_people, on= 'degree'), df_5k_people, on='degree')
df_rich


# <a id = "rich_money"></a>
# 
# <font color ='green'> The following histogram shows how many people from each degree make a lot of money</font >

# In[ ]:


plt.figure(figsize=(8,8))
# ax= sns.barplot(df_rich['degree'], df_rich[['200-300','300-400','400-500','500+']])
df_rich.plot(x= 'degree', y= ['200-300','300-400','400-500','500+'], kind ='bar', figsize = (10,5), title= '$200,000 + earners and their degrees')


# <a id ="rich_observ"></a>
# # <font color = 'red'>OBSERVATION</font>
# ## It seems that if you are earning $200,000+ a year, you are more likely than not to have a Masters degree. 

# <a id = "ml"></a>
# # <font color ='purple'>One last question that I have: How much more would you make if your company employs machine learning </font>

# In[ ]:


#  ML Questions: Q9, Q10, Q25

df_ml = df_mcq[['Q9', 'Q10', 'Q25']]
df_ml.columns = ['salary', 'yes/no','years of ml']
df_ml = df_ml.drop(df_ml.index[[0]])
df_ml = df_ml.dropna()
df_ml= df_ml[df_ml['salary'] !='I do not wish to disclose my approximate yearly compensation']
df_ml = df_ml[df_ml['yes/no']!= 'I do not know']
df_ml['yes/no'].unique()
# df_ml


# <a id = "ml_responses"></a>
# ## So there are four answers given to the question of whether the companies employing these data scientists are using machine learning. I am going to map them onto numbers as such:
# 
# * No (we do not use ML methods) : **0**
# * We recently started using ML methods (i.e., models in production for less than 2 years) : **1**
# * We have well established ML methods (i.e., models in production for more than 2 years) : **1**
# * We are exploring ML methods (and may one day put a model into production) : **1**
# * We use ML methods for generating insights (but do not put working models into production) : **1**
# 
# ## <font color = 'green'> Since, I am only interested in whether a company uses machine learning or not. I don't care how long they have been using it. So, I am going to map all the choices other than *No* to 1 </font>
# 
# ** For salaries, I will be using the same method as that which I used to study degree and their correlation to yearly salaries **
# 

# In[ ]:


ml_dict = {'No (we do not use ML methods)': 0,
          'We recently started using ML methods (i.e., models in production for less than 2 years)':1,
           'We have well established ML methods (i.e., models in production for more than 2 years)':1,
           'We are exploring ML methods (and may one day put a model into production)':1,
           'We use ML methods for generating insights (but do not put working models into production)':1           
          }
for k,v in ml_dict.items():
    print(k,':',v)


# In[ ]:


df_ml_salary = df_ml[['salary', 'yes/no']]
# df_edu_money['salary'] = df_edu_money['salary'].map({k:v for k, v in salary_scales.items()})
df_ml_salary['salary'] = df_ml_salary['salary'].map({k:v for k,v in salary_scales.items()})
df_ml_salary['yes/no'] = df_ml_salary['yes/no'].map({k:v for k,v in ml_dict.items()})
df_ml_salary.sample(10)


# <a id = "ml_salary"></a>
# # <font color = 'green'> In the box plot below you can see the number of people making over $100,000 a year or more at the companies using Machine Learning</font>

# In[ ]:


df_ml_yes = df_ml_salary[df_ml_salary['yes/no']==1]
df_ml_ycount= pd.DataFrame(df_ml_yes['salary'].value_counts().reset_index())
df_ml_ycount.columns= ['salary scale', 'count']
df_ml_y100K = df_ml_ycount[df_ml_ycount['salary scale']>=10]
df_ml_y100K.plot(x='salary scale', y= 'count', kind= 'bar', title = 'Salary scale for Companies using Machine Learning(only people making over $100K+)', figsize= (10,5))


# In[ ]:


print('At the comapnies using Some level of machine learning',round(df_ml_y100K['count'].sum()/df_ml_ycount['count'].sum()*100), '% of employees make over $100k+ a year')
print('The mean salary scale at companies using machine learnig is:',df_ml_ycount['salary scale'].mean())
print('An 8.5 scale translates to $80,000 to $100,000 per year as a mean salary for the employees')
print(df_ml_ycount['count'].sum(), 'people who responded to this question said they worked for a company where machine learning was involved at some level')


# <a id = "noml_salary"></a>
# # <font color = 'green'> In the box plot below you can see the number of people making over $100,000 a year or more at the companies which DON'T Machine Learning</font>

# In[ ]:


df_ml_no = df_ml_salary[df_ml_salary['yes/no']==0]
df_ml_ncount= pd.DataFrame(df_ml_no['salary'].value_counts().reset_index())
df_ml_ncount.columns= ['salary scale', 'count']
df_ml_n100K = df_ml_ncount[df_ml_ncount['salary scale']>=10]
df_ml_n100K.plot(x='salary scale', y= 'count', kind= 'bar', title = 'Salary scale for Companies not using Machine Learning(only people making over $100K+)', figsize= (10,5))


# In[ ]:


print('At the comapnies not using any kind machine learning',round(df_ml_n100K['count'].sum()/df_ml_ncount['count'].sum()*100), '% of employees make over $100k+ a year')
print('The mean salary scale at companies not using machine learnig is:',df_ml_ncount['salary scale'].mean())
print('An 8 scale translates to $80,000 to $90,000 per year as a mean salary for the employees')
print(df_ml_ncount['count'].sum(), 'people who responded to this question said they worked for a company where there was no machine learning used')


# <a id = "observ_ml"></a>
# # <font color ='red'> OBSERVATIONS ABOUT SALARIES AND COMPANIES USING MACHINE LEARNING  </font>
# 
# **Companies using machine learning:**
#    * At least 9395 people who responded to questions regarding the use of machine learning at their work said responded with a yes. 
#    * Almost 19% of the respondents working at such firms earned at least $100,000/year. 
#    
#    * The mean salary at such companies was from $$80,000 to $100,000 a year.
#    
#    **Companies not using machine learning:**
#    * At least 2938  people who responded to questions regarding the use of machine learning at their work said responded with a yes.
#    *  Almost 6% of the respondets working at such firms earned at least $100,000/ year.
#    
#    * The mean salary at such companies was from $$80,000 to $90,000 a year.
#    
#    ## <font color = 'purple'>In short, if you work at a company which employs machine learning at some level, you are likely to be paid more </font>

# <a id = "final_thoughts"></a>
# 
# # <font color = 'purple'> I had meant to answer a few questions and  I have answered them</font>
# 
# * Data Science is a male dominated field where most of the practitioners are between 20 and 34 years old. 
# * The most common background of people coming to this field is some kind of science/technical/business background
# * The most common degree held by Data Science practitioners is a Masters Degree in a technical/science field.
# *  I found out that people with Masters Degrees are more likely to earn more than $200k a year in salary. 
# * People working at places which employ machine learning at some levels, have a higher mean pay than those who work at places which DON'T use machine learning at all
