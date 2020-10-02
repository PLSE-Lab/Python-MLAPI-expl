#!/usr/bin/env python
# coding: utf-8

# ![](https://i.imgur.com/G0jD7fS.jpg)
# 
# # A Day in the life of a Software Developer
# > Software developers are the creative minds behind computer programs. Some develop the applications that allow people to do specific tasks on a computer or another device. Others develop the underlying systems that run the devices or that control networks. In this notebook, I will try to find the typical day in a software developer. I will be analysing Stackoverflow Developer Survey 2018 dataset for this notebook. Feel free for any suggestion and if you find this useful please upvote. 

# In[1]:


# load our necessary libary
import pandas as pd
from pandas import Series
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
df_survey_results = pd.read_csv('../input/survey_results_public.csv', low_memory=False)


# # 1. How's the day start? 
# > In this section, I will try to understand how developers start his day. Like when he wakes up, does he exercise. And how his morning effect in her professional life. Let's begin!

# In[2]:


plt.figure(figsize=(10,8))

df_survey_results.WakeTime.value_counts().sort_values(ascending=True).plot.barh(width=0.9,color=sns.color_palette('inferno_r',15))
plt.savefig('sta.png')
plt.show()


# We can see that majority of developers are the early riser. They wake up between 6 to 8 am. That was a very healthy sign. But how this routine help in their professional workflow. Let's find out! 

# In[3]:


# plt.figure(figsize=(15, 8))
f,ax=plt.subplots(1,2,figsize=(25,10))

sns.countplot(x="WakeTime", hue="JobSatisfaction", data=df_survey_results, ax=ax[0])
sns.countplot(x='WakeTime', hue='CareerSatisfaction', data=df_survey_results, ax=ax[1])
ax[0].set_title('Effect of wake up time in job satisfaction', fontsize=18)
ax[1].set_title('Effect of wake up time in career satisfaction', fontsize=18)
ax[0].tick_params(axis='x', labelsize=18,rotation = 90)
ax[1].tick_params(axis='x', labelsize=18, rotation=90)
plt.setp(ax[0].get_legend().get_texts(), fontsize='18') # for legend text
plt.setp(ax[0].get_legend().get_title(), fontsize='20') # for legend title
plt.setp(ax[1].get_legend().get_texts(), fontsize='18') # for legend text
plt.setp(ax[1].get_legend().get_title(), fontsize='20') # for legend title
plt.show()


# Awesome! That is a very good information. Let's understand the two plots. Those who wake up between 6 to 7 am they are **Extremely Satisfied** with their job. And those who wake up between 7 to 8 am they are **moderately satisfied** with their job. Also, those who wake up between 6 to 7 am they are also **Extremely satisfied** with their career. Okay now let's see the exercise status of our developers. 

# In[4]:


f,ax=plt.subplots(1,2,figsize=(25,10))

df_survey_results.Exercise.value_counts().sort_values(ascending=True).plot.barh(width=0.9,color=sns.color_palette('inferno_r',15), ax = ax[0])
df_survey_results.SkipMeals.value_counts().sort_values(ascending=True).plot.barh(width=0.9,color=sns.color_palette('inferno_r',15), ax = ax[1])
ax[0].tick_params(axis='y', labelsize=18)
ax[1].tick_params(axis='y', labelsize=18)
ax[0].set_title('How developers exercise in a week', fontsize=18)
ax[1].set_title('How developers skip their meals in a week', fontsize=18)
plt.show()


# Oops! That was embarrassing. Majority of developers don't exercise. And also good news that majority of developers never skip meals for their works. Now we see how the developers start his day. Now let's see what developers do in his workspace. 

# # 2. What do developers do in his workspace?
# > In this section, we will look into what developers do in his workspace, how much time he spends on his computer, how much time spends on outside and many more. Let's dive in!

# In[ ]:


f,ax=plt.subplots(1,2,figsize=(25,10))

df_survey_results.HoursComputer.value_counts().sort_values(ascending=True).plot.barh(width=0.9,color=sns.color_palette('inferno_r',15), ax = ax[0])
df_survey_results.HoursOutside.value_counts().sort_values(ascending=True).plot.barh(width=0.9,color=sns.color_palette('inferno_r',15), ax = ax[1])
ax[0].tick_params(axis='y', labelsize=18)
ax[1].tick_params(axis='y', labelsize=18)
ax[0].set_title('Time spent on computer in a day', fontsize=18)
ax[1].set_title('Time spent outside in a day', fontsize=18)
plt.show()


# Majority of developers work 9 to 12 hours in a day. And 1 to 2 hours outside. Now let's see developers use ergonomic devices use or not. 

# In[ ]:


plt.figure(figsize=(15,10))

df_survey_results.ErgonomicDevices.value_counts().sort_values(ascending=True).plot.barh(width=0.9,color=sns.color_palette('inferno_r',15))
plt.title('Ergonomic devices use by developers', fontsize=18)
plt.yticks(fontsize=18)
plt.show()


# Awesome! Majority of developers use Ergonomic Keyboard and mouse and a standing desk. Okay, now we see what ergonomic devices they use now let's see what tools they use. Like the programming language, IDE, database, framework, number of monitors, operating system and many more!

# In[ ]:



f,ax=plt.subplots(2,4,figsize=(25,25))

from pandas import Series

s = df_survey_results['LanguageWorkedWith'].str.split(';').apply(Series, 1).stack()
s.index = s.index.droplevel(-1)
s.name = 'Language'
df_language = df_survey_results.join(s)
df_language.Language.value_counts().sort_values(ascending=True).plot.barh(width=0.9,color=sns.color_palette('inferno_r',15), ax=ax[0][0])



s = df_survey_results['DatabaseWorkedWith'].str.split(';').apply(Series, 1).stack()
s.index = s.index.droplevel(-1)
s.name = 'Database'
df_database = df_survey_results.join(s)
df_database.Database.value_counts().sort_values(ascending=True).plot.barh(width=0.9,color=sns.color_palette('inferno_r',15), ax=ax[0][1])

s = df_survey_results['PlatformWorkedWith'].str.split(';').apply(Series, 1).stack()
s.index = s.index.droplevel(-1)
s.name = 'Platform'
df_platform = df_survey_results.join(s)
df_platform.Platform.value_counts().sort_values(ascending=True).plot.barh(width=0.9,color=sns.color_palette('inferno_r',15), ax=ax[0][2])

s = df_survey_results['FrameworkWorkedWith'].str.split(';').apply(Series, 1).stack()
s.index = s.index.droplevel(-1)
s.name = 'Framework'
df_framework = df_survey_results.join(s)
df_framework.Framework.value_counts().sort_values(ascending=True).plot.barh(width=0.9,color=sns.color_palette('inferno_r',15), ax=ax[0][3])


s = df_survey_results['IDE'].str.split(';').apply(Series, 1).stack()
s.index = s.index.droplevel(-1)
s.name = 'ide'
df_ide = df_survey_results.join(s)
df_ide.ide.value_counts().sort_values(ascending=True).plot.barh(width=0.9,color=sns.color_palette('inferno_r',15), ax=ax[1][0])

s = df_survey_results['Methodology'].str.split(';').apply(Series, 1).stack()
s.index = s.index.droplevel(-1)
s.name = 'Working_Methodology'
df_methodology= df_survey_results.join(s)
df_methodology.Working_Methodology.value_counts().sort_values(ascending=True).plot.barh(width=0.9,color=sns.color_palette('inferno_r',15), ax=ax[1][1])

s = df_survey_results['VersionControl'].str.split(';').apply(Series, 1).stack()
s.index = s.index.droplevel(-1)
s.name = 'Version Control'
df_version= df_survey_results.join(s)
df_version['Version Control'].value_counts().sort_values(ascending=True).plot.barh(width=0.9,color=sns.color_palette('inferno_r',15), ax=ax[1][2])


df_survey_results.OperatingSystem.value_counts().sort_values(ascending=True).plot.barh(width=0.9,color=sns.color_palette('inferno_r',15), ax=ax[1][3])






plt.subplots_adjust(wspace=0.8)
ax[0][0].set_title('Top programming languages used by our developers')
ax[0][1].set_title('Top Database solutions used by our developers')
ax[0][2].set_title('Top platforms used by our developers')
ax[0][3].set_title('Top framework used by our developers')
ax[1][0].set_title('Top IDE used by our developers')
ax[1][1].set_title('Top methodology used by our developers')
ax[1][2].set_title('Top Version control used by our developers')
ax[1][3].set_title('Top operating used by our developers')
plt.show()


# Wait! There was a lot of information. Surely, JavaScript wins the programming language challenge. Because JavaScript is used by Web developers, Mobile developers and other fields. The top database is MySQL and SQL server. 
# 
# Top platform is Linux (As expected). The top framework is node.js because it is a JavaScript library. And top IDE is visual studio notepad, vim and sublime text. 
# 
# After that, we see Agile and Scrum are two popular methodologies used by our developers. And as always Git is popular among developers. Lastly, windows is the most used operating system used by our developers in daily life. 
# 
# Now let's see some more!

# In[2]:


MULTIPLE_CHOICE = [
    'DatabaseWorkedWith','PlatformWorkedWith',
    'PlatformDesireNextYear','Methodology','VersionControl','LanguageWorkedWith']
temp_df = df_survey_results[MULTIPLE_CHOICE]
# Go through all object columns
for c in MULTIPLE_CHOICE:
    
    # Check if there are multiple entries in this column
    temp = temp_df[c].str.split(';', expand=True)

    # Get all the possible values in this column
    new_columns = pd.unique(temp.values.ravel())
    for new_c in new_columns:
        if new_c and new_c is not np.nan:
            
            # Create new column for each unique column
            idx = temp_df[c].str.contains(new_c, regex=False).fillna(False)
            temp_df.loc[idx, f"{c}_{new_c}"] = 1

    # Info to the user
    print(f">> Multiple entries in {c}. Added {len(new_columns)} one-hot-encoding columns")

    # Drop the original column
    temp_df.drop(c, axis=1, inplace=True)
        
# For all the remaining categorical columns, create dummy columns
temp_df = pd.get_dummies(temp_df)
temp_df = temp_df.fillna(0)


# In[ ]:


use_features = [x for x in temp_df.columns if x.find('LanguageWorkedWith_') != -1]

def plot_corr(df,size=10):
    '''Function plots a graphical correlation matrix for each pair of columns in the dataframe.

    Input:
        df: pandas DataFrame
        size: vertical and horizontal size of the plot'''
    
    df = df[use_features]
    df.rename(columns=lambda x: x.split('_')[1], inplace=True)
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(size, size))
    ax.matshow(corr)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90);
    plt.yticks(range(len(corr.columns)), corr.columns);
    
plot_corr(temp_df)


# Wait! There is a lot of information. The plot is self explanotary. But my favorite part is that the first few language like JavaScript, Python, HTML, CSS, Bash, Java, C++ are highly correleted with each other. We can also see some relation between Ruby and R. That's sound interesting. 

# # 3. Future developers
# > Let's see what our future developers want to do. What career they do want to choose? Their undergrad major, formal education and job search status. Let's do this!

# In[ ]:


plt.figure(figsize=(15,13))
df3 = df_survey_results.dropna(subset=['Student', 'HopeFiveYears'])
sns.heatmap(pd.crosstab(df3.HopeFiveYears, df3.Student))
plt.title('Plan for five years of our future developers')
plt.yticks(fontsize=18)
plt.xticks(fontsize=18)
plt.ylabel('Hope for five years',fontsize=18)
plt.xlabel('Student',fontsize=18)
plt.show()


# That was a very good information. We can see that our future developers, I mean those are currently students want to open a company and work there as a founder or co-founder of their own company. And others group want to work in a more specialized technical role. 

# In[ ]:


f,ax=plt.subplots(3,1,figsize=(10,25))

sns.heatmap(pd.crosstab(df_survey_results.FormalEducation, df_survey_results.Student), ax=ax[0])
sns.heatmap(pd.crosstab(df_survey_results.UndergradMajor, df_survey_results.Student), ax=ax[1])
sns.heatmap(pd.crosstab(df_survey_results.JobSearchStatus, df_survey_results.Student), ax=ax[2])
plt.subplots_adjust(wspace=0.8)
plt.show()


# Oh! That's a lot of cool information. Let's break down those plot into text: 
# 
# **First plot: ** It shows the formal education of our students. Also, we can compare that to non-student and part-time student. We can see that our full-time student's formal education are most often Bachelor degree. Also, non-students formal education Bachelor degree and that is also true for part-time students. 
# 
# **Second plot: ** It shows us that those who participate in this survey their undergrad major is computer science. 
# 
# **Third plot: ** It shows us that our students are not actively looking for jobs but they are open to opportunity. 
# 
# We have learnt a lot about our future developers. Let's see what we can find more!
# 
# Let's see how our developers using Stackoverflow!

# # 4. How are our developers using Stackoverflow?
# > StackOverflow is a popular question and answer site for developers. You can find nearly any answer you want. In my daily life, I visit StackOverflow lot of times. Facing problem and bugs is a daily scenario for developers. So, let's see how our developers using StackOverflow. 

# In[ ]:


f,ax=plt.subplots(2,4,figsize=(25,25))
df_survey_results.StackOverflowRecommend.value_counts().sort_values(ascending=True).plot.barh(width=0.9,color=sns.color_palette('inferno_r',15), ax=ax[0][0])
df_survey_results.StackOverflowVisit.value_counts().sort_values(ascending=True).plot.barh(width=0.9,color=sns.color_palette('inferno_r',15), ax=ax[0][1])
df_survey_results.StackOverflowHasAccount.value_counts().sort_values(ascending=True).plot.barh(width=0.9,color=sns.color_palette('inferno_r',15), ax=ax[0][2])
df_survey_results.StackOverflowParticipate.value_counts().sort_values(ascending=True).plot.barh(width=0.9,color=sns.color_palette('inferno_r',15), ax=ax[0][3])
df_survey_results.StackOverflowJobs.value_counts().sort_values(ascending=True).plot.barh(width=0.9,color=sns.color_palette('inferno_r',15), ax=ax[1][0])
df_survey_results.StackOverflowDevStory.value_counts().sort_values(ascending=True).plot.barh(width=0.9,color=sns.color_palette('inferno_r',15), ax=ax[1][1])
df_survey_results.StackOverflowJobsRecommend.value_counts().sort_values(ascending=True).plot.barh(width=0.9,color=sns.color_palette('inferno_r',15), ax=ax[1][2])
df_survey_results.StackOverflowConsiderMember.value_counts().sort_values(ascending=True).plot.barh(width=0.9,color=sns.color_palette('inferno_r',15), ax=ax[1][3])




plt.subplots_adjust(wspace=0.8)
ax[0][0].set_title('Developers recommend Stackoverflow')
ax[0][1].set_title('Developers visit Stackoverflow')
ax[0][2].set_title('Developers have account on Stackoverflow')
ax[0][3].set_title('Developers participate in Stackoverflow')
ax[1][0].set_title('Developers using Stackoverflow job board')
ax[1][1].set_title('Developers using Stackoverflow Developer Story feature')
ax[1][2].set_title('Developers recommend Stackoverflow Jobs')
ax[1][3].set_title('Developers who are Stackoverflow members')
plt.show()


# We found a lot of information. Let's break down these plots part by part. 
# 
# ** First Plot: ** We can see from the first plot that developers are very likely to recommend StackOverflow. They are very positive about StackOverflow. 
# 
# ** Second Plot: ** Bugs in code is a daily problem for developers. So they visit Stackoverflow most often, multiple times per day. The plot shows the same trend. 
# 
# ** Third Plot: ** We can see most of the developers have an account on StackOverflow. 
# 
# ** Fourth Plot: ** The plot shows us that developers participate in StackOverflow Q&A in less than once per month.
# 
# ** Fifth Plot: ** We see that most of the developers hear about StackOverflow Jobs Board. But we can also see that a large portion of developers hear about it but never visited. 
# 
# ** Sixth Plot: ** The sixth plot says that most of the developers don't use Developer Story feature on StackOverflow and they also don't know what it is.
# 
# ** Seventh Plot: ** Developers are very likely to recommend StackOverflow Jobs. 
# 
# ** Eighth Plot: ** Most of the developers considers themselves a StackOverflow member. 
# 
# Awesome! That's a lot of information. Let's see what we can find about StackOverflow visiter. Let's see some correleation between StackOverflow Vister vs others column. And see what we find!

# In[ ]:


f,ax=plt.subplots(3,2,figsize=(25,30))

sns.heatmap(pd.crosstab(df_survey_results.StackOverflowDevStory, df_survey_results.StackOverflowParticipate), ax=ax[0][0])
sns.heatmap(pd.crosstab(df_survey_results.StackOverflowVisit, df_survey_results.JobSearchStatus), ax = ax[0][1])
sns.heatmap(pd.crosstab(df_survey_results.StackOverflowVisit, df_survey_results.Employment), ax= ax[1][0])
sns.heatmap(pd.crosstab(df_survey_results.StackOverflowVisit, df_survey_results.OpenSource), ax= ax[1][1])
sns.heatmap(pd.crosstab(df_survey_results.StackOverflowVisit, df_survey_results.Age), ax= ax[2][0])
sns.heatmap(pd.crosstab(df_survey_results.StackOverflowVisit, df_survey_results.AdBlocker), ax= ax[2][1])

ax[0][0].tick_params(axis='y', labelsize=18)
ax[0][0].tick_params(axis='x', labelsize=15)
ax[0][1].tick_params(axis='y', labelsize=18)
ax[0][1].tick_params(axis='x', labelsize=15)
ax[1][0].tick_params(axis='y', labelsize=18)
ax[1][0].tick_params(axis='x', labelsize=15)
ax[1][1].tick_params(axis='y', labelsize=18)
ax[1][1].tick_params(axis='x', labelsize=18)
ax[2][0].tick_params(axis='y', labelsize=18)
ax[2][0].tick_params(axis='x', labelsize=18)
ax[2][1].tick_params(axis='y', labelsize=18)
ax[2][1].tick_params(axis='x', labelsize=18)

plt.subplots_adjust(wspace=0.8, hspace=.99)
plt.show()


# There is a lot of information. Take a deep breath and let's see what we find. 
# 
# In the first plot, we see those who participate less than once don't know what is dev story is all about. But we see in a previous plot that most of the developers participate in StackOverflow less than once per month. That's a very embarrassing news. 
# 
# In the second plot, we see that those who visit Stackoverflow frequently they are not actively looking for a job opportunity but they are open to offers. 
# 
# The third plot is self-explanatory and shows us expected results. Because we see full time employed developers visit Stackoverflow more frequently. In the fourth plot, we see that those who visit Stackoverflow daily have a higher chance that they don't participate in Open source. 
# 
# The fifth correlation shows some interesting information. It shows us that developers whose age is between 25 to 34 have a higher rate for visiting StackOverflow more frequently. And then the next position is dominated by our young developers whose age is 18 - 24 years old. 
# 
# The last plot shows us some bad news. It shows us that most frequently visited developers have a higher rate of using AdBlocker. That's a very sad new for StackOverflow. 

# # 5. But wait! How our developer's ethics are? 
# > Our developers can make awesome solutions. But how do they response to un ethical projects? Let's find out! 

# In[ ]:


f,ax=plt.subplots(2,2,figsize=(25,25))
df_survey_results.EthicsChoice.value_counts().sort_values(ascending=True).plot.barh(width=0.9,color=sns.color_palette('inferno_r',15), ax=ax[0][0])
df_survey_results.EthicsReport.value_counts().sort_values(ascending=True).plot.barh(width=0.9,color=sns.color_palette('inferno_r',15), ax=ax[0][1])
df_survey_results.EthicsResponsible.value_counts().sort_values(ascending=True).plot.barh(width=0.9,color=sns.color_palette('inferno_r',15), ax=ax[1][0])
df_survey_results.EthicalImplications.value_counts().sort_values(ascending=True).plot.barh(width=0.9,color=sns.color_palette('inferno_r',15), ax=ax[1][1])



plt.subplots_adjust(wspace=0.8)
ax[0][0].set_title('Response for writing code for a project that is unethical')
ax[0][1].set_title('Do developers want to report for the unethical project')
ax[1][0].set_title('Who is reponsible for code that is unethical')
ax[1][1].set_title('Do developers think about ethical purpose of their code')
plt.show()


# We found some important information. We see that if developers request for writing code for unethical purpose then they will not write that code. We also see they are ready to report for that request. They think that upper management of the company is responsible for this kind of unethical purpose. We also see that developers think about the ethical side of their code. That is a very good sign. 

# # 6. Hey! What about salary?
# > Our developers done a lot of hard work. What about their salary. Do they get what they deserve?  Let's find out! Shall we? 

# In[ ]:


# df_survey_results_without_nan_salry = df_survey_results.dropna(subset=['ConvertedSalary'])
# sns.distplot(df_survey_results_without_nan_salry.ConvertedSalary)

data_dem = df_survey_results[(df_survey_results['ConvertedSalary']>5000) & (df_survey_results['ConvertedSalary']<1000000)]

plt.subplots(figsize=(15,8))
sns.distplot(data_dem['ConvertedSalary'])
plt.title('Income histograms and fitted distribtion',size=15)
plt.show();


# In[ ]:


print('The median salary of developers: {} USD'.format(data_dem['ConvertedSalary'].median()
))
print('The mean salary of developers: {:0.2f} USD'.format(data_dem['ConvertedSalary'].mean()
))


# We can see from the plot that most of the developer's annual salary lies in under 100k $. We can also see that the median salary of developers is 58752.0 USD and mean salary is 79812.63 USD. 
# 
# Seaborn's 'distplot' fits a univariate distribution using kernel density estimation KDE. We notice with the bins that most developers have an income between 70k and 130k and that the fitted distribution is skewed right which means there's much more outliers towards the right (unusually high incomes) than towards the left.
# 
# Here's a more sophisticated plot for the distribution of the annual income.

# In[ ]:


plt.figure(figsize=(15,8))
sns.violinplot(x='ConvertedSalary', data=data_dem)
plt.title("Salary distribution of our developers", fontsize=16)
plt.xlabel("Annual Salary", fontsize=16)
plt.show();


# We can see something similar here. We see that most of the developers salary is under 100k. 

# ### Salary vs Gender

# In[ ]:


temp=data_dem[data_dem.Gender.isin(['Male','Female'])]
plt.figure(figsize=(10,8))
sns.violinplot( y='ConvertedSalary', x='Gender',data=temp)
plt.title("Salary distribution Vs Gender", fontsize=16)
plt.ylabel("Annual Salary", fontsize=16)
plt.xlabel("Gender", fontsize=16)
plt.show();


# The salary gap between men and women isn't too big. But still in favour of man. 
# 
# The avarage salary of male developers is a bit high compare to women developers. 

# ### Developers salary in different country
# > Let's find out how our developers getting paid in different country. I found some problem when directly calculate top country wise salary. So I decided to count popular respondent countries. 

# In[ ]:


resp_coun=df_survey_results['Country'].value_counts()[:15].to_frame()

f,ax=plt.subplots(1,1,figsize=(18,8))
max_coun=df_survey_results.groupby('Country')['ConvertedSalary'].median().to_frame()
max_coun=max_coun[max_coun.index.isin(resp_coun.index)]
max_coun.sort_values(by='ConvertedSalary',ascending=True).plot.barh(width=0.8,ax=ax,color=sns.color_palette('RdYlGn'))
ax.axvline(df_survey_results['ConvertedSalary'].median(),linestyle='dashed')
ax.set_title('Compensation of Top 15 Respondent Countries')
ax.set_xlabel('')
ax.set_ylabel('')
plt.subplots_adjust(wspace=0.8)
plt.show()


# ## Wow! we find the top paying country. But wait there is a problem. 
# 
# When you're travelling across the country you live in, you notice that prices for food, rent and common goods vary from one city to another. This happens on a much larger scale when changing countries!
# Let's take an example: Consider 1 USD, Google tells you that it equals 64.92 Indian Rupee. An average dinner in the US may cost you 10-12 USD which corresponds to 640-760 INR. But in fact, it appears that you could get yourself a good meal with 150-200INR in India. In other words: You can do with 1USD in India more than what you would do with 1USD in the US.
# 
# As a basic example, let's say we find that 1USD in India = 3USD in the US, that would mean that if you get paid 3000 USD in India you would lead the same someone would be leading in the US if he's getting paid 9000 USD.
# 
# Here comes the Purchasing Power Parity (PPP from now on): Faced with this problematic of different costs of livings, economists came up with a ratio to convert currencies without using market exchange rates. To do so, they select a basket of goods, say orange/Milk/tomato ..., and check its value in the USD and in the country they're interested in.
# Let's say they find that :
# 
# Value of the basket in the US = 10 USD
# Value of the basket in India = 200 INR
# Then they would say that PPP(India) = 200/10 = 20, in other words, you need 20 local currency units to get what you would get with 1 USD in the US.
# 
# All in all, The purchasing power of a currency refers to the quantity of the currency needed to purchase a given unit of a good, or common basket of goods and services.
# 
# [This text is copied from the kernel of Mhamed Jabri. [link](https://www.kaggle.com/mhajabri/salary-and-purchasing-power-parity/notebook)]
# 
# So let's adjust annual income salary based on PPP. 

# In[ ]:


rates_ppp={'Countries':['United States','India','United Kingdom','Germany','France','Brazil','Canada','Spain','Australia','Russian Federation','Italy',"People 's Republic of China",'Netherlands', 'Sweden', 'Poland', 'Ukraine'],
           'Currency':['USD','INR','GBP','EUR','EUR','BRL','CAD','EUR','AUD','RUB','EUR','CNY','EUR', 'SEK', 'PLN', 'UAH'],
           'PPP':[1.00,17.7,0.7,0.78,0.81,2.05,1.21,0.66,1.46,25.13,0.74,3.51,0.8, 9.125, 1.782, 8.56],
          'exchange_rate': [1,67.56, 0.75, 0.85, 0.85, 3.74, 1.30, 0.85, 1.32, 62.37, 0.85, 6.41, 0.85, 8.72, 3.64, 26.16]}

rates_ppp = pd.DataFrame(data=rates_ppp)
rates_ppp


# We notice that the currency used for each country respondents is most of the time the local currency but not always so we can't directly use the PPP rates.
# What we'll do instead is the following :
# 
# Convert all incomes to USD using Market Exchange Rates 
# Calculature the ratio of PPP rates to MER rates
# Calculate the adjusted salaries using the ratio adjusting factor

# In[ ]:


rates_ppp['PPP/MER']=rates_ppp['PPP']*rates_ppp['exchange_rate']

#keep the PPP/MER rates plus the 'Countries' column that will be used for the merge
rates_ppp


# In[ ]:


pd.set_option('display.float_format', lambda x: '%.3f' % x)

temp = df_survey_results.loc[df_survey_results['Country'].isin(rates_ppp.Countries)]
temp = temp.dropna(subset=['ConvertedSalary'])
temp = temp[(temp['ConvertedSalary']>500) & (temp['ConvertedSalary']<1000000)]
temp = temp.merge(rates_ppp,left_on='Country',right_on='Countries',how='left')[['Country', 'ConvertedSalary','PPP/MER', 'exchange_rate']]


# In[ ]:


temp['AdjustedSalary']=temp['ConvertedSalary']*temp['exchange_rate']/temp['PPP/MER']


d_salary = {}
for country in temp['Country'].value_counts().index :
    d_salary[country]=temp[temp['Country']==country]['AdjustedSalary'].median()
    
median_wages = pd.DataFrame.from_dict(data=d_salary, orient='index').round(2)
median_wages.sort_values(by=list(median_wages),axis=0, ascending=True, inplace=True)
ax = median_wages.plot(kind='barh',figsize=(15,8),width=0.7,align='center')
ax.legend_.remove()
ax.set_title("Adjusted incomes over the world",fontsize=16)
ax.set_xlabel("Amount", fontsize=14)
ax.set_ylabel("Country", fontsize=14)
for tick in ax.get_xticklabels():
    tick.set_rotation(0)
    tick.set_fontsize(10)
plt.tight_layout()
plt.show()


# **Awesome!** Now we see the difference. If you don't see the difference then I will make another plot comparing regular country level annual salary vs adjusted country level annual salary. 

# In[ ]:


f,ax=plt.subplots(1,2,figsize=(18,8))



max_coun.sort_values(by='ConvertedSalary',ascending=True).plot.barh(width=0.8,color=sns.color_palette('RdYlGn'), ax=ax[0])
ax[0].axvline(df_survey_results['ConvertedSalary'].median(),linestyle='dashed')
ax[0].set_title('Compensation of Top 15 Respondent Countries')
ax[0].set_xlabel('')
ax[0].set_ylabel('')


ax[1] = median_wages.plot(kind='barh',figsize=(15,8),width=0.7,align='center', ax=ax[1])
ax[1].legend_.remove()
ax[1].set_title("Adjusted incomes over the world",fontsize=16)
ax[1].set_xlabel("Amount", fontsize=14)
ax[1].set_ylabel("Country", fontsize=14)
for tick in ax[1].get_xticklabels():
    tick.set_rotation(0)
    tick.set_fontsize(10)
plt.tight_layout()
plt.show()


# Now! You are seeing the difference. We see that the United States holds his position. But the United Kingdom jumps to the second position. Russia also changes its position. And many more countries also changes its position. So now we have a more accurate country wise annual income of our developers. Feel free to tell some suggestion. 

# ## Can we predict our the salary of our awesome developers? Let's give it a try?
# 

# In[3]:


df_survey_predict = df_survey_results.copy()
df_survey_predict = df_survey_predict[['FormalEducation', 'YearsCodingProf', 'Age', 'Gender','ConvertedSalary']]
df_survey_predict = df_survey_predict.dropna()
df_survey_predict = df_survey_predict[df_survey_predict.Gender.isin(['Male','Female'])]
df_survey_predict = df_survey_predict[(df_survey_predict['ConvertedSalary']>100) & (df_survey_predict['ConvertedSalary']<1000000)]
df_survey_predict['YearsCodingProf'] = df_survey_predict['YearsCodingProf'].astype(str).str.replace(' years','').str.replace(' or more', '').str.split('-', expand=True).astype(float).mean(axis=1)
df_survey_predict['Age'] = df_survey_predict['Age'].astype(str).str.replace(' years old','').str.replace('Under ', '').str.replace(' years or older', '').str.split('-', expand=True).astype(float).mean(axis=1)
df_survey_predict['FormalEducation'] = df_survey_predict['FormalEducation'].str.replace(r"\(.*\)","")
df_survey_predict['FormalEducation'] = df_survey_predict['FormalEducation'].str.replace('Some college/university study without earning a degree','Some college W.E.D')
df_survey_predict['FormalEducation'] = df_survey_predict['FormalEducation'].str.replace('I never completed any formal education','Never completed')

df_survey_predict.head()


# In[4]:


import statsmodels.api as sm
from statsmodels.formula.api import ols

model = ols("ConvertedSalary ~ Age + Gender + FormalEducation + YearsCodingProf", data=df_survey_predict).fit()
model.summary()


# We can see that Age, Years of professional coding are statistically significant. Also, we can see that doctoral degree and Master's degree also statistically significant. Let's plot some regression plot against Years of Professional coding. 

# In[5]:


fig = plt.figure(figsize=(12,8))
fig = sm.graphics.plot_regress_exog(model, "YearsCodingProf", fig=fig)


# In[66]:


fig = plt.figure(figsize=(12,8))
fig = sm.graphics.plot_regress_exog(model, "Age", fig=fig)


# There is a lot of information. Coming in recent versions. .....

# In[68]:


fig, ax = plt.subplots(figsize=(12,8))
temp = df_survey_predict.groupby('FormalEducation')[['ConvertedSalary', 'YearsCodingProf', 'Age']].mean()
fig = sm.graphics.plot_partregress("ConvertedSalary", "Age", ["YearsCodingProf"],  ax=ax, data=temp)


# We can see from the partial plot that Salary is increasing with Age and Years Of professional coding. 

# # Conclusion
# > We have come a long way. Thank you for reading this kernel. If you find useful then please upvote. We see what our developers did in daily life and their lifestyle and salary. If you have any suggestion please fill free to reach me. Thank You!
