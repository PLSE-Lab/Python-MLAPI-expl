#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np
import os
import pandas as pd
import sys
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from wordcloud import WordCloud,STOPWORDS

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[2]:


plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15
plt.rcParams['font.size'] = 15


# In[3]:


data = pd.read_csv('../input/HackerRank-Developer-Survey-2018-Values.csv')


# In[4]:


data.head()


# In[5]:


data.isnull().sum(axis=0)


# ## Top 10 countries

# In[6]:


top_countries = data['CountryNumeric'].value_counts().head(10)
plt.figure(figsize=(18,8))
plt.tick_params(labelsize=11)
sns.barplot(top_countries.index, top_countries.values)
plt.xlabel("Country")
plt.ylabel("Number of respondents")
plt.show()


# Most respondents belong to **Ghana** followed by **India** and **United States**.

# ## Developers based on Gender

# In[7]:


top_gender = data['q3Gender'].value_counts().head()
plt.figure(figsize=(12,8))
sns.barplot(top_gender.index, top_gender.values)
plt.xlabel("Gender")
plt.ylabel("Number of respondents")
plt.show()


# In[8]:


top_gender.values


# In[9]:


explode = (0.1, 0, 0, 0)  # explode 1st slice
# Plot
plt.figure(figsize=(8,8))
plt.pie(top_gender.values, explode=explode, labels=top_gender.index)
plt.axis('equal')
plt.show()


# ## Developers based on Age

# In[10]:


top_age = data['q2Age'].value_counts().head()
plt.figure(figsize=(12,8))
sns.barplot(top_age.values,top_age.index)
plt.xlabel("Number of respondents")
plt.ylabel("Age")
plt.show()


# Most developers belong to 18-24 years category.

# ## When they started?

# In[11]:


begin_coding_age = data['q1AgeBeginCoding'].value_counts().head()
plt.figure(figsize=(12,8))
sns.barplot(begin_coding_age.values,begin_coding_age.index)
plt.xlabel("Number of respondents")
plt.ylabel("Age")
plt.show()


# Most developers begin coding between 16-20 years of age.

# ## What about their Education?

# In[12]:


highest_education = data['q4Education'].value_counts().head()
plt.figure(figsize=(12,8))
sns.barplot(highest_education.values,highest_education.index)
plt.xlabel("Number of respondents")
plt.ylabel("Education")
plt.show()


# Mostly have some college degree.

# ## What kind of degree is most popular among developers?

# In[13]:


highest_degree = data['q5DegreeFocus'].value_counts().head()
plt.figure(figsize=(6,6))
sns.barplot(highest_degree.values,highest_degree.index)
plt.xlabel("Number of respondents")
plt.ylabel("Degree")
plt.show()


# Computer science degree....as expected.

# ## How many of them learned coding at university or school or bootcamp training or self ?

# In[14]:


learn_code_univ = data['q6LearnCodeUni'].value_counts().head()
learn_code_self = data['q6LearnCodeSelfTaught'].value_counts().head()
learn_code_train = data['q6LearnCodeAccelTrain'].value_counts().head()
learn_code_dontknow = data['q6LearnCodeDontKnowHowToYet'].value_counts().head()
learn_code_other = data['q6LearnCodeOther'].value_counts().head()

temp = pd.DataFrame()
temp['counting'] = [learn_code_univ.values[0], learn_code_self.values[0], learn_code_train.values[0], learn_code_dontknow.values[0], learn_code_other.values[0]]
temp.index = ['School or university','Self-taught (i.e. books, online)','Accelerated training (i.e. bootcamp)', 'I do not know how to code yet', 'Other']

temp = temp.sort_values(['counting'], ascending=[0])

plt.figure(figsize=(6,6))
sns.barplot(temp.counting,temp.index)
plt.xlabel("Number of respondents")
plt.ylabel("Learning Approach")
plt.show()


# *Self-taught* is most popular whereas *bootcamps* are not that popular in order to learn code. It shows that if one has desire and if one practices sufficiently, one can easily learn code.

# ## What are their Job level?

# In[15]:


highest_level = data['q8JobLevel'].value_counts().head()
plt.figure(figsize=(6,6))
sns.barplot(highest_level.values,highest_level.index)
plt.xlabel("Number of respondents")
plt.ylabel("Job level")
plt.show()


# Most participants were students.

# ## Current Industry

# In[16]:


curr_ind = data['q10Industry'].value_counts().head()
plt.figure(figsize=(6,6))
sns.barplot(curr_ind.values,curr_ind.index)
plt.xlabel("Number of respondents")
plt.ylabel("Industry")
plt.show()


# ## Developers' criteria to choose a job

# In[17]:


criteria_1 = data['q12JobCritPrefTechStack'].value_counts().head()
criteria_2 = data['q12JobCritCompMission'].value_counts().head()
criteria_3 = data['q12JobCritCompCulture'].value_counts().head()
criteria_4 = data['q12JobCritWorkLifeBal'].value_counts().head()
criteria_5 = data['q12JobCritCompensation'].value_counts().head()
criteria_6 = data['q12JobCritProximity'].value_counts().head()
criteria_7 = data['q12JobCritPerks'].value_counts().head()
criteria_8 = data['q12JobCritSmartPeopleTeam'].value_counts().head()
criteria_9 = data['q12JobCritImpactwithProduct'].value_counts().head()
criteria_10 = data['q12JobCritInterestProblems'].value_counts().head()
criteria_11 = data['q12JobCritFundingandValuation'].value_counts().head()
criteria_12 = data['q12JobCritStability'].value_counts().head()
criteria_13 = data['q12JobCritProfGrowth'].value_counts().head()
#criteria_14 = data['q12JobCritOther'].value_counts().head()

li = [criteria_1.values[0], criteria_2.values[0], criteria_3.values[0], criteria_4.values[0], criteria_5.values[0],
                   criteria_6.values[0], criteria_7.values[0], criteria_8.values[0], criteria_9.values[0], criteria_10.values[0],
                    criteria_11.values[0], criteria_12.values[0], criteria_13.values[0] #, criteria_14.values[0]
                   ]

temp = pd.DataFrame()
temp['counting'] = li
temp.index = ['Preferred Tech Stack','Company Mission','Company culture', 'Work-life balance', 'Compensation',
             'Proximity', 'Perks', 'Smart people team','Impact with product','Interesting Problems', 'Funding and Valuation',
             'Stability', 'Professional Growth']

temp = temp.sort_values(['counting'], ascending=[0])

plt.figure(figsize=(8,8))
sns.barplot(temp.counting,temp.index)
plt.xlabel("Number of respondents")
plt.ylabel("Job Criteria")
plt.show()


# **Professional Growth** and **Work-life balance** are most popular criteria before choosing a job.

# ## How did employers measure developer's skills?

# In[18]:


criteria_1 = data['q13EmpMeasWhiteboard'].value_counts().head()
criteria_2 = data['q13EmpMeasHackerRank'].value_counts().head()
criteria_3 = data['q13EmpMeasOtherCodingChallenge'].value_counts().head()
criteria_4 = data['q13EmpMeasTechPhoneInt'].value_counts().head()
criteria_5 = data['q13EmpMeasTakeHomeProject'].value_counts().head()
criteria_6 = data['q13EmpMeasResume'].value_counts().head()
criteria_7 = data['q13EmpMeasPastWork'].value_counts().head()
criteria_8 = data['q13EmpMeasOther'].value_counts().head()

li = [criteria_1.values[0], criteria_2.values[0], criteria_3.values[0], criteria_4.values[0], criteria_5.values[0],
                   criteria_6.values[0], criteria_7.values[0], criteria_8.values[0]
                   ]

temp = pd.DataFrame()
temp['counting'] = li
temp.index = ['Whiteboard','HackerRank','Other Coding Challenge', 'Phone interview', 'Take Home Project',
             'Resume', 'Past Work', 'Other']

temp = temp.sort_values(['counting'], ascending=[0])

plt.figure(figsize=(8,8))
sns.barplot(temp.counting,temp.index)
plt.xlabel("Number of respondents")
plt.ylabel("Employer Measures")
plt.show()


# **Resume, phone interview and past works** are top 3 measures of developers skills.

# ## Which core competencies employers look for?

# In[19]:


criteria_1 = data['q21CoreCompProbSolv'].value_counts().head()
criteria_2 = data['q21CoreCompProgLang'].value_counts().head()
criteria_3 = data['q21CoreCompFrameworkProf'].value_counts().head()
criteria_4 = data['q21CoreCompDebugging'].value_counts().head()
criteria_5 = data['q21CoreCompCodebaseNav'].value_counts().head()
criteria_6 = data['q21CoreCompPerfOpt'].value_counts().head()
criteria_7 = data['q21CoreCompCodeReview'].value_counts().head()
criteria_8 = data['q21CoreCompDatabaseDesign'].value_counts().head()
criteria_9 = data['q21CoreCompSysDesign'].value_counts().head()
criteria_10 = data['q21CoreCompTesting'].value_counts().head()

li = [criteria_1.values[1], criteria_2.values[1], criteria_3.values[1], criteria_4.values[1], criteria_5.values[1],
                   criteria_6.values[1], criteria_7.values[1], criteria_8.values[1],criteria_9.values[1],criteria_10.values[1]
                   ]

temp = pd.DataFrame()
temp['counting'] = li
temp.index = ['Problem-solving','Programming language proficiency','Framework Proficiency', 'Debugging', 'Codebase navigation',
             'Performance optimization', 'Code Review capability', 'Database design', 'System design', 'Testing capability']

temp = temp.sort_values(['counting'], ascending=[0])

plt.figure(figsize=(8,8))
sns.barplot(temp.counting,temp.index)
plt.xlabel("Number of respondents")
plt.ylabel("Core Competency")
plt.show()


# **Problem-solving skill** is the most important qualification employers look for. Apart from that **programming language proficiency** and **debugging** skills are also very important.

# ## Which languages do employers look for?

# In[20]:


criteria_1 = data['q22LangProfR'].value_counts().head()
criteria_2 = data['q22LangProfC'].value_counts().head()
criteria_3 = data['q22LangProfCPlusPlus'].value_counts().head()
criteria_4 = data['q22LangProfJava'].value_counts().head()
criteria_5 = data['q22LangProfPython'].value_counts().head()
criteria_6 = data['q22LangProfRuby'].value_counts().head()
criteria_7 = data['q22LangProfJavascript'].value_counts().head()
criteria_8 = data['q22LangProfCSharp'].value_counts().head()
criteria_9 = data['q22LangProfGo'].value_counts().head()
criteria_10 = data['q22LangProfScala'].value_counts().head()
criteria_11 = data['q22LangProfPerl'].value_counts().head()
criteria_12 = data['q22LangProfSwift'].value_counts().head()
criteria_13 = data['q22LangProfPascal'].value_counts().head()
criteria_14 = data['q22LangProfClojure'].value_counts().head()
criteria_15 = data['q22LangProfPHP'].value_counts().head()
criteria_16 = data['q22LangProfHaskell'].value_counts().head()
criteria_17 = data['q22LangProfLua'].value_counts().head()

li = [criteria_1.values[1], criteria_2.values[1], criteria_3.values[1], criteria_4.values[1], criteria_5.values[1],
                   criteria_6.values[1], criteria_7.values[1], criteria_8.values[1],criteria_9.values[1],criteria_10.values[1],criteria_11.values[1], criteria_12.values[1],
      criteria_13.values[1], criteria_14.values[1], criteria_15.values[1],
                   criteria_16.values[1], criteria_17.values[1]
                   ]

temp = pd.DataFrame()
temp['counting'] = li
temp.index = ['R','C','C++', 'Java', 'Python',
             'Ruby', 'Javascript', 'C#', 'Go', 'Scala','Perl','Swift','Pascal','Clojure','PHP','Haskell','Lua']

temp = temp.sort_values(['counting'], ascending=[0])

plt.figure(figsize=(20,5))
sns.barplot(temp.index,temp.counting)
plt.xlabel("Language")
plt.ylabel("Number of respondents")
plt.show()


# **Javascript, Java, Python and C++** are the most popular languages.

# ## Which frameworks do employers look for?

# In[21]:


criteria_1 = data['q23FrameAngularJS'].value_counts().head()
criteria_2 = data['q23FrameReact'].value_counts().head()
criteria_3 = data['q23FrameVueDotJS'].value_counts().head()
criteria_4 = data['q23FrameEmber'].value_counts().head()
criteria_5 = data['q23FrameBackboneDotJS'].value_counts().head()
criteria_6 = data['q23FrameSpring'].value_counts().head()
criteria_7 = data['q23FrameJSF'].value_counts().head()
criteria_8 = data['q23FrameStruts'].value_counts().head()
criteria_9 = data['q23FrameNodeDotJS'].value_counts().head()
criteria_10 = data['q23FrameExpressJS'].value_counts().head()
criteria_11 = data['q23FrameMeteor'].value_counts().head()
criteria_12 = data['q23FrameDjango'].value_counts().head()
criteria_13 = data['q23FramePyramid'].value_counts().head()
criteria_14 = data['q23FrameRubyonRails'].value_counts().head()
criteria_15 = data['q23FramePadrino'].value_counts().head()
criteria_16 = data['q23FrameASP'].value_counts().head()
criteria_17 = data['q23FrameNetCore'].value_counts().head()
criteria_18 = data['q23FrameCocoa'].value_counts().head()
criteria_19 = data['q23FrameReactNative'].value_counts().head()
criteria_20 = data['q23FrameRubyMotion'].value_counts().head()

li = [criteria_1.values[1], criteria_2.values[1], criteria_3.values[1], criteria_4.values[1], criteria_5.values[1],
                   criteria_6.values[1], criteria_7.values[1], criteria_8.values[1],criteria_9.values[1],criteria_10.values[1],criteria_11.values[1], criteria_12.values[1],
      criteria_13.values[1], criteria_14.values[1], criteria_15.values[1],
                   criteria_16.values[1], criteria_17.values[1],criteria_18.values[1],criteria_19.values[1],criteria_20.values[1]
                   ]

temp = pd.DataFrame()
temp['counting'] = li
temp.index = ['AngularJS','React','Vue.js', 'Ember', 'Backbone.js',
             'Spring', 'JSF', 'Struts', 'Node.js', 'ExpressJS','Meteor','Django','Pyramid','Ruby on Rails','Padrino','ASP','.NETCore'
             ,'Cocoa','ReactNative','Ruby Motion'
             ]

temp = temp.sort_values(['counting'], ascending=[0])

plt.figure(figsize=(20,5))
temp_plot = sns.barplot(temp.index,temp.counting)
for item in temp_plot.get_xticklabels():
    item.set_rotation(45)
plt.xlabel("Framework")
plt.ylabel("Number of respondents")
plt.show()


# **Angular**, **Node.js** and **React** are the top 3 frameworks which employers want.

# ## Vim or Emacs?

# In[24]:


editor = data['q24VimorEmacs'].value_counts().head()
#explode = (0.1, 0)  # explode 1st slice
# Plot
plt.figure(figsize=(8,8))
plt.pie(editor.values, labels=editor.index)
plt.axis('equal')
plt.show()


# **Vim** is most popular among developers.

# ## Which languages do developers already know?

# In[70]:


columns = data.columns[data.columns.str.startswith('q25')]
li = []
inx = []
for i in columns:
    if i == 'q25LangOther':
        continue
    li.append(data[data[i] == 'Know'].shape[0])
    x = str(i).replace('q25','')
    x = x.replace('Lang','')
    inx.append(x)

temp = pd.DataFrame()
temp['counting'] = li
temp.index = inx

temp = temp.sort_values(['counting'], ascending=[0])

plt.figure(figsize=(20,5))
temp_plot = sns.barplot(temp.index,temp.counting)
for item in temp_plot.get_xticklabels():
    item.set_rotation(45)
plt.xlabel("Already Known Languages")
plt.ylabel("Number of respondents")
plt.show()


# **Java, C/C++ and JavaScript** are most popular languages among developers.

# ## Which languages are developers willing to learn next?

# In[71]:


columns = data.columns[data.columns.str.startswith('q25')]
li = []
inx = []
for i in columns:
    if i == 'q25LangOther':
        continue
    li.append(data[data[i] == 'Will Learn'].shape[0])
    x = str(i).replace('q25','')
    x = x.replace('Lang','')
    inx.append(x)

temp = pd.DataFrame()
temp['counting'] = li
temp.index = inx

temp = temp.sort_values(['counting'], ascending=[0])

plt.figure(figsize=(20,5))
temp_plot = sns.barplot(temp.index,temp.counting)
for item in temp_plot.get_xticklabels():
    item.set_rotation(45)
plt.xlabel("Language willing to learn")
plt.ylabel("Number of respondents")
plt.show()


# **Go, Python, Ruby and Kotlin** are the next choice of programming language for most of the developers. 

# ## Which frameworks do developers already know?

# In[68]:


columns = data.columns[data.columns.str.startswith('q26')]
li = []
inx = []
for i in columns:
    if (i == 'q26FrameLearnPadrino2') or (i == 'q26FrameLearnDjango2') or (i == 'q26FrameLearnPyramid2'):
        continue
    li.append(data[data[i] == 'Know'].shape[0])
    x = str(i).replace('q26FrameLearn','')
    inx.append(x)

temp = pd.DataFrame()
temp['counting'] = li
temp.index = inx

temp = temp.sort_values(['counting'], ascending=[0])

plt.figure(figsize=(20,5))
temp_plot = sns.barplot(temp.index,temp.counting)
for item in temp_plot.get_xticklabels():
    item.set_rotation(45)
plt.xlabel("Already known Frameworks")
plt.ylabel("Number of respondents")
plt.show()


# **Node.js, Angular.js and Spring** are the top 3 most popular frameworks.

# ## Which frameworks are developers willing to learn next?

# In[67]:


columns = data.columns[data.columns.str.startswith('q26')]
li = []
inx = []
for i in columns:
    if (i == 'q26FrameLearnPadrino2') or (i == 'q26FrameLearnDjango2') or (i == 'q26FrameLearnPyramid2'):
        continue
    li.append(data[data[i] == 'Will Learn'].shape[0])
    x = str(i).replace('q26FrameLearn','')
    inx.append(x)

temp = pd.DataFrame()
temp['counting'] = li
temp.index = inx

temp = temp.sort_values(['counting'], ascending=[0])

plt.figure(figsize=(20,5))
temp_plot = sns.barplot(temp.index,temp.counting)
for item in temp_plot.get_xticklabels():
    item.set_rotation(45)
plt.xlabel("Framework willing to learn")
plt.ylabel("Number of respondents")
plt.show()


# Most developers want to learn **React, AngularJS and Node.js** 

# ## Emerging Technology

# In[103]:


plt.figure(figsize=(12,5))
data['q27EmergingTechSkill'].value_counts().plot.barh()
plt.ylabel("Technology", fontsize=15)
plt.xlabel("Count", fontsize=15)
plt.show()


# **Machine Learning/Deep Learning** is the most popular technology among developers.

# ## Most loved programming language

# In[105]:


columns = data.columns[data.columns.str.startswith('q28')]
li = []
inx = []
for i in columns:
    if (i == 'q28LoveOther'):
        continue
    li.append(data[data[i] == 'Love'].shape[0])
    x = str(i).replace('q28Love','')
    inx.append(x)

temp = pd.DataFrame()
temp['counting'] = li
temp.index = inx

temp = temp.sort_values(['counting'], ascending=[0])

plt.figure(figsize=(20,5))
temp_plot = sns.barplot(temp.index,temp.counting)
for item in temp_plot.get_xticklabels():
    item.set_rotation(45)
plt.xlabel("Language")
plt.ylabel("Number of respondents")
plt.show()


# **Python** is on the top of love-hate index. It won the heart of many developers including me ^_^

#  ## Most hated programming language

# In[106]:


columns = data.columns[data.columns.str.startswith('q28')]
li = []
inx = []
for i in columns:
    if (i == 'q28LoveOther'):
        continue
    li.append(data[data[i] == 'Hate'].shape[0])
    x = str(i).replace('q28Love','')
    inx.append(x)

temp = pd.DataFrame()
temp['counting'] = li
temp.index = inx

temp = temp.sort_values(['counting'], ascending=[0])

plt.figure(figsize=(20,5))
temp_plot = sns.barplot(temp.index,temp.counting)
for item in temp_plot.get_xticklabels():
    item.set_rotation(45)
plt.xlabel("Language")
plt.ylabel("Number of respondents")
plt.show()


# **PHP** and **Java** are on the top of the hate list. May be due to their syntax :P

# ## Most loved Framework

# In[107]:


columns = data.columns[data.columns.str.startswith('q29')]
li = []
inx = []
for i in columns:
    if (i == 'q29FrameLoveOther'):
        continue
    li.append(data[data[i] == 'Love'].shape[0])
    x = str(i).replace('q29FrameLove','')
    inx.append(x)

temp = pd.DataFrame()
temp['counting'] = li
temp.index = inx

temp = temp.sort_values(['counting'], ascending=[0])

plt.figure(figsize=(20,5))
temp_plot = sns.barplot(temp.index,temp.counting)
for item in temp_plot.get_xticklabels():
    item.set_rotation(45)
plt.xlabel("Framework")
plt.ylabel("Number of respondents")
plt.show()


# **AngularJS, NodeJS and React** touched the heart of many developers.

# ## Most hated Frameworks

# In[108]:


columns = data.columns[data.columns.str.startswith('q29')]
li = []
inx = []
for i in columns:
    if (i == 'q29FrameLoveOther'):
        continue
    li.append(data[data[i] == 'Hate'].shape[0])
    x = str(i).replace('q29FrameLove','')
    inx.append(x)

temp = pd.DataFrame()
temp['counting'] = li
temp.index = inx

temp = temp.sort_values(['counting'], ascending=[0])

plt.figure(figsize=(20,5))
temp_plot = sns.barplot(temp.index,temp.counting)
for item in temp_plot.get_xticklabels():
    item.set_rotation(45)
plt.xlabel("Framework")
plt.ylabel("Number of respondents")
plt.show()


# **ASP, Structs and JSF** are top 3 most hated frameworks.

# ## Besides HackerRank, which platforms do developers use to learn how to code?

# In[112]:


criteria_1 = data['q30LearnCodeStackOverflow'].value_counts().head()
criteria_2 = data['q30LearnCodeYoutube'].value_counts().head()
criteria_3 = data['q30LearnCodeMOOC'].value_counts().head()
criteria_4 = data['q30LearnCodeCompCodingSites'].value_counts().head()
criteria_5 = data['q30LearnCodeOnlineTutorial'].value_counts().head()
criteria_6 = data['q30LearnCodeBooks'].value_counts().head()
criteria_7 = data['q30LearnCodeAcademicPaper'].value_counts().head()
criteria_8 = data['q30LearnCodeOther'].value_counts().head()

li = [criteria_1.values[0], criteria_2.values[0], criteria_3.values[0], criteria_4.values[0], criteria_5.values[0],
                   criteria_6.values[0], criteria_7.values[0], criteria_8.values[0]
                   ]

temp = pd.DataFrame()
temp['counting'] = li
temp.index = ['StackOverflow','Youtube','MOOC', 'Competitive coding sites', 'Online tutorials',
             'Books', 'Academic Papers', 'Other']

temp = temp.sort_values(['counting'], ascending=[0])

plt.figure(figsize=(12,8))
sns.barplot(temp.counting,temp.index)
plt.xlabel("Number of respondents")
plt.ylabel("Ways to learn")
plt.show()


# **StackOverflow** is on the top as always. It looks like developers are preferring **Youtube** more than books to learn coding.

# **More to come... Stay tuned!!!**
# **Happy learning ^_^**
