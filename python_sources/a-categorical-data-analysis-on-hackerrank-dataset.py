#!/usr/bin/env python
# coding: utf-8

# # From students to professionals: an analysis over technology perception
# Inspired by the *HackerRank Developer Survey 2018*, this data analysis seeks to answer the posed question: **How are responses from students different from professionals? Is there anything we can learn from their different priorities or preferences?**  
# 
# As a homage to the International Women Day, it will primarily consider only the answer by those who declared female gender *(q3Gender: 2)*, resulting in a number of 4,122 respondents.  
# 
# 
# ## Table of Contents
# * <a href="#sec1">1. Introduction</a>
#   * <a href="#sec1.1"> 1.1. Initial statements </a>
#   * <a href="#sec1.2"> 1.2. Questions of interest </a>
#   * <a href="#sec1.3"> 1.3. Analysis methodology </a>
# 
# * <a href="#sec2">2. Data Wrangling</a>
#   * <a href="#sec2.1"> 2.1. Creating the *class* and *profile* attributes </a>
# 
# * <a href="#sec3">3. Data Analysis</a>
#   * <a href="#sec3.1"> 3.1. From past to present</a>
#   * <a href="#sec3.2"> 3.2. Emerging technologies enrollment</a>
#   * <a href="#sec3.3"> 3.3. Which language do women love?</a>

# <a id='sec1'></a>
# ## 1. Introduction
# <a id='sec1.1'></a>
# ### 1.1. Initial statements
# This section sets up import statements for all the packages that will be used throughout this python notebook.

# In[1]:


# Data analysis packages:
import pandas as pd
import numpy as np

# Visualization packages:
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates
from math import pi

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


## Forcing pandas to display any number of elements
pd.set_option('display.max_columns', None)
pd.options.display.max_seq_items = 2000


# <a id='sec1.2'></a>
# ### 1.2. Question of interest
# The first step in this data analysis was to explore the dataset, understanding the data and seeking for those attributes that could help us finding the answers to the main topic of this analysis. So...

# In[3]:


## Reading the data:
hackerRank_codebook = pd.read_csv('../input/HackerRank-Developer-Survey-2018-Codebook.csv')
hackerRank_numericMapping = pd.read_csv('../input/HackerRank-Developer-Survey-2018-Numeric-Mapping.csv')
hackerRank_numeric = pd.read_csv('../input/HackerRank-Developer-Survey-2018-Numeric.csv')
hackerRank_values = pd.read_csv('../input/HackerRank-Developer-Survey-2018-Values.csv')


# The data exploration started by looking at each dataset, making associations among the attributes and so on. In the way to share how it was done but avoiding the extensive results from these steps, the used commands will be written down just as a code snippet:
# > ** Understanding each survey question:**
# ```
# hackerRank_codebook.head()  
# for ix,item in hackerRank_codebook.iterrows():  
#     print('{0}: {1}\n'.format(item[0], item[1]))
# ```
# 
# > ** Discovering the possible answers:**
# ```
# for ix,item in hackerRank_numericMapping.iterrows():
#     print('{0}: {1} : {2}\n'.format(item[0], item[1], item[2]))
# ```

# From the steps above, the following attributes seem to apply to this analysis:
# * General questions:
#     * At what age did you start coding
#     * How old are you now?
#     * What gender do you identify with?
#     * What is the highest level of education you have (or plan to obtain)?
#     * What is the focus area of your degree?
# 	* How did you learn how to code?
# 
# * Student or professional?
#     * Which of the following best matches your employment level?
#     * Student vs. Non-student
#     * Which one of these best describes your current role?
#     * Which best describes the industry you work in?
# 
# * Job expectations:
#     * What are the top 3 most important things you look for in a company when looking for job opportunities?
# 
# * Technologies engagement:
#     * Which emerging tech skill are you currently learning or looking to learn in the next year?
#     * Which programming language do you love or hate?

# <a id='sec1.3'></a>
# ### 1.3. Analysis methodology
# Considering we are interested in finding the technology perception differences among students and professionals, this analysis should be set up under the following groups:
# 
# **Category** | **Profile** | **Attributes**
# --- | --- | ---
# Student | Under college | *q8JobLevel=1 & q4Education = [1,2]*
# Student | College | *q8JobLevel=1 & q4Education = [3,4,5]*
# Student | Graduate | *q8JobLevel=1 & q4Education = [6,7]*
# Professional | Junior | *q8JobLevel = [2,4]*
# Professional | Senior | *q8JobLevel = [5,6,7]*
# Professional | Freelancer | *q8JobLevel = [3]*
# Professional | Executive | *q8JobLevel = [9,10]*

# --------------------

# <a id='sec2'></a>
# ## 2. Data wrangling
# The goal now is to construct the dataset based on the criteria defined in the previous section, which will be done considering the original categorical dataset.

# In[4]:


## Selecting the female respondents:
dataset = hackerRank_values[hackerRank_values['q3Gender'] == 'Female']


# In[5]:


dataset.info()


# In[6]:


## Attributes of interest:
attributes = ['q1AgeBeginCoding', 'q2Age', 'q3Gender', 'q4Education', 'q0004_other',
       'q5DegreeFocus', 'q0005_other', 'q6LearnCodeUni',
       'q6LearnCodeSelfTaught', 'q6LearnCodeAccelTrain',
       'q6LearnCodeDontKnowHowToYet', 'q6LearnCodeOther', 'q0006_other',
       'q8JobLevel', 'q0008_other', 'q8Student', 'q9CurrentRole',
       'q0009_other', 'q10Industry', 'q0010_other', 'q12JobCritPrefTechStack',
       'q12JobCritCompMission', 'q12JobCritCompCulture',
       'q12JobCritWorkLifeBal', 'q12JobCritCompensation',
       'q12JobCritProximity', 'q12JobCritPerks', 'q12JobCritSmartPeopleTeam',
       'q12JobCritImpactwithProduct', 'q12JobCritInterestProblems',
       'q12JobCritFundingandValuation', 'q12JobCritStability',
       'q12JobCritProfGrowth', 'q0012_other', 'q27EmergingTechSkill',
       'q0027_other', 'q28LoveC', 'q28LoveCPlusPlus', 'q28LoveJava',
       'q28LovePython', 'q28LoveRuby', 'q28LoveJavascript', 'q28LoveCSharp',
       'q28LoveGo', 'q28LoveScala', 'q28LovePerl', 'q28LoveSwift',
       'q28LovePascal', 'q28LoveClojure', 'q28LovePHP', 'q28LoveHaskell',
       'q28LoveLua', 'q28LoveR', 'q28LoveRust', 'q28LoveKotlin',
       'q28LoveTypescript', 'q28LoveErlang', 'q28LoveJulia', 'q28LoveOCaml',
       'q28LoveOther']


# In[7]:


dataset = dataset[attributes]


# In[8]:


## Checking the type for each attribute
dataset.info()


# <a id='sec2.1'></a>
# ### 2.1 Creating the *class* and *profile* attributes
# As defined in Section 1.3, this analysis will be conduced in relation to some technical women profiles, considering their student or professional roles.

# <a id='sec2.1.2'></a>
# #### 2.1.2 Other educational level
# From the considered respondents, anyone has declared to have a different educational level. In this way, the *q0004_other* attribute will be dropped.

# In[9]:


dataset[dataset['q4Education'] == 'Other (please specify)']


# In[10]:


## Checking the 'q4Education' values:
dataset['q4Education'].unique()


# In[11]:


## Dropping the 'q4Education' #NULL values:
ixNull = dataset[dataset['q4Education']=='#NULL!'].index
dataset = dataset.drop(labels=ixNull)


# In[12]:


## Dropping the 'Other education level' column:
dataset = dataset.drop('q0004_other', axis=1)


# <a id='sec2.1.2'></a>
# #### 2.1.2 Other professional roles
# From the considered respondents, 205 declared other employment level, from which 169 are unique roles and must be categorized manually in a further analysis. For now, in this first kernel version, those instances will be dropped down. 

# In[13]:


dataset['q8JobLevel'].unique()


# In[14]:


dataset['q0008_other'].unique()


# In[15]:


## Counting the different employment levels:
q0008_total = dataset['q0008_other'].count()
q0008_unique = len(dataset['q0008_other'].unique())
print('From {0} different employment levels, {1} are unique.'.format(q0008_total, q0008_unique))


# In[16]:


## Dropping down these instances:
q0008_indexes = dataset[dataset['q8JobLevel'] == dataset['q8JobLevel'].unique()[3]]['q0008_other'].index
dataset = dataset.drop(labels=q0008_indexes)
dataset = dataset.drop('q0008_other', axis=1)


# <a id='sec2.1.3'></a>
# #### 2.1.3. Checking student information consistency
# There are two attributes related to student role:
# - *q8JobLevel: 1 : Student*
# - *q8Student: 1 : Students*  
# 
# As confirmed below, both attributes are consistent and carries the same value when it applies.

# In[17]:


indexq8JobLevel = dataset[dataset['q8JobLevel'] == 'Student'].index  #float64 type
indexq8Student = dataset[dataset['q8Student'] == 'Students'].index  #float64 type


# In[18]:


np.unique(indexq8JobLevel == indexq8Student)


# <a id='sec2.1.5'></a>
# #### 2.1.4. Cleaning '#NULL!' values in all attributes
# Since it was later checked there were many attributes with *#NULL!* values, a function will be defined to clean them all.

# In[19]:


def clean_null(dataset):
    for col in dataset.columns:
        if '#NULL!' in dataset[col].unique():
            ixNull = dataset[dataset[col]=='#NULL!'].index
            dataset = dataset.drop(labels=ixNull)
            print('It was cleaned {0} null instances from {1}'.format(len(ixNull), col))
    return dataset


# In[20]:


dataset = clean_null(dataset)


# <a id='sec2.1.5'></a>
# #### 2.1.5. Setting the profile label
# In this data wrangling step, the category and profile information will be added to each dataset instance. It must be taken into account the attribute types for *q8JobLevel=1* and *q4Education* attributes, which are *float64* and *string object*, respectively.
# 
# 
# **Category** | **Profile** | **Attributes**
# --- | --- | ---
# Student | Under college | *q8JobLevel=1 & q4Education = [1,2]*
# Student | College | *q8JobLevel=1 & q4Education = [3,4,5]*
# Student | Graduate | *q8JobLevel=1 & q4Education = [6,7]*
# Professional | Junior | *q8JobLevel = [2,4]*
# Professional | Senior | *q8JobLevel = [5,6,7]*
# Professional | Freelancer | *q8JobLevel = [3]*
# Professional | Executive | *q8JobLevel = [9,10]*

# In[21]:


def map_q8JobLevel(ix):
    temp = hackerRank_numericMapping[(hackerRank_numericMapping['Data Field']=='q8JobLevel')&(hackerRank_numericMapping['Value']==ix)]['Label']
    temp = temp.values
    return temp[0]

def map_q4Education(ix):
    temp = hackerRank_numericMapping[(hackerRank_numericMapping['Data Field']=='q4Education')&(hackerRank_numericMapping['Value']==ix)]['Label']
    temp = temp.values
    return temp[0]


# In[22]:


## Student - Under college:
education = [map_q4Education(1), map_q4Education(2)]
dataset.loc[(dataset['q8JobLevel']=='Student') & (dataset['q4Education'].isin(education)), 'Category'] = 'Student'
dataset.loc[(dataset['q8JobLevel']=='Student') & (dataset['q4Education'].isin(education)), 'Profile'] = 'Under college'


# In[23]:


## Student - College:
education = [map_q4Education(3), map_q4Education(4), map_q4Education(5)]
dataset.loc[(dataset['q8JobLevel']=='Student') & (dataset['q4Education'].isin(education)), 'Category'] = 'Student'
dataset.loc[(dataset['q8JobLevel']=='Student') & (dataset['q4Education'].isin(education)), 'Profile'] = 'College'


# In[24]:


## Student - Graduate:
education = [map_q4Education(6), map_q4Education(7)]
dataset.loc[(dataset['q8JobLevel']=='Student') & (dataset['q4Education'].isin(education)), 'Category'] = 'Student'
dataset.loc[(dataset['q8JobLevel']=='Student') & (dataset['q4Education'].isin(education)), 'Profile'] = 'Graduate'


# In[25]:


## Professional - Junior:
joblevel = [map_q8JobLevel(2), map_q8JobLevel(4)]
dataset.loc[(dataset['q8JobLevel'].isin([2,4])), 'Category'] = 'Professional'
dataset.loc[(dataset['q8JobLevel'].isin([2,4])), 'Profile'] = 'Junior'


# In[26]:


## Professional - Senior:
joblevel = [map_q8JobLevel(5), map_q8JobLevel(6), map_q8JobLevel(7), map_q8JobLevel(8)]
dataset.loc[(dataset['q8JobLevel'].isin(joblevel)), 'Category'] = 'Professional'
dataset.loc[(dataset['q8JobLevel'].isin(joblevel)), 'Profile'] = 'Senior'


# In[27]:


## Professional - Freelancer:
dataset.loc[(dataset['q8JobLevel']==map_q8JobLevel(3)), 'Category'] = 'Professional'
dataset.loc[(dataset['q8JobLevel']==map_q8JobLevel(3)), 'Profile'] = 'Freelancer'


# In[28]:


## Professional - Executive:
joblevel = [map_q8JobLevel(9), map_q8JobLevel(10)]
dataset.loc[(dataset['q8JobLevel'].isin(joblevel)), 'Category'] = 'Professional'
dataset.loc[(dataset['q8JobLevel'].isin(joblevel)), 'Profile'] = 'Executive'


# --------------------

# <a id='sec3'></a>
# ## 3. Data analysis
# Since the dataset was cleaned and the classes and profiles we want to work with are already set, now it is time to run some interesting analysis on these data.

# In[29]:


def df_column_normalize(dataframe, percent=False):
    '''Normalizes the values of a given pandas.Dataframe by the total sum of each column.
    If percent=True, multiplies the final value by 100.
    Algorithm based on https://stackoverflow.com/questions/26537878/pandas-sum-across-columns-and-divide-each-cell-from-that-value'''
    if percent:
        return dataframe.div(dataframe.sum(axis=0), axis=1)*100
    else:
        return dataframe.div(dataframe.sum(axis=0), axis=1)


# <a id='sec3.1'></a>
# ### 3.1 From past to present
# What is the age most of women start to code? In this section we will seek to find not only this answer, but also check if the students nowadays started coding earlier or later than the today professionals. Furthermore, everyone who now works with technology had to be student once.  
# 
# Ideally it would be interesting to plot both information together: the present age compared to when the woman started coding. But since these data are categorized in different scales, they will have to be shown in different charts.

# In[30]:


##Adjusting the data:
analysis01 = dataset[['q1AgeBeginCoding', 'q2Age','Category']]  #Copy the attributes of interest
analysis01['q1AgeBeginCoding'] = analysis01.q1AgeBeginCoding.apply(lambda x: x[:-9])  #Removing 'year old' text
analysis01['q2Age'] = analysis01.q2Age.apply(lambda x: x[:-9])  #Removing 'year old' text


# In[31]:


## Checking the data before plotting:
analysis01.q2Age.unique()


# In[32]:


## Drawing the barplots of ages for each class
fig1, (ax1,ax2) = plt.subplots(2,1,figsize=(12, 8))
fig1.subplots_adjust(top=.93)
plt.suptitle('Women age distribution among students and professionals', fontsize=14, fontweight='bold')

q2Age_order = ['12 - 18 ','18 - 24 ','25 - 34 ','35 - 44 ','45 - 54 ','55 - 64 ','75 years']
sns.countplot(x="q2Age", hue="Category", data=analysis01, ax=ax1, order=q2Age_order)
ax1.set_yticklabels(ax1.get_yticklabels(), ha="right", fontsize=12, weight='bold');
ax1.set_xticklabels(ax1.get_xticklabels(), fontsize=12, weight='bold');
ax1.set_xlabel('Today Age', fontsize=13, weight='bold')

q1AgeBegin_order = ['5 - 10 ','11 - 15 ','16 - 20 ','21 - 25 ','26 - 30 ',
             '31 - 35 ','36 - 40 ','41 - 50 ','50+ years']
sns.countplot(x="q1AgeBeginCoding", hue="Category", data=analysis01, ax=ax2, order=q1AgeBegin_order)
ax2.set_yticklabels(ax2.get_yticklabels(), ha="right", fontsize=12, weight='bold');
ax2.set_xticklabels(ax2.get_xticklabels(), fontsize=12, weight='bold');
ax2.set_xlabel('Age when started coding', fontsize=13, weight='bold')


# From the chart above, we can notice most of women started coding from their 16. It is also clear to notice that most of women working with technology has around 25 to 44 years old and most of the women students is aged between 18 to 24 years.  
# 
# A question that arouse from this is the relationship between these information: **when did the women that are working now start coding?**

# In[33]:


analysis01_relation = analysis01.groupby(['q1AgeBeginCoding','q2Age']).Category.count()
analysis01_relation = analysis01_relation.unstack(1).replace(np.nan,0)
analysis01_relation = df_column_normalize(analysis01_relation, percent=True)


# In[34]:


# Drawing a heatmap with the numeric values in each cell
fig2, ax = plt.subplots(figsize=(9,6))
fig2.subplots_adjust(top=.925)
plt.suptitle('From past to present: when women started coding and their ages', fontsize=14, fontweight='bold')

ax.set_yticklabels(ax.get_yticklabels(), ha="right", fontsize=12, weight='bold')
ax.set_xticklabels(ax.get_xticklabels(), fontsize=12, weight='bold')

cbar_kws = {'orientation':"vertical", 'pad':0.03, 'aspect':50}
sns.heatmap(analysis01_relation, annot=True, linewidths=.3, fmt='.2f', cmap='RdPu', ax=ax, cbar_kws=cbar_kws);

ax.set_ylabel('Age when started coding', fontsize=13, weight='bold')
ax.set_xlabel('Present Age', fontsize=13, weight='bold')


# There are interesting associations to be noticed from the heatmap above:  
# * **40% of the women above 55 years old started coding after their 50s**, demonstrating a fearless disposition to learn new skills. 
# * Since the scale between both variables are not adjusted, we can infer that 80% of the women around 18 years old are just learning to code, which can be supposed to be at universities program.
# * It also can be noticed that 60% of women with more than 55 years old started learning to code on their 16 to 20 years old, which can be associated to senior professionals that started learning code at their studies.

# <a id='sec3.2'></a>
# ### 3.2 Emerging technologies enrollment

# In[35]:


emergTech = dataset[['Category','q27EmergingTechSkill']]


# In[36]:


analysis01 = emergTech.groupby('q27EmergingTechSkill').Category.value_counts()
analysis01 = analysis01.unstack()
analysis01.fillna(value=0, inplace=True)
analysis01 = df_column_normalize(analysis01, percent=True)
analysis01


# In[37]:


# Drawing a heatmap with the numeric values in each cell
fig1, ax = plt.subplots(figsize=(4, 8))
fig1.subplots_adjust(top=.93)
plt.suptitle('Relative enrollment on emerging technologies by female students and professionals', fontsize=14, fontweight='bold')

ax.set_yticklabels(ax.get_yticklabels(), ha="right", fontsize=12, weight='bold')
ax.set_xticklabels(ax.get_xticklabels(), fontsize=12, weight='bold')

cbar_kws = {'orientation':"vertical", 'pad':0.1, 'aspect':50}
sns.heatmap(analysis01, annot=True, linewidths=.3, ax=ax, cmap='RdPu', cbar_kws=cbar_kws);


# From the heatmap above, it can be noticed that 50% of the interviewed women are concerned on machine learning and deep learning technologies, whether they are students or professionals. Furthermore, there is no significant difference on emerging technologies between students and professionals.

# <a id='sec3.3'></a>
# ### 3.3 Which language do women love?

# In[38]:


def df_row_normalize(dataframe):
    '''Normalizes the values of a given pandas.Dataframe by the total sum of each line.
    Algorithm based on https://stackoverflow.com/questions/26537878/pandas-sum-across-columns-and-divide-each-cell-from-that-value'''
    return dataframe.div(dataframe.sum(axis=1), axis=0)


# In[47]:


language = dataset[['q28LoveC', 'q28LoveCPlusPlus',
       'q28LoveJava', 'q28LovePython', 'q28LoveRuby', 'q28LoveJavascript',
       'q28LoveCSharp', 'q28LoveGo', 'q28LoveScala', 'q28LovePerl',
       'q28LoveSwift', 'q28LovePascal', 'q28LoveClojure', 'q28LovePHP',
       'q28LoveHaskell', 'q28LoveLua', 'q28LoveR', 'q28LoveRust',
       'q28LoveKotlin', 'q28LoveTypescript', 'q28LoveErlang', 'q28LoveJulia',
       'q28LoveOCaml', 'q28LoveOther', 'Category']]


# In[48]:


## Replacing all "hate" and "NaN" values by zero (we're interestede just in the languages they love, for while)
lovelanguage = language.replace('Hate',0)
lovelanguage = lovelanguage.replace('Love', 1)

## Replacing all "Love" and "NaN" values by zero (we're now interested just in the languages they hate)
hatelanguage = language.replace('Love',0)
hatelanguage = hatelanguage.replace('Hate', 1)


# In[49]:


lovelanguage = lovelanguage.groupby('Category').sum()
lovelanguage = df_row_normalize(lovelanguage)*100
lovelanguage.reset_index(inplace=True)


# In[50]:


hatelanguage = hatelanguage.groupby('Category').sum()
hatelanguage = df_row_normalize(hatelanguage)*100
hatelanguage.reset_index(inplace=True)


# In[52]:


## Adjusting the columns names:
lovelanguage.columns
lovelanguage.columns = ['group','C', 'C++', 'Java', 'Python','Ruby', 'Javascript', 'C#', 'Go',
       'Scala', 'Perl', 'Swift', 'Pascal','Clojure', 'PHP', 'Haskell', 'Lua','R', 'Rust',
       'Kotlin', 'Typescript','Erlang', 'Julia', 'OCaml']
hatelanguage.columns = lovelanguage.columns


# In[53]:


# From: https://python-graph-gallery.com/391-radar-chart-with-several-individuals/
categories=list(lovelanguage)[1:]
N = len(categories)
# What will be the angle of each axis in the plot? (we divide the plot / number of variable)
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]
# Initialise the spider plot
fig3 = plt.figure(figsize=(8,8))
ax = plt.subplot(111, polar=True)
plt.title('Which programming language do women love the most?', fontsize=14, fontweight='bold')
 
# If you want the first axis to be on top:
ax.set_theta_offset(pi / 2)
ax.set_theta_direction(-1)

# Draw one axe per variable + add labels labels yet
plt.xticks(angles[:-1], categories) 
# Draw ylabels
ax.set_rlabel_position(0)
plt.yticks([5,10,15], ["5%","10%","15%"], color="grey", size=12)
plt.ylim(0,15)

# Plot each individual = each line of the data 
# Ind1
values=lovelanguage.loc[0].drop('group').values.flatten().tolist()
values += values[:1]
ax.plot(angles, values, linewidth=1, linestyle='solid', label="Professional")
ax.fill(angles, values, 'b', alpha=0.1)
# Ind2
values=lovelanguage.loc[1].drop('group').values.flatten().tolist()
values += values[:1]
ax.plot(angles, values, linewidth=1, linestyle='solid', label="Students")
ax.fill(angles, values, 'r', alpha=0.1)
 
# Add legend
plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))


# From the *spyder chart* above, we can see there is a few difference between what student women prefer in relation to the professional ones. The most loved languages are Java, Python, C++, C and Javascript. 

# In[61]:


hatelanguage


# In[60]:


# From: https://python-graph-gallery.com/391-radar-chart-with-several-individuals/
categories=list(hatelanguage)[1:]
N = len(categories)
# What will be the angle of each axis in the plot? (we divide the plot / number of variable)
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]
# Initialise the spider plot
fig3 = plt.figure(figsize=(8,8))
ax = plt.subplot(111, polar=True)
plt.title('Which programming language do women hate the most?', fontsize=14, fontweight='bold')
 
# If you want the first axis to be on top:
ax.set_theta_offset(pi / 2)
ax.set_theta_direction(-1)

# Draw one axe per variable + add labels labels yet
plt.xticks(angles[:-1], categories) 
# Draw ylabels
ax.set_rlabel_position(0)
plt.yticks([3,6,9], ["3%","6%","9%"], color="grey", size=12)
plt.ylim(0,9)

# Plot each individual = each line of the data 
# Ind1
values=hatelanguage.loc[0].drop('group').values.flatten().tolist()
values += values[:1]
ax.plot(angles, values, linewidth=1, linestyle='solid', label="Professional")
ax.fill(angles, values, 'b', alpha=0.1)
# Ind2
values=hatelanguage.loc[1].drop('group').values.flatten().tolist()
values += values[:1]
ax.plot(angles, values, linewidth=1, linestyle='solid', label="Students")
ax.fill(angles, values, 'r', alpha=0.1)
 
# Add legend
plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))


# Now considering the most hated ones, a slightly difference can be found: professional women do hate C++, PHP and C, while just this last is the hated one by the students. In general, they love more than hate :)
