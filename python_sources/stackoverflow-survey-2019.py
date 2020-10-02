#!/usr/bin/env python
# coding: utf-8

# # What you need to know to become a data scientist?
# #### - Data mining from Stack Overflow 2019 Annual Developer Survey.
# 
# * [1. Business understanding](#1)
# * [2. Data understanding](#2)
#    + [2.1 Missing value](#2.1)  
# * [3. Q1: What skill and education background data scientists have?](#3)
#    + [3.1 Data scientist comparing to other developer](#3.1)
#    + [3.2 Skills](#3.2)
#    + [3.3 Relationship of skills](#3.3)
#    + [3.4 Education](#3.4)
# * [4. Q2: What salary does data scientist earn comparing to other developer?](#4)
#    + [4.1 Age](#4.1)   
#    + [4.2 Salary](#4.2)   
#    + [4.3 Predict US Salary](#4.3)   
# * [5. Q3: What features make data scientist distnct from other developer?](#5)
# 
# 

# <a class="anchor" id="1"></a>
# ## 1. Business understanding
# 
# Data scientist is a very popular job since 2010s. A data scientist job openning will attract hundreds of applicants to submit their resume, I am interested what you should have to become a data scientist, also the key terms that you should add into your resume so you can increase the chance to get a data science job. The goal of this project is to find evidences to understand characteristics of a data scientist in industry. The project attempt to give answers to the following three questions:
# 
# 1. What skill and education background data scientist has?
# 2. What salary does data scientist earn comparing to other developer?
# 3. What features make data scientist distnct from other developer?
# 
# The plan is to perform data mining on the data set of 2019 Stack Overflow Developer Survey which can be downloaded from [here](https://insights.stackoverflow.com/survey) 

# <a class="anchor" id="2"></a>
# ## 2. Data understanding
# 
# The 2019 Stack Overflow Annual Developer Survey contains nearly 90000 responses fielded from over 170 countries and dependent territories. The survey examines all aspects of the developer experience from career satisfaction and job search to education and opinions on open source software. In this project, I will explore this survey dataset, specially focused on mining the people who identify themself as a data scientist.
# 

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import networkx as nx
import random
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score,roc_curve,precision_recall_curve,auc,average_precision_score,accuracy_score
get_ipython().run_line_magic('matplotlib', 'inline')

print(os.listdir("../input"))

pd.set_option("max_colwidth",1000000)
pd.set_option('max_columns', 15000)


# In[ ]:


# Read dataset
survey_2019 = pd.read_csv(r'../input/developer_survey_2019/survey_results_public.csv')
schema = pd.read_csv(r'../input/developer_survey_2019/survey_results_schema.csv')


# Take a look the dimension of the dataset

# In[ ]:


survey_2019.shape


# Check the all column types

# In[ ]:


survey_2019.info()


# Just have 5 numeric variables, the other variables are all strings. The variable Respondent can be ignored since it's just an ID for each respondent. Take a look the statistic of these 5 numeric variables

# In[ ]:


survey_2019.describe()


# It looks like there are multiple abnormal values for these numeric, e.g. Age is 99, WorkWeekHrs is 4850 that is impossible. We should clean these outliers if we need to analyze these variables. I will do it later.

# <a class="anchor" id="2.1"></a>
# ### 2.1 Missing values

# Check columns without missing value. Just see 3 columns without missing values.

# In[ ]:


survey_2019.columns[survey_2019.isnull().mean()==0]


# Check the columns that have highest percentage of missing values. The BlockchainOrg has most missing values, but still less than 50%.

# In[ ]:


survey_2019.isnull().mean().sort_values(ascending=False)


# I will impute the missing values in the later steps.

# <a class="anchor" id="3"></a>
# ## 3. Q1: What skill and education background data scientists have?
# 

# <a class="anchor" id="3.1"></a>
# ### 3.1 Data scientist comparing to other developer
# 
# #### 3.1.1 Data preparation
# 
# In order to subset the data scientist people. Let's look at the DevType column which describes the role of each developer. The majority of people are "full-stack developer", they were allowed to select multiple roles so that I can anticipate data scientist can be a single role or a mixed data scientist who also know other skills.

# In[ ]:


# List all possible role
survey_2019['DevType'].value_counts(dropna=False)


# I am curious that some roles might be similar to each other, for instance, full-stack developer usually means a developer can do both front-end and back-end, data scientist usually works with data analyst and data engineer. Let's transform DevType into multiple columns of which each column represent an individual roles.

# In[ ]:


# Check any missing DevType
np.sum(survey_2019['DevType'].isnull())


# In[ ]:


# The transpose data set should have 88883-7548=81335 observations.
df = pd.get_dummies(survey_2019['DevType'].str.split(';', expand=True)
   .stack()
   ).sum(level=0)
df.shape


# In[ ]:


df.head()


# The transformed data set looks good. Now we can generate a graph indicating the correlation among different roles. i use seaborn package to plot this correlation. 

# #### 3.1.2 Descriptive plot

# In[ ]:


fig, ax = plt.subplots()
fig.set_size_inches(10, 10)
plt.title("Correlation of Developer")
ax = sns.heatmap(df.corr(),vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);


# The graph shows data scientist or machine learning specialist is highly correlated with data or business analyst, data engineer and I missed one, the researcher and scientist also uses data science to do research. The interesting thing is data scientist has negative correlation with front-end and full-stack developer. That means, data scientist would rather work as back-end, but would not play a role as front-end? Let's see how many samples support this view.

# In[ ]:


# Counter the number of developer who is a data scientist and a front-end or full-stack developer
num1 = survey_2019[survey_2019['DevType'].str.contains("Data scientist",na = False)
           & (survey_2019['DevType'].str.contains("front-end",na = False)
           | survey_2019['DevType'].str.contains("full-stack",na = False))].shape[0]
# Counter the number of developer who is a data scientist
num2 = survey_2019[survey_2019['DevType'].str.contains("Data scientist",na = False)].shape[0]
print(num1)
print(num2)
print(num1/num2)


# In[ ]:


# Count the number of developer who is a front-end or full-stack developer
num3 = survey_2019[survey_2019['DevType'].str.contains("front-end",na = False)
           | survey_2019['DevType'].str.contains("full-stack",na = False)].shape[0]
print(num3)
print(num2/num3)


# Now we see that only 42% of the data scientist is also working as front-end or full-stack developer, while only 12% of front-end or full-starck developers are working as data scientist. The negative correlation make senses. 

# <a class="anchor" id="3.2"></a>
# ### 3.2 Skills

# #### 3.2.1  Data preparation
# 
# To compare the different skills between data scientist and non data scientist, I would like to divide the developer into 3 groups:
# 
# 1. non data scientist - who does not identify himself as a data scientist
# 2. mixed data scientist - who is a data scientist as well as other role
# 3. pure data scientist - who has only one data scientist role
# 
# I assume these 3 groups of people have diverse skill distribution, then we know what skill make data scientist different. 

# In[ ]:


# Divide the data set into 3 subset and check the number of record for each group
non_data_scientist = survey_2019[~survey_2019['DevType'].str.contains("Data scientist",na = False)]
mixed_data_scientist = survey_2019[survey_2019['DevType'].str.contains("Data scientist",na = False)
                                  & (survey_2019['DevType']!='Data scientist or machine learning specialist')]
pure_data_scientist = survey_2019[survey_2019['DevType']=='Data scientist or machine learning specialist']
print(non_data_scientist.shape[0])
print(mixed_data_scientist.shape[0])
print(pure_data_scientist.shape[0])


# In[ ]:


# Define a function to show the top feature for each group.
def total_count(df, col1):
    '''
    INPUT:
    df - the pandas dataframe you want to search
    col1 - the column name you want to look through
    
    OUTPUT:
    new_df - a dataframe shows the percentage account for the total observation. 
    '''
    new_df = df[col1].str.split(';', expand=True).stack().value_counts(dropna=False).reset_index()
    new_df.rename(columns={0: 'count'}, inplace=True)
    new_df['percentage'] = new_df['count']/np.sum(df[col1].notnull())
    return new_df
    


# #### 3.2.2 Descriptive charts and plots

# First, we check the LanguageWorkedWith, see what programming language the developer is using.

# In[ ]:


total_count(non_data_scientist,'LanguageWorkedWith')


# The top three languages are JavaScript, HTML and SQL among non data scientist, that makes sense due to most people taking part in this survey who are web developers. To understand the programming language for the data scientist group, I'd like to make graphs to compare the diversity. Let's make a barplot to display the result. Here I define the barplot function across 3 group:

# In[ ]:


def barplot_group(col1,width=20,height=8):
    '''
    INPUT
    col1 - column name you want to analyze
    width - width of the graph
    height -height of the graph
    
    OUTPUT
    output the a barplot graph showing percentage of column accounting for the total by each group
    '''
    
    df1 = total_count(non_data_scientist,col1)
    df2 = total_count(mixed_data_scientist,col1)
    df3 = total_count(pure_data_scientist,col1)
    df1['role'] = 'non ds'
    df2['role'] = 'mixed ds'
    df3['role'] = 'pure ds'
    df = pd.concat([df1,df2,df3])
 
    fig, ax = plt.subplots()
    fig.set_size_inches(width, height)
    ax = sns.barplot(x="index", y="percentage", hue="role", data=df)
    plt.legend(loc=1, prop={'size': 20})


# Again, we show the LanguageWorkedWith across 3 group.

# In[ ]:


barplot_group('LanguageWorkedWith',30,8)
plt.title("Language worked with")


# The top 3 languages for mixed data scientist are Python, SQL and JavaScript. While more than 80% of pure data scientist use Python, SQL and R are also popular. This is so true that if you search "Python SQL R" in a job board website, it will return data scientist jobs. Beyond the programming language, we take a look DatabaseWorkedWith, PlatformWorkedWith, WebFrameWorkedWith, MiscTechWorkedWith, DevEnviron, OpSys. These developer tools might differ across 3 group. 

# In[ ]:


barplot_group('DatabaseWorkedWith')


# Non data scientists use MySQL a lot while pure data scientist use PostgreSQL, I think the reason is they have a different working platform. Let's check the platform.

# In[ ]:


barplot_group('PlatformWorkedWith')


# Linux is the top working platform for every developer, Windows also is a important working platform traditionally. There are recently new platforms such as AWS and Docker.

# In[ ]:


barplot_group('WebFrameWorkedWith')


# Non data scientists use JQuery while pure data scientist prefers Flask, which is a Python based web framework. It looks like WebFrameWorkedWith is somewhat correlated with LanguageWorkedWith, as well as they are correlated with MiscTechWorkedWith:

# In[ ]:


barplot_group('MiscTechWorkedWith')


# Data scientists use Pandas, Tensorflow, a machine learning library, also Spark, Hadoop and PyTorch. All these tools are based on Python or with a Python access API. No wonder more than 80% of pure data scientists are using Python! I expect that DevEnviron is also correlated.

# In[ ]:


barplot_group('DevEnviron',30,8)


# Top 3 developing editors for pure data scientist are Jupyter, PyCharm and RStudio, which is corresponding to what we've found, they use Python and R.

# In[ ]:


barplot_group('OpSys')


# Let's see the the survey question of OpSys: What is the primary operating system in which you work? It indicates the system you are commonly working with including sending emails with Outlook and using MS Office production. Therefore, it's no doubt the result returns Windows as the top operation system. Data scientists prefer Linux then MacOS as we've known they work on Linux mostly. 
# 
# 

# <a class="anchor" id="3.3"></a>
# ### 3.3 Relationship of skills
# 
# I am interested if a network graph is able to explain the correlation among these technologies and tools. I want to gather all these tools and use networkx package to generate a technology network for data scientist. The first step, I need to create a co-occurence matrix for these technology variables.

# #### 3.3.1 Data preparation

# In[ ]:


# Explore the pure data scientist group
# data_scientist = survey_2019[survey_2019['DevType'].str.contains("Data scientist",na = False)]

# Gather the technology variables
temp = pure_data_scientist[['LanguageWorkedWith'
                                ,'DatabaseWorkedWith'
                                ,'PlatformWorkedWith'
                                ,'WebFrameWorkedWith'
                                ,'MiscTechWorkedWith'
                                ,'DevEnviron'
                                ]]

# Create a tech combining all technologies into one variable.
temp['tech'] = temp['LanguageWorkedWith'].map(str)+";"+temp['DatabaseWorkedWith'].map(str)+";"+temp['PlatformWorkedWith'].map(str)+";"+temp['WebFrameWorkedWith'].map(str)+";"+temp['MiscTechWorkedWith'].map(str)+";"+temp['DevEnviron'].map(str)

# Transpose tech to build a one hot matrix 
df = pd.get_dummies(temp['tech'].str.split(';', expand=True)
   .stack()
   ).sum(level=0)

# drop the nan column
df = df.drop(columns=['nan'])

# Convert the value to integer.
df_asint = df.astype(int)

# Create co-occurrence matrix
coocc = df_asint.T.dot(df_asint)
coocc


# In[ ]:


# networkx time              
# create edges with weight, and a note list
edge_list = []
node_list = []
for index, row in coocc.iterrows():
    i = 0
    for col in row:
        weight = float(col)/df.shape[0]
        
        if weight >=0.2:    # ignore weak weight.
            
            if index != coocc.columns[i]:
                edge_list.append((index, coocc.columns[i], weight))
            
            #create a note list
            if index == coocc.columns[i]:
                node_list.append((index, weight))
        i += 1


# In[ ]:


# networkx graph
G = nx.Graph()
for i in sorted(node_list):
    G.add_node(i[0], size = i[1])
G.add_weighted_edges_from(edge_list)

# create a list for edges width.
test = nx.get_edge_attributes(G, 'weight')
edge_width = []
for i in nx.edges(G):
    for x in iter(test.keys()):
        if i[0] == x[0] and i[1] == x[1]:
            edge_width.append(test[x])


# #### 3.3.2 Descriptive plot

# In[ ]:


plt.subplots(figsize=(14,14))
node_scalar = 5000
width_scalar = 10
sizes = [x[1]*node_scalar for x in node_list]
widths = [x*width_scalar for x in edge_width]

#draw the graph
pos = nx.spring_layout(G, k=0.4, iterations=15,seed=1234)

nx.draw(G, pos, with_labels=True, font_size = 8, font_weight = 'bold', 
        node_size = sizes, width = widths,alpha=0.6,edge_color="green")
plt.title("Data Science Tool Relationship Map")


# The bigger note indicates the larger percentage of pure data scientists who use the tool. It's obvious Python is a core skill for data scientist, it connects to almost all data science tool. It can be concluded that data scientist must know Python as well as learn Python if you want to break into the field.

# <a class="anchor" id="3.3"></a>
# ### 3.3 Education

# Next we move to analyze the education background of data scientist. First, we look into EdLevel, which shows the highest education level of developers.

# In[ ]:


barplot_group('EdLevel',50,8)


#  It looks like the data scientists tend to have a higher education degree than non data scientists. The majority of pure data scientists has a master degree, and it seems Ph.Ds are more common in data science. I am interested in what field the data scientist study. Let's look into UndergradMajor.

# In[ ]:


barplot_group('UndergradMajor',50,10)


# No surprise that most data scientists earn a computer science or engineer degree. There is a large portion of pure data scientist earning a mathematics and statistics degree. I am curious about the people who has social science background. Let's see the data.

# In[ ]:


survey_2019['UndergradMajor'].value_counts(dropna=False)


# In[ ]:


# Check the number of data scientist with social science background.
df = survey_2019[survey_2019['DevType'].str.contains("Data scientist",na = False)
                                  & (survey_2019['UndergradMajor']=='A social science (ex. anthropology, psychology, political science)')]
df.shape[0]


# In[ ]:


# What country do they reside in?
df['Country'].value_counts(dropna=False)

It seems more than half of social data scientist are from United States. I am curious the portion of US data scientists grouping by different education field.
# In[ ]:


# The percentage of data scientists who reside in US. 
data_scientist = pd.concat([pure_data_scientist,mixed_data_scientist])
print(data_scientist[data_scientist['Country']=='United States'].shape[0]/data_scientist.shape[0])


# In[ ]:


# Distribution of undergraduate major of data scientist
total_count(data_scientist,'UndergradMajor')


# In[ ]:


total_count(data_scientist,'EduOther')


# In[ ]:


total_count(data_scientist,'EdLevel')


# In[ ]:


# The percentage of data scientists who reside in US grouped by undergraduate major.
df = data_scientist['UndergradMajor'].value_counts(dropna=False).sort_index().reset_index()
df1 = data_scientist[data_scientist['Country']=='United States']['UndergradMajor'].value_counts(dropna=False).sort_index().reset_index()
df2 = pd.merge(df,df1, on='index')
df2['PCT of US'] = df2['UndergradMajor_y']/df2['UndergradMajor_x']
df2


# Comparing to other countries, the US accounts for a majority of data scientist who has an undergraduate degree of social science, art or humanities discipline. I wonder if data science recently grows on these fields in the US. Next, take a look the EduOther which shows other education like online course.

# In[ ]:


survey_2019['EduOther'].value_counts(dropna=False)


# In[ ]:


total_count(non_data_scientist,'EduOther')


# In[ ]:


barplot_group('EduOther',50,10)


# That seems not much difference between data scientist and non data scientist. Taught yourself a new language and taking online course are the most suggested education method beyond earning a degree.

# <a class="anchor" id="4"></a>
# ## 4. Q2: What salary does data scientist earn comparing to other developer?
# 
# 

# <a class="anchor" id="4.1"></a>
# ### 4.1 Age

# First check the distribution of age among 3 groups. Can we distinguish data scientist with their age?

# In[ ]:


df1 = non_data_scientist[non_data_scientist['Age'].notnull()]
df2 = mixed_data_scientist[mixed_data_scientist['Age'].notnull()]
df3 = pure_data_scientist[pure_data_scientist['Age'].notnull()]

# density plot of age among 3 groups
fig, ax = plt.subplots()
fig.set_size_inches(8, 8)
sns.distplot(df1[['Age']], hist=False,color='blue',norm_hist=True)
sns.distplot(df2[['Age']], hist=False,color='red',norm_hist=True)
sns.distplot(df3[['Age']], hist=False,color='green',norm_hist=True)


# Most developers are young between 20 and 40. A few developers is joking almost 100 years old. It looks like a larger portion of younger population for the data scientist.

# In[ ]:


fig, ax = plt.subplots()
fig.set_size_inches(8, 8)

# density plot of age of pure data scientist between residing in US and non-US.
sns.distplot(df3[df3['Country']=='United States'][['Age']], hist=False,color='green',norm_hist=True)
sns.distplot(df3[df3['Country']!='United States'][['Age']], hist=False,color='red',norm_hist=True)


# <a class="anchor" id="4.2"></a>
# ### 4.2 Salary

# There are two variables about the salary, ConvertedComp and CompTotal. ConvertedComp is USD converted from the CompTotal which is amount of local currency. There is a CurrencySymbol that indicates the type of currency.  

# In[ ]:


survey_2019['CurrencySymbol'].value_counts(dropna=False)


# In[ ]:


# Salaries among 3 groups 


df1 = non_data_scientist[non_data_scientist['ConvertedComp'].notnull()]
df2 = mixed_data_scientist[mixed_data_scientist['ConvertedComp'].notnull()]
df3 = pure_data_scientist[pure_data_scientist['ConvertedComp'].notnull()]
fig, ax = plt.subplots()
fig.set_size_inches(8, 8)
sns.distplot(df1[['ConvertedComp']], hist=False,color='blue',norm_hist=True)
sns.distplot(df2[['ConvertedComp']], hist=False,color='red',norm_hist=True)
sns.distplot(df3[['ConvertedComp']], hist=False,color='green',norm_hist=True)


# It looks like data scientist has a higher salary than other developer.

# In[ ]:


# Salary vs Age
df = survey_2019[survey_2019['DevType'].str.contains("Data scientist",na = False)]
df['US'] = df['Country'].apply(lambda x: 1 if x=='United States' else 0)


fig, ax = plt.subplots()
fig.set_size_inches(10, 10)
ax = sns.scatterplot(x="ConvertedComp", y="Age", data=df, hue='US',alpha=0.6)


# Data scientists in the US have higher salaries than rest of the countries. We also see some abnormal number of population of salary at 1 million and 2 million.

# In[ ]:


# Zoom in the compensations less than 500000 USD.
# Just ignore the big salaries, focus on common and reasonable salary.
df1 = df[df["ConvertedComp"]<=500000]

fig, ax = plt.subplots()
fig.set_size_inches(10, 10)
ax = sns.scatterplot(x="ConvertedComp", y="Age", data=df1, hue='US',alpha=0.8)


# In[ ]:


# Filter the data scientist in the US.

df2 = pd.concat([pure_data_scientist,mixed_data_scientist])
df2 = df2[(df2["ConvertedComp"]<=500000) & (df2["Country"]=='United States')]

fig, ax = plt.subplots()
fig.set_size_inches(10, 10)
ax = sns.scatterplot(x="ConvertedComp", y="Age", data=df2, color="#f28e2b")
ax = sns.regplot(x="ConvertedComp", y="Age", data=df2,color="#f28e2b")


# In[ ]:


# Just curious whether US data scientist has any dependent.
df = survey_2019[survey_2019['DevType'].str.contains("Data scientist",na = False)]
df = df[df["Country"]=='United States']

fig, ax = plt.subplots()
fig.set_size_inches(10, 10)
ax = sns.boxplot(x="Dependents",y="Age" , data=df)


# We can check Employment status against salary, perhaps they are correlated.

# In[ ]:


# Employment vs salary

fig, ax = plt.subplots()
fig.set_size_inches(20, 8)
ax = sns.boxplot(x="Employment",y="CompTotal" , data=df2)


# Also, look into OrgSize, the company size may affect their wage payment.

# In[ ]:


# OrgSize vs salary

fig, ax = plt.subplots()
fig.set_size_inches(20, 8)
ax = sns.boxplot(x="OrgSize",y="CompTotal" , data=df2)


# We can see that the organization with 10000 or more employees tends to be generous, willing to pay more than others. 

# In[ ]:


# df = survey_2019[survey_2019['DevType'].str.contains("Data scientist",na = False)]
# df = df[(df["Country"]=='United States')]
# df['CompTotal1'] = df.apply(lambda row : min(row['CompTotal'],row['ConvertedComp']),axis=1)

# fig, ax = plt.subplots()
# fig.set_size_inches(20, 20)
# ax = sns.scatterplot(x='CompTotal1', y="Age", data=df)


# In[ ]:


# df = survey_2019[survey_2019['DevType'].str.contains("Data scientist",na = False)]
# df = df[(df["Country"]=='United States')]
# df['CompTotal1'] = df.apply(lambda row : min(row['CompTotal'],row['ConvertedComp']),axis=1)
# df = df[df['CompTotal1'] <600000] 

# fig, ax = plt.subplots()
# fig.set_size_inches(20, 20)
# ax = sns.scatterplot(x='CompTotal1', y="Age", data=df)


# <a class="anchor" id="4.3"></a>
# ### 4.3 Predict US Salary

# The question is: What feature affect a data scientist salary? Skill, age, or company size? We already know that who resides in the US usually earns more than other countries. So we focus on analyzing salary within the US. 

# #### 4.3.1 Data understanding and preparation

# In[ ]:


# get the data first
df = survey_2019[survey_2019['DevType'].str.contains("Data scientist",na = False)]
df = df[df["Country"]=='United States']
df.shape


# In[ ]:


# Check the statistics of numeric variables, might remove some outliers.
df.describe()


# In[ ]:


# Check CompTotal vs ConvertedComp, see which salary makes sense.
fig, ax = plt.subplots()
fig.set_size_inches(10, 10)
ax = sns.scatterplot(x='CompTotal', y="ConvertedComp", data=df)


# In[ ]:


# if we set CompTotal<700000, will filter out most abnormal salaries.
fig, ax = plt.subplots()
fig.set_size_inches(10, 10)
ax = sns.scatterplot(x='CompTotal', y="ConvertedComp", data=df[df["CompTotal"]<700000])


# Theotically they should be aligned, but not all. In the original survey form, there is only a box for compensation. It's possible ConvertedComp is a calculated variable. So I just consider CompTotal and drop ConvertedComp.
# 

# In[ ]:


df['Employment'].value_counts(dropna=False)


# In[ ]:


# Remove records with missing CompTotal and outliers;
df = df[(df['CompTotal'].notnull()) & (df['CompTotal']<700000) & (df['CompTotal']>0)]

# Exclude the unemployed.
df = df[df['Employment']!='Not employed, but looking for work']

# Drop ConvertedComp, Respondent, Country, DevType, CurrencySymbol,CurrencyDesc
df = df.drop(['ConvertedComp', 'Respondent', 'Country', 'DevType', 'CurrencySymbol','CurrencyDesc'],axis=1)
df.shape


# In[ ]:


# Retrieve the categorical variables
cat_vars_int = df.select_dtypes(include=['object']).copy().columns
len(cat_vars_int)


# In[ ]:


# Split and transpose the categorical variable into one-hot columns.
for var in  cat_vars_int:
    # for each cat add dummy var, drop original column
    df = pd.concat([df.drop(var, axis=1), df[var].str.get_dummies(sep=';').rename(lambda x: var+'_' + x, axis='columns')], axis=1)

df.describe()


# In[ ]:


df.isnull().mean().sort_values(ascending=False)


# In[ ]:


# Impute missing values with column median
# I choose median because the distributions are not normal
# No needed to impute the missing value for the one-hot columns
df['CodeRevHrs'] = df['CodeRevHrs'].fillna(df['CodeRevHrs'].median())
df['Age'] = df['Age'].fillna(df['Age'].median())
df['WorkWeekHrs'] = df['WorkWeekHrs'].fillna(df['WorkWeekHrs'].median())


# In[ ]:


# An option reducing features to prevent overfitting?
# df = df.iloc[:, np.where((X.sum() > 10) == True)[0]]
# df.shape


# In[ ]:


# Plot the distribution of numeric variables, see if any skewed distribution that needs to be normailized. 

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10,10))
plt.tight_layout(w_pad=2.0, h_pad=5.0)

plt.subplot(2,2,1)
plt.xlabel('CompTotal')
p1 = sns.distplot(df[['CompTotal']], hist=False,color='red',norm_hist=True)

plt.subplot(2,2,2)
plt.xlabel('WorkWeekHrs')
p2 = sns.distplot(df[['WorkWeekHrs']], hist=False,color='red',norm_hist=True)

plt.subplot(2,2,3)
plt.xlabel('CodeRevHrs')
p3 = sns.distplot(df[['CodeRevHrs']], hist=False,color='red',norm_hist=True)

plt.subplot(2,2,4)
plt.xlabel('Age')
p4 = sns.distplot(df[['Age']], hist=False,color='red',norm_hist=True)


# In[ ]:


# Use logarithm function to transform the numeric columns
# df['CompTotal_log'] = np.log(df['CompTotal']+100000)
df['WorkWeekHrs_log'] =np.log(df['WorkWeekHrs']+100)
df['CodeRevHrs_log'] = np.log(df['CodeRevHrs']-30)

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10,10))
plt.tight_layout(w_pad=2.0, h_pad=5.0)

# plt.subplot(2,2,1)
# plt.xlabel('CompTotal_log')
# p1 = sns.distplot(df[['CompTotal_log']], hist=False,color='red',norm_hist=True)

plt.subplot(2,2,2)
plt.xlabel('WorkWeekHrs_log')
p2 = sns.distplot(df[['WorkWeekHrs_log']], hist=False,color='red',norm_hist=True)

plt.subplot(2,2,3)
plt.xlabel('CodeRevHrs_log')
p3 = sns.distplot(df[['CodeRevHrs_log']], hist=False,color='red',norm_hist=True)

plt.subplot(2,2,4)
plt.xlabel('Age')
p4 = sns.distplot(df[['Age']], hist=False,color='red',norm_hist=True)


# In[ ]:


# Drop the original numeric columns
df = df.drop(['WorkWeekHrs','CodeRevHrs'],axis=1)


# #### 4.3.2 Modeling

# In[ ]:


# Split the data into train and test
y = df['CompTotal'].values
X = df.drop(['CompTotal'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=0)


# Now we use Xgboost to predict the salary of data scientist. Xgboost is one of the best tree learning algorithm by which most people have won a Kaggle competition.

# In[ ]:


# Xgboost modelling
dtrain = xgb.DMatrix(X_train, label=y_train)
dvalid = xgb.DMatrix(X_test, label=y_test)
watchlist = [(dtrain, 'train'), (dvalid, 'valid')]


# In[ ]:


# Set the parameters
# Set the regularization lambda to 100000
# Set the evalutaion metric as rmse (root mean square error)
# Set the early stopping rounds to 5

evals_result = {}


xgb_pars = {'min_child_weight': 5, 'eta':0.5, 'colsample_bytree': 0.8, 
            'max_depth': 10,
'subsample': 0.8, 'lambda': 100000, 'nthread': -1, 'booster' : 'gbtree', 'silent': 1,
'eval_metric': 'rmse', 'objective': 'reg:linear','seed':1234}

# xgb_pars = {'lambda': 100000, 'booster' : 'gbtree', 
# 'eval_metric': 'rmse', 'objective': 'reg:linear','seed':1234}

model = xgb.train(xgb_pars, dtrain, 10000000, watchlist, early_stopping_rounds=5,
      maximize=False, verbose_eval=1000, evals_result=evals_result)
print('Modeling RMSE %.5f' % model.best_score)


# #### 4.3.3 Evaluation

# In[ ]:


# Model evaluation graph
plt.plot(evals_result['train']['rmse'], linewidth=2, label='Train')
plt.plot(evals_result['valid']['rmse'], linewidth=2, label='Test')
plt.legend(loc='upper right')
plt.title('Model RMSE')
plt.ylabel('rmse')
plt.xlabel('Epoch')
plt.show()


# In[ ]:


# confusion matrix
dtest = xgb.DMatrix(X_test)
y_pred = model.predict(dtest)

fig, ax = plt.subplots()
fig.set_size_inches(8, 8)

ax = sns.scatterplot(x=y_pred,y=y_test)
plt.title('Residual plot');
plt.xlabel('predicted');
plt.ylabel('actual'); 


# In[ ]:


# histogram of residual
sns.distplot(y_pred-y_test)


# We see there is an overfitting issue for the model. The residual is close to normal distribution, we can't say the model is bias. The next job maybe is tuning the model to find better parameters?

# In[ ]:


# Display the feature importance.
fig, ax = plt.subplots()
fig.set_size_inches(10, 10)
xgb.plot_importance(model, max_num_features=28, ax=ax)


# We can see Age and Working hour are important features, data scientist who has more years of working experience and working hours can earn more, which likely make senses. Other features seem less relevant. This perhaps is the fact of the data scientist market, working harder and gaining experience, you would earn more, so does it mean that the boss does not care about their skills?

# In[ ]:


# Plot WorkWeekHrs vs CompTotal
# Check the tendency
fig, ax = plt.subplots()
fig.set_size_inches(10, 10)

df['WorkWeekHrs'] = np.exp(df['WorkWeekHrs_log'])-100
# ax = sns.boxplot(y='CompTotal', x="MgrMoney_Not sure", data=df[df["CompTotal"]<10000000])
# ax = sns.boxplot(y='CompTotal', x="OrgSize_10,000 or more employees", data=df[df["CompTotal"]<10000000])
ax = sns.scatterplot( x="WorkWeekHrs",y='CompTotal', data=df[df["CompTotal"]<10000000])
ax = sns.regplot(x="WorkWeekHrs", y="CompTotal", data=df[df["CompTotal"]<10000000])


# In[ ]:


# model = xgb.XGBRegressor(colsample_bytree=0.4,
#                  gamma=0,                 
#                  learning_rate=0.07,
#                  max_depth=3,
#                  min_child_weight=1.5,
#                  n_estimators=10000,                                                                    
#                  reg_alpha=0.75,
#                  reg_lambda=0.45,
#                  subsample=0.6,
#                  seed=42,
#                  verbose=10) 


# In[ ]:


# model.fit(X_train,y_train)


# In[ ]:


# predictions = model.predict(X_test)
# # print(explained_variance_score(predictions,y_test))
# from sklearn.metrics import mean_absolute_error
# print("Mean Absolute Error : " + str(mean_absolute_error(predictions, y_test)))


# <a class="anchor" id="5"></a>
# ## 5. Q3: What features make data scientist distnct from other developer?

# The basic idea to answer this question is to build a classifier model that distinguishes data scientist and non data scentist, then we can dig out the distinct feature by exploring the feature importance of the model.

# ### 5.1 Data understanding and preparation

# In[ ]:


# Retrieve the character variables.
# df = survey_2019[survey_2019['Country']=='United States']
df = survey_2019
cat_vars_int = survey_2019.select_dtypes(include=['object']).copy().columns
len(cat_vars_int)
df.shape


# In[ ]:


# Again split and transpose the categorical variable into one-hot columns.
for var in  cat_vars_int:
    # for each cat add dummy var, drop original column
    df = pd.concat([df.drop(var, axis=1), df[var].str.get_dummies(sep=';').rename(lambda x: var+'_' + x, axis='columns')], axis=1)

df.describe()


# In[ ]:


df.shape


# In[ ]:


# Drop Respondent and CompTotal. Comptotal is the salary before curreny conversion 
# It does not provide valuable information unless we focus on one country.
df = df.drop(['Respondent','CompTotal'],axis=1)


# In[ ]:


# Check the ratio of data scientist and non data scientist
sns.countplot(df['DevType_Data scientist or machine learning specialist'])


# In[ ]:


# Impute missing values with column median
# The distribution of these variables are not normal
# so I use column median rather than column mean.
# No needed to impute the one-hot columns.
df['CodeRevHrs'] = df['CodeRevHrs'].fillna(df['CodeRevHrs'].median())
df['Age'] = df['Age'].fillna(df['Age'].median())
df['WorkWeekHrs'] = df['WorkWeekHrs'].fillna(df['WorkWeekHrs'].median())
df['ConvertedComp'] = df['ConvertedComp'].fillna(df['ConvertedComp'].median())


# In[ ]:


# See if the numeric variables need to be normalized.
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10,10))
plt.tight_layout(w_pad=2.0, h_pad=5.0)

plt.subplot(2,2,1)
plt.xlabel('ConvertedComp')
p1 = sns.distplot(df[['ConvertedComp']], hist=False,color='red',norm_hist=True)

plt.subplot(2,2,2)
plt.xlabel('WorkWeekHrs')
p2 = sns.distplot(df[['WorkWeekHrs']], hist=False,color='red',norm_hist=True)

plt.subplot(2,2,3)
plt.xlabel('CodeRevHrs')
p3 = sns.distplot(df[['CodeRevHrs']], hist=False,color='red',norm_hist=True)

plt.subplot(2,2,4)
plt.xlabel('Age')
p4 = sns.distplot(df[['Age']], hist=False,color='red',norm_hist=True)


# In[ ]:


# Use logarithm function to transform the numeric columns
df['ConvertedComp_log'] = np.log(df['ConvertedComp']-90000)
df['WorkWeekHrs_log'] = np.log(df['WorkWeekHrs']-70)
df['CodeRevHrs_log'] = np.log(df['CodeRevHrs']-30)
df['Age_log'] = np.log(df['Age']+30)


fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10,10))
plt.tight_layout(w_pad=2.0, h_pad=5.0)

plt.subplot(2,2,1)
plt.xlabel('ConvertedComp_log')
p1 = sns.distplot(df[['ConvertedComp_log']], hist=False,color='red',norm_hist=True)

plt.subplot(2,2,2)
plt.xlabel('WorkWeekHrs_log')
p2 = sns.distplot(df[['WorkWeekHrs_log']], hist=False,color='red',norm_hist=True)

plt.subplot(2,2,3)
plt.xlabel('CodeRevHrs_log')
p3 = sns.distplot(df[['CodeRevHrs_log']], hist=False,color='red',norm_hist=True)

plt.subplot(2,2,4)
plt.xlabel('Age_log')
p4 = sns.distplot(df[['Age_log']], hist=False,color='red',norm_hist=True)


# ### 5.2 Modeling

# In[ ]:


# Split data set into response vector and feature matrix.
y = df['DevType_Data scientist or machine learning specialist'].values
X = df.drop(['DevType_Data scientist or machine learning specialist'], axis=1)


# In[ ]:


# Drop original numeric columns. 
# We already know that researcher and data engineer might share the role of data scientist
# Drop DevType as it does not provide the information why data scientist is distinct. 

X = X.drop([col for col in X.columns if 'DevType_' in col],axis=1)
X = X.drop(['CodeRevHrs','ConvertedComp','WorkWeekHrs','Age'],axis=1)
X.shape


# In[ ]:


# Split X y into training set and test set.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=0,stratify=y)


# In[ ]:


# Define an Xgboost classifer
# Using AUPRC as the evaluation metric which is more sensitive to the minor class 
# As we know the population of data scientist is just 1/8 of other developers

model = xgb.XGBClassifier(
    learning_rate =0.1, n_estimators=1000,
    gamma=0,
    subsample=0.8,
    colsample_bytree=0.8,
    objective= 'binary:logistic', 
    nthread=4,
    scale_pos_weight=7,
    seed=27,
    max_depth = 5,
    min_child_weight = 5
)

def evalauc(preds, dtrain):
    labels = dtrain.get_label()
    precision, recall, thresholds = precision_recall_curve(labels, preds)
    area = auc(recall, precision)
    return 'AUPRC', -area


model.fit(X_train, y_train,
          eval_metric=evalauc,
          eval_set=[(X_train, y_train), (X_test, y_test)],
          early_stopping_rounds=5,
         verbose=True)


# ### 5.3 Evaluation

# In[ ]:


# See the prediction result
predict = model.predict(X_test)
print(classification_report(y_test, predict))
print(confusion_matrix(y_test,predict))
print("Accuracy: ")
print(accuracy_score(y_test,predict))


# Accuracy is high but it is not a good metric due to bias classes. F1-score is fair 0.53 for the data scientist prediction. We can see many non data scientist are being classified as data scientist.

# In[ ]:


fpr, tpr, thresholds = precision_recall_curve(y_test, model.predict_proba(X_test)[:,1])
auprc = average_precision_score(y_test, predict)

plt.plot(fpr, tpr, lw=1, label='AUPRC = %0.2f'%(auprc))
plt.plot([0, 1], [0, 1], '--k', lw=1)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('XGBOOST AUPRC')
plt.legend(loc="lower right", frameon = True).get_frame().set_edgecolor('black')


# In[ ]:


fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:,1])
roc_auc = roc_auc_score(y_test, predict)

plt.plot(fpr, tpr, lw=1, label='AUC = %0.2f'%(roc_auc))
plt.plot([0, 1], [0, 1], '--k', lw=1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('XGBOOST ROC')
plt.legend(loc="lower right", frameon = True).get_frame().set_edgecolor('black')


# In[ ]:


feature_name=X.columns.tolist()
#feature_name.remove('DevType_Data scientist or machine learning specialist')
dtrain = xgb.DMatrix(X, label=y,feature_names=feature_name)


# In[ ]:


# model.get_booster().get_score().items()


# In[ ]:


# mapper = {'f{0}'.format(i): v for i, v in enumerate(dtrain.feature_names)}
# mapped = { mapper[k]: v for k, v in model.get_booster().get_score().items()}

fig,ax  =  plt.subplots (figsize=(10, 5))
xgb.plot_importance(model, max_num_features=20,ax=ax)
plt.show()


# In[ ]:



df.groupby(['Employment_Not employed, and not looking for work', 'DevType_Data scientist or machine learning specialist']).size()


# The classifier does not return a clear bountary between data scientist and non data scientist. I am interested if we remove the mixed data scientist from the training and test data, will the model perform better? Let's do it.

# In[ ]:


# Retrieve the character variables.
# df = survey_2019[survey_2019['Country']=='United States']
df = pd.concat([pure_data_scientist,non_data_scientist])
cat_vars_int = df.select_dtypes(include=['object']).copy().columns
len(cat_vars_int)
df.shape


# In[ ]:


# Again split and transpose the categorical variable into one-hot columns.
for var in  cat_vars_int:
    # for each cat add dummy var, drop original column
    df = pd.concat([df.drop(var, axis=1), df[var].str.get_dummies(sep=';').rename(lambda x: var+'_' + x, axis='columns')], axis=1)

df.describe()


# In[ ]:


df.shape


# In[ ]:


sns.countplot(df['DevType_Data scientist or machine learning specialist'])


# The classes are extremely bias.

# In[ ]:


np.sum(df['DevType_Data scientist or machine learning specialist'])


# In[ ]:


# Impute missing values with column median
df['CodeRevHrs'] = df['CodeRevHrs'].fillna(df['CodeRevHrs'].median())
df['Age'] = df['Age'].fillna(df['Age'].median())
df['WorkWeekHrs'] = df['WorkWeekHrs'].fillna(df['WorkWeekHrs'].median())
df['ConvertedComp'] = df['ConvertedComp'].fillna(df['ConvertedComp'].median())


# In[ ]:


# See if the numeric variables need to be normalized.
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10,10))
plt.tight_layout(w_pad=2.0, h_pad=5.0)

plt.subplot(2,2,1)
plt.xlabel('ConvertedComp')
p1 = sns.distplot(df[['ConvertedComp']], hist=False,color='red',norm_hist=True)

plt.subplot(2,2,2)
plt.xlabel('WorkWeekHrs')
p2 = sns.distplot(df[['WorkWeekHrs']], hist=False,color='red',norm_hist=True)

plt.subplot(2,2,3)
plt.xlabel('CodeRevHrs')
p3 = sns.distplot(df[['CodeRevHrs']], hist=False,color='red',norm_hist=True)

plt.subplot(2,2,4)
plt.xlabel('Age')
p4 = sns.distplot(df[['Age']], hist=False,color='red',norm_hist=True)


# In[ ]:


# Use logarithm function to transform the numeric columns
df['ConvertedComp_log'] = np.log(df['ConvertedComp']+1000)
df['WorkWeekHrs_log'] = np.log(df['WorkWeekHrs']-70)
df['CodeRevHrs_log'] = np.log(df['CodeRevHrs']-30)
df['Age_log'] = np.log(df['Age']+30)


fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10,10))
plt.tight_layout(w_pad=2.0, h_pad=5.0)

plt.subplot(2,2,1)
plt.xlabel('ConvertedComp_log')
p1 = sns.distplot(df[['ConvertedComp_log']], hist=False,color='red',norm_hist=True)

plt.subplot(2,2,2)
plt.xlabel('WorkWeekHrs_log')
p2 = sns.distplot(df[['WorkWeekHrs_log']], hist=False,color='red',norm_hist=True)

plt.subplot(2,2,3)
plt.xlabel('CodeRevHrs_log')
p3 = sns.distplot(df[['CodeRevHrs_log']], hist=False,color='red',norm_hist=True)

plt.subplot(2,2,4)
plt.xlabel('Age_log')
p4 = sns.distplot(df[['Age_log']], hist=False,color='red',norm_hist=True)


# In[ ]:


# Split data set into response vector and feature matrix.
y = df['DevType_Data scientist or machine learning specialist'].values
X = df.drop(['DevType_Data scientist or machine learning specialist'], axis=1)


# In[ ]:


X = X.drop(['Respondent','CompTotal'],axis=1)
X = X.drop([col for col in X.columns if 'DevType_' in col],axis=1)
X = X.drop(['CodeRevHrs','ConvertedComp','WorkWeekHrs','Age'],axis=1)
X.shape


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=0,stratify=y)


# In[ ]:


model = xgb.XGBClassifier(
    learning_rate =0.01, n_estimators=1000,
    gamma=0,
    subsample=0.8,
    colsample_bytree=0.8,
    objective= 'binary:logistic', 
    nthread=4,
    scale_pos_weight=100,
    seed=27,
    max_depth = 5,
    min_child_weight = 3
)

def evalauc(preds, dtrain):
    labels = dtrain.get_label()
    precision, recall, thresholds = precision_recall_curve(labels, preds)
    area = auc(recall, precision)
    return 'AUPRC', -area


model.fit(X_train, y_train,
          eval_metric=evalauc,
          eval_set=[(X_train, y_train), (X_test, y_test)],
          early_stopping_rounds=5,
         verbose=True)


# In[ ]:


# See the prediction result
predict = model.predict(X_test)
print(classification_report(y_test, predict))
print(confusion_matrix(y_test,predict))
print("Accuracy: ")
print(accuracy_score(y_test,predict))


# In[ ]:


fpr, tpr, thresholds = precision_recall_curve(y_test, model.predict_proba(X_test)[:,1])
auprc = average_precision_score(y_test, predict)

plt.plot(fpr, tpr, lw=1, label='AUPRC = %0.2f'%(auprc))
plt.plot([0, 1], [0, 1], '--k', lw=1)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('XGBOOST AUPRC')
plt.legend(loc="lower right", frameon = True).get_frame().set_edgecolor('black')


# In[ ]:


fig,ax  =  plt.subplots (figsize=(10, 5))
xgb.plot_importance(model, max_num_features=20,ax=ax)
plt.show()


# In[ ]:


df.groupby(['DevType_Data scientist or machine learning specialist','MiscTechWorkedWith_TensorFlow']).size()


# Still, the model is not as good as I anticipate, there are many other developers being classified as data scientist. We can see somehow the top important feature is TensorFlow known by almost 50% of pure data scientist, while part of the non data scientist also know. The most distinct feature seems not to distinguish a data scientist very well. 
