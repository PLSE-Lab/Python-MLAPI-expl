#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd 
import os

import nltk
from nltk.stem import WordNetLemmatizer 
from collections import Counter
import wordcloud
import matplotlib.pyplot as plt
import scipy.stats as ss
import seaborn as sns

#for dirname, _, filenames in os.walk('/kaggle/input'):
#    for filename in filenames:
#        print(os.path.join(dirname, filename))


# # Functions

# In[ ]:


def business_type_plot_hist(_df2019, current_question, x_title, topn='all'):
    fig, axs = plt.subplots(1,4, figsize=(16,4), constrained_layout=True)

    bs_axis = {'micro' : [0], 'small' : [1], 'medium_large' : [2], 
               'enterprise' : [3]}#, 'large_enterprise' : [4]}

    for business_size, coords in bs_axis.items():
        if topn == 'all':
            df = _df2019.loc[_df2019['Business_size'] == business_size][current_question].value_counts(normalize=True)
        else:
            df = _df2019.loc[_df2019['Business_size'] == business_size][current_question].value_counts(normalize=True).head(topn)            
        bcplot = sns.barplot(x=df.index, y=df.values, ax=axs[coords[0]]);
        axs[coords[0]].set_title(business_size);
        for p in axs[coords[0]].patches:
            axs[coords[0]].annotate(str("{0:.0%}".format(p.get_height())), (p.get_x() * 0.97, p.get_height() * 0.97))
        for item in bcplot.get_xticklabels():
            item.set_rotation(90);
    for ax in axs.flat:
        ax.set(xlabel=x_title, ylabel='%_of_employees')

    #for ax in axs.flat:
    #    ax.label_outer()    

def salary_plot(_df2019, current_question, x_title, topn='all'):
    fig, axs = plt.subplots(1,5, figsize=(15,5), constrained_layout=True)

    bs_axis = {'micro' : [0], 'small' : [1], 'medium_large' : [2], 
               'enterprise' : [3], 'large_enterprise' : [4]}

    for sgroup, coords in bs_axis.items():
        if topn == 'all':
            df = _df2019.loc[_df2019['Salary'] == sgroup][current_question].value_counts(normalize=True)
        else:
            df = _df2019.loc[_df2019['Salary'] == sgroup][current_question].value_counts(normalize=True).head(topn)            
        bcplot = sns.barplot(x=df.index, y=df.values, ax=axs[coords[1]]);
        axs[coords[0]].set_title(business_size);
        for p in axs[coords[0]].patches:
            axs[coords[0]].annotate(str("{0:.0%}".format(p.get_height())), (p.get_x() * 0.97, p.get_height() * 0.97))
        for item in bcplot.get_xticklabels():
            item.set_rotation(90);
    for ax in axs.flat:
        ax.set(xlabel=x_title, ylabel='N_of_employees')

    for ax in axs.flat:
        ax.label_outer()    
        
        
def business_type(row):
    if row == '0-49 employees':
        return 'micro'
    elif row == '50-249 employees':
        return 'small'
    elif row == '250-999 employees':
        return 'medium_large'
    elif row == '1000-9,999 employees':
        return 'enterprise'
    elif row == '> 10,000 employees':
        return 'enterprise' #'large_enterprise'
    else:
        'no_answer'
        
        
def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x,y)
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
    rcorr = r-((r-1)**2)/(n-1)
    kcorr = k-((k-1)**2)/(n-1)
    return np.sqrt(phi2corr/min((kcorr-1),(rcorr-1)))

def associations(dataset, return_results=False, plot=True, **kwargs):
    columns = dataset.columns
    corr = pd.DataFrame(index=columns, columns=columns)
    for i in range(0,len(columns)):
        for j in range(i,len(columns)):
            if i == j:
                corr[columns[i]][columns[j]] = 1.0
            else:
                cell = cramers_v(dataset[columns[i]], dataset[columns[j]])
                corr[columns[i]][columns[j]] = cell
                corr[columns[j]][columns[i]] = cell
    corr.fillna(value=np.nan, inplace=True)
    if plot:
        plt.figure(figsize=kwargs.get('figsize',None))
        sns.heatmap(corr, annot=kwargs.get('annot',True), fmt=kwargs.get('fmt','.2f'),
                   cbar_kws = dict(use_gridspec=False,location="top"))
        #plt.show()
    if return_results:
        return corr


# # Data

# In[ ]:


df_2019 = pd.read_csv('../input/kaggle-survey-2019/multiple_choice_responses.csv', low_memory=False);
df_2019.columns = df_2019.iloc[0];
df_2019 = df_2019.drop([0]);
print ("Columns count in 2019: ", df_2019.shape[1])


# # Small companies VS corporations

# In[ ]:


q = ['What is your age (# years)?',
'What is your gender? - Selected Choice',
'In which country do you currently reside?',
'What is the highest level of formal education that you have attained or plan to attain within the next 2 years?',
'Select the title most similar to your current role (or most recent title if retired): - Selected Choice',
'What is the size of the company where you are employed?',
'Approximately how many individuals are responsible for data science workloads at your place of business?',
'Does your current employer incorporate machine learning methods into their business?',
'What is your current yearly compensation (approximate $USD)?'
    ]


# Let's define a business type based on a total employee number. I suppose, we should exclude '0-49' group from small business and consider them as startups or family bisuness or something like this. Then, '50-249' is a small business. Should we consider small and medium in union or separately? Let's see.

# In[ ]:


_df2019 = df_2019[q].copy()
_df2019['Business_size'] = _df2019['What is the size of the company where you are employed?'].apply(lambda x : business_type(x) )
bcount = _df2019['Business_size'].value_counts()
bcount_plt = sns.barplot(bcount.index, bcount.values, alpha=0.8);
for item in bcount_plt.get_xticklabels():
    item.set_rotation(45);


# Most of respondents work in enterprise segment. Let's select only Data Scientists

# In[ ]:


bcount = _df2019.loc[_df2019['Select the title most similar to your current role (or most recent title if retired): - Selected Choice'] == 'Data Scientist']['Business_size'].value_counts()
bcount_plt = sns.barplot(bcount.index, bcount.values, alpha=0.8);
for item in bcount_plt.get_xticklabels():
    item.set_rotation(45);


# Well, nothing changes. Let's go deeper and compare age, gender, education level in each group. 

# ## Job Title

# In[ ]:


current_question = 'Select the title most similar to your current role (or most recent title if retired): - Selected Choice'

business_type_plot_hist(_df2019, 
                   current_question=current_question,
                   x_title='Job_Title',
                    topn='all')


# Job titles are far more interesting. We can assume that there are industry-specific companies in "small-medium" group. Micro , small and enterprise segments are alike and may have DS as departments to solve business tasks or provide DS as a service.

# ## Age

# In[ ]:


business_type_plot_hist(_df2019.loc[_df2019['Select the title most similar to your current role (or most recent title if retired): - Selected Choice'] == 'Data Scientist'], 
                   current_question='What is your age (# years)?',
                   x_title='Age_Group')


# People of 25-29 is the most represented group of respondents. We should pay attention to the fact that the staff is younger in micro companies and older in enterprise.

# # Gender

# In[ ]:


business_type_plot_hist(_df2019.loc[_df2019['Select the title most similar to your current role (or most recent title if retired): - Selected Choice'] == 'Data Scientist'], 
                   current_question='What is your gender? - Selected Choice',
                   x_title='Gender_Group')


# Gender statistics does not depend on the size of company. We can just note that 2% more women work in enterprise segment.

# # Countries - top 10 for each type

# In[ ]:


current_question = 'In which country do you currently reside?'

_df2019[current_question] = _df2019[current_question].replace(['United States of America'], 'USA')
_df2019[current_question] = _df2019[current_question].replace(['United Kingdom of Great Britain and Northern Ireland'], 'UK')

business_type_plot_hist(_df2019.loc[_df2019['Select the title most similar to your current role (or most recent title if retired): - Selected Choice'] == 'Data Scientist'], 
                   current_question=current_question,
                   topn=10,
                   x_title='Country')


# Only top2 is significant here. Data scientists who work in micro and small segment are likely to reside India. And employees of  medium\large and enterprise are reside USA. Unfortunately, we do not have a statistics about remote and in-office work so we can not make conclusions about conuntry-spesific business types. However, we can check the total distribution of countries to find some ideas.

# In[ ]:


_df2019.loc[_df2019['Select the title most similar to your current role (or most recent title if retired): - Selected Choice']  == 'Data Scientist']['In which country do you currently reside?'].value_counts().head(5).plot.pie()


# India and USA are the most frequent so it is not surprising that they are in top in all groups. 

# ## Education

# In[ ]:


current_question = 'What is the highest level of formal education that you have attained or plan to attain within the next 2 years?'

business_type_plot_hist(_df2019.loc[_df2019['Select the title most similar to your current role (or most recent title if retired): - Selected Choice'] == 'Data Scientist'], 
                   current_question=current_question,
                   x_title='Education')


# Speaking about education. Masters degree is the most popular type of formal education. By the way, medium \ large segments attracts more Doctors than Bachelors while micro, small and enterprise demonstrate the opposite situation.
# 

# ## Size of Data Science teams

# In[ ]:


current_question = 'Approximately how many individuals are responsible for data science workloads at your place of business?'

business_type_plot_hist(_df2019.loc[_df2019['Select the title most similar to your current role (or most recent title if retired): - Selected Choice'] == 'Data Scientist'], 
                   current_question=current_question,
                   x_title='DS_Staff')


# The statistics is quite obvious. It's enough 1-2 or 3-4 Data scientists to solve tasks for micro-business but enterprise can afford to hire an entire DS department. Medium \ large companies take an intermediate position.

# # Salaries

# In[ ]:


current_question = 'What is your current yearly compensation (approximate $USD)?'
business_type_plot_hist(_df2019.loc[_df2019['Select the title most similar to your current role (or most recent title if retired): - Selected Choice'] == 'Data Scientist'], 
                   current_question=current_question,
                   topn=10,
                   x_title='Salaries')


# Well, salaries are the most interesting. Histograms show that there is an enourmous gap between the most frequent salaries and the others in micro segment. And, moreover, the most popular YEARLY compensation in micro, small and medium\large is less then 1000$! We can assume that this is an intern positions. Well, in small and medium \ large sement the distribution is far more balanced. Enterprise offers really huge salaries. 
# 
# See [this article](https://datasciencedegree.wisconsin.edu/data-science/data-scientist-salary/) by University of Wisconsin to figure out the market situation.

# # What's the correlation between all these questions?

# Many thanks to Shaked Zychlinski (see the article for more info: 
# https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9 )

# In[ ]:


np.set_printoptions(precision=2)
qdict = {'What is your age (# years)?' : 'Age',
'What is your gender? - Selected Choice' : 'Gender',
'In which country do you currently reside?' : 'Country',
'What is the highest level of formal education that you have attained or plan to attain within the next 2 years?' : 'Education',
'Select the title most similar to your current role (or most recent title if retired): - Selected Choice' : 'Job Title',
'What is the size of the company where you are employed?' : 'Company Size',
'Approximately how many individuals are responsible for data science workloads at your place of business?' : 'DS_Staff_Size',
'Does your current employer incorporate machine learning methods into their business?' : 'ML_Usage',
'What is your current yearly compensation (approximate $USD)?' : 'Salary'}
_df2019 = _df2019.rename(columns=qdict)
_df2019.columns
associations(_df2019[qdict.values()])


# I computed correlation between categorical values using Kramer's  V coefficient. Well, Age is highly correlated with Education and Job Title, Gender is highly correlated with Country and Job Title, Education and Job Title are highly correlated with Age, Company size is highly correlated with CompanySize and so on. Let's go deeper and discover the portrait of rich and happy data scientist.

# # Skills and technologies

# ## Experience

# In[ ]:


q = ['For how many years have you used machine learning methods?', 'What is the size of the company where you are employed?', 'Select the title most similar to your current role (or most recent title if retired): - Selected Choice']
_df2019 = df_2019[q].copy()
_df2019['Business_size'] = _df2019['What is the size of the company where you are employed?'].apply(lambda x : business_type(x) )

current_question = 'For how many years have you used machine learning methods?'
business_type_plot_hist(_df2019.loc[_df2019['Select the title most similar to your current role (or most recent title if retired): - Selected Choice'] == 'Data Scientist'], 
                   current_question=current_question,
                   topn=10,
                   x_title='ML_Experience')


# Well, wee see that most of Data scientists were attracted at the industry 1-3 years ago. But enterprise have more experienced employees so this is one of the reasion of large salaries.

# ## How long have you been writing code to analyze data (at work or at school)?

# In[ ]:


q = ['How long have you been writing code to analyze data (at work or at school)?', 'What is the size of the company where you are employed?', 'Select the title most similar to your current role (or most recent title if retired): - Selected Choice']
_df2019 = df_2019[q].copy()
_df2019['Business_size'] = _df2019['What is the size of the company where you are employed?'].apply(lambda x : business_type(x) )

current_question = 'How long have you been writing code to analyze data (at work or at school)?'
business_type_plot_hist(_df2019.loc[_df2019['Select the title most similar to your current role (or most recent title if retired): - Selected Choice'] == 'Data Scientist'], 
                   current_question=current_question,
                   topn=10,
                   x_title='CodingExperience')


# Well, speaking about programming experience. We can conclude that data scientists have more solid programming background than ML background. And the size of the company has and influence on the experiense. 

# ## Select any activities that make up an important part of your role at work

# In[ ]:


qacts = [ 'Select any activities that make up an important part of your role at work: (Select all that apply) - Selected Choice - Analyze and understand data to influence product or business decisions',
 'Select any activities that make up an important part of your role at work: (Select all that apply) - Selected Choice - Build and/or run the data infrastructure that my business uses for storing, analyzing, and operationalizing data',
 'Select any activities that make up an important part of your role at work: (Select all that apply) - Selected Choice - Build prototypes to explore applying machine learning to new areas',
 'Select any activities that make up an important part of your role at work: (Select all that apply) - Selected Choice - Build and/or run a machine learning service that operationally improves my product or workflows',
 'Select any activities that make up an important part of your role at work: (Select all that apply) - Selected Choice - Experimentation and iteration to improve existing ML models',
 'Select any activities that make up an important part of your role at work: (Select all that apply) - Selected Choice - Do research that advances the state of the art of machine learning',
 'Select any activities that make up an important part of your role at work: (Select all that apply) - Selected Choice - None of these activities are an important part of my role at work',
 'What is the size of the company where you are employed?',
 'What is your current yearly compensation (approximate $USD)?',
 'Select the title most similar to your current role (or most recent title if retired): - Selected Choice',
  'In which country do you currently reside?'
]

_df2019_qa = df_2019[qacts].copy()
_df2019_qa['Business_size'] = _df2019_qa['What is the size of the company where you are employed?'].apply(lambda x : business_type(x) )
for qa in qacts[:-5]:
    _df2019_qa[qa] = _df2019_qa[qa].apply(lambda x: 1 if not pd.isnull(x) else 0)
   # _df2019_qa[qa] = _df2019_qa[qa].astype('int8')

_df2019_qa.columns = [ 'Analyze and understand data to influence product or business decisions',
 'Build and/or run the data infrastructure that my business uses for storing, analyzing, and operationalizing data',
 'Build prototypes to explore applying machine learning to new areas',
 'Build and/or run a machine learning service that operationally improves my product or workflows',
 'Experimentation and iteration to improve existing ML models',
 'Do research that advances the state of the art of machine learning',
 'None of these activities are an important part of my role at work',
 'CompanySize',
  'Salary',
 'Job Title',
   'Country',
   'Business_size'
  ]


# In[ ]:


fig, ax = plt.subplots(2, 3, figsize = (15,6))
plt.subplots_adjust(hspace = 1)
cords = { 'Analyze and understand data to influence product or business decisions' : [0,0],
 'Build and/or run the data infrastructure that my business uses for storing, analyzing, and operationalizing data' : [0,1],
 'Build prototypes to explore applying machine learning to new areas' : [0,2],
 'Build and/or run a machine learning service that operationally improves my product or workflows' : [1,0],
 'Experimentation and iteration to improve existing ML models' : [1,1],
 'Do research that advances the state of the art of machine learning' : [1,2]} 
from textwrap import wrap
for qq in _df2019_qa.columns[:-6]:
    _df = _df2019_qa.loc[_df2019_qa['Job Title'] == 'Data Scientist']
    confusion_matrix = pd.crosstab(_df[qq]
                                   , _df['Business_size']
                                  , normalize='columns')
    df = confusion_matrix.loc[confusion_matrix.index == 1].T.reset_index()
    df.columns = ['Business_size', '%_Employees']    
    bcplot = sns.barplot(x=df['Business_size'], y=df['%_Employees'], ax=ax[cords[qq][0], cords[qq][1]]);
    ax[cords[qq][0], cords[qq][1]].set_title("\n".join(wrap(qq, 30)));
    for p in ax[cords[qq][0], cords[qq][1]].patches: 
        ax[cords[qq][0], cords[qq][1]].annotate(str("{0:.0%}".format(p.get_height())), (p.get_x() + p.get_width() / 2., p.get_height()+0.009),
                   ha='center', va='center', textcoords='offset points')
    for item in bcplot.get_xticklabels():
        item.set_rotation(90);
    ax[cords[qq][0], cords[qq][1]].set(xlabel='', ylabel='%_Employees')
    ax[cords[qq][0], cords[qq][1]].label_outer()   
    bcplot.set(yticks=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])


# Well, we computed % of employees doing each type of tasks VS not doing and show those who answered positively. Data analisys and building ML applications are the core competense for DS and the research  is not.

# ## Hardware

# In[ ]:


q = ['Have you ever used a TPU (tensor processing unit)?',
 'Which types of specialized hardware do you use on a regular basis?  (Select all that apply) - Selected Choice - CPUs',
 'Which types of specialized hardware do you use on a regular basis?  (Select all that apply) - Selected Choice - GPUs',
 'Which types of specialized hardware do you use on a regular basis?  (Select all that apply) - Selected Choice - None / I do not know',
 'Which types of specialized hardware do you use on a regular basis?  (Select all that apply) - Selected Choice - Other',
 'Which types of specialized hardware do you use on a regular basis?  (Select all that apply) - Selected Choice - TPUs',
 'What is the size of the company where you are employed?',
 'Select the title most similar to your current role (or most recent title if retired): - Selected Choice',
]
_df2019 = df_2019[q].copy()
_df2019['Business_size'] = _df2019['What is the size of the company where you are employed?'].apply(lambda x : business_type(x) )

for qa in _df2019.columns[1:-3]:
    _df2019[qa] = _df2019[qa].apply(lambda x: 1 if not pd.isnull(x) else 0)
   # _df2019_qa[qa] = _df2019_qa[qa].astype('int8')


# Let's look at hardware. Well, the most basic question is about TPU's.  

# In[ ]:


business_type_plot_hist(_df2019, 
                   current_question='Have you ever used a TPU (tensor processing unit)?',
                   x_title='TPUS')


# Most of respondents never used it but employees of micro and small busines are a little bit more open to innovations. Ok, more wide question will show us a fuul picture.

# In[ ]:


_df2019.columns = [ 'Have you ever used a TPU (tensor processing unit)?',
                    'CPUs', 'GPUs', 'None \ I dont know', 'Other', 'TPUs', 
                   'What is the size of the company where you are employed?',
                  'Job Title',  'Business_size' ]


fig, ax = plt.subplots(2, 3, figsize = (15,6))
plt.subplots_adjust(hspace = 1)
cords = { 'CPUs' : [0,0],
 'GPUs' : [0,1],
 'TPUs' : [0,2],
 'Other' : [1,0],
 'None \ I dont know' : [1,1]} 
from textwrap import wrap
for qq in _df2019.columns[1:-3]:
    _df = _df2019.loc[_df2019['Job Title'] == 'Data Scientist']
    confusion_matrix = pd.crosstab(_df[qq]
                                   , _df['Business_size']
                                  , normalize='columns')
    df = confusion_matrix.loc[confusion_matrix.index == 1].T.reset_index()
    df.columns = ['Business_size', '%_Employees']    
    bcplot = sns.barplot(x=df['Business_size'], y=df['%_Employees'], ax=ax[cords[qq][0], cords[qq][1]]);
    ax[cords[qq][0], cords[qq][1]].set_title("\n".join(wrap(qq, 30)));
    for p in ax[cords[qq][0], cords[qq][1]].patches: 
        ax[cords[qq][0], cords[qq][1]].annotate(str("{0:.0%}".format(p.get_height())), (p.get_x() + p.get_width() / 2., p.get_height()+0.009),
                   ha='center', va='center', textcoords='offset points')
    for item in bcplot.get_xticklabels():
        item.set_rotation(90);
    ax[cords[qq][0], cords[qq][1]].set(xlabel='', ylabel='%_Employees')
    ax[cords[qq][0], cords[qq][1]].label_outer()   
    bcplot.set(yticks=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
fig.delaxes(ax[1][2])


# Nothing special, you see. 

# ## Software

# In[ ]:


q = ['What data visualization libraries or tools do you use on a regular basis?  (Select all that apply) - Selected Choice -  Altair ',
 'What data visualization libraries or tools do you use on a regular basis?  (Select all that apply) - Selected Choice -  Bokeh ',
 'What data visualization libraries or tools do you use on a regular basis?  (Select all that apply) - Selected Choice -  D3.js ',
 'What data visualization libraries or tools do you use on a regular basis?  (Select all that apply) - Selected Choice -  Geoplotlib ',
 'What data visualization libraries or tools do you use on a regular basis?  (Select all that apply) - Selected Choice -  Ggplot / ggplot2 ',
 'What data visualization libraries or tools do you use on a regular basis?  (Select all that apply) - Selected Choice -  Leaflet / Folium ',
 'What data visualization libraries or tools do you use on a regular basis?  (Select all that apply) - Selected Choice -  Matplotlib ',
 'What data visualization libraries or tools do you use on a regular basis?  (Select all that apply) - Selected Choice -  Plotly / Plotly Express ',
 'What data visualization libraries or tools do you use on a regular basis?  (Select all that apply) - Selected Choice -  Seaborn ',
 'What data visualization libraries or tools do you use on a regular basis?  (Select all that apply) - Selected Choice -  Shiny ',
 'What data visualization libraries or tools do you use on a regular basis?  (Select all that apply) - Selected Choice - None',
     'What data visualization libraries or tools do you use on a regular basis?  (Select all that apply) - Selected Choice - Other',
     'What is the size of the company where you are employed?',
 'Select the title most similar to your current role (or most recent title if retired): - Selected Choice',
]
_df2019 = df_2019[q].copy()
_df2019['Business_size'] = _df2019['What is the size of the company where you are employed?'].apply(lambda x : business_type(x) )
for qa in _df2019.columns[:-3]:
    _df2019[qa] = _df2019[qa].apply(lambda x: 1 if not pd.isnull(x) else 0)
    


# In[ ]:



_df2019.columns = [ 'Altair', 'Bokeh', ' D3.js', 'Geoplotlib', 'GGplot', 'Leaflet / Folium', 
                   'Matplotlib', 'Plotly', 'Seaborn', 'Shiny', 'None', 'Other',
                   'What is the size of the company where you are employed?',
                  'Job Title',  'Business_size' ]

dd = _df2019.loc[_df2019['Job Title'] == 'Data Scientist'][[ 'Altair', 'Bokeh', ' D3.js', 'Geoplotlib', 'GGplot', 'Leaflet / Folium', 
                   'Matplotlib', 'Plotly', 'Seaborn', 'Shiny', 'None', 'Other',  'Business_size']].groupby('Business_size').sum()
dd =  dd / _df2019.loc[_df2019['Job Title'] == 'Data Scientist'].shape[0]
dd = dd.reset_index()

topn = 'all'
fig, axs = plt.subplots(2,2, figsize=(15,6), constrained_layout=True)

bs_axis = {'micro' : [0,0], 'small' : [0,1], 'medium_large' : [1,0], 
           'enterprise' : [1,1]}

for business_size, coords in bs_axis.items():
    if topn == 'all':
        df = dd.loc[dd['Business_size'] == business_size].set_index('Business_size').T#[current_question].value_counts(normalize=True)
    else:
        df = dd.loc[dd['Business_size'] == business_size].set_index('Business_size').T.head(topn)#[current_question].value_counts(normalize=True).head(topn)            
    bcplot = sns.barplot(x=df.index, y=df[business_size], ax=axs[coords[0],coords[1] ]);
    axs[coords[0],coords[1] ].set_title(business_size);
    for p in axs[coords[0],coords[1] ].patches:
        axs[coords[0],coords[1] ].annotate(str("{0:.0%}".format(p.get_height())), (p.get_x() * 0.97, p.get_height() * 0.97))
    for item in bcplot.get_xticklabels():
        item.set_rotation(90);
    bcplot.set(yticks=[0.1, 0.2, 0.3, 0.4, 0.5]);
for ax in axs.flat:
    ax.set(xlabel='', ylabel='N_of_employees')

for ax in axs.flat:
    ax.label_outer() 

#fig.delaxes(axs[1][2])


# The sum is not equal to 100% because a lot of people answered NaN. Well , Matplotlib, seaborn, ggplot and Plotly are the most popular visualisation instruments. But you can see that micro and large enterprise campanies are more interested in visualisation task because the sum of four most popular framewors is above 40%. 

# ## What programming languages do you use on a regular basis?

# In[ ]:


q = [ 'What programming languages do you use on a regular basis? (Select all that apply) - Selected Choice - Bash',
 'What programming languages do you use on a regular basis? (Select all that apply) - Selected Choice - C',
 'What programming languages do you use on a regular basis? (Select all that apply) - Selected Choice - C++',
 'What programming languages do you use on a regular basis? (Select all that apply) - Selected Choice - Java',
 'What programming languages do you use on a regular basis? (Select all that apply) - Selected Choice - Javascript',
 'What programming languages do you use on a regular basis? (Select all that apply) - Selected Choice - MATLAB',
 'What programming languages do you use on a regular basis? (Select all that apply) - Selected Choice - None',
 'What programming languages do you use on a regular basis? (Select all that apply) - Selected Choice - Other',
 'What programming languages do you use on a regular basis? (Select all that apply) - Selected Choice - Python',
 'What programming languages do you use on a regular basis? (Select all that apply) - Selected Choice - R',
 'What programming languages do you use on a regular basis? (Select all that apply) - Selected Choice - SQL',
 'What programming languages do you use on a regular basis? (Select all that apply) - Selected Choice - TypeScript',
 'What is the size of the company where you are employed?',
 'Select the title most similar to your current role (or most recent title if retired): - Selected Choice',
]
_df2019 = df_2019[q].copy()
_df2019['Business_size'] = _df2019['What is the size of the company where you are employed?'].apply(lambda x : business_type(x) )
for qa in _df2019.columns[:-2]:
    _df2019[qa] = _df2019[qa].apply(lambda x: 1 if not pd.isnull(x) else 0)


# In[ ]:


_df2019.columns = [ 'Bash', 'C','C++','Java', 'Javascript',  'MATLAB', 'None', 
                   'Other', 'Python', 'R', 'SQL', 'Typescript',
                   'What is the size of the company where you are employed?',
                  'Job Title',  'Business_size' ]

dd = _df2019.loc[_df2019['Job Title'] == 'Data Scientist'][['Bash', 'C','C++','Java', 'Javascript',  'MATLAB', 'None', 
                   'Other', 'Python', 'R', 'SQL', 'Typescript',  'Business_size']].groupby('Business_size').sum()
dd =  dd / _df2019.loc[_df2019['Job Title'] == 'Data Scientist'].shape[0]
dd = dd.reset_index()
topn = 'all'
fig, axs = plt.subplots(2,2, figsize=(15,6), constrained_layout=True)

bs_axis = {'micro' : [0,0], 'small' : [0,1], 'medium_large' : [1,0], 
           'enterprise' : [1,1]}

for business_size, coords in bs_axis.items():
    if topn == 'all':
        df = dd.loc[dd['Business_size'] == business_size].set_index('Business_size').T#[current_question].value_counts(normalize=True)
    else:
        df = dd.loc[dd['Business_size'] == business_size].set_index('Business_size').T.head(topn)#[current_question].value_counts(normalize=True).head(topn)            
    bcplot = sns.barplot(x=df.index, y=df[business_size], ax=axs[coords[0],coords[1] ]);
    axs[coords[0],coords[1] ].set_title(business_size);
    for p in axs[coords[0],coords[1] ].patches:
        axs[coords[0],coords[1] ].annotate(str("{0:.0%}".format(p.get_height())), (p.get_x() * 0.97, p.get_height() * 0.97))
    for item in bcplot.get_xticklabels():
        item.set_rotation(90);
    bcplot.set(yticks=[0.1, 0.2, 0.3]);
for ax in axs.flat:
    ax.set(xlabel='', ylabel='N_of_employees')

for ax in axs.flat:
    ax.label_outer() 

#fig.delaxes(axs[1][2])


# Looks like most of DS use Python as primary language and also the work with tabluar dat (SQL). But you see that medium \ large companies have the lowest rates. But let's look at programming languages employees of each group recommend to learn first:

# In[ ]:


q = ['What programming language would you recommend an aspiring data scientist to learn first? - Selected Choice',
      'What is the size of the company where you are employed?',
 'Select the title most similar to your current role (or most recent title if retired): - Selected Choice',
]

_df2019 = df_2019[q].copy()
_df2019['Business_size'] = _df2019['What is the size of the company where you are employed?'].apply(lambda x : business_type(x) )
business_type_plot_hist(_df2019.loc[_df2019['Select the title most similar to your current role (or most recent title if retired): - Selected Choice'] == 'Data Scientist'], 
                   current_question=q[0],
                   x_title='Recommended_Language')


# ## Which automated machine learning tools (or partial AutoML tools) do you use on a regular basis?

# In[ ]:


q = [  'Which automated machine learning tools (or partial AutoML tools) do you use on a regular basis?  (Select all that apply) - Selected Choice -   Auto-Keras ',
 'Which automated machine learning tools (or partial AutoML tools) do you use on a regular basis?  (Select all that apply) - Selected Choice -   Auto-Sklearn ',
 'Which automated machine learning tools (or partial AutoML tools) do you use on a regular basis?  (Select all that apply) - Selected Choice -   Auto_ml ',
 'Which automated machine learning tools (or partial AutoML tools) do you use on a regular basis?  (Select all that apply) - Selected Choice -   MLbox ',
 'Which automated machine learning tools (or partial AutoML tools) do you use on a regular basis?  (Select all that apply) - Selected Choice -   Tpot ',
 'Which automated machine learning tools (or partial AutoML tools) do you use on a regular basis?  (Select all that apply) - Selected Choice -   Xcessiv ',
 'Which automated machine learning tools (or partial AutoML tools) do you use on a regular basis?  (Select all that apply) - Selected Choice -  DataRobot AutoML ',
 'Which automated machine learning tools (or partial AutoML tools) do you use on a regular basis?  (Select all that apply) - Selected Choice -  Databricks AutoML ',
 'Which automated machine learning tools (or partial AutoML tools) do you use on a regular basis?  (Select all that apply) - Selected Choice -  Google AutoML ',
 'Which automated machine learning tools (or partial AutoML tools) do you use on a regular basis?  (Select all that apply) - Selected Choice -  H20 Driverless AI  ',
 'Which automated machine learning tools (or partial AutoML tools) do you use on a regular basis?  (Select all that apply) - Selected Choice - None',
 'Which automated machine learning tools (or partial AutoML tools) do you use on a regular basis?  (Select all that apply) - Selected Choice - Other',
 'What is the size of the company where you are employed?',
 'Select the title most similar to your current role (or most recent title if retired): - Selected Choice',
]
_df2019 = df_2019[q].copy()
_df2019['Business_size'] = _df2019['What is the size of the company where you are employed?'].apply(lambda x : business_type(x) )

for qa in _df2019.columns[:-2]:
    _df2019[qa] = _df2019[qa].apply(lambda x: 1 if not pd.isnull(x) else 0)
    


# In[ ]:



_df2019.columns = [ 'Auto-Keras', 'Auto-Sklearn', 'Auto_ml', 'MLbox', 'Tpot', 'Xcessiv',
                   'DataRobot AutoML', 'Databricks AutoML', 'Google AutoML', 
                   'H20 Driverless AI', 'None', 'Other',
                   'What is the size of the company where you are employed?',
                  'Job Title',  'Business_size' ]

dd = _df2019.loc[_df2019['Job Title'] == 'Data Scientist'][['Auto-Keras', 'Auto-Sklearn', 
                    'Auto_ml', 'MLbox', 'Tpot', 'Xcessiv',
                   'DataRobot AutoML', 'Databricks AutoML', 'Google AutoML', 
                   'H20 Driverless AI', 'None', 'Other',  
                    'Business_size']].groupby('Business_size').sum()
dd =  dd / _df2019.loc[_df2019['Job Title'] == 'Data Scientist'].shape[0]
dd = dd.reset_index()
topn = 'all'
fig, axs = plt.subplots(2,2, figsize=(15,6), constrained_layout=True)

bs_axis = {'micro' : [0,0], 'small' : [0,1], 'medium_large' : [1,0], 
           'enterprise' : [1,1]}

for business_size, coords in bs_axis.items():
    if topn == 'all':
        df = dd.loc[dd['Business_size'] == business_size].set_index('Business_size').T#[current_question].value_counts(normalize=True)
    else:
        df = dd.loc[dd['Business_size'] == business_size].set_index('Business_size').T.head(topn)#[current_question].value_counts(normalize=True).head(topn)            
    bcplot = sns.barplot(x=df.index, y=df[business_size], ax=axs[coords[0],coords[1] ]);
    axs[coords[0],coords[1] ].set_title(business_size);
    for p in axs[coords[0],coords[1] ].patches:
        axs[coords[0],coords[1] ].annotate(str("{0:.0%}".format(p.get_height())), (p.get_x() * 0.97, p.get_height() * 0.97))
    for item in bcplot.get_xticklabels():
        item.set_rotation(90);
    bcplot.set(yticks=[0.1, 0.2]);
for ax in axs.flat:
    ax.set(xlabel='', ylabel='N_of_employees')

for ax in axs.flat:
    ax.label_outer() 

#fig.delaxes(axs[1][2])


# Most of respondents do not use Auto ML on reular basis (may be there are experiments but not production) but micro companies and enterprises look more innovative because its employees are familiar with AutoKeras and AutoSklearn.

# ##  Which categories of ML tools do you use on a regular basis?

# In[ ]:


q = [  'Which categories of ML tools do you use on a regular basis?  (Select all that apply) - Selected Choice - Automated data augmentation (e.g. imgaug, albumentations)',
 'Which categories of ML tools do you use on a regular basis?  (Select all that apply) - Selected Choice - Automated feature engineering/selection (e.g. tpot, boruta_py)',
 'Which categories of ML tools do you use on a regular basis?  (Select all that apply) - Selected Choice - Automated hyperparameter tuning (e.g. hyperopt, ray.tune)',
 'Which categories of ML tools do you use on a regular basis?  (Select all that apply) - Selected Choice - Automated model architecture searches (e.g. darts, enas)',
 'Which categories of ML tools do you use on a regular basis?  (Select all that apply) - Selected Choice - Automated model selection (e.g. auto-sklearn, xcessiv)',
 'Which categories of ML tools do you use on a regular basis?  (Select all that apply) - Selected Choice - Automation of full ML pipelines (e.g. Google AutoML, H20 Driverless AI)',
 'Which categories of ML tools do you use on a regular basis?  (Select all that apply) - Selected Choice - None',
 'Which categories of ML tools do you use on a regular basis?  (Select all that apply) - Selected Choice - Other',
 'What is the size of the company where you are employed?',
 'Select the title most similar to your current role (or most recent title if retired): - Selected Choice',
]
_df2019 = df_2019[q].copy()
_df2019['Business_size'] = _df2019['What is the size of the company where you are employed?'].apply(lambda x : business_type(x) )

for qa in _df2019.columns[:-2]:
    _df2019[qa] = _df2019[qa].apply(lambda x: 1 if not pd.isnull(x) else 0)


# In[ ]:


_df2019.columns = [ 'data augmentation', 'feature engineering/selection',
                   'hyperparameter tuning', 'model architecture searches',
                   'model selection', 'full ML pipeline', 'None', "Other",
                   'What is the size of the company where you are employed?',
                  'Job Title',  'Business_size' ]

dd = _df2019.loc[_df2019['Job Title'] == 'Data Scientist'][['data augmentation', 
                  'feature engineering/selection',
                   'hyperparameter tuning', 'model architecture searches',
                   'model selection', 'full ML pipeline',  'None', "Other",
                    'Business_size']].groupby('Business_size').sum()
dd =  dd / _df2019.loc[_df2019['Job Title'] == 'Data Scientist'].shape[0]
dd = dd.reset_index()
topn = 'all'
fig, axs = plt.subplots(2,2, figsize=(15,6), constrained_layout=True)

bs_axis = {'micro' : [0,0], 'small' : [0,1], 'medium_large' : [1,0], 
           'enterprise' : [1,1]}

for business_size, coords in bs_axis.items():
    if topn == 'all':
        df = dd.loc[dd['Business_size'] == business_size].set_index('Business_size').T#[current_question].value_counts(normalize=True)
    else:
        df = dd.loc[dd['Business_size'] == business_size].set_index('Business_size').T.head(topn)#[current_question].value_counts(normalize=True).head(topn)            
    bcplot = sns.barplot(x=df.index, y=df[business_size], ax=axs[coords[0],coords[1] ]);
    axs[coords[0],coords[1] ].set_title(business_size);
    for p in axs[coords[0],coords[1] ].patches:
        axs[coords[0],coords[1] ].annotate(str("{0:.0%}".format(p.get_height())), (p.get_x() * 0.97, p.get_height() * 0.97))
    for item in bcplot.get_xticklabels():
        item.set_rotation(90);
    bcplot.set(yticks=[0.1, 0.2]);
for ax in axs.flat:
    ax.set(xlabel='', ylabel='N_of_employees')

for ax in axs.flat:
    ax.label_outer() 

#fig.delaxes(axs[1][2])


# Most of enterprise employees never use ML tools listed, but feature engineering and model selection are actually used. Probably, it means that we have not enough data because a lot of people ignored the answer. Micro companies look alike enterprise in sense of tools usage. By the way, We have another proof that medium and large companies work in the industries in which Data science is not used now or there are no rules and best practices. 

# ## Which of the following ML algorithms do you use on a regular basis?

# In[ ]:


q = [  'Which of the following ML algorithms do you use on a regular basis? (Select all that apply): - Selected Choice - Bayesian Approaches',
 'Which of the following ML algorithms do you use on a regular basis? (Select all that apply): - Selected Choice - Convolutional Neural Networks',
 'Which of the following ML algorithms do you use on a regular basis? (Select all that apply): - Selected Choice - Decision Trees or Random Forests',
 'Which of the following ML algorithms do you use on a regular basis? (Select all that apply): - Selected Choice - Dense Neural Networks (MLPs, etc)',
 'Which of the following ML algorithms do you use on a regular basis? (Select all that apply): - Selected Choice - Evolutionary Approaches',
 'Which of the following ML algorithms do you use on a regular basis? (Select all that apply): - Selected Choice - Generative Adversarial Networks',
 'Which of the following ML algorithms do you use on a regular basis? (Select all that apply): - Selected Choice - Gradient Boosting Machines (xgboost, lightgbm, etc)',
 'Which of the following ML algorithms do you use on a regular basis? (Select all that apply): - Selected Choice - Linear or Logistic Regression',
 'Which of the following ML algorithms do you use on a regular basis? (Select all that apply): - Selected Choice - None',
 'Which of the following ML algorithms do you use on a regular basis? (Select all that apply): - Selected Choice - Other',
 'Which of the following ML algorithms do you use on a regular basis? (Select all that apply): - Selected Choice - Recurrent Neural Networks',
 'Which of the following ML algorithms do you use on a regular basis? (Select all that apply): - Selected Choice - Transformer Networks (BERT, gpt-2, etc)',
     'What is the size of the company where you are employed?',
 'Select the title most similar to your current role (or most recent title if retired): - Selected Choice',
]
_df2019 = df_2019[q].copy()
_df2019['Business_size'] = _df2019['What is the size of the company where you are employed?'].apply(lambda x : business_type(x) )

for qa in _df2019.columns[:-2]:
    _df2019[qa] = _df2019[qa].apply(lambda x: 1 if not pd.isnull(x) else 0)    


# In[ ]:


_df2019.columns = ['Bayesian Approaches',
                   'Convolutional Neural Networks',
                   'Decision Trees or Random Forests',
                   'Dense Neural Networks (MLPs, etc)',
                   'Evolutionary Approaches',
                   'Generative Adversarial Networks',
                   'Gradient Boosting Machines',
                   'Linear or Logistic Regression',
                   'None', 'Other',
                   'Recurrent Neural Networks',
                   'Transformer Networks',
                   'What is the size of the company where you are employed?',
                  'Job Title',  'Business_size' ]

dd = _df2019.loc[_df2019['Job Title'] == 'Data Scientist'][['Bayesian Approaches',
                   'Convolutional Neural Networks',
                   'Decision Trees or Random Forests',
                   'Dense Neural Networks (MLPs, etc)',
                   'Evolutionary Approaches',
                   'Generative Adversarial Networks',
                   'Gradient Boosting Machines',
                   'Linear or Logistic Regression',
                   'None', 'Other',
                   'Recurrent Neural Networks',
                   'Transformer Networks',
                    'Business_size']].groupby('Business_size').sum()
dd =  dd / _df2019.loc[_df2019['Job Title'] == 'Data Scientist'].shape[0]
dd = dd.reset_index()
topn = 'all'
fig, axs = plt.subplots(2,2, figsize=(15,6), constrained_layout=True)

bs_axis = {'micro' : [0,0], 'small' : [0,1], 'medium_large' : [1,0], 
           'enterprise' : [1,1]}

for business_size, coords in bs_axis.items():
    if topn == 'all':
        df = dd.loc[dd['Business_size'] == business_size].set_index('Business_size').T#[current_question].value_counts(normalize=True)
    else:
        df = dd.loc[dd['Business_size'] == business_size].set_index('Business_size').T.head(topn)#[current_question].value_counts(normalize=True).head(topn)            
    bcplot = sns.barplot(x=df.index, y=df[business_size], ax=axs[coords[0],coords[1] ]);
    axs[coords[0],coords[1] ].set_title(business_size);
    for p in axs[coords[0],coords[1] ].patches:
        axs[coords[0],coords[1] ].annotate(str("{0:.0%}".format(p.get_height())), (p.get_x() * 0.97, p.get_height() * 0.97))
    for item in bcplot.get_xticklabels():
        item.set_rotation(90);
    bcplot.set(yticks=[0.1, 0.2, 0.3]);
for ax in axs.flat:
    ax.set(xlabel='', ylabel='N_of_employees')

for ax in axs.flat:
    ax.label_outer() 

#fig.delaxes(axs[1][2])


# Micro, small and enterprise companies use traditional ML algos and exploring neural networks. Medium \ large provide very poor statistics.

# ## Which of the following cloud computing platforms do you use on a regular basis?

# In[ ]:


q = ['Which of the following cloud computing platforms do you use on a regular basis? (Select all that apply) - Selected Choice -  Alibaba Cloud ',
 'Which of the following cloud computing platforms do you use on a regular basis? (Select all that apply) - Selected Choice -  Amazon Web Services (AWS) ',
 'Which of the following cloud computing platforms do you use on a regular basis? (Select all that apply) - Selected Choice -  Google Cloud Platform (GCP) ',
 'Which of the following cloud computing platforms do you use on a regular basis? (Select all that apply) - Selected Choice -  IBM Cloud ',
 'Which of the following cloud computing platforms do you use on a regular basis? (Select all that apply) - Selected Choice -  Microsoft Azure ',
 'Which of the following cloud computing platforms do you use on a regular basis? (Select all that apply) - Selected Choice -  Oracle Cloud ',
 'Which of the following cloud computing platforms do you use on a regular basis? (Select all that apply) - Selected Choice -  Red Hat Cloud ',
 'Which of the following cloud computing platforms do you use on a regular basis? (Select all that apply) - Selected Choice -  SAP Cloud ',
 'Which of the following cloud computing platforms do you use on a regular basis? (Select all that apply) - Selected Choice -  Salesforce Cloud ',
 'Which of the following cloud computing platforms do you use on a regular basis? (Select all that apply) - Selected Choice -  VMware Cloud ',
 'Which of the following cloud computing platforms do you use on a regular basis? (Select all that apply) - Selected Choice - None',
 'Which of the following cloud computing platforms do you use on a regular basis? (Select all that apply) - Selected Choice - Other',
      'What is the size of the company where you are employed?',
 'Select the title most similar to your current role (or most recent title if retired): - Selected Choice',
]
_df2019 = df_2019[q].copy()
_df2019['Business_size'] = _df2019['What is the size of the company where you are employed?'].apply(lambda x : business_type(x) )

for qa in _df2019.columns[:-2]:
    _df2019[qa] = _df2019[qa].apply(lambda x: 1 if not pd.isnull(x) else 0)  


# In[ ]:


_df2019.columns = ['Alibaba Cloud', 'Amazon Web Services', 'Google Cloud Platform',
                   'IBM Cloud', 'Microsoft Azure', 'Oracle Cloud', 'Red Hat Cloud',
                   'SAP Cloud', 'Salesforce Cloud', 'VMware Cloud', 'None', 'Other',
                   'What is the size of the company where you are employed?',
                  'Job Title',  'Business_size' ]

dd = _df2019.loc[_df2019['Job Title'] == 'Data Scientist'][['Alibaba Cloud',
                    'Amazon Web Services', 'Google Cloud Platform',
                   'IBM Cloud', 'Microsoft Azure', 'Oracle Cloud', 'Red Hat Cloud',
                   'SAP Cloud', 'Salesforce Cloud', 'VMware Cloud', 'None', 'Other',
                    'Business_size']].groupby('Business_size').sum()
dd =  dd / _df2019.loc[_df2019['Job Title'] == 'Data Scientist'].shape[0]
dd = dd.reset_index()
topn = 'all'
fig, axs = plt.subplots(2,2, figsize=(15,6), constrained_layout=True)

bs_axis = {'micro' : [0,0], 'small' : [0,1], 'medium_large' : [1,0], 
           'enterprise' : [1,1]}
for business_size, coords in bs_axis.items():
    if topn == 'all':
        df = dd.loc[dd['Business_size'] == business_size].set_index('Business_size').T#[current_question].value_counts(normalize=True)
    else:
        df = dd.loc[dd['Business_size'] == business_size].set_index('Business_size').T.head(topn)#[current_question].value_counts(normalize=True).head(topn)            
    bcplot = sns.barplot(x=df.index, y=df[business_size], ax=axs[coords[0],coords[1] ]);
    axs[coords[0],coords[1] ].set_title(business_size);
    for p in axs[coords[0],coords[1] ].patches:
        axs[coords[0],coords[1] ].annotate(str("{0:.0%}".format(p.get_height())), (p.get_x() * 0.97, p.get_height() * 0.97))
    for item in bcplot.get_xticklabels():
        item.set_rotation(90);
    bcplot.set(yticks=[0.1, 0.2, 0.3]);
for ax in axs.flat:
    ax.set(xlabel='', ylabel='N_of_employees')

for ax in axs.flat:
    ax.label_outer() 

#fig.delaxes(axs[1][2])


# Amazon and Google are leaders of cloud services. And small and medium\large companies are not interested in clouds.

# ## Which of the following hosted notebook products do you use on a regular basis?

# In[ ]:


q = [ 'Which of the following hosted notebook products do you use on a regular basis?  (Select all that apply) - Selected Choice -  Binder / JupyterHub ',
 'Which of the following hosted notebook products do you use on a regular basis?  (Select all that apply) - Selected Choice -  FloydHub ',
 'Which of the following hosted notebook products do you use on a regular basis?  (Select all that apply) - Selected Choice -  Google Cloud Notebook Products (AI Platform, Datalab, etc) ',
 'Which of the following hosted notebook products do you use on a regular basis?  (Select all that apply) - Selected Choice -  Google Colab ',
 'Which of the following hosted notebook products do you use on a regular basis?  (Select all that apply) - Selected Choice -  IBM Watson Studio ',
 'Which of the following hosted notebook products do you use on a regular basis?  (Select all that apply) - Selected Choice -  Kaggle Notebooks (Kernels) ',
 'Which of the following hosted notebook products do you use on a regular basis?  (Select all that apply) - Selected Choice -  Microsoft Azure Notebooks ',
 'Which of the following hosted notebook products do you use on a regular basis?  (Select all that apply) - Selected Choice -  Paperspace / Gradient ',
 'Which of the following hosted notebook products do you use on a regular basis?  (Select all that apply) - Selected Choice - AWS Notebook Products (EMR Notebooks, Sagemaker Notebooks, etc) ',
 'Which of the following hosted notebook products do you use on a regular basis?  (Select all that apply) - Selected Choice - Code Ocean ',
 'Which of the following hosted notebook products do you use on a regular basis?  (Select all that apply) - Selected Choice - None',
 'Which of the following hosted notebook products do you use on a regular basis?  (Select all that apply) - Selected Choice - Other',
      'What is the size of the company where you are employed?',
 'Select the title most similar to your current role (or most recent title if retired): - Selected Choice',
]
_df2019 = df_2019[q].copy()
_df2019['Business_size'] = _df2019['What is the size of the company where you are employed?'].apply(lambda x : business_type(x) )

for qa in _df2019.columns[:-2]:
    _df2019[qa] = _df2019[qa].apply(lambda x: 1 if not pd.isnull(x) else 0)  


# In[ ]:


_df2019.columns = ['Binder / JupyterHub', 'FloydHub', ' Google Cloud Notebook Products',
                   'Google Colab', 'IBM Watson Studio', 'Kaggle Notebooks (Kernels)',
                   'Microsoft Azure Notebooks', 'Paperspace / Gradient', 'AWS Notebook Products',
                   'Code Ocean', 'None', 'Other',
                   'What is the size of the company where you are employed?',
                  'Job Title',  'Business_size' ]

dd = _df2019.loc[_df2019['Job Title'] == 'Data Scientist'][['Binder / JupyterHub', 
                    'FloydHub', ' Google Cloud Notebook Products',
                   'Google Colab', 'IBM Watson Studio', 'Kaggle Notebooks (Kernels)',
                   'Microsoft Azure Notebooks', 'Paperspace / Gradient', 'AWS Notebook Products',
                   'Code Ocean', 'None', 'Other',
                    'Business_size']].groupby('Business_size').sum()
dd =  dd / _df2019.loc[_df2019['Job Title'] == 'Data Scientist'].shape[0]
dd = dd.reset_index()
topn = 'all'
fig, axs = plt.subplots(2,2, figsize=(15,6), constrained_layout=True)

bs_axis = {'micro' : [0,0], 'small' : [0,1], 'medium_large' : [1,0], 
           'enterprise' : [1,1]}

for business_size, coords in bs_axis.items():
    if topn == 'all':
        df = dd.loc[dd['Business_size'] == business_size].set_index('Business_size').T#[current_question].value_counts(normalize=True)
    else:
        df = dd.loc[dd['Business_size'] == business_size].set_index('Business_size').T.head(topn)#[current_question].value_counts(normalize=True).head(topn)            
    bcplot = sns.barplot(x=df.index, y=df[business_size], ax=axs[coords[0],coords[1] ]);
    axs[coords[0],coords[1] ].set_title(business_size);
    for p in axs[coords[0],coords[1] ].patches:
        axs[coords[0],coords[1] ].annotate(str("{0:.0%}".format(p.get_height())), (p.get_x() * 0.97, p.get_height() * 0.97))
    for item in bcplot.get_xticklabels():
        item.set_rotation(90);
    bcplot.set(yticks=[0.1, 0.2, 0.3]);
for ax in axs.flat:
    ax.set(xlabel='', ylabel='N_of_employees')

for ax in axs.flat:
    ax.label_outer() 

#fig.delaxes(axs[1][2])


# Well, looks like there is an intersection of work and personal use-cases. So, Kaggle notebooks is a popular tool for personal research projects and Google Collab probably used in working environment. 

# ## Which of the following integrated development environments (IDE's) do you use on a regular basis?

# In[ ]:


q = [ 
 "Which of the following integrated development environments (IDE's) do you use on a regular basis?  (Select all that apply) - Selected Choice -   Notepad++  ",
 "Which of the following integrated development environments (IDE's) do you use on a regular basis?  (Select all that apply) - Selected Choice -   Spyder  ",
 "Which of the following integrated development environments (IDE's) do you use on a regular basis?  (Select all that apply) - Selected Choice -   Sublime Text  ",
 "Which of the following integrated development environments (IDE's) do you use on a regular basis?  (Select all that apply) - Selected Choice -   Vim / Emacs  ",
 "Which of the following integrated development environments (IDE's) do you use on a regular basis?  (Select all that apply) - Selected Choice -  Atom ",
 "Which of the following integrated development environments (IDE's) do you use on a regular basis?  (Select all that apply) - Selected Choice -  MATLAB ",
 "Which of the following integrated development environments (IDE's) do you use on a regular basis?  (Select all that apply) - Selected Choice -  PyCharm ",
 "Which of the following integrated development environments (IDE's) do you use on a regular basis?  (Select all that apply) - Selected Choice -  RStudio ",
 "Which of the following integrated development environments (IDE's) do you use on a regular basis?  (Select all that apply) - Selected Choice -  Visual Studio / Visual Studio Code ",
 "Which of the following integrated development environments (IDE's) do you use on a regular basis?  (Select all that apply) - Selected Choice - Jupyter (JupyterLab, Jupyter Notebooks, etc) ",
 "Which of the following integrated development environments (IDE's) do you use on a regular basis?  (Select all that apply) - Selected Choice - None",
 "Which of the following integrated development environments (IDE's) do you use on a regular basis?  (Select all that apply) - Selected Choice - Other",
 'What is the size of the company where you are employed?',
 'Select the title most similar to your current role (or most recent title if retired): - Selected Choice',
]
_df2019 = df_2019[q].copy()
_df2019['Business_size'] = _df2019['What is the size of the company where you are employed?'].apply(lambda x : business_type(x) )

for qa in _df2019.columns[:-2]:
    _df2019[qa] = _df2019[qa].apply(lambda x: 1 if not pd.isnull(x) else 0)  


# In[ ]:


_df2019.columns = ['Notepad++', 'Spyder', 'Sublime Text', 'Vim / Emacs', 'Atom', 'MATLAB', 'PyCharm',
                   'RStudio', 'Visual Studio / Visual Studio Code', 'Jupyter', 'None', 'Other',
                   'What is the size of the company where you are employed?',
                  'Job Title',  'Business_size' ]

dd = _df2019.loc[_df2019['Job Title'] == 'Data Scientist'][['Notepad++', 'Spyder', 'Sublime Text', 
                    'Vim / Emacs', 'Atom', 'MATLAB', 'PyCharm',
                   'RStudio', 'Visual Studio / Visual Studio Code', 'Jupyter', 'None', 'Other',
                    'Business_size']].groupby('Business_size').sum()
dd =  dd / _df2019.loc[_df2019['Job Title'] == 'Data Scientist'].shape[0]
dd = dd.reset_index()
topn = 'all'
fig, axs = plt.subplots(2,2, figsize=(15,6), constrained_layout=True)

bs_axis = {'micro' : [0,0], 'small' : [0,1], 'medium_large' : [1,0], 
           'enterprise' : [1,1]}

for business_size, coords in bs_axis.items():
    if topn == 'all':
        df = dd.loc[dd['Business_size'] == business_size].set_index('Business_size').T#[current_question].value_counts(normalize=True)
    else:
        df = dd.loc[dd['Business_size'] == business_size].set_index('Business_size').T.head(topn)#[current_question].value_counts(normalize=True).head(topn)            
    bcplot = sns.barplot(x=df.index, y=df[business_size], ax=axs[coords[0],coords[1] ]);
    axs[coords[0],coords[1] ].set_title(business_size);
    for p in axs[coords[0],coords[1] ].patches:
        axs[coords[0],coords[1] ].annotate(str("{0:.0%}".format(p.get_height())), (p.get_x() * 0.97, p.get_height() * 0.97))
    for item in bcplot.get_xticklabels():
        item.set_rotation(90);
    bcplot.set(yticks=[0.1, 0.2, 0.3, 0.4]);
for ax in axs.flat:
    ax.set(xlabel='', ylabel='N_of_employees')

for ax in axs.flat:
    ax.label_outer() 

#fig.delaxes(axs[1][2])


# Jupyter is the leader of IDE's but enterprise employees also use PyCharm as an alternative option for Python and, more interesting, the also use RStudio (well we accidently found where the R language has chances)

# ## Which of the following machine learning frameworks do you use on a regular basis?

# In[ ]:


q = [  'Which of the following machine learning frameworks do you use on a regular basis? (Select all that apply) - Selected Choice -   Scikit-learn ',
 'Which of the following machine learning frameworks do you use on a regular basis? (Select all that apply) - Selected Choice -   TensorFlow ',
 'Which of the following machine learning frameworks do you use on a regular basis? (Select all that apply) - Selected Choice -  Caret ',
 'Which of the following machine learning frameworks do you use on a regular basis? (Select all that apply) - Selected Choice -  Fast.ai ',
 'Which of the following machine learning frameworks do you use on a regular basis? (Select all that apply) - Selected Choice -  Keras ',
 'Which of the following machine learning frameworks do you use on a regular basis? (Select all that apply) - Selected Choice -  LightGBM ',
 'Which of the following machine learning frameworks do you use on a regular basis? (Select all that apply) - Selected Choice -  PyTorch ',
 'Which of the following machine learning frameworks do you use on a regular basis? (Select all that apply) - Selected Choice -  RandomForest',
 'Which of the following machine learning frameworks do you use on a regular basis? (Select all that apply) - Selected Choice -  Spark MLib ',
 'Which of the following machine learning frameworks do you use on a regular basis? (Select all that apply) - Selected Choice -  Xgboost ',
 'Which of the following machine learning frameworks do you use on a regular basis? (Select all that apply) - Selected Choice - None',
 'Which of the following machine learning frameworks do you use on a regular basis? (Select all that apply) - Selected Choice - Other',
'What is the size of the company where you are employed?',
 'Select the title most similar to your current role (or most recent title if retired): - Selected Choice',
]
_df2019 = df_2019[q].copy()
_df2019['Business_size'] = _df2019['What is the size of the company where you are employed?'].apply(lambda x : business_type(x) )

for qa in _df2019.columns[:-2]:
    _df2019[qa] = _df2019[qa].apply(lambda x: 1 if not pd.isnull(x) else 0)  


# In[ ]:


_df2019.columns = ['Scikit-learn', 'TensorFlow', 'Caret', 'Fast.ai', 'Keras',
                   'LightGBM', 'PyTorch', 'RandomForest', 'Spark MLib', 'Xgboost',
                    'None', 'Other',
                   'What is the size of the company where you are employed?',
                  'Job Title',  'Business_size' ]

dd = _df2019.loc[_df2019['Job Title'] == 'Data Scientist'][['Scikit-learn', 'TensorFlow', 'Caret', 'Fast.ai', 'Keras',
                   'LightGBM', 'PyTorch', 'RandomForest', 'Spark MLib', 'Xgboost', 'None', 'Other',
                    'Business_size']].groupby('Business_size').sum()
dd =  dd / _df2019.loc[_df2019['Job Title'] == 'Data Scientist'].shape[0]
dd = dd.reset_index()
topn = 'all'
fig, axs = plt.subplots(2,2, figsize=(15,6), constrained_layout=True)

bs_axis = {'micro' : [0,0], 'small' : [0,1], 'medium_large' : [1,0], 
           'enterprise' : [1,1]}

for business_size, coords in bs_axis.items():
    if topn == 'all':
        df = dd.loc[dd['Business_size'] == business_size].set_index('Business_size').T#[current_question].value_counts(normalize=True)
    else:
        df = dd.loc[dd['Business_size'] == business_size].set_index('Business_size').T.head(topn)#[current_question].value_counts(normalize=True).head(topn)            
    bcplot = sns.barplot(x=df.index, y=df[business_size], ax=axs[coords[0],coords[1] ]);
    axs[coords[0],coords[1] ].set_title(business_size);
    for p in axs[coords[0],coords[1] ].patches:
        axs[coords[0],coords[1] ].annotate(str("{0:.0%}".format(p.get_height())), (p.get_x() * 0.97, p.get_height() * 0.97))
    for item in bcplot.get_xticklabels():
        item.set_rotation(90);
    bcplot.set(yticks=[0.1, 0.2, 0.3]);
for ax in axs.flat:
    ax.set(xlabel='', ylabel='N_of_employees')

for ax in axs.flat:
    ax.label_outer() 

#fig.delaxes(axs[1][2])


# Scikit-learn and Keras are popular in micro and enterprise companies. ; Tensorlow, XGboost, Random forest also ploy an important role.  Enterprise employees emonstrate an interest in LightGBM. 

# ## Databases

# ### Which of the following relational database products do you use on a regular basis?

# In[ ]:


q = [   'Which of the following relational database products do you use on a regular basis? (Select all that apply) - Selected Choice - AWS DynamoDB',
 'Which of the following relational database products do you use on a regular basis? (Select all that apply) - Selected Choice - AWS Relational Database Service',
 'Which of the following relational database products do you use on a regular basis? (Select all that apply) - Selected Choice - Azure SQL Database',
 'Which of the following relational database products do you use on a regular basis? (Select all that apply) - Selected Choice - Google Cloud SQL',
 'Which of the following relational database products do you use on a regular basis? (Select all that apply) - Selected Choice - Microsoft Access',
 'Which of the following relational database products do you use on a regular basis? (Select all that apply) - Selected Choice - Microsoft SQL Server',
 'Which of the following relational database products do you use on a regular basis? (Select all that apply) - Selected Choice - MySQL',
 'Which of the following relational database products do you use on a regular basis? (Select all that apply) - Selected Choice - None',
 'Which of the following relational database products do you use on a regular basis? (Select all that apply) - Selected Choice - Oracle Database',
 'Which of the following relational database products do you use on a regular basis? (Select all that apply) - Selected Choice - Other',
 'Which of the following relational database products do you use on a regular basis? (Select all that apply) - Selected Choice - PostgresSQL',
 'Which of the following relational database products do you use on a regular basis? (Select all that apply) - Selected Choice - SQLite',
 'What is the size of the company where you are employed?',
 'Select the title most similar to your current role (or most recent title if retired): - Selected Choice',
]
_df2019 = df_2019[q].copy()
_df2019['Business_size'] = _df2019['What is the size of the company where you are employed?'].apply(lambda x : business_type(x) )

for qa in _df2019.columns[:-2]:
    _df2019[qa] = _df2019[qa].apply(lambda x: 1 if not pd.isnull(x) else 0)  


# In[ ]:


_df2019.columns = ['WS DynamoDB', 'AWS Relational Database Service', 'Azure SQL Database', 'Google Cloud SQL',
                   'Microsoft Access', 'Microsoft SQL Server', 'MySQL', 'None', 'Oracle Database', 'Other',            
                    'PostgresSQL', 'SQLite',
                   'What is the size of the company where you are employed?',
                  'Job Title',  'Business_size' ]

dd = _df2019.loc[_df2019['Job Title'] == 'Data Scientist'][['WS DynamoDB', 'AWS Relational Database Service', 
                        'Azure SQL Database', 'Google Cloud SQL',
                   'Microsoft Access', 'Microsoft SQL Server', 'MySQL', 'None', 'Oracle Database', 'Other',            
                    'PostgresSQL', 'SQLite',
                    'Business_size']].groupby('Business_size').sum()
dd =  dd / _df2019.loc[_df2019['Job Title'] == 'Data Scientist'].shape[0]
dd = dd.reset_index()
topn = 'all'
fig, axs = plt.subplots(2,2, figsize=(15,6), constrained_layout=True)

bs_axis = {'micro' : [0,0], 'small' : [0,1], 'medium_large' : [1,0], 
           'enterprise' : [1,1]}

for business_size, coords in bs_axis.items():
    if topn == 'all':
        df = dd.loc[dd['Business_size'] == business_size].set_index('Business_size').T#[current_question].value_counts(normalize=True)
    else:
        df = dd.loc[dd['Business_size'] == business_size].set_index('Business_size').T.head(topn)#[current_question].value_counts(normalize=True).head(topn)            
    bcplot = sns.barplot(x=df.index, y=df[business_size], ax=axs[coords[0],coords[1] ]);
    axs[coords[0],coords[1] ].set_title(business_size);
    for p in axs[coords[0],coords[1] ].patches:
        axs[coords[0],coords[1] ].annotate(str("{0:.0%}".format(p.get_height())), (p.get_x() * 0.97, p.get_height() * 0.97))
    for item in bcplot.get_xticklabels():
        item.set_rotation(90);
    bcplot.set(yticks=[0.1, 0.2]);
for ax in axs.flat:
    ax.set(xlabel='', ylabel='N_of_employees')

for ax in axs.flat:
    ax.label_outer() 

#fig.delaxes(axs[1][2])


# MySQL is a leader in both micro and enterprise segments; the key difference heere is Oracle Database which is used ny enterprise employees (probably at work).

# ## Who/what are your favorite media sources that report on data science topics?

# In[ ]:


q = [ 'Who/what are your favorite media sources that report on data science topics? (Select all that apply) - Selected Choice - Blogs (Towards Data Science, Medium, Analytics Vidhya, KDnuggets etc)',
'Who/what are your favorite media sources that report on data science topics? (Select all that apply) - Selected Choice - Course Forums (forums.fast.ai, etc)',
'Who/what are your favorite media sources that report on data science topics? (Select all that apply) - Selected Choice - Hacker News (https://news.ycombinator.com/)',
'Who/what are your favorite media sources that report on data science topics? (Select all that apply) - Selected Choice - Journal Publications (traditional publications, preprint journals, etc)',
'Who/what are your favorite media sources that report on data science topics? (Select all that apply) - Selected Choice - Kaggle (forums, blog, social media, etc)',
'Who/what are your favorite media sources that report on data science topics? (Select all that apply) - Selected Choice - None',
'Who/what are your favorite media sources that report on data science topics? (Select all that apply) - Selected Choice - Other',
'Who/what are your favorite media sources that report on data science topics? (Select all that apply) - Selected Choice - Podcasts (Chai Time Data Science, Linear Digressions, etc)',
'Who/what are your favorite media sources that report on data science topics? (Select all that apply) - Selected Choice - Reddit (r/machinelearning, r/datascience, etc)',
'Who/what are your favorite media sources that report on data science topics? (Select all that apply) - Selected Choice - Slack Communities (ods.ai, kagglenoobs, etc)',
'Who/what are your favorite media sources that report on data science topics? (Select all that apply) - Selected Choice - Twitter (data science influencers)',
'Who/what are your favorite media sources that report on data science topics? (Select all that apply) - Selected Choice - YouTube (Cloud AI Adventures, Siraj Raval, etc)',
     'What is the size of the company where you are employed?',
'Select the title most similar to your current role (or most recent title if retired): - Selected Choice',
]
_df2019 = df_2019[q].copy()
_df2019['Business_size'] = _df2019['What is the size of the company where you are employed?'].apply(lambda x : business_type(x) )

for qa in _df2019.columns[:-2]:
   _df2019[qa] = _df2019[qa].apply(lambda x: 1 if not pd.isnull(x) else 0)  


# In[ ]:


_df2019.columns = ['Blogs', 'Course forums', 'Hacker News', 'Journal Publications', 'Kaggle',
                   'None', 'Other',  'Podcasts',  'Reddit', 'Slack', 'Twitter', 'YouTube',
                   'What is the size of the company where you are employed?',
                  'Job Title',  'Business_size' ]

dd = _df2019.loc[_df2019['Job Title'] == 'Data Scientist'][['Blogs', 'Course forums', 'Hacker News', 'Journal Publications', 'Kaggle',
                   'None', 'Other',  'Podcasts',  'Reddit', 'Slack', 'Twitter', 'YouTube',
                    'Business_size']].groupby('Business_size').sum()
dd =  dd / _df2019.loc[_df2019['Job Title'] == 'Data Scientist'].shape[0]
dd = dd.reset_index()
topn = 'all'
fig, axs = plt.subplots(2,2, figsize=(15,6), constrained_layout=True)

bs_axis = {'micro' : [0,0], 'small' : [0,1], 'medium_large' : [1,0], 
           'enterprise' : [1,1]}

for business_size, coords in bs_axis.items():
    if topn == 'all':
        df = dd.loc[dd['Business_size'] == business_size].set_index('Business_size').T#[current_question].value_counts(normalize=True)
    else:
        df = dd.loc[dd['Business_size'] == business_size].set_index('Business_size').T.head(topn)#[current_question].value_counts(normalize=True).head(topn)            
    bcplot = sns.barplot(x=df.index, y=df[business_size], ax=axs[coords[0],coords[1] ]);
    axs[coords[0],coords[1] ].set_title(business_size);
    for p in axs[coords[0],coords[1] ].patches:
        axs[coords[0],coords[1] ].annotate(str("{0:.0%}".format(p.get_height())), (p.get_x() * 0.97, p.get_height() * 0.97))
    for item in bcplot.get_xticklabels():
        item.set_rotation(90);
    bcplot.set(yticks=[0.1, 0.2]);
for ax in axs.flat:
    ax.set(xlabel='', ylabel='N_of_employees')

for ax in axs.flat:
    ax.label_outer() 

#fig.delaxes(axs[1][2])


# Distributions for micro, small, medium \ large and enterprise empliyees are very similar. The most popular media are Kaggle and blogs.

# ## ML Applications: 

# ### Which categories of computer vision methods do you use on a regular basis?

# In[ ]:


q = [ 'Which categories of computer vision methods do you use on a regular basis?  (Select all that apply) - Selected Choice - General purpose image/video tools (PIL, cv2, skimage, etc)',
 'Which categories of computer vision methods do you use on a regular basis?  (Select all that apply) - Selected Choice - Generative Networks (GAN, VAE, etc)',
 'Which categories of computer vision methods do you use on a regular basis?  (Select all that apply) - Selected Choice - Image classification and other general purpose networks (VGG, Inception, ResNet, ResNeXt, NASNet, EfficientNet, etc)',
 'Which categories of computer vision methods do you use on a regular basis?  (Select all that apply) - Selected Choice - Image segmentation methods (U-Net, Mask R-CNN, etc)',
 'Which categories of computer vision methods do you use on a regular basis?  (Select all that apply) - Selected Choice - None',
 'Which categories of computer vision methods do you use on a regular basis?  (Select all that apply) - Selected Choice - Object detection methods (YOLOv3, RetinaNet, etc)',
 'Which categories of computer vision methods do you use on a regular basis?  (Select all that apply) - Selected Choice - Other',
 'What is the size of the company where you are employed?',
 'Select the title most similar to your current role (or most recent title if retired): - Selected Choice',
]
_df2019 = df_2019[q].copy()
_df2019['Business_size'] = _df2019['What is the size of the company where you are employed?'].apply(lambda x : business_type(x) )

for qa in _df2019.columns[:-2]:
    _df2019[qa] = _df2019[qa].apply(lambda x: 1 if not pd.isnull(x) else 0)    


# In[ ]:


_df2019.columns = ['General purpose image/video tools \n (PIL, cv2, skimage, etc)',
                   'Generative Networks \n (GAN, VAE, etc)',
                   'Image classification \n (VGG, Inception, ResNet, etc)',
                   'Image segmentation methods \n (U-Net, Mask R-CNN, etc)',
                   'None',
                   'Object detection methods \n (YOLOv3, RetinaNet, etc)',
                   'Other',
                   'What is the size of the company where you are employed?',
                  'Job Title',  'Business_size' ]

dd = _df2019.loc[_df2019['Job Title'] == 'Data Scientist'][['General purpose image/video tools \n (PIL, cv2, skimage, etc)',
                   'Generative Networks \n (GAN, VAE, etc)',
                   'Image classification \n (VGG, Inception, ResNet, etc)',
                   'Image segmentation methods \n (U-Net, Mask R-CNN, etc)',
                   'None',
                   'Object detection methods \n (YOLOv3, RetinaNet, etc)',
                   'Other',
                    'Business_size']].groupby('Business_size').sum()
dd =  dd / _df2019.loc[_df2019['Job Title'] == 'Data Scientist'].shape[0]
dd = dd.reset_index()
topn = 'all'
fig, axs = plt.subplots(2,2, figsize=(15,6), constrained_layout=True)

bs_axis = {'micro' : [0,0], 'small' : [0,1], 'medium_large' : [1,0], 
           'enterprise' : [1,1]}

for business_size, coords in bs_axis.items():
    if topn == 'all':
        df = dd.loc[dd['Business_size'] == business_size].set_index('Business_size').T#[current_question].value_counts(normalize=True)
    else:
        df = dd.loc[dd['Business_size'] == business_size].set_index('Business_size').T.head(topn)#[current_question].value_counts(normalize=True).head(topn)            
    bcplot = sns.barplot(x=df.index, y=df[business_size], ax=axs[coords[0],coords[1] ]);
    axs[coords[0],coords[1] ].set_title(business_size);
    for p in axs[coords[0],coords[1] ].patches:
        axs[coords[0],coords[1] ].annotate(str("{0:.0%}".format(p.get_height())), (p.get_x() * 0.97, p.get_height() * 0.97))
    for item in bcplot.get_xticklabels():
        item.set_rotation(90);
    bcplot.set(yticks=[0.1, 0.2, 0.3]);
for ax in axs.flat:
    ax.set(xlabel='', ylabel='N_of_employees')

for ax in axs.flat:
    ax.label_outer() 

#fig.delaxes(axs[1][2])


# Computer vision methods usage is considered as an additional question and image classification is the most popular task in micro and enterprise companies. Small and medium \ large probably do not solve computer vision tasks at all. 

# ### Which of the following natural language processing (NLP) methods do you use on a regular basis? 

# In[ ]:


q = [  'Which of the following natural language processing (NLP) methods do you use on a regular basis?  (Select all that apply) - Selected Choice - Contextualized embeddings (ELMo, CoVe)',
 'Which of the following natural language processing (NLP) methods do you use on a regular basis?  (Select all that apply) - Selected Choice - Encoder-decorder models (seq2seq, vanilla transformers)',
 'Which of the following natural language processing (NLP) methods do you use on a regular basis?  (Select all that apply) - Selected Choice - None',
 'Which of the following natural language processing (NLP) methods do you use on a regular basis?  (Select all that apply) - Selected Choice - Other',
 'Which of the following natural language processing (NLP) methods do you use on a regular basis?  (Select all that apply) - Selected Choice - Transformer language models (GPT-2, BERT, XLnet, etc)',
 'Which of the following natural language processing (NLP) methods do you use on a regular basis?  (Select all that apply) - Selected Choice - Word embeddings/vectors (GLoVe, fastText, word2vec)',
 'What is the size of the company where you are employed?',
 'Select the title most similar to your current role (or most recent title if retired): - Selected Choice',
]
_df2019 = df_2019[q].copy()
_df2019['Business_size'] = _df2019['What is the size of the company where you are employed?'].apply(lambda x : business_type(x) )

for qa in _df2019.columns[:-2]:
    _df2019[qa] = _df2019[qa].apply(lambda x: 1 if not pd.isnull(x) else 0)    


# In[ ]:


_df2019.columns = ['Contextualized embeddings', 'Encoder-decorder models',
                   'None', 'Other', 'Transformer language models', 'Word embeddings/vectors',
                   'What is the size of the company where you are employed?',
                  'Job Title',  'Business_size' ]

dd = _df2019.loc[_df2019['Job Title'] == 'Data Scientist'][['Contextualized embeddings', 'Encoder-decorder models',
                   'None', 'Other', 'Transformer language models', 'Word embeddings/vectors',
                    'Business_size']].groupby('Business_size').sum()
dd =  dd / _df2019.loc[_df2019['Job Title'] == 'Data Scientist'].shape[0]
dd = dd.reset_index()
topn = 'all'
fig, axs = plt.subplots(2,2, figsize=(15,6), constrained_layout=True)

bs_axis = {'micro' : [0,0], 'small' : [0,1], 'medium_large' : [1,0], 
           'enterprise' : [1,1]}

for business_size, coords in bs_axis.items():
    if topn == 'all':
        df = dd.loc[dd['Business_size'] == business_size].set_index('Business_size').T#[current_question].value_counts(normalize=True)
    else:
        df = dd.loc[dd['Business_size'] == business_size].set_index('Business_size').T.head(topn)#[current_question].value_counts(normalize=True).head(topn)            
    bcplot = sns.barplot(x=df.index, y=df[business_size], ax=axs[coords[0],coords[1] ]);
    axs[coords[0],coords[1] ].set_title(business_size);
    for p in axs[coords[0],coords[1] ].patches:
        axs[coords[0],coords[1] ].annotate(str("{0:.0%}".format(p.get_height())), (p.get_x() * 0.97, p.get_height() * 0.97))
    for item in bcplot.get_xticklabels():
        item.set_rotation(90);
    bcplot.set(yticks=[0.1, 0.2, 0.3]);
for ax in axs.flat:
    ax.set(xlabel='', ylabel='N_of_employees')

for ax in axs.flat:
    ax.label_outer() 

#fig.delaxes(axs[1][2])


# NLP is a rare application but Word embeddings and encoder-decoder models are most popular methods now.
