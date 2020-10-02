#!/usr/bin/env python
# coding: utf-8

# ## Data Science Survey Analysis
# 
# This notebook analyses the Data Science survey data offered by Kaggle and finds insights about which technologies and areas young developers to focus in order to land a successful career as a Data Scientist. <br>
# 
# ### Content
# [1. Geography](#1)<br>
# [2. Gender of Respondents](#2)<br>
# [3. Developer Role](#3)<br>
# &emsp;[3.1 Developer Type](#3.1)<br>
# &emsp;[3.2 Industry Specific to the current Client](#3.2)<br>
# [4. Education](#4)<br>
# [5. Experience and Work Life](#5)<br>
# &emsp;[5.1 Number of years of experience](#5.1)<br>
# &emsp;[5.2 Does the Current Employer incorporate Machine Learning?](#5.2)<br>
# [6. Tools/Frameworks/Language Used by Developers](#6)<br>
# &emsp;[6.1 What are the tools which are most often used by Developers at Work?](#6.1)<br>
# &emsp;[6.2 Integrated environment used at Work or School](#6.2)<br>
# &emsp;[6.3 Most Preferred Notebook in last 5 years](#6.3)<br>
# &emsp;[6.4 Most Preferred Cloud Providers](#6.4)<br>
# &emsp;[6.5 Programming Language used and Recommended](#6.5)<br>
# &emsp;[6.6 Most Preferred Framework](#6.6)<br>
# &emsp;[6.7 Mostly Used Cloud Computing Product in past 5 years](#6.7)<br>
# &emsp;[6.8 Most Used Products in past 5 years](#6.8)<br>
# &emsp;&emsp;[6.8.1 Most used Machine Learning and Database products](#6.8.1)<br>
# &emsp;&emsp;[6.8.2 Most Used Big Data and Analytics Products](#6.8.2)<br>
# &emsp;&emsp;[6.8.3 Most interacted Data](#6.8.3)<br>
# &emsp;&emsp;[6.8.4 Where do people find public data?](#6.8.4)<br>
# [7. How Data Scientists distribute their time while working on Machine Learning Project?](#7)<br>
# [8. How people learn implementing techniques and concepts of Machine Learning?](#8)<br>
# &emsp;[8.1 How people learn Machine Learning?](#8.1)<br>
# &emsp;[8.2 How much learning from each of the online course offering platform?](#8.2)<br>
# &emsp;[8.3 Most liked Social media platform that report on Data Science](#8.3)<br>
# [9. Learning Data Science](#9)<br>
# &emsp;[9.1 Online Learning in comparison to Traditional Institutional Learning?](#9.1)<br>
# &emsp;[9.2 Which better demonstrate expertise in Data Science- Academic Achievements or Independents Projects?](#9.2)<br>
# [10. Importance of Different AI Topics](#10)<br>
# &emsp;[10.1 Importance of Fairness in ML algorithm, Explaining ML model, Reproducibility in Data Science](#10.1)<br>
# &emsp;[10.2 How to choose Metric?](#10.2)<br>
# &emsp;[10.3 How much proportion of data project involved exploring unfair bias in the dataset?](#10.3)<br>
# &emsp;[10.4 What is most difficult to ensure the algorithm is fair and unbiased?](#10.4)<br>
# &emsp;[10.5 When to explore model insights about model prediction?](#10.5)<br>
# &emsp;[10.6 what percent of data projects involve exploring model insights?](#10.6)<br>
# &emsp;[10.7 What methods are preferred to interpret decisions of ML model?](#10.7)<br>
# &emsp;[10.8 If ML model Black Box?](#10.8)<br>
# [11. How to do reproducible Coding or work? And what are the difficulties in it?](#11)<br>
# &emsp;[11.1 Tools used to make code reproducible](#11.1)<br>
# &emsp;[11.2 Barriers that prevent from making work reusable?](#11.2)<br>
# [12. Salary of Respondents](#12)<br>
# [13. Summary](#13)<br>
#     

# In[ ]:


import pandas as pd 
import numpy as np
import matplotlib
import matplotlib.pyplot as plt 
import seaborn as sns 
import itertools
color = sns.color_palette()
import plotly.offline as py
py.init_notebook_mode(connected=True)
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.offline as offline
offline.init_notebook_mode()
from plotly import tools
from numpy import array
from matplotlib import cm
import missingno as msno
import cufflinks as cf
from wordcloud import WordCloud, STOPWORDS

from sklearn import preprocessing
import warnings
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_colwidth', 5000)
pd.set_option('display.max_rows', None)
cf.go_offline()


# In[ ]:


freeFormResponse = pd.read_csv('../input/freeFormResponses.csv')
mcqResponse = pd.read_csv('../input/multipleChoiceResponses.csv')
schema = pd.read_csv('../input/SurveySchema.csv')
mcqResponse = mcqResponse[1:].reset_index(drop=True)


# ### <a id="1">1. Geography</a>

# In[ ]:


def filter_data(df):
    return df[~df['Q3'].isin(['Other', 'I do not wish to disclose my location'])].reset_index(drop=True)


# In[ ]:


world_map = filter_data(mcqResponse)
world_map_count = world_map['Q3'].value_counts()
data = [ dict(
        type = 'choropleth',
        locations = world_map_count.index,
        locationmode = 'country names',
        z = world_map_count.values,
        text = world_map_count.values,
        colorscale = [[0.0, 'rgb(242,240,247)'],[0.2, 'rgb(218,218,235)'],[0.4, 'rgb(188,189,220)'],\
            [0.6, 'rgb(158,154,200)'],[0.8, 'rgb(117,107,177)'],[1.0, 'rgb(84,39,143)']],
        autocolorscale = False,
        reversescale = False,
        marker = dict(
            line = dict (
                color = 'rgb(190,190,190)',
                width = 0.5
            ) ),
        colorbar = dict(
            autotick = False,
            tickprefix = '',
            title = 'Survey Participation'),
      ) ]

layout = dict(
    title = 'Participation in survey from different countries',
    geo = dict(
        showframe = False,
        showcoastlines = False,
        projection = dict(
            type = 'Mercator'
        )
    )
)

fig = dict( data=data, layout=layout )
py.iplot( fig, validate=False)


print('Other: ', mcqResponse[mcqResponse['Q3'] == 'Other']['Q3'].count(), ', Dont want to disclose: ',
      mcqResponse[mcqResponse['Q3'] == 'I do not wish to disclose my location']['Q3'].count())


# More participation is being done by the developers from **India and America.** It also shows that these countires uses the Kaggle platform more in comparison to other countries.

# ### <a id="2">2. Gender of Respondents</a>

# In[ ]:


mcqResponse['Q1'].value_counts().plot.bar(title='Gender Count')


# ### <a id="3">3. Developer Role</a>
# #### <a id="3.1">3.1 Developer Type</a>

# In[ ]:


role = mcqResponse[mcqResponse['Q6'] != 'Other']['Q6'].value_counts()
role.iplot(kind='barh', title='Developer Type', margin=go.Margin(l=160))

print('Other: ', mcqResponse[mcqResponse['Q6'] == 'Other']['Q6'].count())


# **There are developers from wide cateogories of skills:**<br>
# 1) Most of them are Students or it can be said that there are many young people which are using the Kaggle platform and learning from it. <br>
# 2) Other big ratio comprises professional Data Scientists, Analysts and Software Engineers.<br>

# #### <a id="3.2">3.2 Industry specific to the current client</a>

# In[ ]:


client_specific_indistry_DS = mcqResponse[((mcqResponse['Q6'] == 'Data Scientist')&(mcqResponse['Q7'] != 'Other'))]['Q7']
client_specific_indistry = mcqResponse[mcqResponse['Q7'] != 'Other']['Q7'].value_counts().to_frame().reset_index()
role = client_specific_indistry_DS.value_counts().to_frame().reset_index()


trace0 = go.Bar(
    x = list(client_specific_indistry['Q7']),
    y = list(client_specific_indistry['index']),
    orientation = 'h'
)

trace1 = go.Bar(
    x = list(role['Q7']),
    y = list(role['index']),
    orientation = 'h'
)


fig = tools.make_subplots(rows=2, cols=1, subplot_titles=('Client Specific Industry', 'Client Specific Industry for Data Science'))
fig.append_trace(trace0, 1, 1)
fig.append_trace(trace1, 2, 1)
fig['layout'].update(height=1000,margin=go.Margin(l=250))
py.iplot(fig)


print('Other: ', mcqResponse[mcqResponse['Q7'] == 'Other']['Q7'].count())


# Most of the clients are from Computers/Technology Industries for Developers from Data Scientists as well as for other Developers. There are many Data Science Developers who has their clients in Accounting/Finance Industries.

# There are <b>far more male respondents in comparison to Female Respondents</b> in the Data Science Survey. Hope this gap get filled soon in near future.

# ### <a id="4">4. Education</a>

# In[ ]:


formal_edu_DS = mcqResponse[mcqResponse['Q6'] == 'Data Scientist']['Q4'].value_counts().to_frame().reset_index()
formal_edu = mcqResponse['Q4'].value_counts().to_frame().reset_index()


trace0 = go.Bar(
    x = list(formal_edu['Q4']),
    y = list(formal_edu['index']),
    orientation = 'h'
)

trace1 = go.Bar(
    x = list(formal_edu_DS['Q4']),
    y = list(formal_edu_DS['index']),
    orientation = 'h'
)


fig = tools.make_subplots(rows=2, cols=1, subplot_titles=('Formal Education', 'Formal Education of Data Scientist'))
fig.append_trace(trace0, 1, 1)
fig.append_trace(trace1, 2, 1)
fig['layout'].update(height=500,margin=go.Margin(l=370))
py.iplot(fig)


# Most of the Developers have Master's and Bachelor's Degree. Some have Doctoral Degree.

# ### <a id="5">5. Experience and Work Life</a>
# #### <a id="5.1">5.1 Number of years of experience</a>

# In[ ]:


experience_DS = mcqResponse[mcqResponse['Q6'] == 'Data Scientist']['Q8'].value_counts().to_frame().reset_index()
experience = mcqResponse['Q8'].value_counts().to_frame().reset_index()


trace0 = go.Bar(
    x = list(experience['Q8']),
    y = list(experience['index']),
    orientation = 'h'
)

trace1 = go.Bar(
    x = list(experience_DS['Q8']),
    y = list(experience_DS['index']),
    orientation = 'h'
)


fig = tools.make_subplots(rows=2, cols=1, subplot_titles=('Experience (in yrs)', 'Experience of Data Scientists (in yrs)'))
fig.append_trace(trace0, 1, 1)
fig.append_trace(trace1, 2, 1)
fig['layout'].update(height=500,margin=go.Margin(l=50))
py.iplot(fig)


# **Most of the developers are young talents with less than 5 years of experience.**<br>
# 1) As infered in the previous section, many are students and are enthusist to learn data science so does this graph shows having higher percentage of people having 0-1 years of experience.<br>
# 2) There are few Data Scientists with over 10 years of experience and many with less than 5 years of experience. So it can be said that more people are learning Machine Learning.

# #### <a id="5.2">5.2 Does the Current Employer incorporate Machine Learning?</a>

# In[ ]:


series = mcqResponse['Q10'].value_counts()
series.iplot(kind='barh', title='Does Current Employer Incorporate Machine Learning?', margin=go.Margin(l=560))


# - Above graph shows that people have started using Machine Learning and many have even put their model in production for more than 2 years. <br>
# - Still the majority of Developer's Employer are either exploring ML Methods or don't use them.

# #### <a id="5.3">5.3 What is your important part of work?</a>

# In[ ]:


cols = ['Q11_Part_1', 'Q11_Part_2', 'Q11_Part_3', 'Q11_Part_4', 'Q11_Part_5', 'Q11_Part_6', 'Q11_Part_7']
skills_df = {}
skills_df['Skills'] = []
skills_df['Count'] = []
for col in cols:
    name = str(mcqResponse[col].dropna().unique()).strip("[]''")
    count = mcqResponse[mcqResponse[col] == name][col].count()
    skills_df['Skills'].append(name)
    skills_df['Count'].append(count)
    
skills_df = pd.DataFrame(skills_df)
skills_df.sort_values(by=['Count'], inplace=True)

trace0 = go.Bar(
    x = skills_df['Count'],
    y = skills_df['Skills'],
    orientation = 'h'
)

layout = go.Layout(
    margin = dict(l=700),
    title = 'Respondents important role at work'
)
trace = [trace0]
fig = go.Figure(data=trace, layout=layout)
py.iplot(fig)

print('Select all that apply')


# 1) Almost 9K people say that their main role is to analyse the data so that to influence product or becision decisions. So that means Data analyses plays a major role in any organisation and before building AI models. <br>
# 2) The second majority of people build protypes of Machine Learning models to new areas which **indicates that people wants to migrate the boring hand-made stuff to automatic Machine Learnt stuff**

# ### <a id="6">6. Tools/Frameworks/Language Used by Developers</a>
# #### <a id="6.1">6.1 What are the tools which are most often used by Developers at Work?</a>

# In[ ]:


tools_used = mcqResponse[mcqResponse['Q12_MULTIPLE_CHOICE'] != 'Other']['Q12_MULTIPLE_CHOICE'].value_counts()
tools_used.iplot(kind='barh', title='Mostly used Tool', margin=go.Margin(l=450))

print('Other: ', mcqResponse[mcqResponse['Q12_MULTIPLE_CHOICE'] == 'Other']['Q12_MULTIPLE_CHOICE'].count())


# 1) **RStudio and JupyterLab are most used tools by developers.** This shows that people prefer to use R or Python while working on Machine Learning projects. <br>
# 2) Almost 4K developers also use Basic Statistical Software like Excel, Google Sheets.<br>
# 3) Very few developers use Business Intelligence Softwares, Cloud Based Softwares.<br>
# 4) **This graph shows that few people use Advance tools. So, having knowledge of how to use such tools can help to boost the skills and the chance to land a good Data Science Job, can stand above many other emerging Data Scientists.**

# #### <a id="6.2">6.2 Integrated environment used at Work or School</a>

# In[ ]:


cols = ['Q13_Part_1', 'Q13_Part_2', 'Q13_Part_3', 'Q13_Part_4', 'Q13_Part_5', 'Q13_Part_6', 'Q13_Part_7',
        'Q13_Part_8', 'Q13_Part_9', 'Q13_Part_10', 'Q13_Part_11', 'Q13_Part_12', 'Q13_Part_13', 'Q13_Part_14',
        'Q13_Part_15']

ide_df = {}
ide_df['IDE'] = []
ide_df['Count'] = []
for col in cols:
    name = str(mcqResponse[col].dropna().unique()).strip("[]''")
    count = mcqResponse[mcqResponse[col] == name][col].count()
    ide_df['IDE'].append(name)
    ide_df['Count'].append(count)
    
ide_df = pd.DataFrame(ide_df)
ide_df.sort_values(by=['Count'], inplace=True)
trace0 = go.Bar(
    x = ide_df['Count'],
    y = ide_df['IDE'],
    orientation = 'h'
)

layout = go.Layout(
    margin = dict(l=120),
    title = 'Most Preferred IDE in last 5 years'
)
trace = [trace0]
fig = go.Figure(data=trace, layout=layout)
py.iplot(fig)

print('Select all that apply')


# 1) As infered from section 6.1, most people use Jupyter or R while working and hence Jupyter/Ipython or RStudio.<br>
# 2) Notepad++, PyCharm and Sublime text are among other IDE which people use.<br>

# In[ ]:


def generate_choices_df(cols, to_skip_cols):
    df = {}
    df['Choice'] = []
    df['Count'] = []
    for col in cols:
        name = str(mcqResponse[~mcqResponse[col].isin(to_skip_cols)][col].dropna().unique()).strip("[]''")
        count = mcqResponse[mcqResponse[col] == name][col].count()
        df['Choice'].append(name)
        df['Count'].append(count)

    df = pd.DataFrame(df)
    df.sort_values(by=['Count'], inplace=True)
    
    return df

def plot_choices(df, title, left_margin=50):
    trace0 = go.Bar(
        x = df['Count'],
        y = df['Choice'],
        orientation = 'h'
    )

    layout = go.Layout(
        margin = dict(l=left_margin),
        title = title
    )
    trace = [trace0]
    fig = go.Figure(data=trace, layout=layout)
    py.iplot(fig)

    print('Select all that apply')
    return None


# #### <a id="6.3">6.3 Most Preferred Notebook in last 5 years</a>

# In[ ]:


columns =  ['Q14_Part_1', 'Q14_Part_2', 'Q14_Part_3', 'Q14_Part_4', 'Q14_Part_5', 'Q14_Part_6',
            'Q14_Part_7', 'Q14_Part_8', 'Q14_Part_9', 'Q14_Part_10', 'Q14_Part_11']
notebook_choices_df = generate_choices_df(columns, ['Other', 'None'])
plot_choices(notebook_choices_df, 'Most Preferred Notebook in last 5 years', left_margin=150)


# 1) **So, KAGGLE Kernels are the most preferred Notebooks by the developers.**<br>
# 2) Other most used Notebooks are JupyterHub, Google Colab.<br>
# 3) Very few people use Domino Datalab, Crestle, Floydhub.

# #### <a id="6.4">6.4 Most Preferred Cloud Providers</a>

# In[ ]:


columns =  ['Q15_Part_1', 'Q15_Part_2', 'Q15_Part_3', 'Q15_Part_4', 'Q15_Part_5', 'Q15_Part_6', 'Q15_Part_7']
cloud_choices_df = generate_choices_df(columns, ['I have not used any cloud providers'])
plot_choices(cloud_choices_df, 'Most Preferred Cloud Computing Service in last 5 years', left_margin=250)

print(mcqResponse[mcqResponse['Q15_Part_6'] == 'I have not used any cloud providers']['Q15_Part_6'].count(), ''' people have not used any Cloud Provider''')


# 1) **People are more using AWS or Google Cloud plaforms as their Cloud Computing Services.**<br>
# 2) Knowledge of such skills to young aspirants can help to get a good position in indistry.

# #### <a id="6.5">6.5 Programming Language used and Recommended</a>

# In[ ]:


lang_used_columns =  ['Q16_Part_1', 'Q16_Part_2', 'Q16_Part_3', 'Q16_Part_4', 'Q16_Part_5', 'Q16_Part_6', 'Q16_Part_7',
                       'Q16_Part_8', 'Q16_Part_9', 'Q16_Part_10', 'Q16_Part_11', 'Q16_Part_12', 'Q16_Part_13', 'Q16_Part_14',
                       'Q16_Part_15', 'Q16_Part_16', 'Q16_Part_17', 'Q16_Part_18']
lang_choices_df = generate_choices_df(lang_used_columns, ['None'])
lang_recommended = mcqResponse['Q18'].value_counts().to_frame().reset_index()

f, ax = plt.subplots(2, 2, figsize=(50,40))

sns.barplot(list(lang_choices_df['Count']), list(lang_choices_df['Choice']), ax=ax[0,0])

for index, value in enumerate(lang_choices_df['Count']):
    ax[0,0].text(0.8, index, value, color='k', fontsize=25)
    
ax[0,0].set_title('Programming Languages USed', fontsize=35)
ax[0,0].set_yticklabels(lang_choices_df['Choice'], fontsize=25)


sns.barplot(list(lang_recommended['Q18']), list(lang_recommended['index']), ax=ax[1,0])

for index, value in enumerate(lang_recommended['Q18']):
    ax[1,0].text(0.8, index, value, color='k', fontsize=25)
    
ax[1,0].set_title('Recommended Programming Languages', fontsize=35)
ax[1,0].set_yticklabels(lang_recommended['index'], fontsize=25)


word_cloud = WordCloud(height=460, width=300).generate("".join(mcqResponse['Q17'].dropna()))
ax3 = plt.subplot2grid((2,2), (0,1), rowspan=2)
ax3.imshow(word_cloud)
ax3.axis('off')
ax3.set_title('Specific Programming Language Used', fontsize=45)
plt.show()


# 1) As analysed earlier, People mostly use Python or R as their programming language.<br>
# 2) Sql, c++ are also among the widely used languages.<br>
# 3) Python, Typescript are the specific languages which people use.

# #### <a id="6.6">6.6 Most Preferred Framework</a>

# In[ ]:


columns_ml =  ['Q19_Part_1', 'Q19_Part_2', 'Q19_Part_3', 'Q19_Part_4', 'Q19_Part_5', 'Q19_Part_6', 'Q19_Part_7',
           'Q19_Part_8', 'Q19_Part_9', 'Q19_Part_10', 'Q19_Part_11', 'Q19_Part_12', 'Q19_Part_13', 'Q19_Part_14',
           'Q19_Part_15', 'Q19_Part_16', 'Q19_Part_17', 'Q19_Part_18', 'Q19_Part_19']

columns_vis =  ['Q21_Part_1', 'Q21_Part_2', 'Q21_Part_3', 'Q21_Part_4', 'Q21_Part_5', 'Q21_Part_6', 'Q21_Part_7',
           'Q21_Part_8', 'Q21_Part_9', 'Q21_Part_10', 'Q21_Part_11', 'Q21_Part_12', 'Q21_Part_13']

framework_choices_df = generate_choices_df(columns_ml, ['None'])
vis_lib_choices_df = generate_choices_df(columns_vis, ['None'])

trace1 = go.Bar(
    x=framework_choices_df['Count'],
    y=framework_choices_df['Choice'],
    orientation = 'h'
)
trace2 = go.Bar(
    x=vis_lib_choices_df['Count'],
    y=vis_lib_choices_df['Choice'],
    orientation = 'h'
)

fig = tools.make_subplots(rows=1, cols=2, subplot_titles=('Most Preferred Machine Learning Framework',
                                                          'Mostly Used Visualization Library'))
fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 1, 2)
fig['layout'].update(height=500, width=900, title='Frameworks preferred by Developers in last 5 years', 
                     margin=go.Margin(l=100, r=0))

py.iplot(fig)


print(mcqResponse[mcqResponse['Q19_Part_18'] == 'None']['Q19_Part_18'].count(), ''' people have not used any ML Framework specified in the list''')
print(mcqResponse[mcqResponse['Q21_Part_12'] == 'None']['Q21_Part_12'].count(), ''' people have not used any Visualization Library specified in the list''')


# 1) The Machine LEarning Frameworks: <br>
#     - scikit-learn, Tensorflow, keras are the top three Machine LEarning frameworks which people prefer to use.<br>
#     - random forest, xgboost are among the other preferred Machine learning frameworks.<br>
# 2) The Visualization Frameworks:<br>
#     - matplotlib, seaborn, ggplot2, plotly are the top 4 most preferred Visualization frameworks.<br>
#     - Very few people prefer Altair or Geoplotlib, etc.

# #### <a id="6.7">6.7 Mostly Used Cloud Computing Product in past 5 years</a>

# In[ ]:


columns =  ['Q27_Part_1', 'Q27_Part_2', 'Q27_Part_3', 'Q27_Part_4', 'Q27_Part_5', 'Q27_Part_6', 'Q27_Part_7',
           'Q27_Part_8', 'Q27_Part_9', 'Q27_Part_10', 'Q27_Part_11', 'Q27_Part_12', 'Q27_Part_13', 'Q27_Part_14',
           'Q27_Part_15', 'Q27_Part_16', 'Q27_Part_17', 'Q27_Part_18', 'Q27_Part_19', 'Q27_Part_20']
cloud_prod_choices_df = generate_choices_df(columns, ['None'])
plot_choices(cloud_prod_choices_df, 'Mostly Used Cloud Computing Products in last 5 years', left_margin=220)

print(mcqResponse[mcqResponse['Q27_Part_19'] == 'None']['Q27_Part_19'].count(), ''' people have not used any Cloud Computing Product specified in the list''')


# 1) As discussed in section 6.4, **AWS and Google Cloud are the most preferred Cloud Computing platforms. So the above graph adds-on information to it and shows that AWS Elastic Compute Cloud and Google Compute Engine are also most preferred Cloud Computing Products.**<br>
# 2) People also prefer AWS Lambda, Azure Virtual Machines, Google App Engine.

# #### <a id="6.8">6.8 Most Used Products in past 5 years</a>
# #### <a id="6.8.1">6.8.1 Most used Machine Learning and Database products</a>

# In[ ]:


columns_ml_product =  list(mcqResponse.columns[mcqResponse.columns.str.startswith('Q28')])
columns_ml_product.remove('Q28_OTHER_TEXT')
ml_prod_choices_df = generate_choices_df(columns_ml_product, ['None'])

columns_db_product =  list(mcqResponse.columns[mcqResponse.columns.str.startswith('Q29')])
columns_db_product.remove('Q29_OTHER_TEXT')
db_choices_df = generate_choices_df(columns_db_product, ['None'])

trace1 = go.Bar(
    x=ml_prod_choices_df['Count'],
    y=ml_prod_choices_df['Choice'],
    orientation = 'h'
)
trace2 = go.Bar(
    x=db_choices_df['Count'],
    y=db_choices_df['Choice'],
    orientation = 'h'
)

fig = tools.make_subplots(rows=2, cols=1, subplot_titles=('Most Used ML Products in last 5 years',
                                                          'Most Used Database Products in last 5 years'))
fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 2, 1)
fig['layout'].update(height=950, width=800, 
                     margin=go.Margin(l=240))

py.iplot(fig)


print(mcqResponse[mcqResponse['Q28_Part_42'] == 'None']['Q28_Part_42'].count(), ''' people have not used any ML Product specified in the list''')
print(mcqResponse[mcqResponse['Q29_Part_27'] == 'None']['Q29_Part_27'].count(), ''' people have not used any Database Product specified in the list''')


# 1) Most used Machine Learning Products:<br>
#     - People prefer wide range of Machine Learning Products, may be depending upon each one's importance.<br>
#     - SAS, Cloudera, Azure Machine Learning Studio are among the most prefered Machine learning products.<br>
# 2) Most used Database products: <br>
#     - SQL is most used Database product.<br>
#     - MySQL, postgresSQL, SQLite, Miscrosoft SQL Server are other preferred Database products.<br>
#     - Oracle Database, AWS Database are also among top preferred database products.<br>

# #### <a id="6.8.2">6.8.2 Most Used Big Data and Analytics Products</a>

# In[ ]:


def plot_choices_graph(col, to_skip_cols, title, print_title=None, col_num=None, left_margin=50):
    columns =  list(mcqResponse.columns[mcqResponse.columns.str.startswith(col)])
    columns.remove(col+'_OTHER_TEXT')
    db_choices_df = generate_choices_df(columns, to_skip_cols)
    plot_choices(db_choices_df, title, left_margin=left_margin)
    
    if col_num:
        print(mcqResponse[mcqResponse[col+'_Part_'+col_num] == 'None'][col+'_Part_'+col_num].count(), print_title)
    
    return None


# In[ ]:


plot_choices_graph('Q30', ['None'], 'Most Used Big Data and Analytics Products in last 5 years',
                  'people have not used any Big Data and Anaytics Product specified in the list', '24', 200)


# 1) **Google Big Query, AWS Redshift, Databricks, AWS Elastic MapReduce** are top 4 most preferred Big Data and Analytics products.<br>

# #### <a id="6.8.3">6.8.3 Most interacted Data</a>

# In[ ]:


plot_choices_graph('Q31', ['None'], 'Most interacted Data', left_margin=110)


# 1) The above graph shows that mostly people interact with the Numeric Data, Test data, categorical data, time series data or Tabular Data. <br>
# 2) Very less people interact with audio data, video data or image data.<br>
# 
# This shows that people at work or students mostly interact with the above top dataset categories.<br>
# **Learning how to work on image data, audio data, Genetic Data can help to grow the career as the demand for such skills may grow exponentially in future.**

# #### <a id="6.8.4">6.8.4 Where do people find public data?</a>

# In[ ]:


plot_choices_graph('Q33', ['None'], 'Where People find Public Data', left_margin=500)


# 1) KAGGLE, Socrata, are the top used platoforms where people find public Datasets. So people can try to publish their datasets on these platforms so that other people can learn and work on it.<br>
# 2) Google Search, Github, web-scraping or Goverment websites are other sources where people fing public data.<br>
# 3) People also find data from University research group websites but may be all are not free.

# ### <a id="7">7. How Data Scientists distribute their time while working on Machine Learning Project?</a>

# In[ ]:


def find_avg_proportion(col, names):
    columns = list(mcqResponse.columns[mcqResponse.columns.str.startswith(col)])
    columns.remove(col+'_OTHER_TEXT')
    proportions = []
    for c in columns:
        proportions.append(mcqResponse[c].astype(float).mean())
    
    df = pd.DataFrame({'Name': names, 'Proportion': proportions})
    df.sort_values(by=['Proportion'], inplace=True)
    return df


# In[ ]:


ml_project_distribution = find_avg_proportion('Q34', ['Data Gathering', 'Data Cleaning', 'Visualizing Data',
                                                     'Model Building/Model Selection', 'Putting model in production',
                                                     'Finding insights and communicating with stakeholders'])

def plot_proportions(df, title, left_margin=50):
    trace0 = go.Bar(
        x = df['Proportion'],
        y = df['Name'],
        orientation = 'h'
    )

    layout = go.Layout(
        margin = dict(l=left_margin),
        title = title
    )
    trace = [trace0]
    fig = go.Figure(data=trace, layout=layout)
    py.iplot(fig)

    return None

plot_proportions(ml_project_distribution, 'Distribution of Time in ML Project in %', left_margin=350)


# - All phases of building Machine learning model demands time of developers but **Data Cleaning and Model Selection takes most of the time of developers.**<br>
# - It is definitely important to spend time in data cleaning so that unwanted data is not fed into the machine learning model which may affect the performance of the model.

# ### <a id="8">8. How people learn implementing techniques and concepts of Machine Learning?</a>
# #### <a id="8.1">8.1 How people learn Machine Learning?</a>

# In[ ]:


ml_learning_distribution = find_avg_proportion('Q35', ['Self-taught', 'Online Courses', 'Work',
                                                     'University', 'Kaggle Competitions',
                                                     'Other'])

plot_proportions(ml_learning_distribution, 'Distribution of Learning in ML Project', left_margin=150)


# #### <a id="8.2">8.2 How much learning from each of the online course offering platform?</a>

# In[ ]:


plot_choices_graph('Q36', ['None'], 'Distribution of Learning from Online Platforms', left_margin=165)


# In[ ]:


most_used_online_platforms = mcqResponse['Q37'].value_counts()
most_used_online_platforms.iplot(title='Most Used Online platform for learning AI', margin=go.Margin(b=150))


# #### <a id="8.3">8.3 Most liked Social media platform  that report on Data Science</a>

# In[ ]:


plot_choices_graph('Q38', ['None'], title = 'Favourite Social Media platform that report on Data Science', left_margin=200)


# 1) **How people learn Machine learning?** <br>
#     - Mostly people learn own their own or from online courses.<br>
#     - Arounf 18% people learn from university (as their academics), 10% from while working (this shows that industry is demanding machine learning) and 8% people from kaggel competitions.<br>
# 2) **Most used online learning platform-**<br>
#     - Most of the poeple use Coursera for online courses.<br>
#     - Kaggle Learning, Datacamp, udemy are other most used online learning platforms.<br>
# 3) **Favourite Social media platform for Data Science.**<br>
#     - Most of the people like Kaggle Forum and Medium Blog posts. <br>
#     
# **So in order to share information, people can use Kaggle Forum or Medium Bolg posts as most of the poeple like to read them.**

# ### <a id="9">9. Learning Data Science</a>
# #### <a id="9.1">9.1 Online Learning in comparison to Traditional Institutional Learning?</a>

# In[ ]:


online_learning = mcqResponse['Q39_Part_1'].value_counts().to_frame().reset_index()
institutional_learning = mcqResponse['Q39_Part_2'].value_counts().to_frame().reset_index()

trace0 = go.Bar(
    x = online_learning['Q39_Part_1'],
    y = online_learning['index'],
    orientation = 'h'
)
trace1 = go.Bar(
    x = institutional_learning['Q39_Part_2'],
    y = institutional_learning['index'],
    orientation = 'h'
)

fig = tools.make_subplots(rows=2, cols=1, subplot_titles=('Online Learning and MOOCs',
                                                          'In-person Bootcamp'))
fig.append_trace(trace0, 1, 1)
fig.append_trace(trace1, 2, 1)
fig['layout'].update(height=450, width=1000, 
                     margin=go.Margin(l=200))

py.iplot(fig)


# People have mix views but mostly people feel that online learning is either better or equivalent to traditional institutional learning.<br>
# 

# #### <a id="9.2">9.2 Which better demonstrate expertise in Data Science- Academic Achievements or Independents Projects?</a>

# In[ ]:


series = mcqResponse['Q40'].value_counts()
series.iplot(kind='barh', title='Academic Achievement V/S Independent Projects', margin=go.Margin(l=500))


# Most of the people say that independent projects are more important than academic achievemets. People learn a lot while working on projects so independent projects can help to grow knowledge.

# ### <a id="10">10. Importance of Different AI Topics</a>
# #### <a id="10.1">10.1 Importance of Fairness in ML algorithm, Explaining ML model, Reproducibility in Data Science</a>

# In[ ]:


columns = mcqResponse.columns[mcqResponse.columns.str.startswith('Q41')]

ml_algo = mcqResponse[columns[0]].value_counts().to_frame().reset_index()
explaining_ml = mcqResponse[columns[1]].value_counts().to_frame().reset_index()
reproducibility_DS = mcqResponse[columns[2]].value_counts().to_frame().reset_index()

trace0 = go.Bar(
    x = ml_algo['index'],
    y = ml_algo['Q41_Part_1'],
    name = 'Fairness in ML algorithm'
)
trace1 = go.Bar(
    x = explaining_ml['index'],
    y = explaining_ml['Q41_Part_2'],
    name = 'Explaining ML model'
)
trace2 = go.Bar(
    x = reproducibility_DS['index'],
    y = reproducibility_DS['Q41_Part_3'],
    name = 'Reproducibility in Data Science'
)
trace = [trace0, trace1, trace2]

layout = go.Layout(
    barmode = 'group',
    title='Importance of Different ML Topics'
)

fig = go.Figure(data=trace, layout = layout)
py.iplot(fig)


# Most of the developers say that all the three- Fairness in ML algorithm, Explaining the ML model and Reproducibility in Data Science are **very important** in Data Science.

# #### <a id="10.2">10.2 How to choose Metric?</a>

# In[ ]:


plot_choices_graph('Q42', ['None'], 'Metrics preferred which determine the success of model', left_margin=500)


# 1) Mostly people say they use the metrics that considers the accuracy of the model. <br>
# 2) But there are people who say that they consider business goals while chosing the metric. And around 2700 developers prefer the metrics that considers the unfair bias.

# #### <a id="10.3">10.3 How much proportion of data project involved exploring unfair bias in the dataset?</a>

# In[ ]:


exploring_data = mcqResponse['Q43'].value_counts().to_frame().reset_index()
trace0 = go.Bar(
    x = exploring_data['Q43'],
    y = exploring_data['index'],
    orientation = 'h'
)

layout = go.Layout(
    title = 'Proportion of data project involved exploring unfair bias in the dataset'
)
trace = [trace0]
fig = go.Figure(data=trace, layout=layout)
py.iplot(fig)


# For most of the developers, 0-10 or 10-20 proportion of their data science project requires exploring unfair bias. This means that people find it in less time.

# #### <a id="10.4">10.4 What is most difficult to ensure the algorithm is fair and unbiased?</a>

# In[ ]:


columns =  list(mcqResponse.columns[mcqResponse.columns.str.startswith('Q44')])
ensure_fairness_df = generate_choices_df(columns, ['None'])
plot_choices(ensure_fairness_df, title='Difficulty in ensuring the algorithm is fair and unbiased', left_margin=650)


# 1) Mostly people say that they find difficult to collect data that may be found unfairly targeted or selecting the appropriate evaluation metric.<br>
# 2) Dificulty in identifying groups that are unfairly targeted and lack communication between data collectors and analyzers are other difficulties people face.<br>
# 3) Around 4K people say they have not performed this task.

# #### <a id="10.5">10.5 When to explore model insights about model prediction?</a>

# In[ ]:


columns =  list(mcqResponse.columns[mcqResponse.columns.str.startswith('Q45')])
model_df = generate_choices_df(columns, ['None'])
plot_choices(model_df, title='When to explore model insights about model prediction?', left_margin=500)


# #### <a id="10.6">10.6 what percent of data projects involve exploring model insights?</a>

# In[ ]:


exploring_data = mcqResponse['Q46'].value_counts().to_frame().reset_index()
trace0 = go.Bar(
    x = exploring_data['Q46'],
    y = exploring_data['index'],
    orientation = 'h'
)

layout = go.Layout(
    title = 'How much data projects involve exploring model insights?'
)
trace = [trace0]
fig = go.Figure(data=trace, layout=layout)
py.iplot(fig)


# #### <a id="10.7">10.7 What methods are preferred to interpret decisions of ML model?</a>

# In[ ]:


columns =  list(mcqResponse.columns[mcqResponse.columns.str.startswith('Q47')])
interpret_model_df = generate_choices_df(columns, ['None'])
plot_choices(interpret_model_df, title='Methods preferred to interpret decisions of ML model', left_margin=500)


# #### <a id="10.8">10.8 If ML model Black Box?</a>

# In[ ]:


black_box_or_not = mcqResponse['Q48'].value_counts().to_frame().reset_index()
trace0 = go.Bar(
    x = black_box_or_not['Q48'],
    y = black_box_or_not['index'],
    orientation = 'h'
)

layout = go.Layout(
    title = 'If ML box is black box or not?',
    margin = go.Margin(
        l = 650
    ),
    height = 400
)
trace = [trace0]
fig = go.Figure(data=trace, layout=layout)
py.iplot(fig)


# This Section 10 explaines the challenges, difficulties developers face in building Machine Learning models.<br><br>
# 1) **When to explore model insights:** It can ce said from section 10.5 that model analysis is important when building the model and before putting the model into production. This is fairly right for the model to perform correctly while in production.<br><br>
# 2) **Proportion of time developers spend on finding model insights:** In section 10.6, it can be analysed that some people spend less proportion of their time in finding model insights and some spend more. But majority of people spend less than 30 % of their time in this phase of Machine learning.<br><br>
# 3) **How to interpret the decisions of ML model:** Section 10.7 shows that<br>
#     - Mostly people use Predicted result v/s actual results to interpret ML decisions.<br>
#     - There are other ways as well which are almost equally important ans used by people like examining important features and finding correlations among them, reducing the dimensionality, ploting the decision boundry, etc.<br><br>
# 4) **If ML BLACK BOX?**<br>
#     - Around 6400 people say that they can understand and explain the outputs but not all the ML models.<br>
#     - Around 3000 people say that experts can explain the model outputs.

# ### <a id="11">11. How to do reproducible Coding or work? And what are the difficulties in it?</a>
# #### <a id="11.1">11.1 Tools used to make code reproducible</a>

# In[ ]:


plot_choices_graph('Q49', ['None'], 'Tools that can make work easy and reproducible', left_margin=484)


# #### <a id="11.2">11.2 Barriers that prevent from making work reusable?</a>

# In[ ]:


plot_choices_graph('Q50', ['None'], 'Barriers that prevent from making work easy and reproducible', left_margin=450)


# **Section 11 explains how people write reusale codes and what barriers they face**<br>
# 1) Tools used to make code reusable:<br>
#     - People say that well documented and commented code helps to use that code again and again.<br>
#     - Sharing on code sharing platforms like Github also helps other to use the same code.<br>
# 2) Difficulties that prevent from making work reusable:<br>
#     - Almost 6500 respondents say that it is too time consuming to make the work reusable for other.<br>
#     - Few say that this process requires more incentives and more technical knowledge and are expensive.<br>
#     - Some are even afraid that other can use their work without giving any credit.

# ### <a id="12">12. Salary of Respondents</a>

# In[ ]:


salary = mcqResponse[mcqResponse['Q9'] != 'I do not wish to disclose my approximate yearly compensation']['Q9'].value_counts().to_frame().reset_index()
salary_DS = mcqResponse[((mcqResponse['Q9'] != 'I do not wish to disclose my approximate yearly compensation')&(mcqResponse['Q6'] == 'Data Scientist'))]['Q9'].value_counts().to_frame().reset_index()

trace0 = go.Bar(
    x = list(salary['Q9']),
    y = list(salary['index']),
    orientation = 'h'
)

trace1 = go.Bar(
    x = list(salary_DS['Q9']),
    y = list(salary_DS['index']),
    orientation = 'h'
)


fig = tools.make_subplots(rows=2, cols=1, subplot_titles=('Salary (in $USD)', 'Salary of Data Scientists (in $USD)'))
fig.append_trace(trace0, 1, 1)
fig.append_trace(trace1, 2, 1)
fig['layout'].update(height=700,margin=go.Margin(l=100))
py.iplot(fig)


# 1) Most of the respondents have salary below 20K US Dollars.<br>
# 2) There are Data Scientist who have salary greater than 500K US Dollars as well.
# 

# ### <a id="13">13. Summary</a>
# 
# **Geography**<br><br>
# 1) Most of the developers are from America and India and the ratio of men v/s women is very high. there are far
#     more male developers in comparison to female developers.<br><br>
#     
# **Work**<br><br>
# 1) Most of the Respondents are from Computer Science Industry and a large proportion of them are students. **This shows growing interest in Data Science**. And most of the professional developers have either Bachlor's or Master degree.<br>
# 2) Many companies are migrating to building Machine Learning Models as good proportion of people have learnt AL at work and most of the time of developers is spent in analysing the data.<br>
# 3) For doing work, Jupyter and RStudio, Python and R are the most widely used IDE and Programming language while Kaggle Kernels, Jupyter Hub are the most used Notebooks.<br>
# 4) There is growing interest of scikit-learn, tensorfow, keras, matplotlib, seabron, plotly frameworks among people and they find data for building models using these frameworks from Kaggle, Socrata like platforms which provide free datasets.<br>
# 5) **For any Cloud Computing**, AWS and Google Cloud are most preferred.<br><br>
# 
# **About Machine learning Topics**<br><br>
# 1) Online learning platforms like Coursera, udemy, Kaggle Kernels, Medium blog posts are most used platform to gain and share information.<br>
# 2)While making Machine learning model, it is required to consider fairness in Machine Larning models, reproducubility in Data science.<br>
# 3) And how to check whether the model is performng better or not? For this evaluation metrics like accuracy, metrics which consider unfair bias are considered.<br>
# 4) There are challenges as well which data scientists face in initiating the machine learning project like finding and collecting data. Mostly data scientists try to find data on publically available platforms like Kaggle, Github, etc and some try to collect the data on their own.<br>
# 5) While collecting data, one should consider the biased target problem and the analysis needs to communicate with the data collector in order to generate best data and find best insights from them.<br>
# 6) Bt the job is not done here only. Data scientists needs to check the model insights as well. They can check either while building the AI model or before putting the model into production as this notebook's analysis explains.<br><br>
# 
# **Experience**<br><br>
# 1) As most of the developers are students so does their experience graph depicts. Most of the respondents have
#     experience less than 5 years.<br>
#     
# **Salary**<br>
# 1) Average salary of Developers lie between 0-20K USD
# 
# ### Hope the above analysis helps to get to know about the Data Science Career.
# ### Thank You!
