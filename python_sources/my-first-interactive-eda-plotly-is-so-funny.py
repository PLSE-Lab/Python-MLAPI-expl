#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd # package for high-performance, easy-to-use data structures and data analysis
import numpy as np # fundamental package for scientific computing with Python
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt # for plotting
import seaborn as sns # for making plots with seaborn
import random


plt.style.use('seaborn')
sns.set(font_scale=2)

from plotly import tools
import plotly.offline as py
import plotly.plotly as py_2
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.figure_factory as ff

import missingno as msno
import pycountry

from collections import Counter

import cufflinks as cf
cf.go_offline()

import warnings
warnings.filterwarnings("ignore")


# Contents
# - <a href='#1'>1. Read dataset</a>
#     - <a href='#1_1'>1.1. Checking dataset</a>
#     - <a href='#1_2'>1.2. Making dataframe containing response rate</a>
# - <a href='#2'>2. Personal information</a>
#     - <a href='#2_1'>2.1. Gender</a>
#     - <a href='#2_2'>2.2. Country</a>
#     - <a href='#2_3'>2.3. EducationTypes</a>
#     - <a href='#2_4'>2.4. SelfTaughtTypes</a>
#     - <a href='#2_5'>2.5. UndergradMajor</a>
#     - <a href='#2_6'>2.6. FormalEducation</a>
# - <a href='#3'>3. What is coding to you?</a>
#     - <a href='#3_1'>3.1. HackathonReasons</a>
#     - <a href='#3_2'>3.2. Hobby</a>
#     - <a href='#3_3'>3.3. Hobby for various age (sankey plot)</a>
#     - <a href='#3_4'>3.4. Hobby for various age (bar plot)</a>
# - <a href='#4'>4. Programming tools</a>
#     - <a href='#2_1'>2.1. Language</a>
#     - <a href='#2_2'>2.2. Platform</a>
#     - <a href='#2_3'>2.3. Database</a>
#     - <a href='#2_4'>2.4. Framework</a>
#     - <a href='#2_5'>2.5. IDE</a>
#     - <a href='#2_6'>2.6. CommunicationTools</a>
#     - <a href='#2_7'>2.7. CommunicationTOols for various companySize</a>
#     - <a href='#2_8'>2.8. VersionControl</a>
#     - <a href='#2_9'>2.9. VersionControl for various companySize</a> 
# - <a href='#5'>5. Equipment</a>
#     - <a href='#5_1'>5.1. NumberMonitors</a>
#     - <a href='#5_2'>5.2. ErgonomicDevices</a>
# - <a href='#6'>6. Work</a>
#     - <a href='#6_1'>6.1. JobSatisfaction</a>
#     - <a href='#6_2'>6.2. JobSatisfaction to various company Size</a>
#     - <a href='#6_3'>6.3. JobSatisfaction by Gender</a>
# - <a href='#7'>7. Artificial inteligence</a>
#     - <a href='#7.1'>7.1. AIDangerous</a>
#     - <a href='#7.2'>7.2. AIInteresting</a>
#     - <a href='#7_3'>7.3. AIResponsible</a>
#     - <a href='#7_3'>7.3. AIFutre</a>
# - <a href='#8'>8. Daily life</a>
#     - <a href='#2_1'>8.1. WakeTime</a>
#     - <a href='#2_2'>8.2. WakeTime for various companySize</a>
#     - <a href='#2_3'>8.3. Exercise</a>
#     - <a href='#2_4'>8.4. CheckInCode</a>
#     - <a href='#2_5'>8.5. HoursComputer</a>
#     - <a href='#2_6'>8.6. HoursComputer for various companySize</a>
#     - <a href='#2_7'>8.7. SkipMeals</a>
#     - <a href='#2_8'>8.8. SkipMeals for various companySize</a>
# - <a href='#9'>9. Platform</a>
#     - <a href='#9_1'>9.1. Salary by coutry with considering salaries less than 75% quartile</a>
#     - <a href='#9_2'>9.2. Salary by companySize with considering salaries less than 75% quartile</a>
#     - <a href='#9_2'>9.2. Salary by age with considering salaries less than 75% quartile</a>

# # <a id='1'>1. Read dataset</a>

# - I really wanted to learn plotly for a month.  Many kernel contributers shared their codes, So, I have learned from them. I really appreciate many guys who shared their know-how. 

# ## <a id='1_1'>1.1. Check dataset</a>

# In[2]:


df_response = pd.read_csv('../input/survey_results_public.csv')
df_response.head()


# In[3]:


df_response.shape


# ## <a id='1_2'>1.2. Make meta dataframe containing response rate and schema</a>

# In[4]:


schema = pd.read_csv('../input/survey_results_schema.csv')
response_rate = (100 * df_response.isnull().sum()/df_response.shape[0]).values

df_reference = pd.DataFrame({'Column': df_response.columns,
                            'response_rate': response_rate})
df_reference = pd.merge(df_reference, schema).sort_values('response_rate', ascending=False).reset_index(drop=True)


# In[5]:


df_reference


#  # <a id='2'>2. Personal information</a>

# ## <a id='2_1'>2.1. Gender</a>

# In[6]:


def bar_horizontal_plot(choice_type, question, width=800, height=800, left=600):
    if choice_type == 'multiple':
        temp1 = pd.DataFrame(df_response[question].dropna().str.split(';').tolist()).stack()
        cnt_srs = temp1.value_counts()
    else:
        cnt_srs = df_response[question].value_counts().head(20)
    
    trace = go.Bar(
        y = cnt_srs.index[::-1],
        x = cnt_srs.values[::-1],
        text=['{:.1f}%'.format(percent) for percent in (100 * cnt_srs / cnt_srs.sum())[::-1]], 
        textposition = 'auto',
        textfont=dict(
            size=12,
            color='rgb(0, 0, 0)'
        ),
        orientation = 'h',
        marker = dict(
            color = random_color_generator(30),
            line=dict(color='rgb(8,48,107)',
              width=1.5,)
        ),
        opacity = 0.7,
    )

    layout = dict(
        title = question,
        titlefont = dict(
            size=15
        ),
        margin = dict(
            l = left
        ),
        xaxis=dict(
            title = 'Count',
            tickfont=dict(
                size=12,
            )
        ),
        yaxis=dict(
            tickfont=dict(
                size=12,
            )
        ),
        width = width,
        height = height,
    )

    data = [trace]
    fig = go.Figure(data=data, layout=layout)
    print('Q:', df_reference.loc[(df_reference.Column == question), 'QuestionText'].values[0])
    py.iplot(fig)
    
def random_color_generator(number_of_colors):
    color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
                 for i in range(number_of_colors)]
    return color


# In[7]:


bar_horizontal_plot(choice_type='multiple', question='Gender', width=800, height=400, left=400)


# - 91.7 % of respondents are male
# - 6.8% of respondents are female

# ## <a id='2_2'>2.2. Country</a>

# In[8]:


bar_horizontal_plot(choice_type='single', question='Country', width=800, height=800, left=300)


# - United States and India  are almost half of all respondents.

# In[9]:


countries = df_response['Country'].value_counts()
countries = countries.to_frame().reset_index()
mapping = {country.name: country.alpha_3 for country in pycountry.countries}

for i, country in enumerate(countries['index']):
    temp_country = country
    countries.loc[i, 'code'] = mapping.get(temp_country)
    
data = [dict(
    type = 'choropleth',
    locations = countries['code'],
    z = countries['Country'],
    text = countries['index'],
    colorscale = [[0,"rgb(5, 10, 172)"],[0.35,"rgb(40, 60, 190)"],[0.5,"rgb(70, 100, 245)"],\
            [0.6,"rgb(90, 120, 245)"],[0.7,"rgb(106, 137, 247)"],[1,"rgb(220, 220, 220)"]],
    autocolorscale=False,
    reversescale = True,
    marker = dict(
        line = dict(
            color = 'rgb(180, 180, 180)',
            width =0.5
        )
    ),
    colorbar = dict(
        autotick = False,
        tickprefix = '',
        title = 'Total Count',
    )
)]

layout = dict(
    title = 'countries which responded to the survey',
    height = 800,
    geo = dict(
        showframe = False,
        showcoastlines = False,
        projection = dict(
            type = 'Mercator'
        )
    )
)

fig = dict(data=data, layout=layout)
print('Q:', df_reference.loc[(df_reference.Column == 'Country'), 'QuestionText'].values[0])
py.iplot(fig, validate=False)


# ## <a id='2_3'>2.3.  EducationTypes</a>

# In[10]:


bar_horizontal_plot(choice_type='multiple', question='EducationTypes', width=800, height=600, left=600)


# - Selftaught and online course is main way to learn programming.

# ## <a id='2_4'>2.4.  SelfTaughtTypes</a>

# In[11]:


bar_horizontal_plot(choice_type='multiple', question='SelfTaughtTypes', width=800, height=600, left=500)


# - Official documentation is shown as important way to learn for selftaught-people. Please, keep care for documetation to help selftaught-people
# - Community power(Stack overflow is also good source!

# ## <a id='2.5'>2.5.  UndergradMajor</a>

# In[12]:


bar_horizontal_plot(choice_type='multiple', question='UndergradMajor', width=800, height=600, left=500)


# -  Engineering people are about 73%.

# # <a id='3'>3. What is coding to you?</a>

# ## <a id='3_1'>3.1. HackathonReasons</a>

# In[13]:


bar_horizontal_plot(choice_type='multiple', question='HackathonReasons', width=900, height=600, left=600)


# - Enjoy and self improvement are prior to win prizes or cash awards. How about kaggler? I think kaggler is like them.

# ## <a id='3_2'>3.2.  Hobby</a>

# In[14]:


question = 'Hobby'
temp1 = pd.DataFrame(df_response[question].dropna().str.split(';').tolist()).stack()
cnt_srs = temp1.value_counts()

trace = go.Pie(
    labels = cnt_srs.index,
    values = (100 * cnt_srs / cnt_srs.sum()).values,

     textfont=dict(
         size=20,
         color='rgb(0, 0, 0)'
    ),
)

layout = dict(
    title = question,
    titlefont = dict(
        size=20
    ),
    width = 800,
    height = 500
)

data = [trace]
fig = go.Figure(data=data, layout=layout)
print('Q:', df_reference.loc[(df_reference.Column == question), 'QuestionText'].values[0])

py.iplot(fig)


# - 80.8% of repondents is saying programming is hobby.

# In[15]:


temp = df_response[question].value_counts()
temp = pd.DataFrame({'labels':temp.index,
                  'values': temp.values})

temp.iplot(kind='pie', labels='labels', values='values', title=question, hole=0.6)


# - With Cufflink, we can draw a pretty plot easily . We just make a dataframe and write 'iplot'. It will give an insight to us.

# ## <a id='3_3'>3.3. Hobby for various age (sankey plot)</a>

# In[16]:


data = df_response[['Hobby', 'Age']]

label = np.concatenate((np.array(df_response['Age'].dropna().unique()), np.array(['Yes', 'No'])), axis=0)

data = dict(
    type = 'sankey',
    node = dict(
        pad = 15,
        thickness = 20,
        line = dict(
            color = random_color_generator(1),
            width = 0.5
        ),
        label = label,
        color = random_color_generator(20)
    ),
    link = dict(
        source = [0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6],
        target = [7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8],
        value = [len(df_response[(df_response['Age'] == 'Under 18 years old') & (df_response['Hobby'] == 'Yes')]),
                 len(df_response[(df_response['Age'] == '18 - 24 years old') & (df_response['Hobby'] == 'Yes')]),
                 len(df_response[(df_response['Age'] == '25 - 34 years old') & (df_response['Hobby'] == 'Yes')]),
                 len(df_response[(df_response['Age'] == '35 - 44 years old') & (df_response['Hobby'] == 'Yes')]),
                 len(df_response[(df_response['Age'] == '45 - 54 years old') & (df_response['Hobby'] == 'Yes')]),
                 len(df_response[(df_response['Age'] == '55 - 64 years old') & (df_response['Hobby'] == 'Yes')]),
                 len(df_response[(df_response['Age'] == '65 years or older') & (df_response['Hobby'] == 'Yes')]),
                 len(df_response[(df_response['Age'] == 'Under 18 years old') & (df_response['Hobby'] == 'No')]),
                 len(df_response[(df_response['Age'] == '18 - 24 years old') & (df_response['Hobby'] == 'No')]),
                 len(df_response[(df_response['Age'] == '25 - 34 years old') & (df_response['Hobby'] == 'No')]),
                 len(df_response[(df_response['Age'] == '35 - 44 years old') & (df_response['Hobby'] == 'No')]),
                 len(df_response[(df_response['Age'] == '45 - 54 years old') & (df_response['Hobby'] == 'No')]),
                 len(df_response[(df_response['Age'] == '55 - 64 years old') & (df_response['Hobby'] == 'No')]),
                 len(df_response[(df_response['Age'] == '65 years or older') & (df_response['Hobby'] == 'No')]),
                ]
    )
)

layout = dict(
    height = 800,
    title = 'Hobby' + ' for variuos age',
    titlefont = dict(
        size=20
    ),
    margin = dict(
#         l = 800
    ),
    xaxis=dict(
        tickfont=dict(
            size=18,
        )
    ),
    yaxis=dict(
        tickfont=dict(
            size=18,
        )
    ) 
)

fig = dict(data=[data], layout=layout)
py.iplot(fig, validate=False)


# ## <a id='3_4'>3.4. Hobby for various age (bar plot)</a>

# In[17]:


question = 'Hobby'
temp_age1 = 'Under 18 years old'
temp1 = pd.DataFrame(df_response.loc[df_response.Age == temp_age1, question].dropna().str.split(';').tolist()).stack()
cnt_srs1 = temp1.value_counts()

temp_age2 = '18 - 24 years old'
temp2 = pd.DataFrame(df_response.loc[df_response.Age == temp_age2, question].dropna().str.split(';').tolist()).stack()
cnt_srs2 = temp2.value_counts()

temp_age3 = '25 - 34 years old'
temp3 = pd.DataFrame(df_response.loc[df_response.Age == temp_age3, question].dropna().str.split(';').tolist()).stack()
cnt_srs3 = temp3.value_counts()

temp_age4 = '35 - 44 years old'
temp4 = pd.DataFrame(df_response.loc[df_response.Age == temp_age4, question].dropna().str.split(';').tolist()).stack()
cnt_srs4 = temp4.value_counts()

temp_age5 = '45 - 54 years old'
temp5 = pd.DataFrame(df_response.loc[df_response.Age == temp_age5, question].dropna().str.split(';').tolist()).stack()
cnt_srs5 = temp5.value_counts()

temp_age6 = '55 - 64 years old'
temp6 = pd.DataFrame(df_response.loc[df_response.Age == temp_age6, question].dropna().str.split(';').tolist()).stack()
cnt_srs6 = temp6.value_counts()

temp_age7 = '65 years or older'
temp7 = pd.DataFrame(df_response.loc[df_response.Age == temp_age7, question].dropna().str.split(';').tolist()).stack()
cnt_srs7 = temp7.value_counts()

colors = random_color_generator(10)



trace1 = go.Bar(
    x = cnt_srs1.index,
    y = (100 * cnt_srs1 / cnt_srs1.sum()),
    text=['{:.1f}%'.format(percent) for percent in (100 * cnt_srs1 / cnt_srs1.sum())], 
    textposition = 'auto',
    textfont=dict(
        size=20,
        color='rgb(0, 0, 0)'
    ),
    orientation = 'v',
    marker = dict(
        color = colors[0]
    ),
    opacity = 0.7,
    name = temp_age1
)

trace2 = go.Bar(
    x = cnt_srs2.index,
    y = (100 * cnt_srs2 / cnt_srs2.sum()),
    text=['{:.1f}%'.format(percent) for percent in (100 * cnt_srs2 / cnt_srs2.sum())], 
    textposition = 'auto',
    textfont=dict(
        size=20,
        color='rgb(0, 0, 0)'
    ),
    orientation = 'v',
    marker = dict(
        color = colors[1]
    ),
    opacity = 0.7,
    name = temp_age2
)

trace3 = go.Bar(
    x = cnt_srs3.index,
    y = (100 * cnt_srs3 / cnt_srs3.sum()),
    text=['{:.1f}%'.format(percent) for percent in (100 * cnt_srs3 / cnt_srs3.sum())], 
    textposition = 'auto',
    textfont=dict(
        size=20,
        color='rgb(0, 0, 0)'
    ),
    orientation = 'v',
    marker = dict(
        color = colors[2]
    ),
    opacity = 0.7,
    name = temp_age3
)

trace4 = go.Bar(
    x = cnt_srs4.index,
    y = (100 * cnt_srs4 / cnt_srs4.sum()),
    text=['{:.1f}%'.format(percent) for percent in (100 * cnt_srs4 / cnt_srs4.sum())], 
    textposition = 'auto',
    textfont=dict(
        size=20,
        color='rgb(0, 0, 0)'
    ),
    orientation = 'v',
    marker = dict(
        color = colors[3]
    ),
    opacity = 0.7,
    name = temp_age4
)

trace5 = go.Bar(
    x = cnt_srs5.index,
    y = (100 * cnt_srs5 / cnt_srs5.sum()),
    text=['{:.1f}%'.format(percent) for percent in (100 * cnt_srs5 / cnt_srs5.sum())], 
    textposition = 'auto',
    textfont=dict(
        size=20,
        color='rgb(0, 0, 0)'
    ),
    orientation = 'v',
    marker = dict(
        color = colors[4]
    ),
    opacity = 0.7,
    name = temp_age5
)

trace6 = go.Bar(
    x = cnt_srs6.index,
    y = (100 * cnt_srs6 / cnt_srs6.sum()),
    text=['{:.1f}%'.format(percent) for percent in (100 * cnt_srs6 / cnt_srs6.sum())], 
    textposition = 'auto',
    textfont=dict(
        size=20,
        color='rgb(0, 0, 0)'
    ),
    orientation = 'v',
    marker = dict(
        color = colors[5]
    ),
    opacity = 0.7,
    name = temp_age6
)

trace7 = go.Bar(
    x = cnt_srs7.index,
    y = (100 * cnt_srs7 / cnt_srs7.sum()),
    text=['{:.1f}%'.format(percent) for percent in (100 * cnt_srs7 / cnt_srs7.sum())], 
    textposition = 'auto',
    textfont=dict(
        size=20,
        color='rgb(0, 0, 0)'
    ),
    orientation = 'v',
    marker = dict(
        color = colors[6]
    ),
    opacity = 0.7,
    name = temp_age7
)

layout = dict(
    height = 600,
    width = 800,
    title = question + ' for variuos age',
    titlefont = dict(
        size=20
    ),
    margin = dict(
#         l = 800
    ),
    xaxis=dict(
        tickfont=dict(
            size=18,
        )
    ),
    yaxis=dict(
        tickfont=dict(
            size=18,
        )
    ) 
)

data = [trace1, trace2, trace3, trace4, trace5, trace6, trace7]
fig = go.Figure(data=data, layout=layout)
print('Q:', df_reference.loc[(df_reference.Column == question), 'QuestionText'].values[0])
py.iplot(fig)


# - I made two plots(bar, sankey)  which have same information. 
# - I think plotting is like conversation. We can say same things with different languages. It makes me funny.
# - Anyway, Hobby tendency for age has a bit parabolic shape. Programming before working and after retirement may brings easy mind to do.

# # <a id='4'>4. Programming tools</a>

# ## <a id='4_1'>4.1. Language</a>

# In[18]:


def bar_plots_for_various_company(question):
    temp_company_size1 = 'Fewer than 10 employees'
    temp1 = pd.DataFrame(df_response.loc[df_response.CompanySize == temp_company_size1, question].dropna().str.split(';').tolist()).stack()
    cnt_srs1 = temp1.value_counts()

    temp_company_size2 = '10 to 19 employees'
    temp2 = pd.DataFrame(df_response.loc[df_response.CompanySize == temp_company_size2, question].dropna().str.split(';').tolist()).stack()
    cnt_srs2 = temp2.value_counts()

    temp_company_size3 = '20 to 99 employees'
    temp3 = pd.DataFrame(df_response.loc[df_response.CompanySize == temp_company_size3, question].dropna().str.split(';').tolist()).stack()
    cnt_srs3 = temp3.value_counts()

    temp_company_size4 = '100 to 499 employees'
    temp4 = pd.DataFrame(df_response.loc[df_response.CompanySize == temp_company_size4, question].dropna().str.split(';').tolist()).stack()
    cnt_srs4 = temp4.value_counts()

    temp_company_size5 = '500 to 999 employees'
    temp5 = pd.DataFrame(df_response.loc[df_response.CompanySize == temp_company_size5, question].dropna().str.split(';').tolist()).stack()
    cnt_srs5 = temp5.value_counts()

    temp_company_size6 = '1,000 to 4,999 employees'
    temp6 = pd.DataFrame(df_response.loc[df_response.CompanySize == temp_company_size6, question].dropna().str.split(';').tolist()).stack()
    cnt_srs6 = temp6.value_counts()

    temp_company_size7 = '5,000 to 9,999 employees'
    temp7 = pd.DataFrame(df_response.loc[df_response.CompanySize == temp_company_size7, question].dropna().str.split(';').tolist()).stack()
    cnt_srs7 = temp7.value_counts()
    
    temp_company_size8 = '10,000 or more employees'
    temp8 = pd.DataFrame(df_response.loc[df_response.CompanySize == temp_company_size8, question].dropna().str.split(';').tolist()).stack()
    cnt_srs8 = temp8.value_counts()
    
    colors = random_color_generator(10)



    trace1 = go.Bar(
        x = cnt_srs1.index,
        y = (100 * cnt_srs1 / cnt_srs1.sum()),
        text=['{:.1f}%'.format(percent) for percent in (100 * cnt_srs1 / cnt_srs1.sum())], 
        textposition = 'auto',
        textfont=dict(
            size=20,
            color='rgb(0, 0, 0)'
        ),
        orientation = 'v',
        marker = dict(
            color = colors[0]
        ),
        opacity = 0.7,
        name = temp_company_size1
    )

    trace2 = go.Bar(
        x = cnt_srs2.index,
        y = (100 * cnt_srs2 / cnt_srs2.sum()),
        text=['{:.1f}%'.format(percent) for percent in (100 * cnt_srs2 / cnt_srs2.sum())], 
        textposition = 'auto',
        textfont=dict(
            size=20,
            color='rgb(0, 0, 0)'
        ),
        orientation = 'v',
        marker = dict(
            color = colors[1]
        ),
        opacity = 0.7,
        name = temp_company_size2
    )

    trace3 = go.Bar(
        x = cnt_srs3.index,
        y = (100 * cnt_srs3 / cnt_srs3.sum()),
        text=['{:.1f}%'.format(percent) for percent in (100 * cnt_srs3 / cnt_srs3.sum())], 
        textposition = 'auto',
        textfont=dict(
            size=20,
            color='rgb(0, 0, 0)'
        ),
        orientation = 'v',
        marker = dict(
            color = colors[2]
        ),
        opacity = 0.7,
        name = temp_company_size3
    )

    trace4 = go.Bar(
        x = cnt_srs4.index,
        y = (100 * cnt_srs4 / cnt_srs4.sum()),
        text=['{:.1f}%'.format(percent) for percent in (100 * cnt_srs4 / cnt_srs4.sum())], 
        textposition = 'auto',
        textfont=dict(
            size=20,
            color='rgb(0, 0, 0)'
        ),
        orientation = 'v',
        marker = dict(
            color = colors[3]
        ),
        opacity = 0.7,
        name = temp_company_size4
    )

    trace5 = go.Bar(
        x = cnt_srs5.index,
        y = (100 * cnt_srs5 / cnt_srs5.sum()),
        text=['{:.1f}%'.format(percent) for percent in (100 * cnt_srs5 / cnt_srs5.sum())], 
        textposition = 'auto',
        textfont=dict(
            size=20,
            color='rgb(0, 0, 0)'
        ),
        orientation = 'v',
        marker = dict(
            color = colors[4]
        ),
        opacity = 0.7,
        name = temp_company_size5
    )

    trace6 = go.Bar(
        x = cnt_srs6.index,
        y = (100 * cnt_srs6 / cnt_srs6.sum()),
        text=['{:.1f}%'.format(percent) for percent in (100 * cnt_srs6 / cnt_srs6.sum())], 
        textposition = 'auto',
        textfont=dict(
            size=20,
            color='rgb(0, 0, 0)'
        ),
        orientation = 'v',
        marker = dict(
            color = colors[5]
        ),
        opacity = 0.7,
        name = temp_company_size6
    )
    
    trace7 = go.Bar(
        x = cnt_srs7.index,
        y = (100 * cnt_srs7 / cnt_srs7.sum()),
        text=['{:.1f}%'.format(percent) for percent in (100 * cnt_srs7 / cnt_srs7.sum())], 
        textposition = 'auto',
        textfont=dict(
            size=20,
            color='rgb(0, 0, 0)'
        ),
        orientation = 'v',
        marker = dict(
            color = colors[6]
        ),
        opacity = 0.7,
        name = temp_company_size7
    )
    
    trace8 = go.Bar(
        x = cnt_srs8.index,
        y = (100 * cnt_srs8 / cnt_srs8.sum()),
        text=['{:.1f}%'.format(percent) for percent in (100 * cnt_srs8 / cnt_srs8.sum())], 
        textposition = 'auto',
        textfont=dict(
            size=20,
            color='rgb(0, 0, 0)'
        ),
        orientation = 'v',
        marker = dict(
            color = colors[7]
        ),
        opacity = 0.7,
        name = temp_company_size8
    )
    
    layout = dict(
        height = 1000,
        title = question + ' for various company size',
        titlefont = dict(
            size=20
        ),
        margin = dict(
    #         l = 800
        ),
        xaxis=dict(
            tickfont=dict(
                size=12,
            )
        ),
        yaxis=dict(
            title = 'Count',
            tickfont=dict(
                size=12,
            )
        ) 
    )

    data = [trace1, trace2, trace3, trace4, trace5, trace6, trace7, trace8]
    fig = go.Figure(data=data, layout=layout)
    print('Q:', df_reference.loc[(df_reference.Column == question), 'QuestionText'].values[0])
    py.iplot(fig)


# In[19]:


def two_bar_plot(question1, question2, title, legend):
    temp1 = pd.DataFrame(df_response[question1].dropna().str.split(';').tolist()).stack()
    cnt_srs_1 = temp1.value_counts().sort_values(ascending=False).head(20)
    temp2 = pd.DataFrame(df_response[question2].dropna().str.split(';').tolist()).stack()
    cnt_srs_2 = temp2.value_counts().sort_values(ascending=False).head(20)

    trace1 = go.Bar(
        x = cnt_srs_1.index,
        y = cnt_srs_1.values,
        orientation = 'v',
        opacity = 0.9,
        name=legend[0]
    )

    trace2 = go.Bar(
        x = cnt_srs_2.index,
        y = cnt_srs_2.values,
        orientation = 'v',
        opacity = 0.9,
        name=legend[1]
    )

    layout = go.Layout(
        width = 800,
        height=600,
        title = title,
        titlefont = dict(
            size=20
        ),
        xaxis=dict(
            tickfont=dict(
                size=12,
            )
        ),
        yaxis=dict(
            tickfont=dict(
                size=12,
            )
        ) 
    )

    data = [trace1, trace2]
    fig = go.Figure(data=data, layout=layout)
    py.iplot(fig, filename='Platform')


# In[20]:


question1 = 'LanguageWorkedWith'
question2 = 'LanguageDesireNextYear'
title = 'Language 2018 vs 2019'
legend = ['Language 2018', 'Language 2019']

temp1 = pd.DataFrame(df_response[question1].dropna().str.split(';').tolist()).stack()
cnt_srs_1 = temp1.value_counts().sort_values(ascending=False).head(20)
temp2 = pd.DataFrame(df_response[question2].dropna().str.split(';').tolist()).stack()
cnt_srs_2 = temp2.value_counts().sort_values(ascending=False).head(20)

trace1 = go.Bar(
    y = cnt_srs_1.index[::-1],
    x = cnt_srs_1.values[::-1],
    orientation = 'h',
    opacity = 0.9,
    name = legend[0],
)

trace2 = go.Bar(
    y = cnt_srs_2.index[::-1],
    x = cnt_srs_2.values[::-1],
    orientation = 'h',
    opacity = 0.9,
    name = legend[1],
)

fig = tools.make_subplots(rows=1, cols=2)

fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 1, 2)

fig['layout'].update(height=800, width=800, title=title)
py.iplot(fig, filename='Language')


# -  Javascript is the most famous language.
# - Python will be more famous nextyear.

# In[21]:


question1 = 'LanguageWorkedWith'
question2 = 'LanguageDesireNextYear'
title = 'Language 2018 vs 2019'
legend = ['Language 2018', 'Language 2019']
two_bar_plot(question1, question2, title, legend)


# - The 2019 demand of Python, TypeScript, Swift and Go is higher than 2018. 
# - If you start programming, how about learning them?

# ## <a id='4_2'>4.2. Platform</a>

# In[22]:


question1 = 'PlatformWorkedWith'
question2 = 'PlatformDesireNextYear'
title = 'Platform 2018 vs 2019'
legend = ['Platform 2018', 'Platform 2019']
two_bar_plot(question1, question2, title, legend)


# - Cloud services such as AWS, Azure and Google Cloud is attracting many software engineers.
# - Because of IoT world, Rasberry Pi is also famous.

# ## <a id='4_3'>4.3. Database</a>

# In[23]:


question1 = 'DatabaseWorkedWith'
question2 = 'DatabaseDesireNextYear'
title = 'Database 2018 vs 2019'
legend = ['Database 2018', 'Database 2019']
two_bar_plot(question1, question2, title, legend)


# * - Aws, Google Cloud is also important service for database.

# ## <a id='4_4'>4.4. Framework</a>

# In[24]:


question1 = 'FrameworkWorkedWith'
question2 = 'FrameworkDesireNextYear'
title = 'Framework 2018 vs 2019'
legend = ['Framework 2018', 'Framework 2019']
two_bar_plot(question1, question2, title, legend)


# - React, Tensorflow and Pytorch is soaring in popularity.

# ## <a id='4_5'>4.5. IDE</a>

# In[25]:


bar_horizontal_plot(choice_type='multiple', question='IDE', width=800, height=700, left=200)


# - There are many IDE in the world. Surprisingly, vim is still famous tool. I also use Vim when coding on my server or my lab server. 
# - Let's use favorite IDE. All IDE is useful. The most important thing is our programming skill!

# ## <a id='4_6'>4.6. CommunicationTools</a>

# In[26]:


bar_horizontal_plot(choice_type='multiple', question='CommunicationTools', width=800, height=700, left=400)


# - Slack is the most popular communication tool.
# - I want to see this subject more deeply. Let's check the change with differing companysize.

# ## <a id='4_7'>4.7. CommunicationTools for various companySize</a>

# In[27]:


bar_plots_for_various_company('CommunicationTools')


# - Interestingly, The larger companySize, The less frequency of use slack, Trello, SNS(Google Hangouts, Facebook).
# - Inversely, The larger companySize, The more frequency of use a formal  documentations(wick, Office).

# ## <a id='4_8'>4.8. VersionControl</a>

# In[28]:


bar_horizontal_plot(choice_type='multiple', question='VersionControl', width=800, height=600, left=200)


# - Gits is the most popular version control tool. 

# ## <a id='4_9'>4.9. VersionControl for various companySize</a>

# In[29]:


bar_plots_for_various_company('VersionControl')


# - The larger companySize, The more frequency of use Subversion, Tem Foundation VersionControl.

# # <a id='5'>5. equipments</a>

# ## <a id='5_1'>5.1. NumberMonitors</a>

# In[30]:


question = 'NumberMonitors'
temp1 = pd.DataFrame(df_response[question].dropna().str.split(';').tolist()).stack()
cnt_srs = temp1.value_counts()

trace = go.Pie(
    labels = cnt_srs.index,
    values = (100 * cnt_srs / cnt_srs.sum()).values,

     textfont=dict(
         size=20,
         color='rgb(0, 0, 0)'
    ),
    marker = dict(
        colors = random_color_generator(20),
    ), 
    opacity = 0.7
)

layout = dict(
    title = question,
    titlefont = dict(
        size=20
    ),
    width = 800,
    height = 500
)

data = [trace]
fig = go.Figure(data=data, layout=layout)
print('Q:', df_reference.loc[(df_reference.Column == question), 'QuestionText'].values[0])

py.iplot(fig)


# - More than I thought,  one monitor people is quite

# ## <a id='5_2'>5.2. ErgonomicDevices</a>

# In[31]:


bar_horizontal_plot(choice_type='multiple', question='ErgonomicDevices', width=800, height=400, left=200)


#  - Ergonomic keyboard, mouse, stading desk is also goods I need!

# # <a id='6'>6. Work</a>

# ## <a id='6_1'>6.1. JobSatisfaction</a>

# In[32]:


question = 'JobSatisfaction'
temp1 = pd.DataFrame(df_response[question].dropna().str.split(';').tolist()).stack()
cnt_srs = temp1.value_counts()

trace = go.Pie(
    labels = cnt_srs.index,
    values = (100 * cnt_srs / cnt_srs.sum()).values,

     textfont=dict(
         size=20,
         color='rgb(0, 0, 0)'
    ),
    marker = dict(
        colors = random_color_generator(20),
    ), 
    opacity = 0.7
)

layout = dict(
    title = question,
    titlefont = dict(
        size=20
    ),
    width = 800,
    height = 500
)

data = [trace]
fig = go.Figure(data=data, layout=layout)
print('Q:', df_reference.loc[(df_reference.Column == question), 'QuestionText'].values[0])

py.iplot(fig)


# - Generally, many respondents are satisfactory for their job.

# ## <a id='6_1'>6.2. JobSatisfaction to various company size</a>

# In[33]:


bar_plots_for_various_company('JobSatisfaction')


# - A company which have Fewer than 10 employees has the highest position in both "Extremely satisfied" and "Extremely disstisfied". It is interesting. 

# ## <a id='6_3'>6.3. JobSatisfaction by Gender</a>

# In[34]:


bar_plots_for_various_company('Gender')


# - There are few change between various genders for jobsatisfaction.

# # <a id='7'>7. Artificial inteligence</a>

# In[35]:


df_reference[df_reference.Column.map(lambda x: True if 'AI' in x else False)]


# ## <a id='7_11'>7.1. AIDangerous</a>

# In[36]:


bar_horizontal_plot(choice_type='multiple', question='AIDangerous', width=800, height=400, left=500)


# - Can AI surpass people? or Can people who makes AI surpass people?

# ## <a id='7_2'>7.2. AIInteresting</a>

# In[37]:


bar_horizontal_plot(choice_type='multiple', question='AIInteresting', width=800, height=400, left=500)


# - Automation on Job and decision is the most interesting aspect of AI for stack over flow people.

# ## <a id='7_3'>7.3. AIResponsible</a>

# In[38]:


bar_horizontal_plot(choice_type='multiple', question='AIResponsible', width=800, height=400, left=300)


# - Any creator has a copyright on own creature. I think AI is same as the others.[](http://)

# ## <a id='7_4'>7.4. AIFuture</a>

# In[39]:


bar_horizontal_plot(choice_type='multiple', question='AIFuture', width=800, height=400, left=500)


# # <a id='8'>8. Daily life</a>

# ## <a id='8_1'>8.1. WakeTime</a>

# In[40]:


bar_horizontal_plot(choice_type='multiple', question='WakeTime', width=800, height=500, left=200)


# - I think It can be more easy to see how software engineer sleep by varying companySize.

# ## <a id='8_2'>8.2. WakeTime for various companySize</a>

# In[41]:


bar_plots_for_various_company('WakeTime')


# - There is a tendency that the larger companysize, the earlier wakeTime. 

# ## <a id='8_3'>8.3. Exercise</a>

# In[42]:


bar_horizontal_plot(choice_type='multiple', question='Exercise', width=800, height=400, left=200)


# - Relatively, many software engineer do exercise.

# ## <a id='8_4'>8.4. CheckInCode</a>

# In[43]:


bar_horizontal_plot(choice_type='multiple', question='CheckInCode', width=800, height=400, left=300)


# - About 10% people don't check code frequently. Let's check your repository!

# ## <a id='8_5'>8.5. HoursComputer</a>

# In[44]:


bar_horizontal_plot(choice_type='multiple', question='HoursComputer', width=800, height=400, left=300)


# ## <a id='8_6'>8.6. HoursComputer for various companySize</a>

# In[45]:


bar_plots_for_various_company('HoursComputer')


# - Programmers who work in small company tend to work more.

# ## <a id='8_7'>8.7. SkipMeals</a>

# In[46]:


bar_horizontal_plot(choice_type='multiple', question='SkipMeals', width=800, height=400, left=200)


# ## <a id='8_8'>8.8. SkipMeals for various companySize</a>

# In[47]:


bar_plots_for_various_company('SkipMeals')


# - Programmers who work in small company tend to skip meals frequently.

# # <a id='9'>9. Salary</a>

# ## <a id='9_1'>9.1. Salary by country with considering salaries less than 75% quartile</a>

# In[48]:


countries = df_response['Country'].value_counts().index[:10]
df = df_response[df_response['Country'].isin(countries)]
plt.figure(figsize=(15, 12))
sns.boxplot(x='ConvertedSalary', y='Country', data=df[df.ConvertedSalary < df.ConvertedSalary.quantile(.75)])
plt.title('Sarary Comparison', y = 1.02)
plt.show()


# - United states and Australia respondents got high salaries.

# In[49]:


salary_medians = df_response[['Country', 'ConvertedSalary']].dropna().groupby('Country')['ConvertedSalary'].median().sort_values(ascending=False)
data = [dict(
        type='choropleth',
        locations = salary_medians.index,
        locationmode ='country names',
        z = salary_medians.values,
        text = salary_medians.index,
        colorscale = 'Red',
        marker = dict(
            line = dict(
                color = 'rgb(180, 180, 180)',
                width =0.5
            )
        ),
        colorbar = dict(
            tickprefix = '',
            title = 'Median Salary ($)',
        )
)]

layout = dict(
    title = 'Median salary ($) in Earth',
    geo = dict(
        showframe = False,
        projection = dict(
            type = 'Mercatorodes'
        )
    ),
    height = 800,
#     width = 800,
)

fig = dict(data=data, layout=layout)
py.iplot(fig)


# ## <a id='9_2'>9.2. Salary by companySize with considering salaries less than 75% quartile</a>

# In[50]:


plt.figure(figsize=(15, 12))
sns.boxplot(x='ConvertedSalary', y='CompanySize', data=df_response[df_response.ConvertedSalary < df_response.ConvertedSalary.quantile(.75)],
              order=['Fewer than 10 employees', '10 to 19 employees', '20 to 99 employees', '100 to 499 employees', '500 to 999 employees',
                    '1,000 to 4,999 employees', '5,000 to 9,999 employees', '10,000 or more employees'])
plt.title('Sarary Comparison', y = 1.02)
plt.show()


# - The larger companySize, The lower salary

# ## <a id='9_3'>9.3. Salary by age with considering salaries less than 75% quartile</a>

# In[51]:


plt.figure(figsize=(15, 12))
sns.boxplot(x='ConvertedSalary', y='Age', data=df_response[df_response.ConvertedSalary < df_response.ConvertedSalary.quantile(.75)],
              order=['Under 18 years old', '18 - 24 years old', '25 - 34 years old', '35 - 44 years old', '45 - 54 years old',
                    '55 - 64 years old', '65 years or older'])
plt.title('Sarary Comparison', y = 1.02)
plt.show()


# - Salary tendency for various age shows a quadratic pattern. But some young engineer earn a bit high salary than the older engineer.

# # PlotLy is So funny!!
