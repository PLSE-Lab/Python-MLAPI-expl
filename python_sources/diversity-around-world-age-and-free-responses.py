#!/usr/bin/env python
# coding: utf-8

# # <center>Diversity around world, age and free responses</center>

# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The world is diverse by various aspects. Using 2018 Kaggle Survey data, we will explore how diverse it is among countries and between different age groups. Also, we will explore free responses given by survey takers that couldn't find their desired option.<br>
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;It is my attempt to make visualization more informative, making data speak for itself with the use of interactive plots, also making data easy to absorb.<br>
# ### Content
# 1. [Diversity around world in data science.](#chap1)
# 2. [Age is just not any number.](#chap2)
# 3. [Exploring free responses.](#chap3)

# In[ ]:


import pandas as pd
import numpy as np

from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go
from plotly import tools

init_notebook_mode(connected=True)

import warnings
warnings.filterwarnings('ignore')

#print(__version__) 
import os
#print(os.listdir('kaggle_survey/'))


# In[ ]:


mcq = pd.read_csv('../input/multipleChoiceResponses.csv')

col_replace = {'Time from Start to Finish (seconds)' : 'Time required'}
col_tuple = [(14, 21), (22, 28), (29, 44), (45, 56), (57, 64), (65, 83), (88, 107), (110, 123), (130, 150), (151, 194), (195, 223), (224, 249), (250, 262), (265, 276), (277, 283), (284,290), (291, 304), (307, 329), (336, 341), (343, 349), (349, 355), (356, 371), (373, 385), (386, 394)]

for i in col_tuple:
    for j in range(i[0], i[1]):
        col_replace[mcq.columns[j]] = mcq.columns[j][:3] + '_' + mcq[mcq.columns[j]].iloc[0][mcq[mcq.columns[j]].iloc[0].rindex('- ')+2:]

mcq.drop(index = 0, inplace = True)
mcq.rename(columns = col_replace, inplace = True)

mcq['Q3'] = mcq['Q3'].replace('Iran, Islamic Republic of...', 'Iran') 


code_dict = {'Argentina': 'ARG',
 'Australia': 'AUS',
 'Austria': 'AUT',
 'Bangladesh': 'BGD',
 'Belarus': 'BLR',
 'Belgium': 'BEL',
 'Brazil': 'BRA',
 'Canada': 'CAN',
 'Chile': 'CHL',
 'China': 'CHN',
 'Colombia': 'COL',
 'Czech Republic': 'CZE',
 'Denmark': 'DNK',
 'Egypt': 'EGY',
 'Finland': 'FIN',
 'France': 'FRA',
 'Germany': 'DEU',
 'Greece': 'GRC',
 'Hungary': 'HUN',
 'India': 'IND',
 'Indonesia': 'IDN',
 'Iran': 'IRN',
 'Ireland': 'IRL',
 'Israel': 'ISR',
 'Italy': 'ITA',
 'Japan': 'JPN',
 'Kenya': 'KEN',
 'Malaysia': 'MYS',
 'Mexico': 'MEX',
 'Morocco': 'MAR',
 'Netherlands': 'NLD',
 'New Zealand': 'NZL',
 'Nigeria': 'NGA',
 'Norway': 'NOR',
 'Pakistan': 'PAK',
 'Peru': 'PER',
 'Philippines': 'PHL',
 'Poland': 'POL',
 'Portugal': 'PRT',
 'Romania': 'ROU',
 'Russia': 'RUS',
 'Singapore': 'SGP',
 'South Africa': 'ZAF',
 'Spain': 'ESP',
 'Sweden': 'SWE',
 'Switzerland': 'CHE',
 'Thailand': 'THA',
 'Tunisia': 'TUN',
 'Turkey': 'TUR',
 'Ukraine': 'UKR',
 'Hong Kong (S.A.R.)': 'HKG',
 'Republic of Korea': 'PRK',
 'South Korea': 'KOR',
 'United Kingdom of Great Britain and Northern Ireland': 'GBR',
 'United States of America': 'USA',
 'Viet Nam': 'VNM',
 'I do not wish to disclose my location': 'Do not wish to disclose',
 'Other': 'OTHER'}

mcq['Q3_CODE'] = mcq['Q3'].apply(lambda l : code_dict[l])

#mcq.head()


# In[ ]:


default_codes = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#7cfc00', '#ffa500', '#ff1493', '#adff2f', '#0000cd']

def pie_with_bar(x, y, labels, values, title, xtitle, ytitle, dx = [0.20, 1], dy = [0.20, 1], showlegend = True, legend_pos = 'v', rotation = 0):
    
    
    if legend_pos == 'v':
        legend = dict(orientation = 'v')
    else:
        legend = dict(orientation = 'h', x = 0, y = 0)

    trace1 = go.Bar(x = x, 
                    y = y, 
                    text = y,
                    hoverinfo = 'text',
                    marker = dict(color = default_codes),
                    textposition = 'auto',
                    showlegend = False)


    trace2 = go.Pie(labels = labels, 
                    values = values, 
                    domain = dict(x = dx, 
                                  y = dy),
                    hoverinfo = 'label+percent',
                    marker = dict(colors = default_codes),
                    hole = 0.40,
                    sort = False,
                    showlegend = showlegend,
                    rotation = rotation)

    layout = go.Layout(dict(title = title,
                           xaxis = dict(title = xtitle),
                           yaxis = dict(title = ytitle),
                           legend = legend))
    fig = dict(data = [trace1, trace2], layout = layout)
    iplot(fig)
    
def stacked_bar(index, column, title, legend_pos = 'v', extra_suffix = 'Object', showlegend = True):
    c_mat = count_percent_mat(index, column, 12, extra_suffix)
    p_mat = c_mat[1]
    c_mat = c_mat[0]
    data = []
    
    if legend_pos == 'v':
        legend = dict(orientation = 'v')
    else:
        legend = dict(orientation = 'h')
    for i in c_mat.columns:
        data.append(go.Bar(x = p_mat.index,
                           y = p_mat[i],
                           name = i,
                           text = c_mat[i].apply(str) + '<br>' + p_mat[i].apply(lambda l : format(l, '.2f')) + '%',
                           hoverinfo = 'text+name',
                           showlegend = showlegend)
                   )
    layout = go.Layout(dict(barmode = 'stack',
                           title = title,
                           yaxis = dict(title = 'Percentage'),
                           legend = legend))
    fig = go.Figure(data, layout)
    iplot(fig)
    
def multi_stacked_bar(index, column, title, legend_pos = 'v', extra_suffix = 'Objects'):
    c_mat = multi_count_percent_mat(index, column, 12, extra_suffix)
    p_mat = c_mat[1]
    c_mat = c_mat[0]
    data = []
    if legend_pos == 'v':
        legend = dict(orientation = 'v')
    else:
        legend = dict(orientation = 'h')
    for i in c_mat.columns:
        data.append(go.Bar(x = p_mat.index,
                           y = p_mat[i],
                           name = i,
                           text =  c_mat[i].apply(str) + '<br>' + p_mat[i].apply(lambda l : format(l, '.2f')) + '%',
                           hoverinfo = 'text+name',
                           showlegend = True)
                   )
    layout = go.Layout(dict(barmode = 'stack', 
                           title = title,
                           yaxis = dict(title = 'Percentage'),
                           legend = legend))
    fig = go.Figure(data, layout)
    iplot(fig)


from sklearn.preprocessing import LabelEncoder

def draw_map(index, title):
    if type(index) == str:
        c_mat = count_percent_mat('Q3', index)
    else:
        c_mat = multi_count_percent_mat('Q3', index)
    p_mat = c_mat[1].transpose()
    c_mat = c_mat[0].transpose()

    del c_mat['Other']
    del c_mat['I do not wish to disclose my location']
    del p_mat['Other']
    del p_mat['I do not wish to disclose my location']
    
    c_mat.sort_index(inplace = True)
    p_mat.sort_index(inplace = True)

    l = LabelEncoder()
    l.fit(c_mat.index)


    c_list = []
    l_list = []
    z_list = []
    t_list = []

    for i in c_mat.columns:
        c_list.append(i)
        z_list.append(l.transform([c_mat[i].idxmax()])[0])
        t = i+'<br>Max count, '+c_mat[i].idxmax()+' : '+str(max(c_mat[i]))+', '+format(max(p_mat[i]), '.2f')+' %'
        for (x, y, z) in zip(c_mat.index, c_mat[i], p_mat[i]):
            t += '<br>'+x+' : '+str(y)+', '+format(z, '.2f')+' %'
        t_list.append(t)

    l_list = list(map(lambda l : code_dict[l], c_list))

    data = dict(type='choropleth',
                locations = l_list,
                z = z_list,
                text = t_list,
                hoverinfo = 'text',
                autocolorscale = False,
                colorscale = 'Jet',
                showscale = False
                ) 
    
    title = '<b>' + title + '</b><br>Hover over for more details'
    layout = dict(title = title,
                  geo = dict(
                showframe = False,
                showcoastlines = False,
                showocean = True,
                oceancolor = '#3f3f4f',
                projection = dict(
                type = 'robinson')))
    choromap = go.Figure(data = [data],layout = layout)
    iplot(choromap)

def box_dist(index, columns, title):
    columns = list(columns)
    columns.append(index)
    d = mcq[columns].sort_values(index)
    columns.remove(index)

    traces = []

    for i in range(len(columns)):
        traces.append(go.Box(
                             x = d[index],
                             y = d[columns[i]],
                             fillcolor = default_codes[i],
                             showlegend = False))

    columns = list(map(lambda l : l[4:], columns))
    for i in range(len(columns)):
        if len(columns[i]) > 30:
            t = list(map(lambda l : l + ' ', columns[i].split()))
            j = 0
            k = 0
            columns[i] = ''
            while j <= 30:
                j += len(t[k])
                j += 1
                k += 1
            columns[i] = ''.join(t[:k - 1]) + '<br>' + ''.join(t[k - 1:])
            columns[i].rstrip()
    
    fig = tools.make_subplots(rows = len(columns) // 2,
                              cols = 2,  
                              shared_xaxes = True,
                              subplot_titles = columns,
                              vertical_spacing = 0.05)
                  
    for i in range(len(traces)):
           fig.append_trace(traces[i], (i // 2) + 1, (i % 2) + 1)
        
    fig['layout'].update(height = 1200, width = 900, title = title)        
    iplot(fig)            


# for single column

def count_percent_mat(index, column, limit = 0, suffix = 'Objects'):

    group = mcq.groupby([column, index]).count()['Time required']

    indexu = mcq[index].unique()
    indexu = indexu[~pd.isnull(indexu)]
    indexu.sort()
    colu = mcq[column].unique()
    colu = colu[~pd.isnull(colu)].tolist()
    
    if limit == 0 or limit >= len(colu):
        limit = len(colu)

    col_list = mcq.groupby(column).count().sort_values('Time required', ascending = False).index.tolist()
    if 'Other' in col_list:
        col_list.remove('Other')
        col_list.append('Other')
    col_list = col_list[:limit]

    col_len = len(col_list)

    if limit > 0 and limit < len(colu):
        others = 'Other ' + suffix
        col_list += [others]
        
    count_mat = pd.DataFrame(np.zeros((len(indexu), len(col_list))), index = indexu, columns = col_list)
    per_mat = pd.DataFrame(np.zeros((len(indexu), len(col_list))), index = indexu, columns = col_list)

    for i in range(limit):
        for j in group.loc[col_list[i]].index:
            count_mat.loc[j][col_list[i]] = group.loc[col_list[i]][j]
        colu.remove(col_list[i])

    # for 'other<suffix>' columns, if limit 0, nothing left in colu
    for i in colu:
        for j in group.loc[i].index:
            count_mat.loc[j][others] += group.loc[i][j]
    
    for i in count_mat.index:
        total = sum(count_mat.loc[i])
        for j in count_mat.columns:
            per_mat.loc[i][j] = (count_mat.loc[i][j] / total) * 100
    
    return (count_mat, per_mat)


# for (select all that apply) questions
# below method calculates percentage w.r.t. total non-nan values.
# For example, if a group has members a, b, c, d and there an attribute T with multiple selectable values x, y and z.
# a selects x and y, b selects y, z, c selects all x, y, z.
# percentage users of x is 50.0%, y is 75.0%, z is 50.0%.
# But with multiple options selected, it becomes more as a set problem and a bit complicated to interpret with large number of values of T.
# Thus, I have simply considered non - nan values for each values of x, y, z, and calculate percentage w.r.t. sum of all those non - nan values.
# This gives more direct comparison between x, y, z where I calculate which value (x, y, z) has majority.
# It also dissloves an uncertainty of 'd' not selecting any value, which is possible if question never appeared to 'd' as d could have selected negative answer to some 
# previous question OR could have selected values like 'None' or 'Other' which I have not considered for some questions. 
# Also makes easy for comparison between different groups like (a, b, c, d). 

def multi_count_percent_mat(index, columns, limit = 0, suffix = 'Objects'):

    count_mat = mcq.groupby(index).count()[columns]
    count_mat.columns = list(map(lambda l : l[4:], count_mat.columns))

    if limit >= len(count_mat.columns):
        limit = 0

    if limit > 0 and limit < len(count_mat.columns):
        l = []
        for i in count_mat.columns:
            l.append(sum(count_mat[i]))
        
        count_mat.loc['Total'] = l
        count_mat.sort_values('Total', axis = 1, ascending = False, inplace = True)
        if 'Other' in count_mat.columns:
            t = count_mat['Other']
            del count_mat['Other']
            count_mat['Other'] = t
        
        others = 'Other '+suffix
        count_mat[others] = np.zeros(len(count_mat))
    
        for i in count_mat.columns[limit:-1]:
            count_mat[others] += count_mat[i]
            del count_mat[i]
        count_mat.drop(index = 'Total', inplace = True)  

    per_mat = pd.DataFrame(np.zeros((len(count_mat.index), len(count_mat.columns))), index = count_mat.index, columns = count_mat.columns)
    for i in count_mat.index:
        total = sum(count_mat.loc[i])
        for j in count_mat.columns:
            per_mat.loc[i][j] = (count_mat.loc[i][j] / total) * 100
    return (count_mat, per_mat)


# <a id='chap1'></a>
# # Chapter 1 - Diversity around world in data science

# Our world is diverse. There is lot of diversity in ideas, what we like, what tools we use most, .As the name suggests, in this story, we will explore what are most popular answers for our survey questions across world. Mostly discussing about what features/tools/trends appear in different parts of world. 

# Starting with number count.

# In[ ]:


data = mcq.groupby('Q3').count()[['Time required']]
data.sort_values('Time required', ascending = False, inplace = True)

t = data.loc['Other']
data.drop(index = 'Other', inplace = True)
data = data.append(t)

data.drop(index = 'I do not wish to disclose my location', inplace = True)

c = 0
for i in data.index[13:]:
    c += data.loc[i]['Time required']

data = data[:13]
t.name = 'Rest of World'
t['Time required'] = c
data = data.append(t)

pie_with_bar(data.index, data['Time required'], data.index, data['Time required'], 'Countries', '', 'Count', [0.35, 1], [0.25, 0.9], showlegend=True, legend_pos = 'h', rotation = 180)


# Most of survey takers are from **United States of America (4716, 20.1%), followed by India (4417, 18.8%), China (1644, 7.01%), Russia (879, 3.75%), Brazil (736, 3.14%), Germany (734, 3.13%), United Kingdom of Great Britain and Northern Ireland (702, 2.99%), France (604, 2.57%), Canada (604, 2.57%), Japan (597, 2.54%), Spain (485, 2.07%), Italy (355, 1.51%), Australia (330, 1.41%) and all 44 remaining countries contribute (6662, 28.4%) records.**<br> Note that records in which **people have not disclosed their location** are not considered. Also, for this above graph **Other** option is considered. <br>However, for rest of this chapter **both of these categories are excluded.** 

# In[ ]:


draw_map('Q1', 'Gender')


# The **count of survey takers across world** favours men with **average percentage of men and women across world being 82.54% and 15.89% respectively.** However, it is worth noting that **some of countries have fine female% to male% ratio are Tunisia (31.08% - female, 67.56% - male), Malaysia (26.54%, 71.68%), Morocco (23.94%, 76.05%), Iran (23.89%, 75.22%), United States of America (22.94%, 74.85%), Egypt (22.91%, 76.04%), Romania (22.78%, 77.21%), Singapore (22.04%, 75.80%) and so on...
# <br>View below output for complete details...**
# 

# In[ ]:


d = count_percent_mat('Q3', 'Q1')[1]
d.sort_values('Female', ascending = False)
# to hide O/P


# In[ ]:


draw_map('Q4', 'Highest Level of Education')


# Most of survey takers in **Europe, North America, South America (excluding Brazil), South Africa, Morocco, Tunisia, Iran, Pakistan, China, South Korea, Thailand, Japan and New Zealand have Master's Degree, while Brazil, Nigeria, Egypt, Kenya, India, South East Asian Countries (excluding Thailand), Republic of Korea have Bachelors's Degree.** Percentage wise top 5 countries having **Doctoral Degree** are **Morocco (28.98%), Switzerland (28.65%), Germany (27.07%), Iran (25.92%) and United Kingdom of Great Britain and Northern Ireland (25.5%)**, having **Master's Degree** are **France (70.08%), Iran (66.66%), Belgium (64.86%), Poland (61.35%) and Spain (59.74%),** and those having **Bachelor's Degree** are **Nigeria (58.17%), Kenya (52.94%), Egypt (51.61%), Vietnam (51.06%), Indonesia and India (both 50%)** of survey takers.

# In[ ]:


draw_map('Q5', 'Undergraduate Major')


# Survey takers from **South Africa (29.08%) and New Zealand(25.97%)** come from **Mathematics/Statistics** background. Otherwise, most of them around world come from **Computer Science** background.

# In[ ]:


draw_map('Q6', 'JobTitles')


# Majority of survey takers from **Canada, Peru, Brazil, Morocco, Nigeria, Tunisia, Egypt, Kenya, Portugal, Ireland, Switzerland, Italy, Denmark, Iran, Indian-Subcontinent, China, Republic of Korea, South Korea, Japan, South East Asia (excluding Thailand and Philippines) and Australia are Students**, while those from **United States, Mexico, Chile, Colombia, Argentina, Great Britain, France, Spain, Belgium, Netherlands, Germany, Poland, Austria, Hungary, Belarus, Norway, Sweden, Finland, Russia, Turkey, Greece, South Africa and Thailand have Data Scientists as their job title.** Also, those from **Philippines are Data Analyst and those from Romania, Czech Republic and Ukraine are Software Engineer.** 

# In[ ]:


draw_map('Q8', 'Years of Experience in current role')


# Majority of survey takers from **Mexico, Belarus, Hungary and Iran have 1-2 years experience**, and **from Colombia have 2-3 years experience**, while those **from Argentina with have 5-10 years experience (21.24%)**. Otherwise **rest of world has majority with 0-1 years experience.**

# In[ ]:


draw_map(mcq.columns[29:44], 'IDEs (by use)')


# In **New Zealand**, majority of user use **RStudio (38, 15.97%) and Jupyter/IPython Notebooks (37, 15.55%).** While rest of world use **Jupyter/IPython Notebooks with global average of 17.4%**

# In[ ]:


draw_map(mcq.columns[45:54], 'Hosted Notebooks (by use)')


# It is obvious to see **Kaggle Kernels** used widely with **global average 33%.**, followed by **JupyterHub  (25.46%), Google Colab (20.43%), Azure Notebook (7.88%), Google Cloud Datalab (7.19%) and so on.**

# In[ ]:


draw_map(mcq.columns[57:62], 'Cloud Services (by use)')


# Excluding people who don't use cloud service or have given free responses, **AWS emerges as most popular cloud service among survey takers with global average 37.90%, followed by Google Cloud Platform (GCP) 28.81%, Microsoft Azure 23.55%, IBM Cloud 8.11% (widely used in Morocco 45.23%) and Alibaba Cloud 1.61% which used widely in China (48.26%).**

# In[ ]:


draw_map('Q17', 'Programming Languages')


# The result seems to be copy of that for IDEs, with more use of **R (16, 28.57%) than Python (14, 25.0%)**. Otherwise, it is favours Python. Also, top 5 countries of which survey takers use R are in **Kenya (33.3% - R, 36.6% Python), Finland (30% - R, 58% - Python), Republic of Korea (28.94% - R, 55.26% - Python), New Zealand (28.57% - R, 25% - Python) and Malaysia (27.05% - R, 34.11% - Python).**   

# In[ ]:


draw_map('Q18', 'Recommended Language to young aspirants')


# This result is more obvious as **Python is most easy language to use.** It has got **global average percentage of 75.03%** and from survey takers **in New Zealand 63.93% recommend it.** It is followed by **R with global average of 13.8%.**

# In[ ]:


draw_map('Q20', 'ML Framework')


# **Sci-kit Learn** is widely preferred with **global average of 45.22%.** Also, survey takers from **Republic of Korea prefer Keras (30.3% - Keras, 12.12% - Scikit Learn) and those from South Korea use Tensorflow (32.67% - Tensorflow, 29.70%) than Sci-kit Learn.**

# In[ ]:


draw_map('Q22', 'Data Visualization Libraries')


# **Matplotlib**, a low level Data Visualization Library in Python , is widely used with **global average of 55%, followed by ggplot2 (24.08%).** Also, survey takers from **Australia and New Zealand** prefer **ggplot2**, a Data Visualization package, in R with percentage count of **(40.43% - ggplot2, 35.11%) and (51.35% - ggplot2, 24.32 - Python), respectively. 

# In[ ]:


draw_map('Q23', 'Time spent on coding')


# **Most of survey takers prefer spending 25%-49% and 50%-74% of their time on coding**. But, survey takers in **Austria, Argentina, Bangladesh and Japan** seem to spent **1%-25%**. However, it is worth noting that for **Austria** count for **1%-25% is 13 and that for 50%-74% is 12.**

# In[ ]:


draw_map('Q24', 'Years writing code to analyse data')


# With a quite mixture of **<1 year , 1-2 year and 3-5 year,** majority of people are writing code to analyse data for **1-2 years (29.36%),** followed by **<1 year (24.09%) and 3-5 years(22.04%).**

# In[ ]:


draw_map('Q25', 'Years using Machine Learning Algorithm')


# With most of survey takers from **United States and some countries in Europe** have **1-2 years** experience of using ML algorithm, **rest of the world** has majority with **<1 years** followed by those having **1-2 years of experience.**

# In[ ]:


draw_map(mcq.columns[130:148], 'Cloud Computing Framework (by use)')


# As **AWS** happened to be most popular cloud service, it's **EC2 cloud computing framework** is preferred by most of survey takers, followed by **Google Compute Engine in Peru, Chile, Nigeria, Kenya, Egypt, Turkey, Thailand and Vietnam,** and **Azure Virtual Machine in Norway**, and, **Google App Engine in Austria**, and **IBM Virtual Cloud Servers in Morocco.**

# In[ ]:


draw_map(mcq.columns[195:221], 'Relational Database (by use)')


# Majority of survey takers use **MySQL with global average of 23.48%**, followed by **PostgresSQL (14.9%) and SQLite (14.52%).** 

# In[ ]:


draw_map(mcq.columns[224:247], 'BigData and Analytics Products (by use)')


# Most of survey takers use **Google BigQuery with global average 15.85%**, followed by **DataBricks (10.12%), AWS Redshift (8.48%), Microsoft Analysis Service (6.37%), Teradata (6.02%), AWS Elastic MapReduce (5.98%), and so on.**

# In[ ]:


draw_map('Q32', 'Type of Data')


# With most of survey takers work on **Numerical data with global average 25.36%,**, followed by **Tabular Data (18.73%) used by South Africa (24.53%), Kenya (32.08%), Ireland (32.31%), Finland (30.61%), Hungary (30.65%), Czech Republic (26.19%) and Belgium (22.22%),** and **Text data (14.02%) in Tunisia (23.33%) and Pakistan (23.53%)** and **Image Data (12.44%) in Belarus (30.0%), China (25.15%) and Vietnam (26.83%)** and **Time Series data (11.96%) in Austria(23.53%)**

# In[ ]:


draw_map(mcq.columns[265:274], 'Find Public Datasets (by use)')


# Most of survey takers find datasets on **Dataset aggregation sites like Kaggle with global average 18.15%**, followed by **Google Search (16.23%) and Github (12.94%).** Also, those from **Netherands collect their own data (through web scraping, etc) (17.89%)**. And, it is good to know that survey takers **find datasets on government websites in Australia (15.91%) and New Zealand (17.51%).**   

# In[ ]:


draw_map('Q37', 'Online Learning Platform')


# **Coursera** is popular among most of survey takers **with global average (28.35%)**, followed by **DataCamp (33.33%)** popular in South East Asian Countries, **Udemy (11.36%)**, **edX (9.23%)**, and so on.

# In[ ]:


draw_map('Q40', 'Independent Projects v/s Academic Achievements')


# Survey takers from **Canada, African Countries, Finland, Iran, Pakistan, India, New Zealand, Republic of Korea and Phillippines** highly prefer **Independent Project**, while those in **United States, South Asian Countries, Australia, and some countries in Europe** slightly prefer **Independent Project**. The rest of world have **equal importance**. Thus, we can conclude that **survey takers think Independent Projects demonstrate more expertise in Data Science than Academics Achievements.**

# In[ ]:


draw_map(mcq.columns[336:341], 'Metrics that determine ML model\'s sucess (by use)')


# Most of survey takers prefer **metrics that consider accuracy.** Even those in Peru have preferred **Revenue and/or business goals are 18**, those who prefer **metrics that consider accuracy are 17.** 

# **Conclusion :** On most of the aspects the world seems less diverse. However, R being originated in New Zealand has quite a bit of influence than Python, which unanimously most popular. Practical knowledge is gaining more importance as compared to academic acheivements. Participation of students and recents grads is quite in countries where Data Science and ML as a work field have recently emerged and booming.

# <a id="chap2"></a>
# # Chapter 2 -  Age is not just any number

# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;For recent years, data science being among hot tech jobs and also increase in computer based solutions, has attracted lot of recent graduates, mostly with computer science background. In this chapter, we will explore how this new addition has changed use of tech., promoted by this young grads. 
# <br>The age group given in this datasets are 

# In[ ]:


d = mcq['Q2'].unique()
d.sort()
for i in d[:-2]:
    print(i, end = ', ')
print(d[-2], 'and', d[-1],'\b.')


# Lets check how numbers of people are distributed among these age groups.

# In[ ]:


data = mcq.groupby('Q2').count()
pie_with_bar(data.index, data['Time required'], data.index, data['Time required'], 'Age-wise count', 'Age Group', 'Count', [0.55, 1], showlegend=True)


# For most of data scientists on Kaggle being of age between 25-29 (25.8%), are followed by group 22-24, 30-34, 18-21 (21.5%, 15.8%, 12.7% respectively) and so on. **If age was a numerical attribute, it's distribution would look like a left leaning curve.** <br>
# 

# In[ ]:


draw_map('Q2', 'What country has what majority of age group on Kaggle?')


# It is clear to say that most of countries have majority of 25-29 age group data scientists. Followed by 22-24 for countries like **China, Romania, Peru, Chile, Egypt, Tunisia, Morocco, Pakistan, and countries in South East Asia.** With countries **Australia, New Zealand, Belgium and Hungary** having majority age group 30-34 highly experienced people, only **India, Belarus and Ukraine** have 18-21 majority saying how popular data science is among college go-ers and recent graduates like me.

# In[ ]:


stacked_bar('Q2', 'Q4', 'Highest Level of Education', legend_pos = 'h')


# It is very obvious that 18-21 group has majority of **Bachelor's Degree, 64%.** I wonder how Master's and Doctoral degree appear in this group, but otherwise most of remaining groups have **Master's Degree, around 48%**. Also, **Bachelor's Degree** and **Doctoral Degree** go on decreasing and increasing respectively as we go right. Also, with **people with professional degree and those who attended college/university without Bachelor's Degree** have almost very thin difference, with each being high and low than other interchangably for different groups.

# In[ ]:


# is not displayed well when notebook is published
stacked_bar('Q2', 'Q5', 'Undergraduate major<br>Hover over bars for legends', legend_pos = 'h', extra_suffix = 'Undergraduate majors', showlegend = False)


# In recent years, data science jobs are on raise attracting **students with Computer Science degree.** Also, we can see that number of people with **Engineering (non-computer focused) and Mathematics/Statistic major** increase as we walk right. They are mostly people who were originally in this field when it wasn't booming or there were not much computer based solutions available.

# In[ ]:


stacked_bar('Q2', 'Q6', 'Job Titles', extra_suffix = 'JobTitles')


# Starting with 18-21 group having large number of **Students**, all remaining groups tend to have more number of **Data Scientists, and Consultants** in 60-69 and 70-79. For **Data Scientist**, trends goes up from **14.56% in 22-24 to 23.14% in 30-34, gradually falling in 35-39 (21.24%) to 10.78% in 60-69.** **Research Scientist** that require highly experienced people, start with **2.14% in 22-24, more than doubling to 4.97% in 25-29, growing to 7.82% and 8.64% in 30-34 and 35-39 respectively goes upto 20.41% in 70-79 group. For 80+, 17.64%** have free response which we will explore in last story.  

# In[ ]:


stacked_bar('Q2', 'Q8', 'Years of Experience in Current Role')


# For age group **18-21**, it starts with **highest 55.06% gradually decreasing to 0.81% at 60-69.** However, it is good to see having **15.38% of people in 80+group**, as they must have started/switched recently. It bring to a quote by a noble prize winner, ***The excitement of learning separates youth from old age. As long as you're learning, you're not old***. Otherwise, the trend seems obvious with **5-10 years experience being high in 30-34, followed by 15-20 years in 45-49 and 30+ in 70-79.**

# In[ ]:


multi_stacked_bar('Q2', mcq.columns[45:54], 'Hosted Notebooks')


# **Kaggle Kernels (around 33%)** and **JupyterHub (around 21%)** are popular among every group who use hosted notebook. Mid-range age group people seem to use **Azure Notebook (around 7% in 40-59 and 5.13% overall)** than people at both tails. **Google Colab** seems popular to left, but not increasing as we walk right **(23% to 12%).** Also, **Google Cloud Datalab** being most popular in **70-79 and 80+ group (15%).**

# In[ ]:


stacked_bar('Q2', 'Q17', 'Programming Languages', extra_suffix = 'Languages')


# **Python** being most popular has not changed a lot. As **18-21** group has majority of Computer Science Students, we can a large contribution of CS popular programming languages like **C/C++ and JAVA**, but it decreases towards right. Also, **R/RStudio** has a growing trends towards right. **SQL** is like common database language everyone knows. **Visual Basic** is programming languages used, a lot before for Windows based applications.

# In[ ]:


stacked_bar('Q2', 'Q18', 'Recommended Language for Young Aspirants', legend_pos = 'h', extra_suffix = 'Languages')


# **Python (around 69%)** is highly recommended language to young data science aspirants by any group, followed by **R and SQL (around 15% and 4% respectively). R** is more recommended towards towards right with **starting with 8% in 18-21 group, 13.15% in 30-34 group, 17.14% in 45-49 group, all the way to 27.27% in group 70-79. ** 

# In[ ]:


stacked_bar('Q2', 'Q20', 'Machine Language Framework', extra_suffix = 'ML Frmaeworks')


# **Sci-kit Learn (around 40%)** is most popular ML framework among all age groups. **Tensorflow** , a low level neural networks library and **Keras**,  which is, high level in nature, neural networks library, is widely used by **18-21 group, 22.3% and 16.98% repectively.** With **mid-range group equally supporting** both these libraries, groups **55-59 and 60-69 prefer Keras more than Tensorflow.**  Also, **Caret**, which is a machine learning package for **R is quite popular within groups (mid range- ) that have good number of R preferred survey takers than groups towards left .**

# In[ ]:


stacked_bar('Q2', 'Q22', 'Data Visualization Library')


# **Matplotlib (around 50%)**, a low level data visualization library is highly used by most of the groups as it is first library for every Python user making it most popular in **18-21 group**. **ggplot2 (around 28%)**, data visualization package for R follow same trend as that of **R in programming languages** is most popular in **50-59 group**. Other popular libraries are **Seaborn, Plotly, D3 and Shiny** that are all high level data visualization libraries. 

# In[ ]:


stacked_bar('Q2', 'Q32', 'Types of Data', legend_pos = 'h')


# Every group largely works on **Numerical data (around 25%). 18-21 group** works more **on Image data (20.07%)** than any other group. It may contributed by academic projects. Also, **Time Series data and Text data** gets popular after **22-24 to 55-59 (12.78% to 17.15%)**. 

# In[ ]:


box_dist('Q2', mcq.columns[277:283], 'Proportion of time devoted to various Data Science Tasks')


# With all groups spending almost equal time with **Gathering data (15%), Cleaning data (20%), Visualizing data (10%), Model building/selection (20%), Putting model into production (5%) and finding insights and communicating with stakeholders (10%)**, *values in () taken from overall median.*

# In[ ]:


box_dist('Q2', mcq.columns[284:290], 'Proportion of training')


# Most of survey takers are **self taught and aided their learning with Online Courses**. The **mid age group** seems to learn quite at **work place**, while **groups near to left** have learned more through University than any other group as Universities have recently started including ML as a part of their courseware. For **Kaggle Competition, 18-21 group seems to learn higher (q3 = 15) compared to other groups (q3 = 10).**

# In[ ]:


stacked_bar('Q2', 'Q48', 'Do you consider ML models as black box?', legend_pos = 'h')


# With most of groups having **similar proportion of opinion, proportion of those who are confident to explain outputs of most ML model improves as we walk right.** Also, for **group 70 -79,**  proportion of survey takers considering **ML models as black boxes (orange and green) is quite low (15.15%).**

# In[ ]:


multi_stacked_bar('Q2', mcq.columns[343:349], 'Difficulty in identifying if ML model is fair/unbiased', legend_pos = 'h')


# Most of survey takers find it **difficult in identifying and selecting appropriate evaluation metrics and collecting enough data points that may be unfairly targets**. As we walk right, survey takers **don't find any difficulty** in identifying if model is fair/unbiased.

# In[ ]:


multi_stacked_bar('Q2', mcq.columns[356:371], 'Methods for explaining ML models output', extra_suffix = 'methods')


# Groups towards left use **visual methods such as plotting predicted v/s actual results, printing decision tree, plotting decision boundaries, etc.** more than those towards right using **advanced  technique such as Sensitivity Analysis/Perturbation Importance, examine model coefficient, Dimensionality reduction techniques, etc.** to explain ML models output.

# In[ ]:


multi_stacked_bar('Q2', mcq.columns[386:394], 'Barriers Preventing from Share Coding', legend_pos = 'h')


# Groups towards left find sharing their work more than groups towards right, as **they think it would require more technical knowledge** or **are afraid of their work being used by others without givving them much credit.** As we walk right, it is less likely for survey takers, to find difficulties because of any of above reason. Instead, it is more likely that their work is confidential and bound to their organization that normally won't allow to open source it.

# **Conclusion :** Most of experienced survey takers come from Mathematical/Statistical background allowing them to explore deep and explain tasks involved in designing machine learning algorithms. They tend to use scripting language R, which is specifically designed for data analysis, along with packages and tools that come along. With the boom of computer solutions, most of young survey takers happen to have Computer Science background. It can be more clear by how Python being more popular among them, as Python, a general purpose language is widely taught through academics in computer science. Obviously, experience (which can be mostly derived by age) plays an important role for being a more sophisticated (technically) machine learning engineer.

# <a id='chap3'></a>
# # Chapter 3 - Exploring free responses

# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;In this chapter, we will explore free responses using WordClouds. These responses show a **wide diversity of options** given by survey takers.

# In[ ]:


from wordcloud import WordCloud

import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')

import string
from nltk.corpus import stopwords

def normalize_text(text):
    
    # lowercase it
    text = text.lower()
    # remove punctuation
    text = ''.join([t if t not in string.punctuation else ' ' for t in text])
    # remove stopwords
    text = [t for t in text.split() if t not in stopwords.words('english')]
    # return text
    return ' '.join(text)

def make_text(col):
    text = ''
    for i in col:
        if pd.notnull(i):
            text += ' ' + normalize_text(i)
    return text

def generate_wordcloud(col):
    wordcloud = WordCloud(background_color = 'black', height = 1500, width = 2350, random_state = 21)
    wordcloud.generate(make_text(col))
    plt.figure(figsize=(15, 7))
    plt.axis('off')
    plt.imshow(wordcloud)

freeres = pd.read_csv('../input/freeFormResponses.csv')
freeres.drop(index = 0, inplace = True)


# **#1 - Gender**

# In[ ]:


generate_wordcloud(freeres['Q1_OTHER_TEXT'])


# Valid free form responses are **transgender, male and female.**

# **#2 - JobTitles**

# In[ ]:


generate_wordcloud(freeres['Q6_OTHER_TEXT'])


# Common Job Title include **Professor, Engineer, Analyst, Manager, Instructor, Machine Learning Engineer, etc.**

# **#3 - Industry** 

# In[ ]:


generate_wordcloud(freeres['Q7_OTHER_TEXT'])


# Common response for Industry include **Consultancy, Telecommunication, Analytics, Management, Agriculture, Healthcare, Automotive, etc.**

# **#4 - Important Part of Your Role**

# In[ ]:


generate_wordcloud(freeres['Q11_OTHER_TEXT'])


# Free response answers for important part of role included keywords **Student, Data, Machine Learning, Model, Research, Data Science, Development, Analysis, etc.** 

# **#5 - Primary Tool to Analyse Data**

# In[ ]:


generate_wordcloud(freeres['Q12_OTHER_TEXT'])


# Free responses for primary tools include **Python, Matlab, Pandas, Excel, Tensorflow, Jupyter Notebook, etc.**

# **#6 - IDEs**

# In[ ]:


generate_wordcloud(freeres['Q13_OTHER_TEXT'])


# Most of free response for IDEs include **Eclipse, Emac, Netbeans, XCode, NetBeans, Anaconda, Android Studio, Octave, SAS, Nano, Watson Studio, etc.**

# **#7 - Hosted Notebooks**

# In[ ]:


generate_wordcloud(freeres['Q14_OTHER_TEXT'])


# Common free response for Host Notebooks include **Jupyter Notebook, RStudio, Databricks, Github, Sagemaker, Zeppelin, Github, etc..**

# **#8 - Cloud Computing Service**

# In[ ]:


generate_wordcloud(freeres['Q15_OTHER_TEXT'])


# Free response for Cloud Computing Service include **OpenStack, Digital Ocean, Oracle, Tencent Cloud, Cloudera, Baidu, OneDrive, etc.**

# **#9 - Programming Languages**

# In[ ]:


generate_wordcloud(freeres['Q16_OTHER_TEXT'])


# Free responses for Programming Languages Used include **Swift, Perl, Fortran, Rust, Haskell, Kotlin, Clojure, etc.**

# **#10 - Often Used Programming Language**

# In[ ]:


generate_wordcloud(freeres['Q17_OTHER_TEXT'])


# Most often used lanuages free response include **Swift, Clojure, Spss, Erlang, Kotlin, Perl, Delphi, etc.**

# **#11 - Recommended Programming Language**

# In[ ]:


generate_wordcloud(freeres['Q18_OTHER_TEXT'])


# Free response of first language recommendatin for young aspirants include **Python, Julia, Octave, etc.**

# **#12 - Machine Learning Framework**

# In[ ]:


generate_wordcloud(freeres['Q19_OTHER_TEXT'])


# Most common free response for Machine Learning Framework include **Theano, Chainer, Weka, Darknet, NLTK, etc.**

# **#13 - Most Used Machine Learning Framework**

# In[ ]:


generate_wordcloud(freeres['Q20_OTHER_TEXT'])


# Free response for most preferred ML framework include **Theano, Weka, Chainer, Caret, NLTK, etc.**

# **#14 - Data Visualization Library**

# In[ ]:


generate_wordcloud(freeres['Q21_OTHER_TEXT'])


# Most common free responses for Data Visualization include **Tableau, Matlab, GNUPlot, TensorBoard, Dash, etc.**

# **#15 - Most Used Data Visualization Library**

# In[ ]:


generate_wordcloud(freeres['Q22_OTHER_TEXT'])


# Common free responses for specific most used Data Visualization include **Tableau, Matlab, Powerbi, GNUPlot, Tensorboard, etc.**

# **#16 - Cloud Computing Products**

# In[ ]:


generate_wordcloud(freeres['Q27_OTHER_TEXT'])


# Keywords in free responses for Cloud Computing Products include **AWS, Azure, Alibaba, Digital Ocean, Sagemaker, Cloud, Google, Oracle, etc.**

# **#17 - Machine Learning Products**

# In[ ]:


generate_wordcloud(freeres['Q28_OTHER_TEXT'])


# Common products in free response for Machine Learning Products include **Knime, Weka, Databricks, IBM Watson, SAP, etc.**

# **#18 - Relational Databases**

# In[ ]:


generate_wordcloud(freeres['Q29_OTHER_TEXT'])


# Free response for Database include **MongoDB, SAP Hana, RedShift, Teradata, Snowflake, etc.**

# **#19 - Big Data and Analytics Products**

# In[ ]:


generate_wordcloud(freeres['Q30_OTHER_TEXT'])


# Most common free response for Big Data and Analytics product include **Spark, Hadoop, Cloudera, Splunk, Hive, Apache, Kafka, etc.**

# **#20 - Types of Data**

# In[ ]:


generate_wordcloud(freeres['Q31_OTHER_TEXT'])


# Keywords in free response for Types of Data survey takers interact with include **Network, Log, Survey, Signals, Medical, ClickStream, Financial,etc.**

# **#21 - Most Often Interacted Type of Data**

# In[ ]:


generate_wordcloud(freeres['Q32_OTHER'])


# Free response for most interacted type of include **Logs, Network, Web, Malware, Medical, Security, etc.**

# **#22 - Places to Find Datasets**

# In[ ]:


generate_wordcloud(freeres['Q33_OTHER_TEXT'])


# Keywords in free response for Places where DataSets are found include **Kaggle, Company, UCI, Private, Repository, Bloomberg, etc.**

# **#23 - Machine Learning Training Sources**

# In[ ]:


generate_wordcloud(freeres['Q35_OTHER_TEXT'])


# Free response for learning ML include **Youtube, Bootcamp, Book, Blog, Competition, Research, Course, etc.** 

# **#24 - Online Learning Platform**

# In[ ]:


generate_wordcloud(freeres['Q36_OTHER_TEXT'])


# Most common free response for online learning platform include **Lynda, MLCourse, Youtube, Codecademy, etc.**

# **#25 - Highly Used Online Learning Platform**

# In[ ]:


generate_wordcloud(freeres['Q37_OTHER_TEXT'])


# For highly used online leaning platform, free response include **Youtube, Lynda, NPTEL, PluralSight, LinkedIn, Codecademy, etc.**

# **#26 - Favourite Media Sources**

# In[ ]:


generate_wordcloud(freeres['Q38_OTHER_TEXT'])


# Keywords in free response for favourite media source include **AnalyticsVidhya, Podcast, Blog, Newsletter, etc.**

# **#27 - Metrics used to Determine Models Sucess**

# In[ ]:


generate_wordcloud(freeres['Q42_OTHER_TEXT'])


# Metrics related keywords in free response include **precision, recall, accuracy, evaluation, goal, confusion, etc.**

# **#28 - Methods Used to Reproduce Work**

# In[ ]:


generate_wordcloud(freeres['Q49_OTHER_TEXT'])


# Keywords in free response for methods used to reproduce work include **publish, share, work, reproducible, etc.**

# **#29 - Barrier preventing to reproduce/reuse your work**

# In[ ]:


generate_wordcloud(freeres['Q50_OTHER_TEXT'])


# Keywords in free response for barrier preventing to share work include **Proprietary, Privacy, Company, Confidentiality, Sensitivity, etc.**

# # <center></ Thank You></center>

# In[ ]:




