#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.plotly as py
from scipy import stats


survey = pd.read_csv('../input/SurveySchema.csv', low_memory=False)
free = pd.read_csv('../input/freeFormResponses.csv', low_memory=False)
multi = pd.read_csv('../input/multipleChoiceResponses.csv', low_memory=False)


# # Rising need of data scientists in public sector
# 
# Data scientist in the public sector. Public sector includes following three communities ('Government/Public Service', 'Non-profit/Service','Military/Security/Defense'). Respondents of the Kaggle ML survey are 23,859, and 4.2% out of them are from and working in the public sector. 
# As a person who worked in both side of the sector, it is highly intrigued topic that I've been wondered for a long time. Public sector required a different kind of tasks and consisted of people from various backgrounds while industry based data-scientists are trained/raised in Brick&Mortar learning environment. This study will show the rising demand of data scientists in the public sector and the future of public data-loving data scientists. 
# 

# In[ ]:


import plotly.plotly as py
import plotly.tools as tls
from plotly.graph_objs import *
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import cufflinks as cf
cf.set_config_file(offline=True, world_readable=True)

from scipy import stats


# In[ ]:


sns.set(style="darkgrid")


# In[ ]:


plt.figure(figsize=(10,10))
ax=sns.countplot(y="Q7", data=multi[2:], color="c",order = multi['Q7'][2:].value_counts().index)
ax.set_ylabel('')
ax.set_title('# of respondents in their employed industry field',fontsize=20)


# In[ ]:


gvds=multi.loc[multi['Q7'].isin(['Government/Public Service', 'Non-profit/Service','Military/Security/Defense'])]
nongvds=multi.loc[~multi['Q7'].isin(['Government/Public Service', 'Non-profit/Service','Military/Security/Defense'])]
print("Total " , str(round((gvds.shape[0]/(len(multi)-1))*100,2)) , "% of respondents are public sector data scientists.")


# In[ ]:


multi['publicSector_da'] = np.where(multi['Q7'].isin(['Government/Public Service', 'Non-profit/Service','Military/Security/Defense']), "Public sector", "Non-Public sector")

def mk_piechart(df,colnam, title):
    data=pd.DataFrame(df[colnam].value_counts())
    data=data.reset_index()
    data=data.sort_values(by='index', ascending = True)
    data.iplot(kind='pie',labels='index',values=colnam,pull=.2,hole=.2,colorscale='blues',textposition='auto',textinfo='value+percent',title =title)

mk_piechart(multi,'publicSector_da','Porportion of public sector data scientists')


# # Who are they
# 1.  Background (Q4) and level of experiences (Q25) in data scientist using public data 
# 1. Which country they are working (Q3)?
# 1. Which tool they use (Q17)? 
# 

# In[ ]:


def mk_sidebysidebar(df1,df2, colnam,title):
    plt.subplot(121)
    ax=sns.countplot(y=colnam, data=df1[2:], color="c",order = df1[colnam][2:].value_counts().index) #
    ax.set_title('Public sector')
    ax.set_ylabel('')


    plt.subplot(122)
    ax=sns.countplot(y=colnam, data=df2[2:], color="c")
    ax.set_title('Non public sector')
    ax.set(yticks=[])
    ax.set_ylabel('')

    plt.suptitle(title, fontsize=16)
    plt.subplots_adjust(top=0.8)
    plt.show()

mk_sidebysidebar(gvds,nongvds, "Q4" ,'# respondents per education level')


# Majority of public sector data scientists holding Masters' degree unlike the non-public sector data scientists are mostly from the groups holding Bachelor's degree. Doctoral degree holders are the third majority of the data scientist demographic across both sectors. 

# In[ ]:


mk_sidebysidebar(gvds,nongvds, "Q25" ,'# of respondnets per years using ML methods ')


# Most of data scientists in the public sector are newly entered in the ML community than non-public sector data scientists and missing human resources of highly skilled/experienced data scientist. This may interpretable as rising need of ML approaches in the public sector. 

# In[ ]:


ax=sns.countplot(y="Q3", data=gvds[2:], color="c",order = gvds['Q3'][2:].value_counts().iloc[:10].index)
ax.set_title('Top 10 countries with # of respondents in Public sector',fontsize=20)
plt.subplots_adjust(top=0.7)
ax.set_ylabel('')

plt.show()


# Top 10 countries where they have most of the respondents for public sector employees are the U.S., India, Other, Brazil, U.K.&N.I., Canada, Australia, Spain, France, and Russia, while U.S. based public sector respondents are dramatically bigger (6X) portion than all other countries in the above countries. 

# In[ ]:


import warnings 
warnings.filterwarnings("ignore", category=DeprecationWarning)


# In[ ]:


lists = ['Q16_Part_1','Q16_Part_2','Q16_Part_3','Q16_Part_4','Q16_Part_5','Q16_Part_6','Q16_Part_7',
        'Q16_Part_8','Q16_Part_9','Q16_Part_10','Q16_Part_11','Q16_Part_12','Q16_Part_13','Q16_Part_14',
        'Q16_Part_15','Q16_Part_16','Q16_Part_17','Q16_Part_18']
tempdf=gvds[lists]
x = tempdf.stack()
tempdfout=pd.DataFrame(pd.Series(pd.Categorical(x[x!=0].index.get_level_values(1))))
tempdfout.columns=['Value']
tempdfout['Value'].value_counts()

tempdfout['output']=''    
for outputlabel in lists:
    tempdfout.ix[tempdfout.Value == outputlabel, 'output'] = multi[outputlabel].value_counts().index.tolist()[0:1]
tempdfout=pd.DataFrame(tempdfout) 
gvdsQ16=tempdfout

lists = ['Q16_Part_1','Q16_Part_2','Q16_Part_3','Q16_Part_4','Q16_Part_5','Q16_Part_6','Q16_Part_7',
        'Q16_Part_8','Q16_Part_9','Q16_Part_10','Q16_Part_11','Q16_Part_12','Q16_Part_13','Q16_Part_14',
        'Q16_Part_15','Q16_Part_16','Q16_Part_17','Q16_Part_18']
tempdf=nongvds[lists]
x = tempdf.stack()
tempdfout=pd.DataFrame(pd.Series(pd.Categorical(x[x!=0].index.get_level_values(1))))
tempdfout.columns=['Value']
tempdfout['Value'].value_counts()

tempdfout['output']=''    
for outputlabel in lists:
    tempdfout.ix[tempdfout.Value == outputlabel, 'output'] = multi[outputlabel].value_counts().index.tolist()[0:1]
tempdfout=pd.DataFrame(tempdfout) 

nongvdsQ16=tempdfout


mk_sidebysidebar(gvdsQ16,nongvdsQ16, "output" ,'# of respondents per programming languages use on a regular basis')


# In[ ]:


mk_sidebysidebar(gvds,nongvds, "Q17",'# of respondents per most commonly used coding language' ) # this is suprising 


# The interesting difference between the two sectors' data scientists comes from the type of coding language in regular base vs. most commonly used cases. Public sector data scientists use python as a significant tool for coding, while non-public sector data scientists interchangeably use R and Python.

# # What do they do
# 1. Main types of task (Q11) 
# 2. exact job title (Q6) 
# 3. KPIs (Q42)
# 
# 

# In[ ]:


lists = ['Q11_Part_1','Q11_Part_2','Q11_Part_3','Q11_Part_4','Q11_Part_5','Q11_Part_6','Q11_Part_7']
tempdf=gvds[lists]
x = tempdf.stack()
tempdfout=pd.DataFrame(pd.Series(pd.Categorical(x[x!=0].index.get_level_values(1))))
tempdfout.columns=['Value']
tempdfout['Value'].value_counts()

tempdfout['output']=''    
for outputlabel in lists:
    tempdfout.ix[tempdfout.Value == outputlabel, 'output'] = multi[outputlabel].value_counts().index.tolist()[0:1]
tempdfout=pd.DataFrame(tempdfout) 
gvdsQ11=tempdfout

lists = ['Q11_Part_1','Q11_Part_2','Q11_Part_3','Q11_Part_4','Q11_Part_5','Q11_Part_6','Q11_Part_7']
tempdf=nongvds[lists]
x = tempdf.stack()
tempdfout=pd.DataFrame(pd.Series(pd.Categorical(x[x!=0].index.get_level_values(1))))
tempdfout.columns=['Value']
tempdfout['Value'].value_counts()

tempdfout['output']=''    
for outputlabel in lists:
    tempdfout.ix[tempdfout.Value == outputlabel, 'output'] = multi[outputlabel].value_counts().index.tolist()[0:1]
tempdfout=pd.DataFrame(tempdfout) 

nongvdsQ11=tempdfout


mk_sidebysidebar(gvdsQ11,nongvdsQ11, "output" ,'Type of main tasks')


# The primary type of tasks differs between the two sectors where the public sector data scientists mainly focus on analyzing and understand data to influence on decision-making processes and about half of the respondents answered that their primary task as prototype building or data infrastructure development. 

# In[ ]:


mk_sidebysidebar(gvds,nongvds, "Q6" ,'Job title') 


# Most of the respondents employed in public sector answered their job title as a data scientist. While the absolute number of data scientist respondents in the non-public sector is the much bigger size of a population, however, the majority of the respondents in non-public sector employees are either consultants or data analysts. 

# In[ ]:


lists = ['Q42_Part_1','Q42_Part_2','Q42_Part_3','Q42_Part_4','Q42_Part_5']
tempdf=gvds[lists]
x = tempdf.stack()
tempdfout=pd.DataFrame(pd.Series(pd.Categorical(x[x!=0].index.get_level_values(1))))
tempdfout.columns=['Value']
tempdfout['Value'].value_counts()

tempdfout['output']=''    
for outputlabel in lists:
    tempdfout.ix[tempdfout.Value == outputlabel, 'output'] = multi[outputlabel].value_counts().index.tolist()[0:1]
tempdfout=pd.DataFrame(tempdfout) 
gvdsQ42=tempdfout

lists = ['Q42_Part_1','Q42_Part_2','Q42_Part_3','Q42_Part_4','Q42_Part_5']
tempdf=nongvds[lists]
x = tempdf.stack()
tempdfout=pd.DataFrame(pd.Series(pd.Categorical(x[x!=0].index.get_level_values(1))))
tempdfout.columns=['Value']
tempdfout['Value'].value_counts()

tempdfout['output']=''    
for outputlabel in lists:
    tempdfout.ix[tempdfout.Value == outputlabel, 'output'] = multi[outputlabel].value_counts().index.tolist()[0:1]
tempdfout=pd.DataFrame(tempdfout) 

nongvdsQ42=tempdfout


mk_sidebysidebar(gvdsQ42,nongvdsQ42, "output" ,'KPIs')


# This shows another nature of public sector data scientists. Working as a supporter to help evidence-based decision making as a public sector data scientists, most important metrics to measure their performance is the accuracy while the non-public sector data scientists are focus more on building unbiased outcome.

# # What do they use 
# 1. Main source of public data 
# 2. Main source of public data per where data scientist are reside 

# In[ ]:



lists = ['Q33_Part_1','Q33_Part_2','Q33_Part_3','Q33_Part_4','Q33_Part_5','Q33_Part_6','Q33_Part_7','Q33_Part_8','Q33_Part_9','Q33_Part_10','Q33_Part_11']
tempdf=gvds[lists]
x = tempdf.stack()
tempdfout=pd.DataFrame(pd.Series(pd.Categorical(x[x!=0].index.get_level_values(1))))
tempdfout.columns=['Value']
tempdfout['Value'].value_counts()

tempdfout['output']=''    
for outputlabel in lists:
    tempdfout.ix[tempdfout.Value == outputlabel, 'output'] = multi[outputlabel].value_counts().index.tolist()[0:1]
tempdfout=pd.DataFrame(tempdfout) 
 


tempdfout.head()

def mk_stack_2(df):
    tempdf=df[lists]
    x = tempdf.stack()
    tempdfout=pd.DataFrame(pd.Series(pd.Categorical(x[x!=0].index.get_level_values(1))))
    tempdfout.columns=['Value']
    tempdfout['Value'].value_counts()

    tempdfout['output']=''    
    for outputlabel in lists:
        tempdfout.ix[tempdfout.Value == outputlabel, 'output'] = multi[outputlabel].value_counts().index.tolist()[0:1]
    tempdfout=pd.DataFrame(tempdfout) 


mk_stack_2(gvds)
gvdsQ33=tempdfout

mk_stack_2(nongvds)
nongvdsQ33=tempdfout

mk_sidebysidebar(gvdsQ33,nongvdsQ33, "output",'Source of public data' )


# The primary source of public data amongst the public sector employed respondents is from the government website. While non-public sector respondents utilize google data search as the primary source, respondents in public sector yet use references in the Google data search, which shows blue ocean to the public sector data scientists where they can get the further source of information added to the government sourced data. 

# In[ ]:


gvds['datasource_gv'] = np.where(gvds['Q33_Part_1'].isin(['Government websites']), "Government websites", "Else")
test=pd.crosstab(gvds['Q3'], gvds['datasource_gv']).apply(lambda r: r/r.sum() * 100, axis=1)
test=test.sort_values(by='Government websites', ascending = False)
test=test.reset_index()


# In[ ]:


f, ax = plt.subplots(1, figsize=(20,5))

bar_width = 1
bar_l = [i for i in range(len(test['Government websites']))] 
tick_pos = [i+(bar_width/2) for i in bar_l] 

totals = [i+j for i,j in zip(test['Government websites'], test['Else'])]
pre_rel = [i / j * 100 for  i,j in zip(test['Government websites'], totals)]
mid_rel = [i / j * 100 for  i,j in zip(test['Else'], totals)]

ax.bar(bar_l, 
       pre_rel, 
       label='Government websites', 
       alpha=0.9, 
       color='mediumslateblue',
       width=bar_width,
       edgecolor='white'
       )

ax.bar(bar_l, 
       mid_rel, 
       bottom=pre_rel, 
       label='Else', 
       alpha=0.9, 
       color='white', 
       width=bar_width,
       edgecolor='white'
       )


plt.xticks(tick_pos, test['Q3'])

ax.set_ylabel("Percentage")
ax.set_xlabel("")

plt.xlim([min(tick_pos)-bar_width, max(tick_pos)+bar_width])
plt.ylim(0, 100)
plt.yticks(np.arange(0, 110, step=10))

plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')
plt.title('% of using government data by country as source of public data',fontsize=20)
plt.subplots_adjust(top=0.7)

plt.show()


# Respondents from Austria and Ireland uses government data as a complete source of their public data. Following countries are New Zealand, Kenya, Philippines, Greece, South Korea, Mexico, which uses government data as more than 70% of the source of their public data pulling pool. This may means the abundancy public data provided by the government. 
# 
# 
# 

# 
