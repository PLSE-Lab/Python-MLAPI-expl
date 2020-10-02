#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
pd.options.mode.chained_assignment = None

from IPython.display import HTML

import plotly.plotly as py
import plotly.graph_objs as go
from plotly import tools
from plotly.graph_objs import *
from plotly.offline import iplot, init_notebook_mode
init_notebook_mode()


# In[8]:


df=pd.read_csv('../input/Reveal_EEO1_for_2016.csv')
print("Lets take a look at what our data  looks like")
df.head()


# In[9]:


df['count'].replace(to_replace='na',value=0,inplace=True)
df['count']=df['count'].astype(int)


# In[10]:


d=df.groupby(['company']).agg({'count': lambda x: sum((x).astype(int))})
plt.figure(figsize=(10,8))
sns.set_style('white')
sns.barplot(x=d.index.get_values(),y=d['count'],palette='viridis')
plt.title('Number of employees by company',size=16)
plt.ylabel('Number of employees',size=14)
plt.xlabel('Company',size=14)
plt.yticks(size=14)
plt.xticks(size=14,rotation=90)
sns.despine()
plt.show()


# In[11]:


labels = df.groupby(['gender']).agg({'count':sum}).index.get_values()
values = df.groupby(['gender']).agg({'count':sum})['count'].values
trace = go.Pie(labels=labels, values=values,textinfo='label+value+percent')
layout=go.Layout(title='Distribution of Female and Male Employee')
data=[trace]


fig = dict(data=data,layout=layout)
iplot(fig, filename='Distribution of Female and Male Employee')


# ### For every women employee there are two men employed by the given companies.
# ### **Let us plot the number of male and female employee by each of the companies to see the distribution of employees**</h3>

# In[12]:


d=df.groupby(['gender','company']).agg({'count':sum}).reset_index()
trace1 = go.Bar(
    x=d[d.gender=='male']['company'],
    y=d[d.gender=='male']['count'],
    name='Males'
)
trace2 = go.Bar(
    x=d[d.gender=='female']['company'],
    y=d[d.gender=='female']['count'],
    name='Females'
)
data = [trace1, trace2]
layout = go.Layout(
    barmode='group',title='Distribution of Male and Female Employees by Company')


fig = dict(data=data, layout=layout)
iplot(fig, filename='Distribution of Male and Female Employees by Company')


# <h3><b> We can see that some of the biggest companies such as Intel, Apple, Cisco, Google have a huge gap between the number of male and female employees. Let us dig a little deeper to see how wide the gap is</b></h3>

# In[13]:


d=d.groupby(['company','gender']).agg({'count':sum})
d=d.unstack()
d=d['count']
d=d.iloc[:,:].apply(lambda x: (x/x.sum())*100,axis=1)
d['Disparity']=d['male']/d['female']
d.sort_values(by='Disparity',inplace=True,ascending=False)
d.columns=['Female %','Male %','Disparity']


# In[14]:


d


# #### Even big companies such as <b>Nvidia, Intel, Cisco, Uber, Google, Apple, Facebook </b> and many more have more than <b>2 male employees for each female employee.</b>
# #### <b>Nvidia</b> seems to have a major disparity in the number of male and female employees, <b>with almost 5 men for each female employee</b>

# In[15]:


trace1 = go.Bar(
    x=d.index.get_values(),
    y=d['Disparity'],text=np.round(d['Disparity'],2),textposition='auto'
)




data = [trace1]
layout = go.Layout(
    barmode='group',title='Disparity in Number of Male and Female Employees')

fig = dict(data=data, layout=layout)
iplot(fig, filename='Disparity in Number of Male and Female Employees')


# ### **Let us take a look at the number of employees by Company, Race and Gender** 
# ### **Use the Company button from the dropdown menu to see the number of male and female employee by race for each of the company**

# In[16]:


y=df.groupby(['company','race','gender']).agg({'count':sum}).reset_index()


# In[17]:


from collections import deque

## We have to build a list of true/false, in order to support a dropdown menu
## The for loop right shifts True by 1 position
visibility=[True]+[False]*((y.company.nunique()+1))
d=[0]*((y.company.nunique()+3))
for i in range(0,len(d)):
    a=deque(visibility)
    a.rotate(i) 
    d[i]=list(a)

a={}
al=[]
data=[]

unique_company=y.company.unique()
for i in range(0,22):
    m=y[y.company==unique_company[i]]
    data.append(Bar(x=m['race'].unique(),y=m[m.gender=='male']['count'].values+m[m.gender=='female']['count'].values))
    max_annotations=[]
    xcord=np.arange(0,m['race'].nunique())
    for j in range(0,len(m['race'].unique())):
        max_annotations.append([dict(x=xcord[j], y=m[m.gender=='male']['count'].values[j]+m[m.gender=='female']['count'].values[j],
                       xref='x', yref='y',text='<b>Males:</b> '+str(m[m.gender=='male']['count'].values[j])
                       +"<br><b>Females:</b> "+str(m[m.gender=='female']['count'].values[j]))]) 
    al.append(dict(label=unique_company[i],method='update',args=[{'visible':d[i]},{'title':"<b>"+unique_company[i]+"</b><br>Total Employees:"+str(sum(m['count'])),
                                                                                   'annotations':max_annotations[0]+max_annotations[1]+max_annotations[2]+max_annotations[3]
                                                                                  +max_annotations[4]+max_annotations[5]+max_annotations[6]}]))
    

    

data=Data(data)
updatemenus=list([
        dict(
            x=0,
            y=1,
            xanchor='left',
            active=0,
            yanchor='top',
            buttons=al,
        )
    ])

layout = Layout(updatemenus=updatemenus,showlegend=False,)

fig = dict(data=data,layout=layout)
iplot(fig,filename='Number of employees by Company, Race and Gender')


# ### **Now let us see the number of male and female employees by Gender and Race**

# In[18]:


d=df.groupby(['gender','race']).agg({'count':sum}).reset_index()


# In[19]:


d=df.groupby(['gender','race']).agg({'count':sum}).reset_index()
trace1 = go.Bar(
    x=d[d.gender=='male']['race'],
    y=d[d.gender=='male']['count'],
    name='Males'
)
trace2 = go.Bar(
    x=d[d.gender=='female']['race'],
    y=d[d.gender=='female']['count'],
    name='Females'
)

xcord=np.arange(0,7)
annotations_1=[]
annotations_2=[]
for i in range(0,7):
    annotations_1.append(dict(x=xcord[i]-0.2,y=d[d.gender=='male']['count'].values[i]+100,text='%d' %d[d.gender=='male']['count'].values[i],
                             font=dict(family='Arial', size=10,
                                  color='rgba(0,0,0,1)'),
                                  showarrow=True,))
    
for i in range(0,7):
    annotations_2.append(dict(x=xcord[i]+0.3,y=d[d.gender=='female']['count'].values[i]+100,text='%d' %d[d.gender=='female']['count'].values[i],
                             font=dict(family='Arial', size=10,
                                  color='rgba(0,0,0,1)'),
                                  showarrow=True,))
    
annotations=annotations_1+annotations_2
data = [trace1, trace2]
layout = go.Layout(
    barmode='group',title='Distribution of Male and Female Employees by Race')
layout['annotations'] = annotations

fig = dict(data=data, layout=layout)
iplot(fig, filename='Distribution of Male and Female Employees by Race')


# ### <b> Let us plot the number of employees by gender and Job Category</b>

# In[20]:


d=df.groupby(['gender','race','job_category']).agg({'count':sum}).reset_index()


# In[21]:


plt.figure(figsize=(15,12))
sns.set_style('white')
sns.barplot(x='job_category',y='count',hue='gender',data=d, palette="muted",ci=None)
plt.title('Number of employee by Job Category and Gender',size=16)
plt.yticks(size=14)
plt.ylabel('Number of Employees',size=14)
plt.xlabel('Job Category',size=14)
plt.xticks(rotation=90,size=14)
plt.show()


# In[22]:


HTML('''<script>
code_show=true; 
function code_toggle() {
 if (code_show){
 $('div.input').hide();
 } else {
 $('div.input').show();
 }
 code_show = !code_show
} 
$( document ).ready(code_toggle);
</script>
To toggle code, click <a href="javascript:code_toggle()">here</a>.''')


# In[ ]:




