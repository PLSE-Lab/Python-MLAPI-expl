#!/usr/bin/env python
# coding: utf-8

# In[2]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# ## Mission of the competition
# DonorsChoose belives that if first time donors gave yet another donation, most of the project requests will be satisfied. The goal of this competition is to figure out the most effective email campaign which would elicit first time donors to donate again.
# 

# In[10]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plot

donationsfile = '../input/Donations.csv'
projectsfile = '../input/Projects.csv'
donorsfile = '../input/Donors.csv'
resourcesfile = '../input/Resources.csv'

donations = pd.read_csv(donationsfile)
projects = pd.read_csv(projectsfile)
donors = pd.read_csv( donorsfile )
resources = pd.read_csv( resourcesfile )


# ## Step 1:
# We will find out those unique donors who made only a single donation and understand their contribution.

# In[4]:


#Find unique single donation givers

#donors who have donated
t = donations['Donor ID'].value_counts() 
singledonors = pd.DataFrame(columns=['Donor ID'])
singledonors['Donor ID'] = t[t==1].index 

donorprojects = donations[['Donor ID', 'Project ID', 'Donation Amount']]
singledonations = pd.merge(singledonors, donorprojects, on='Donor ID')

#project cost, status and amount 
p = projects[['Project ID', 'Project Cost', 'Project Current Status']] 

# donors who donated to a single project, amount they donated, total project cost
singledonorprojects = pd.merge(singledonations, p, on='Project ID')

print(singledonorprojects.describe())


# ## Step 2:  
# First time donors donated in the range \$0.01 to \$31856. Let us bin the donors by donations and see if we can come up with inferences.

# In[3]:


bins = [0, 5, 10, 50, 100, 500, 1000, 5000, 10000, 35000]
singledonorprojects['bins'] = pd.cut(singledonorprojects['Donation Amount'], bins=bins)

bins_count = pd.DataFrame( singledonorprojects['bins'].value_counts().sort_index())
bins_count.columns=['Bin Count']
donations_per_bin = pd.pivot_table(singledonorprojects,index=['bins'],values='Donation Amount',aggfunc=np.sum)

explode = (0,0,0.1,0,0,0,0,0,0)
fig = plot.figure()
fig.set_size_inches(9,9)
ax = plot.subplot(111)
colors = plot.cm.Set1(np.linspace(0, 1, 10))
patches, texts, autotexts = ax.pie(bins_count['Bin Count'], labels=bins_count['Bin Count'], 
                                   autopct='%1.1f%%', explode=explode, labeldistance=1.05,
                                   colors=colors)
for t in texts:
    t.set_size('medium')
for t in autotexts:
    t.set_size('small')
plot.title('First time donors and their contribution bracket', fontsize=20)
ax.legend(bins_count.index, loc = "lower center", ncol=3, fontsize='medium')
plot.show()


# ## Inference 1:
# Majority of first time donors donated in the \$10 - \$50 range

# In[4]:


fig2 = plot.figure()
fig2.set_size_inches(9,9)
ax = plot.subplot(111)
colors = plot.cm.Set2(np.linspace(0, 1, 10))
patches, texts, autotexts = plot.pie(donations_per_bin['Donation Amount'], 
                                     labels = round(donations_per_bin['Donation Amount']),
                                     autopct='%1.1f%%', explode=explode, labeldistance=1.05,
                                     colors=colors)
for t in texts:
    t.set_size('medium')
for t in autotexts:
    t.set_size('small')
plot.title('Total contribution by first time donors in their contribution bracket', fontsize=20)
ax.legend(bins_count.index, loc = "lower center", ncol=3, fontsize='medium')
plot.show()


# ## Inference 2:
# 
# The **single largest donation bracket is \$10 - \$50**. They amount to** USD 32 Million** in donations.
# 
# \$50-\$100 and \$100-\$500 are the next largest contributors amounting to a combined donation amount of **USD 35 Million**.
# 
# ## Step 3:
# Let us study the grades whose projects and project resources that were of interest to the 3 donor brackets of interest.

# In[5]:


#let us work on the major donor bins
a = singledonorprojects.copy()
under50donors = a[(a['Donation Amount']>10.0) & (a['Donation Amount']<=50.0)]
under100donors = a[(a['Donation Amount']>50.0) & (a['Donation Amount']<=100.0)]
under500donors = a[(a['Donation Amount']>100.0) & (a['Donation Amount']<=500.0)]
  
#under $50 donor interests
x = pd.merge(under50donors, projects, on='Project ID') 
under50projectdata = x[['Project ID', 'Project Grade Level Category', 'Project Subject Category Tree',
                        'Project Resource Category']]

temp = under50projectdata.copy()
p = temp.groupby(['Project Grade Level Category', 'Project Resource Category']).size().reset_index()
p.columns = ['Grade Category','Resource Category','Count']
labels = pd.DataFrame(temp['Project Grade Level Category'].value_counts()).reset_index()
labels.columns = ['Grade Category','Count']
graph_data50 = pd.pivot_table(p,index=['Grade Category'],
                            columns=['Resource Category'],
                            values='Count',aggfunc=np.sum).fillna(0)

colors = plot.cm.PiYG(np.linspace(0, 1, 20))
ax = graph_data50.plot(kind='barh', stacked=True, color=colors, 
                  title='Resource Categories for Projects supported by First time Donors (\$10-\$50]',
                  figsize=(17,10))

patches, labels = ax.get_legend_handles_labels()
ax.legend(patches, labels, loc='right')


# **The study of project subject category and project subject subcategory did not give any conclusive information, but project resource category did.**
# 
# ## Inference 3 (\$10 - \$50 bracket) :  
# 
# Most donations were made to **Grades PreK-2**, followed by **Grades 3-5**.  
# For Grades PreK-2, the maximum donation was for **Supplies**, followed by **Technology**.  
# For Grades 3-5, **Supplies and Technology **shared almost equal contributions.

# In[6]:


#under $100 donor interests
x = pd.merge(under100donors, projects, on='Project ID') 
under100projectdata = x[['Project ID', 'Project Grade Level Category', 'Project Subject Category Tree',
                         'Project Resource Category']]

temp = under100projectdata.copy()
p = temp.groupby(['Project Grade Level Category', 'Project Resource Category']).size().reset_index()
p.columns = ['Grade Category','Resource Category','Count']
labels = pd.DataFrame(temp['Project Grade Level Category'].value_counts()).reset_index()
labels.columns = ['Grade Category','Count']
graph_data100 = pd.pivot_table(p,index=['Grade Category'],
                            columns=['Resource Category'],
                            values='Count',aggfunc=np.sum).fillna(0)

colors = plot.cm.BrBG(np.linspace(0, 1, 20))
ax = graph_data100.plot(kind='barh', stacked=True, color=colors, 
                   title='Resource Categories for Projects supported by First time Donors (\$50-\$100]',
                  figsize=(17,10))

patches, labels = ax.get_legend_handles_labels()
ax.legend(patches, labels, loc='right')


# ## Inference 4 (\$50 - \$100 bracket) :  
# 
# Once again, maximum donations were made to **Grades PreK-2**, followed by **Grades 3-5**.  
# For **Grades PreK-2**, the donation was split between **Supplies and Technology**.  
# For **Grades 3-5**, maximum donation was for **Technology** followed by **Supplies**.

# In[7]:


#under $500 donor interests
x = pd.merge(under500donors, projects, on='Project ID') 
under500projectdata = x[['Project ID', 'Project Grade Level Category', 'Project Subject Category Tree',
                         'Project Resource Category']]

temp = under500projectdata.copy()
p = temp.groupby(['Project Grade Level Category', 'Project Resource Category']).size().reset_index()
p.columns = ['Grade Category','Resource Category','Count']
labels = pd.DataFrame(temp['Project Grade Level Category'].value_counts()).reset_index()
labels.columns = ['Grade Category','Count']
graph_data500 = pd.pivot_table(p,index=['Grade Category'],
                            columns=['Resource Category'],
                            values='Count',aggfunc=np.sum).fillna(0)

colors = plot.cm.bwr(np.linspace(0, 1, 20))
ax = graph_data500.plot(kind='barh', stacked=True, color=colors, 
                   title='Resource Categories for Projects supported by First time Donors (\$100-\$500]',
                  figsize=(17,10))

patches, labels = ax.get_legend_handles_labels()
ax.legend(patches, labels, loc='right')


# ## Inference 5 (\$100 - \$500 bracket) :  
# 
# Once again, maximum donations were made to **Grades PreK-2**, followed by **Grades 3-5**.  
# For **Grades PreK-2,** the donation was more for **Supplies** followed by **Technology**.  
# For **Grades 3-5**, maximum donation was for **Technology** followed by **Supplies**.

# > **Recommendation 1:**
# 
# > Surprisingly first time donors donating in the \$10-\$50 bracket  **contributed to a significant portion** of the donations.  
# > An email highlighting how **any  donation amount, however small  makes a significant impact **should be the focus of the campaign to first time donors.  
# > Additionally, it should present **Grades PreK-2 and Grades 3-5 projects **which require **Supplies and Technology**, as these were the areas to which first time donors contributed heavily.  
# >  
# > Such an email campagin can be run towards potential donors regardless of the donation bracket.
# 
# We will continue to study the donation patterns of donors to make further recommendations before the deadline.
# 

# ## Step 4: 
# 
# We would now like to study the contributions of people who have donated just twice. We wish to see if there is a pattern in their donation which will help us predict the kind of projects first time donors may prefer donating to, the second time around.
# 
# We first find donors who have donated twice and then get the data on the projects they donated to and when they made their donation. 

# In[1]:


#let us now work on donors who have donated to more than 1 project

newdonors = pd.DataFrame(donations['Donor ID'].value_counts()).reset_index()
newdonors.columns=['Donor ID', 'Count']
newdonors = newdonors[newdonors['Count'] >= 2 ]

#donors who have donated **only** to 2 projects
twicedonors = newdonors[newdonors['Count'] == 2 ]

#Get the donation information for these donors
tDonors = donations[donations['Donor ID'].isin(twicedonors['Donor ID'])]
temp_list = [1,2] * int(len(tDonors)/2)

tDonors = tDonors.sort_values(by=['Donor ID','Donation Received Date'])
tDonors['temp_label'] = temp_list

t1= pd.pivot_table(tDonors,index=['Donor ID'],
                            columns=['temp_label'],
                            values=['Project ID', 'Donation ID', 'Donation Received Date'],
                            aggfunc=lambda x: ' '.join(x) )
t1 = t1.reset_index()
t2 = pd.pivot_table(tDonors,index=['Donor ID'],
                            columns=['temp_label'],
                            values=['Donation Amount'])
t2 = t2.reset_index() 
df = pd.merge(t1, t2, on='Donor ID')

#Find the pattern in their donations
df['Donation Diff'] = df[('Donation Amount', 1)] - df[('Donation Amount', 2 )]
df['Donation Diff'] = df['Donation Diff'].abs()
print( df['Donation Diff'].describe() )


# 
# 
# 

# ## Step 5:
# 
# There are around 275000 people who donated exactly twice. We calculate the difference in their donations and study they distribution.  
# 
# It shows that upto the 75th percentile, the difference in donation amount by two time donors is in the order of $35. 
# 
# ## Inference 6:
# 
# **This shows that majority of the people who donate the second time around, donate close to their first time donation amount.**

# In[2]:


p = df['Donation Diff'].value_counts().reset_index()
top10count = p[0:10]
top10count.columns = ['Donation Difference', 'Count']
donationdiff = top10count.sort_values(by='Donation Difference')

fig3 = plot.figure()
fig3.set_size_inches(9,9)
ax = plot.subplot(111)
colors = plot.cm.Set2(np.linspace(0, 1, 10))
patches, texts, autotexts = ax.pie(donationdiff['Count'], 
                                   labels=donationdiff['Donation Difference'], 
                                   autopct='%1.1f%%', labeldistance=1.05,
                                   colors=colors)
for t in texts:
    t.set_size('medium')
for t in autotexts:
    t.set_size('small')
plot.title('Distribution of top 10 donation amount differences \n comparing the 2 donations made by second time donors.')


# ## Inference 7:
# 
# We plotted the top 10 donation difference amounts which range from \$0 to \$75. \$0 means the donation amount second time around was exactly as the same as the first donation. We see that **>50 percent donated exactly the same amount** as the first time.

# In[ ]:


df['t1'] = pd.to_datetime(df[('Donation Received Date', 1)])
df['t2'] = pd.to_datetime(df[('Donation Received Date', 2)])

df['Date Diff'] = df['t2'] - df['t1']
print( df[('Date Diff')].describe() )


# ## Inference 8:
# 
# The second time donation can happen any time from the first donation and **can even happen almost a year **after the first donation.
# 
# ## Step 8:
# 
# Now we wish to see if the two time donors donated to the same grade projects as the first time and also if they were influenced by the resources used in the projects.

# In[ ]:


dp1 = pd.DataFrame(columns=['Donor ID', 'Project ID'])
dp1['Project ID'] = df[('Project ID', 1)]
dp1['Donor ID'] = df['Donor ID']
temp = pd.merge(dp1, projects, on='Project ID')
dp1_final = temp[['Donor ID', 'Project ID', 'Project Grade Level Category', 'Project Resource Category']]
dp1_final.columns=['Donor ID', 'Project ID1', 'Grade1', 'Resource1']
dp2 = pd.DataFrame(columns=['Donor ID', 'Project ID'])
dp2['Project ID'] = df[('Project ID', 2)]
dp2['Donor ID'] = df['Donor ID']
temp = pd.merge(dp2, projects, on='Project ID')
dp2_final = temp[['Donor ID', 'Project ID', 'Project Grade Level Category', 'Project Resource Category']]
dp2_final.columns=['Donor ID', 'Project ID2', 'Grade2', 'Resource2']

merged_data = pd.merge(dp1_final, dp2_final, on='Donor ID')
merged_data['Same Grade?'] = (merged_data['Grade1'] == merged_data['Grade2'])
merged_data['Same Resource?'] = (merged_data['Resource1'] == merged_data['Resource2'])


# ## Step 9:
# 
# Let us plot the data which tells us what kind of projects the second time donors donated to when compared to their first time donation.

# In[ ]:


fig4 = plot.figure()
fig4.set_size_inches(9,9)
graph1 = merged_data['Same Grade?'].value_counts().reset_index()
graph1.columns = ['Same Grade?','Count']
plot.title('Did a second time donor contribute to \n a project in the same grade as the first time?')
plot.pie(graph1['Count'], labels=graph1['Same Grade?'], colors=['y','g'], autopct='%1.1f%%')
plot.show()


# ## Inference 9:
# 
# More than **60 percent** of two time donors donated to the **same grade projects as the first time**.

# In[ ]:


fig5 = plot.figure()
fig5.set_size_inches(9,9)
graph2 = merged_data['Same Resource?'].value_counts().reset_index()
graph2.columns = ['Same Resource?','Count']
plot.title('Did a second time donor contribute to \n a project with the same resource category as the first time?')
plot.pie(graph2['Count'], labels=graph2['Same Resource?'], autopct='%1.1f%%')
plot.show()


# ## Inference 10:
# 
# There is **no viable pattern **on the project resources favored by two time donors for their second donation.
# 
# Let us try to find the seasonality pattern for the donations.

# In[8]:


#Took data between 2013-2017
ts = donations.loc[:,['Donation Received Date', 'Donation Amount']]
ts.set_index('Donation Received Date', inplace=True)
ts = ts[(ts.index>='2013-01-01') & (ts.index<'2018-01-01') ]
ts['month'] = pd.to_datetime(ts.index, format='%Y-%m-%d %H:%M:%S.%f').month
ts.groupby('month').sum()['Donation Amount'].plot(kind='bar', color='darkorange', figsize=(10,5))
plot.xlabel('Donation Amount')
plot.show()


# Total donation is the lowest in June as its summer vacation and highest in December and August (when schools are about to open).

# In[1]:


# import cufflinks and offline mode
import cufflinks as cf
cf.go_offline()
#Number of donations per project
custom_bucket = [0, 1, 5, 10, 20, 1000000]
custom_bucket_label = ['Single Donor', '1-5 Donors', '6-10 Donors', '11-20 Donors', 'More than 20 Donors']
num_of_don = donations['Project ID'].value_counts().to_frame(name='Donation Count').reset_index()
num_of_don['Donation Cnt'] = pd.cut(num_of_don['Donation Count'], custom_bucket, labels=custom_bucket_label)
num_of_don = num_of_don['Donation Cnt'].value_counts().sort_index()

num_of_don.iplot(kind='bar', xTitle = 'Number of Donors', yTitle = 'Number of Projects', 
                title = 'Distribution on Number of Donors and Project Count')


# ## Inference 11:
# 
# Most projects have less than 5 Donors.

# In[2]:


import plotly.graph_objs as go
from plotly import tools
from plotly.offline import iplot

donations['Donation Date'] = pd.to_datetime(donations['Donation Received Date'])
donations['Donation Year'] = donations['Donation Date'].dt.year
donations['Donation Month'] = donations['Donation Date'].dt.month

donations_g = donations.groupby(['Donation Month']).agg({'Donation Year' : 'count', 'Donation Amount' : 'mean'}).reset_index().rename(columns={'Donation Year' : 'Total Donations', 'Donation Amount' : 'Average Amount'})
x = donations_g['Donation Month']
y1 = donations_g['Total Donations']
y2 = donations_g['Average Amount']
trace1 = go.Scatter(x=x, y=y1, fill='tozeroy', fillcolor = '#kcc49f', mode= 'none')
trace2 = go.Scatter(x=x, y=y2, fill='tozeroy', fillcolor = "#a993f9", mode= 'none')
fig = tools.make_subplots(rows=1, cols=2, print_grid=False, subplot_titles = ["<b>Donations per Month</b>", "<b>Average Donation Amount per Month</b>"])
fig.append_trace(trace1, 1, 1);
fig.append_trace(trace2, 1, 2);

fig['layout'].update(height=300, showlegend=False, yaxis=dict(
        autorange=True,
        showgrid=False,
        zeroline=False,
        showline=False,
        autotick=True,
        ticks='',
        showticklabels=False
    ), yaxis2=dict(
        autorange=True,
        showgrid=False,
        zeroline=False,
        showline=False,
        autotick=True,
        ticks='',
        showticklabels=False
    ));


iplot(fig);


# ## Inference 12:
# 
# 1. **Highest number of donations are received **in the month of **August** with 577,274 Donations **followed by September** month with 518,908 donations while least number of donations are received in the month of June equal to 191,221 donations.
# 2. The** highest average donation amount** is received in the month of **December** with USD 80 while the least is in the month of July equal to USD 51. 
# 3. On an average the **average donation amount per year is close to USD 59.**

# # Conclusion & Recommendation:
# 
# 1. Donors donate close to school reopening time (August/September) and during December (end of the tax year)
# 2. First time donors donated maximum to Grades PreK-2 projects
# 3. First time donors favored projects that required Supplies and Technology
# 4. Majority first time donors were in the USD 10 - USD 50 bracket
# 4. Two time donors mostly donated same amount as the first time
# 5. Two time donors donated to projects for the same grade as the first time
# 
# **An email campaign in August or December to first time donors emphasizing that every donation counts and presenting projects in the same grade level as their first time donation project will elicit the maximum response as per this study.**
# 
# Thank you for this lovely opportunity to analyze and participate in this competition.
# 
# 
# 

# In[ ]:




