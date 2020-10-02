#!/usr/bin/env python
# coding: utf-8

# In[3]:


import seaborn as sns
sns.set(style='whitegrid')
get_ipython().run_line_magic('pylab', 'inline')
import numpy as np
import pandas as pd
import missingno as msn


# In[4]:


df = pd.read_csv("../input/openpowerlifting.csv")


# In[3]:


df.describe(include='all')


# In[4]:


df.info()


# In[5]:


msn.matrix(df, sort='ascending')


# <h1>Missing data</h1>
# <h3>As we can see there is almost no data available for *Squat4Kg*, *Bench4Kg* and *Deadlift4Kg*. Further *Age* is also rahter incomplete. Lets quickly look at the other feature how much data is actually missing.</h3>
# 

# In[6]:


for feature in df:
    missing = df[feature].isnull().sum()
    perc_missing = round(missing/df.shape[0]*100,2)
    print("{} has {} missing entries.".format(feature, missing))
    print("That's {} % missing.".format(perc_missing))
    print('*'*44)


# <h3>Ok so let's start looking at the features. First we are going to take a look at the meetings and the names. How many meeting are combined in this dataset and how many unique individuals are in the data?</h3>

# In[7]:


grouped = df.groupby("Name")

m = 0
f = 0
for i in grouped:
    if i[1]["Sex"].iloc[0] == 'M':
        m += 1
    else:
        f += 1


# In[8]:


print("The data is composed of {} different meetings.".format(len(df["MeetID"].value_counts())))
print("Overall {} individual athletes are in the dataset.".format(len(df["Name"].value_counts())))
print("Of those {} are male, and {} female.".format(m, f))
print("The type of Equipment used:")
print(df["Equipment"].value_counts())


# ## The sport is dominated by man and most of the entires are either *Raw* or *Single-ply*. To give you a quick overview:
# ## 1. Raw
# >###    No equipment is allowed 
# 
# ## 2. Single-ply 
# >###   Supportive clothing. 

# In[9]:


for i in df:
    if df[i].dtype != 'O':
        vmin = df[i].min()
        vmax = df[i].max()
        vmean = df[i].mean()
        vmedian = df[i].median()
        print(i)
        print("min: {}".format(vmin))
        print("max: {}".format(vmax))
        print("mean: {}".format(round(vmean,2)))
        print("median: {}".format(vmedian))
        print('*'*20)


# In[10]:


plt.figure(figsize=(15,30))
plt.subplot(6,2,1)
#plt.hist(df["BodyweightKg"].dropna(), bins=30);
sns.countplot(pd.qcut(df["Age"],20), hue=df["Sex"])

plt.subplot(6,2,2)
#plt.hist(df["BodyweightKg"].dropna(), bins=30);
sns.countplot(pd.qcut(df["BodyweightKg"],20), hue=df["Sex"])

plt.subplot(6,2,3)
#plt.hist(df["BodyweightKg"].dropna(), bins=30);
sns.countplot(pd.qcut(df["Squat4Kg"],20), hue=df["Sex"])

plt.subplot(6,2,4)
#plt.hist(df["BodyweightKg"].dropna(), bins=30);
sns.countplot(pd.qcut(df["BestSquatKg"],20), hue=df["Sex"])

plt.subplot(6,2,5)
#plt.hist(df["BodyweightKg"].dropna(), bins=30);
sns.countplot(pd.qcut(df["Bench4Kg"],20), hue=df["Sex"])

plt.subplot(6,2,6)
#plt.hist(df["BodyweightKg"].dropna(), bins=30);
sns.countplot(pd.qcut(df["BestBenchKg"],20), hue=df["Sex"])

plt.subplot(6,2,7)
#plt.hist(df["BodyweightKg"].dropna(), bins=30);
sns.countplot(pd.qcut(df["Deadlift4Kg"],20), hue=df["Sex"])

plt.subplot(6,2,8)
#plt.hist(df["BodyweightKg"].dropna(), bins=30);
sns.countplot(pd.qcut(df["BestDeadliftKg"],20), hue=df["Sex"])

plt.subplot(6,2,9)
#plt.hist(df["BodyweightKg"].dropna(), bins=30);
sns.countplot(pd.qcut(df["Wilks"],20), hue=df["Sex"])

plt.subplot(6,2,10)
#plt.hist(df["BodyweightKg"].dropna(), bins=30);
sns.countplot(pd.qcut(df["TotalKg"],20), hue=df["Sex"])

plt.subplot(6,2,11)
#plt.hist(df["BodyweightKg"].dropna(), bins=30);
sns.countplot(pd.qcut(df["Wilks"],20), hue=df["Sex"])


# <h3>So what exactly is this equipment?</h3>
# <ul>
#     <li>
#         <h3>Raw: Lifting w/o euqipment. However, lifting belts, sleeves and chalk is allowed, though.</h3>
#     </li><li>
#         <h3>Single-ply: Lifting w/o euqipment. However, lifting belts, sleeves and chalk is allowed, though.</h3>
#     </li>
# </ul>

# In[11]:


plt.figure(figsize=(15,15))
plt.subplot(6,2,1)
#plt.hist(df["BodyweightKg"].dropna(), bins=30);
plt.scatter(df["Age"][df["Sex"] == 'M'], df["BodyweightKg"][df["Sex"] == 'M'], alpha=.1)
plt.scatter(df["Age"][df["Sex"] == 'F'], df["BodyweightKg"][df["Sex"] == 'F'], alpha=.1)

plt.figure(figsize=(15,30))
plt.subplot(6,2,1)
#plt.hist(df["BodyweightKg"].dropna(), bins=30);
plt.scatter(df["Age"][df["Sex"] == 'M'], df["BestSquatKg"][df["Sex"] == 'M'], alpha=.1)
plt.scatter(df["Age"][df["Sex"] == 'F'], df["BestSquatKg"][df["Sex"] == 'F'], alpha=.1)

plt.subplot(6,2,2)
#plt.hist(df["BodyweightKg"].dropna(), bins=30);
plt.scatter(df["BodyweightKg"][df["Sex"] == 'M'], df["BestSquatKg"][df["Sex"] == 'M'], alpha=.1)
plt.scatter(df["BodyweightKg"][df["Sex"] == 'F'], df["BestSquatKg"][df["Sex"] == 'F'], alpha=.1)

plt.subplot(6,2,3)
plt.scatter(df["Age"][df["Sex"] == 'M'], df["BestBenchKg"][df["Sex"] == 'M'], alpha=.1)
plt.scatter(df["Age"][df["Sex"] == 'F'], df["BestBenchKg"][df["Sex"] == 'F'], alpha=.1)

plt.subplot(6,2,4)
plt.scatter(df["BodyweightKg"][df["Sex"] == 'M'], df["BestBenchKg"][df["Sex"] == 'M'], alpha=.1)
plt.scatter(df["BodyweightKg"][df["Sex"] == 'F'], df["BestBenchKg"][df["Sex"] == 'F'], alpha=.1)

plt.subplot(6,2,5)
plt.scatter(df["Age"][df["Sex"] == 'M'], df["BestDeadliftKg"][df["Sex"] == 'M'], alpha=.1)
plt.scatter(df["Age"][df["Sex"] == 'F'], df["BestDeadliftKg"][df["Sex"] == 'F'], alpha=.1)

plt.subplot(6,2,6)
plt.scatter(df["BodyweightKg"][df["Sex"] == 'M'], df["BestDeadliftKg"][df["Sex"] == 'M'], alpha=.1)
plt.scatter(df["BodyweightKg"][df["Sex"] == 'F'], df["BestDeadliftKg"][df["Sex"] == 'F'], alpha=.1)

plt.subplot(6,2,7)
plt.scatter(df["Age"][df["Sex"] == 'M'], df["TotalKg"][df["Sex"] == 'M'], alpha=.1)
plt.scatter(df["Age"][df["Sex"] == 'F'], df["TotalKg"][df["Sex"] == 'F'], alpha=.1)

plt.subplot(6,2,8)
plt.scatter(df["BodyweightKg"][df["Sex"] == 'M'], df["TotalKg"][df["Sex"] == 'M'], alpha=.1)
plt.scatter(df["BodyweightKg"][df["Sex"] == 'F'], df["TotalKg"][df["Sex"] == 'F'], alpha=.1)

plt.subplot(6,2,9)
plt.scatter(df["Age"][df["Sex"] == 'M'], df["Wilks"][df["Sex"] == 'M'], alpha=.1)
plt.scatter(df["Age"][df["Sex"] == 'F'], df["Wilks"][df["Sex"] == 'F'], alpha=.1)

plt.subplot(6,2,10)
plt.scatter(df["BodyweightKg"][df["Sex"] == 'M'], df["Wilks"][df["Sex"] == 'M'], alpha=.1)
plt.scatter(df["BodyweightKg"][df["Sex"] == 'F'], df["Wilks"][df["Sex"] == 'F'], alpha=.1)


# In[12]:


plt.figure(figsize=(15,15))
plt.subplot(6,2,1)
#plt.hist(df["BodyweightKg"].dropna(), bins=30);
plt.scatter(df["Age"][df["Place"] != 'DQ'], df["BodyweightKg"][df["Place"] != 'DQ'], alpha=.1)
# plt.scatter(df["Age"][df["Sex"] == 'F'], df["BodyweightKg"][df["Sex"] == 'F'], alpha=.1)

plt.figure(figsize=(15,30))
plt.subplot(6,2,1)
#plt.hist(df["BodyweightKg"].dropna(), bins=30);
plt.scatter(df["Age"][df["Place"] != 'DQ'], df["BestSquatKg"][df["Place"] != 'DQ'], alpha=.1)
# plt.scatter(df["Age"][df["Sex"] == 'F'], df["BestSquatKg"][df["Sex"] == 'F'], alpha=.1)

plt.subplot(6,2,2)
#plt.hist(df["BodyweightKg"].dropna(), bins=30);
plt.scatter(df["BodyweightKg"][df["Place"] != 'DQ'], df["BestSquatKg"][df["Place"] != 'DQ'], alpha=.1)
# plt.scatter(df["BodyweightKg"][df["Sex"] == 'F'], df["BestSquatKg"][df["Sex"] == 'F'], alpha=.1)

plt.subplot(6,2,3)
plt.scatter(df["Age"][df["Place"] != 'DQ'], df["BestBenchKg"][df["Place"] != 'DQ'], alpha=.1)
# plt.scatter(df["Age"][df["Sex"] == 'F'], df["BestBenchKg"][df["Sex"] == 'F'], alpha=.1)

plt.subplot(6,2,4)
plt.scatter(df["BodyweightKg"][df["Place"] != 'DQ'], df["BestBenchKg"][df["Place"] != 'DQ'], alpha=.1)
# plt.scatter(df["BodyweightKg"][df["Sex"] == 'F'], df["BestBenchKg"][df["Sex"] == 'F'], alpha=.1)

plt.subplot(6,2,5)
plt.scatter(df["Age"][df["Place"] != 'DQ'], df["BestDeadliftKg"][df["Place"] != 'DQ'], alpha=.1)
# plt.scatter(df["Age"][df["Sex"] == 'F'], df["BestDeadliftKg"][df["Sex"] == 'F'], alpha=.1)

plt.subplot(6,2,6)
plt.scatter(df["BodyweightKg"][df["Place"] != 'DQ'], df["BestDeadliftKg"][df["Place"] != 'DQ'], alpha=.1)
# plt.scatter(df["BodyweightKg"][df["Sex"] == 'F'], df["BestDeadliftKg"][df["Sex"] == 'F'], alpha=.1)

plt.subplot(6,2,7)
plt.scatter(df["Age"][df["Place"] != 'DQ'], df["TotalKg"][df["Place"] != 'DQ'], alpha=.1)
# plt.scatter(df["Age"][df["Sex"] == 'F'], df["TotalKg"][df["Sex"] == 'F'], alpha=.1)

plt.subplot(6,2,8)
plt.scatter(df["BodyweightKg"][df["Place"] != 'DQ'], df["TotalKg"][df["Place"] != 'DQ'], alpha=.1)
# plt.scatter(df["BodyweightKg"][df["Sex"] == 'F'], df["TotalKg"][df["Sex"] == 'F'], alpha=.1)

plt.subplot(6,2,9)
plt.scatter(df["Age"][df["Place"] != 'DQ'], df["Wilks"][df["Place"] != 'DQ'], alpha=.1)
# plt.scatter(df["Age"][df["Sex"] == 'F'], df["Wilks"][df["Sex"] == 'F'], alpha=.1)

plt.subplot(6,2,10)
plt.scatter(df["BodyweightKg"][df["Place"] != 'DQ'], df["Wilks"][df["Place"] != 'DQ'], alpha=.1)
# plt.scatter(df["BodyweightKg"][df["Sex"] == 'F'], df["Wilks"][df["Sex"] == 'F'], alpha=.1)


# In[44]:


df = df[df["Place"] != 'DQ']


# ### Ok so the negative entries are from athlete that have been disqualified. It might be that the negative sign indicates the disqualification while the number represents their attempt ? We can further see in the plots that age clearly affects the performance and that athlete in their mid 20's to  30's show the best performance. The weight also seems to be correlated. The *Wilk* index accounts for athlete's weight.

# In[13]:


cdf = df.select_dtypes(exclude='O')
corr = cdf.drop('MeetID',axis=1).corr()
plt.figure(figsize=(15,15))
sns.heatmap(corr,annot=True, square=True, cmap='magma')


# In[14]:


df["BestSquatKg"].fillna(df["Squat4Kg"], inplace=True)
df["BestBenchKg"].fillna(df["Bench4Kg"], inplace=True)
df["BestDeadliftKg"].fillna(df["Deadlift4Kg"], inplace=True)

df.drop(["Squat4Kg", "Bench4Kg", "Deadlift4Kg"], axis=1, inplace=True)


# In[15]:


cdf = df.select_dtypes(exclude='O')
corr = cdf.drop('MeetID',axis=1).corr()
plt.figure(figsize=(15,15))
sns.heatmap(corr,annot=True, square=True, cmap='magma')


# # Feature correlation
# ### There is correlation between totalKg -> wilks -> the single lifts. Which is not suprising. Squat shows the highest correlation with totalkg and wilks. Also Bodyweight shows a positive correlation towards totalKg. Age is mostly uncorrelated. Best Bench shows the lowest correlation towards totalkg and wilks, since it is probably the category in which the least weight is lifted, interestingly Age shows a weak positive correaltion towards bench. 
# ### Taking another look at missing values reveals that Age also has quite a lot of missing entries. It will be hard to impute those values properly, since there seems to be not too much correlation towards the other features... maybe we can just take the mean/median?

# In[16]:


msn.matrix(df, sort='ascending')


# In[17]:


import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls


trace1 = go.Scatter3d(
    x=df["BestSquatKg"].iloc[:555],
    y=df["BestBenchKg"].iloc[:555],
    z=df["BestDeadliftKg"].iloc[:555],
    mode='markers',
    marker=dict(
        size=12,
        line=dict(
            color='rgba(217, 217, 217, 0.14)',
            width=0.5
        ),
        opacity=0.8
    )
)

data = [trace1]


layout = go.Layout(        
    title='Relation of single presses',
    hovermode='closest',
    scene = dict(
                    xaxis = dict(
                        title='BestSquatKg'),
                    yaxis = dict(
                        title='BestBenchKg'),
                    zaxis = dict(
                        title='BestDeadliftKg'),),
                    width=700,
                    
                  )
  

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='simple-3d-scatter')


# In[18]:


df.columns


# # Divisions
# 
# ### So after we looked a little bit at the relationship of age and weight on the lifting performance as well as performed some preliminary data cleaning lets further explore the data. The Division feature might be interesting to look into in a bit more detail.

# In[20]:


stat_min = 500
div = (df["Division"].value_counts() < stat_min)
df["Division"].fillna('Misc', inplace=True)
df["Division"] =  df["Division"].apply(lambda x: 'Misc' if div.loc[x] == True else x)


# In[54]:


grouped = df.groupby("Equipment")
plt.figure(figsize=(15,15))
for e,i in enumerate(grouped):
    print("Stats for {}".format(i[0]))
    for j in i[1]:
        if i[1][j].dtype != 'O' and j != 'MeetID':
            vmin = i[1][j].min()
            vmax = i[1][j].max()
            vmean = i[1][j].mean()
            vmedian = i[1][j].median()
            print(j)
            print("    min: {}".format(vmin))
            print("    max: {}".format(vmax))
            print("    mean: {}".format(round(vmean,2)))
            print("    median: {}".format(vmedian))
            print('- '*10)
    print('*'*20)
    
    plt.subplot(3,2,e+1)
    plt.title(i[0])
    sns.barplot(x=i[1]["Division"].value_counts()[:10], y=i[1]["Division"].value_counts()[:10].index)


# In[57]:


sns.violinplot(x='Equipment', y='TotalKg', data=df, hue='Sex')


# # To be continued...
