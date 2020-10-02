#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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


# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re


# In[ ]:


dataset = pd.read_csv('../input/data.csv')
pd.options.display.max_columns = None


# In[ ]:


from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# In[ ]:


dataset.head(5)


# In[ ]:


dataset.describe(include = 'all')


# In[ ]:


dataset.drop(['Unnamed: 0','Photo', 'Flag','Club Logo', 'Loaned From' ], axis = 1, inplace = True)


# In[ ]:


dataset.columns


# In[ ]:


dataset.info()


# In[ ]:


dataset.isna().sum()


# In[ ]:


#top 5 countries with highest no of players
dataset['Nationality'].value_counts().head(5)


# In[ ]:


sns.barplot(x=dataset['Nationality'].value_counts().head(5).index, y= dataset['Nationality'].value_counts().head(5).values,
           data = dataset)


# In[ ]:


df1 = dataset.loc[dataset['Nationality'].isin(['England', 'Germany', 'Spain', 'Argentina', 'France', 'Brazil', 'Italy',
       'Colombia', 'Japan', 'Netherlands', 'Sweden', 'China PR', 'Chile',
       'Republic of Ireland', 'Mexico', 'United States', 'Poland', 'Norway',
       'Saudi Arabia', 'Denmark', 'Korea Republic', 'Portugal', 'Turkey',
       'Austria', 'Scotland'])]
df1.head()


# In[ ]:


# 25 Countries with max no of players
plot1 = sns.countplot(x =df1['Nationality'] , data = dataset, )
plot1.set_xticklabels(plot1.get_xticklabels(),rotation=90)


# In[ ]:


#Total number of clubs present and top 5 clubs with highest number of players
dataset['Club'].value_counts().head(10)
dataset['Club'].nunique()


# In[ ]:


#Player with maximum Potential and Overall Performance
dataset.loc[(dataset['Overall']==max(dataset['Overall']))|(dataset['Potential']==max(dataset['Potential']))]


# In[ ]:


print('maximum overall performance: ', dataset.loc[dataset['Overall']==max(dataset['Overall'])]['Name'][0])


# In[ ]:


print('maximum performance:', dataset.loc[dataset['Potential']==max(dataset['Potential'])]['Name'][25])


# In[ ]:


pr_columns = ['Crossing', 'Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys',
       'Dribbling', 'Curve', 'FKAccuracy', 'LongPassing', 'BallControl',
       'Acceleration', 'SprintSpeed', 'Agility', 'Reactions', 'Balance',
       'ShotPower', 'Jumping', 'Stamina', 'Strength', 'LongShots',
       'Aggression', 'Interceptions', 'Positioning', 'Vision', 'Penalties',
       'Composure', 'Marking', 'StandingTackle', 'SlidingTackle', 'GKDiving',
       'GKHandling', 'GKKicking', 'GKPositioning', 'GKReflexes']


# In[ ]:


i=0
while(i<len(pr_columns)):
    print('Best '+ pr_columns[i]+ ':' , dataset.iloc[dataset[pr_columns[i]].idxmax()]['Name'])
    i=i+1


# In[ ]:


#Converting the Value and wage column to string to compare
i=0
while (i<10):
    dataset['Value'][i]= int(re.search(r'\d+', dataset['Value'][i]).group())
    i=i+1     


# In[ ]:


i=0
while (i<10):
    dataset['Wage'][i]= int(re.search(r'\d+', dataset['Wage'][i]).group())
    i=i+1  


# In[ ]:


dataset.head()


# In[ ]:


dataset.rename(columns={'Value' : 'Value M Eu', 'Wage' : 'Wage M Eu'}, inplace=True)


# In[ ]:


#Top valuable players among head(10)
print('Top valuable Player :', dataset.loc[dataset['Value M Eu']==max(dataset['Value M Eu'].head(10))]['Name'][2])


# In[ ]:


#Top earning players among head(10)
print('Top earning Player :', dataset.loc[dataset['Wage M Eu']==max(dataset['Wage M Eu'].head(10))]['Name'][0])


# In[ ]:


#freq dist plot
sns.distplot(dataset['Age'])


# In[ ]:


#plot between age and portetial
sns.jointplot(x='Age', y='Potential', data= dataset,kind='reg')


# In[ ]:


dataset.columns


# In[ ]:


player_attribute = ['Crossing', 'Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys',
       'Dribbling', 'Curve', 'FKAccuracy', 'LongPassing', 'BallControl',
       'Acceleration', 'SprintSpeed', 'Agility', 'Reactions', 'Balance',
       'ShotPower', 'Jumping', 'Stamina', 'Strength', 'LongShots',
       'Aggression', 'Interceptions', 'Positioning', 'Vision', 'Penalties',
       'Composure', 'Marking']
player_attribute[26]


# In[ ]:


dataset.loc[dataset['Name']=='L. Messi']


# In[ ]:


#top 5 player attribute for Lionel messi
series= dataset.loc[dataset['Name']=='L. Messi'][['Crossing', 'Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys',
       'Dribbling', 'Curve', 'FKAccuracy', 'LongPassing', 'BallControl',
       'Acceleration', 'SprintSpeed', 'Agility', 'Reactions', 'Balance',
       'ShotPower', 'Jumping', 'Stamina', 'Strength', 'LongShots',
       'Aggression', 'Interceptions', 'Positioning', 'Vision', 'Penalties',
       'Composure', 'Marking']]
series.transpose().sort_values(by=0, ascending=False).head()
                       


# In[ ]:


df1 = dataset.groupby(by=dataset['Position'])[player_attribute].agg('mean')
df1


# In[ ]:


#plotting radar graph
#plotting spider web
idx = 1
i=0
plt.figure(figsize=(15,45))
while i<=len(df1.index):
    df2 = df1.iloc[i, :].nlargest(5)
    cat = list(df2.index)
    values= list(df2.values)
    N = len(cat)
    x_as = [n/float(N)*2*np.pi for n in range(N)]
    x_as = x_as + [x_as[0]]
    values = values + [values[0]]
    
    ax = plt.subplot(10, 3, idx, polar=True)
    ax.set_theta_offset(np.pi/3)
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(20)
    
    plt.xticks(x_as[:-1],cat)
    plt.yticks([20,40,60,80,100], ['20','40','60','80','100'])
    
    ax.plot(x_as,values)
    ax.fill(x_as, values, 'b', alpha=0.1)
    plt.ylim(0, 100)
    plt.title(df1.index[i])
    i=i+1
    idx=idx+1


# In[ ]:


sns.heatmap(data=df1,cmap='viridis', linewidths=0.1)


# In[ ]:


#age vs sprint speed
sns.regplot(x='Age',y ='SprintSpeed', data=dataset, scatter_kws={'alpha':0.1, 's':10,'color':'blue'}, )


# In[ ]:


#Better is left footed or right footed
dataset['Preferred Foot'].value_counts()
sns.countplot(dataset['Preferred Foot'] )
dataset.columns
dataset.groupby(by=dataset['Preferred Foot'])['Overall'].agg(['count', 'mean'])


# In[ ]:


#leftfoot vs rightfoot
sns.lmplot(x='BallControl', y='Dribbling', data=dataset, scatter_kws={'alpha':0.1,'s': 5, 'color':'red'}, col='Preferred Foot' )


# In[ ]:


df2 = dataset[['Crossing', 'Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys',
       'Dribbling', 'Curve', 'FKAccuracy', 'LongPassing', 'BallControl',
       'Acceleration', 'SprintSpeed', 'Agility', 'Reactions', 'Balance',
       'ShotPower', 'Jumping', 'Stamina', 'Strength', 'LongShots',
       'Aggression', 'Interceptions', 'Positioning', 'Vision', 'Penalties',
       'Composure', 'Marking', 'StandingTackle', 'SlidingTackle']].corr()
df2


# In[ ]:


plt.figure(figsize=(8,4))
sns.heatmap(df2,linewidths=0.01, cmap='coolwarm')


# In[ ]:


sns.pairplot(df2.iloc[0:5,0:5])


# In[ ]:


#Year of service
import datetime
tday = datetime.date.today()
tday


# In[ ]:


dataset['Joined']=pd.to_datetime(dataset['Joined']) 
dataset.head(5)


# In[ ]:


#player with max days of service to their club
dataset['Service']= datetime.date(2019, 5,7)
dataset['Service'].astype(pd.datetime)


# In[ ]:


dataset['Service']= dataset['Service']- dataset['Joined']
dataset[['Name','Club','Service']].nlargest(5,'Service')
dataset.head()


# In[ ]:


#Messi rank in loyality for his team (as no of days he played)
dataset.sort_values(by=['Service'], ascending=False).loc[dataset['Service']>='5422 days'].reset_index(drop=False, inplace=False)


# In[ ]:


#Create a position value for all the players
dataset.columns
list =['ID','Nationality','Club','Value M Eu', 'Wage M Eu','Special','International Reputation','Body Type', 'Real Face',
       'Jersey Number', 'Joined','Contract Valid Until', 'Height', 'Weight', 'LS', 'ST', 'RS', 'LW',
       'LF', 'CF', 'RF', 'RW', 'LAM', 'CAM', 'RAM', 'LM', 'LCM', 'CM', 'RCM',
       'RM', 'LWB', 'LDM', 'CDM', 'RDM', 'RWB', 'LB', 'LCB', 'CB', 'RCB', 'RB','Release Clause', 'Service']
dataset.drop(list, axis=1, inplace= True)


# In[ ]:


dataset=dataset.dropna()


# In[ ]:


dataset['Position'].value_counts()
plot1 = sns.countplot(dataset['Position'])
plot1.set_xticklabels(plot1.get_xticklabels(),rotation=90)


# In[ ]:


dataset['Preferred Foot']= dataset['Preferred Foot'].map({'Left' : 0, 'Right':1})


# In[ ]:


#Building decision tree
X=dataset.loc[:,['Age', 'Overall', 'Potential', 'Preferred Foot','Crossing', 'Finishing',
       'HeadingAccuracy', 'ShortPassing', 'Volleys', 'Dribbling', 'Curve',
       'FKAccuracy', 'LongPassing', 'BallControl', 'Acceleration',
       'SprintSpeed', 'Agility', 'Reactions', 'Balance', 'ShotPower',
       'Jumping', 'Stamina', 'Strength', 'LongShots', 'Aggression',
       'Interceptions', 'Positioning', 'Vision', 'Penalties', 'Composure',
       'Marking', 'StandingTackle', 'SlidingTackle', 'GKDiving', 'GKHandling',
       'GKKicking', 'GKPositioning', 'GKReflexes']].values
Y=dataset.loc[:,'Position'].values


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test= train_test_split(X,Y,test_size=0.25, random_state=1)


# In[ ]:


from sklearn.tree import DecisionTreeClassifier
classifier= DecisionTreeClassifier('entropy',random_state=1)


# In[ ]:


classifier.fit(X_train, Y_train)


# In[ ]:


Y_pred= classifier.predict(X_test)
Y_pred


# In[ ]:


final =pd.DataFrame(data=[Y_test, Y_pred]).transpose()
final.head(5)


# In[ ]:


i=0
correct=0
wrong=0
while i<4537:
    if(final.iloc[i,0] == final.iloc[i,1]):
        correct=correct+1
    else:
        wrong=wrong+1
    i=i+1 
    
print(i)
print(correct)
print(wrong)


# In[ ]:


accuracy = correct/(correct+wrong)
accuracy

