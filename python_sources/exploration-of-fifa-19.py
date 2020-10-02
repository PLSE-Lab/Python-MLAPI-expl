#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import re
import missingno as msno
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.offline as py
import plotly.graph_objs as go
pd.options.display.max_columns = 999
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
plt.style.use('seaborn-bright')


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# **Quick look**

# In[ ]:


df = pd.read_csv("../input/data.csv")
df.drop(columns='Unnamed: 0', inplace=True)
df.head(5)


# In[ ]:


columns_to_drop = ['ID', 'Real Face', 'Joined', 'Loaned From', 'Contract Valid Until', 'LS',
                   'ST', 'RS', 'LW', 'LF', 'CF','RF', 'RW', 'LAM','CAM','RAM', 'LM',
                   'LCB', 'CB', 'RCB', 'RB', 'LCM', 'CM', 'RCM', 'RM', 'LWB', 'LDM',
                   'CDM', 'RDM', 'RWB', 'LB','Flag', 'Club Logo']
df.drop(columns_to_drop, inplace=True, axis=1)


# In[ ]:


df.head(5)


# **Missing data**

# In[ ]:


null = df.isnull().sum()/df.shape[0]
plt.figure(figsize=(16,12))
null.plot.bar()
plt.title('Missing data in percent', fontsize=20)
plt.xticks(rotation=60)
plt.show()


# In[ ]:


column_null = null[9:].index
msno.matrix(df[column_null])


# Variables in which occurs missing data are often specific rows which has most of the data missing. That's why I'm going to drop these rows.

# In[ ]:


df.dropna(inplace=True)


# In[ ]:


df.isnull().sum()


# **Feature enginnering need for visualization**

# In[ ]:


def dollar_to_number(df_value):
    try:
        value = float(df_value[1:-1])
        dollar = df_value[-1:]

        if dollar == 'M':
            value = value * 1000000
        elif dollar == 'K':
            value = value * 1000
    except ValueError:
        value = 0
    return value

def height_to_cm(df_value):
    try:
        feet = int(df_value[0])
        inch = int(df_value[2:])
        
        new_value = (feet*30.48)+(inch*2.54)
    except ValueError:
        new_value = 0
    return new_value

def weight_to_kg(df_value):
    try:
        lbs = int(df_value[:-3])
        
        new_value = lbs*0.453592
    except ValueError:
        new_value = 0
    return new_value


# In[ ]:


df['Value'] = df['Value'].apply(dollar_to_number)
df['Release Clause'] = df['Release Clause'].apply(dollar_to_number)
df['Wage'] = df['Wage'].apply(dollar_to_number)
df['CM'] = df['Height'].apply(height_to_cm)
df['KG'] = df['Weight'].apply(weight_to_kg)

#positions = {['ST', 'RW', 'LW', 'CF', 'LF', 'LS', 'RS', 'RF']:'Attacker',
            #['CAM', 'CM', 'LM', 'RM', 'CDM', 'RCM','LCM', 'LDM', 'RDM', 'LAM', 'RAM']:'Middlefielder',
             #['LWB', 'RWB', 'CB', 'RB', 'LB', 'LCB', 'RCB']:'Defender'}
df['Position_Cat'] = df['Position'].replace(['ST', 'RW', 'LW', 'CF', 'LF', 'LS', 'RS', 'RF'], 'Attacker')
df['Position_Cat'] = df['Position_Cat'].replace(['CAM', 'CM', 'LM', 'RM', 'CDM', 'RCM','LCM', 'LDM', 'RDM', 'LAM', 'RAM'], 'Middlefielder')
df['Position_Cat'] = df['Position_Cat'].replace(['LWB', 'RWB', 'CB', 'RB', 'LB', 'LCB', 'RCB'], 'Deffender')


# In[ ]:


field_players = df[df['Position'] != 'GK']
field_players.drop(columns=['GKDiving','GKHandling', 'GKKicking',
                            'GKPositioning', 'GKReflexes'], inplace=True)
goalkeepers = df[df['Position'] == 'GK']
cat_columns = df.select_dtypes(include='object')
numeric_columns = df.select_dtypes(exclude='object')
numeric_columns_field = field_players.select_dtypes(exclude='object')
numeric_columns_GK = goalkeepers.select_dtypes(exclude='object')


# **Correlations heatmaps for field players and goalkeepers**

# In[ ]:


numeric_columns_field_corr = numeric_columns_field.corr()
numeric_columns_GK_corr = numeric_columns_GK.corr()
mask1 = np.zeros_like(numeric_columns_field_corr)
mask2 = np.zeros_like(numeric_columns_GK_corr)
mask1[np.triu_indices_from(mask1)] = True
mask2[np.triu_indices_from(mask2)] = True
fig = plt.figure(figsize=(15,20))
ax1 = fig.add_subplot(211)
ax1.title.set_text('Field players')
sns.heatmap(numeric_columns_field_corr, cmap='YlGnBu', annot=True, fmt='.1f', mask=mask1)
ax2 = fig.add_subplot(212)
ax2.title.set_text('Goalkeepers')
sns.heatmap(numeric_columns_GK_corr, cmap='YlGnBu', annot=True, fmt='.1f', mask=mask2)


# **Countries with most players**

# In[ ]:


top_10 = df['Nationality'].value_counts()[:20]

plt.figure(figsize=(16,10))
sns.barplot(top_10.index, top_10.values)
plt.xticks(rotation=45)
plt.title('Most frequent nationality of player')
plt.show()


# **Basic statistics**

# In[ ]:


plt.figure(figsize=(16,10))
sns.countplot(x='Preferred Foot', data=df)
plt.title('Foot preferation')


# In[ ]:


fix, (ax1,ax2) = plt.subplots(1, 2, figsize=(16,10))
sns.barplot(x=df['Work Rate'].value_counts().index, y=df['Work Rate'].value_counts().values, data=df, ax=ax1)
ax1.tick_params(rotation=45)
ax1.title.set_text('Work rate')
sns.countplot(x='Body Type', data=df, ax=ax2, order=df['Body Type'].value_counts().index)
ax2.tick_params(rotation=45)
ax2.title.set_text('Body Type')


# In[ ]:


f, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,10))
sns.countplot(x='Position_Cat', data=df, ax=ax1, order=['GK', 'Deffender', 'Middlefielder', 'Attacker'])
sns.violinplot(x='Position_Cat', y='Overall', data=df, ax=ax2, order=['GK', 'Deffender', 'Middlefielder', 'Attacker'])
plt.suptitle('Positions')
ax1.set_xlabel('Position category')
ax2.set_xlabel('Position category')
plt.show()


# In[ ]:


plt.figure(figsize=(16,10))
sns.scatterplot(df['KG'], df['CM'], alpha=0.5)
plt.title('Height vs Weight')
plt.arrow(110.222856,170, 0, 5, head_width=0.5)
plt.annotate('A. Akinfenwa', (107, 168), fontsize=12)
plt.show()


# In[ ]:


corr = round(df[['Value', 'Wage']].corr().iloc[1,0], 2)
plt.figure(figsize=(16,10))
sns.scatterplot(df['Value']/1000000, 'Wage', data=df, style='Position_Cat', hue='Preferred Foot', markers=['^','v', 'o','X'], palette='Set1')
plt.text(x=40,y=500000, s='Correlattion {}'.format(corr), fontsize=15)
plt.xlabel('Value in milions')
plt.title('Wage vs Value')
#plt.xlim(0,150)
#plt.ylim(0,600000)
plt.show()


# **Top 10 clubs**

# In[ ]:


top_10_club = df.groupby(by='Club').mean()['Overall'].sort_values(ascending=False)[:10].index
df10 = df[df['Club'].isin(top_10_club)]
plt.figure(figsize=(16,10))
sns.boxplot(x='Club', y='Overall', data=df10, order=top_10_club)
plt.title('Top 10 clubs rating based on overall of players')
plt.show()


# In[ ]:


top_10_value = df.groupby(by='Club').mean()['Value'].sort_values(ascending=False)[:10].index
value10 = df[df['Club'].isin(top_10_value)]
plt.figure(figsize=(16,10))
sns.boxplot(x='Club', y=df['Value']/1000000, data=value10, order=top_10_value)
plt.title('Top 10 clubs rating based on value of players')
plt.xticks(rotation=45)
plt.show()


# In[ ]:


top_10_sum = df.groupby(by='Club').sum()['Value'].sort_values(ascending=False)[:10]
plt.figure(figsize=(16,10))
sns.barplot(top_10_sum.index, top_10_sum.values/1000000)
plt.title('Most valuable clubs in milions')
plt.xticks(rotation=45)
plt.show()


# **Effect of jersey number**

# In[ ]:


plt.figure(figsize=(16,10))
sns.scatterplot(y='Overall', x='Jersey Number', data=df, hue='Position_Cat', size=df['Wage'])
plt.arrow(30, 95, -17, 0, head_width=0.85, head_length=0.5, fc='k', ec='k')
plt.annotate('Number 10 and 7', (31, 94), fontsize=15)
plt.xticks(np.linspace(0,100,11))
plt.show()


# **Radar graphs**

# In[ ]:


data_sort = pd.DataFrame()
best_features = df[numeric_columns.columns].groupby(df['Position_Cat']).mean()
for i, j in zip(range(best_features.shape[0]), best_features.index):
    best_9 = best_features.iloc[i,:].sort_values(ascending=False)
    #print(best_9)
    data_sort[j] = best_9[:15].index
    
best_Attacker = ['SprintSpeed', 'Acceleration', 'Agility', 'Balance', 'ShotPower', 'Jumping']
best_GK = ['GKReflexes', 'GKDiving', 'GKPositioning', 'GKHandling', 'GKKicking', 'Reactions']
best_Middlefielder = ['Balance', 'Agility', 'Acceleration', 'SprintSpeed', 'Stamina', 'ShortPassing']
best_Deffender = ['Strength', 'Jumping', 'Stamina', 'StandingTackle', 'Aggression', 'SlidingTackle']

labels = [best_Attacker,best_Middlefielder, best_Deffender, best_GK]


# In[ ]:


player_atk = df[df['Position_Cat'] == 'Attacker'][best_Attacker].sample(1)
player_def = df[df['Position_Cat'] == 'Deffender'][best_Deffender].sample(1)
player_gk = df[df['Position_Cat'] == 'GK'][best_GK].sample(1)
player_mid = df[df['Position_Cat'] == 'Middlefielder'][best_Middlefielder].sample(1)
#
stats1=player_atk.values.T
stats2=player_def.values.T
stats3=player_gk.values.T
stats4=player_mid.values.T
#
angles1=np.linspace(0, 2*np.pi, len(best_Attacker), endpoint=False)
angles2=np.linspace(0, 2*np.pi, len(best_Deffender), endpoint=False)
angles3=np.linspace(0, 2*np.pi, len(best_GK), endpoint=False)
angles4=np.linspace(0, 2*np.pi, len(best_Middlefielder), endpoint=False)
#
stats1=np.concatenate((stats1,[stats1[0]]))
stats2=np.concatenate((stats2,[stats2[0]]))
stats3=np.concatenate((stats3,[stats3[0]]))
stats4=np.concatenate((stats4,[stats4[0]]))
#
angles1=np.concatenate((angles1,[angles1[0]]))
angles2=np.concatenate((angles2,[angles2[0]]))
angles3=np.concatenate((angles3,[angles3[0]]))
angles4=np.concatenate((angles4,[angles4[0]]))
#
player = [player_atk, player_mid, player_def, player_gk]
angles = [angles1, angles2, angles3, angles4]
stats = [stats1, stats2, stats3, stats4]

fig  = plt.figure(figsize=(15,14))
for p, s in zip([0, 1, 2, 3],[1, 2, 3, 4]):
    ax = fig.add_subplot(2, 2, s, polar=True)
    ax.plot(angles[p], stats[p], 'o-', linewidth=2, label='Messi')
    ax.fill(angles[p], stats[p], alpha=0.25)
    ax.set_thetagrids(angles[p] * 180/np.pi, labels[p])
    ax.set_title(df.loc[player[p].index[0]]['Position_Cat'] + ': ' + df.loc[player[p].index[0]]['Name']
                 + '\n Nationality: ' + df.loc[player[p].index[0]]['Nationality']
                 + '\n Overall: ' + np.str(df.loc[player[p].index[0]]['Overall'])
                )
    fig.suptitle('Random players for each position', fontsize=16)


# **3D Scatter**

# In[ ]:


def scatter_3d(x, y, z):
    """Choose X, Y, Z."""
    trace1 = go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='markers',
        marker=dict(
            size=12,
            color=z,                
            colorscale='Viridis',  
            opacity=0.8

        ),text=df['Name']
    )

    data = [trace1]
    layout = go.Layout(
        scene=dict(
        xaxis=dict(
            title=x.name),
        yaxis=dict(
            title=y.name),
        zaxis=dict(
            title=z.name)),
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=0
        )
    )
    fig = go.Figure(data=data, layout=layout)
    py.iplot(fig, filename='3d-scatter-colorscale')


# In[ ]:


scatter_3d(df['Value'], df['Wage'], df['Overall'])


# **Finding similar player**

# In[ ]:


def find_player(c, player):
    sc = StandardScaler()
    pos = df[df['Name'] == player]['Position_Cat'].values[0]
    base_data = df[df['Position_Cat'] == pos].reset_index(drop=True)
    base_scales = sc.fit_transform(base_data[col_to_cluster])
    base = pd.DataFrame(columns=col_to_cluster, data=base_scales)
    
    kmeans = KMeans(n_clusters=c, random_state=1)
    k = kmeans.fit_predict(base[col_to_cluster])
    pred = pd.concat([base_data[col_to_cluster], base_data['Name'], base_data['Overall'],
                      pd.Series(k).rename('Cluster')], axis=1)
    pred['Cluster'] = pred['Cluster'].astype('category')
    
    player_predict = pred[pred['Name'] == player]['Cluster']
    
    top_5_similar = pred[pred['Cluster'] == player_predict.values[0]]
    top_5_similar = top_5_similar.sort_values(by='Overall', ascending=False)
    
    print(top_5_similar[:5])
    
    fig, ax = plt.subplots(figsize=(16,10))
    x = np.array(pred[col_to_cluster[0]])
    y = np.array(pred[col_to_cluster[1]])
    cluster = np.array(pred['Cluster'])
    for g in np.unique(cluster):
        i = np.where(cluster == g)
        ax.scatter(x[i], y[i], label=g)
        
    ax.legend()
    plt.xlabel(col_to_cluster[0])
    plt.ylabel(col_to_cluster[1])
    plt.show()
    
    return pred
    
    


# In[ ]:


col_to_cluster = ['Stamina', 'Strength']

pred = find_player(5, 'L. Messi')

