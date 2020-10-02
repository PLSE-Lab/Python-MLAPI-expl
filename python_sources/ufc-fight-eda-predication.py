#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


fighter_details = pd.read_csv("../input/ufcdata/raw_fighter_details.csv")
df = pd.read_csv("../input/ufcdata/raw_total_fight_data.csv", sep=';')


# First lets see the dataframe of the UFC Fight from 1993-2019

# In[ ]:


df.head() # The Match Data


# In[ ]:


fighter_details.head() #The fighter data 


# Come take a look the info of the match data

# In[ ]:


df.info()


# # Column definitions:
# 
# - `R_` and `B_` prefix signifies red and blue corner fighter stats respectively
# - `KD` is number of knockdowns
# - `SIG_STR` is no. of significant strikes 'landed of attempted'
# - `SIG_STR_pct` is significant strikes percentage
# - `TOTAL_STR` is total strikes 'landed of attempted'
# - `TD` is no. of takedowns
# - `TD_pct` is takedown percentages
# - `SUB_ATT` is no. of submission attempts
# - `PASS` is no. times the guard was passed?
# - `REV?`
# - `HEAD` is no. of significant strinks to the head 'landed of attempted'
# - `BODY` is no. of significant strikes to the body 'landed of attempted'
# - `CLINCH` is no. of significant strikes in the clinch 'landed of attempted'
# - `GROUND` is no. of significant strikes on the ground 'landed of attempted'
# - `win_by` is method of win
# - `last_round` is last round of the fight (ex. if it was a KO in 1st, then this will be 1)
# - `last_round_time` is when the fight ended in the last round
# - `Format` is the format of the fight (3 rounds, 5 rounds etc.)
# - `Referee` is the name of the Ref
# - `date` is the date of the fight
# - `location` is the location in which the event took place
# - `Fight_type` is which weight class and whether it's a title bout or not
# - `Winner` is the winner of the fight

# # First lets gather the data from the raw match data above 
# 1. Split the attemp and strike landed
# 2. Make Percentage to Fraction
# 3. Get the Fighter Division
# 4. Get Number of Round
# 5. Replace The Winner Name Column with R/B
# ****

# # Split The Strike Attemp and Strike Landed

# In[ ]:


df.columns


# In[ ]:


columns = ['R_SIG_STR.', 'B_SIG_STR.', 'R_TOTAL_STR.', 'B_TOTAL_STR.',
       'R_TD', 'B_TD', 'R_HEAD', 'B_HEAD', 'R_BODY','B_BODY', 'R_LEG', 'B_LEG', 
        'R_DISTANCE', 'B_DISTANCE', 'R_CLINCH','B_CLINCH', 'R_GROUND', 'B_GROUND']


# In[ ]:


attemp = '_att'
landed = '_landed'

for column in columns:
    df[column+attemp] = df[column].apply(lambda X: int(X.split('of')[1]))
    df[column+landed] = df[column].apply(lambda X: int(X.split('of')[0]))
    
df.drop(columns, axis=1, inplace=True)


# In[ ]:


df.head()


# # Make The Percentage to Fraction

# In[ ]:


pct_columns = ['R_SIG_STR_pct','B_SIG_STR_pct', 'R_TD_pct', 'B_TD_pct']

for column in pct_columns:
    df[column] = df[column].apply(lambda X: float(X.replace('%', ''))/100)


# # Get Fighter Division

# In[ ]:


def Division(X):
    for Division in weight_classes:
        if Division in X:
            return Division
    if X == 'Catch Weight Bout' or 'Catchweight Bout':
        return 'Catch Weight'
    else:
        return 'Open Weight'


# In[ ]:


weight_classes = ['Women\'s Strawweight', 'Women\'s Bantamweight', 
                  'Women\'s Featherweight', 'Women\'s Flyweight', 'Lightweight', 
                  'Welterweight', 'Middleweight','Light Heavyweight', 
                  'Heavyweight', 'Featherweight','Bantamweight', 'Flyweight', 'Open Weight']

df['weight_class'] = df['Fight_type'].apply(Division)


# In[ ]:


df['weight_class'].value_counts()


# # Number of Round

# In[ ]:


def get_rounds(X):
    if X == 'No Time Limit':
        return 1
    else:
        return len(X.split('(')[1].replace(')', '').split('-'))

df['no_of_rounds'] = df['Format'].apply(get_rounds)


# # Replace Winner Name with R/B

# In[ ]:


df['Winner'].isnull().sum()


# In[ ]:


df['Winner'].fillna('Draw', inplace=True) #fill the null value with draw


# In[ ]:


def get_renamed_winner(row):
    if row['R_fighter'] == row['Winner']:
        return 'Red'
    elif row['B_fighter'] == row['Winner']:
        return 'Blue'
    elif row['Winner'] == 'Draw':
        return 'Draw'

df['Winner'] = df[['R_fighter', 'B_fighter', 'Winner']].apply(get_renamed_winner, axis=1)


# In[ ]:


df['Winner'].value_counts()


# # Deal with fighter details data
# 1. `Inch` to `CM`
# 2. Merger `Fighter Data` with `Match Data`
# 3. Get Fighter `Age`

# # Inch to CM

# In[ ]:


def convert_to_cms(X):
    if X is np.NaN:
        return X
    elif len(X.split("'")) == 2:
        feet = float(X.split("'")[0])
        inches = int(X.split("'")[1].replace(' ', '').replace('"',''))
        return (feet * 30.48) + (inches * 2.54)
    else:
        return float(X.replace('"','')) * 2.54


# In[ ]:


fighter_details['Height'] = fighter_details['Height'].apply(convert_to_cms)
fighter_details['Reach'] = fighter_details['Reach'].apply(convert_to_cms)


# In[ ]:


fighter_details['Weight'] = fighter_details['Weight'].apply(lambda X: float(X.replace(' lbs.', '')) if X is not np.NaN else X)


# In[ ]:


fighter_details.head()


# # Merger Fighter Data with Match Data

# In[ ]:


new = df.merge(fighter_details, left_on='R_fighter', right_on='fighter_name', how='left')


# In[ ]:


new = new.drop('fighter_name', axis=1)


# In[ ]:


new.rename(columns={'Height':'R_Height',
                          'Weight':'R_Weight',
                          'Reach':'R_Reach',
                          'Stance':'R_Stance',
                          'DOB':'R_DOB'}, 
                 inplace=True)


# In[ ]:


new = new.merge(fighter_details, left_on='B_fighter', right_on='fighter_name', how='left')


# In[ ]:


new = new.drop('fighter_name', axis=1)


# In[ ]:


new.rename(columns={'Height':'B_Height',
                          'Weight':'B_Weight',
                          'Reach':'B_Reach',
                          'Stance':'B_Stance',
                          'DOB':'B_DOB'}, 
                 inplace=True)


# In[ ]:


new.head()


# # Get Age

# In[ ]:


new['R_DOB'] = pd.to_datetime(new['R_DOB'])
new['B_DOB'] = pd.to_datetime(new['B_DOB'])
new['date'] = pd.to_datetime(new['date'])


# In[ ]:


new['R_year'] = new['R_DOB'].apply(lambda x: x.year)
new['B_year'] = new['B_DOB'].apply(lambda x: x.year)
new['date_year'] = new['date'].apply(lambda x: x.year)


# In[ ]:


def get_age(row):
    B_age = (row['date_year'] - row['B_year'])
    R_age = (row['date_year'] - row['R_year'])
    if np.isnan(B_age)!=True:
        B_age = B_age
    if np.isnan(R_age)!=True:
        R_age = R_age
    return pd.Series([B_age, R_age], index=['B_age', 'R_age'])


# In[ ]:


new[['B_age', 'R_age']]= new[['date_year', 'R_year', 'B_year']].apply(get_age, axis=1)


# In[ ]:


new.drop(['R_DOB', 'B_DOB','date_year','R_year','B_year'], axis=1, inplace=True)


# # Fighter Country

# In[ ]:


new['country'] = new['location'].apply(lambda x : x.split(',')[-1])


# # EDA & Visualization

# In[ ]:


new['date_year'] = new['date'].apply(lambda x: x.year)


# In[ ]:


values = new['date_year'].sort_values(ascending=False).value_counts().sort_index()
labels = values.index

clrs = ['navy' if (y < max(values)) else 'black' for y in values ]

plt.figure(figsize=(15,8))
bar = sns.barplot(x=labels, y=values, palette=clrs)


ax = plt.gca()
y_max = values.max() 
ax.set_ylim(1)
for p in ax.patches:
    ax.text(p.get_x() + p.get_width()/2., p.get_height(), p.get_height(), 
        fontsize=10, color='black', ha='center', va='bottom')
    
plt.xlabel('Tahun')
plt.ylabel('Jumlah Pertandingan')
plt.title('UFC Event Per Year')
plt.show()


# UFC become popular since 2011 and have the most event happened on 2014

# In[ ]:


plt.figure(figsize=(10,5))
bar = sns.countplot(new['country'])
plt.xticks(rotation=90)
ax = plt.gca()
y_max = new['country'].value_counts().max() 
ax.set_ylim(1)
for p in ax.patches:
    ax.text(p.get_x() + p.get_width()/2., p.get_height(), p.get_height(), 
        fontsize=10, color='black', ha='center', va='bottom')

plt.title('Event by Country')    
plt.show()


# The first UFC match Held in Denver,Colorado, USA. And until now the sport is become so popular in USA 

# In[ ]:


women = new.weight_class.str.contains('Women')


# In[ ]:


women1 = len(new[women])
men = (len(new['weight_class'])) - len(new[women])


# In[ ]:


labels = 'Men Fight', 'Women Fight'
sizes = [men,women1]
explode = (0, 0.1,)  

fig1, ax1 = plt.subplots(figsize=(10,8))
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90 )
ax1.axis('equal') 

plt.show()


# The UFC Fight still dominated by the Male Match with 93.9% (4830) and The Female Match only have 6.1% (314)

# # Let see the win distribution between red and blue side

# In[ ]:


plt.figure(figsize=(15,8))
new['Winner'].value_counts()[:10].plot.pie(explode=[0.05,0.05,0.05],autopct='%1.1f%%',shadow=True)
plt.show()


# we can see that the red side win more often

# ## Then let see how the `Age` affect the winner of the match

# In[ ]:


new['R_age'] = new['R_age'].fillna(new['R_age'].median())


# In[ ]:


new['B_age'] = new['B_age'].fillna(new['B_age'].median())


# In[ ]:


f,ax=plt.subplots(1,2,figsize=(10,8))
sns.distplot(new['R_age'], ax=ax[0])

ax[0].set_title('R_age')
ax[0].set_ylabel('')
hist = sns.distplot(new['B_age'],ax=ax[1])

ax[1].set_title('B_age')
plt.show()


# Most of fighter are on their 27-35, because `Age` is a big factor in UFC, 
# in UFC the combination of of strength, agility and speed (among other skills) are important. These skills peak at around `Age` 27-35 and fighter's fighting at this `Age` should have higher likelyhood of winning the fight. 
# 
# Let's validate by grouping `Age` for Red and Blue fighters who have won the fight.

# In[ ]:


f,ax=plt.subplots(1,2,figsize=(10,8))
new[new['Winner']=='Red']['R_age'].value_counts().plot.bar(ax=ax[0])

ax[0].set_title('R_age')
ax[0].set_ylabel('')
bar = new[new['Winner']=='Blue']['B_age'].value_counts().plot.bar(ax=ax[1])

ax[1].set_title('B_age')
plt.show()


# As we can see Fighter with `Age` around 27-35 tahun tend to win more often

# ## Then Now lets see how Height can affect the match 

# In[ ]:


sns.lmplot(x='R_Height', y='R_Reach', data=new)
plt.show()


# The graph show that Higher height also got higher point in reach..
# Higher fighter can reach enemy easily

# In[ ]:


new['R_Height'] = new['R_Height'].fillna(new['R_Height'].mean())


# In[ ]:


new['B_Height'] = new['B_Height'].fillna(new['R_age'].mean())


# In[ ]:


fig, ax = plt.subplots(figsize=(14, 6))
sns.kdeplot(new.R_Height, shade=True, color='indianred', label='Red')
sns.kdeplot(new.B_Height, shade=True, label='Blue')
plt.xlabel('Height')
plt.title('Height Different')

plt.show()


# From the Graph Above Red Fighter tend to have higher Height thats why Red Side Have Higher Win.

# The fighter Division

# In[ ]:


plt.figure(figsize=(15,8))
sns.countplot(y=new['weight_class'])

sns.set()
sns.set(style="white")
plt.show()


# ## How the win happened ?

# In[ ]:


values = new['win_by'].value_counts()
labels = values.index

plt.figure(figsize=(15,8))

sns.barplot(x=values,y=labels, palette='RdBu')

plt.title('UFC Fight Win By')
plt.show()


# The win mostly Cause by Decision, following by KO/TKO and Submission

# 1. **DEC**: Decision (Dec) is a result of the fight or bout that does not end in a knockout in which the judges' scorecards are consulted to determine the winner; a majority of judges must agree on a result. A fight can either end in a win for an athlete, a draw, or a no decision.
# 
# 2. **SUB**: also referred to as a "tap out" or "tapping out" - is often performed by visibly tapping the floor or the opponent with the hand or in some cases with the foot, to signal the opponent and/or the referee of the submission
# 
# 3. **KO/TKO**: Knockout (KO) is when a fighter gets knocked out cold. (i.e.. From a standing to not standing position from receiving a strike.). Technical Knockout (TKO) is when a fighter is getting pummeled and is unable to defend him/herself further. The referee will step in and make a judgement call to end it and prevent the fighter from receiving any more unnecessary or permanent damage, and call it a TKO.[](http://)

# In[ ]:


bar = new.groupby(['weight_class', 'win_by']).size().reset_index().pivot(columns='win_by', index='weight_class', values=0)
bar.plot(kind='barh', stacked=True, figsize=(15,8))
plt.legend(bbox_to_anchor=(1.23, 0.99), loc=1, borderaxespad=0.)
plt.title('UFC Fight Outcome by Division')
plt.xlabel('Jumlah')
plt.ylabel('Divisi')
plt.show()


# on heavy division the match was mainly ended by TKO

# In[ ]:


bar = new.groupby(['date_year', 'win_by']).size().reset_index().pivot(columns='win_by', index='date_year', values=0)
bar.plot(kind='barh', stacked=True, figsize=(15,8))
plt.legend(bbox_to_anchor=(1.23, 0.99), loc=1, borderaxespad=0.)
plt.title('UFC Fight Outcome over the Years')
plt.xlabel('Jumlah')
plt.ylabel('Tahun')
plt.show()


# As year goes on, winning by submission is more common and winning by decission is increasing,while KO/TKO is remaining Steady  

# In[ ]:


Attempt = pd.concat([new['R_TOTAL_STR._att'], new['B_TOTAL_STR._att']], ignore_index=True)
Landed = pd.concat([new['R_TOTAL_STR._landed'], new['B_TOTAL_STR._landed']], ignore_index=True)


# In[ ]:


sns.jointplot(x=Attempt , y=Landed)
plt.show()


# The Higher Strike Attemp The Landed Strike tend to get higher as well

# In[ ]:


r_landed = new['R_TOTAL_STR._landed']
r_index = r_landed.index


# In[ ]:


b_landed = new['B_TOTAL_STR._landed']
b_index = b_landed.index


# In[ ]:


new['Winner'].head(9)


# In[ ]:


sns.lineplot(x=r_index[0:9], y=r_landed[0:9], color='r')
sns.lineplot(x=b_index[0:9], y=b_landed[0:9])
plt.show()


# In[ ]:


Fighter = pd.concat([new['R_fighter'], new['B_fighter']], ignore_index=True)


# In[ ]:


plt.figure(figsize=(10,8))
sns.countplot(y = Fighter, order=pd.value_counts(Fighter).iloc[:10].index)
plt.show()


# # UFC Prediction

# In[ ]:


from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, RandomizedSearchCV
from xgboost import XGBClassifier

import warnings
warnings.filterwarnings("ignore")


# # Deal with null Values

# In[ ]:


df = new.copy()


# In[ ]:


df.isnull().sum()


# In[ ]:


df = df.fillna(df.mean())


# In[ ]:


from statistics import mode 
df['B_Stance'] = df['B_Stance'].fillna(df['B_Stance'].mode()[0])
df['R_Stance'] = df['R_Stance'].fillna(df['R_Stance'].mode()[0])


# # Data Encoding

# In[ ]:


enc = LabelEncoder()


# In[ ]:


data_enc1 = df['weight_class']
data_enc1 = enc.fit_transform(data_enc1)

data_enc2 = df['R_Stance']
data_enc2 = enc.fit_transform(data_enc2)

data_enc3 = df['B_Stance']
data_enc3= enc.fit_transform(data_enc3)


# In[ ]:


data_enc1 = pd.DataFrame(data_enc1, columns=['weight_class'])
data_enc2 = pd.DataFrame(data_enc2, columns=['R_Stance'])
data_enc3 = pd.DataFrame(data_enc3, columns=['B_Stance'])


# In[ ]:


df[['weight_class']] = data_enc1[['weight_class']]
df[['R_Stance']] = data_enc2[['R_Stance']]
df[['B_Stance']] = data_enc3[['B_Stance']]


# In[ ]:


df = pd.concat([df,pd.get_dummies(df['win_by'], prefix='win_by')],axis=1)
df.drop(['win_by'],axis=1, inplace=True)


# In[ ]:


df['Winner_num'] = df.Winner.map({'Red':0,'Blue':1,'Draw':2})


# In[ ]:


df.head()


# In[ ]:


encode = df[['R_fighter','B_fighter','weight_class']].apply(enc.fit_transform)
encode.head()


# In[ ]:


df[['R_fighter','B_fighter','weight_class']] = encode[['R_fighter','B_fighter','weight_class']] 


# In[ ]:


df = df.dropna()
sum(df.isnull().sum())


# In[ ]:


plt.figure(figsize=(10,15))
sns.heatmap(df.corr()[['Winner_num']].sort_values(by='Winner_num', ascending=False),annot=True)
plt.show()


# # Normalize data with Standard Scaler

# In[ ]:


numerical = df.drop(['R_fighter','B_fighter','weight_class','no_of_rounds','Winner_num'], axis=1)


# In[ ]:


std = StandardScaler()
df_num = numerical.select_dtypes(include=[np.float, np.int])


# In[ ]:


numerical[list(df_num.columns)] = std.fit_transform(numerical[list(df_num.columns)])


# In[ ]:


df_fix = numerical.join(df[['R_fighter','B_fighter','weight_class','no_of_rounds','Winner_num']])


# In[ ]:


df_fix.head()


# ## Drop Unecessary Column

# In[ ]:


df_fix = df_fix.drop(['country','location','date_year','date','Referee','Format','last_round_time','Fight_type','Winner'], axis=1)


# # Modeling XGBoost

# In[ ]:


model = XGBClassifier()


# In[ ]:


X = df_fix.drop(['Winner_num'], axis=1)
y = df_fix['Winner_num']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.25, random_state = 42)


# In[ ]:


model.fit(X_train,y_train)


# In[ ]:


pred = model.predict(X_test)


# In[ ]:


Score = model.score(X_test,y_test)
print("Score: %.2f%%" % (Score * 100.0))


# In[ ]:


from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from itertools import cycle
lw=1


# In[ ]:


X1 = df_fix.drop(['Winner_num'], axis=1)
y1 = df_fix['Winner_num']


# In[ ]:


y1 = label_binarize(y1, classes=[0, 1, 2])
n_classes = y1.shape[1]


# In[ ]:


X1_train,X1_test,y1_train,y1_test = train_test_split(X1,y1,test_size = 0.25, random_state = 42)


# In[ ]:


pred_proba = model.predict_proba(X1_test)


# In[ ]:


fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y1_test[:, i], pred_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
colors = cycle(['blue', 'red', 'green'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([-0.05, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic for multi-class data')
plt.legend(loc="lower right")
plt.show()


# In[ ]:




