#!/usr/bin/env python
# coding: utf-8

# # AFL Time Series Data Analysis with Result and AFL Fantasy Predictions

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import ttest_ind, zscore

get_ipython().run_line_magic('matplotlib', 'inline')

#Supresses scientific notation
pd.set_option('display.float_format', lambda x: '%.2f' % x)

import warnings
warnings.filterwarnings('ignore')


# In[ ]:


df = pd.read_csv('../input/stats.csv')

df.head()


# In[ ]:


df.dtypes


# ### Creating "Age" column:

# In[ ]:


#Converting date objects to datetime:
df['D.O.B'] = pd.to_datetime(df['D.O.B'], format='%Y'+'-'+'%m'+'-'+'%d')
df['Date'] = pd.to_datetime(df['Date'], format='%Y'+'-'+'%m'+'-'+'%d')

#Creating Age column:
df.insert(2,'Age',(df['Date']-df['D.O.B'])/np.timedelta64(1,'Y'))

df.drop('D.O.B',axis=1,inplace=True)

#Changing WinLoss to numerical values
df.WinLoss.replace(['W', 'L', 'D'],[1,0,.5],inplace=True)


# ### Fixing Null Values:

# In[ ]:


#Removing post season "Rounds" (they also have null values)
round_list = ['QF', 'SF', 'PF', 'GF', 'EF']
df = df[~df.Round.isin(round_list)]


# ### Creating a alphabetically sorted "Team, Opposition" column to be able to group by game:

# In[ ]:


df['Teams'] = df[['Team','Opposition']].values.tolist()


# In[ ]:


df.Teams = df.Teams.apply(sorted).apply(', '.join)


# ### Creating Points Per Percent Played column:

# In[ ]:


#Creating "points per percent of game played" columns
df.insert(10, 'PointsPerPercentPlayed',(df['Goals']+df['Behinds'])/df['PercentPlayed'])


# ## Feature info:

# In[ ]:


df.describe()


# In[ ]:


number_list = df.select_dtypes(include=['number']).columns


# In[ ]:


fig,ax = plt.subplots(6, 5)

m=0
for i in range(6):
    for j in range(5):

        df[number_list[m]].plot(kind='hist',bins=20,ax=ax[i,j],figsize=(30, 30),
                                edgecolor='k').set_title(number_list[m])
        m+=1


# In[ ]:


df = df.set_index(['Season','Round']).sort_index()


# # Predicting the outcome of a game:

# ## Normalizing all players by round (n1):

# In[ ]:


df_ML = df.copy()


# In[ ]:


scaling_list = ['PointsPerPercentPlayed','Disposals', 'Kicks', 'Marks','Handballs',
                'Goals', 'Behinds','Hitouts', 'Tackles', 'Rebound50s','Inside50s',
                'Clearances','Clangers', 'FreesFor', 'FreesAgainst','BrownlowVotes',
                'ContendedPossessions', 'UncontendedPossessions','ContestedMarks',
                'MarksInside50', 'OnePercenters', 'Bounces', 'GoalAssists']

normalize_list = ['Age', 'Height', 'Weight', 'Score', 'Margin']

#Copying lists for AFL Fantasy analysis below:
new_scaling_list = scaling_list.copy()
new_normalize_list = normalize_list.copy()


lose = ['Age', 'Height', 'Weight'] #For rows that I don't want to shift


# In[ ]:


templist=[]
for col in scaling_list:
    df_ML[col+'_n1'] = df_ML.groupby(['Season','Round'])[col].transform(lambda x:(x.astype(float) - min(x))/(max(x)-min(x)))    
    templist.append(col+'_n1')

scaling_list += templist


# In[ ]:


templist=[]
for col in normalize_list:
    df_ML[col+'_n1'] = df_ML.groupby(['Season','Round'])[col].transform(lambda x: zscore(x,ddof=1))
    templist.append(col+'_n1')
    
    if col.startswith('Age')|col.startswith('Height')|col.startswith('Weight'):
        lose.append(col+'_n1')

normalize_list += templist


# ## Normalizing further by game (n2):

# In[ ]:


df_ML = df_ML.set_index(['Teams'],append=1).sort_index()


# In[ ]:


templist=[]
for col in scaling_list:
    df_ML[col+'_n2'] = df_ML.groupby(['Season','Round','Teams'])[col].transform(lambda x:(x.astype(float) - min(x))/(max(x)-min(x)))
    templist.append(col+'_n2')

scaling_list += templist


# In[ ]:


templist=[]
for col in normalize_list:
    df_ML[col+'_n2'] = df_ML.groupby(['Season','Round','Teams'])[col].transform(lambda x: zscore(x,ddof=1))
    templist.append(col+'_n2')
    
    if col.startswith('Age')|col.startswith('Height')|col.startswith('Weight'):
        lose.append(col+'_n2')

normalize_list += templist


# In[ ]:


df_ML.fillna(0,inplace=True)


# # Comparing points made per percent of game to various characteristics relative to players in each game:

# In[ ]:


df_ML.reset_index(inplace=True)
df_ML.drop(['Season','Round'],axis=1,inplace=True)

df_ML.set_index(['Date','Teams','Team'],inplace=True)
df_ML.sort_index(inplace=True)


# ## Scaling the data by the percent played:

# In[ ]:


templist=[]
for col in scaling_list:
    df_ML[col+'Scale'] = df_ML[col].multiply(df_ML.PercentPlayed/100,axis=0)
    templist.append(col+'Scale')
    
scaling_list += templist


# In[ ]:


number_list = scaling_list+normalize_list


# In[ ]:


df_ML[number_list] = df_ML[number_list].groupby(['Date','Teams','Team']).mean()


# ## Offsetting columns for time series analysis:

# In[ ]:


df_ML.reset_index(inplace=True)
df_ML.drop(['Player','Position','PercentPlayed','Teams'],axis=1,inplace=True)
df_ML.drop_duplicates(inplace=True)


# In[ ]:


lose #Columns whose rows that I don't want to shift


# In[ ]:


Offset_List = number_list + ['WinLoss']

for x in lose:
    Offset_List.remove(x)


# In[ ]:


PointsList=[]

for column_name in Offset_List:
    df_ML[column_name+'Shift1'] = df_ML.set_index('Date',append=1).sort_index(level=1)        .groupby('Team')[column_name].shift().reset_index(['Date'], drop=1)
    
    PointsList.append(column_name+'Shift1')
    
    for i in range(2,6):
        df_ML[column_name+'Shift'+str(i)] = df_ML.set_index('Date',append=1).sort_index(level=1)            .groupby('Team')[column_name].shift(i).reset_index(['Date'], drop=1)
        
        df_ML[column_name+'Exp'+str(i)] = df_ML.set_index('Date',append=1).sort_index(level=1)            .groupby('Team')[column_name].rolling(window=i,min_periods=i,win_types='exponential')            .mean().groupby('Team').shift().reset_index(['Team','Date'], drop=1)
        
        PointsList.append(column_name+'Shift'+str(i))
        PointsList.append(column_name+'Exp'+str(i))


# In[ ]:


Offset_List.remove('WinLoss')

df_ML.drop(Offset_List,axis=1,inplace=True)


# In[ ]:


#Fixing nulls again because of shifting:
df_ML.dropna(inplace=True)


# ## Win and Loss Distributions of Features:

# In[ ]:


df_ML_win = df_ML.select_dtypes(include=['number'])[df_ML.WinLoss==1]


# In[ ]:


df_ML_loss = df_ML.select_dtypes(include=['number'])[df_ML.WinLoss==0]


# In[ ]:


number_list = df_ML_win.columns
len(number_list)


# ### Checking distribution differences between winners and losers via hypothesis testing:

# In[ ]:


series_list=[]

for m in range(1750):
    test_stat1, p_value1 = ttest_ind(df_ML_win[number_list[m]], df_ML_loss[number_list[m]])
    
    series_list.append([number_list[m], test_stat1])

df_stat = pd.DataFrame(series_list, columns=['column_name','test_stat_mean'])
df_stat.set_index('column_name',inplace=True)
df_stat.replace([np.inf, -np.inf], np.nan,inplace=True)


# In[ ]:


df_stat.dropna().abs().sort_values('test_stat_mean',ascending=False)['test_stat_mean'].head(20)


# ### The features with the biggest mean discrepancy between the winning and losing teams are Age, previous Margin, Inside 50s, and previous Win-Loss record. 

# # Machine Learning:
# ## Making dummy variables:

# In[ ]:


df_ML.select_dtypes(include=['object']).head()


# In[ ]:


categorical_list = df_ML.select_dtypes(include=['object']).columns


# In[ ]:


df_ML = pd.get_dummies(df_ML,columns=categorical_list,drop_first=True)


# In[ ]:


df_ML = df_ML[df_ML.WinLoss!=.5]


# ### Alternating between dropping wins and losses to remove one team from each game:

# In[ ]:


row_list=[]
for row in range(len(df_ML)):
    if (row % 2!=0) & (df_ML.WinLoss.iloc[row]==0):
        row_list.append(df_ML.index[row])
    elif (row % 2==0) & (df_ML.WinLoss.iloc[row]!=0):
        row_list.append(df_ML.index[row])


# In[ ]:


df_ML.drop(row_list,inplace=True)


# In[ ]:


df_ML.WinLoss.value_counts()


# ## Spliting Data and Creating Model:

# In[ ]:


X = df_ML.drop(['WinLoss','Date'],axis=1)
y = df_ML['WinLoss']


# In[ ]:


from sklearn.linear_model import LogisticRegression

#for function below
from sklearn.model_selection import TimeSeriesSplit

from time import time
from sklearn.metrics import make_scorer,confusion_matrix,accuracy_score,    precision_score,recall_score,f1_score,roc_auc_score,matthews_corrcoef


# In[ ]:


def metrics_function1(target,pred):
    return accuracy_score(target, pred),precision_score(target, pred),        recall_score(target, pred),f1_score(target, pred),        roc_auc_score(target, pred),matthews_corrcoef(target, pred)

def TSer1_TEST(clf,X_all,y_all,folds_num,row_factor):
    start=time()
    
    TSer1=TimeSeriesSplit(n_splits=folds_num)
    print ('{}:'.format(clf.__class__.__name__),'\n')
    
    acc_list_train=[]
    acc_list_test=[]
    prc_list_train=[]
    prc_list_test=[]
    rcal_list_train=[]
    rcal_list_test=[]
    f1_list_train=[]
    f1_list_test=[]
    matt_list_train=[]
    matt_list_test=[]
    AUC_list_train=[]
    AUC_list_test=[]
    
    samp_size=X_all.shape[0]//row_factor
    
    for fold,(train_index,target_index) in enumerate(TSer1.split(X_all[:samp_size],
                                                                y_all[:samp_size])):
        X_train=X_all.iloc[train_index].values
        y_train=y_all.iloc[train_index].values

        X_test=X_all.iloc[target_index].values
        y_test=y_all.iloc[target_index].values
        
        clf.fit(X_train,y_train)
        y_pred1=clf.predict(X_train)
        y_pred2=clf.predict(X_test)

        train_acc,train_prc,train_rcal,train_f1,train_auc,train_matt=metrics_function1(y_train,y_pred1)
        
        test_acc,test_prc,test_rcal,test_f1,test_auc,test_matt=metrics_function1(y_test,y_pred2)
        
        acc_list_train.append(train_acc)
        acc_list_test.append(test_acc)
        prc_list_train.append(train_prc)
        prc_list_test.append(test_prc)
        rcal_list_train.append(train_rcal)
        rcal_list_test.append(test_rcal)
        
        f1_list_train.append(train_f1)
        f1_list_test.append(test_f1)
        matt_list_train.append(train_matt)
        matt_list_test.append(test_matt)
        AUC_list_train.append(train_auc)
        AUC_list_test.append(test_auc)
    
    print("Averages:"'\n')
    
    print("Train acc: {}, Test acc: {}".format(np.mean(acc_list_train),
                                               np.mean(acc_list_test)))
    print("Train prc: {}, Test prc: {}".format(np.mean(prc_list_train),
                                               np.mean(prc_list_test)))
    print("Train recall: {}, Test recall: {}".format(np.mean(rcal_list_train),
                                                     np.mean(rcal_list_test)),'\n')
    
    print("Train f1: {}, Test f1: {}".format(np.mean(f1_list_train),
                                             np.mean(f1_list_test)))
    print("Train MattCC: {}, Test MattCC: {}".format(np.mean(matt_list_train),
                                                     np.mean(matt_list_test)))
    print("Train AUC: {}, Test AUC: {}".format(np.mean(AUC_list_train),
                                               np.mean(AUC_list_test)),'\n'*2)
        
    print("Sample Size: {}, Folds Num: {}, Time: {}".format(samp_size,folds_num,
                                                            time()-start),'\n'*2)


# In[ ]:


clf_A1 = LogisticRegression(penalty='l1',tol=1e-1,C=.15,solver='liblinear',random_state=0)


# In[ ]:


TSer1_TEST(clf_A1, X, y, 5, 1)


# ### Previous win percentage of odds favorites from 2009-2013
# (according to https://www.bigfooty.com/forum/threads/how-often-do-favourites-win.1004437/)
# 
# ##### In 2009 it was 50/72 or 69.4% - (only data available)
# ##### In 2010 it was 109/176 or 61.9%
# ##### In 2011 it was 142/187 or 75.9%
# ##### In 2012 it was 155/198 or 78.2%
# ##### So far in 2013 it was 41/54 or 75.9%
# 
# ### This model predicts the winner over 80% of the time.

# # Predicting AFL Fantasy points:

# In[ ]:


df_F = df.copy()
fantasy_points = {'Kicks':3,'Handballs':2,'Marks':3,'Tackles':4,'FreesFor':1,
                  'FreesAgainst':-3,'Hitouts':1,'Goals':6,'Behinds':1}


# In[ ]:


new_list=[]

#Creating fantasy columns
for keys in fantasy_points.keys():
    df_F[keys+'_fantasy'] = df_F[keys]*fantasy_points[keys]
    new_list.append(keys+'_fantasy')
    
df_F['fantasy_points'] = pd.Series()   
    
for cols in fantasy_points.keys():
    df_F.fantasy_points = df_F.fantasy_points.add(df_F[cols+'_fantasy'],fill_value=0)

df_F['target'] = df_F.fantasy_points


# In[ ]:


df_F[new_list+['fantasy_points']].head()


# In[ ]:


df_F.insert(11, 'FantasyPerPercentPlayed',df_F['fantasy_points']/df_F['PercentPlayed'])


# In[ ]:


df_F.drop(new_list,axis=1,inplace=True)
new_scaling_list = new_scaling_list + ['fantasy_points'] + ['FantasyPerPercentPlayed']


# ## Fantasy points by position:

# In[ ]:


print(df_F.Position.value_counts())

hist_names = df_F.Position.value_counts().index.drop('Midfield, Ruck')


# In[ ]:


for names in hist_names:
    print(names+':','\n')
    print(df_F[df_F.Position==names].fantasy_points.describe(),'\n'*2)


# ### There is a commonly held belief that, in AFL Fantasy, midfielders are the best position for points, and this information confirms that.

# ## Normalizing all players by round (n1):

# In[ ]:


templist=[]
for col in new_scaling_list:
    df_F[col+'_n1'] = df_F.groupby(['Season','Round'])[col].transform(lambda x:(x.astype(float) - min(x))/(max(x)-min(x)))    
    templist.append(col+'_n1')

new_scaling_list += templist


# In[ ]:


templist=[]
for col in new_normalize_list:
    df_F[col+'_n1'] = df_F.groupby(['Season','Round'])[col].transform(lambda x: zscore(x,ddof=1))
    templist.append(col+'_n1')

new_normalize_list += templist


# ## Normalizing further by game (n2):

# In[ ]:


df_F = df_F.set_index(['Teams'],append=1).sort_index()


# In[ ]:


templist=[]
for col in new_scaling_list:
    df_F[col+'_n2'] = df_F.groupby(['Season','Round','Teams'])[col].transform(lambda x:(x.astype(float) - min(x))/(max(x)-min(x)))
    templist.append(col+'_n2')

new_scaling_list += templist


# In[ ]:


templist=[]
for col in new_normalize_list:
    df_F[col+'_n2'] = df_F.groupby(['Season','Round','Teams'])[col].transform(lambda x: zscore(x,ddof=1))
    templist.append(col+'_n2')

new_normalize_list += templist


# In[ ]:


df_F.fillna(0,inplace=True)


# # Comparing points made per percent of game to various characteristics relative to players in each game:

# In[ ]:


df_F.reset_index(inplace=True)
df_F.drop(['Season','Round'],axis=1,inplace=True)

df_F = df_F.set_index(['Date','Teams','Team']).sort_index()


# ## Scaling the data by the percent played:

# In[ ]:


templist=[]
for col in new_scaling_list:
    df_F[col+'Scale'] = df_F[col].multiply(df_F.PercentPlayed/100,axis=0)
    templist.append(col+'Scale')
    
new_scaling_list += templist


# In[ ]:


new_number_list = new_scaling_list+new_normalize_list


# ## Offsetting columns for time series analysis:

# In[ ]:


df_F.reset_index(inplace=True)
df_F.drop(['Teams','PercentPlayed'],axis=1,inplace=True)


# In[ ]:


Offset_Fantasy_List = new_number_list + ['WinLoss']

for x in lose:
    Offset_Fantasy_List.remove(x)


# ### Creating a 5-game auto-regression and exponentially weighted moving average for all relevant features:

# In[ ]:


FantasyList=[]

for column_name in Offset_Fantasy_List:
    df_F[column_name+'Shift1'] = df_F.set_index('Date',append=1).sort_index(level=1)        .groupby('Player')[column_name].shift().reset_index(['Date'], drop=1)
 
    FantasyList.append(column_name+'Shift1')

    for i in range(2,6):
        df_F[column_name+'Shift'+str(i)] = df_F.set_index('Date',append=1).sort_index(level=1)            .groupby('Player')[column_name].shift(i).reset_index(['Date'], drop=1)

        df_F[column_name+'Exp'+str(i)] = df_F.set_index('Date',append=1).sort_index(level=1)            .groupby('Player')[column_name].rolling(window=i,min_periods=i,win_types='exponential')            .mean().groupby('Player').shift().reset_index(['Player','Date'], drop=1)

        FantasyList.append(column_name+'Shift'+str(i))
        FantasyList.append(column_name+'Exp'+str(i))


# In[ ]:


df_F.drop(Offset_Fantasy_List,axis=1,inplace=True)


# ### Offsetting the target to see the error for predictions within 5 games:

# In[ ]:


target_list = []

for i in range(1,5):
    df_F['targetShift'+str(-i)] = df_F.set_index('Date',append=1).sort_index(level=1).groupby('Player')['target'].shift(-i).reset_index('Date', drop=1)
    
    target_list.append('targetShift'+str(-i))


# In[ ]:


target_list = ['target'] + target_list
target_list


# In[ ]:


#Fixing nulls again because of shifting:
df_F.dropna(inplace=True)


# ## Correlation Between AFL Fantasy Points and Other Features by Position:

# In[ ]:


column_list = df_F.select_dtypes(include=['number']).drop(['target']+target_list,axis=1).columns


# ## Defender:

# In[ ]:


df_F[df_F.Position=='Defender'][column_list].corrwith(df_F[df_F.Position=='Defender'].target).sort_values(ascending=False)


# ## Forward:

# In[ ]:


df_F[df_F.Position=='Forward'][column_list].corrwith(df_F[df_F.Position=='Forward'].target).sort_values(ascending=False)


# ## Midfield:

# In[ ]:


df_F[df_F.Position=='Midfield'][column_list].corrwith(df_F[df_F.Position=='Midfield'].target).sort_values(ascending=False)


# ## Midfield, Forward:

# In[ ]:


df_F[df_F.Position=='Midfield, Forward'][column_list].corrwith(df_F[df_F.Position=='Midfield, Forward'].target).sort_values(ascending=False)


# ## Ruck:

# In[ ]:


df_F[df_F.Position=='Ruck'][column_list].corrwith(df_F[df_F.Position=='Ruck'].target).sort_values(ascending=False)


# ## Forward, Ruck:

# In[ ]:


df_F[df_F.Position=='Forward, Ruck'][column_list].corrwith(df_F[df_F.Position=='Forward, Ruck'].target).sort_values(ascending=False)


# ## Defender, Midfield:

# In[ ]:


df_F[df_F.Position=='Defender, Midfield'][column_list].corrwith(df_F[df_F.Position=='Defender, Midfield'].target).sort_values(ascending=False)


# ## Defender, Forward:

# In[ ]:


df_F[df_F.Position=='Defender, Forward'][column_list].corrwith(df_F[df_F.Position=='Defender, Forward'].target).sort_values(ascending=False)


# # Machine Learning:
# ## Making dummy variables:

# In[ ]:


df_F.select_dtypes(include=['object']).head()


# In[ ]:


categorical_list2 = df_F.select_dtypes(include=['object']).drop('Player',axis=1).columns


# In[ ]:


df_F = pd.get_dummies(df_F,columns=categorical_list2,drop_first=True)


# ## Spliting Data and Creating Model:

# In[ ]:


X2 = df_F.drop(target_list+['Date','Player'],axis=1)

#Assigns variables to offset target values: 
for i in range(len(target_list)):
    globals()['y2_'+str(i)] = df_F[target_list[i]]


# In[ ]:


from sklearn.linear_model import Ridge

from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score


# In[ ]:


def metrics_function2(target,pred):
    return mean_squared_error(target,pred),mean_absolute_error(target,pred),r2_score(target,pred)

def TSer2_TEST(clf,X_all,y_all,folds_num,row_factor):
    start=time()
    
    TSer2=TimeSeriesSplit(n_splits=folds_num)
    print ('{}:'.format(clf.__class__.__name__),'\n')
    
    samp_size=X_all.shape[0]//row_factor
    
    MSE_list_train=[]
    MSE_list_test=[]
    MAE_list_train=[]
    MAE_list_test=[]
    R2_list_train=[]
    R2_list_test=[]
    
    for fold,(train_index,target_index) in enumerate(TSer2.split(X_all[:samp_size],
                                                                y_all[:samp_size])):
        X_train=X_all.iloc[train_index].values
        y_train=y_all.iloc[train_index].values

        X_test=X_all.iloc[target_index].values
        y_test=y_all.iloc[target_index].values
        
        clf.fit(X_train,y_train)
        y_pred1=clf.predict(X_train)
        y_pred2=clf.predict(X_test)

        train_MSE,train_MAE,train_R2=metrics_function2(y_train,y_pred1)
        test_MSE,test_MAE,test_R2=metrics_function2(y_test,y_pred2)
        
        MSE_list_train.append(train_MSE)
        MSE_list_test.append(test_MSE)
        MAE_list_train.append(train_MAE)
        MAE_list_test.append(test_MAE)
        R2_list_train.append(train_R2)
        R2_list_test.append(test_R2)
        
    print("Train MSE: {}, Test MSE: {}".format(np.mean(MSE_list_train),
                                               np.mean(MSE_list_test)),'\n'*2) 
    print("Train MAE: {}, Test MAE: {}".format(np.mean(MAE_list_train),
                                               np.mean(MAE_list_test)),'\n'*2)
    print("Train R2: {}, Test R2: {}".format(np.mean(R2_list_train),
                                             np.mean(R2_list_test)),'\n'*2)   
        
    print("Sample Size: {}, Folds Num: {}, Time: {}".format(samp_size,folds_num,
                                                            time()-start),'\n'*2)


# In[ ]:


clf_A2 = Ridge(alpha=1e3,random_state=0)


# ## Predicting AFL Fantasy points for the next 5 games:

# ### Current game's point prediction:

# In[ ]:


TSer2_TEST(clf_A2, X2, y2_0, 5, 1)


# ### Two games prediction:

# In[ ]:


TSer2_TEST(clf_A2, X2, y2_1, 5, 1)


# ### Three games out:

# In[ ]:


TSer2_TEST(clf_A2, X2, y2_2, 5, 1)


# ### Four games out:

# In[ ]:


TSer2_TEST(clf_A2, X2, y2_3, 5, 1)


# ### Five games out:

# In[ ]:


TSer2_TEST(clf_A2, X2, y2_4, 5, 1)

