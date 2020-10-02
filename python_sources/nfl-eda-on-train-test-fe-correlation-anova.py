#!/usr/bin/env python
# coding: utf-8

# # NFL EDA on Train & Test / FE, Correlation, ANOVA
# This notebook explores the training and the testing data on the following contents.
# 
# ## Content
# * Train/Test data period
# * Play Visualization
# * Feature Engineering & Data Manipulation
# * Correlation between Yards and numeric features
#     * Plotting Top 10 Correlated Features
# * One-way ANOVA on Categorical Features against Yards
#     * Plotting Most Significant 10 Features

# In[ ]:


from kaggle.competitions import nflrush
import pandas as pd
import numpy as np
import gc

import seaborn as sns; sns.set()
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")


# We are loading training data and testing data first. We will utilize the API provided by Kaggle to get the test data.

# In[ ]:


train_df = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2020/train.csv', low_memory=False)


# In[ ]:


env = nflrush.make_env()
iter_test = env.iter_test()

for idx, (test_df_tmp, sample_prediction_df) in enumerate(iter_test):
    if idx == 0:
        test_df = test_df_tmp.copy()
    else:
        test_df = pd.concat([test_df, test_df_tmp])
        
    env.predict(sample_prediction_df)


# The training data has 2,3171 plays and each play data has 22 rows corresponding to 22 players in the game from two different teams. Those data have tracking information (X, Y coordinate, etc) at the moment of handoffs being made:

# In[ ]:


train_df[train_df.PlayId == 20170907000118][['PlayId','TimeHandoff','Team','DisplayName','NflId','NflIdRusher','X','Y','Yards']]


# # Train/Test data period

# In[ ]:


def convert_dt(df):

    df["TimeHandoff"] = pd.to_datetime(df.TimeHandoff).dt.tz_localize(None)
    df["TimeSnap"] = pd.to_datetime(df.TimeSnap).dt.tz_localize(None)
    
    return df


# In[ ]:


train_df = convert_dt(train_df)
test_df = convert_dt(test_df)


# In[ ]:


train_df['TimeHandoff'].hist(bins=100, figsize=(13, 4),                             label='train', xrot=45)
test_df['TimeHandoff'].hist(bins=8, figsize=(13, 4),                            label='test', xrot=45)
plt.title('Train/Test Data Period', fontsize=14)
plt.legend(loc='upper right', prop={'size': 12})


# We can see the training data is made from the last two seasons (every NFL season starts from September and ends in December or early January). The test-set is the first two months of this season. Keep in mind this test data is just a temporal one and the actual test-set will be December 2019.

# # Play Visualization
# I'm applying the play data standarization scheme which the host explained here: https://www.kaggle.com/statsbymichaellopez/nfl-tracking-initial-wrangling-voronoi-areas. It makes offence directions to be right direction in every play.

# In[ ]:


def std_cols(df):
    
    df['X_std'] = df.apply(lambda x: x.X if x.PlayDirection == 'right'                           else 120-x.X, axis=1) 
    df['Y_std'] = df.apply(lambda x: x.Y if x.PlayDirection == 'right'                           else 53.3-x.Y, axis=1) 
    df['Orientation_std'] = df.apply(lambda x: x.Orientation                                      if x.PlayDirection == 'right'                                      else x.Orientation + 180, axis=1)
    df['YardLine_std'] = df.apply(lambda x: x.YardLine + 10                                   if (x.PlayDirection == 'right') &                                   (x.FieldPosition == x.PossessionTeam)                                   | (x.PlayDirection == 'left') &                                   (x.FieldPosition == x.PossessionTeam)                                   else 60 + (50-x.YardLine), axis=1)
    df['FieldPosition_std'] = df.apply(lambda x: 'left'                                        if x.FieldPosition ==                                        x.PossessionTeam                                        else 'right', axis=1) 
    
    df['OffenceDefence'] =     df.apply(lambda x: "offence" if ((x.Team == 'home')                                      & (x.PossessionTeam ==                                         x.HomeTeamAbbr)) |                                     ((x.Team == 'away') &                                      (x.PossessionTeam ==                                       x.VisitorTeamAbbr))                                     else "defence", axis=1)
    
    df.drop(['X', 'Y', 'Orientation', 'YardLine', 'FieldPosition'],             axis=1, inplace=True)
    
    return df


# In[ ]:


train_df = std_cols(train_df)
test_df = std_cols(test_df)


# In[ ]:


plays = train_df.PlayId.unique()[:8]

fig, axes = plt.subplots(4,2, figsize=(14,18))

for i, ax in enumerate(axes.flatten()):
    
    play = train_df[train_df.PlayId == plays[i]].copy()
    play['OffenceDefence'][play['NflId'] ==                            play['NflIdRusher'].values[0]]                             = "rusher"
    play.sort_values(by=['OffenceDefence'], inplace=True)
    
    ax.set_xlim(0,120)
    ax.set_ylim(0,53.3)
    
    YardLine = play.YardLine_std.values[10]
    ax.plot([YardLine, YardLine], [0, 53.3],             linestyle='--', color="gray")
    
    sns.scatterplot(x="X_std", y="Y_std",                     hue="OffenceDefence", data=play, ax=ax)

    ax.set_title('Yards: %d' % play['Yards'].values[0], fontsize=14)
    plt.legend(loc='upper right', prop={'size': 12})
    
    for player in play.iterrows():
        deg=player[1].Orientation_std
        sp=player[1].S
        # sp =100
        x_pos = player[1].X_std
        y_pos = player[1].Y_std
        x_direct = np.cos(deg/180*np.pi) * sp
        y_direct = np.sin(deg/180*np.pi) * sp
#         x_direct = np.cos((deg+90)/180*np.pi) * sp  # does not look correct
#         y_direct = np.sin((deg+90)/180*np.pi) * sp
        col = "g" if player[1].OffenceDefence == "rusher"     else "r" if player[1].OffenceDefence == "offence" else "b"

        ax.quiver(x_pos,y_pos,x_direct,y_direct, scale=50, color=col)

    fig.tight_layout()


# In the pictures above, offence directions are always right directions. Arrows are orientation and a length of an arrow means speed of that player.  

# # Feature Engineering & Data Manipulation

# We need to summarize 22 rows from a single play into a single row to predict Yards corresponding that play. First I'm adding some additional features.

# In[ ]:


def add_cols(df):
    
    df["DefenceTeam"] =     df.apply(lambda x: x.HomeTeamAbbr              if x.HomeTeamAbbr != x.PossessionTeam              else x.VisitorTeamAbbr, axis=1)
    df["OffenceisHome"] =     df.apply(lambda x: True if x.HomeTeamAbbr ==              x.PossessionTeam else False, axis=1)
    df["HandoffSnapDiff"] =     (df.TimeHandoff - df.TimeSnap).dt.seconds
    df["OffenceScoreBeforePlay"] =     df.apply(lambda x: x.HomeScoreBeforePlay              if x.HomeTeamAbbr == x.PossessionTeam              else x.VisitorScoreBeforePlay, axis=1)
    df["DefenceScoreBeforePlay"] =     df.apply(lambda x: x.HomeScoreBeforePlay              if x.HomeTeamAbbr != x.PossessionTeam              else x.VisitorScoreBeforePlay, axis=1)
    df["PlayerHeight"] =     df.PlayerHeight.apply(lambda x: int(x.split('-')[0])                           + int(x.split('-')[1]) /12)
    df["PlayerBirthDate"] = pd.to_datetime(df.PlayerBirthDate)
    df['age'] = ((df.TimeHandoff.dt.date -                   df.PlayerBirthDate.dt.date).dt.days /365.25)
    
    return df


# In[ ]:


train_df = add_cols(train_df)
test_df = add_cols(test_df)


# As some columns only have single values in 22 rows and some other columns have different values across those rows, we need different strategies on different columns. The picture below shows unique counts per play across the columns.

# In[ ]:


uniq_cnt=[];cols=[]

for col in train_df.columns:
    if col != 'PlayId':
        cols.append(col)
        uniq_cnt.append(train_df[['PlayId',col]]                        .drop_duplicates(subset=['PlayId',col])                        ['PlayId'].value_counts()                        .value_counts(sort=False).index[-1])
    
uniq_cnt_df = pd.DataFrame({'columns':cols,                             'uniq_cnt_per_play': uniq_cnt})
plt.figure(figsize=(15,5))
ax = sns.barplot(x='columns', y="uniq_cnt_per_play",                  data=uniq_cnt_df)
plt.title('Unique value counts per play by columns',           fontsize=14)

for item in ax.get_xticklabels():
    item.set_rotation(90)


# ### keeping unique per play columns as is

# In[ ]:


uniq_cols = ['PossessionTeam', 'OffenseFormation', 'OffensePersonnel',             'DefensePersonnel', 'Stadium', 'Location', 'StadiumType',             'Turf', 'GameWeather', 'WindSpeed', 'WindDirection',              'DefenceTeam', 'OffenceisHome', 'HandoffSnapDiff',              'OffenceScoreBeforePlay', 'DefenceScoreBeforePlay',              'Dis', 'Season', 'YardLine_std', 'Quarter', 'Down',              'Distance', 'DefendersInTheBox', 'Temperature',              'Humidity', 'FieldPosition_std', 'PlayId']


# In[ ]:


X_train = train_df[uniq_cols+['Yards']].drop_duplicates(subset='PlayId').set_index('PlayId')
X_test = test_df[uniq_cols].drop_duplicates(subset='PlayId').set_index('PlayId')

y_train = X_train['Yards']
X_train.drop(['Yards'], axis=1, inplace=True)


# ### Taking average on numeric columns

# In[ ]:


agg_cols = ['X_std', 'Y_std', 'S', 'A', 'age',             'PlayerWeight', 'PlayerHeight']
X_train = X_train.join(train_df.groupby('PlayId')[agg_cols].mean())
X_test = X_test.join(test_df.groupby('PlayId')[agg_cols].mean())


# ### Creating features specific to rusher

# In[ ]:


def create_rusher_cols(df, X):
    
    Rusher_df = df[['PlayId', 'Position', 'X_std', 'Y_std', 'S',                     'A', 'PlayerCollegeName' ,'age', 'PlayerHeight',                    'Orientation_std', 'Dir']][df['NflId'] ==                                                df['NflIdRusher']]

    for col in Rusher_df.columns:
        if col != 'PlayId':
            Rusher_df.rename(columns={col: col + '_rusher'},                             inplace=True)

    Rusher_df.set_index('PlayId', inplace=True)
    X = X.join(Rusher_df)
    
    return X


# In[ ]:


X_train = create_rusher_cols(train_df, X_train)
X_test = create_rusher_cols(test_df, X_test)


# ### Calculating distances between rusher and defences

# In[ ]:


def create_dist_cols(df, X):
    
    dist_df = pd.DataFrame()
    for play in df.groupby('PlayId'):
        play_df = play[1]
        rusher = play_df[play_df.NflId == play_df.NflIdRusher]
        dist = play_df[play_df['OffenceDefence'] == 'defence']        .apply(lambda x: np.sqrt((x.X_std - rusher.X_std)**2                                  + (x.Y_std - rusher.Y_std)**2),                axis=1).values
        dist_df.loc[play[0], "AveDistToDef"] = dist.mean()
        dist_df.loc[play[0], "MinDistToDef"] = dist.min()
        dist.sort()
        dist_df.loc[play[0], "AveDistToNearest3Def"]         = np.mean(dist[:3])

    X = X.join(dist_df)

    return X


# In[ ]:


X_train = create_dist_cols(train_df, X_train)
X_test = create_dist_cols(test_df, X_test)


# # Correlation between Yards and numeric features

# In[ ]:


pd.options.display.float_format = '{:.4f}'.format
corr = np.abs(X_train.corrwith(y_train)).sort_values(ascending=False)
plt.figure(figsize=(8,15))
ax = sns.barplot(x=corr[:40], y=corr.index[:40])
plt.title('Correlation coefficient on Yards and columns', fontsize=14)


# ## Plotting Top 10 Correlated Features
# 
# The left pictures show Yards vs. each feature and the right pictures show train/test distributions on those features.

# In[ ]:


def plot_num_col(var):
    
    fig, ax = plt.subplots(1, 2, figsize=(12,5))
    
    sns.regplot(x=X_train[var], y=y_train, ax=ax[0], scatter_kws={'s':5})
    ax[0].set_title("%s vs Yards in Train-set" % var, size=14)
    
    sns.distplot(X_train[var].dropna(), label="train", ax=ax[1])
    sns.distplot(X_test[var].dropna(), label="test", ax=ax[1])
    if var in ylims:
        ax[1].set_ylim(0,ylims[var])
    fig.tight_layout()
    ax[1].legend(prop={'size': 12})
    ax[1].set_title("Train vs Test by %s Density Plot" % var, size=14)


# In[ ]:


ylims = {'A_rusher':0.6, 'AveDistToDef': 0.3,'DefendersInTheBox': 1, 'YardLine_std': 0.025,
       'X_std_rusher': 0.025, 'X_std': 0.025, 'S_rusher': 0.6, 'PlayerWeight':0.1, 'Distance':0.4, 'A':1.5, 'PlayerHeight':16}


# In[ ]:


for col in corr.index[:10]:
    plot_num_col(col)


# # One-way ANOVA on Categorical Features against Yards
# To check the correlation between categorical features and Yards, I'm calculating one-way ANOVA here. 

# In[ ]:


import statsmodels.api as sm
from statsmodels.formula.api import ols

f=[];p=[]

cat_cols = list()
for idx, col in enumerate(X_train.dtypes):
    if col == 'object':
        cat_cols.append(X_train.dtypes.index[idx])

for col in cat_cols:
    
    indat = pd.DataFrame({'Yards': y_train, 'cat_col': X_train[col]})
    model = ols('Yards ~ C(cat_col)', data=indat).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    f.append(anova_table.iloc[0, 2])
    p.append(anova_table.iloc[0, 3])


# In[ ]:


anova = pd.DataFrame({'column': cat_cols, 'F': f, 'p':p})
anova = anova.sort_values(by=['F'], ascending=False).reset_index(drop=True)

plt.figure(figsize=(8,8))
ax = sns.barplot(x=anova.F, y=anova.column)
plt.title('ANOVA F-values on Yars', fontsize=14)


# ### Plotting Most Significant 10 Features
# The left pictures show average Yards on each categories and the right pictures show train/test distributions on those features.

# In[ ]:


def plot_cat_col(var):
    
    fig, ax = plt.subplots(1, 2, figsize=(15,5))
    
    indat = pd.DataFrame({'Yards': y_train, '%s' % var: X_train[var]})
    means = indat.groupby(['%s' % var])['Yards'].mean()    .sort_values(ascending=False)
    if len(means) > 20:
        means = means.iloc[:20]
    sns.barplot(x=means.index, y=means, ax=ax[0])

    ax[0].set_title("Average Yards by %s" % var, size=14)
    for item in ax[0].get_xticklabels():
        item.set_rotation(90)
    
    var_dist_trn = X_train[var].value_counts()
    var_dist_trn = var_dist_trn / len(X_train)
    var_dist_tst = X_test[var].value_counts()
    var_dist_tst = var_dist_tst / len(X_test)

    df_join = var_dist_trn.to_frame().join(var_dist_tst.to_frame(),                                            lsuffix='_1', rsuffix='_2')    .reset_index()
    df_join.columns = [var, 'train', 'test']
    df_join = df_join[df_join[var].isin(means.index) ]
    df_join = pd.melt(df_join, id_vars=var, var_name="data",                       value_name="distribution")
    df_join.sort_values(by=["data","distribution"],                         ascending=[True,False], inplace=True)

    sns.barplot(x=var, y="distribution", hue="data",                 data=df_join, order= means.index, ax=ax[1])
    ax[1].legend(loc='upper right', prop={'size': 12})
    ax[1].set_title("Train vs Test  Distribution by %s" % var, size=14)
    for item in ax[1].get_xticklabels():
        item.set_rotation(90)


# In[ ]:


for col in anova['column'][:10]:
    plot_cat_col(col)


# That's it! Hope this notebook can be helpful for you.
