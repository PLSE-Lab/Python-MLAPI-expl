#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import lightgbm as lgb
from tqdm import tqdm_notebook
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold,GroupKFold,KFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
import shap
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


from IPython.display import Image
Image("/kaggle/input/nflanalyticsimages/breakawaybanner.JPG")


# 

# In[ ]:


import numpy as np
import pandas as pd
from IPython.display import HTML, Image
import warnings
import plotly
import plotly.express as px

pd.set_option("display.max_columns", 100)
th_props = [('font-size', '13px'), ('background-color', 'white'), 
            ('color', '#666666')]
td_props = [('font-size', '15px'), ('background-color', 'white')]
styles = [dict(selector="td", props=td_props), dict(selector="th", 
            props=th_props)]


# In[ ]:


play = pd.read_csv('../input/nfl-playing-surface-analytics/PlayList.csv')


# In[ ]:


oneplayer = pd.read_parquet('/kaggle/input/breakaway-cleats-loading-track-data/oneplayer.parq')


# Replace Weather conditions with 2 factors - wet & dry.  Group players into blockers (OL/DL) and ball handlers (RB, WR, TE, LB, CB, S)

# In[ ]:


Weatherdict = {"10% Chance of Rain": "Dry",
               "30% Chance of Rain": "Dry",
              "Clear":"Dry",
"Clear Skies":"Dry",
"Clear and Cool":"Dry",
"Clear and Sunny":"Dry",
"Clear and cold":"Dry",
"Clear and sunny":"Dry",
"Clear and warm":"Dry",
"Clear skies":"Dry",
"Clear to Partly Cloudy":"Dry",
"Cloudy":"Dry",
"Cloudy and Cool":"Dry",
"Cloudy and cold":"Dry",
"Cloudy with periods of rain, thunder possible. Winds shifting to WNW, 10-20 mph.":"Dry",
"Cloudy, 50% change of rain":"Dry",
"Cloudy, Rain":"Wet",
"Cloudy, chance of rain":"Dry",
"Cloudy, fog started developing in 2nd quarter":"Dry",
"Cloudy, light snow accumulating 1-3":"Dry",
"Cold":"Dry",
"Controlled Climate":"Dry",
"Coudy":"Dry",
"Fair":"Dry",
"Hazy":"Dry",
"Heat Index 95":"Dry",
"Heavy lake effect snow":"Wet",
"Indoor":"Dry",
"Indoors":"Dry",
"Light Rain":"Wet",
"Mostly Cloudy":"Dry",
"Mostly Coudy":"Dry",
"Mostly Sunny":"Dry",
"Mostly Sunny Skies":"Dry",
"Mostly cloudy":"Dry",
"Mostly sunny":"Dry",
"N/A (Indoors)":"Dry",
"N/A Indoor":"Dry",
"Overcast":"Dry",
"Partly Cloudy":"Dry",
"Partly Clouidy":"Dry",
"Partly Sunny":"Dry",
"Partly clear":"Dry",
"Partly cloudy":"Dry",
"Partly sunny":"Dry",
"Party Cloudy":"Dry",
"Rain":"Wet",
"Rain Chance 40%":"Dry",
"Rain likely, temps in low 40s.":"Wet",
"Rain shower":"Wet",
"Rainy":"Wet",
"Scattered Showers":"Wet",
"Showers":"Wet",
"Snow":"Wet",
"Sun & clouds":"Dry",
"Sunny":"Dry",
"Sunny Skies":"Dry",
"Sunny and clear":"Dry",
"Sunny and cold":"Dry",
"Sunny and warm":"Dry",
"Sunny, Windy":"Dry",
"Sunny, highs to upper 80s":"Dry",
"cloudy":"Dry",
}
Rosterdict1 = {"Cornerback":"Fast",
"Defensive Lineman":"Slow",
"Linebacker":"Medium",
"Offensive Lineman":"Slow",
"Running Back":"Fast",
"Safety":"Medium",
"Tight End":"Medium",
"Wide Receiver":"Fast"}
Rosterdict2 = {
"Cornerback":"Defense",
"Defensive Lineman":"Defense",
"Linebacker":"Defense",
"Offensive Lineman":"Offense",
"Running Back":"Offense",
"Safety":"Defense",
"Tight End":"Offense",
"Wide Receiver":"Offense"}
Rosterdict3={
"Cornerback":"Defender",
"Defensive Lineman":"Blocker",
"Linebacker":"Defender",
"Offensive Lineman":"Blocker",
"Running Back":"Catcher",
"Safety":"Defender",
"Tight End":"Catcher",
"Wide Receiver":"Catcher"}
Rosterdict4={
"Cornerback":"Ball",
"Defensive Lineman":"Blocker",
"Linebacker":"Ball",
"Offensive Lineman":"Blocker",
"Running Back":"Ball",
"Safety":"Ball",
"Tight End":"Ball",
"Wide Receiver":"Ball"}
Bodydict={
"Ankle":"Joint",
"Foot":"Feet",
"Heel":"Feet",
"Knee":"Joint",
"Toes":"Feet"}

RosterEncode1 = {"Fast":2,
"Medium":1,
"Slow": 0}

RosterEncode = {"Ball":0,
"Blocker":1}

TurfEncode = {'Synthetic':1,'Natural':0}
WeatherEncode = {"Dry":0,"Wet":1}


play['Dry']=play['Weather']
play = play.replace({"Dry": Weatherdict})

play['Dry'] = np.where(play['Dry'].isin(["Dry",'Wet']),play['Dry'],'Wet')
play['Roster']=play['RosterPosition']
play = play.replace({"Roster": Rosterdict4})
play['Roster1']=play['RosterPosition']
play = play.replace({"Roster2": Rosterdict1})




play['IsDry'] = play['Dry']
play['Turf'] = play['FieldType']


play = play.replace({'Roster':RosterEncode})
play = play.replace({'Roster1':RosterEncode1})
play = play.replace({'IsDry':WeatherEncode})
play = play.replace({'Turf':TurfEncode})

play = play[play.Roster.isin([0,1])].copy()

play['IsPass'] = np.where(play['PlayType'].isin(['Rush','Pass']),play['PlayType'],2)
play['IsPass'] = np.where(play['PlayType']=='Pass',0,play.IsPass)
play['IsPass'] = np.where(play['PlayType']=='Rush',1,play.IsPass)

play['Dry']=play['Weather']
play = play.replace({"Dry": Weatherdict})

play['Dry'] = np.where(play['Dry'].isin(["Dry",'Wet']),play['Dry'],'Wet')
play['Roster']=play['RosterPosition']
play = play.replace({"Roster": Rosterdict4})
play['Roster1']=play['RosterPosition']
play = play.replace({"Roster2": Rosterdict1})

play['IsDry'] = play['Dry']
play['Turf'] = play['FieldType']

play['RosterType'] = play.Roster
play = play.replace({'Roster':RosterEncode})
play = play.replace({'Roster1':RosterEncode1})
play = play.replace({'IsDry':WeatherEncode})
play = play.replace({'Turf':TurfEncode})

play = play[play.Roster.isin([0,1])].copy()

play['IsPass'] = np.where(play['PlayType'].isin(['Rush','Pass']),play['PlayType'],2)
play['IsPass'] = np.where(play['PlayType']=='Pass',0,play.IsPass)
play['IsPass'] = np.where(play['PlayType']=='Rush',1,play.IsPass)

play['PlayKey'] = play.PlayKey.fillna('0-0-0')
id_array = play.PlayKey.str.split('-', expand=True).to_numpy()
play['PlayerKey'] = id_array[:,0].astype(int)
play['GameID'] = id_array[:,1].astype(int)
play['PlayKey'] = id_array[:,2].astype(int)


# In[ ]:


Play_trk=play.merge(oneplayer,on=['PlayerKey','GameID','PlayKey'])
Play_trk['PlayerKeyGroup'] =np.round(Play_trk['PlayerKey'].astype(int)/100).fillna(0)
Play_trk=Play_trk.drop(columns=['index'])


# In[ ]:


X = Play_trk.replace([np.inf,-np.inf],np.nan).fillna(0).copy()    
X.drop([
     'Turf','StadiumType' ,'Temperature'  ,'GameID','PlayKey','PlayerGame','PlayerDay','PlayerKeyGroup'
       ], axis=1, inplace=True)


# In[ ]:


cat_features=['PlayerKey']
for col in cat_features:
    X[col]= X[col].astype('category')

features = list(X.select_dtypes(include=[np.number]).columns.values)
X[features] = X[features].fillna(0)
scaler = StandardScaler()
#train_val[features] = scaler.fit_transform(train_val[features])
X[features] = scaler.fit_transform(X[features])
train_df = X.copy()
features = features+cat_features


# In[ ]:


features = ['IsDry', 'IsPass',  'Delta_Dis_2', 'Delta_Dir_2', 'Delta_Angle_2', 'Delta_O_2', 'Delta_Total_Acc_2', 'abs_Delta_Dis_2', 'abs_Delta_Dir_2', 'abs_Delta_Angle_2', 'abs_Delta_O_2', 'abs_Delta_Total_Acc_2', 'Delta_Dis_3', 'Delta_Dir_3', 'Delta_Angle_3', 'Delta_O_3', 'Delta_Total_Acc_3', 'abs_Delta_Dis_3', 'abs_Delta_Dir_3', 'abs_Delta_Angle_3', 'abs_Delta_O_3', 'abs_Delta_Total_Acc_3', 'Delta_Dis_4', 'Delta_Dir_4', 'Delta_Angle_4', 'Delta_O_4', 'Delta_Total_Acc_4', 'abs_Delta_Dis_4', 'abs_Delta_Dir_4', 'abs_Delta_Angle_4', 'abs_Delta_O_4', 'abs_Delta_Total_Acc_4', 'Delta_Dis_5', 'Delta_Dir_5', 'Delta_Angle_5', 'Delta_O_5', 'Delta_Total_Acc_5', 'abs_Delta_Dis_5', 'abs_Delta_Dir_5', 'abs_Delta_Angle_5', 'abs_Delta_O_5', 'abs_Delta_Total_Acc_5', 'Delta_Dis_6', 'Delta_Dir_6', 'Delta_Angle_6', 'Delta_O_6', 'Delta_Total_Acc_6', 'abs_Delta_Dis_6', 'abs_Delta_Dir_6', 'abs_Delta_Angle_6', 'abs_Delta_O_6', 'abs_Delta_Total_Acc_6', 'Delta_Dis_7', 'Delta_Dir_7', 'Delta_Angle_7', 'Delta_O_7', 'Delta_Total_Acc_7', 'abs_Delta_Dis_7', 'abs_Delta_Dir_7', 'abs_Delta_Angle_7', 'abs_Delta_O_7', 'abs_Delta_Total_Acc_7', 'Delta_Dis_8', 'Delta_Dir_8', 'Delta_Angle_8', 'Delta_O_8', 'Delta_Total_Acc_8', 'abs_Delta_Dis_8', 'abs_Delta_Dir_8', 'abs_Delta_Angle_8', 'abs_Delta_O_8', 'abs_Delta_Total_Acc_8', 'Delta_Dis_9', 'Delta_Dir_9', 'Delta_Angle_9', 'Delta_O_9', 'Delta_Total_Acc_9', 'abs_Delta_Dis_9', 'abs_Delta_Dir_9', 'abs_Delta_Angle_9', 'abs_Delta_O_9', 'abs_Delta_Total_Acc_9', 'Rolling_5_Dis_std', 'Rolling_5_Dir_std', 'Rolling_5_Angle_std', 'Rolling_5_O_std', 'Rolling_5_Total_Acc_std', 'Rolling_abs_5_Dis_mean', 'Rolling_abs_5_Dir_mean', 'Rolling_abs_5_Angle_mean', 'Rolling_abs_5_O_mean', 'Rolling_abs_5_Total_Acc_mean', 'Rolling_10_Dis_std', 'Rolling_10_Dir_std', 'Rolling_10_Angle_std', 'Rolling_10_O_std', 'Rolling_10_Total_Acc_std', 'Rolling_abs_10_Dis_mean', 'Rolling_abs_10_Dir_mean', 'Rolling_abs_10_Angle_mean', 'Rolling_abs_10_O_mean', 'Rolling_abs_10_Total_Acc_mean', 'PlayerKey']


# In[ ]:



target = Play_trk['Turf']


params = {'objective': 'binary',
 'metric': 'binary_error',
 'boosting_type': 'gbdt',
 'learning_rate': 0.1,
 'lambda_l1': 9.934222857123041,
 'lambda_l2': 0.9573358221807761,
 'num_leaves': 42,
 'feature_fraction': 0.705808436693177,
 'bagging_fraction': 0.76018681204692047,
 'bagging_freq': 2,
 'min_child_samples': 74,
 'random_state': 42,
 'verbose': -1}





folds = GroupKFold(n_splits=5)

oof = np.zeros(len(train_df))
predictions = np.zeros(len(train_df))
feature_importance_df = pd.DataFrame()
predict_df = pd.DataFrame()
models = []
for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_df.values, target.values,Play_trk['PlayerGamePlay'])):
    print("Fold {}".format(fold_))
    trn_data = lgb.Dataset(train_df.iloc[trn_idx][features], label=target.iloc[trn_idx])
    val_data = lgb.Dataset(train_df.iloc[val_idx][features], label=target.iloc[val_idx])

    num_round = 300
    clf = lgb.train(params, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=10, early_stopping_rounds = 50)
    oof[val_idx] = clf.predict(train_df.iloc[val_idx][features], num_iteration=clf.best_iteration)
    
    fold_importance_df = pd.DataFrame()
    fold_importance_df["Feature"] = features
    fold_importance_df["importance"] = clf.feature_importance()
   
    fold_importance_df["fold"] = fold_ + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    models.append(clf)

    
    
    #predictions += clf.predict(train_df[features], num_iteration=clf.best_iteration) / folds.n_splits
    #predict_fold = pd.DataFrame(clf.predict(train_df[features], num_iteration=clf.best_iteration))
    #predict_fold['Fold']= fold_ + 1
    #predict_fold['PlayId'] = Play_trk.PlayKey #combine with time
    #predict_df = pd.concat([predict_df,predict_fold])


# In[ ]:



feature_importance_df[["Feature", "importance"]].groupby("Feature").mean().sort_values(by="importance", ascending=False).reset_index().to_csv('FeatureImportance.csv')


# In[ ]:


## With all Features
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
df1 = pd.DataFrame(oof)
df1['target'] = target

fpr, tpr, _ = roc_curve(df1.target, df1[0])
roc_auc = auc(fpr, tpr)


plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.savefig('roc.png')


# In[ ]:


explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(train_df[features])
shap.summary_plot(shap_values, train_df[features], show=False)
plt.savefig('shap.png')


# In[ ]:




