#!/usr/bin/env python
# coding: utf-8

# # Road Safety EDA
# Exploratory data analysis and modelling of 2015 road traffic accident data provided
# publicly by the UK government. The data provides 
# detailed road safety information about the circumstances of personal injury 
# road accidents. The statistics relate only to personal injury accidents on public roads 
# that are reported to the police, and subsequently recorded.
# 
# This notebook is comprised of data preprocessing, exploration and visualisation, modelling and evaluation.
# 
# The objective is to build a model to predict whether a police officer attended an accident scene.
# 
# ## Import packages

# In[ ]:


from matplotlib import pyplot as plt
from sklearn import preprocessing, model_selection, metrics, ensemble
import seaborn as sns
import pandas as pd
import numpy as np
import datetime as dt
import xgboost
pd.options.mode.chained_assignment = None  # default='warn'


# In[ ]:


with open('../input/Accidents0515.csv', 'r') as f:
    df = pd.read_csv(f,encoding='utf-8')


# ## Preprocessing

# In[ ]:


df['Date']=pd.to_datetime(df['Date'], format='%d/%m/%Y')
df['Time']=pd.to_datetime(df['Time'], format='%H:%M')
#could engineer hour of day


# In[ ]:


print('Police responded: '+str(float(len(df[df['Did_Police_Officer_Attend_Scene_of_Accident']==1]))/float(len(df))*100)[:5]+'%')


# Moderately unbalanced datasets, perhaps use average precision score or AUROC as evaluation metrics

# In[ ]:


#GBM requires y labels to start at 0
df['Did_Police_Officer_Attend_Scene_of_Accident']=df['Did_Police_Officer_Attend_Scene_of_Accident'].replace(2,0)


# In[ ]:


#drop na
df=df.dropna()

#encode categorical variables as integers for GBM
le=preprocessing.LabelEncoder()
lizt=df.select_dtypes([object]).columns.values
for i in lizt:
    df[i]=le.fit_transform(df[i])


# ## EDA

# In[ ]:


total = df.isnull().sum().sort_values(ascending=False)
percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(5)
#some missing LSOA data before encoding


# ### Location
# Accidents are largely in densely populated roads/areas. No clear separation of police arrival on inspection

# In[ ]:


plt.figure(figsize=(10,15))
plt.scatter(df['Longitude'],df['Latitude'],c=df['Did_Police_Officer_Attend_Scene_of_Accident'])
plt.ylim((49.8,61))
plt.xlim((-8.5,2))
plt.show()


# ### Casualties

# In[ ]:


plt.figure(figsize=(13, 6))
sns.boxplot(y='Did_Police_Officer_Attend_Scene_of_Accident', x='Number_of_Casualties', data=df, orient='h')
plt.title('Casualties versus attendance')
plt.xlabel('Number of Casualties')
plt.show()


# ### Number of Vehicles

# In[ ]:


f, ax = plt.subplots(figsize=(12, 8))
ax = sns.violinplot(x="Did_Police_Officer_Attend_Scene_of_Accident", y="Number_of_Vehicles",data=df, palette="muted")


# Incidents with high number of vehicles involved is likely to have police attendance
# ### Feature Correlations

# In[ ]:


sns.set(style="white")

corrmat = df.corr()

mask = np.zeros_like(corrmat, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(12, 9))

cmap = sns.diverging_palette(220,10, as_cmap=True)

sns.heatmap(corrmat, vmax=.8, square=True, linewidths=.5, cbar_kws={"shrink": .5},mask=mask, cmap=cmap, ax=ax)
plt.title("Correlation Matrix")
plt.show()


# ### Scatterplot

# In[ ]:


cols=['Number_of_Vehicles','1st_Road_Number','Accident_Severity']
sns.pairplot(df,vars=cols,hue='Did_Police_Officer_Attend_Scene_of_Accident', hue_order=[1,0], size = 2.5)


# Difficult to draw conclusions on predictive features due to many overlapping values but number of vehicles and accident severity features seem to have good separation of data points for y

# # Modelling
# Bayesian tuned XGBoost
# 
# Reasoning:
# * GBMs generally perform well with contrained datasets and on mixed categorical and numerical data. 
# * Don't require one-hot encoding on categorical data which would increase the sparsity of the data. 
# * Not really enough data/time available for a neural network
# 
# Firstly build train and test sets:

# In[ ]:


X=df.drop(['Did_Police_Officer_Attend_Scene_of_Accident','Accident_Index','Time','Date'], axis=1) #drop
y=df['Did_Police_Officer_Attend_Scene_of_Accident']
X_train, X_test, y_train, y_test=model_selection.train_test_split(X,y, test_size=0.20)


# Check data has been formatted correctly:

# In[ ]:


print(X_train.dtypes)


# In[ ]:


#xg_params=xgb_bayesopt.res['max']['max_params']
#xg_params['max_depth']=int(xg_params['max_depth'])
#xg_params['n_estimators']=int(xg_params['n_estimators'])

num_rounds=3000
xg_params={'colsample_bytree': 0.85179589789548693, 'learning_rate': 0.028386065836928077, 'min_child_weight': 9.5803134857929955, 'n_estimators': 472, 'subsample': 0.79923622162397667, 'max_depth': 5, 'gamma': 0.23443198383493641}

xg_clf=xgboost.XGBClassifier(num_boost_round=num_rounds)
xg_clf.set_params(**xg_params)
xg_clf.fit(X_train, y_train)
xg_preds=xg_clf.predict_proba(X_test)
xg_preds=[x[1] for x in xg_preds]


# ## Evaluation
# Function for measuring performance of each modelling approach:

# In[ ]:


def metric(preds):
    auc=metrics.roc_auc_score(y_test.clip(0,1), preds)
    aps=metrics.average_precision_score(y_test.clip(0,1), preds)
    print('preds AUC: '+ str(auc))
    print('preds APS: '+ str(aps))
    print('\n')
    predictions=pd.DataFrame({'true':y_test.clip(0,1),'pred':[int(round(x)) for x in preds], 'prob':preds})
    print(predictions.sort_values('prob', ascending=False).head())
    
    cnf_matrix=metrics.confusion_matrix(y_test.clip(0,1), [int(round(x)) for x in preds], labels=None, sample_weight=None)
    cnf_matrix=pd.DataFrame(cnf_matrix, columns=['0 Pred','1 Pred']).rename(index={0: '0 True', 1: '1 True'})
    ax = sns.heatmap(cnf_matrix, annot=True)
    
    fpr, tpr, thresholds = metrics.roc_curve(y_test.clip(0,1), preds) #roc_curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange')


# ### XGBoost Output

# In[ ]:


Ometric(xg_preds)


# Oops.
# 
# ### Feature Importances

# In[ ]:


xgboost.plot_importance(xg_clf, max_num_features=15)


# # Conclusion
# Something went wrong, probably that I forgot to clip the truth values to 0,1 before training, or that the hyperparameters were tuned in a different notebook on only 2015 data from this dataset. Will continue to investigate. Please upvote if you enjoyed.

# In[ ]:




