#!/usr/bin/env python
# coding: utf-8

# ## Heart Disease Diagnosis and Prediction

# <img src="https://cdn-images-1.medium.com/max/1600/1*CQXQxHDKi0Q2IpdjhufEcw.jpeg" style="height:300px;" />

# ### <b>Kernel Goals </b>:
# #### Data exploration and cleaning.
# #### Data modeling(Random Forest + LighGBM + XGBoost ).
# #### Feature importance. 
# #### Investigation some correlation.

# ### <b>Introduction </b>

# #### Dataset information 

# This database contains 76 attributes, but all published experiments refer to using a subset of 14 of them. In particular, the Cleveland database is the only one that has been used by ML researchers to this date. The "goal" field refers to the presence of heart disease in the patient. It is integer valued from 0 (no presence) to 4.
# 

# #### Attribute Information: 
# > 1. Age 
# > 2. Sex 
# > 3. Chest pain type (4 values) 
# > 4. Resting blood pressure 
# > 5. Serum cholestoral in mg/dl 
# > 6. Fasting blood sugar > 120 mg/dl
# > 7. Resting electrocardiographic results (values 0,1,2)
# > 8. Maximum heart rate achieved 
# > 9. Exercise induced angina 
# > 10. Oldpeak = ST depression induced by exercise relative to rest 
# > 11. The slope of the peak exercise ST segment 
# > 12. Number of major vessels (0-3) colored by flourosopy 
# > 13. Thal: 3 = normal; 6 = fixed defect; 7 = reversable defect

# ### <b>Contents</b>
# 
# 1. [Data Cleaning](#dc)
# 2. [Models](#md) <br>
#     2.1 [Random Forest](#md)<br>
#     2.2 [LightGBM](#lgm)<br>
#     2.3 [XGBoost](#xgb)
# 3. [Top feature causing heart disease](#tf)
# 4. [Feature correlation ](#fc)
# 5. [Investigating strongly correlated features](#ic)

# In[ ]:


get_ipython().system('pip install git+https://github.com/fastai/fastai@2e1ccb58121dc648751e2109fc0fbf6925aa8887 2>/dev/null 1>/dev/null')


# In[ ]:


from fastai.imports import *
from fastai.structured import *
from pandas_summary import DataFrameSummary
from sklearn.ensemble import RandomForestClassifier
from IPython.display import display
from sklearn import metrics
from sklearn.model_selection import train_test_split
import numpy as np 
import pandas as pd

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')
pd.options.mode.chained_assignment = None


# ## Data Cleaning <a class="anchor" id="dc"></a>

# In[ ]:


df = pd.read_csv("../input/heart.csv")


# Now we will first convert our normal attributes like age,sex etc into numerical form so that we can further analyse it.

# In[ ]:


df.columns = ['age', 'sex', 'chest_pain_type', 'resting_blood_pressure', 'cholesterol', 'fasting_blood_sugar', 'rest_ecg', 'max_heart_rate_achieved',
       'exercise_induced_angina', 'st_depression', 'st_slope', 'num_major_vessels', 'thalassemia', 'target']

df['sex'][df['sex'] == 0] = 'female'
df['sex'][df['sex'] == 1] = 'male'

df['chest_pain_type'][df['chest_pain_type'] == 1] = 'typical angina'
df['chest_pain_type'][df['chest_pain_type'] == 2] = 'atypical angina'
df['chest_pain_type'][df['chest_pain_type'] == 3] = 'non-anginal pain'
df['chest_pain_type'][df['chest_pain_type'] == 4] = 'asymptomatic'

df['fasting_blood_sugar'][df['fasting_blood_sugar'] == 0] = 'lower than 120mg/ml'
df['fasting_blood_sugar'][df['fasting_blood_sugar'] == 1] = 'greater than 120mg/ml'

df['rest_ecg'][df['rest_ecg'] == 0] = 'normal'
df['rest_ecg'][df['rest_ecg'] == 1] = 'ST-T wave abnormality'
df['rest_ecg'][df['rest_ecg'] == 2] = 'left ventricular hypertrophy'

df['exercise_induced_angina'][df['exercise_induced_angina'] == 0] = 'no'
df['exercise_induced_angina'][df['exercise_induced_angina'] == 1] = 'yes'

df['st_slope'][df['st_slope'] == 1] = 'upsloping'
df['st_slope'][df['st_slope'] == 2] = 'flat'
df['st_slope'][df['st_slope'] == 3] = 'downsloping'

df['thalassemia'][df['thalassemia'] == 1] = 'normal'
df['thalassemia'][df['thalassemia'] == 2] = 'fixed defect'
df['thalassemia'][df['thalassemia'] == 3] = 'reversable defect'


# In[ ]:


df.head()


# In[ ]:


def missing_data_ratio(df):
    all_data_na = (df.isnull().sum() / len(df)) * 100
    all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]
    missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
    return missing_data


# In[ ]:


import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import imp


# In[ ]:


import pandas_profiling


# In[ ]:


profile = pandas_profiling.ProfileReport(df)


# #### <b>Quick exploration with pandas profiling. </b>

# In[ ]:


missing_data_ratio(df)


# In[ ]:


profile


# We will now process the data to use algorithms on it.

# In[ ]:


df.columns


# In[ ]:


df.chest_pain_type = df.chest_pain_type.astype("category")
df.exercise_induced_angina = df.exercise_induced_angina.astype("category")
df.fasting_blood_sugar = df.fasting_blood_sugar.astype("category")
df.rest_ecg = df.rest_ecg.astype("category")
df.sex = df.sex.astype("category")
df.st_slope = df.st_slope.astype("category")
df.thalassemia = df.thalassemia.astype("category")


# In[ ]:


df = pd.get_dummies(df, drop_first=True)


# In[ ]:


df_p,y,_=proc_df(df,"target")


# In[ ]:


df_p.head()


# ## Models <a class="anchor" id="md"></a>

# #### <b>Random Forest</b>  <a class="anchor" id="rd"></a>

# **How does random forest work?**
# 
#     To classify a new object with attributes a tree is planted and a tree votes for a specific class and the forest chooses the class with most votes.

# In[ ]:


from sklearn.model_selection import RandomizedSearchCV


# In[ ]:


rf_param_grid = {
                 'max_depth' : [4, 6, 8,10],
                 'n_estimators': range(1,30),
                 'max_features': ['sqrt', 'auto', 'log2'],
                 'min_samples_split': [2, 3, 10,20],
                 'min_samples_leaf': [1, 3, 10,18],
                 'bootstrap': [True, False],
                 
                 }


# In[ ]:


m = RandomForestClassifier()


# In[ ]:


m_r = RandomizedSearchCV(param_distributions=rf_param_grid, 
                                    estimator = m, scoring = "accuracy", 
                                    verbose = 0, n_iter = 100, cv = 5)


# In[ ]:


get_ipython().run_line_magic('time', 'm_r.fit(df_p, y)')


# In[ ]:


m_r.best_score_


# In[ ]:


m_r.best_params_


# In[ ]:


rf_bp = m_r.best_params_


# In[ ]:


rf_classifier=RandomForestClassifier(n_estimators=rf_bp["n_estimators"],
                                     min_samples_split=rf_bp['min_samples_split'],
                                     min_samples_leaf=rf_bp['min_samples_leaf'],
                                     max_features=rf_bp['max_features'],
                                     max_depth=rf_bp['max_depth'],
                                     bootstrap=rf_bp['bootstrap'])


# In[ ]:


rf_classifier.fit(df_p,y)


# In[ ]:


fi = rf_feat_importance(rf_classifier,df_p)


# In[ ]:


def plot_fi(fi):
    return fi.plot('cols', 'imp', 'barh', figsize=(12,7), legend=False)


# #### <b>Random Forest Feature importance</b> 

# . It can perform both classification and regression.
# . Handles the missing values and maintains the accuracy for missing data.
# . It dosen't overfit the model.

# In[ ]:


plot_fi(fi)


# #### <b>LightGBM</b>  <a class="anchor" id="lgm"></a>

# LightGBM is a gradient boosting framework that uses tree based learning algorithms.

# In[ ]:


import lightgbm as lgbm


# In[ ]:


lgbm_model = lgbm.LGBMClassifier()


# In[ ]:


lgbm_params = {
    "n_estimators":[10,100,1000,2000],
    'boosting_type': ['dart','gbdt'],          
    'learning_rate': [0.05,0.1,0.2],       
    'min_split_gain': [0.0,0.1,0.5,0.7],     
    'min_child_weight': [0.001,0.003,0.01],     
    'num_leaves': [10,21,41,61],            
    'min_child_samples': [10,20,30,60,100]
              }


# In[ ]:


lgbm_model = lgbm.LGBMClassifier()


# In[ ]:


lgbm_c = RandomizedSearchCV(param_distributions=lgbm_params, 
                                    estimator = lgbm_model, scoring = "accuracy", 
                                    verbose = 0, n_iter = 100, cv = 4)


# In[ ]:


lgbm_c.fit(df_p,y)


# In[ ]:


lgbm_bp =lgbm_c.best_params_


# In[ ]:


lgbm_model = lgbm.LGBMClassifier(num_leaves=lgbm_bp["num_leaves"],
                                 n_estimators=lgbm_bp["n_estimators"],
                                 min_split_gain=lgbm_bp["min_split_gain"],
                                 min_child_weight=lgbm_bp["min_child_weight"],
                                 min_child_samples=lgbm_bp["min_child_samples"],
                                 learning_rate=lgbm_bp["learning_rate"],
                                 boosting_type=lgbm_bp["boosting_type"])


# In[ ]:


lgbm_model.fit(df_p,y)


# #### <b>LightGBM Feature importance </b>

# In[ ]:


def feature_imp(df,model):
    fi = pd.DataFrame()
    fi["feature"] = df.columns
    fi["importance"] = model.feature_importances_
    return fi.sort_values(by="importance", ascending=False)


# In[ ]:


feature_imp(df_p,lgbm_model).plot('feature', 'importance', 'barh', figsize=(12,7), legend=False)


# #### <b>Xgboost</b>  <a class="anchor" id="xgb"></a>

# What is eXtreme Gradient Boosting(Xgboost)?
# 
# 
# To understand xgboost we need to first understand Gradient boosting first. Gradient boosting is a machine learning technique for regression and classification problems, which produces a prediction model in the form of an ensemble of weak prediction models, typically decision trees.
# 
# The name xgboost, though, actually refers to the engineering goal to push the limit of computations resources for boosted tree algorithms. Which is the reason why many people use xgboost.

# In[ ]:


import xgboost as xgb


# In[ ]:


xgb_classifier = xgb.XGBClassifier()


# In[ ]:


gbm_param_grid = {
    'n_estimators': range(1,20),
    'max_depth': range(1, 10),
    'learning_rate': [.1,.4, .45, .5, .55, .6],
    'colsample_bytree': [.6, .7, .8, .9, 1],
    'booster':["gbtree"],
     'min_child_weight': [0.001,0.003,0.01],
}


# In[ ]:


xgb_random = RandomizedSearchCV(param_distributions=gbm_param_grid, 
                                    estimator = xgb_classifier, scoring = "accuracy", 
                                    verbose = 0, n_iter = 100, cv = 4)


# In[ ]:


xgb_random.fit(df_p,y)


# In[ ]:


xgb_bp = xgb_random.best_params_


# In[ ]:


xgb_model=xgb.XGBClassifier(n_estimators=xgb_bp["n_estimators"],
                            min_child_weight=xgb_bp["min_child_weight"],
                            max_depth=xgb_bp["max_depth"],
                            learning_rate=xgb_bp["learning_rate"],
                            colsample_bytree=xgb_bp["colsample_bytree"],
                            booster=xgb_bp["booster"])


# In[ ]:


xgb_model.fit(df_p,y)


# #### <b>XGboost Feature importance </b>

# In[ ]:


feature_imp(df_p,xgb_model).plot('feature', 'importance', 'barh', figsize=(12,7), legend=False)


# In[ ]:


from IPython.core.display import HTML

def multi_table(table_list):
    ''' Acceps a list of IpyTable objects and returns a table which contains each IpyTable in a cell
    '''
    return HTML(
        '<table><tr style="background-color:white;">' + 
        ''.join(['<td>' + table._repr_html_() + '</td>' for table in table_list]) +
        '</tr></table>'
    )


# In[ ]:


rf_fm = rf_feat_importance(rf_classifier,df_p)
lgbm_fm = feature_imp(df_p,lgbm_model)
xgb_fm = feature_imp(df_p,xgb_model)


# ### Top feature causing heart disease according to RF,LGBM,XGB <a class="anchor" id="tf"></a>

# In[ ]:


multi_table([rf_fm,lgbm_fm,xgb_fm])


# **Feature correlation**

# In[ ]:


import seaborn as sns
corr = df_p.corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
f, ax = plt.subplots(figsize=(11, 9))

sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values,mask=mask,cmap='summer_r',vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


# In[ ]:


from scipy.cluster import hierarchy as hc


# In[ ]:


def hierarchy_tree(df):
    corr = np.round(scipy.stats.spearmanr(df).correlation, 4)
    corr_condensed = hc.distance.squareform(1-df.corr())
    z = hc.linkage(corr_condensed, method='average')
    fig = plt.figure(figsize=(16,10))
    dendrogram = hc.dendrogram(z, labels=df.columns, orientation='left', leaf_font_size=16)
    plt.show()


# In[ ]:


hierarchy_tree(df_p)


# ##  Investigating strongly correlated features <a class="anchor" id="ic"></a>

# ### crosstabing correlated features

# In[ ]:


df_p["target"] = y  


# In[ ]:


df_p.columns


# In[ ]:


max_heart_rate_achieved = pd.cut(df_p.max_heart_rate_achieved,4,labels=["71-104","105-137","138-170","171-202"])


# In[ ]:


df_p.columns


# In[ ]:


cross1=pd.crosstab([df_p.st_slope_flat[df_p.st_slope_flat==1],df_p.target],max_heart_rate_achieved).style.background_gradient(cmap='summer_r')
cross1


# In[ ]:


cross1=pd.crosstab([df_p.st_slope_flat[df_p.st_slope_flat==1],df_p["thalassemia_fixed defect"][df_p["thalassemia_fixed defect"]==1],
                    df_p.target],max_heart_rate_achieved).style.background_gradient(cmap='summer_r')
cross1


# In[ ]:


cross1=pd.crosstab([df_p.st_slope_flat[df_p.st_slope_flat==1],df_p["thalassemia_fixed defect"][df_p["thalassemia_fixed defect"]==1],df_p["chest_pain_type_typical angina"][df_p["chest_pain_type_typical angina"]==1],
                    df_p.target],max_heart_rate_achieved).style.background_gradient(cmap='summer_r')
cross1


# ###### according to this data 93% of poeple that have st_slope_flat + thalassemia_fixed_defect + chest_pain_typical_argina have heart disease 

# In[ ]:


cross1=pd.crosstab([df_p["thalassemia_fixed defect"]],df_p.sex_male,margins=True).style.background_gradient(cmap='summer_r')
cross1


# In[ ]:


cross1=pd.crosstab([df_p["thalassemia_fixed defect"][df_p["thalassemia_fixed defect"]==1],df_p.target],df_p.sex_male).style.background_gradient(cmap='summer_r')
cross1


# In[ ]:


cross1=pd.crosstab([df_p.exercise_induced_angina_yes[df_p.exercise_induced_angina_yes==1],df_p.target],[df_p.st_slope_upsloping[df_p.st_slope_upsloping==1],df_p.st_depression]).style.background_gradient(cmap='summer_r')
cross1


# In[ ]:


age = pd.cut(df_p.age,6,labels=["(28.952, 37.0)", "(37.0, 45.0)", "(45.0, 53.0)","(53.0, 61.0)","(61.0, 69.0)", "(69.0, 77.0)"])


# In[ ]:


cross1=pd.crosstab([pd.cut(df_p.resting_blood_pressure,3),df_p.target],[age]).style.background_gradient(cmap='summer_r')
cross1


# #### <b> resting_blood_pressure is strong factory but only in early age </b>
