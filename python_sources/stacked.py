#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.svm import SVC,LinearSVC
from sklearn import preprocessing
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier, OutputCodeClassifier
import category_encoders as ce
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from scipy.stats import randint as sp_randint,uniform as sp_ranprop
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV,RFE,SelectFromModel
import mlxtend as mx
from mlxtend.classifier import  StackingCVClassifier
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis,RadiusNeighborsClassifier
from sklearn.naive_bayes import GaussianNB , BernoulliNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import KBinsDiscretizer
from tempfile import mkdtemp
from shutil import rmtree

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

def misclass(labels,results,wilderness):
    analysis = pd.DataFrame({'labels':labels,'results':results,'wilderness':wilderness})
    analysis['correct'] = (analysis.labels == analysis.results)+0
    print(pd.crosstab(margins=True,index=analysis.labels,columns=analysis_db.wilderness,values=analysis.correct,aggfunc=np.mean,dropna=False))
    print(pd.crosstab(margins=True,index=analysis.labels,columns=analysis_db.wilderness,values=analysis.correct,aggfunc=np.sum,dropna=False))
    
def examine(est,validation_set,targets,raw_targets=None,importance_flag=True):
    if (raw_targets is None):
        raw_targets=targets
    if (importance_flag):    
        low_importances = pd.DataFrame({'importances':est.feature_importances_,
                                     'names':validation_set.columns})
        print(low_importances.sort_values('importances'))
        
    scores =  est.predict_proba(validation_set.values)[:,1]
     
    spreads = pd.DataFrame({'target':targets,
                  'scores':scores,'raw_target': raw_targets})
     
    g = sns.FacetGrid( row='raw_target', data=spreads,sharey=False)
    g.map(sns.distplot, "scores")
    g.add_legend()
   
def examine_score(scores,targets,raw_targets=None):
    if (raw_targets is None):
        raw_targets=targets
      
    spreads = pd.DataFrame({'target':targets,
                  'scores':scores,'raw_target': raw_targets})
     
    g = sns.FacetGrid( row='raw_target', data=spreads,sharey=False)
    g.map(sns.distplot, "scores")
    g.add_legend()
   


# In[ ]:


import pandas_profiling


# In[ ]:


SHOWGRAPHS=False


# In[ ]:


#Input training data
training_db = pd.read_csv('/kaggle/input/learn-together/train.csv')


# In[ ]:


#training_db.profile_report()


# In[ ]:


def transform_db(db):
    new_db = db.drop(columns=['Soil_Type15','Soil_Type7']).eval(
        "calc_slope=Elevation/(Horizontal_Distance_To_Hydrology+0.01)").eval(
        "calc_slope2=Vertical_Distance_To_Hydrology/(Horizontal_Distance_To_Hydrology+0.01)").eval(
        "calc_slope3=Vertical_Distance_To_Hydrology/(Horizontal_Distance_To_Roadways+0.01)").eval(
         "sin_aspect=sin(Aspect/180*3.14156926)"
         ).eval("sin_slope=sin(Slope/180*3.14156926)").eval("h_ratio1=Hillshade_9am/(Hillshade_3pm+0.01)").eval(
            "hratio2=Hillshade_9am/(Hillshade_Noon+0.01)").eval(
            "hratio3=Hillshade_3pm/(Hillshade_Noon+0.01)").eval(
    "diff1 = Elevation - Vertical_Distance_To_Hydrology"
    ).eval(
    "diff2 = Horizontal_Distance_To_Roadways  - Horizontal_Distance_To_Hydrology"
    ).eval(
    "diff3 = Horizontal_Distance_To_Roadways  - Horizontal_Distance_To_Fire_Points"
    ).eval(
    "diff4 = Horizontal_Distance_To_Hydrology- Horizontal_Distance_To_Fire_Points"
    ).eval("sum1 = Elevation + Vertical_Distance_To_Hydrology"
    ).eval(
    "sum2 = Horizontal_Distance_To_Roadways  + Horizontal_Distance_To_Hydrology"
    ).eval(
    "sum3 = Horizontal_Distance_To_Roadways  + Horizontal_Distance_To_Fire_Points"
    ).eval(
    "sum = Horizontal_Distance_To_Hydrology+ Horizontal_Distance_To_Fire_Points"
    ).eval("rockoutcropcomplex= Soil_Type1 + Soil_Type3 + Soil_Type4 + Soil_Type5 + Soil_Type6 \
                  + Soil_Type11  + Soil_Type28 + Soil_Type33").eval(
        "rubbly = Soil_Type3 + Soil_Type4 + Soil_Type5 + Soil_Type10 + Soil_Type11 +\
         Soil_Type13").eval(
        "Vanet = Soil_Type2 +Soil_Type5 + Soil_Type6").eval("Bulwark = Soil_Type10 +Soil_Type11").eval(
        "Leighcan = Soil_Type21 + Soil_Type22 + Soil_Type23 + Soil_Type24 + Soil_Type25+\
           Soil_Type27 + Soil_Type28 + Soil_Type31 + Soil_Type32 + Soil_Type33+\
           Soil_Type38").eval(
        "ext_stony = Soil_Type1 + Soil_Type24 + Soil_Type25 + Soil_Type27 + Soil_Type28 +\
            Soil_Type29 + Soil_Type30 + Soil_Type31 + Soil_Type32 + Soil_Type33 +\
            Soil_Type34 + Soil_Type36 + Soil_Type37 + Soil_Type38 + Soil_Type39 +\
            Soil_Type40").eval(
           "very_stony = Soil_Type2 + Soil_Type9 + Soil_Type18").eval(
           "stony = Soil_Type6 + Soil_Type12").eval(
            "interact1 = Elevation*Horizontal_Distance_To_Hydrology"
    ).eval(
            "interact2 = Elevation*Vertical_Distance_To_Hydrology"
    ).eval(
            "interact3 = Vertical_Distance_To_Hydrology*Horizontal_Distance_To_Hydrology"
    ).eval(
     "Elevation2 = Elevation*Elevation"
    ).eval(
     "Vertical_Distance_To_Hydrology2 = Vertical_Distance_To_Hydrology*Vertical_Distance_To_Hydrology"
    ).eval(
     "Horizontal_Distance_To_Hydrology2 = Horizontal_Distance_To_Hydrology*Horizontal_Distance_To_Hydrology"
    ).eval("total_light = Hillshade_9am+Hillshade_3pm+Hillshade_Noon"
                                                               ).eval(
    "morning=Hillshade_9am/(total_light+0.01)"
    ).eval(
    "noon=Hillshade_Noon/(total_light+0.01)"
    ).eval(
    "afternoon=Hillshade_3pm/(total_light+0.01)"
    ).eval(
    "lightdiff1 = Hillshade_9am -Hillshade_3pm"
    ).eval(
    "lightdiff2 = Hillshade_9am -Hillshade_Noon"
    ).eval(
    "lightdiff3 = Hillshade_Noon -Hillshade_3pm"
    ).eval(
    "lightsum1 = Hillshade_9am +Hillshade_3pm"
    ).eval(
    "lightsum2 = Hillshade_9am +Hillshade_Noon"
    ).eval(
    "lightsum3 = Hillshade_Noon +Hillshade_3pm"
    ).eval(
    "North = ((Aspect >= 0) & (Aspect <=45) ) | ((Aspect>=315)&(Aspect<=360) )"
    ).eval(
    "East = ((Aspect > 45) & (Aspect <=135) )"
    ).eval(
    "South = ((Aspect > 135) & (Aspect <=225) )"
    ).eval(
    "West = ((Aspect > 225) & (Aspect <=315) )"
    ).eval(
    "SouthSlope = cos(South*Slope*3.141592653/180)"
    ).eval(
    "WestSlope = cos(West*Slope*3.141592653/180)"
    ).eval(
    "EastSlope = cos(East*Slope*3.141592653/180)"
    ).eval(
    "NorthSlope = cos(North*Slope*3.141592653/180)"
    )

    
    
    new_db["accumulate"]=0
    soils = new_db.columns.values[
                             new_db.columns.str.startswith("Soil_Type")]
    for i,name in enumerate(soils):
            new_db.accumulate = new_db.accumulate + new_db[name]*int(name.replace('Soil_Type',''))
        
      
    #return new_db.drop(columns=soils)
    return new_db

transformed_training_db = transform_db(training_db)


# In[ ]:


def analyse_db(training_db):
   analysis_db = training_db.eval("wilderness = Wilderness_Area1+Wilderness_Area2*2+Wilderness_Area3*4+Wilderness_Area4*8") 
   analysis_db["Cover_Type"] = "c"+analysis_db.Cover_Type.astype(str)
   analysis_db["wilderness"] = "w"+analysis_db.wilderness.astype(str)
   analysis_db["Grp_Flags"] = analysis_db.Cover_Type.replace(["c5","c7","c2","c1"],"High").replace(["c4","c3","c6"],"Low")
   analysis_db = pd.concat((analysis_db,pd.get_dummies(analysis_db.Cover_Type,prefix='',dtype=int)),axis=1)
   analysis_db["accumulate"]=0
   soils = analysis_db.columns.values[
                             analysis_db.columns.str.startswith("Soil_Type")]
   for i,name in enumerate(soils):
            analysis_db.accumulate = analysis_db.accumulate + analysis_db[name]*(2<<(i+1))

   return analysis_db.eval("total_light = Hillshade_9am+Hillshade_3pm+Hillshade_Noon"
                                                               ).eval(
    "morning=Hillshade_9am/(total_light+0.01)"
    ).eval(
    "noon=Hillshade_Noon/(total_light+0.01)"
    ).eval(
    "afternoon=Hillshade_3pm/(total_light+0.01)"
    ).eval(
    "lightdiff1 = Hillshade_9am -Hillshade_3pm"
    ).eval(
    "lightdiff2 = Hillshade_9am -Hillshade_Noon"
    ).eval(
    "lightdiff3 = Hillshade_Noon -Hillshade_3pm"
    ).eval(
    "lightsum1 = Hillshade_9am +Hillshade_3pm"
    ).eval(
    "lightsum2 = Hillshade_9am +Hillshade_Noon"
    ).eval(
    "lightsum3 = Hillshade_Noon +Hillshade_3pm"
    ).eval(
    "North = ((Aspect >= 0) & (Aspect <=45) ) | ((Aspect>=315)&(Aspect<=360) )"
    ).eval(
    "East = ((Aspect > 45) & (Aspect <=135) )"
    ).eval(
    "South = ((Aspect > 135) & (Aspect <=225) )"
    ).eval(
    "West = ((Aspect > 225) & (Aspect <=315) )"
    ).eval("direction = North + West*2 + South*3 +East*4").eval(
    "SouthSlope = cos(South*Slope*3.141592653/180)"
    ).eval(
    "WestSlope = cos(West*Slope*3.141592653/180)"
    ).eval(
    "EastSlope = cos(East*Slope*3.141592653/180)"
    ).eval(
    "NorthSlope = cos(North*Slope*3.141592653/180)"
    )

analysis_db = analyse_db(training_db)


# In[ ]:


pd.crosstab(index=analysis_db.Cover_Type,columns=analysis_db.wilderness)


# In[ ]:


pd.crosstab(index=analysis_db.Cover_Type,columns=analysis_db.direction)


# In[ ]:


if (SHOWGRAPHS):
 sns.boxplot(x="wilderness",y="Horizontal_Distance_To_Hydrology",data=analysis_db)


# In[ ]:


if (SHOWGRAPHS):
  sns.boxplot(x="wilderness",y="Vertical_Distance_To_Hydrology",data=analysis_db) 


# In[ ]:


if (SHOWGRAPHS):
    sns.boxplot(x="Cover_Type",y="Horizontal_Distance_To_Hydrology",data=analysis_db)


# In[ ]:


if (SHOWGRAPHS):
    sns.boxplot(x="Cover_Type",y="WestSlope",data=analysis_db)


# In[ ]:


if (SHOWGRAPHS):
    sns.boxplot(x="Cover_Type",y="EastSlope",data=analysis_db)


# In[ ]:


if (SHOWGRAPHS):
    sns.boxplot(x="Cover_Type",y="SouthSlope",data=analysis_db)


# In[ ]:


if (SHOWGRAPHS):
    sns.boxplot(x="Cover_Type",y="NorthSlope",data=analysis_db)


# In[ ]:


if (SHOWGRAPHS):
  sns.boxplot(x="Cover_Type",y="Vertical_Distance_To_Hydrology",data=analysis_db)


# In[ ]:


if (SHOWGRAPHS):
   sns.boxplot(x="Cover_Type",y="Slope",data=analysis_db)


# In[ ]:



if (SHOWGRAPHS):
    sns.boxplot(x="wilderness",y="Elevation",data=analysis_db)


# In[ ]:


if (SHOWGRAPHS):
  sns.boxplot(x="Cover_Type",y="Elevation",data=analysis_db)


# In[ ]:


if (SHOWGRAPHS):
   sns.scatterplot(x="Elevation",y="Vertical_Distance_To_Hydrology",hue="Cover_Type", 
                data=analysis_db)


# In[ ]:


if (SHOWGRAPHS):
      sns.violinplot(x="Cover_Type",y='Horizontal_Distance_To_Fire_Points',data=analysis_db)


# In[ ]:


if (SHOWGRAPHS):
    sns.violinplot(x="Cover_Type",y='Horizontal_Distance_To_Roadways',data=analysis_db)


# In[ ]:


if (SHOWGRAPHS):
 sns.violinplot(x="Cover_Type",y="total_light",data=analysis_db)


# In[ ]:


if (SHOWGRAPHS):
  sns.boxplot(x="Cover_Type",y="Aspect",data=analysis_db)


# In[ ]:


if (SHOWGRAPHS):
   sns.boxplot(x="Cover_Type",y="Slope",data=analysis_db)


# In[ ]:


if (SHOWGRAPHS):
  sns.violinplot(x="Cover_Type",y="morning",data=analysis_db)


# In[ ]:


if (SHOWGRAPHS):
  sns.boxplot(x="Cover_Type",y="noon",data=analysis_db)


# In[ ]:


if (SHOWGRAPHS):
  sns.boxplot(x="Cover_Type",y="afternoon",data=analysis_db)


# In[ ]:


if (SHOWGRAPHS):
  sns.violinplot(x="Cover_Type",y="lightdiff1",data=analysis_db)


# In[ ]:


if (SHOWGRAPHS):
  sns.boxplot(x="Cover_Type",y="lightdiff2",data=analysis_db)


# In[ ]:


if (SHOWGRAPHS):
  sns.boxplot(x="Cover_Type",y="lightdiff3",data=analysis_db)


# In[ ]:


if (SHOWGRAPHS):
 sns.boxplot(x="Cover_Type",y="lightsum1",data=analysis_db)


# In[ ]:


if (SHOWGRAPHS):
  sns.boxplot(x="Cover_Type",y="lightsum3",data=analysis_db)


# In[ ]:


if (SHOWGRAPHS):
 sns.boxplot(x="Cover_Type",y="lightsum2",data=analysis_db)


# In[ ]:


if (SHOWGRAPHS):
  sns.scatterplot(x="Elevation",y="Vertical_Distance_To_Hydrology",hue="Grp_Flags", 
                data=analysis_db)


# In[ ]:


if (SHOWGRAPHS):
  sns.scatterplot(x="Elevation",y="Horizontal_Distance_To_Hydrology",hue="Grp_Flags", 
                data=analysis_db)


# In[ ]:


def diffgraph(cover,xaxis,yaxis):
   g = sns.FacetGrid(col="wilderness",row="accumulate", data=analysis_db,hue=cover)
   g.map(sns.scatterplot, xaxis,yaxis)
   g.add_legend();


if (SHOWGRAPHS):
  diffgraph("_c5","Elevation","Vertical_Distance_To_Hydrology")
  diffgraph("_c5","Vertical_Distance_To_Hydrology","Horizontal_Distance_To_Hydrology") 


# So "w1" and w3 have no Low group
# W8 has no high 
# w4 (Wilderness_Area2) has mixed 

# In[ ]:


if (SHOWGRAPHS):
  g = sns.FacetGrid(col="wilderness", data=analysis_db.query("Grp_Flags=='High'"),sharex=False,col_wrap=2)
  g.map(sns.violinplot, "Cover_Type", "Elevation")
  g.add_legend();


# Is there a misclassification of something else as Cover type 2 

# In[ ]:


if (SHOWGRAPHS):
  g = sns.FacetGrid(col="wilderness", data=analysis_db.query("Grp_Flags=='Low'"),sharex=False,col_wrap=2)
  g.map(sns.violinplot, "Cover_Type", "Elevation")
  g.add_legend();


# In[ ]:


if (SHOWGRAPHS):
  g = sns.FacetGrid(col="wilderness", data=analysis_db,sharex=False,col_wrap=2)
  g.map(sns.violinplot, "Cover_Type", "Elevation")
  g.add_legend();


# In[ ]:


if (SHOWGRAPHS):
  sns.boxplot(x="Cover_Type",y="Elevation", data=analysis_db.query("Grp_Flags=='High'"))


# In[ ]:





# In[ ]:


if (SHOWGRAPHS):
  g = sns.FacetGrid(col="wilderness", data=analysis_db,sharex=False,col_wrap=2)
  g.map(sns.boxplot, "Cover_Type", "Aspect")
  g.add_legend();


# In[ ]:


if (SHOWGRAPHS):
 g = sns.FacetGrid(col="wilderness", data=analysis_db,sharex=False,col_wrap=2)
 g.map(sns.boxplot, "Cover_Type", "Hillshade_3pm")
 g.add_legend();


# In[ ]:


if (SHOWGRAPHS):
  g = sns.FacetGrid(col="wilderness", data=analysis_db,sharex=False,col_wrap=2)
  g.map(sns.boxplot, "Cover_Type", "Hillshade_9am")
  g.add_legend();


# In[ ]:


if (SHOWGRAPHS):
 g = sns.FacetGrid(col="wilderness", data=analysis_db,sharex=False,col_wrap=2)
 g.map(sns.boxplot, "Cover_Type", "Hillshade_Noon")
 g.add_legend();


# In[ ]:


if (SHOWGRAPHS):
  g = sns.FacetGrid(col="wilderness", data=analysis_db,sharex=False,col_wrap=2,hue="Cover_Type")
  g.map(sns.scatterplot, "Hillshade_3pm", "Hillshade_Noon")
  g.add_legend();


# In[ ]:


if (SHOWGRAPHS):
 g = sns.FacetGrid(col="wilderness", data=analysis_db,sharex=False,col_wrap=2,hue="Grp_Flags")
 g.map(sns.scatterplot, "Hillshade_3pm", "Hillshade_Noon")
 g.add_legend();


# In[ ]:


if (SHOWGRAPHS):
 g = sns.FacetGrid(col="wilderness", data=analysis_db,sharex=False,col_wrap=2,hue="Grp_Flags")
 g.map(sns.scatterplot, "Hillshade_3pm", "Hillshade_9am")
 g.add_legend();


# In[ ]:


if (SHOWGRAPHS):
 g = sns.FacetGrid(col="wilderness", data=analysis_db,sharex=False,col_wrap=2,hue="Grp_Flags")
 g.map(sns.scatterplot, "Aspect", "Slope")
 g.add_legend();


# Cover types "c5","c7","c2","c1" vs  "c3","c6","c4" is easy (Elevation, Distance to Hydrology)
# 
# Cover type  "c7" vs "c5","c2","c1" based on Elevation should be easy.(Elevation)
# 
# Then need to differentiate "c3","c6","c4" and "c5","c2","c1"
# 

# In[ ]:


transformed_training_db.accumulate.unique().shape


# In[ ]:





# # Split into training and validation

# In[ ]:



transformed_training_db.reset_index(inplace=True)
Ids_train,Ids_validate, y_train,y_validate = train_test_split(transformed_training_db['Id'],
                                                          transformed_training_db['Cover_Type'],
                                                          train_size=0.8,
                                                          random_state=42,
                                                         stratify=transformed_training_db['Cover_Type']
                                                         ) 

transformed_training_db.set_index('Id',inplace=True)
X_train=transformed_training_db.drop(columns=['Cover_Type','index'],errors='ignore').loc[Ids_train]
print(X_train.head())
X_validate=transformed_training_db.drop(columns=['Cover_Type','index'],errors='ignore').loc[Ids_validate]


# In[ ]:


print(y_train.value_counts(),y_validate.value_counts(),transformed_training_db['Cover_Type'].value_counts())


# # Soil Type

# In[ ]:


#For each cover type get list of soil types it is associated with

types = {}

weights = {}
for cover,grp in transformed_training_db.loc[Ids_train].groupby('Cover_Type'):
     types[cover] = [ 'Soil_Type'+str(soil_type) for soil_type in grp.accumulate.unique()]
     weights[cover] = grp[types[cover]].sum()/grp.shape[0]

    
def soil_types(df):
    for cover in types:   
        df["soil_for"+str(cover)]= (df[types[cover]]*weights[cover]).sum(axis=1)
        
    df["Low_soils"] = df[["soil_for1","soil_for2","soil_for7","soil_for5"]].sum(axis=1)
    df["High_soils"] = df[["soil_for4","soil_for3","soil_for6"]].sum(axis=1)
    
soil_types(X_train)
soil_types(X_validate)
soil_types(analysis_db)


# # Setup Stacking and Gridsearch

# In[ ]:



    
params ={

 'randomforestclassifier__min_weight_fraction_leaf': [0.0,0.1,0.2],
 'randomforestclassifier__n_estimators': sp_randint(100,4000),
 'extratreesclassifier__n_estimators':  sp_randint(100,4000),
 'pipeline-2__Features__estimator__n_estimators': sp_randint(50,400),
  'pipeline-2__Features__max_features': sp_randint(5,50),
 'pipeline-2__clf__colsample_bytree': sp_ranprop(),
 'pipeline-2__clf__learning_rate': [0.2,0.1,0.05,0.01],
 'pipeline-2__clf__max_depth': sp_randint(3,20),
 'pipeline-2__clf__n_estimators': sp_randint(100,4000),
 'pipeline-2__clf__reg_alpha': sp_randint(0,4),
 'pipeline-2__clf__reg_lambda': sp_randint(0,4),
 'pipeline-2__clf__subsample': sp_ranprop(),
 'meta_classifier__Features__estimator__n_estimators': sp_randint(100,4000),
 'meta_classifier__Features__max_features': sp_randint(3,60),
 'meta_classifier__clf__colsample_bytree': sp_ranprop(),
 'meta_classifier__clf__learning_rate': [0.2,0.1,0.05,0.01],
 'meta_classifier__clf__max_depth':sp_randint(3,20),
 'meta_classifier__clf__n_estimators': [1000,2000,4000,8000],
 'meta_classifier__clf__reg_alpha': sp_randint(0,4),
 'meta_classifier__clf__reg_lambda': sp_randint(0,4),
 'pipeline-1__Features__estimator__n_estimators': sp_randint(50,400),
 'pipeline-1__Features__max_features': sp_randint(2,7),    
 'pipeline-1__NN__n_neighbors': sp_randint(1,4)
}  


# In[ ]:


best_params = {'extratreesclassifier__n_estimators': 1250, 
               'meta_classifier__Features__estimator__n_estimators': 3981, 
               'meta_classifier__Features__max_features': 42, 
               'meta_classifier__clf__colsample_bytree': 0.4547010139728621, 
               'meta_classifier__clf__learning_rate': 0.01, 
               'meta_classifier__clf__max_depth': 19, 
               'meta_classifier__clf__n_estimators': 8000, 
               'meta_classifier__clf__reg_alpha': 1, 
               'meta_classifier__clf__reg_lambda': 0, 
               'pipeline-1__Features__estimator__n_estimators': 340, 
               'pipeline-1__Features__max_features': 2, 
               'pipeline-1__NN__n_neighbors': 2, 
               'pipeline-2__Features__estimator__n_estimators': 101, 
               'pipeline-2__Features__max_features': 27, 
               'pipeline-2__clf__colsample_bytree': 0.7363377647573023,
               'pipeline-2__clf__learning_rate': 0.2,
               'pipeline-2__clf__max_depth': 7,
               'pipeline-2__clf__n_estimators': 1409,
               'pipeline-2__clf__reg_alpha': 1, 
               'pipeline-2__clf__reg_lambda': 2,
               'pipeline-2__clf__subsample': 0.8865179551312595, 
               'randomforestclassifier__min_weight_fraction_leaf': 0.0, 
               'randomforestclassifier__n_estimators': 1627}


# In[ ]:


RANDOM_SEED = 42


clf1 = Pipeline([ 
                                        ('Features',
                                         SelectFromModel(RandomForestClassifier(random_state=RANDOM_SEED,
                                                                                #class_weight=class_weights,
                                                                                n_estimators=50,n_jobs=4)
                                    ,max_features=4,
                                                         threshold=-np.inf))
                                      ,('Std', StandardScaler())
                                      ,('metric',NeighborhoodComponentsAnalysis())
                                     ,('NN',KNeighborsClassifier(n_neighbors=3,n_jobs=4))
                                         ]#, memory=caches[1]
)

clf2 = RandomForestClassifier(random_state=47,n_estimators=3000,n_jobs=4,min_weight_fraction_leaf=0.1
                              #,class_weight=class_weights

 )


clf3 = ExtraTreesClassifier(random_state=47,n_estimators=1000,n_jobs=4
                            #,class_weight=class_weights

 )
                         
 

 



  
  
clf4 = Pipeline([ ('Features',SelectFromModel(RandomForestClassifier(random_state=RANDOM_SEED,
                                                                     #class_weight=class_weights,
                                                                     n_estimators=500,n_jobs=4)
                                    ,max_features=15,threshold=-np.inf)
                     ),
                                     
                       ('clf', xgb.XGBClassifier(n_jobs=4,n_estimators=2000
                                  ,learning_rate=0.2
                                ,max_depth=12
                                ,random_state=54                  
                                ,colsample_bynode= 0.67
                                 ,reg_alpha =  1
                                 ))
                    ]
                   #,memory=caches[4]
                                   )




  



gb = Pipeline([ ('Features',SelectFromModel(RandomForestClassifier(random_state=RANDOM_SEED,
                                                                 #  class_weight=class_weights,
                                                                   n_estimators=100,n_jobs=4)
                                   ,max_features=50,threshold=-np.inf)
                     ),
                                     
                       ('clf', 
                        xgb.XGBClassifier(n_jobs=4, colsample_bynode= 0.5465213014678215, 
                     colsample_bytree= 0.021671507502719622, 
                       learning_rate= 0.01, 
                       max_depth= 15, 
                        n_estimators= 4000, 
                        reg_alpha= 2, 
                       reg_lambda= 3,                             
                       random_state=65               
                                 )
                       )
                    ]
                  # ,memory=caches[0]
             )


stack_model = StackingCVClassifier(classifiers=[
                                  clf1
                                  ,clf2
                                 ,clf3
                                  ,clf4
                    
                            ]   ,
                            meta_classifier=gb,
                            use_probas=True,
                            use_features_in_secondary=True,      
                                cv=StratifiedKFold(2),
                            n_jobs=1,      
                            random_state=RANDOM_SEED)

stack_model = stack_model.set_params(**best_params)

                        


# In[ ]:


minimal_columns = X_train.columns[~X_train.columns.str.startswith("Soil_Type")]

#grid = RandomizedSearchCV(estimator=stack_model, 
#                   param_distributions=params, 
#                    cv=StratifiedKFold(2),n_iter=20,scoring='accuracy',
#                    refit='accuracy')

def std_output(model,X_train,y_train,X_validate):
    
    model.fit(X=X_train,y=y_train
              #,sample_weight=low_weights.loc[y_train_low].values
             )
    
    return           (model.predict_proba(X=X_validate) , 
                      model.predict(X=X_validate))



# # Fit model 

# In[ ]:



scores,predictions = std_output(stack_model,X_train[minimal_columns].values,y_train.values,X_validate[minimal_columns].values)
#misclass(y_validate,predictions,analyse_db(training_db.loc[Ids_validate]).wilderness.values)


# # Evaluate Results

# In[ ]:


for i in range(scores.shape[1]):
  examine_score(scores[:,i],y_validate)
  tpr,fpr,thresh = roc_curve(y_validate,scores[:,i])
  sns.lineplot(x=fpr,y=tpr)
  print(roc_auc_score(y_validate,scores[:,i]))
print(confusion_matrix(y_validate,predictions))


# In[ ]:





# # Score test data

# In[ ]:


test_db = pd.read_csv("/kaggle/input/learn-together/test.csv",index_col='Id')
prediction_db = transform_db(test_db)
soil_types(prediction_db)
#prediction_db = leave1out.transform(transform_db(test_db))
prediction_db.head()


# In[ ]:


predictions = stack_model.predict(prediction_db[minimal_columns])


# In[ ]:


ids = prediction_db.index.values


# In[ ]:


output_frame = pd.DataFrame({'Id':ids,'Cover_Type':predictions})


# In[ ]:


weights = output_frame.Cover_Type.value_counts()/output_frame.shape[0]
print(weights)


# In[ ]:


output_frame.head()


# In[ ]:


output_frame.to_csv('submission.csv',index=False)

