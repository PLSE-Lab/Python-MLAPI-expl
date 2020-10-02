#!/usr/bin/env python
# coding: utf-8

# # Strategy
# 
# While I could just build the build the best machine learning model possible, it doesn't give a way forward.
# My way forward is:
# * Understand the data
# * Identify high level groupings
# * Identify easy and problematic groups within the high level groupings
# * Work on separating the problematic groups

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
from sklearn.metrics import confusion_matrix,accuracy_score
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
from lightgbm import LGBMClassifier

import pandas_profiling

def misclass(labels,results,wilderness):
    analysis = pd.DataFrame({'labels':labels,'results':results,'wilderness':wilderness})
    analysis['correct'] = (analysis.labels == analysis.results)+0
    print(pd.crosstab(margins=True,index=analysis.labels,columns=analysis_db.wilderness,values=analysis.correct,aggfunc=np.mean,dropna=False))
    print(pd.crosstab(margins=True,index=analysis.labels,columns=analysis_db.wilderness,values=analysis.correct,aggfunc=np.sum,dropna=False))

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:





# In[ ]:


#Input training data
training_db = pd.read_csv('/kaggle/input/learn-together/train.csv')


# In[ ]:


#training_db.profile_report()


# # Feature generation

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


# # EDA

# In[ ]:


pd.crosstab(index=analysis_db.Cover_Type,columns=analysis_db.wilderness)


# In[ ]:


sns.boxplot(x="Cover_Type",y="Horizontal_Distance_To_Hydrology",data=analysis_db)


# In[ ]:


sns.boxplot(x="Cover_Type",y="Vertical_Distance_To_Hydrology",data=analysis_db)


# In[ ]:


sns.scatterplot(x="Elevation",y="Vertical_Distance_To_Hydrology",hue="Grp_Flags", 
                data=analysis_db)


# In[ ]:





# In[ ]:



g = sns.FacetGrid(col="wilderness", data=analysis_db,hue="Grp_Flags")
g.map(sns.scatterplot, "Elevation","Horizontal_Distance_To_Hydrology")
g.add_legend();


# In[ ]:


g = sns.FacetGrid(row="wilderness",col="Cover_Type" ,data=analysis_db)
g.map(sns.boxplot, "Elevation")
g.add_legend();


# So "w1" and w3 have no Low group
# W8 has no high 
# w4 (Wilderness_Area2) has mixed 

# Cover types "c5","c7","c2","c1" vs  "c3","c6","c4" is easy (Elevation, Distance to Hydrology)
# 
# Cover type  "c7" vs "c5","c2","c1" based on Elevation should be easy.(Elevation)
# 
# Then need to differentiate "c3","c6","c4" and "c5","c2","c1"
# 

# # Split into training and validation

# In[ ]:



#transformed_training_db.reset_index(inplace=True)
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


# # Soil weights

# In[ ]:



# A little biased but I don't think nitpicking will help
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


# # Differentiate High and Low groups

# In[ ]:


#Just for Wilderness Area 3
W2_train=X_train.Wilderness_Area3.values==1
W2_validate=X_validate.Wilderness_Area3.values==1
#high_low_model = LogisticRegressionCV(penalty='l1',random_state=13,solver="liblinear",cv=5,max_iter=400)
#high_low_model = SVC(probability=True,gamma=1.0e-12)
#high_low_model = LogisticRegression()
high_low_model = xgb.XGBClassifier(n_estimators=900,n_jobs=4,learning_rate=0.1,colsample_bytree=0.4,max_depth=5)
y_high_low_train=y_train.replace([5,7,2,1],"High").replace([4,3,6],"Low").loc[W2_train]
y_high_low_validate=y_validate.replace([5,7,2,1],"High").replace([4,3,6],"Low").loc[W2_validate]
high_low_predictors = ["Elevation","Vertical_Distance_To_Hydrology","Horizontal_Distance_To_Hydrology",
                       "High_soils",
                       "Low_soils",
                       "soil_for1",
                       "soil_for2",
                       "soil_for3",
                       "soil_for4",
                       "soil_for5",
                       "soil_for6",
                       "soil_for7",
                       "diff1",
                       "diff2",
                       "diff3",
                       "diff4"
                       #,"Elevation2"
                       #,"interact2"
                       #,"Vertical_Distance_To_Hydrology2"
                       #,"Horizontal_Distance_To_Hydrology2"
                       #,"interact1"
                       #,"interact3"
                      ]
high_low_model.fit(X_train.loc[W2_train,high_low_predictors].values,y_high_low_train)
high_low_model.score(X_validate.loc[W2_validate,high_low_predictors].values,y_high_low_validate)
high_low_scores= high_low_model.predict_proba(X_validate.loc[W2_validate,high_low_predictors].values)[:,1]
tpr,fpr,thresh = roc_curve(y_high_low_validate,
                           high_low_scores,
                           pos_label="High") 
plt.plot(fpr,tpr)
print(roc_auc_score(y_high_low_validate,
                           high_low_scores))


# In[ ]:


spreads = pd.DataFrame({'target':y_validate.loc[W2_validate].values,
'scores':high_low_scores})
spreads["grpd_score"]=pd.cut(spreads.scores,10)
g = sns.FacetGrid(col="target", data=spreads,sharey=False,col_wrap=2)
g.map(sns.distplot, "scores")
g.add_legend();


# # Within "High" Group distinguish Krumholz from everything else

# In[ ]:


#krummholz_model = LogisticRegression()
krummholz_model = xgb.XGBClassifier(n_estimators=1400,max_depth=20,n_jobs=4,learning_rate=0.07)

krummholz_train = y_train.isin([5,7,2,1]).values
krummholz_validate = y_validate.isin([5,7,2,1]).values
y_krummholz_train=  y_train[krummholz_train] ==7 #y_train ==7
y_krummholz_validate=  y_validate[krummholz_validate] == 7 #y_validate == 7

predictors_krummholz=['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology', 
            'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways', 
            'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm', 
            'Horizontal_Distance_To_Fire_Points', 'Wilderness_Area1', 
            'Wilderness_Area2', 'Wilderness_Area3', "soil_for1",
                       "soil_for2",
                       "soil_for3",
                       "soil_for4",
                       "soil_for5",
                       "soil_for6",
                       "soil_for7",
                        'h_ratio1', 'hratio2', 'hratio3', 
            'diff1', 'diff2', 'diff3', 'diff4', 'sum1', 'sum2', 'sum3', 'sum', 
            'rockoutcropcomplex', 'rubbly', 'Bulwark', 'Leighcan', 'ext_stony', 'stony', 'accumulate',
            'Wilderness_Area1',
            'Wilderness_Area2',
            'Wilderness_Area3',
            'Wilderness_Area4'
           ]
krummholz_model.fit(X_train.loc[krummholz_train,
                                predictors_krummholz].values,y_krummholz_train.values)
krummholz_model.score(X_validate.loc[krummholz_validate,
                                     predictors_krummholz].values,
                      y_krummholz_validate)

scores =  krummholz_model.predict_proba(X_validate.loc[krummholz_validate,
                                                    predictors_krummholz].values)[:,1]
tpr,fpr,thresh = roc_curve(y_krummholz_validate,
                          scores,
                           pos_label=False
                          ) 
plt.plot(fpr,tpr)

print(roc_auc_score(y_krummholz_validate,
                    scores
     ))



spreads = pd.DataFrame({'target':y_krummholz_validate,
'scores':scores,'raw_target': y_validate[krummholz_validate]})
spreads["grpd_score"]=pd.cut(spreads.scores,10)
g = sns.FacetGrid( row='raw_target', data=spreads,sharey=False)
g.map(sns.distplot, "scores")
g.add_legend();


# # Differentiate between 4,3,6 within Low

# In[ ]:


Low_predictors = ['Elevation', 'Aspect', 'Slope', 
                  'Horizontal_Distance_To_Hydrology', 
                  'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways', 
                  'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points', 'calc_slope', 'calc_slope2', 'calc_slope3', 'sin_aspect', 'h_ratio1', 
                  'hratio2', 'hratio3', 'diff1', 'diff2', 'diff3', 'diff4', 'sum1', 'sum2', 'sum3', 'sum', 
                  'rockoutcropcomplex', 'rubbly', 'Vanet', 'Bulwark', 'Leighcan', 'ext_stony', 'very_stony', 'stony',
                   'interact1', 'interact2', 'interact3',
      'Vertical_Distance_To_Hydrology2',
       'total_light', 'morning',
       'noon', 'afternoon', 'lightdiff1', 'lightdiff2', 'lightdiff3',
       'lightsum1', 'lightsum2', 'lightsum3', 'North',  
        'SouthSlope', 'WestSlope', 'EastSlope', 'NorthSlope',
                  "soil_for1",
                       "soil_for2",
                       "soil_for3",
                       "soil_for4",
                       "soil_for5",
                       "soil_for6",
                       "soil_for7",
                  
                       
                 #,'Wilderness_Area4'
                  #, 'Wilderness_Area3'
                 ]


# In[ ]:


#Set up datasets 
Low_train_indices = y_train.isin([4,3,6]).values
Low_validate_indices = y_validate.isin([4,3,6]).values

#Low_predictors = X_train.columns[~(X_train.columns.str.startswith("Soil") | (X_train.columns == "accumulate"))]

X_train_low = X_train.loc[Low_train_indices,Low_predictors]
X_validate_low = X_validate.loc[Low_validate_indices,Low_predictors]
y_train_low = y_train.loc[Low_train_indices]
y_validate_low = y_validate.loc[Low_validate_indices]

#Low_model = OneVsOneClassifier(xgb.XGBClassifier(n_estimators=1200,max_depth=10,n_jobs=4,learning_rate=0.05))

Low_model = xgb.XGBClassifier(n_estimators=4000,max_depth=5,n_jobs=4,learning_rate=0.1)
Low_model = LGBMClassifier(n_estimators=2000,max_depth=5,n_jobs=4,learning_rate=0.1)
#low_weights = pd.DataFrame({'weight':[10000,1,1000]},index=pd.Index([3,4,6]))
Low_model.fit(X_train_low.values,y_train_low.values
              #,sample_weight=low_weights.loc[y_train_low].values
             )
predictions = Low_model.predict(X_validate_low.values)

misclass(y_validate_low,predictions,analyse_db(training_db.loc[Ids_validate].loc[Low_validate_indices]).wilderness.values)
print(confusion_matrix(y_validate_low,predictions))


# In[ ]:



scores = Low_model.predict_proba(X_validate_low.values)
for i in range(scores.shape[1]):
  spreads = pd.DataFrame({
  'scores':scores[:,i],'raw_target':y_validate_low  })
  g = sns.FacetGrid( row='raw_target', data=spreads,sharey=False)
  g.map(sns.distplot, "scores")
  g.add_legend();


# In[ ]:


low_importances = pd.DataFrame({'importances':Low_model.feature_importances_,
                             'names':X_train_low.columns})
low_importances.sort_values('importances')


# In[ ]:


print(low_importances.names[low_importances.importances>0.001].tolist())


# # Differentiate between 5 and pines (1,2) within High 

# In[ ]:


#Set up datasets 
#high_columns=X_train.columns[~X_train.columns.str.startswith('Soil_Type')]
high_columns=['Elevation', 'Aspect', 'Slope',
              'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology', 
              'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon', 
              'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points', 'Wilderness_Area1', 
              'Wilderness_Area3', 'calc_slope', 'calc_slope2', 'calc_slope3', 'sin_aspect', 
              'h_ratio1', 'hratio2', 'hratio3', 'diff1', 'diff2', 'diff3', 'diff4',
              'sum1', 'sum2', 'sum3', 'sum', 'rockoutcropcomplex', 'rubbly',
              'Vanet', 'Bulwark', 'Leighcan', 'ext_stony', 'very_stony', 
              'interact1', 'interact2', 'interact3', 'Vertical_Distance_To_Hydrology2',
              'total_light', 'morning', 'noon', 'afternoon', 'lightdiff1', 'lightdiff2',
              'lightdiff3', 'lightsum1', 'lightsum2', 'lightsum3', 'North', 'SouthSlope', 
              'WestSlope', 'EastSlope', 'NorthSlope', 'accumulate', 'soil_for1', 'soil_for2', 
              'soil_for3', 'soil_for4', 'soil_for5', 'soil_for6', 'soil_for7', 
              'Low_soils', 'High_soils']
High_train_indices = y_train.isin([1,2,5]).values
High_validate_indices = y_validate.isin([1,5,2]).values
X_train_high = X_train.loc[High_train_indices,high_columns]
X_validate_high = X_validate.loc[High_validate_indices,high_columns]
y_train_high = y_train.loc[High_train_indices].replace([2],1)
y_validate_high = y_validate.loc[High_validate_indices].replace([2],1)
#high_weights = pd.DataFrame({'weight':[1000,1000,1]},index=pd.Index([1,2,5]))
high_weights = {1:1000}#,5:1}

High_model = LGBMClassifier(n_estimators=2000,max_depth=10,n_jobs=4,
                                                  learning_rate=0.1
                            #,class_weight=high_weights
                            #objective='multiclass'
                           )


High_model.fit(X_train_high.values,y_train_high.values,
               #sample_weight=high_weights.loc[y_train_high].values
              )
predictions = High_model.predict(X_validate_high.values)


misclass(y_validate_high,predictions,analyse_db(training_db.loc[Ids_validate].loc[High_validate_indices]).wilderness.values)


# In[ ]:


scores = High_model.predict_proba(X_validate_high.values)
for i in range(scores.shape[1]):
  spreads = pd.DataFrame({
  'scores':scores[:,i],'raw_target':y_validate_high  })
  g = sns.FacetGrid( row='raw_target', data=spreads,sharey=False)
  g.map(sns.distplot, "scores")
  g.add_legend();


# In[ ]:


tpr,fpr,thresh = roc_curve(y_validate_high,
                           scores[:,0],
                           pos_label=5
                          ) 
plt.plot(fpr,tpr)
print(roc_auc_score(y_validate_high,
                           scores[:,1],
     ))


# In[ ]:


high_importances = pd.DataFrame({'importances':High_model.feature_importances_,
                             'names':X_train_high.columns})
high_importances.sort_values('importances')


# In[ ]:


print(high_importances.names[high_importances.importances>0].to_list())


# # Pines - 1 vs 2

# In[ ]:


keep_preds=['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',
            'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways', 
            'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points', 
            'Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3', 
            'calc_slope', 'calc_slope2', 'calc_slope3', 'sin_aspect', 'h_ratio1', 'hratio2', 'hratio3', 'diff1', 'diff2', 'diff3', 'diff4', 'sum1', 'sum2', 'sum3', 'sum', 'Leighcan', 'ext_stony', 'interact1', 'interact2', 'interact3', 'Vertical_Distance_To_Hydrology2', 'total_light', 'morning', 'noon', 'afternoon', 'lightdiff1', 'lightdiff2', 'lightdiff3', 'lightsum1', 'lightsum2', 'lightsum3', 'SouthSlope', 'WestSlope', 'EastSlope', 'NorthSlope', 'accumulate', 'soil_for1', 'soil_for2', 'soil_for3', 'soil_for5', 'soil_for6', 'soil_for7', 'Low_soils', 'High_soils']


# In[ ]:


#Set up datasets 
pine_train_indices = (y_train.isin([1,2]).values ) & (X_train.Wilderness_Area4 == 0 ).values
pine_validate_indices = (y_validate.isin([1,2]).values) & (X_validate.Wilderness_Area4 == 0 ).values
#Cover_Type2 in Wilderness Area 4 looks wrong
X_train_pine = X_train.loc[pine_train_indices]
X_validate_pine = X_validate.loc[pine_validate_indices]
y_train_pine = y_train.loc[pine_train_indices]
y_validate_pine = y_validate.loc[pine_validate_indices]

#pine_model = xgb.XGBClassifier(n_estimators=4000,max_depth=12,n_jobs=4,learning_rate=0.01
#                               )
pine_model = LGBMClassifier(n_estimators=4000,max_depth=5,n_jobs=4,learning_rate=0.1
                               )

#High_model = OneVsOneClassifier(RandomForestClassifier(n_estimators=1600,n_jobs=4,oob_score=True))
#predictors = ["diff1","stony","Hillshade_Noon","Leighcan","Wilderness_Area2",
#              "Wilderness_Area1","Wilderness_Area3","Wilderness_Area4",
#             "Bulwark","Horizontal_Distance_To_Roadways"
#             ] + X_train_pine.columns[X_train.columns.str.startswith("Soil")].values.tolist()
#pine_predictors = X_train.columns[~X_train.columns.str.startswith('Soil_Type')]
pine_predictors=keep_preds
#predictors = X_train_pine.columns.values
pine_model.fit(X_train_pine[pine_predictors].values,y_train_pine.values)
predictions = pine_model.predict(X_validate_pine[pine_predictors].values)



# In[ ]:




misclass(y_validate_pine,predictions,analyse_db(training_db.loc[Ids_validate].loc[pine_validate_indices]).wilderness.values)
print(confusion_matrix(y_validate_pine,predictions))
scores =  pine_model.predict_proba(X_validate_pine[#.loc[krummholz_validate,
                                                    pine_predictors].values)[:,0]

tpr,fpr,thresh = roc_curve(y_validate_pine,
                          scores,
                           pos_label=2
                          ) 
plt.plot(fpr,tpr)

print(roc_auc_score(y_validate_pine,
                    1-scores
     ))


# In[ ]:


importances = pd.DataFrame({'importances':pine_model.feature_importances_,
                             'names':pine_predictors})
importances.sort_values('importances')


# In[ ]:


keep_preds = importances.names[importances.importances >30]


# In[ ]:


print(keep_preds.to_list())


# In[ ]:


print(X_validate.eval("wilderness= Wilderness_Area1 + 2*Wilderness_Area2 +3*Wilderness_Area3 + 4*Wilderness_Area4",inplace=True))


# # Manual adjustments

# In[ ]:


def  score_dset(df,high_low_cutoff=0.45,krummholz_cutoff=0.98,cutoff_5=0.98):
    results = df.copy()
    results["HighvsLow"] = 'High'
    results.loc[(df.Wilderness_Area4==1).values,"HighvsLow"] = "Low"
    high_low_flag = high_low_model.predict_proba(df[high_low_predictors].values)[:,0]> high_low_cutoff
    print(high_low_flag.shape)
    results.loc[(df.Wilderness_Area3==1).values & high_low_flag] = "High" 
    results["Krummholz"] = krummholz_model.predict_proba(df[predictors_krummholz].values)[:,1]
    results.loc[(df.Wilderness_Area4==1).values,"Krummholz"] =0.0
    results["Low"] = Low_model.predict(df[X_train_low.columns].values)
    results["High"] = High_model.predict_proba(df[X_train_high.columns].values)[:,1]
    results["Pine"] = pine_model.predict(df[pine_predictors].values)
    results["final_result"] = results.Low
    
    results.loc[(results.HighvsLow =='High') & (results.Krummholz>=krummholz_cutoff),"final_result"] = 7
    #results.loc[(results.HighvsLow =='High') & (results.Krummholz>=0.5),"final_result"] = results.High
    #results.loc[(results.HighvsLow =='Low'),"final_result"] = results.Low
    
    results.loc[(results.HighvsLow =='High') & (results.Krummholz<krummholz_cutoff) & (results.High > cutoff_5) ,
                "final_result"] = 5
    results.loc[(results.HighvsLow =='High')  & (results.Krummholz<krummholz_cutoff) & (results.High <= cutoff_5) 
                ,"final_result"] = results.Pine
  
    return results


result = score_dset(X_validate)


misclass(y_validate.values,result.final_result.values,X_validate.wilderness)
class_weights=np.array([0,0.5,0.4,0.1,0.0,0.1,0.1,0.0,0.0])
print(class_weights[y_validate.values])
max_score=0 
for kc in np.linspace(0.9,0.98,5):
    for hlc in np.linspace(0.3,0.7,8):
        for c5 in np.linspace(0.9,0.98,8):        
            result = score_dset(X_validate,krummholz_cutoff=kc,high_low_cutoff=hlc,cutoff_5=c5)
            print(kc,hlc,c5)
            acc=accuracy_score(y_validate.values,result.final_result.values,sample_weight=class_weights[y_validate.values])
            if (acc >max_score):
                max_score=acc
                max_state={'krummholz_cutoff':kc,'high_low_cutoff':hlc,'cutoff_5':c5}
                print(acc,max_state)


# # Generate submission

# In[ ]:


test_db = pd.read_csv("/kaggle/input/learn-together/test.csv",index_col='Id')
prediction_db = transform_db(test_db)
soil_types(prediction_db)
#prediction_db = leave1out.transform(transform_db(test_db))
prediction_db.head()


# In[ ]:


result_test = score_dset(prediction_db[X_validate.drop(columns=['wilderness']).columns],**max_state)


# In[ ]:


predictions = result_test.final_result


# In[ ]:


ids = prediction_db.index.values


# In[ ]:


output_frame = pd.DataFrame({'Id':ids,'Cover_Type':predictions})


# In[ ]:


weights = output_frame.Cover_Type.value_counts()/output_frame.shape[0]
print(weights)


# In[ ]:


pd.crosstab(index=analysis_test.Cover_Type,columns=analysis_test.wilderness)


# In[ ]:


output_frame.head()


# In[ ]:


output_frame.to_csv('submission.csv',index=False)

