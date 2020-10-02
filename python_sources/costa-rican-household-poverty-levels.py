#!/usr/bin/env python
# coding: utf-8

# ## Costa Rican Household Poverty Level Prediction

# ### Overview

# The data for this project is taken from a Kaggle competition for poverty levels prediction of Costa Rican housholds, hosted by the The Inter-American Development Bank: <br>
# https://www.kaggle.com/c/costa-rican-household-poverty-prediction
# <br>
# <br>
# The training data conatains 9558 rows, each row represting a person. There are 143 columns, including the person ID, its household identifer and the target columns (our labels). Some of the features refer to the person and some are aggregated per houshold. 
# <br>
# For example: 
# Age, is male/female if owns a tablet and number of years of education refer to person. <br>
# Number people under 12, rent, the matirial the house is made of and the region -  refer to a household. <br>
# <br>
# The prediction should be on the houshold level, i.e. all the persons under the same household should have the same poverty level.
# <br>
# The poverty level are:<br>
# 1 = extreme poverty <br>
# 2 = moderate poverty <br>
# 3 = vulnerable households <br>
# 4 = non vulnerable households <br>
# 

# ### Exploration and Preprocessing

# In[ ]:


import pandas as pd
import matplotlib
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import train_test_split as split
from sklearn.metrics import confusion_matrix, SCORERS, classification_report, accuracy_score, recall_score, f1_score, roc_auc_score
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import warnings
import time


# In[ ]:


warnings.simplefilter('ignore')


# In[ ]:


data=pd.read_csv('../input/train.csv')
data.head()


# In[ ]:


data.columns


# In[ ]:


data.shape


# As mentioned, the data should be aggregated to houshold level, since the prediction should be per household.<br>
# So we explored each column to understand the data range and understand how to aggregate.

# In[ ]:


val_exsplore=(data.agg(['min','max','dtype',lambda df: df.nunique(),pd.unique])
                  .transpose()
                  .rename(index=str, columns={'<lambda>':'nunique'})
             )


# In[ ]:


pd.set_option('display.max_colwidth', 100)
val_exsplore.sort_values(by=['nunique'],ascending = False)


# In[ ]:


# Check for NaN values
nan_sum=pd.DataFrame(data.isnull().sum())
nan_sum[nan_sum[0]>0]


# After consideration we decided to replace nan values with 0. 

# In[ ]:


# replace nan values with 0
data_clean = data.copy()
data_clean.fillna(value=0,inplace=True)


# We note some inconsistencies with houshole lables. <br>
# Some households have more than one lable. In those cases the lable of the head of the household label for the houshold

# In[ ]:


house_labels=pd.DataFrame(data_clean
                                    .groupby(['idhogar'])
                                    .Target
                                    .nunique()
                         )
house_labels[house_labels['Target']>1].size


# We will find the labels of the heads of houshold:

# In[ ]:


data_clean[data_clean['parentesco1']==1][['idhogar','Target']]


# In[ ]:


#size of targets of heads of households
data_clean[data_clean['parentesco1']==1][['idhogar','Target']].shape[0]


# In[ ]:


#number of unique housholds
data_clean['idhogar'].nunique()


# In[ ]:


data_clean['idhogar'].nunique()-data_clean[data_clean['parentesco1']==1][['idhogar','Target']].shape[0]


# So there are 15 households with no head of household in the data. We will finde their lables and concatenate to what we have

# In[ ]:


data_household_ishead=pd.pivot_table(data_clean, index='idhogar',aggfunc = sum, values = ['parentesco1'])
data_household_nohead=data_household_ishead[data_household_ishead['parentesco1']==0].index


# In[ ]:


missing_labels=data_clean[data_clean['idhogar'].isin(data_household_nohead)][['idhogar','Target']].drop_duplicates()
all_house_labels=pd.concat([data_clean[data_clean['parentesco1']==1][['idhogar','Target']],missing_labels], axis=0)
all_house_labels.rename(index=str, columns={'Target': 'Target_new'}, inplace = True)
all_house_labels.head()


# In[ ]:


house_labels_new=pd.DataFrame(all_house_labels
                                    .groupby(['idhogar'])
                                    .Target_new
                                    .nunique()
                         )
house_labels_new[house_labels_new['Target_new']>1].size


# Join the new target labels into the data_clean table

# In[ ]:


data_clean = pd.merge(data_clean,all_house_labels, on='idhogar')


# In[ ]:


data_clean.shape


# How the target lables are split in the data:

# In[ ]:


labels = (pd
            .DataFrame(data_clean
                            .Target_new
                            .value_counts()
                      )
            .sort_index()
         )
labels['Target_all_%']=((labels['Target_new']/(labels['Target_new'] .sum()))
                                                            .round(3)
                       )
labels


# Since we need to lable in the household label, we look at the target labels groupd by houshold

# In[ ]:


labels['house_target']=all_house_labels['Target_new'].value_counts().sort_index()
labels['house_target_%']=(labels['house_target']/labels['house_target'].sum()).round(3)
#  .plot.bar())
labels


# In[ ]:


names = pd.Series(['extreme', 'moderate', 'vulnerable', 'non vulnerable'])
labels.set_index(names,inplace=True)
labels['house_target'].plot(kind='bar')


# In[ ]:


(pd
     .concat([labels,pd.DataFrame(labels.sum()).transpose()], axis=0)
     .rename(index={0: 'Total'})
)


# We see that the label ratio is similar on the person and on the household level.

# We note issues in calculating dependency. The dependency in the data was not according to the promised calculations, so we calculated our own dependecy columns, based on the age data of the members of each housholds.

# In[ ]:


def is_adult(s):
    if (s<=64) & (s>=19):
        return 1
    return 0
    
def is_minor(s):
    if s<19:
        return 1
    return 0
    
def is_senior(s):
    if s>64:
        return 1
    return 0
    
data['is_adult']=data['age'].apply(is_adult)
data['is_minor']=data['age'].apply(is_minor)
data['is_senior']=data['age'].apply(is_senior)


# In[ ]:


agg_ages=pd.pivot_table(data, index='idhogar', values = ['is_minor','is_adult','is_senior'], aggfunc = sum) 
agg_ages['all_ages']=agg_ages[['is_minor','is_adult','is_senior']].sum(axis=1)
agg_ages.head(10)


# In[ ]:


agg_ages['dependency_our']=agg_ages[['is_minor','is_senior']].sum(axis=1)/agg_ages['is_adult']
# agg_ages.head(20)

agg_ages['dependency_our'].replace(np.inf, 10, inplace=True)


# In[ ]:


agg_ages.head()


# Drop columns that don't contibute to the household data

# In[ ]:


(data_clean.drop(
                    axis=1, 
                    columns=['Id','hogar_nin','hogar_adul','hogar_mayor',
                        'hogar_total','dependency','qmobilephone','age','agesq','Target'],
                    inplace =True
                )
)


# In[ ]:


# replace string values with yes with 1 and no with 0
data_clean.edjefe=data_clean.edjefe.replace(['yes'], 1, inplace=True)
data_clean.edjefe=data_clean.edjefe.replace(['no'], 0, inplace=True)
data_clean.edjefa=data_clean.edjefa.replace(['yes'], 1, inplace=True)
data_clean.edjefa=data_clean.edjefa.replace(['no'], 0, inplace=True)


# Aggregate the data by houshold

# In[ ]:


data_household_max=pd.pivot_table(data_clean, index='idhogar',aggfunc = max, values = ['v2a1','hacdor','rooms','hacapo',
                    'v14a',
                    'refrig',
                    'v18q',
                    'v18q1',
                    'r4h1',
                    'r4h2',
                    'r4h3',
                    'r4m1',
                    'r4m2',
                    'r4m3',
                    'r4t1',
                    'r4t2',
                    'r4t3',
                    'tamhog',
                    'tamviv',
                    'hhsize',
                    'paredblolad',
                    'paredzocalo',
                    'paredpreb',
                    'pareddes',
                    'paredmad',
                    'paredzinc',
                    'paredfibras',
                    'paredother',
                    'pisomoscer',
                    'pisocemento',
                    'pisoother',
                    'pisonatur',
                    'pisonotiene',
                    'pisomadera',
                    'techozinc',
                    'techoentrepiso',
                    'techocane',
                    'techootro',
                    'cielorazo',
                    'abastaguadentro',
                    'abastaguafuera',
                    'abastaguano',
                    'public',
                    'planpri',
                    'noelec',
                    'coopele',
                    'sanitario1',
                    'sanitario2',
                    'sanitario3',
                    'sanitario5',
                    'sanitario6',
                    'energcocinar1',
                    'energcocinar2',
                    'energcocinar3',
                    'energcocinar4',
                    'elimbasu1',
                    'elimbasu2',
                    'elimbasu3',
                    'elimbasu4',
                    'elimbasu5',
                    'elimbasu6',
                    'epared1',
                    'epared2',
                    'epared3',
                    'etecho1',
                    'etecho2',
                    'etecho3',
                    'eviv1',
                    'eviv2',
                    'eviv3',
                    'dis',
                    'male',
                    'female',
                    'estadocivil1',
                    'estadocivil2',
                    'estadocivil3',
                    'estadocivil4',
                    'estadocivil5',
                    'estadocivil6',
                    'estadocivil7',
                    'parentesco1',
                    'parentesco2',
                    'parentesco3',
                    'parentesco4',
                    'parentesco5',
                    'parentesco6',
                    'parentesco7',
                    'parentesco8',
                    'parentesco9',
                    'parentesco10',
                    'parentesco11',
                    'parentesco12',
                    'edjefe',
                    'edjefa',
                    'meaneduc',
                    'bedrooms',
                    'overcrowding',
                    'tipovivi1',
                    'tipovivi2',
                    'tipovivi3',
                    'tipovivi4',
                    'tipovivi5',
                    'computer',
                    'television',
                    'mobilephone',
                    'lugar1',
                    'lugar2',
                    'lugar3',
                    'lugar4',
                    'lugar5',
                    'lugar6',
                    'area1',
                    'area2',
                    'SQBescolari',
                    'SQBage',
                    'SQBhogar_total',
                    'SQBedjefe',
                    'SQBhogar_nin',
                    'SQBovercrowding',
                    'SQBdependency',
                    'SQBmeaned','Target_new']
                                 )


# In[ ]:


data_household_sum=pd.pivot_table(data_clean, index='idhogar',aggfunc = sum, values = ['escolari',
                                                                                        'rez_esc',
                                                                                        'instlevel1',
                                                                                        'instlevel2',
                                                                                        'instlevel3',
                                                                                        'instlevel4',
                                                                                        'instlevel5',
                                                                                        'instlevel6',
                                                                                        'instlevel7',
                                                                                        'instlevel8',
                                                                                        'instlevel9']
                                 )


# In[ ]:


data_houshold=(data_household_max
                                .join(data_household_sum)
                                .join(agg_ages)
              )
data_houshold.head()


# In[ ]:


data_houshold = pd.concat([data_houshold,(pd.DataFrame(agg_ages['dependency_our']))], axis=1, sort=False)


# In[ ]:


data_houshold.shape


# In[ ]:


feature_names = ['v18q','mobilephone','refrig','computer','television']


# In[ ]:


for feature in feature_names:
    data_houshold.groupby(['Target_new',feature]).size().unstack().plot(kind='bar', stacked=True)


# In[ ]:


region = data_houshold[['lugar1','lugar2','lugar3','lugar4','lugar5','lugar6','Target_new']].copy()
region['lugar2'] = region['lugar2'].replace(1,2)
region['lugar3'] = region['lugar3'].replace(1,3)
region['lugar4'] = region['lugar4'].replace(1,4)
region['lugar5'] = region['lugar5'].replace(1,5)
region['lugar6'] = region['lugar6'].replace(1,6)
region['Region']= (region[['lugar1','lugar2','lugar3','lugar4','lugar5','lugar6']].max(axis =1)
      .replace([1,2,3,4,5,6],['Central','Chorotega','PacÃ­fico central','Brunca','Huetar AtlÃ¡ntica','Huetar Norte']))
region['Target_new']=region['Target_new'].replace([1,2,3,4],['extreme', 'moderate', 'vulnerable', 'non vulnerable'])
region.groupby(['Region','Target_new']).size().groupby(level=0).apply(lambda x: 100 * x / x.sum()).unstack().plot(kind='bar',stacked=True)


# # Fitting the models

# Now our data is clean and aggregated by household.<br>
# We will create 3 datasets for predicting different labels.<br>
# 
# 1. **data_houshold_2 - 2 Labels of balanced data:** <br>
#     1 = extreme poverty ==> 1<br>
#     2 = moderate poverty ==> 1 <br>
#     3 = vulnerable households ==> 1 <br>
#     4 = non vulnerable households ==> 0 <br>
#     <br>
# 2. **data_houshold_2_i - 2 Labels of imbalanced data:** <br>
#     1 = extreme poverty ==> 1<br>
#     2 = moderate poverty ==> 1 <br>
#     3 = vulnerable households ==> 0 <br>
#     4 = non vulnerable households ==> 0<br>
#     <br>
# 3. **data_houshold - 4 Labels of imbalanced data:** <br>
#     1 = extreme poverty <br>
#     2 = moderate poverty <br>
#     3 = vulnerable households <br>
#     4 = non vulnerable households <br>

# Most features consist of binary data but some features have int/float type data that is much larger than 1, so data will be scaled with min-max scaler.<br> 
# 
# For each dataset we will run 2 methods of dimensionality reduction:
# * Feature selection with lasso logistic regression
# * Feature extraction with PCA
#   
# and 2 classifiers: 
# * Logistic regression
# * Random forest
#  
# We will run a grid search cross validation (CV=10, since the data size is fairly small, using accuracy score for multiclass and roc-auc for the 2 class datasets), with diffrent parameters to find the best classifires for each dataset.<br>
# <br>
# <br>
# We will also look at logistic regression and random forest calssifiers, with all the features

# **The hyper-parameter grids for each configuration:**

# In[ ]:


param_grid_log_reg_L1 = {'Feature_selection__estimator__C': [0.1, 1, 10],
                          'clf__C' : [0.1, 1, 10], 
                          'Feature_selection__threshold': [0.05, 0.1, 0.2]
                         }

param_grid_log_reg_PCA = {'Feature_extraction__n_components': [30, 60, 90, 120],
                          'clf__C' : [0.1, 1, 10]
                         }

param_grid_log_reg = {'clf__C' : [0.01, 0.1, 1, 10, 100]}

param_grid_RF_L1 = {'Feature_selection__estimator__C': [0.1, 1, 10],
                    'Feature_selection__threshold': [0.05, 0.1, 0.2], 
                    'clf__min_samples_split': [2,8,15,20]
                   }

param_grid_RF_PCA = {'Feature_extraction__n_components': [30, 60, 90, 120],
                     'clf__min_samples_split': [2,8,15,20]
                    }

param_grid_RF = {'clf__min_samples_split': [2,8,15,20]}


# **Define scaler, feature extraction and selection models and classifiers:**

# In[ ]:


max_iter_param=100

#Scaler
minmax_scaler = MinMaxScaler()

#Feature selection - 2 labels
Lasso_log_reg2 = LogisticRegression(penalty='l1',
                                    class_weight='balanced',
                                    solver='liblinear'
                                   )
                                    
Feature_selection2=SelectFromModel(Lasso_log_reg2)                

#Feature selection - 4 labels
Lasso_log_reg4 =LogisticRegression(penalty='l1' ,
                                   max_iter = max_iter_param,
                                   multi_class='multinomial',
                                   class_weight='balanced', 
                                   solver='saga'
                                  )

Feature_selection4=SelectFromModel(Lasso_log_reg4)     

#Feature Extraction
PCA_features=PCA()

#Logistic regression classifier - 2 labels
log_reg_clf2=LogisticRegression(solver='lbfgs',
                                class_weight='balanced'
                               )

#Logistic regression classifier - 4 labels
log_reg_clf4=LogisticRegression(solver='saga',
                                max_iter = max_iter_param,
                                multi_class='multinomial',
                                class_weight='balanced'
                               )

#Random forest classifier
RF_clf = RandomForestClassifier(class_weight='balanced' ,
                                n_estimators=100,
                                random_state=123
                               )


# **Define the piplines:**

# In[ ]:


#Pipelines - 2 labels

Log_reg_pipe2_L1 = Pipeline(steps=[('Scaler', minmax_scaler),
                                   ('Feature_selection', Feature_selection2),
                                   ('clf',log_reg_clf2)
                                  ]
                            )

Log_reg_pipe2_PCA = Pipeline(steps=[('Scaler', minmax_scaler),
                                     ('Feature_extraction', PCA_features),
                                     ('clf',log_reg_clf2)
                                   ]
                            ) 

Log_reg_pipe2= Pipeline (steps=[('Scaler', minmax_scaler),
                                ('clf',log_reg_clf2)
                               ]
                        )

RF_pipe_2_L1 = Pipeline(steps=[('Scaler', minmax_scaler),
                               ('Feature_selection', Feature_selection2),
                               ('clf',RF_clf)
                                  ]
                            )


# In[ ]:


#Pipelines - 4 labels
Log_reg_pipe4_L1 = Pipeline(steps=[('Scaler', minmax_scaler),
                                   ('Feature_selection', Feature_selection4),
                                   ('clf',log_reg_clf4)
                                  ]
                            )

Log_reg_pipe4_PCA = Pipeline(steps=[('Scaler', minmax_scaler),
                                    ('Feature_extraction', PCA_features),
                                    ('clf',log_reg_clf4)
                                   ]
                            ) 

Log_reg_pipe4= Pipeline (steps=[('Scaler', minmax_scaler),
                                ('clf',log_reg_clf4)
                               ]
                        )

RF_pipe_4_L1 = Pipeline(steps=[('Scaler', minmax_scaler),
                               ('Feature_selection', Feature_selection4),
                               ('clf',RF_clf)
                                  ]
                            )


# In[ ]:


#pipelines for 2 or 4 labels
RF_pipe_PCA = Pipeline(steps=[('Scaler', minmax_scaler),
                              ('Feature_extraction', PCA_features),
                              ('clf',RF_clf)
                             ]
                      )
                                   

RF_pipe = Pipeline(steps=[('Scaler', minmax_scaler),
                          ('clf',RF_clf)
                         ]
                  )


# **Helpful functions:**

# In[ ]:


#Runs the GridSearchCV fit and finds the best classifer:
def fit_model (pipe, grid, X_train, y_train, num_cv, scoring_param):
    best_clf=GridSearchCV(pipe, grid, cv=num_cv,scoring = scoring_param)
    print ('Begin GridSearchCV fit')
    t0 = time.time()
    best_clf.fit(X_train, y_train)
    t1 = time.time()
    h, m ,s=time_convert(t1-t0)
    print('GridSearchCV ended. Elapsed time: {0:.0f} hours, {1:.0f} minutes and {2:.0f} seconds'.format(h,m,s))
    return best_clf

#Returns DF with relevant columns (parameter, mean_test_score and rank_test_score)
#from cv_results_ of the after GridSearchCV fit, sorted by test score rank
def cv_results (clf) :
    df_cv_results = pd.DataFrame(clf.cv_results_)
    param_list=[i for i in list(df_cv_results.columns) if 'param_' in i]
    df_cv_results_filter=df_cv_results[param_list+['mean_test_score','rank_test_score']].sort_values(by=['rank_test_score'])
    return df_cv_results_filter


#Returns DF with best classifer params and train/test scores of for each pipeline:
def results_test_df(rdf,name,clf,X_test,y_test,scoring_param):
    if scoring_param == 'roc_auc':
        rdf=rdf.append ({ 'Model':name,
                          'Best_params':clf.best_params_,
                          'Best_Train_Score':clf.best_score_.round(5),
                          'Best_Test_Score':roc_auc_score(y_true=y_test, y_score=clf.predict(X_test)).round(5)}
                          ,ignore_index = True
                        )
    elif scoring_param == 'accuracy':
        rdf=rdf.append ({ 'Model':name,
                          'Best_params': clf.best_params_,
                          'Best_Train_Score':clf.best_score_.round(5),
                          'Best_Test_Score':accuracy_score(y_true=y_test, y_pred=clf.predict(X_test)).round(5)}
                          ,ignore_index = True
                       )
    return rdf

def time_convert (t):
    h,m1=divmod(t, 3600)
    m,s=divmod(m1, 60) 
    return h, m ,s

def find_best(model_dict, X_train,y_train,X_test,y_test,num_cv,scoring_param,df_init,df_all_best_results):
    for name, (pipe,grid) in model_dict.items():
        print ('Model name:',name)
        best_clf = fit_model (pipe, grid, X_train,y_train, num_cv,scoring_param)
        best_clf_cv_results=cv_results(best_clf)
        display(best_clf_cv_results)
        df_all_best_results = df_all_best_results.append(results_test_df(df_init,name,best_clf,X_test,y_test, scoring_param))
        print('======================================================')
    return df_all_best_results


# Define some helpful parameters:

# In[ ]:


#feature dictionary - 2 labels
model_dict2={'Feature selection and logistic regression':(Log_reg_pipe2_L1, param_grid_log_reg_L1),
            'PCA and logistic regression':               (Log_reg_pipe2_PCA, param_grid_log_reg_PCA),
            'All features and logistic regression':      (Log_reg_pipe2, param_grid_log_reg),
            'Feature selection and random forest':       (RF_pipe_2_L1,param_grid_RF_L1),
            'PCA and random forest':                     (RF_pipe_PCA,param_grid_RF_PCA),
            'All features and random forest':            (RF_pipe , param_grid_RF)
            }

#feature dictionary - 4 labels
model_dict4={'Feature selection and logistic regression':(Log_reg_pipe4_L1, param_grid_log_reg_L1),
            'PCA and logistic regression':               (Log_reg_pipe4_PCA, param_grid_log_reg_PCA),
            'All features and logistic regression':      (Log_reg_pipe4, param_grid_log_reg),
            'Feature selection and random forest':       (RF_pipe_4_L1,param_grid_RF_L1),
            'PCA and random forest':                     (RF_pipe_PCA,param_grid_RF_PCA),
            'All features and random forest':            (RF_pipe , param_grid_RF)
            }

#init dataframe for best scores of each pipline
df_init=pd.DataFrame(columns=['Model','Best_params','Best_Train_Score','Best_Test_Score'])

#number of cross valisation folds
num_cv = 10

#scoring parameter - 2 labels
scoring_param2 = 'roc_auc'

#scoring parameter - 4 labels
scoring_param4 = 'accuracy'

#Columns display definition
pd.set_option('display.max_colwidth', 0)


# ## <font color=red>2 Labels - balanced data</font> 

# In[ ]:


data_houshold_2=data_houshold.copy()


# In[ ]:


#Replace the label values to create 2 labels
data_houshold_2['Target_new']=data_houshold_2['Target_new'].replace([2,3,4], [1,1,0])


# In[ ]:


data_houshold_2['Target_new'].value_counts()


# The labels of this data are more balanced and they represent non_vulnerable (0) and vulnerable/poor (1) households. <br>
# We'll split the data to train and test:

# In[ ]:


X_train2, X_test2, y_train2, y_test2 = split(data_houshold_2.drop(axis=1, columns=['Target_new']), 
                                             data_houshold_2['Target_new'], 
                                             test_size =0.3, random_state=123)


# In[ ]:



X_train, X_test, y_train, y_test = X_train2, X_test2, y_train2, y_test2
df_all_best_results=pd.DataFrame()


df_all_best_results2=find_best(model_dict2, X_train,y_train,X_test,y_test,num_cv,scoring_param2,df_init,df_all_best_results)


# ## <font color=green>2 Labels - imbalanced data</font> 

# In[ ]:


data_houshold_2_i=data_houshold.copy()


# In[ ]:


data_houshold_2_i['Target_new']=data_houshold_2_i['Target_new'].replace([2,3,4], [1,0,0])


# In[ ]:


data_houshold_2_i['Target_new'].value_counts()


# The labels of this data are not balanced, but the split makes more sense, since the lables now represent poor (1) and not-poor (0) households <br>
# We'll split the data to train and test

# In[ ]:


X_train2_i, X_test2_i, y_train2_i, y_test2_i = split(data_houshold_2_i.drop(axis=1, columns=['Target_new']), 
                                                     data_houshold_2_i['Target_new'], 
                                                     test_size =0.3, random_state=123)


# In[ ]:



df_all_best_results=pd.DataFrame()
X_train, X_test, y_train, y_test = X_train2_i, X_test2_i, y_train2_i, y_test2_i

df_all_best_results2_i=find_best(model_dict2, X_train,y_train,X_test,y_test,num_cv,scoring_param2,df_init,df_all_best_results)


# ## <font color=blue>4 labels of imbalanced data</font>

# In[ ]:


data_houshold['Target_new'].value_counts()


# Split the data to train and test

# In[ ]:


X_train4, X_test4, y_train4, y_test4 = split(data_houshold.drop(axis=1, columns=['Target_new']), 
                                             data_houshold['Target_new'], 
                                             test_size =0.3, random_state=123
                                            )


# In[ ]:


df_all_best_results=pd.DataFrame()
X_train, X_test, y_train, y_test = X_train4, X_test4, y_train4, y_test4

df_all_best_results4=find_best(model_dict4, X_train,y_train,X_test,y_test,num_cv,scoring_param4,df_init,df_all_best_results)


# # Conclusions

# For Each dataset we have a all the best estimators from each type of pipeline. we will compare them to each other on the test data.

# ## <font color=red>2 Labels - balanced data</font> 

# In[ ]:


df_all_best_results2.sort_values(['Best_Test_Score','Best_Train_Score'],ascending = False)


# Logistic regression looks like the better method (with C=1). Seems like keeping all the features is be as good as PCA.

# ## <font color=green>2 Labels - imbalanced data</font> 

# In[ ]:


df_all_best_results2_i.sort_values(['Best_Test_Score','Best_Train_Score'],ascending = False)


# Again, using all features or extracting with PCA seem like equal methods, combined with logistic regression as classifier, with C=1, and they preform the best on the test data. 

# ## <font color=blue>4 labels of imbalanced data</font>

# In[ ]:


df_all_best_results4.sort_values(['Best_Test_Score','Best_Train_Score'],ascending = False)


# for multiclass, random forest classifier performs better than logistic regresstion, with feature selection (Threshold = 0.2) having the best test score.
