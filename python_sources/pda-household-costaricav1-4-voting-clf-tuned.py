#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
from numpy.random import randint,uniform
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from mpl_toolkits.mplot3d import Axes3D
from collections import OrderedDict
import matplotlib.pyplot as plt
from IPython.display import display
import warnings
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set(style = 'darkgrid') #
import os
print(os.listdir("../input"))
from pylab import rcParams
import lightgbm as lgb
from datetime import datetime
#from lightgbm import LGBMClassifier,LGBMModel
from lightgbm.sklearn import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier,VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold,cross_val_score,GridSearchCV,RandomizedSearchCV,train_test_split
from sklearn.metrics import classification_report,f1_score,make_scorer, precision_score, recall_score, confusion_matrix
rcParams['figure.figsize'] = 25, 12.5
rcParams['font.size'] = 50
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
pd.set_option('display.max_columns',None)
warnings.filterwarnings('ignore')
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train.head()


# In[ ]:


submission_sample = pd.read_csv('../input/sample_submission.csv')


# In[ ]:


train.info()


# In[ ]:


test.info()


# In[ ]:


int_uniq_values = train.select_dtypes('int').nunique().value_counts()
df = pd.DataFrame(int_uniq_values,columns = ['# columns']).reset_index().rename(columns = {'index':'Unique value'})
sns.barplot(df['Unique value'],df['# columns'])
plt.title('Frequency distribution of unique values in a integer column')
plt.show()
#int_uniq_values
#sns.barplot(int_uniq_values)


# In[ ]:


continuos_var = train.select_dtypes('float').columns.tolist()
continuos_var


# In[ ]:


color_dict = OrderedDict({1: 'red', 2: 'orange', 3: 'blue', 4: 'green'})
poverty_dict = OrderedDict({1: 'extreme', 2: 'moderate', 3: 'vulnerable', 4: 'non vulnerable'})


# In[ ]:


def plot_continuos(data,var):
    for key,clr in color_dict.items():
        sns.kdeplot(train[train.Target==key][var],color=clr,label = poverty_dict[key])
    plt.xlabel(var)
    plt.ylabel('Density')


# In[ ]:


plot_continuos(train,continuos_var[0])
 


# In[ ]:


plot_continuos(train,continuos_var[1])


# In[ ]:


plot_continuos(train,continuos_var[2])


# In[ ]:


plot_continuos(train,continuos_var[3])


# In[ ]:


plot_continuos(train,continuos_var[4])


# In[ ]:


plot_continuos(train,continuos_var[6])


# In[ ]:


train.select_dtypes('object').head()


# In[ ]:


test.select_dtypes('object').head()


# In[ ]:


train.loc[:,['dependency','edjefe','edjefa']] =train[['dependency','edjefe','edjefa']] .replace(to_replace={'yes':1,'no':0}).astype(np.float64) 
test.loc[:,['dependency','edjefe','edjefa']] = test[['dependency','edjefe','edjefa']].replace(to_replace={'yes':1,'no':0}).astype(np.float64) 


# ** Target label distribution**
# 1. How many households  are non-vulnerable to extremely poor?

# In[ ]:


household_train = train[train.parentesco1==1]
print('Number of households in the train data:{}'.format(len(household_train)))
sns.countplot(household_train['Target'])
plt.ylabel('Number of households')
plt.xlabel('Household type')
plt.xticks([x - 1 for x in poverty_dict.keys()], list(poverty_dict.values()))
plt.show()


# In[ ]:


household_checktarget =train.groupby('idhogar')['Target'].apply(lambda x: x.nunique() == 1)
idhogar_mult_target = household_checktarget[household_checktarget!=True].index
print('Number of households where the poverty level is more than one:{}'.format(len(idhogar_mult_target)))


# In[ ]:


for idh in idhogar_mult_target:
    actual_povertylevel = train.loc[((train.parentesco1==1) &(train.idhogar==idh)),'Target']
    train.loc[(train.idhogar==idh),'Target'] = actual_povertylevel


# In[ ]:


household_num_head = train.groupby('idhogar')['parentesco1'].sum()
idhogar_nohead = household_num_head[household_num_head!=1].index
print('Number of households where there are no head:{}'.format(len(idhogar_nohead)))


# In[ ]:


idhogar_nohead


# In[ ]:


train[train.idhogar == 'f2bfa75c4']


# In[ ]:


household_test= test[test['parentesco1'] == 1]

# Create a gender mapping
household_train['gender'] = household_train['male'].replace({1: 'M', 0: 'F'})

# Boxplot
sns.boxplot(x = 'Target', y = 'meaneduc', hue = 'gender', data = household_train)
plt.title('Mean Education vs poverty level by Gender')
plt.xticks([x - 1 for x in poverty_dict.keys()], list(poverty_dict.values()))
plt.show()


# In[ ]:


sns.violinplot(x = 'Target',y='meaneduc',hue = 'gender', data = household_train)
plt.title('Mean Education vs poverty level by Gender')
plt.xticks([x - 1 for x in poverty_dict.keys()], list(poverty_dict.values()))
plt.show()


# In[ ]:


id_columns =  ['Id', 'idhogar', 'Target']

household_boolean = ['hacdor', 'hacapo', 'v14a', 'refrig', 'paredblolad', 'paredzocalo', 
           'paredpreb','pisocemento', 'pareddes', 'paredmad',
           'paredzinc', 'paredfibras', 'paredother', 'pisomoscer', 'pisoother', 
           'pisonatur', 'pisonotiene', 'pisomadera',
           'techozinc', 'techoentrepiso', 'techocane', 'techootro', 'cielorazo', 
           'abastaguadentro', 'abastaguafuera', 'abastaguano',
            'public', 'planpri', 'noelec', 'coopele', 'sanitario1', 
           'sanitario2', 'sanitario3', 'sanitario5',   'sanitario6',
           'energcocinar1', 'energcocinar2', 'energcocinar3', 'energcocinar4', 
           'elimbasu1', 'elimbasu2', 'elimbasu3', 'elimbasu4', 
           'elimbasu5', 'elimbasu6', 'epared1', 'epared2', 'epared3',
           'etecho1', 'etecho2', 'etecho3', 'eviv1', 'eviv2', 'eviv3', 
           'tipovivi1', 'tipovivi2', 'tipovivi3', 'tipovivi4', 'tipovivi5', 
           'computer', 'television', 'lugar1', 'lugar2', 'lugar3',
           'lugar4', 'lugar5', 'lugar6', 'area1', 'area2']

household_continuos = ['v2a1', 'dependency', 'edjefe', 'edjefa', 'meaneduc', 'overcrowding']
household_ordered = [ 'rooms', 'r4h1', 'r4h2', 'r4h3', 'r4m1','r4m2','r4m3', 'r4t1',  'r4t2', 
              'r4t3', 'v18q1', 'tamhog','tamviv','hhsize','hogar_nin',
              'hogar_adul','hogar_mayor','hogar_total',  'bedrooms', 'qmobilephone']

###########################################################################################################
ind_bool = ['v18q', 'dis', 'male', 'female', 'estadocivil1', 'estadocivil2', 'estadocivil3', 
            'estadocivil4', 'estadocivil5', 'estadocivil6', 'estadocivil7', 
            'parentesco1', 'parentesco2',  'parentesco3', 'parentesco4', 'parentesco5', 
            'parentesco6', 'parentesco7', 'parentesco8',  'parentesco9', 'parentesco10', 
            'parentesco11', 'parentesco12', 'instlevel1', 'instlevel2', 'instlevel3', 
            'instlevel4', 'instlevel5', 'instlevel6', 'instlevel7', 'instlevel8', 
            'instlevel9', 'mobilephone']
ind_ordered = ['rez_esc', 'escolari', 'age']

squared_common = ['SQBescolari','SQBage', 'SQBhogar_total','SQBedjefe','SQBhogar_nin','SQBovercrowding', 
           'SQBdependency', 'SQBmeaned']


# In[ ]:


household_train.head(2)


# **Drop all squared variables as they exhibit significant  correlation with the original variables**

# In[ ]:


feature_train_household = household_train.drop(squared_common+['gender'],axis=1)[id_columns[0:3]+household_boolean+household_continuos+household_ordered]
feature_test_household = household_test.drop(squared_common,axis=1)[id_columns[0:2]+household_boolean+household_continuos+household_ordered]
#target_train = household_train['Target']-1

print('Household training features data shape:{}\nHousehold Test data shape:{}'.format(feature_train_household.shape,feature_test_household.shape))


# In[ ]:


##Target variable is present in train data but not in test data


# **Create aggregate features out of individual rows**

# In[ ]:


feature_train_ind_bool = train.groupby('idhogar')[ind_bool].sum().reset_index()
feature_test_ind_bool = test.groupby('idhogar')[ind_bool].sum().reset_index()

print('Individual training boolean features data shape:{}\nIndividual test boolean features data shape:{}'.format(feature_train_ind_bool.shape,feature_test_ind_bool.shape))


# In[ ]:


feature_train_ind_bool.head(2)


# In[ ]:





# In[ ]:


feature_train_ind_cont = train.groupby('idhogar')[ind_ordered].agg(['mean', 'max', 'min', 'sum'])
feature_test_ind_cont = test.groupby('idhogar')[ind_ordered].agg(['mean', 'max', 'min', 'sum'])

new_cols = []
for col in feature_train_ind_cont.columns.levels[0]:
    for stat in feature_train_ind_cont.columns.levels[1]:
        new_cols.append(f'{col}-{stat}')
feature_train_ind_cont.columns = new_cols
feature_test_ind_cont.columns = new_cols

feature_train_ind_cont.reset_index(inplace=True)
feature_test_ind_cont.reset_index(inplace=True)


# In[ ]:


print('Individual training continuous features data shape:{}\nIndividual test continuous features data shape:{}'.format(feature_train_ind_cont.shape,feature_test_ind_cont.shape))


# In[ ]:


feature_train_ind_cont.head(2)


# In[ ]:


feature_train_ind_cont['rez_esc-mean'].isnull().sum()/len(feature_train_ind_cont)


# **Merging individual features**

# In[ ]:


feature_train_ind = pd.merge(feature_train_ind_bool,feature_train_ind_cont,on='idhogar',how='inner')
feature_test_ind = pd.merge(feature_test_ind_bool,feature_test_ind_cont,on='idhogar',how='inner')
print('Individual training  features data shape:{}\nIndividual test  features data shape:{}'.format(feature_train_ind.shape,feature_test_ind.shape))


# In[ ]:


feature_train_ind.head(2)


# **Merging individual features with household features**

# In[ ]:


feature_train = pd.merge(feature_train_household,feature_train_ind,on='idhogar',how='left')
#feature_train.drop('Target',axis=1,inplace=True)
feature_test = pd.merge(feature_test_household,feature_test_ind,on='idhogar',how='left')
print('Combined features train data shape:{}\nCombined features test data shape:{}'.format(feature_train.shape,feature_test.shape))


# In[ ]:


feature_train.head(2)


# **Drop id columns which are not required and align  the test columns  with train columns **

# In[ ]:


target_train = feature_train['Target']-1
test_idhogar = feature_test['idhogar'].values
try:
    feature_train.drop('Target',axis=1,inplace=True)
    feature_train.drop(['Id','idhogar'],axis=1,inplace=True)
    feature_test.drop(['Id','idhogar'],axis=1,inplace=True)
except:
    print("Couldnot drop all columns")
feature_train, feature_test = feature_train.align(feature_test, axis = 1, join = 'inner')
feature_X_train,feature_X_valid,target_y_train,target_y_valid = train_test_split(feature_train,target_train,test_size=0.2,stratify=target_train,random_state=123)



# In[ ]:


feature_train.head(2)


# In[ ]:


feature_test.head(2)


# In[ ]:


print('train features data shape:{}\ntvalid features data shape:{}\ntrain target shape:{}\nvalid target shape:{}'.format(feature_X_train.shape,feature_X_valid.shape,target_y_train.shape,target_y_valid.shape))


# **Custom Performance Metric**

# In[ ]:


def macro_f1_score(labels, predictions):
    # Reshape the predictions as needed
    predictions = predictions.reshape(len(np.unique(labels)), -1 ).argmax(axis = 0)
    
    metric_value = f1_score(labels, predictions, average = 'macro')
    
    # Return is name, value, is_higher_better
    return 'macro_f1', metric_value, True


# **Initialize  cross validation folds**

# In[ ]:


nfolds=5
sfold = StratifiedKFold(n_splits= nfolds,shuffle=True)


# In[ ]:


'''grid_params ={
              'num_leaves': randint(12,30,5), 
              'learning_rate':[0.001,0.01,0.1,0.2,0.3],
              'min_child_samples': randint(12,30,5), 
             #'min_child_weight': [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4],
              'subsample': uniform(0.5,1,4), 
              'colsample_bytree': uniform(0.5,1,4),
              'random_state':[100,200,300]             
            }

#my_scorer = make_scorer(macro_f1_score, greater_is_better=True)
def get_best_params(train_X,train_y):
    clf_lgb = lgb.LGBMClassifier(max_depth=-1,objective='multiclass',silent=True, metric='None',n_jobs=-1, class_weight='balanced')
    grid = RandomizedSearchCV(clf_lgb, grid_params,verbose=20, cv=5,n_jobs=-1,scoring='f1_macro')
    grid.fit(train_X,train_y)
    return grid.best_params_,grid.best_score_
    '''


# In[ ]:


#best_params,best_score = get_best_params(feature_train,target_train)


# In[ ]:


#best_params


# In[ ]:


'''
colsample_bytree = best_params['colsample_bytree']
learning_rate = best_params['learning_rate']
min_child_samples = best_params['min_child_samples']
num_leaves = best_params['num_leaves']
random_state = best_params['random_state']
subsample = best_params['subsample']
'''


# In[ ]:


dict_optimum_params = {'colsample_bytree': 0.605756972263847,
 'learning_rate': 0.01,
 'min_child_samples': 19,
 'num_leaves': 29,
 'subsample': 0.5522796688057564}


# **Baseline model(with no tuning)**
# 
# 1.LightGBM

# In[ ]:


def train_eval(feature_train,feature_test,target_train):
    '''
    clf_lgb = lgb.LGBMClassifier(objective = 'multiclass',
                                 n_jobs = -1,
                                 metric = 'None',
                                 colsample_bytree=dict_optimum_params['colsample_bytree'],
                                 learning_rate=learning_rate,
                                 min_child_samples=min_child_samples,
                                 num_leaves=num_leaves,
                                 subsample=subsample,
                                 random_state=random_state,
                                 class_weight='balanced')
                                 '''
    valid_scores_list = []
    test_predictions_df = pd.DataFrame()
    feature_columns = feature_train.columns
    feature_importance = np.zeros(len(feature_columns))
    featuresNames = []
    featureImps =[]

    feature_train_arr = feature_train.values
    feature_test_arr = feature_test.values
    target_train_arr = target_train.values
    
    clfs =[]
    for i in range(10):
        clf=LGBMClassifier(objective = 'multiclass',
                           colsample_bytree= 0.605,
                           learning_rate= 0.01,
                           min_child_samples= 19,
                           num_leaves=29,
                           subsample=0.552,
                           n_jobs = -1,
                           metric = 'None',
                           random_state=100+i,
                           class_weight='balanced')
        clfs.append(('lgbm{}'.format(i), clf))
        
    
    vc = VotingClassifier(clfs, voting='soft')
    
    for i, (train_index,valid_index) in enumerate(sfold.split(feature_train,target_train)):
        fold_predictions_df = pd.DataFrame()        
        # Training and validation data
        X_train = feature_train_arr[train_index]
        X_valid = feature_train_arr[valid_index]
        y_train = target_train_arr[train_index]
        y_valid = target_train_arr[valid_index]
        
        '''
        fit_params={"early_stopping_rounds":100,
            "eval_metric" : macro_f1_score, 
            "eval_set" : [(X_train,y_train), (X_valid,y_valid)],
            'eval_names': ['train', 'valid'],
            'verbose': 100,
            'categorical_feature': 'auto'}
        '''
        vc.fit(X_train,y_train)
        score = f1_score(y_valid,vc.predict(X_valid),average='macro')
        print('Mean score on fold:{}:{}'.format(i+1,score))
        #valid_scores_list.append(clf_lgb.best_score_['valid']['macro_f1'])
        #display(f'Fold {i + 1}, Validation Score: {round(valid_scores_list[i], 5)}, Estimators Trained: {clf_lgb.best_iteration_}')
        fold_probabilitites = vc.predict_proba(feature_test_arr)
        for j in range(4):
            fold_predictions_df[(j + 1)] = fold_probabilitites[:, j]
            
        fold_predictions_df['idhogar'] = test_idhogar
        fold_predictions_df['fold'] = (i+1)
        
        test_predictions_df = test_predictions_df.append(fold_predictions_df)
        #fold_feature_importance = vc.feature_importances_
        #fold_feature_importance = 100.0 * (fold_feature_importance / fold_feature_importance.max())
        #feature_importance = (feature_importance+fold_feature_importance)/nfolds
        #predictions.columns = ['Poverty_Extreme','Poverty_Mderate','Poverty_Vulnerable','Poverty_Unvulnerable']
    test_predictions_df = test_predictions_df.groupby('idhogar', as_index = False).mean()
    test_predictions_df['Target'] = test_predictions_df[[1, 2, 3, 4]].idxmax(axis = 1)
    test_predictions_df['Score'] = test_predictions_df[[1, 2, 3, 4]].max(axis = 1)
    #sorted_idx = np.argsort(feature_importance)
    '''
    for item in sorted_idx[::-1][:]:
        featuresNames.append(np.asarray(feature_columns)[item])
        featureImps.append(feature_importance[item])
        featureImportance = pd.DataFrame([featuresNames, featureImps]).transpose()
        featureImportance.columns = ['FeatureName', 'Importance']
        
    ''' 
    return test_predictions_df


# In[ ]:


test_predictions_df=train_eval(feature_train,feature_test,target_train)


# In[ ]:


#sns.barplot(x =featureImportance['Importance'][0:10],y=featureImportance['FeatureName'][0:10] )


# In[ ]:


#featureImportance.tail(30)


# In[ ]:


#mean_feat_importance = np.mean(featureImportance.Importance)
#important_feat_filtered = featureImportance.loc[featureImportance.Importance>mean_feat_importance,'FeatureName'].tolist()


# In[ ]:


#important_feat_filtered


# In[ ]:


'''
feature_train_imp = feature_train[important_feat_filtered]
feature_test_imp= feature_test[important_feat_filtered]
feature_train_imp, feature_test_imp = feature_train_imp.align(feature_test_imp, axis = 1, join = 'inner')
print('train features data shape:{}\ntest features data shape:{}\ntrain target shape:{}'.format(feature_train_imp.shape,feature_test_imp.shape,target_train.shape))
'''


# In[ ]:


#_,test_predictions_df=train_eval(feature_train_imp,feature_test_imp,target_train)


# In[ ]:


test_predictions_df.head(2)


# In[ ]:


test.head(2)


# In[ ]:


submission_df = test.loc[:,['Id','idhogar']]


# In[ ]:


submission = pd.merge(submission_df,test_predictions_df[['idhogar','Target']],on='idhogar',how='left').drop('idhogar',axis=1)


# In[ ]:


submission.head(2)


# In[ ]:


submission['Target'] = submission['Target'].fillna(4).astype(np.int8)


# In[ ]:


submission.head(2)


# In[ ]:


#today = datetime.now()
#sub_file = 'submission_LGB_{}.csv'.format(str(today.strftime('%Y-%m-%d-%H-%M')))
submission.to_csv('sample_submission.csv',index=False)


# In[ ]:




