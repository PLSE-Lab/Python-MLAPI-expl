#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# import os
# __print__ = print
# def print(string):
#     os.system(f'echo \"{string}\"')
#     __print__(string)


# This notebook follows my notebooks:
# * https://www.kaggle.com/arateris/xgb-rf-with-gridsearch-for-forest-classifier/ that looked for hyper-paramters (although it was without all the FE so the parameters may not be ideal anymore)
# * https://www.kaggle.com/arateris/stacked-classifiers-for-forest-cover where I was playing with various FE and original stacking
# * https://www.kaggle.com/arateris/2-layer-k-fold-learning-forest-cover with more stacking and features, but using the imbalanced accuracy that I remove here.
# 
# 
# There was a lot of inspirations from other notebooks which I will mention on the way as much as I remember.

# # TODO
# * Check joint add/drop features (especially the one feature encoding, target encoding etc.)
# * Check binary vs categorized features. 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import gc


from time import time

from collections import Counter
from itertools import combinations

from sklearn.model_selection import cross_val_score,cross_validate, train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder

from sklearn.metrics import accuracy_score, confusion_matrix, balanced_accuracy_score, make_scorer
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline, make_union

from category_encoders import CountEncoder, CatBoostEncoder, TargetEncoder

from mlxtend.classifier import StackingCVClassifier, StackingClassifier
from mlxtend.feature_selection import ColumnSelector

# import lightgbm
from lightgbm import LGBMClassifier, plot_importance
from xgboost import XGBClassifier
from catboost import CatBoostClassifier


from tqdm import tqdm


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import sys
def sizeof_fmt(num, suffix='B'):
    ''' by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified'''
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, 'Yi', suffix)

def show_sys_var():
    for name, size in sorted(((name, sys.getsizeof(value)) for name, value in locals().items()),
                             key= lambda x: -x[1])[:10]:
        print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))
        
def mem_reduce(df):
    for col in df.columns:
        if df[col].dtype=='float64': 
            df[col] = df[col].astype('float32')
        if df[col].dtype=='int64': 
            if df[col].max()<1: df[col] = df[col].astype(bool)
            elif df[col].max()<128: df[col] = df[col].astype('int8')
            elif df[col].max()<32768: df[col] = df[col].astype('int16')
            else: df[col] = df[col].astype('int32')
    return df


pd.options.display.max_seq_items = 500

# Any results you write to the current directory are saved as output.


# #Import datasets

# In[ ]:


X = pd.read_csv("../input/learn-together/train.csv", index_col='Id')
X_test = pd.read_csv("../input/learn-together/test.csv", index_col='Id')

y = X['Cover_Type'] # this is the target
X = X.drop('Cover_Type', axis = 1)

test_index = X_test.index
num_train = len(X)

print('Train set shape : ', X.shape)
print('Test set shape : ', X_test.shape)

#renaming to avoid long names in the feature engineering
X.rename({'Horizontal_Distance_To_Roadways':'HDR',
              'Horizontal_Distance_To_Hydrology':'HDH',
              'Horizontal_Distance_To_Fire_Points':'HDF',
              'Vertical_Distance_To_Hydrology':'VDH'}, axis="columns", inplace=True)
X_test.rename({'Horizontal_Distance_To_Roadways':'HDR',
              'Horizontal_Distance_To_Hydrology':'HDH',
              'Horizontal_Distance_To_Fire_Points':'HDF',
              'Vertical_Distance_To_Hydrology':'VDH'}, axis="columns", inplace=True)


raw_columns = X.columns
categorial_feat = [] 
binary_feat = [] 
X.head()


# Note : large difference between train and test size. Will need to check input distributions.

# In[ ]:


# Helper function to generate submission files.
def to_submission(preds, file_name):
    output = pd.DataFrame({'Id': test_index,
                           'Cover_Type': preds})
    output.to_csv(file_name+'.csv', index=False)


# In[ ]:


# Some getters to eacily grab new models later.

def get_XGB(n_estimators= 500, random_state= 2019):
    return XGBClassifier( n_estimators=n_estimators, 
                    learning_rate= 0.1, 
                    max_depth= 200,  
                    objective= 'binary:logistic',
                    random_state= random_state,
                    n_jobs=-1)

def get_LGBM(n_estimators=500, random_state= 2019):
    return LGBMClassifier(n_estimators=n_estimators,  
                     learning_rate= 0.1,
                     objective= 'multiclass', 
                     num_class=7,
                     random_state= random_state,
                     n_jobs=-1)

def get_RF(n_estimators = 750, random_state = 2019):
    return RandomForestClassifier(n_estimators = n_estimators, 
                            max_depth = None, 
                            random_state=random_state,
                            n_jobs=-1)

def get_EXT(n_estimators = 750, random_state = 2019):
    return ExtraTreesClassifier(n_estimators = n_estimators, 
                            max_depth = None, 
                            random_state=random_state,
                            n_jobs=-1)

def get_CB (n_estimators = 200, random_state = 2019, cat_features = None):
    return CatBoostClassifier(n_estimators = n_estimators, 
                              max_depth = None,
                              learning_rate=0.3,
                              random_state=random_state, 
                              cat_features = cat_features,
                              verbose=False)

def get_LR():
    return LogisticRegression(max_iter=1000, 
                       n_jobs=-1,
                       solver= 'lbfgs', #?
                       multi_class = 'multinomial')



# (Now quite old) List of classifiers and hyper-parameters
# - XGBClassifier
# -- Params: {'n_estimators': 500, 'learning_rate': 0.1, 'max_depth': 10} ?
# -- n_estimators = 719, max_depth = 464 https://www.kaggle.com/phsheth/forestml-part-6-stacking-eval-selected-fets-2  https://www.kaggle.com/joshofg/pure-random-forest-hyperparameter-tuning
# - RFClassifier
# -- {'max_depth': 100, 'max_features': 0.3, 'n_estimators': 2000}  https://www.kaggle.com/arateris/xgb-rf-with-gridsearch-for-forest-classifier/
# -- Params: {n_estimators = 719, max_features = 0.3, max_depth = 464, min_samples_split = 2, min_samples_leaf = 1, bootstrap = False} https://www.kaggle.com/joshofg/pure-random-forest-hyperparameter-tuning
# - ExtraTrees
# -- Params : n_estimators = 750, max_features = 0.3, max_depth = None,  https://www.kaggle.com/arateris/xgb-rf-with-gridsearch-for-forest-classifier/
# - LGBM 
# -- Params : n_estimators=400,  num_leaves=100  ?  https://www.kaggle.com/stevegreenau/stacking-multiple-classifiers-clustering
# -- {'learning_rate': 0.5, 'max_depth': 25, 'n_estimators': 500}  https://www.kaggle.com/arateris/xgb-rf-with-gridsearch-for-forest-classifier/
# - ADABoost 
# -- Params : {max_depth  = 464, min_samples_split = 2, min_samples_leaf = 1,}  https://www.kaggle.com/phsheth/forestml-part-6-stacking-eval-selected-fets-2
# 
# 

# # Feature engineering

# In[ ]:


print('Missing Label? ', y.isnull().any())
print('Missing train data? ', X.isnull().any().any())
print('Missing test data? ', X_test.isnull().any().any())


# --> No missing data.

# In[ ]:


print (X.dtypes.value_counts())
print (X_test.dtypes.value_counts())


# --> Everything in numeric. 
# 
# Soil_type and Wilderness_area are categorial data already put as one hot encoded.

# ## Categories vs OHE
# Originally transformed the binary classes to categorial features but maybe not a good idea so in this version I won't keep it for training at the end. 
# I keep it here to do frequency encoding and may try to use it again later.

# In[ ]:


#transform Soil_Type into categorial
def categorify(df, col_string_search, remove_original=False):
    cols = df.columns
    for key_str in col_string_search:
        new_col_name = key_str+'_cat'
        df[new_col_name]=0
        for col in cols:
            if ~str(col).find(key_str):
                binary_feat.append(col)
                df[new_col_name]= df[new_col_name]+int(str(col).lstrip(key_str))*df[col]
                if remove_original:
                    df.drop(col, axis=1, inplace=True)
        categorial_feat.append(new_col_name)
#         df[new_col_name] = df[new_col_name].astype('category')
        
#keeping track of the categorial features
    return df


# In[ ]:


cols_to_categorify = ['Soil_Type', 'Wilderness_Area']

all_data = X.append(X_test)
all_data = categorify(all_data, cols_to_categorify, remove_original=False)

X= all_data.loc[:num_train]
X_test= all_data.loc[num_train+1:]


X_test.head()


# In[ ]:


X.describe()


# ## Fixing Hillshade_3pm
# Ploting all the histograms I saw one weird thing in the Hillshade_3pm

# In[ ]:


# for col in X.columns:
plt.figure(figsize=(15,5))
sns.distplot(X.Hillshade_3pm)
plt.show()


# In[ ]:


print(X.Hillshade_3pm[(X.Hillshade_3pm<130).to_numpy() &  (X.Hillshade_3pm>120).to_numpy()].value_counts())
print((X.Hillshade_3pm==0).sum())
print((X_test.Hillshade_3pm==0).sum())


# Hill_shade_3pm is missing ~30 values at 126. 
# Hill_shade_3pm has 88 values equal to 0 which probably should not. 1250 zeros in the test set. --> this is not so many values, but still prefer fixing it to avoid spreading bad data in the feature engineering and training.

# In[ ]:


# checking which features correlates with Hillshade_3pm
corr = X[X.Hillshade_3pm!=0].corr()
plt.figure(figsize=(12,12))
sns.heatmap(corr,annot=True)


# In[ ]:


#replacing the zeros for better guess, mainly to avoid zeros in the feature engineering and fake outliers. 
all_data = X.append(X_test)

cols_for_HS = ['Aspect','Slope', 'Hillshade_9am','Hillshade_Noon']
HS_zero = all_data[all_data.Hillshade_3pm==0]
HS_zero.shape

HS_train = all_data[all_data.Hillshade_3pm!=0]
# res = cross_val_score(RandomForestRegressor(n_estimators=100), HS_train.drop('Hillshade_3pm',axis=1), HS_train.Hillshade_3pm, n_jobs=-1, verbose=True)
# print(res) --> Output:  #[0.9996774  0.99989463 0.9999186 ]
##actually, the CV is so close to perfect that there is actually no new information here..keeping it for .. sanity ?

rf_hs = RandomForestRegressor(n_estimators=100).fit(HS_train[cols_for_HS], HS_train.Hillshade_3pm)
out = rf_hs.predict(HS_zero[cols_for_HS]).astype(int)
all_data.loc[HS_zero.index,'Hillshade_3pm'] = out

X['Hillshade_3pm']= all_data.loc[:num_train,'Hillshade_3pm']
X_test['Hillshade_3pm']= all_data.loc[num_train+1:,'Hillshade_3pm']

del(HS_train)


# In[ ]:


baseline_cols = X.columns


# ## Add PCA features and Gaussian Mixture
# 

# In[ ]:



t = time()

pca = PCA(n_components=0.99).fit(all_data)
trans = pca.transform(all_data)
print(trans.shape)

for i in range(trans.shape[1]):
    col_name= 'pca'+str(i+1)
    X[col_name] = trans[:num_train, i]
    X_test[col_name] = trans[num_train:, i]

print('duration: '+ str(time()-t))


# In[ ]:


# Adding Gaussian Mixture features to perform some unsupervised learning hints from the full data
# https://www.kaggle.com/stevegreenau/stacking-multiple-classifiers-clustering

t = time()
components = 10 # TODO check other numbers.  with 10 labels there are a few ones with 0 importances. 
gmix = GaussianMixture(n_components=components) 
gaussian = gmix.fit_predict(StandardScaler().fit_transform(all_data))

X['GM'] = gaussian[:num_train]
X_test['GM'] = gaussian[num_train:]

categorial_feat.append('GM')

for i in range(components):
    X['GM'+str(i)] = gaussian[:num_train]==i  
    X_test['GM'+str(i)] = gaussian[num_train:]==i
    binary_feat.append('GM'+str(i))

print('duration: '+ str(time()-t))


# In[ ]:


sns.violinplot(x='GM',y=y, data=X)


# In[ ]:


X.head()


# In[ ]:


# Helper function to generate some basic FE
def quick_fe(df, cols, operations, max_combination=2):
    
    if max_combination>=2:
        for col1, col2 in combinations(cols, 2):
            for ope in operations:
                if ope=='add': df[col1 + "_add_" + col2] = df[col1]+df[col2]
                elif ope=='minus': df[col1 + "_minus_" + col2] = df[col1]-df[col2]
                elif ope=='minabs': df[col1 + "_minabs_" + col2] = abs(df[col1]-df[col2])
                elif ope=='time': df[col1 + "_time_" + col2] = df[col1]*df[col2]
    if max_combination>=3:
        for col1, col2, col3 in combinations(cols, 3):
            for ope in operations:
                if ope=='add': df[col1 + "_" + col2 + "_" + col3+ "_add" ] = df[col1]+df[col2]+df[col3]
                elif ope=='time': df[col1 + "_" + col2+ "_" + col3+ "_time" ] = df[col1]*df[col2]*df[col3]
    return df

X.head()


# In[ ]:



# group all the FE features
def feature_eng(dataset):
    # https://www.kaggle.com/nadare/eda-feature-engineering-and-modeling-4th-359#nadare's-kernel
    #https://www.kaggle.com/lukeimurfather/adversarial-validation-train-vs-test-distribution
    #https://www.kaggle.com/evimarp/top-6-roosevelt-national-forest-competition
    
    dataset['Distance_hyd'] = (dataset['HDH']**2+dataset['VDH']**2)**0.5

    cols_to_combine = ['Elevation','HDH', 'HDF', 'HDR', 'VDH']
    dataset = quick_fe(dataset, cols_to_combine, ['add','time','minus', 'minabs'], max_combination=3)

#     cols_to_combine = ['Elevation', 'VDH']
#     dataset = quick_fe(dataset, cols_to_combine, ['add','time','minus', 'minabs'], max_combination=2)

    dataset['Mean_Distance']=(dataset.HDF + 
                               dataset.Distance_hyd + 
                               dataset.HDR) / 3 
    dataset['Elevation_Adj_distanceH'] = dataset['Elevation'] - 0.25*dataset['Distance_hyd']
    dataset['Elevation_Adj_distanceV'] = dataset['Elevation'] - 0.19*dataset['HDH']

    
    # Hillshade
    hillshade_col = ["Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm"]
    dataset = quick_fe(dataset,hillshade_col, ['add','minus'], max_combination=3)

    dataset["Hillshade_std"] = dataset[hillshade_col].std(axis=1)
    dataset["Hillshade_max"] = dataset[hillshade_col].max(axis=1)
    dataset["Hillshade_min"] = dataset[hillshade_col].min(axis=1)
   
    #Aspect
    dataset['Aspect'] = dataset['Aspect'].astype(int) % 360

    dataset['Sin_Aspect'] = np.sin(np.radians(dataset['Aspect']))
    dataset['Cos_Aspect'] = np.cos(np.radians(dataset['Aspect']))
    
    dataset['Slope_hyd'] = np.arctan(dataset['VDH']/(dataset['HDH']+0.001))
    dataset.Slope_hyd=dataset.Slope_hyd.map(lambda x: 0 if np.isinf(x) else x) # remove infinite value if any
    
    dataset['Sin_Slope_hyd'] = np.sin(np.radians(dataset['Slope_hyd']))
    dataset['Cos_Slope_hyd'] = np.cos(np.radians(dataset['Slope_hyd']))

    dataset['Sin_Slope'] = np.sin(np.radians(dataset['Slope']))
    dataset['Cos_Slope'] = np.cos(np.radians(dataset['Slope']))
    
    # extremely stony = 4, very stony = 3, stony = 2, rubbly = 1, None = 0
    Soil_to_stony = [4, 3, 1, 1, 1, 2, 0, 0, 3, 1,
                1, 2, 1, 0, 0, 0, 0, 0, 0, 0,
                0, 4, 4, 4, 4, 4, 3, 4, 4, 4, 
                4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]
    dataset['Stonyness'] = [Soil_to_stony[x] for x in (dataset['Soil_Type_cat'].astype(int)-1)]
    dataset.loc[:,'Extremely_Stony']= dataset['Stonyness']==4
    dataset.loc[:,'Very_Stony']= dataset['Stonyness']==3
    dataset.loc[:,'Stony']= dataset['Stonyness']==2
    dataset.loc[:,'Rubbly']= dataset['Stonyness']==1
    dataset.loc[:,'Stony_NA']= dataset['Stonyness']==0
    
    return dataset

categorial_feat.append('Stonyness')
binary_feat.append('Extremely_Stony')
binary_feat.append('Very_Stony')
binary_feat.append('Stony')
binary_feat.append('Rubbly')
binary_feat.append('Stony_NA')

X = feature_eng(X)
X_test = feature_eng(X_test)

columns = X.columns


# ## Categorial combining

# In[ ]:


def combine_features(df, col1, col2):
    new_name= str(col1)+'_'+str(col2)
    mixed = df[col1].astype(str)+'_'+df[col2].astype(str)
#     print(mixed.head())
    label_enc = LabelEncoder()
    df[new_name] = label_enc.fit_transform(mixed)
    return df


# In[ ]:


plt.figure(figsize=(15,8))
sns.violinplot(x='GM',y='Wilderness_Area_cat', data=X)


# In[ ]:


all_data = X.append(X_test)
all_data = combine_features(all_data, 'Stonyness', 'Wilderness_Area_cat')
all_data = combine_features(all_data, 'Soil_Type_cat', 'Wilderness_Area_cat')
all_data = combine_features(all_data, 'GM', 'Wilderness_Area_cat')
all_data = combine_features(all_data, 'GM', 'Soil_Type_cat')
X = all_data[:num_train]
X_test = all_data[num_train:]
categorial_feat.append('Stonyness_Wilderness_Area_cat')
categorial_feat.append('Soil_Type_cat_Wilderness_Area_cat')
categorial_feat.append('GM_Wilderness_Area_cat')
categorial_feat.append('GM_Soil_Type_cat')


# In[ ]:


plt.figure(figsize=(15,8))
sns.violinplot(x=y,y=all_data[:num_train].Stonyness_Wilderness_Area_cat)


# In[ ]:


plt.figure(figsize=(15,8))
sns.violinplot(x='Soil_Type_cat',y='Wilderness_Area_cat', data=X)


# ## Count Encoding

# In[ ]:


all_data = X.append(X_test)

count_enc = CountEncoder()
count_encoded = count_enc.fit_transform(all_data[categorial_feat].astype(str))

all_data = all_data.join(count_encoded.add_suffix("_count"))

X = all_data[:num_train]
X_test = all_data[num_train:]

del(all_data)


# In[ ]:


X.describe()


# ## Target encoding 

# In[ ]:


# Create the encoder itself
target_enc = TargetEncoder(cols=categorial_feat)

# Fit the encoder using the categorical features and target
target_enc.fit(X[categorial_feat], y) 

# Transform the features, rename the columns with _target suffix, and join to dataframe
X = X.join(target_enc.transform(X[categorial_feat]).add_suffix('_target'))
X_test = X_test.join(target_enc.transform(X_test[categorial_feat]).add_suffix('_target'))


# In[ ]:


#TODO : target encode high importance values?


# In[ ]:


columns = X.columns
X_test.head()


# # Feature selection

# In[ ]:


X = mem_reduce(X)
X_test = mem_reduce(X_test)


# In[ ]:


all_features = X.columns
added_features = X.columns.drop(baseline_cols).drop(binary_feat, errors='ignore') #droping binary and keep only categorial..maybe should not ?
all_minus_binary_features = X.drop(binary_feat, axis=1).columns
categorized_baseline = baseline_cols.drop(binary_feat, errors='ignore')


# In[ ]:


#Change here to modify the remaining  notebook classifier of reference.
# some constant to be low for kernel editing/ code checking and higher for commit/real training.

CLASSIFIER = get_XGB  #get_LGBM   # The classifier to test the feature with
ESTIMATORS = 75      # number of estimators for the classifier testing features (don't take too high or the testing will take ages)
MODEL_FACTOR = 1     # a model factor to scale the finale model for offline commit (large) or debugging/coding (small)
CV = 6 #2            # CV

def get_clf(n_estimators=ESTIMATORS, random_state= 2019):
    return CLASSIFIER( n_estimators=n_estimators, random_state = random_state)

def check_selected_features_score(features, n_estimators=ESTIMATORS, verbose=False):
    return np.mean(cross_val_score(get_clf(n_estimators=n_estimators), 
                                   pd.DataFrame(X[features]), 
                                   y, 
                                   cv=CV, 
                                   verbose=verbose))


# ## Check baseline scores with small estimator for speed
# 

# In[ ]:


raw_score = check_selected_features_score(raw_columns)
raw_cat_score = check_selected_features_score(categorized_baseline)
no_bin_score = check_selected_features_score(all_minus_binary_features)
full_feature_score = check_selected_features_score(all_features)

print('Raw score : ', raw_score)
print('Categorized Raw score : ', raw_cat_score)
print('No binary Features score : ', no_bin_score)
print('All Features score : ', full_feature_score)


# ## Check by adding only new features that bring in gains
# - Start from raw categorized data,
# - Add each feature individually to the base if check if it improved CV score.

# In[ ]:


def only_add_improving_features():
    score= pd.Series(index=added_features)
    for feat in added_features:
        cols = X[baseline_cols.drop(binary_feat, errors='ignore')].columns.to_list()
        cols.append(feat)
        score[feat] = check_selected_features_score(cols)
        print(feat, score[feat], score[feat]>raw_cat_score )

    cols =  X[baseline_cols.drop(binary_feat, errors='ignore')].columns
    improving_features = added_features[score>raw_cat_score]
    return cols.append(improving_features)


# In[ ]:


# improving_raw_features = only_add_improving_features()

# print(improving_raw_features)


# ## Check individual feature score
# - Check each feature score alone
# - Sort features by importance
# - Start with best feature,
# - Add each feature successively if they improve CV score.

# In[ ]:



def check_individual_score_and_add():
    score= pd.Series(index= all_minus_binary_features)
    for feat in all_minus_binary_features:
        score[feat] = check_selected_features_score(feat)
        print(feat, score[feat])
    score=score.sort_values(ascending=False)
    #check score by adding one by one. keep only improving feature
    hi_score = 0.0
    cols=[]
    for feat in score.index:
        cols.append(feat)
        res = check_selected_features_score(cols)
        print(feat, res, res>hi_score)
        if res>hi_score:
            hi_score=res
        else:
            cols.pop()
    return cols
    


# In[ ]:


individual_features_add = pd.Index(check_individual_score_and_add())
print(individual_features_add)


# ## Adding and removing from scratch
# additive
# - Start from raw categorized data
# - add each added features one by one successfully if improving the CV score.
# 
# droping
# - Start from all features (non-binary)
# - Remove individuals that reduce CV score
# 

# In[ ]:



# probably would be better to have a randomized approach that add/remove features not in a definite order

#add features one by one from scratch, possibly 'num_it' times.
def additive_feature_selection(num_it=1):
    
    hi_score = 0.0
    kept_cols = []
    for i in range(num_it):
        print( 'ADDITIVE FEATURE stage ', num_it)
        features = all_features.drop(kept_cols)
        for feat in features:
            kept_cols.append(feat)
            res = check_selected_features_score(kept_cols)
            print(feat, res, res>hi_score)
            if res>hi_score:
                hi_score=res
            else:
                kept_cols.pop()
    
        print( 'Adding mode finale score : ', hi_score)
    return kept_cols 

#remove features one by one from all features, possibly 'num_it' times.
def dropping_feature_selection(num_it=1):
    kept_cols = all_features
    for i in range(num_it):
        all_remaining_features = kept_cols
        hi_score = check_selected_features_score(kept_cols)
        print( 'Dropping mode baseline score : ', hi_score)

        for j, feat in reversed(list(enumerate(all_remaining_features))): #reversed to start from fancy added features
            test_cols=kept_cols.drop(feat)
            res = check_selected_features_score(test_cols)
            print(feat, res, res>=hi_score)
            if res>=hi_score:
                hi_score=res
                kept_cols=test_cols
        print( 'Dropping mode FINALE score : ', hi_score)
    return kept_cols 


# In[ ]:


# selected_feat_add = pd.Index(additive_feature_selection(num_it=1))

# print(selected_feat_add)


# In[ ]:


# selected_feat_drop =  pd.Index(dropping_feature_selection(num_it=1))
# print(selected_feat_drop)


# In[ ]:


from sklearn.feature_selection import RFECV

selector = RFECV(get_clf(), step=1, min_features_to_select=2, cv=CV, scoring='accuracy', verbose=False)
# selector = selector.fit(X, y)


# In[ ]:


# print("Optimal number of features : %d" % selector.n_features_)

# # Plot number of features VS. cross-validation scores
# plt.figure()
# plt.xlabel("Number of features selected")
# plt.ylabel("Cross validation score (nb of correct classifications)")
# plt.plot(range(1, len(selector.grid_scores_) + 1), selector.grid_scores_)
# plt.show()


# In[ ]:


# rfecv_selected = all_features[selector.support_]

# print(rfecv_selected)


# ## Checking scores with stronger classifier

# In[ ]:



# print('Raw score : ',check_selected_features_score(raw_columns, n_estimators=ESTIMATORS*MODEL_FACTOR))
# print('Categorized Raw score : ', check_selected_features_score(categorized_baseline, n_estimators=ESTIMATORS*MODEL_FACTOR))
# print('No binary Features score : ', check_selected_features_score(all_minus_binary_features, n_estimators=ESTIMATORS*MODEL_FACTOR))
# print('All Features score : ', check_selected_features_score(X.columns, n_estimators=ESTIMATORS*MODEL_FACTOR))
# # print('improving_raw_features', check_selected_features_score(improving_raw_features, n_estimators=ESTIMATORS*MODEL_FACTOR))
# print('individual_features_add', check_selected_features_score(individual_features_add, n_estimators=ESTIMATORS*MODEL_FACTOR))
# # print('selected_feat_add', check_selected_features_score(selected_feat_add, n_estimators=ESTIMATORS*MODEL_FACTOR))
# # print('selected_feat_drop', check_selected_features_score(selected_feat_drop, n_estimators=ESTIMATORS*MODEL_FACTOR))
# print('rfecv_selected', check_selected_features_score(rfecv_selected, n_estimators=ESTIMATORS*MODEL_FACTOR))
# gc.collect()


# # Model generation

# In[ ]:



def get_pipeline(features, n_estimators= ESTIMATORS*MODEL_FACTOR, random_state=2019):
    selector = ColumnSelector([(col not in features) for col in X.columns])
    return make_pipeline(selector, get_clf(n_estimators= n_estimators, 
                                           random_state=random_state))


# In[ ]:


#single model
# single_model = get_pipeline(rfecv_selected, n_estimators= individual_features_add_estimators)
# single_model.fit(X,y)
# single_model_pred = single_model.predict(X_test)
# to_submission(single_model_pred, 'single_model_pred')


# In[ ]:


# clf = StackingCVClassifier(classifiers=[get_pipeline(improving_raw_features ,n_estimators= 500),
#                                         get_pipeline(individual_features_add,n_estimators= 500),
#                                         get_pipeline(selected_feat_add,n_estimators= selected_feat_add_estimators),
#                                         get_pipeline(selected_feat_drop,n_estimators= selected_feat_drop_estimators),
#                                         get_pipeline(rfecv_selected,n_estimators= rfecv_selected_estimators)],
#                            meta_classifier=get_LGBM(),
#                            cv=CV, 
#                            random_state=666, 
#                            use_probas=True, 
#                            use_features_in_secondary=True, 
#                            verbose=True)
# # clf.fit(X, y)
# # print('Predicting test values...')
# # y_pred = clf.predict(X_test)
# # print('Saving predictions...')
# # to_submission(y_pred, 'lgbm_stack_preds_sub')
# print('Done')
                                        
# gc.collect()

