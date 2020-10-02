#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# standard data analysis modules
# i use these in my work on alternative data gathered on companies
# for my work in the asset management industry
import pandas as pd
import numpy as np
from IPython.display import display
import seaborn as sns

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
import matplotlib.ticker as ticker

# dont need these here
#from matplotlib_venn import venn2
#from matplotlib_venn import venn3

get_ipython().run_line_magic('matplotlib', 'inline')
pd.options.display.max_columns = None


# and also now import the AI libraries
# import catboost
from lightgbm import LGBMClassifier
from mlxtend.classifier import StackingCVClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn import model_selection, tree, preprocessing, metrics, linear_model
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, IsolationForest, BaggingClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from catboost import CatBoostClassifier
# from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
#from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.mixture import GaussianMixture

from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier

# deep learning
import keras
from keras import layers

# filter out all warnings 
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


# check on the panda version and its dependencies
# i run this from time to time to ensure all is up to date
pd.__version__
#pd.show_versions()


# In[ ]:


# i use these to input the relevant file names
# which i downloaded earlier and sit in the same directory as this
file_train = 'train.csv'
file_test = 'test.csv'
file_sampleSubmission = 'sample_submission.csv'


# In[ ]:


# which makes this a standard cell
df_train = pd.read_csv(file_train)
df_test = pd.read_csv(file_test)
df_sampleSubmission = pd.read_csv(file_sampleSubmission)


# In[ ]:


# useful maps of each wilderness area
# this makes it cledar that a model per wildnerness areas is likely to work best
# http://www.sangres.com/colorado/wilderness/rawahwilderness.htm#.XX6Pd3dFyUm
# http://www.sangres.com/colorado/wilderness/neota.htm#.XX6P_HdFyUk
# http://www.sangres.com/colorado/wilderness/comanchepeakwilderness.htm#.XX6QAXdFyUk
# http://www.sangres.com/colorado/wilderness/cachelapoudrewilderness.htm#.XX6QB3dFyUk


# In[ ]:


# names of csv files generated
sub_1  = 'submission20191029k_3a.csv' # the wilderness area models  this generally scores 0.82
sub_1a  = 'submission20191029k_3b.csv' # the wilderness area models with keras.  this generally scores 0.75
sub_2  = 'submission20191029k_3c.csv' # the entire forest model.  this generally scores 0.80
sub_2a = 'submission20191029k_3d.csv' # the entire forest model with keras.  this generally scores 0.75
sub_3  = 'submission20191029k_3e.csv' # the combined version of the above.  this geenrally scores 0.826
sub_4  = 'submission20191029k_3f.csv' # the best of the above (hopefully).  also gets to 0.826


# In[ ]:


# need to define the stacks, can change later for each area etc
# see cell below

random_state = 0

ab_clf = AdaBoostClassifier(n_estimators=500,
                            base_estimator=DecisionTreeClassifier(
                                min_samples_leaf=2,
                                random_state=random_state),
                            random_state=random_state)

et_clf = ExtraTreesClassifier(max_depth=None,
                              n_estimators=750,
                              n_jobs=-1,
                              random_state=random_state)

lg_clf = LGBMClassifier(n_estimators=300,
                         num_class=8,
                         num_leaves=25,
                         learning_rate=5,
                         min_child_samples=20,
                         bagging_fraction=.3,
                         bagging_freq=1,
                         reg_lambda = 10**4.5,
                         reg_alpha = 1,
                         feature_fraction=.2,
                         num_boost_round=4000,
                         max_depth=-1,
                         n_jobs=4,
                         silent=-1,
                         verbose=-1)

lda_clf = LinearDiscriminantAnalysis()

knn_clf = KNeighborsClassifier(n_jobs=-1, n_neighbors =1)

gnb_clf = GaussianNB()

svc_clf = SVC(random_state=random_state,
              probability=True,
              verbose=True)

bg_clf = BaggingClassifier(n_estimators=500,
                           verbose=0,
                           random_state=random_state)

gb_clf = GradientBoostingClassifier(n_estimators=500,
                              min_samples_leaf=100,
                              verbose=0,
                              random_state=random_state)

# oddly was not able to get xgb to work
# xgb_clf = XGBClassifier(n_estimators = 500,
#                        learning_rate = 0.1,
#                        max_depth = 200,
#                        objective = 'binary:logistic',
#                        random_state=random_state,
#                        n_jobs = -1)

cb_clf = CatBoostClassifier(n_estimators = 200,
                           max_depth = None,
                           learning_rate = 0.3,
                           random_state=random_state,
                           cat_features = None,
                           verbose = False)

rf_clf = RandomForestClassifier(n_estimators=750,
                                max_depth = None,
                                verbose=0,
                                random_state=random_state)

hg_clf = HistGradientBoostingClassifier(max_iter = 500,
                                        max_depth = 25,
                                        random_state = 0)


# In[ ]:


# stack_w1 to _w4 are the classifiers used on each wilderness area treated as a seperate model
# stack_all are the classifiers used on the entire forest
# similarly for meta_clf
# in other words, i am trying to optimise the use of the classifiers as well as their paramters

stack_w1 = [hg_clf, svc_clf, rf_clf, et_clf]
stack_w2 = [gnb_clf]
stack_w3 = [et_clf]
stack_w4 =  [hg_clf]
stack_all = [lg_clf, knn_clf, et_clf, ab_clf, rf_clf, hg_clf]
# stack_comb = [lg_clf, et_clf, rf_clf]
meta_clf_w1 = rf_clf
meta_clf_w2 = rf_clf
meta_clf_w3 = rf_clf
meta_clf_w4 = rf_clf
meta_clf_all = lg_clf


# In[ ]:


#
# in the first kernel i got 0.73950 score from XGBoost with all columns used and no adjustments made
# i then tried to drop less informative data and this did not improve the score
# so this time i will go through each data column and improve them one by one
# prior kernals had all the data visualisation, the below is the result of that
#


# In[ ]:


#
# setting up some functions now that we will be using later
#


# In[ ]:


# a function to plot the feature importance

def feature_importances(clf, X, y):
    clf = clf.fit(X, y)
    
    importances = pd.DataFrame({'Features': X.columns, 
                                'Importances': clf.feature_importances_})
    
    importances.sort_values(by=['Importances'], axis='index', ascending=False, inplace=True)

    fig = plt.figure(figsize=(14, 4))
    sns.barplot(x='Features', y='Importances', data=importances)
    plt.xticks(rotation='vertical')
    plt.show()
    return importances


# In[ ]:


# a function to select the key features given importance

def select(importances, edge):
    c = importances.Importances >= edge
    cols = importances[c].Features.values
    return cols


# In[ ]:


df_train.describe()


# In[ ]:


df_test.describe()


# In[ ]:


df_train.head()


# In[ ]:


# making a copy of df_train and df_test
# still getting used to .copy() vs no .copy()!!

df = df_train.copy()
df_test_copy = df_test.copy()


# In[ ]:


df.info()


# In[ ]:


df_test_copy.info()


# In[ ]:


# drop Soil_Type7 and Soil_Type15 because they contain 1 value in the training data but 2 in the test data

df.drop(['Soil_Type7','Soil_Type15'], axis=1, inplace=True)
df_test_copy.drop(['Soil_Type7','Soil_Type15'], axis=1, inplace=True)


# In[ ]:


# df.head()


# In[ ]:


# df_test_copy.head()


# In[ ]:


#
# start making new features
#


# In[ ]:


# First, add PCA features

data_pca = df.append(df_test_copy)
data_pca.drop(['Cover_Type','Id'], axis=1, inplace=True)

pca = PCA(n_components=0.99).fit(data_pca)
trans = pca.transform(data_pca)
print(trans.shape)

for i in range(trans.shape[1]):
    col_name= 'pca'+str(i+1)
    df[col_name] = trans[:len(df), i]
    df_test_copy[col_name] = trans[len(df):, i]


# In[ ]:


# df.tail()


# In[ ]:


# df_test_copy.head()


# In[ ]:


# the actual distance to hydrology sounds sensible

df['walking_distance_to_Hydrology'] = (df['Horizontal_Distance_To_Hydrology']**2 + df['Vertical_Distance_To_Hydrology']**2)**0.5
df_test_copy['walking_distance_to_Hydrology'] = (df_test_copy['Horizontal_Distance_To_Hydrology']**2 + df_test_copy['Vertical_Distance_To_Hydrology']**2)**0.5


# In[ ]:


# total amount of sunlight or shade makes sense
# not sure if i should merely add the columns or something else
# so we will go for total, mean, sum of squares, and max and min

df['Hillshade_total'] = df['Hillshade_9am'] + df['Hillshade_Noon'] + df['Hillshade_3pm']
df_test_copy['Hillshade_total'] = df_test_copy['Hillshade_9am'] + df_test_copy['Hillshade_Noon'] + df_test_copy['Hillshade_3pm']

df['Hillshade_mean'] = (df['Hillshade_9am'] + df['Hillshade_Noon'] + df['Hillshade_3pm'])/3
df_test_copy['Hillshade_mean'] = (df_test_copy['Hillshade_9am'] + df_test_copy['Hillshade_Noon'] + df_test_copy['Hillshade_3pm'])/3

# df['Hillshade_sumsq'] = df['Hillshade_9am']**2 + df['Hillshade_Noon']**2 + df['Hillshade_3pm']**2
# df_test_copy['Hillshade_sumsq'] = df_test_copy['Hillshade_9am']**2 + df_test_copy['Hillshade_Noon']**2 + df_test_copy['Hillshade_3pm']**2

# df['Hillshade_max'] = (df[['Hillshade_9am','Hillshade_Noon','Hillshade_3pm']]).max(axis='columns')
# df_test_copy['Hillshade_max'] = (df_test_copy[['Hillshade_9am','Hillshade_Noon','Hillshade_3pm']]).max(axis='columns')

# df['Hillshade_min'] = (df[['Hillshade_9am','Hillshade_Noon','Hillshade_3pm']]).min(axis='columns')
# df_test_copy['Hillshade_min'] = (df_test_copy[['Hillshade_9am','Hillshade_Noon','Hillshade_3pm']]).min(axis='columns')

# df['Hillshade_variation'] = df['Hillshade_max'] - df['Hillshade_min']
# df_test_copy['Hillshade_variation'] = df_test_copy['Hillshade_max'] - df_test_copy['Hillshade_min']

# df['Hillshade_9plus3'] = df['Hillshade_9am'] + df['Hillshade_3pm']
# df_test_copy['Hillshade_9plus3'] = df_test_copy['Hillshade_9am'] + df_test_copy['Hillshade_3pm']

df['Hillshade_total_noonx4'] = df['Hillshade_9am']*2 + df['Hillshade_Noon']*4 + df['Hillshade_3pm']
df_test_copy['Hillshade_total_noonx4'] = df_test_copy['Hillshade_9am']*2 + df_test_copy['Hillshade_Noon']*4 + df_test_copy['Hillshade_3pm']

# df['Hillshade_total_noonxhalf'] = df['Hillshade_9am'] + df['Hillshade_Noon']*0.5 + df['Hillshade_3pm']
# df_test_copy['Hillshade_total_noonxhalf'] = df_test_copy['Hillshade_9am'] + df_test_copy['Hillshade_Noon']*0.5 + df_test_copy['Hillshade_3pm']

df['Hillshade_morning'] = (df['Hillshade_9am'] + df['Hillshade_Noon'])/2
df_test_copy['Hillshade_morning'] = (df_test_copy['Hillshade_9am'] + df_test_copy['Hillshade_Noon'])/2

df['Hillshade_afternoon'] = (df['Hillshade_Noon'] + df['Hillshade_3pm'])/2
df_test_copy['Hillshade_afternoon'] = (df_test_copy['Hillshade_Noon'] + df_test_copy['Hillshade_3pm'])/2

# df['Hillshade_noon_morning'] = df['Hillshade_9am'] - df['Hillshade_Noon']
# df['Hillshade_afternoon_noon'] = df['Hillshade_Noon'] - df['Hillshade_3pm']
# df['Hillshade_afternoon_morning'] = df['Hillshade_9am'] - df['Hillshade_3pm']
# df_test_copy['Hillshade_noon_morning'] = df_test_copy['Hillshade_9am'] - df_test_copy['Hillshade_Noon']
# df_test_copy['Hillshade_afternoon_noon'] = df_test_copy['Hillshade_Noon'] - df_test_copy['Hillshade_3pm']
# df_test_copy['Hillshade_afternoon_morning'] = df_test_copy['Hillshade_9am'] - df_test_copy['Hillshade_3pm']

df['shade_morning_noon_ratio'] = df['Hillshade_9am']/(df['Hillshade_Noon']+1)
df['shade_noon_afternoon_ratio'] = df['Hillshade_Noon']/(df['Hillshade_3pm']+1)
df['shade_morning_afternoon_ratio'] = df['Hillshade_9am']/(df['Hillshade_3pm']+1)
df_test_copy['shade_morning_noon_ratio'] = df_test_copy['Hillshade_9am']/(df_test_copy['Hillshade_Noon']+1)
df_test_copy['shade_noon_afternoon_ratio'] = df_test_copy['Hillshade_Noon']/(df_test_copy['Hillshade_3pm']+1)
df_test_copy['shade_morning_afternoon_ratio'] = df_test_copy['Hillshade_9am']/(df_test_copy['Hillshade_3pm']+1)


# In[ ]:


# # looking at the soil types to class together part 1

df['extremey_stoney'] = df['Soil_Type1'] + df['Soil_Type24'] + df['Soil_Type25'] + df['Soil_Type27'] + df['Soil_Type28'] + df['Soil_Type29'] + df['Soil_Type30'] + df['Soil_Type31'] + df['Soil_Type32'] + df['Soil_Type33'] + df['Soil_Type34'] + df['Soil_Type36'] + df['Soil_Type37'] + df['Soil_Type38'] + df['Soil_Type39'] + df['Soil_Type40']

df['very_stoney'] = df['Soil_Type2'] + df['Soil_Type9'] + df['Soil_Type18'] + df['Soil_Type26']

df['rubbly'] = df['Soil_Type3'] + df['Soil_Type4'] + df['Soil_Type5'] + df['Soil_Type10'] + df['Soil_Type11'] + df['Soil_Type14']

df['stoney'] = df['Soil_Type6'] + df['Soil_Type12']

df['rest'] = df['Soil_Type8'] + df['Soil_Type14'] + df['Soil_Type16'] + df['Soil_Type17'] + df['Soil_Type19'] + df['Soil_Type20'] + df['Soil_Type21'] + df['Soil_Type22'] + df['Soil_Type23'] + df['Soil_Type35']

df_test_copy['extremey_stoney'] = df_test_copy['Soil_Type1'] + df_test_copy['Soil_Type24'] + df_test_copy['Soil_Type25'] + df_test_copy['Soil_Type27'] + df_test_copy['Soil_Type28'] + df_test_copy['Soil_Type29'] + df_test_copy['Soil_Type30'] + df_test_copy['Soil_Type31'] + df_test_copy['Soil_Type32'] + df_test_copy['Soil_Type33'] + df_test_copy['Soil_Type34'] + df_test_copy['Soil_Type36'] + df_test_copy['Soil_Type37'] + df_test_copy['Soil_Type38'] + df_test_copy['Soil_Type39'] + df_test_copy['Soil_Type40']

df_test_copy['very_stoney'] = df_test_copy['Soil_Type2'] + df_test_copy['Soil_Type9'] + df_test_copy['Soil_Type18'] + df_test_copy['Soil_Type26']

df_test_copy['rubbly'] = df_test_copy['Soil_Type3'] + df_test_copy['Soil_Type4'] + df_test_copy['Soil_Type5'] + df_test_copy['Soil_Type10'] + df_test_copy['Soil_Type11'] + df_test_copy['Soil_Type14']

df_test_copy['stoney'] = df_test_copy['Soil_Type6'] + df_test_copy['Soil_Type12']

df_test_copy['rest'] = df_test_copy['Soil_Type8'] + df_test_copy['Soil_Type14'] + df_test_copy['Soil_Type16'] + df_test_copy['Soil_Type17'] + df_test_copy['Soil_Type19'] + df_test_copy['Soil_Type20'] + df_test_copy['Soil_Type21'] + df_test_copy['Soil_Type22'] + df_test_copy['Soil_Type23'] + df_test_copy['Soil_Type35']


# In[ ]:


# looking at the soil types to class together part 2

df['rock_outcrop'] = df['Soil_Type1'] + df['Soil_Type3'] + df['Soil_Type4'] + df['Soil_Type5'] + df['Soil_Type6'] + df['Soil_Type10'] + df['Soil_Type27'] + df['Soil_Type28'] + df['Soil_Type32'] + df['Soil_Type33'] + df['Soil_Type35']

df['Ratake'] = df['Soil_Type2']
df['limber'] = df['Soil_Type8']
df['rock_land'] = df['Soil_Type11'] + df['Soil_Type12'] + df['Soil_Type13'] + df['Soil_Type34'] + df['Soil_Type36']
df['Aquolis'] = df['Soil_Type14']
df['Cryoborolis'] = df['Soil_Type16']
df['Cryaquolis'] = df['Soil_Type17']
df['Borohemists'] = df['Soil_Type19']
df['Cryaquolls'] = df['Soil_Type20'] + df['Soil_Type23'] + df['Soil_Type21']
df['Catamount'] = df['Soil_Type26'] + df['Soil_Type31']
df['Legault'] = df['Soil_Type29'] + df['Soil_Type30']
df['Cryorthents'] = df['Soil_Type37']
df['other_complex'] = df['Soil_Type9'] + df['Soil_Type18'] + df['Soil_Type22'] + df['Soil_Type24'] + df['Soil_Type25']

df_test_copy['rock_outcrop'] = df_test_copy['Soil_Type1'] + df_test_copy['Soil_Type3'] + df_test_copy['Soil_Type4'] + df_test_copy['Soil_Type5'] + df_test_copy['Soil_Type6'] + df_test_copy['Soil_Type10'] + df_test_copy['Soil_Type27'] + df_test_copy['Soil_Type28'] + df_test_copy['Soil_Type32'] + df_test_copy['Soil_Type33'] + df_test_copy['Soil_Type35']

df_test_copy['Ratake'] = df_test_copy['Soil_Type2']
df_test_copy['limber'] = df_test_copy['Soil_Type8']
df_test_copy['rock_land'] = df_test_copy['Soil_Type11'] + df_test_copy['Soil_Type12'] + df_test_copy['Soil_Type13'] + df_test_copy['Soil_Type34'] + df_test_copy['Soil_Type36']
df_test_copy['Aquolis'] = df_test_copy['Soil_Type14']
df_test_copy['Cryoborolis'] = df_test_copy['Soil_Type16']
df_test_copy['Cryaquolis'] = df_test_copy['Soil_Type17']
df_test_copy['Borohemists'] = df_test_copy['Soil_Type19']
df_test_copy['Cryaquolls'] = df_test_copy['Soil_Type20'] + df_test_copy['Soil_Type23'] + df_test_copy['Soil_Type21']
df_test_copy['Catamount'] = df_test_copy['Soil_Type26'] + df_test_copy['Soil_Type31']
df_test_copy['Legault'] = df_test_copy['Soil_Type29'] + df_test_copy['Soil_Type30']
df_test_copy['Cryorthents'] = df_test_copy['Soil_Type37']
df_test_copy['other_complex'] = df_test_copy['Soil_Type9'] + df_test_copy['Soil_Type18'] + df_test_copy['Soil_Type22'] + df_test_copy['Soil_Type24'] + df_test_copy['Soil_Type25']


# In[ ]:


# looking at the soil types to class together part 3

df['cathedral_f'] = df['Soil_Type1']
df['vanet_f'] = df['Soil_Type2'] + df['Soil_Type5'] + df['Soil_Type6']
df['haploborolis_f'] = df['Soil_Type3']
df['ratake_f'] = df['Soil_Type4']
df['troutville_f'] = df['Soil_Type9']
df['bullwark_f'] = df['Soil_Type10'] + df['Soil_Type11']
df['legault_f'] = df['Soil_Type12'] + df['Soil_Type29']
df['catamount_f'] = df['Soil_Type13'] + df['Soil_Type26'] + df['Soil_Type32']
df['gateview_f'] = df['Soil_Type17']
df['rogert_f'] = df['Soil_Type18']

df['leighcan_f'] = df['Soil_Type21'] + df['Soil_Type22'] + df['Soil_Type23'] + df['Soil_Type24'] + df['Soil_Type25'] + df['Soil_Type27'] + df['Soil_Type28'] + df['Soil_Type31'] + df['Soil_Type33'] + df['Soil_Type38']

df['como_f'] = df['Soil_Type30']
df['bross_f'] = df['Soil_Type36']
df['moran_f'] = df['Soil_Type39']
df['limber_f'] = df['Soil_Type8']

df['other_f'] = df['Soil_Type14'] + df['Soil_Type16'] + df['Soil_Type19'] + df['Soil_Type20'] + df['Soil_Type34'] + df['Soil_Type35'] + df['Soil_Type37']

df_test_copy['cathedral_f'] = df_test_copy['Soil_Type1']
df_test_copy['vanet_f'] = df_test_copy['Soil_Type2'] + df_test_copy['Soil_Type5'] + df_test_copy['Soil_Type6']
df_test_copy['haploborolis_f'] = df_test_copy['Soil_Type3']
df_test_copy['ratake_f'] = df_test_copy['Soil_Type4']
df_test_copy['troutville_f'] = df_test_copy['Soil_Type9']
df_test_copy['bullwark_f'] = df_test_copy['Soil_Type10'] + df_test_copy['Soil_Type11']
df_test_copy['legault_f'] = df_test_copy['Soil_Type12'] + df_test_copy['Soil_Type29']
df_test_copy['catamount_f'] = df_test_copy['Soil_Type13'] + df_test_copy['Soil_Type26'] + df_test_copy['Soil_Type32']
df_test_copy['gateview_f'] = df_test_copy['Soil_Type17']
df_test_copy['rogert_f'] = df_test_copy['Soil_Type18']

df_test_copy['leighcan_f'] = df_test_copy['Soil_Type21'] + df_test_copy['Soil_Type22'] + df_test_copy['Soil_Type23'] + df_test_copy['Soil_Type24'] + df_test_copy['Soil_Type25'] + df_test_copy['Soil_Type27'] + df_test_copy['Soil_Type28'] + df_test_copy['Soil_Type31'] + df_test_copy['Soil_Type33'] + df_test_copy['Soil_Type38']

df_test_copy['como_f'] = df_test_copy['Soil_Type30']
df_test_copy['bross_f'] = df_test_copy['Soil_Type36']
df_test_copy['moran_f'] = df_test_copy['Soil_Type39']
df_test_copy['limber_f'] = df_test_copy['Soil_Type8']

df_test_copy['other_f'] = df_test_copy['Soil_Type14'] + df_test_copy['Soil_Type16'] + df_test_copy['Soil_Type19'] + df_test_copy['Soil_Type20'] + df_test_copy['Soil_Type34'] + df_test_copy['Soil_Type35'] + df_test_copy['Soil_Type37']


# In[ ]:


# linking the soil type to area covered and elevation

data_soil = df.append(df_test_copy)
data_soil.drop(['Cover_Type','Id'], axis=1, inplace=True)

data_soil['Soil_Type'] = sum(i * data_soil['Soil_Type{}'.format(i)] for i in range(1, 41) if i != 7 | i != 15)

# soil_count = data_soil['Soil_Type'].value_counts().to_dict()
# data_soil['Soil_area'] = data_soil['Soil_Type'].map(soil_count)

soil_elevation = data_soil.groupby('Soil_Type')['Elevation'].median().to_dict()
data_soil['Soil_Elev_median'] = data_soil['Soil_Type'].map(soil_elevation)

# df['Soil_area'] = data_soil['Soil_area'][:len(df)]
# df_test_copy['Soil_area'] = data_soil['Soil_area'][len(df):]

df['Soil_Elev_median'] = data_soil['Soil_Elev_median'][:len(df)]
df_test_copy['Soil_Elev_median'] = data_soil['Soil_Elev_median'][len(df):]


# In[ ]:


# add the elevation of hydrology

df['hydrology_elevation'] = df['Elevation'] - df['Vertical_Distance_To_Hydrology']
df_test_copy['hydrology_elevation'] = df_test_copy['Elevation'] - df_test_copy['Vertical_Distance_To_Hydrology']


# In[ ]:


# # add the sum of elevation and hydrology elevation

# df['hydrology_elevation_plus'] = df['Elevation'] + df['Vertical_Distance_To_Hydrology']
# df_test_copy['hydrology_elevation_plus'] = df_test_copy['Elevation'] + df_test_copy['Vertical_Distance_To_Hydrology']


# In[ ]:


# difference and sum between distances to hydrology and roadways

df['hydrology_road_diff'] = df['Horizontal_Distance_To_Hydrology'] - df['Horizontal_Distance_To_Roadways']
df['hydrology_road_diff_abs'] = df['hydrology_road_diff'].abs()
df_test_copy['hydrology_road_diff'] = df_test_copy['Horizontal_Distance_To_Hydrology'] - df_test_copy['Horizontal_Distance_To_Roadways']
df_test_copy['hydrology_road_diff_abs'] = df_test_copy['hydrology_road_diff'].abs()

df['hydrology_road_sum'] = df['Horizontal_Distance_To_Hydrology'] + df['Horizontal_Distance_To_Roadways']
df_test_copy['hydrology_road_sum'] = df_test_copy['Horizontal_Distance_To_Hydrology'] + df_test_copy['Horizontal_Distance_To_Roadways']


# In[ ]:


# difference and sum between distances to hydrology and firepoints

df['hydrology_fire_diff'] = df['Horizontal_Distance_To_Hydrology'] - df['Horizontal_Distance_To_Fire_Points']
df['hydrology_fire_diff_abs'] = df['hydrology_fire_diff'].abs()
df_test_copy['hydrology_fire_diff'] = df_test_copy['Horizontal_Distance_To_Hydrology'] - df_test_copy['Horizontal_Distance_To_Fire_Points']
df_test_copy['hydrology_fire_diff_abs'] = df_test_copy['hydrology_fire_diff'].abs()

df['hydrology_fire_sum'] = df['Horizontal_Distance_To_Hydrology'] + df['Horizontal_Distance_To_Fire_Points']
df_test_copy['hydrology_fire_sum'] = df_test_copy['Horizontal_Distance_To_Hydrology'] + df_test_copy['Horizontal_Distance_To_Fire_Points']


# In[ ]:


# difference and sum between distances to roadways and firepoints

df['road_fire_diff'] = df['Horizontal_Distance_To_Roadways'] - df['Horizontal_Distance_To_Fire_Points']
df['road_fire_diff_abs'] = df['road_fire_diff'].abs()
df_test_copy['road_fire_diff'] = df_test_copy['Horizontal_Distance_To_Roadways'] - df_test_copy['Horizontal_Distance_To_Fire_Points']
df_test_copy['road_fire_diff_abs'] = df_test_copy['road_fire_diff'].abs()

df['road_fire_sum'] = df['Horizontal_Distance_To_Roadways'] + df['Horizontal_Distance_To_Fire_Points']
df_test_copy['road_fire_sum'] = df_test_copy['Horizontal_Distance_To_Roadways'] + df_test_copy['Horizontal_Distance_To_Fire_Points']


# In[ ]:


# # mean of all road / fire / hydrology distances

df['road_fire_hydro_mean'] = (df['Horizontal_Distance_To_Roadways'] + df['Horizontal_Distance_To_Fire_Points'] + df['Horizontal_Distance_To_Hydrology'])/3
df_test_copy['road_fire_hydro_mean'] = (df_test_copy['Horizontal_Distance_To_Roadways'] + df_test_copy['Horizontal_Distance_To_Fire_Points'] + df_test_copy['Horizontal_Distance_To_Hydrology'])/3


# In[ ]:


# distance to highest and lowest tree per wildness area and percentage difference

elevation1 = df[df['Wilderness_Area1']==1]['Elevation'].max()
elevation2 = df[df['Wilderness_Area2']==1]['Elevation'].max()
elevation3 = df[df['Wilderness_Area3']==1]['Elevation'].max()
elevation4 = df[df['Wilderness_Area4']==1]['Elevation'].max()

elevation10 = df[df['Wilderness_Area1']==1]['Elevation'].min()
elevation20 = df[df['Wilderness_Area2']==1]['Elevation'].min()
elevation30 = df[df['Wilderness_Area3']==1]['Elevation'].min()
elevation40 = df[df['Wilderness_Area4']==1]['Elevation'].min()

elevation1_t = df_test_copy[df_test_copy['Wilderness_Area1']==1]['Elevation'].max()
elevation2_t = df_test_copy[df_test_copy['Wilderness_Area2']==1]['Elevation'].max()
elevation3_t = df_test_copy[df_test_copy['Wilderness_Area3']==1]['Elevation'].max()
elevation4_t = df_test_copy[df_test_copy['Wilderness_Area4']==1]['Elevation'].max()

elevation10_t = df_test_copy[df_test_copy['Wilderness_Area1']==1]['Elevation'].min()
elevation20_t = df_test_copy[df_test_copy['Wilderness_Area2']==1]['Elevation'].min()
elevation30_t = df_test_copy[df_test_copy['Wilderness_Area3']==1]['Elevation'].min()
elevation40_t = df_test_copy[df_test_copy['Wilderness_Area4']==1]['Elevation'].min()

df['max_elevation_1'] = np.where(df['Wilderness_Area1']==1,elevation1,0)
df['max_elevation_2'] = np.where(df['Wilderness_Area2']==1,elevation2,0)
df['max_elevation_3'] = np.where(df['Wilderness_Area3']==1,elevation3,0)
df['max_elevation_4'] = np.where(df['Wilderness_Area4']==1,elevation4,0)
df['max_elevation'] = df['max_elevation_1'] + df['max_elevation_2'] + df['max_elevation_3'] + df['max_elevation_4']

df['min_elevation_1'] = np.where(df['Wilderness_Area1']==1,elevation10,0)
df['min_elevation_2'] = np.where(df['Wilderness_Area2']==1,elevation20,0)
df['min_elevation_3'] = np.where(df['Wilderness_Area3']==1,elevation30,0)
df['min_elevation_4'] = np.where(df['Wilderness_Area4']==1,elevation40,0)
df['min_elevation'] = df['min_elevation_1'] + df['min_elevation_2'] + df['min_elevation_3'] + df['min_elevation_4']

df_test_copy['max_elevation_1'] = np.where(df_test_copy['Wilderness_Area1']==1,elevation1_t,0)
df_test_copy['max_elevation_2'] = np.where(df_test_copy['Wilderness_Area2']==1,elevation2_t,0)
df_test_copy['max_elevation_3'] = np.where(df_test_copy['Wilderness_Area3']==1,elevation3_t,0)
df_test_copy['max_elevation_4'] = np.where(df_test_copy['Wilderness_Area4']==1,elevation4_t,0)
df_test_copy['max_elevation'] = df_test_copy['max_elevation_1'] + df_test_copy['max_elevation_2'] + df_test_copy['max_elevation_3'] + df_test_copy['max_elevation_4']

df_test_copy['min_elevation_1'] = np.where(df_test_copy['Wilderness_Area1']==1,elevation10_t,0)
df_test_copy['min_elevation_2'] = np.where(df_test_copy['Wilderness_Area2']==1,elevation20_t,0)
df_test_copy['min_elevation_3'] = np.where(df_test_copy['Wilderness_Area3']==1,elevation30_t,0)
df_test_copy['min_elevation_4'] = np.where(df_test_copy['Wilderness_Area4']==1,elevation40_t,0)
df_test_copy['min_elevation'] = df_test_copy['min_elevation_1'] + df_test_copy['min_elevation_2'] + df_test_copy['min_elevation_3'] + df_test_copy['min_elevation_4']

df['dist_to_area_max_elevation'] = df['max_elevation'] - df['Elevation']
# df['dist_to_area_min_elevation'] = df['Elevation'] - df['min_elevation']

df_test_copy['dist_to_area_max_elevation'] = df_test_copy['max_elevation'] - df_test_copy['Elevation']
# df_test_copy['dist_to_area_min_elevation'] = df_test_copy['Elevation'] - df_test_copy['min_elevation']

# df['pct_dist_to_area_max_elevation'] = (df['max_elevation'] - df['Elevation'])/df['max_elevation']
# df['pct_dist_to_area_min_elevation'] = (df['Elevation'] - df['min_elevation'])/df['min_elevation']

# df_test_copy['pct_dist_to_area_max_elevation'] = (df_test_copy['max_elevation'] - df_test_copy['Elevation'])/df_test_copy['max_elevation']
# df_test_copy['pct_dist_to_area_min_elevation'] = (df_test_copy['Elevation'] - df_test_copy['min_elevation'])/df_test_copy['min_elevation']

df.drop(['max_elevation_1', 'max_elevation_2','max_elevation_3','max_elevation_4',
        'min_elevation_1','min_elevation_2','min_elevation_3','min_elevation_4',
        'max_elevation','min_elevation'], axis=1, inplace=True)

df_test_copy.drop(['max_elevation_1', 'max_elevation_2','max_elevation_3','max_elevation_4',
                   'min_elevation_1','min_elevation_2','min_elevation_3','min_elevation_4',
                   'max_elevation','min_elevation'], axis=1, inplace=True)


# In[ ]:


# # horizontal distance to to furthest tree from hydrology (min is at zero) and percentage difference

# hydrology1 = df[df['Wilderness_Area1']==1]['Horizontal_Distance_To_Hydrology'].max()
# hydrology2 = df[df['Wilderness_Area2']==1]['Horizontal_Distance_To_Hydrology'].max()
# hydrology3 = df[df['Wilderness_Area3']==1]['Horizontal_Distance_To_Hydrology'].max()
# hydrology4 = df[df['Wilderness_Area4']==1]['Horizontal_Distance_To_Hydrology'].max()

# hydrology1_t = df_test_copy[df_test_copy['Wilderness_Area1']==1]['Horizontal_Distance_To_Hydrology'].max()
# hydrology2_t = df_test_copy[df_test_copy['Wilderness_Area2']==1]['Horizontal_Distance_To_Hydrology'].max()
# hydrology3_t = df_test_copy[df_test_copy['Wilderness_Area3']==1]['Horizontal_Distance_To_Hydrology'].max()
# hydrology4_t = df_test_copy[df_test_copy['Wilderness_Area4']==1]['Horizontal_Distance_To_Hydrology'].max()

# df['max_hydrology_1'] = np.where(df['Wilderness_Area1']==1,hydrology1,0)
# df['max_hydrology_2'] = np.where(df['Wilderness_Area2']==1,hydrology2,0)
# df['max_hydrology_3'] = np.where(df['Wilderness_Area3']==1,hydrology3,0)
# df['max_hydrology_4'] = np.where(df['Wilderness_Area4']==1,hydrology4,0)
# df['max_hydrology'] = df['max_hydrology_1'] + df['max_hydrology_2'] + df['max_hydrology_3'] + df['max_hydrology_4']

# df_test_copy['max_hydrology_1'] = np.where(df_test_copy['Wilderness_Area1']==1,hydrology1_t,0)
# df_test_copy['max_hydrology_2'] = np.where(df_test_copy['Wilderness_Area2']==1,hydrology2_t,0)
# df_test_copy['max_hydrology_3'] = np.where(df_test_copy['Wilderness_Area3']==1,hydrology3_t,0)
# df_test_copy['max_hydrology_4'] = np.where(df_test_copy['Wilderness_Area4']==1,hydrology4_t,0)
# df_test_copy['max_hydrology'] = df_test_copy['max_hydrology_1'] + df_test_copy['max_hydrology_2'] + df_test_copy['max_hydrology_3'] + df_test_copy['max_hydrology_4']

# df['dist_to_area_max_hydrology'] = df['max_hydrology'] - df['Horizontal_Distance_To_Hydrology']

# df_test_copy['dist_to_area_max_hydrology'] = df_test_copy['max_hydrology'] - df_test_copy['Horizontal_Distance_To_Hydrology']

# #df['pct_dist_to_area_max_hydrology'] = (df['max_hydrology'] - df['Horizontal_Distance_To_Hydrology'])/df['max_hydrology']

# #df_test_copy['pct_dist_to_area_max_hydrology'] = (df_test_copy['max_hydrology'] - df_test_copy['Horizontal_Distance_To_Hydrology'])/df_test_copy['max_hydrology']

# df.drop(['max_hydrology_1', 'max_hydrology_2','max_hydrology_3','max_hydrology_4',
#         'max_hydrology'], axis=1, inplace=True)

# df_test_copy.drop(['max_hydrology_1', 'max_hydrology_2','max_hydrology_3','max_hydrology_4',
#                   'max_hydrology'], axis=1, inplace=True)


# In[ ]:


# vertical distance to to furthest tree from hydrology and percentage difference

hydrologyv1 = df[df['Wilderness_Area1']==1]['Vertical_Distance_To_Hydrology'].max()
hydrologyv2 = df[df['Wilderness_Area2']==1]['Vertical_Distance_To_Hydrology'].max()
hydrologyv3 = df[df['Wilderness_Area3']==1]['Vertical_Distance_To_Hydrology'].max()
hydrologyv4 = df[df['Wilderness_Area4']==1]['Vertical_Distance_To_Hydrology'].max()

hydrologyv1_t = df_test_copy[df_test_copy['Wilderness_Area1']==1]['Vertical_Distance_To_Hydrology'].max()
hydrologyv2_t = df_test_copy[df_test_copy['Wilderness_Area2']==1]['Vertical_Distance_To_Hydrology'].max()
hydrologyv3_t = df_test_copy[df_test_copy['Wilderness_Area3']==1]['Vertical_Distance_To_Hydrology'].max()
hydrologyv4_t = df_test_copy[df_test_copy['Wilderness_Area4']==1]['Vertical_Distance_To_Hydrology'].max()

df['max_hydrology_v1'] = np.where(df['Wilderness_Area1']==1,hydrologyv1,0)
df['max_hydrology_v2'] = np.where(df['Wilderness_Area2']==1,hydrologyv2,0)
df['max_hydrology_v3'] = np.where(df['Wilderness_Area3']==1,hydrologyv3,0)
df['max_hydrology_v4'] = np.where(df['Wilderness_Area4']==1,hydrologyv4,0)
df['max_hydrologyv'] = df['max_hydrology_v1'] + df['max_hydrology_v2'] + df['max_hydrology_v3'] + df['max_hydrology_v4']

df_test_copy['max_hydrology_v1'] = np.where(df_test_copy['Wilderness_Area1']==1,hydrologyv1_t,0)
df_test_copy['max_hydrology_v2'] = np.where(df_test_copy['Wilderness_Area2']==1,hydrologyv2_t,0)
df_test_copy['max_hydrology_v3'] = np.where(df_test_copy['Wilderness_Area3']==1,hydrologyv3_t,0)
df_test_copy['max_hydrology_v4'] = np.where(df_test_copy['Wilderness_Area4']==1,hydrologyv4_t,0)
df_test_copy['max_hydrologyv'] = df_test_copy['max_hydrology_v1'] + df_test_copy['max_hydrology_v2'] + df_test_copy['max_hydrology_v3'] + df_test_copy['max_hydrology_v4']

df['distVert_to_area_max_hydrology'] = df['max_hydrologyv'] - df['Vertical_Distance_To_Hydrology']

df_test_copy['distVert_to_area_max_hydrology'] = df_test_copy['max_hydrologyv'] - df_test_copy['Vertical_Distance_To_Hydrology']

#df['pct_distVert_to_area_max_hydrology'] = (df['max_hydrologyv'] - df['Vertical_Distance_To_Hydrology'])/df['max_hydrologyv']

#df_test_copy['pct_distVert_to_area_max_hydrology'] = (df_test_copy['max_hydrologyv'] - df_test_copy['Vertical_Distance_To_Hydrology'])/df_test_copy['max_hydrologyv']

df.drop(['max_hydrology_v1', 'max_hydrology_v2','max_hydrology_v3','max_hydrology_v4',
        'max_hydrologyv'], axis=1, inplace=True)

df_test_copy.drop(['max_hydrology_v1', 'max_hydrology_v2','max_hydrology_v3','max_hydrology_v4',
                  'max_hydrologyv'], axis=1, inplace=True)


# In[ ]:


# # horizontal distance to to furthest tree from Roadways (min is at zero) and percentage difference

# Roadways1 = df[df['Wilderness_Area1']==1]['Horizontal_Distance_To_Roadways'].max()
# Roadways2 = df[df['Wilderness_Area2']==1]['Horizontal_Distance_To_Roadways'].max()
# Roadways3 = df[df['Wilderness_Area3']==1]['Horizontal_Distance_To_Roadways'].max()
# Roadways4 = df[df['Wilderness_Area4']==1]['Horizontal_Distance_To_Roadways'].max()

# Roadways1_t = df_test_copy[df_test_copy['Wilderness_Area1']==1]['Horizontal_Distance_To_Roadways'].max()
# Roadways2_t = df_test_copy[df_test_copy['Wilderness_Area2']==1]['Horizontal_Distance_To_Roadways'].max()
# Roadways3_t = df_test_copy[df_test_copy['Wilderness_Area3']==1]['Horizontal_Distance_To_Roadways'].max()
# Roadways4_t = df_test_copy[df_test_copy['Wilderness_Area4']==1]['Horizontal_Distance_To_Roadways'].max()

# df['max_Roadways_1'] = np.where(df['Wilderness_Area1']==1,Roadways1,0)
# df['max_Roadways_2'] = np.where(df['Wilderness_Area2']==1,Roadways2,0)
# df['max_Roadways_3'] = np.where(df['Wilderness_Area3']==1,Roadways3,0)
# df['max_Roadways_4'] = np.where(df['Wilderness_Area4']==1,Roadways4,0)
# df['max_Roadways'] = df['max_Roadways_1'] + df['max_Roadways_2'] + df['max_Roadways_3'] + df['max_Roadways_4']

# df_test_copy['max_Roadways_1'] = np.where(df_test_copy['Wilderness_Area1']==1,Roadways1_t,0)
# df_test_copy['max_Roadways_2'] = np.where(df_test_copy['Wilderness_Area2']==1,Roadways2_t,0)
# df_test_copy['max_Roadways_3'] = np.where(df_test_copy['Wilderness_Area3']==1,Roadways3_t,0)
# df_test_copy['max_Roadways_4'] = np.where(df_test_copy['Wilderness_Area4']==1,Roadways4_t,0)
# df_test_copy['max_Roadways'] = df_test_copy['max_Roadways_1'] + df_test_copy['max_Roadways_2'] + df_test_copy['max_Roadways_3'] + df_test_copy['max_Roadways_4']

# df['dist_to_area_max_Roadways'] = df['max_Roadways'] - df['Horizontal_Distance_To_Roadways']

# df_test_copy['dist_to_area_max_Roadways'] = df_test_copy['max_Roadways'] - df_test_copy['Horizontal_Distance_To_Roadways']

# df['pct_dist_to_area_max_Roadways'] = (df['max_Roadways'] - df['Horizontal_Distance_To_Roadways'])/df['max_Roadways']

# df_test_copy['pct_dist_to_area_max_Roadways'] = (df_test_copy['max_Roadways'] - df_test_copy['Horizontal_Distance_To_Roadways'])/df_test_copy['max_Roadways']

# df.drop(['max_Roadways_1', 'max_Roadways_2','max_Roadways_3','max_Roadways_4',
#         'max_Roadways'], axis=1, inplace=True)

# df_test_copy.drop(['max_Roadways_1', 'max_Roadways_2','max_Roadways_3','max_Roadways_4',
#                   'max_Roadways'], axis=1, inplace=True)


# In[ ]:


# # horizontal distance to to furthest tree from Fire_Points (min is at zero) and percentage difference

# Fire_Points1 = df[df['Wilderness_Area1']==1]['Horizontal_Distance_To_Fire_Points'].max()
# Fire_Points2 = df[df['Wilderness_Area2']==1]['Horizontal_Distance_To_Fire_Points'].max()
# Fire_Points3 = df[df['Wilderness_Area3']==1]['Horizontal_Distance_To_Fire_Points'].max()
# Fire_Points4 = df[df['Wilderness_Area4']==1]['Horizontal_Distance_To_Fire_Points'].max()

# Fire_Points1_t = df_test_copy[df_test_copy['Wilderness_Area1']==1]['Horizontal_Distance_To_Fire_Points'].max()
# Fire_Points2_t = df_test_copy[df_test_copy['Wilderness_Area2']==1]['Horizontal_Distance_To_Fire_Points'].max()
# Fire_Points3_t = df_test_copy[df_test_copy['Wilderness_Area3']==1]['Horizontal_Distance_To_Fire_Points'].max()
# Fire_Points4_t = df_test_copy[df_test_copy['Wilderness_Area4']==1]['Horizontal_Distance_To_Fire_Points'].max()

# df['max_Fire_Points_1'] = np.where(df['Wilderness_Area1']==1,Fire_Points1,0)
# df['max_Fire_Points_2'] = np.where(df['Wilderness_Area2']==1,Fire_Points2,0)
# df['max_Fire_Points_3'] = np.where(df['Wilderness_Area3']==1,Fire_Points3,0)
# df['max_Fire_Points_4'] = np.where(df['Wilderness_Area4']==1,Fire_Points4,0)
# df['max_Fire_Points'] = df['max_Fire_Points_1'] + df['max_Fire_Points_2'] + df['max_Fire_Points_3'] + df['max_Fire_Points_4']

# df_test_copy['max_Fire_Points_1'] = np.where(df_test_copy['Wilderness_Area1']==1,Fire_Points1_t,0)
# df_test_copy['max_Fire_Points_2'] = np.where(df_test_copy['Wilderness_Area2']==1,Fire_Points2_t,0)
# df_test_copy['max_Fire_Points_3'] = np.where(df_test_copy['Wilderness_Area3']==1,Fire_Points3_t,0)
# df_test_copy['max_Fire_Points_4'] = np.where(df_test_copy['Wilderness_Area4']==1,Fire_Points4_t,0)
# df_test_copy['max_Fire_Points'] = df_test_copy['max_Fire_Points_1'] + df_test_copy['max_Fire_Points_2'] + df_test_copy['max_Fire_Points_3'] + df_test_copy['max_Fire_Points_4']

# df['dist_to_area_max_Fire_Points'] = df['max_Fire_Points'] - df['Horizontal_Distance_To_Fire_Points']

# df_test_copy['dist_to_area_max_Fire_Points'] = df_test_copy['max_Fire_Points'] - df_test_copy['Horizontal_Distance_To_Fire_Points']

# df['pct_dist_to_area_max_Fire_Points'] = (df['max_Fire_Points'] - df['Horizontal_Distance_To_Fire_Points'])/df['max_Fire_Points']

# df_test_copy['pct_dist_to_area_max_Fire_Points'] = (df_test_copy['max_Fire_Points'] - df_test_copy['Horizontal_Distance_To_Fire_Points'])/df_test_copy['max_Fire_Points']

# df.drop(['max_Fire_Points_1', 'max_Fire_Points_2','max_Fire_Points_3','max_Fire_Points_4',
#         'max_Fire_Points'], axis=1, inplace=True)

# df_test_copy.drop(['max_Fire_Points_1', 'max_Fire_Points_2','max_Fire_Points_3','max_Fire_Points_4',
#                   'max_Fire_Points'], axis=1, inplace=True)


# In[ ]:


# squaring and logging the elevation

# df['elevation_sqd'] = df['Elevation']**2
# df['elevation_log'] = np.log1p(df['Elevation'])

# df_test_copy['elevation_sqd'] = df_test_copy['Elevation']**2
# df_test_copy['elevation_log'] = np.log1p(df_test_copy['Elevation'])


# In[ ]:


# looking at the slope and using sine and cosine

# #df['slope_sine'] =  np.sin(np.radians(df['Slope']))
df['slope_cosine'] = np.cos(np.radians(df['Slope']))

# #df_test_copy['slope_sine'] =  np.sin(np.radians(df_test_copy['Slope']))
df_test_copy['slope_cosine'] = np.cos(np.radians(df_test_copy['Slope']))


# In[ ]:


# # slope at elevation

# df['slope_elevation'] = df['slope_cosine'] * df['Elevation']
# df_test_copy['slope_elevation'] = df_test_copy['slope_cosine'] * df_test_copy['Elevation']


# In[ ]:


# looking at the aspect and using sine and cosine

df['aspect_sine'] =  np.sin(np.radians(df['Aspect']))
df['aspect_cosine'] = np.cos(np.radians(df['Aspect']))

df_test_copy['aspect_sine'] =  np.sin(np.radians(df_test_copy['Aspect']))
df_test_copy['aspect_cosine'] = np.cos(np.radians(df_test_copy['Aspect']))


# In[ ]:


# slope x aspect

# df['slope_x_aspect_cos_sin'] = df['slope_cosine'] * df['aspect_sine'] 
# df['slope_x_aspect_cos_cos'] = df['slope_cosine'] * df['aspect_cosine']
# df['slope_x_aspect_sin_sin'] = df['slope_sine'] * df['aspect_sine'] 
# df['slope_x_aspect_sin_cos'] = df['slope_sine'] * df['aspect_cosine']

# df_test_copy['slope_x_aspect_cos_sin'] = df_test_copy['slope_cosine'] * df_test_copy['aspect_sine'] 
# df_test_copy['slope_x_aspect_cos_cos'] = df_test_copy['slope_cosine'] * df_test_copy['aspect_cosine']
# df_test_copy['slope_x_aspect_sin_sin'] = df_test_copy['slope_sine'] * df_test_copy['aspect_sine'] 
# df_test_copy['slope_x_aspect_sin_cos'] = df_test_copy['slope_sine'] * df_test_copy['aspect_cosine']


# In[ ]:


# cosine of slope with sunlight at midday

df['midday_exposure'] = df['slope_cosine'] * df['Hillshade_Noon']
df_test_copy['midday_exposure'] = df_test_copy['slope_cosine'] * df_test_copy['Hillshade_Noon']


# In[ ]:


# aspect with total sunlight

# #df['light_exposure_aspect_sine'] = df['aspect_sine'] * df['Hillshade_total']
# df['light_exposure_aspect_cosine'] = df['aspect_cosine'] * df['Hillshade_total']
# #df_test_copy['light_exposure_aspect_sine'] = df_test_copy['aspect_sine'] * df_test_copy['Hillshade_total']
# df_test_copy['light_exposure_aspect_cosine'] = df_test_copy['aspect_cosine'] * df_test_copy['Hillshade_total']


# In[ ]:


#
# now dropping features i dont need any more
#


# In[ ]:


soil_list = []

for a in range(1,41):
    soil_list.append('Soil_Type' + str(a))

soil_list.remove('Soil_Type7')
soil_list.remove('Soil_Type15')

    
df.drop(soil_list, axis=1, inplace=True)
df_test_copy.drop(soil_list, axis=1, inplace=True)


# In[ ]:


df.drop('Aspect', axis=1, inplace=True)
df_test_copy.drop('Aspect', axis=1, inplace=True)


# In[ ]:


df.drop('aspect_cosine', axis=1, inplace=True)
df_test_copy.drop('aspect_cosine', axis=1, inplace=True)


# In[ ]:


df.drop('Hillshade_total', axis=1, inplace=True)
df_test_copy.drop('Hillshade_total', axis=1, inplace=True)


# In[ ]:





# In[ ]:


# df.sample(10)


# In[ ]:


# df_test_copy.sample(5)


# In[ ]:


# df.info()


# In[ ]:





# In[ ]:


#
# lets see if we can build a model for each wilderness area
# the combined 4 models for each wilderness areas will be the first model
#


# In[ ]:


# split the test data into wilderness areas

df_test_copy_w1 = df_test_copy[df_test_copy['Wilderness_Area1']==1]
df_test_copy_w2 = df_test_copy[df_test_copy['Wilderness_Area2']==1]
df_test_copy_w3 = df_test_copy[df_test_copy['Wilderness_Area3']==1]
df_test_copy_w4 = df_test_copy[df_test_copy['Wilderness_Area4']==1]

# and the training data (i needed to do this to add features to the wilderness areas)

df_train_copy_w1 = df[df['Wilderness_Area1']==1]
df_train_copy_w2 = df[df['Wilderness_Area2']==1]
df_train_copy_w3 = df[df['Wilderness_Area3']==1]
df_train_copy_w4 = df[df['Wilderness_Area4']==1]


# In[ ]:


# # drop some columns in certain wildnerness areas


df_test_copy_w1.drop(['dist_to_area_max_elevation', 'midday_exposure'], axis=1, inplace=True)

df_test_copy_w2.drop(['Soil_Elev_median', 'shade_morning_noon_ratio',
                      'shade_noon_afternoon_ratio', 'shade_morning_afternoon_ratio',
                     'dist_to_area_max_elevation', 'midday_exposure'], axis=1, inplace=True)

df_test_copy_w3.drop(['Elevation', 'shade_morning_noon_ratio',
                       'shade_noon_afternoon_ratio', 'shade_morning_afternoon_ratio'
                      ], axis=1, inplace=True)

df_test_copy_w4.drop(['shade_morning_noon_ratio','shade_noon_afternoon_ratio',
                      'shade_morning_afternoon_ratio','Soil_Elev_median',
                     'dist_to_area_max_elevation', 'midday_exposure'], axis=1, inplace=True)


df_train_copy_w1.drop(['dist_to_area_max_elevation', 'midday_exposure'], axis=1, inplace=True)

df_train_copy_w2.drop(['Soil_Elev_median', 'shade_morning_noon_ratio',
                      'shade_noon_afternoon_ratio', 'shade_morning_afternoon_ratio',
                      'dist_to_area_max_elevation', 'midday_exposure'], axis=1, inplace=True)

df_train_copy_w3.drop(['Elevation', 'shade_morning_noon_ratio',
                       'shade_noon_afternoon_ratio', 'shade_morning_afternoon_ratio'
                      ], axis=1, inplace=True)

df_train_copy_w4.drop(['shade_morning_noon_ratio','shade_noon_afternoon_ratio',
                      'shade_morning_afternoon_ratio','Soil_Elev_median',
                      'dist_to_area_max_elevation', 'midday_exposure'], axis=1, inplace=True)


# In[ ]:





# In[ ]:


# wilderness area 1


# In[ ]:


random_state = 0
                            
stack = StackingCVClassifier(classifiers=stack_w1,
                             meta_classifier=meta_clf_w1,
                             cv=5,
                             stratify=True,
                             shuffle=True,
                             use_probas=True,
                             use_features_in_secondary=True,
                             verbose=1,
                             random_state=random_state,
                             n_jobs=-1)


# In[ ]:


# i will try the GaussianMixture in the seperate wilderness areas now
df_GM = df_train_copy_w1.drop('Cover_Type', axis=1)


# In[ ]:


gm = GaussianMixture(n_components = 10)
gm.fit(df_test_copy_w1)


# In[ ]:


df_train_copy_w1['GaussianMixture'] = gm.predict(df_GM)
df_test_copy_w1['GaussianMixture'] = gm.predict(df_test_copy_w1)


# In[ ]:


# Scale & bin features

data_X = pd.concat([df_train_copy_w1.drop(['Id', 'Cover_Type','Wilderness_Area1'], axis=1, inplace=False),                     df_test_copy_w1.drop(['Id','Wilderness_Area1'], axis=1, inplace=False)])

data_X.loc[:, :] = np.floor(MinMaxScaler((0, 100)).fit_transform(data_X))
data_X = data_X.astype('int8')
    
df_train_copy_w1_scaled = data_X.iloc[: len(df_train_copy_w1), :]
df_test_copy_w1_scaled = data_X.iloc[len(df_train_copy_w1):, :]

df_train_copy_w1_scaled['Id'] = df_train_copy_w1['Id'].astype('int8')
df_train_copy_w1_scaled['Wilderness_Area1'] = df_train_copy_w1['Wilderness_Area1'].astype('int8')
df_train_copy_w1_scaled['Cover_Type'] = df_train_copy_w1['Cover_Type'].astype('int8')

df_test_copy_w1_scaled['Id'] = df_test_copy_w1['Id'].astype('int8')
df_test_copy_w1_scaled['Wilderness_Area1'] = df_test_copy_w1['Wilderness_Area1'].astype('int8')


# In[ ]:


for i in range(8):
    print(df_train_copy_w1[df_train_copy_w1['Cover_Type']==i].shape[0])


# In[ ]:


X = df_train_copy_w1_scaled.drop(['Cover_Type'], axis='columns')
y = df_train_copy_w1_scaled['Cover_Type']

max_samples = y.value_counts().iat[0]
classes = y.unique().tolist()
sampling_strategy = dict((clas, max_samples) for clas in classes)

sampler = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=random_state)

x_columns = X.columns.tolist()
X, y = sampler.fit_resample(X, y)
X = pd.DataFrame(X, columns=x_columns)
y = pd.Series(y)


df_train_copy_w1_scaled_ups = X
df_train_copy_w1_scaled_ups['Cover_Type'] = y


# In[ ]:


df_train_copy_w1_scaled_ups.head()


# In[ ]:


for i in range(8):
    print(df_train_copy_w1_scaled_ups[df_train_copy_w1_scaled_ups['Cover_Type']==i].shape[0])


# In[ ]:


# define X and y

feature_cols = [col for col in list(df_train_copy_w1_scaled_ups.columns) if col not in {'Id', 'Cover_Type'}] # i.e. all the columns except the Cover_Type and Id
X = df_train_copy_w1_scaled_ups[feature_cols]
y = df_train_copy_w1_scaled_ups['Cover_Type']

X.shape, y.shape


# In[ ]:


# firstly, i will try and see which features are important
# and then i can select the key features and go from there

clf_feat_imp = RandomForestClassifier(n_estimators=125,
                             min_samples_leaf=1,
                             max_depth=None,
                             verbose=0,
                             random_state=0)


# In[ ]:


# split X and y into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state)


# In[ ]:


importances = feature_importances(clf_feat_imp, X_train, y_train)


# In[ ]:


selected_features = select(importances, 0.0003)

X = X[selected_features]


# In[ ]:


# X.head()


# In[ ]:


# split X and y into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state)


# In[ ]:


X_test.head()


# In[ ]:


# remove outliers

iso_clf = IsolationForest(n_estimators=100, 
                          behaviour='new', 
                          max_samples=100,
                          contamination=0.01,
                          verbose=0,
                          random_state=random_state,
                          n_jobs=-1)

outliers = []
for cl in [1, 2, 3, 4, 5, 6, 7]:
    y_cl = y_train[y_train == cl]

    if not y_cl.empty:
        X_cl = X_train.loc[y_cl.index]

        iso_clf = iso_clf.fit(X_cl, y_cl)
        pred = iso_clf.predict(X_cl)
        outliers += y_cl[pred == -1].index.tolist()
    
if outliers:
    X_train = X_train.drop(outliers, axis='index')
    y_train = y_train.drop(outliers)


# In[ ]:


# training the models

stack = stack.fit(X_train, y_train)


# In[ ]:


# test the model

prediction = stack.predict(X_test)

pred_w1 = metrics.accuracy_score(y_test, prediction)

print(metrics.accuracy_score(y_test, prediction))


# In[ ]:


# # define X and y for the prediction
# #feature_cols_testData = [col for col in list(df_test_copy_w1.columns[1:]) if col not in {'Id','Cover_Type'}] # i.e. all the columns except the Cover_Type and Id

try:
    X_testData = df_test_copy_w1_scaled[selected_features]
    print('selected_features taken')
except:
    X_testData = df_test_copy_w1_scaled[feature_cols]
    print('feature_cols taken')


# In[ ]:


Cover_Type_prediction_w1 = stack.predict(X_testData)


# In[ ]:


df_test_copy_w1['Cover_Type'] = Cover_Type_prediction_w1


# In[ ]:


df_test_copy_w1.head()


# In[ ]:


# save the dataframes for keras use

keras_train_X_w1 = df_train_copy_w1_scaled_ups[selected_features].copy()
keras_train_y_w1 = df_train_copy_w1_scaled_ups['Cover_Type'].copy()
keras_test_w1 = X_testData.copy()
keras_test_w1['Id'] = df_test_copy_w1['Id']


# In[ ]:





# In[ ]:


# wilderness area 2


# In[ ]:


# preparing the models to use

random_state = 0

                             
stack = StackingCVClassifier(classifiers=stack_w2,
                             meta_classifier=meta_clf_w2,
                             cv=5,
                             stratify=True,
                             shuffle=True,
                             use_probas=True,
                             use_features_in_secondary=True,
                             verbose=1,
                             random_state=random_state,
                             n_jobs=-1)


# In[ ]:


# i will try the GaussianMixture in the seperate wilderness areas now
df_GM = df_train_copy_w2.drop('Cover_Type', axis=1)


# In[ ]:


gm = GaussianMixture(n_components = 10)
gm.fit(df_test_copy_w2)


# In[ ]:


df_train_copy_w2['GaussianMixture'] = gm.predict(df_GM)
df_test_copy_w2['GaussianMixture'] = gm.predict(df_test_copy_w2)


# In[ ]:


# Scale & bin features

data_X = pd.concat([df_train_copy_w2.drop(['Id', 'Cover_Type','Wilderness_Area2'], axis=1, inplace=False),                     df_test_copy_w2.drop(['Id','Wilderness_Area2'], axis=1, inplace=False)])

data_X.loc[:, :] = np.floor(MinMaxScaler((0, 100)).fit_transform(data_X))
data_X = data_X.astype('int8')
    
df_train_copy_w2_scaled = data_X.iloc[: len(df_train_copy_w2), :]
df_test_copy_w2_scaled = data_X.iloc[len(df_train_copy_w2):, :]

df_train_copy_w2_scaled['Id'] = df_train_copy_w2['Id'].astype('int8')
df_train_copy_w2_scaled['Wilderness_Area2'] = df_train_copy_w2['Wilderness_Area2'].astype('int8')
df_train_copy_w2_scaled['Cover_Type'] = df_train_copy_w2['Cover_Type'].astype('int8')

df_test_copy_w2_scaled['Id'] = df_test_copy_w2['Id'].astype('int8')
df_test_copy_w2_scaled['Wilderness_Area2'] = df_test_copy_w2['Wilderness_Area2'].astype('int8')


# In[ ]:


for i in range(8):
    print(df_train_copy_w2[df_train_copy_w2['Cover_Type']==i].shape[0])


# In[ ]:


X = df_train_copy_w2_scaled.drop(['Cover_Type'], axis='columns')
y = df_train_copy_w2_scaled['Cover_Type']

max_samples = y.value_counts().iat[0]
classes = y.unique().tolist()
sampling_strategy = dict((clas, max_samples) for clas in classes)

sampler = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=random_state)

x_columns = X.columns.tolist()
X, y = sampler.fit_resample(X, y)
X = pd.DataFrame(X, columns=x_columns)
y = pd.Series(y)


df_train_copy_w2_scaled_ups = X
df_train_copy_w2_scaled_ups['Cover_Type'] = y


# In[ ]:


df_train_copy_w2_scaled_ups.head()


# In[ ]:


for i in range(8):
    print(df_train_copy_w2_scaled_ups[df_train_copy_w2_scaled_ups['Cover_Type']==i].shape[0])


# In[ ]:


# define X and y

feature_cols = [col for col in list(df_train_copy_w2_scaled_ups.columns) if col not in {'Id', 'Cover_Type'}] # i.e. all the columns except the Cover_Type and Id
X = df_train_copy_w2_scaled_ups[feature_cols]
y = df_train_copy_w2_scaled_ups['Cover_Type']

X.shape, y.shape


# In[ ]:


# split X and y into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state)


# In[ ]:


importances = feature_importances(clf_feat_imp, X_train, y_train)


# In[ ]:


selected_features = select(importances, 0.0003)

X = X[selected_features]


# In[ ]:


X.head()


# In[ ]:


# split X and y into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


# In[ ]:


# training the models

stack = stack.fit(X_train, y_train)


# In[ ]:


# test the model

prediction = stack.predict(X_test)

pred_w2 = metrics.accuracy_score(y_test, prediction)

print(metrics.accuracy_score(y_test, prediction))


# In[ ]:


# # define X and y for the prediction
# #feature_cols_testData = [col for col in list(df_test_copy_w1.columns[1:]) if col not in {'Id','Cover_Type'}] # i.e. all the columns except the Cover_Type and Id

try:
    X_testData = df_test_copy_w2_scaled[selected_features]
    print('selected_features taken')
except:
    X_testData = df_test_copy_w2_scaled[feature_cols]
    print('feature_cols taken')


# In[ ]:


Cover_Type_prediction_w2 = stack.predict(X_testData)


# In[ ]:


df_test_copy_w2['Cover_Type'] = Cover_Type_prediction_w2


# In[ ]:


df_test_copy_w2.head()


# In[ ]:


# save the dataframes for keras

keras_train_X_w2 = df_train_copy_w2_scaled_ups[selected_features].copy()
keras_train_y_w2 = df_train_copy_w2_scaled_ups['Cover_Type'].copy()
keras_test_w2 = X_testData.copy()
keras_test_w2['Id'] = df_test_copy_w2['Id']


# In[ ]:


# wilderness area 3


# In[ ]:


# preparing the models to use

random_state = 0
                             
stack = StackingCVClassifier(classifiers=stack_w3,
                             meta_classifier=meta_clf_w3,
                             cv=5,
                             stratify=True,
                             shuffle=True,
                             use_probas=True,
                             use_features_in_secondary=True,
                             verbose=1,
                             random_state=random_state,
                             n_jobs=-1)


# In[ ]:


# i will try the GaussianMixture in the seperate wilderness areas now
df_GM = df_train_copy_w3.drop('Cover_Type', axis=1)


# In[ ]:


gm = GaussianMixture(n_components = 10)
gm.fit(df_test_copy_w3)


# In[ ]:


df_train_copy_w3['GaussianMixture'] = gm.predict(df_GM)
df_test_copy_w3['GaussianMixture'] = gm.predict(df_test_copy_w3)


# In[ ]:


# Scale & bin features

data_X = pd.concat([df_train_copy_w3.drop(['Id', 'Cover_Type','Wilderness_Area3'], axis=1, inplace=False),                     df_test_copy_w3.drop(['Id','Wilderness_Area3'], axis=1, inplace=False)])

data_X.loc[:, :] = np.floor(MinMaxScaler((0, 100)).fit_transform(data_X))
data_X = data_X.astype('int8')
    
df_train_copy_w3_scaled = data_X.iloc[: len(df_train_copy_w3), :]
df_test_copy_w3_scaled = data_X.iloc[len(df_train_copy_w3):, :]

df_train_copy_w3_scaled['Id'] = df_train_copy_w3['Id'].astype('int8')
df_train_copy_w3_scaled['Wilderness_Area3'] = df_train_copy_w3['Wilderness_Area3'].astype('int8')
df_train_copy_w3_scaled['Cover_Type'] = df_train_copy_w3['Cover_Type'].astype('int8')

df_test_copy_w3_scaled['Id'] = df_test_copy_w3['Id'].astype('int8')
df_test_copy_w3_scaled['Wilderness_Area3'] = df_test_copy_w3['Wilderness_Area3'].astype('int8')


# In[ ]:


for i in range(8):
    print(df_train_copy_w3[df_train_copy_w3['Cover_Type']==i].shape[0])


# In[ ]:


X = df_train_copy_w3_scaled.drop(['Cover_Type'], axis='columns')
y = df_train_copy_w3_scaled['Cover_Type']

max_samples = y.value_counts().iat[0]
classes = y.unique().tolist()
sampling_strategy = dict((clas, max_samples) for clas in classes)

sampler = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=random_state)

x_columns = X.columns.tolist()
X, y = sampler.fit_resample(X, y)
X = pd.DataFrame(X, columns=x_columns)
y = pd.Series(y)


df_train_copy_w3_scaled_ups = X
df_train_copy_w3_scaled_ups['Cover_Type'] = y


# In[ ]:


df_train_copy_w3_scaled_ups.head()


# In[ ]:


for i in range(8):
    print(df_train_copy_w3_scaled_ups[df_train_copy_w3_scaled_ups['Cover_Type']==i].shape[0])


# In[ ]:


# define X and y

feature_cols = [col for col in list(df_train_copy_w3_scaled_ups.columns) if col not in {'Id', 'Cover_Type'}] # i.e. all the columns except the Cover_Type and Id
X = df_train_copy_w3_scaled_ups[feature_cols]
y = df_train_copy_w3_scaled_ups['Cover_Type']

X.shape, y.shape


# In[ ]:


# split X and y into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state)


# In[ ]:


importances = feature_importances(clf_feat_imp, X_train, y_train)


# In[ ]:


selected_features = select(importances, 0.0003)

X = X[selected_features]


# In[ ]:


X.head()


# In[ ]:


# split X and y into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


# In[ ]:


# remove outliers

iso_clf = IsolationForest(n_estimators=100, 
                          behaviour='new', 
                          max_samples=100,
                          contamination=0.01,
                          verbose=0,
                          random_state=random_state,
                          n_jobs=-1)

outliers = []
for cl in [1, 2, 3, 4, 5, 6, 7]:
    y_cl = y_train[y_train == cl]

    if not y_cl.empty:
        X_cl = X_train.loc[y_cl.index]

        iso_clf = iso_clf.fit(X_cl, y_cl)
        pred = iso_clf.predict(X_cl)
        outliers += y_cl[pred == -1].index.tolist()
    
if outliers:
    X_train = X_train.drop(outliers, axis='index')
    y_train = y_train.drop(outliers)


# In[ ]:


# training the models

stack = stack.fit(X_train, y_train)


# In[ ]:


# test the model

prediction = stack.predict(X_test)

pred_w3 = metrics.accuracy_score(y_test, prediction)

print(metrics.accuracy_score(y_test, prediction))


# In[ ]:


# # define X and y for the prediction
# #feature_cols_testData = [col for col in list(df_test_copy_w1.columns[1:]) if col not in {'Id','Cover_Type'}] # i.e. all the columns except the Cover_Type and Id

try:
    X_testData = df_test_copy_w3_scaled[selected_features]
    print('selected_features taken')
except:
    X_testData = df_test_copy_w3_scaled[feature_cols]
    print('feature_cols taken')


# In[ ]:


Cover_Type_prediction_w3 = stack.predict(X_testData)


# In[ ]:


df_test_copy_w3['Cover_Type'] = Cover_Type_prediction_w3


# In[ ]:


# df_test_copy_w3.head()


# In[ ]:


# save the dataframes for keras use

keras_train_X_w3 = df_train_copy_w3_scaled_ups[selected_features].copy()
keras_train_y_w3 = df_train_copy_w3_scaled_ups['Cover_Type'].copy()
keras_test_w3 = X_testData.copy()
keras_test_w3['Id'] = df_test_copy_w3['Id']


# In[ ]:


# wilderness area 4


# In[ ]:


# preparing the models to use

random_state = 0

stack = StackingCVClassifier(classifiers=stack_w4,
                             meta_classifier=meta_clf_w4,
                             cv=5,
                             stratify=True,
                             shuffle=True,
                             use_probas=True,
                             use_features_in_secondary=True,
                             verbose=1,
                             random_state=random_state,
                             n_jobs=-1)


# In[ ]:


# i will try the GaussianMixture in the seperate wilderness areas now
df_GM = df_train_copy_w4.drop('Cover_Type', axis=1)


# In[ ]:


gm = GaussianMixture(n_components = 10)
gm.fit(df_test_copy_w4)


# In[ ]:


df_train_copy_w4['GaussianMixture'] = gm.predict(df_GM)
df_test_copy_w4['GaussianMixture'] = gm.predict(df_test_copy_w4)


# In[ ]:


# Scale & bin features

data_X = pd.concat([df_train_copy_w4.drop(['Id', 'Cover_Type','Wilderness_Area4'], axis=1, inplace=False),                     df_test_copy_w4.drop(['Id','Wilderness_Area4'], axis=1, inplace=False)])

data_X.loc[:, :] = np.floor(MinMaxScaler((0, 100)).fit_transform(data_X))
data_X = data_X.astype('int8')
    
df_train_copy_w4_scaled = data_X.iloc[: len(df_train_copy_w4), :]
df_test_copy_w4_scaled = data_X.iloc[len(df_train_copy_w4):, :]

df_train_copy_w4_scaled['Id'] = df_train_copy_w4['Id'].astype('int8')
df_train_copy_w4_scaled['Wilderness_Area4'] = df_train_copy_w4['Wilderness_Area4'].astype('int8')
df_train_copy_w4_scaled['Cover_Type'] = df_train_copy_w4['Cover_Type'].astype('int8')

df_test_copy_w4_scaled['Id'] = df_test_copy_w4['Id'].astype('int8')
df_test_copy_w4_scaled['Wilderness_Area4'] = df_test_copy_w4['Wilderness_Area4'].astype('int8')


# In[ ]:


for i in range(8):
    print(df_train_copy_w4[df_train_copy_w4['Cover_Type']==i].shape[0])


# In[ ]:


X = df_train_copy_w4_scaled.drop(['Cover_Type'], axis='columns')
y = df_train_copy_w4_scaled['Cover_Type']

max_samples = y.value_counts().iat[0]
classes = y.unique().tolist()
sampling_strategy = dict((clas, max_samples) for clas in classes)

sampler = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=random_state)

x_columns = X.columns.tolist()
X, y = sampler.fit_resample(X, y)
X = pd.DataFrame(X, columns=x_columns)
y = pd.Series(y)


df_train_copy_w4_scaled_ups = X
df_train_copy_w4_scaled_ups['Cover_Type'] = y


# In[ ]:


df_train_copy_w4_scaled_ups.head()


# In[ ]:


for i in range(8):
    print(df_train_copy_w4_scaled_ups[df_train_copy_w4_scaled_ups['Cover_Type']==i].shape[0])


# In[ ]:


# define X and y

feature_cols = [col for col in list(df_train_copy_w4_scaled_ups.columns) if col not in {'Id', 'Cover_Type'}] # i.e. all the columns except the Cover_Type and Id
X = df_train_copy_w4_scaled_ups[feature_cols]
y = df_train_copy_w4_scaled_ups['Cover_Type']

X.shape, y.shape


# In[ ]:


# split X and y into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state)


# In[ ]:


importances = feature_importances(clf_feat_imp, X_train, y_train)


# In[ ]:


selected_features = select(importances, 0.0003)

X = X[selected_features]


# In[ ]:


X.head()


# In[ ]:


# split X and y into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


# In[ ]:


# remove outliers

iso_clf = IsolationForest(n_estimators=100, 
                          behaviour='new', 
                          max_samples=100,
                          contamination=0.01,
                          verbose=0,
                          random_state=random_state,
                          n_jobs=-1)

outliers = []
for cl in [1, 2, 3, 4, 5, 6, 7]:
    y_cl = y_train[y_train == cl]

    if not y_cl.empty:
        X_cl = X_train.loc[y_cl.index]

        iso_clf = iso_clf.fit(X_cl, y_cl)
        pred = iso_clf.predict(X_cl)
        outliers += y_cl[pred == -1].index.tolist()
    
if outliers:
    X_train = X_train.drop(outliers, axis='index')
    y_train = y_train.drop(outliers)


# In[ ]:


# training the models

stack = stack.fit(X_train, y_train)


# In[ ]:


# test the model

prediction = stack.predict(X_test)

pred_w4 = metrics.accuracy_score(y_test, prediction)

print(metrics.accuracy_score(y_test, prediction))


# In[ ]:


# # define X and y for the prediction
# #feature_cols_testData = [col for col in list(df_test_copy_w1.columns[1:]) if col not in {'Id','Cover_Type'}] # i.e. all the columns except the Cover_Type and Id

try:
    X_testData = df_test_copy_w4_scaled[selected_features]
    print('selected_features taken')
except:
    X_testData = df_test_copy_w4_scaled[feature_cols]
    print('feature_cols taken')


# In[ ]:


Cover_Type_prediction_w4 = stack.predict(X_testData)


# In[ ]:


df_test_copy_w4['Cover_Type'] = Cover_Type_prediction_w4


# In[ ]:


# df_test_copy_w4.head()


# In[ ]:


# save the dataframes for keras

keras_train_X_w4 = df_train_copy_w4_scaled_ups[selected_features].copy()
keras_train_y_w4 = df_train_copy_w4_scaled_ups['Cover_Type'].copy()
keras_test_w4 = X_testData.copy()
keras_test_w4['Id'] = df_test_copy_w4['Id']


# In[ ]:


# create the submission

frames = [df_test_copy_w1,df_test_copy_w2,df_test_copy_w3,df_test_copy_w4]
df_submission = pd.concat(frames)
df_submission=df_submission[['Id','Cover_Type']].copy()
df_submission.to_csv(sub_1, index=False)


# In[ ]:





# In[ ]:


#
# use keras on each of the wilderness areas
# the combined for each wilderness area will be the second model
#


# In[ ]:


# wilderness area 1


# In[ ]:


#Define Keras sequential classifier
clf_krs = keras.Sequential()


# In[ ]:


# Adding the input layer and the first hidden layer
clf_krs.add(layers.Dense(units = 1024, kernel_initializer='Orthogonal', activation = 'relu', input_dim=(len(keras_train_X_w1.columns))))
clf_krs.add(layers.Dropout(rate = 0.1))

# Adding the second hidden layer
clf_krs.add(layers.Dense(units = 512, kernel_initializer= 'Orthogonal', activation = 'relu'))
clf_krs.add(layers.Dropout(rate = 0.1))

# Adding the third hidden layer
clf_krs.add(layers.Dense(units = 256, kernel_initializer= 'Orthogonal', activation = 'relu'))
clf_krs.add(layers.Dropout(rate = 0.1))

# Adding the fourth hidden layer
clf_krs.add(layers.Dense(units = 128, kernel_initializer= 'Orthogonal', activation = 'relu'))
clf_krs.add(layers.Dropout(rate = 0.1))

# Adding the fifth hidden layer
clf_krs.add(layers.Dense(units = 64, kernel_initializer= 'Orthogonal', activation = 'relu'))
clf_krs.add(layers.Dropout(rate = 0.1))

# Adding the sixth hidden layer
clf_krs.add(layers.Dense(units = 32, kernel_initializer= 'Orthogonal', activation = 'relu'))
clf_krs.add(layers.Dropout(rate = 0.1))

# Adding the seventh hidden layer
clf_krs.add(layers.Dense(units = 16, kernel_initializer= 'Orthogonal', activation = 'relu'))
clf_krs.add(layers.Dropout(rate = 0.1))

# Adding the output layer
clf_krs.add(layers.Dense(units = 8, kernel_initializer= 'Orthogonal', activation = 'softmax'))


# In[ ]:


#summary report of classifier we have just built
print("Summary report of Keras classifier:") 
clf_krs.summary()


# In[ ]:


clf_krs.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])


# In[ ]:


keras_result = clf_krs.fit(keras_train_X_w1,
                           keras_train_y_w1,
                           epochs=200,
                           batch_size=32)


# In[ ]:


# accuracy of training model
np.mean(keras_result.history['accuracy'])


# In[ ]:


X_testData = keras_test_w1.drop('Id', axis=1)


# In[ ]:


Cover_Type_prediction_keras = clf_krs.predict_classes(X_testData)


# In[ ]:


keras_test_w1['Cover_type_keras'] = Cover_Type_prediction_keras


# In[ ]:


keras_test_w1.head()


# In[ ]:





# In[ ]:


# wilderness area 2


# In[ ]:


#Define Keras sequential classifier
clf_krs = keras.Sequential()


# In[ ]:


# Adding the input layer and the first hidden layer
clf_krs.add(layers.Dense(units = 1024, kernel_initializer='Orthogonal', activation = 'relu', input_dim=(len(keras_train_X_w2.columns))))
clf_krs.add(layers.Dropout(rate = 0.1))

# Adding the second hidden layer
clf_krs.add(layers.Dense(units = 512, kernel_initializer= 'Orthogonal', activation = 'relu'))
clf_krs.add(layers.Dropout(rate = 0.1))

# Adding the third hidden layer
clf_krs.add(layers.Dense(units = 256, kernel_initializer= 'Orthogonal', activation = 'relu'))
clf_krs.add(layers.Dropout(rate = 0.1))

# Adding the fourth hidden layer
clf_krs.add(layers.Dense(units = 128, kernel_initializer= 'Orthogonal', activation = 'relu'))
clf_krs.add(layers.Dropout(rate = 0.1))

# Adding the fifth hidden layer
clf_krs.add(layers.Dense(units = 64, kernel_initializer= 'Orthogonal', activation = 'relu'))
clf_krs.add(layers.Dropout(rate = 0.1))

# Adding the sixth hidden layer
clf_krs.add(layers.Dense(units = 32, kernel_initializer= 'Orthogonal', activation = 'relu'))
clf_krs.add(layers.Dropout(rate = 0.1))

# Adding the seventh hidden layer
clf_krs.add(layers.Dense(units = 16, kernel_initializer= 'Orthogonal', activation = 'relu'))
clf_krs.add(layers.Dropout(rate = 0.1))

# Adding the output layer
clf_krs.add(layers.Dense(units = 8, kernel_initializer= 'Orthogonal', activation = 'softmax'))


# In[ ]:


#summary report of classifier we have just built
print("Summary report of Keras classifier:") 
clf_krs.summary()


# In[ ]:


clf_krs.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])


# In[ ]:


keras_result = clf_krs.fit(keras_train_X_w2,
                           keras_train_y_w2,
                           epochs=200,
                           batch_size=32)


# In[ ]:


# accuracy of training model
np.mean(keras_result.history['accuracy'])


# In[ ]:


X_testData = keras_test_w2.drop('Id', axis=1)


# In[ ]:


Cover_Type_prediction_keras = clf_krs.predict_classes(X_testData)


# In[ ]:


keras_test_w2['Cover_type_keras'] = Cover_Type_prediction_keras


# In[ ]:


keras_test_w2.head()


# In[ ]:





# In[ ]:


# wilderness area 3


# In[ ]:


#Define Keras sequential classifier
clf_krs = keras.Sequential()


# In[ ]:


# Adding the input layer and the first hidden layer
clf_krs.add(layers.Dense(units = 1024, kernel_initializer='Orthogonal', activation = 'relu', input_dim=(len(keras_train_X_w3.columns))))
clf_krs.add(layers.Dropout(rate = 0.1))

# Adding the second hidden layer
clf_krs.add(layers.Dense(units = 512, kernel_initializer= 'Orthogonal', activation = 'relu'))
clf_krs.add(layers.Dropout(rate = 0.1))

# Adding the third hidden layer
clf_krs.add(layers.Dense(units = 256, kernel_initializer= 'Orthogonal', activation = 'relu'))
clf_krs.add(layers.Dropout(rate = 0.1))

# Adding the fourth hidden layer
clf_krs.add(layers.Dense(units = 128, kernel_initializer= 'Orthogonal', activation = 'relu'))
clf_krs.add(layers.Dropout(rate = 0.1))

# Adding the fifth hidden layer
clf_krs.add(layers.Dense(units = 64, kernel_initializer= 'Orthogonal', activation = 'relu'))
clf_krs.add(layers.Dropout(rate = 0.1))

# Adding the sixth hidden layer
clf_krs.add(layers.Dense(units = 32, kernel_initializer= 'Orthogonal', activation = 'relu'))
clf_krs.add(layers.Dropout(rate = 0.1))

# Adding the seventh hidden layer
clf_krs.add(layers.Dense(units = 16, kernel_initializer= 'Orthogonal', activation = 'relu'))
clf_krs.add(layers.Dropout(rate = 0.1))

# Adding the output layer
clf_krs.add(layers.Dense(units = 8, kernel_initializer= 'Orthogonal', activation = 'softmax'))


# In[ ]:


#summary report of classifier we have just built
print("Summary report of Keras classifier:") 
clf_krs.summary()


# In[ ]:


clf_krs.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])


# In[ ]:


keras_result = clf_krs.fit(keras_train_X_w3,
                           keras_train_y_w3,
                           epochs=200,
                           batch_size=32)


# In[ ]:


# accuracy of training model
np.mean(keras_result.history['accuracy'])


# In[ ]:


X_testData = keras_test_w3.drop('Id', axis=1)


# In[ ]:


Cover_Type_prediction_keras = clf_krs.predict_classes(X_testData)


# In[ ]:


keras_test_w3['Cover_type_keras'] = Cover_Type_prediction_keras


# In[ ]:


keras_test_w3.head()


# In[ ]:





# In[ ]:


# wilderness area 4


# In[ ]:


#Define Keras sequential classifier
clf_krs = keras.Sequential()


# In[ ]:


# Adding the input layer and the first hidden layer
clf_krs.add(layers.Dense(units = 1024, kernel_initializer='Orthogonal', activation = 'relu', input_dim=(len(keras_train_X_w4.columns))))
clf_krs.add(layers.Dropout(rate = 0.1))

# Adding the second hidden layer
clf_krs.add(layers.Dense(units = 512, kernel_initializer= 'Orthogonal', activation = 'relu'))
clf_krs.add(layers.Dropout(rate = 0.1))

# Adding the third hidden layer
clf_krs.add(layers.Dense(units = 256, kernel_initializer= 'Orthogonal', activation = 'relu'))
clf_krs.add(layers.Dropout(rate = 0.1))

# Adding the fourth hidden layer
clf_krs.add(layers.Dense(units = 128, kernel_initializer= 'Orthogonal', activation = 'relu'))
clf_krs.add(layers.Dropout(rate = 0.1))

# Adding the fifth hidden layer
clf_krs.add(layers.Dense(units = 64, kernel_initializer= 'Orthogonal', activation = 'relu'))
clf_krs.add(layers.Dropout(rate = 0.1))

# Adding the sixth hidden layer
clf_krs.add(layers.Dense(units = 32, kernel_initializer= 'Orthogonal', activation = 'relu'))
clf_krs.add(layers.Dropout(rate = 0.1))

# Adding the seventh hidden layer
clf_krs.add(layers.Dense(units = 16, kernel_initializer= 'Orthogonal', activation = 'relu'))
clf_krs.add(layers.Dropout(rate = 0.1))

# Adding the output layer
clf_krs.add(layers.Dense(units = 8, kernel_initializer= 'Orthogonal', activation = 'softmax'))


# In[ ]:


#summary report of classifier we have just built
print("Summary report of Keras classifier:") 
clf_krs.summary()


# In[ ]:


clf_krs.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])


# In[ ]:


keras_result = clf_krs.fit(keras_train_X_w4,
                           keras_train_y_w4,
                           epochs=200,
                           batch_size=32)


# In[ ]:


# accuracy of training model
np.mean(keras_result.history['accuracy'])


# In[ ]:


X_testData = keras_test_w4.drop('Id', axis=1)


# In[ ]:


Cover_Type_prediction_keras = clf_krs.predict_classes(X_testData)


# In[ ]:


keras_test_w4['Cover_type_keras'] = Cover_Type_prediction_keras


# In[ ]:


keras_test_w4.head()


# In[ ]:


frames = [keras_test_w1,keras_test_w2,keras_test_w3,keras_test_w4]
df_submission = pd.concat(frames)
df_submission=df_submission[['Id','Cover_type_keras']].copy()
df_test_copy_keras = df_submission[['Id','Cover_type_keras']].copy()

df_submission.to_csv(sub_1a, index=False)


# In[ ]:





# In[ ]:


#
# for all wilderness areas
# this is the third models
#


# In[ ]:


# preparing the models to use

random_state = 0
                             
stack = StackingCVClassifier(classifiers=stack_all,
                             meta_classifier=meta_clf_all,
                             cv=5,
                             stratify=True,
                             shuffle=True,
                             use_probas=True,
                             use_features_in_secondary=True,
                             verbose=1,
                             random_state=random_state,
                             n_jobs=-1)


# In[ ]:


# i will try the GaussianMixture in the seperate wilderness areas now
df_GM = df.drop('Cover_Type', axis=1)


# In[ ]:


gm = GaussianMixture(n_components = 10)
gm.fit(df_test_copy)


# In[ ]:


df['GaussianMixture'] = gm.predict(df_GM)
df_test_copy['GaussianMixture'] = gm.predict(df_test_copy)


# In[ ]:


# Scale & bin features

data_X = pd.concat([df.drop(['Id', 'Cover_Type','Wilderness_Area1','Wilderness_Area2','Wilderness_Area3','Wilderness_Area4'], axis=1, inplace=False),                     df_test_copy.drop(['Id','Wilderness_Area1','Wilderness_Area2','Wilderness_Area3','Wilderness_Area4'], axis=1, inplace=False)])

data_X.loc[:, :] = np.floor(MinMaxScaler((0, 100)).fit_transform(data_X))
data_X = data_X.astype('int8')
    
df_scaled = data_X.iloc[: len(df), :]
df_test_copy_scaled = data_X.iloc[len(df):, :]

df_scaled['Id'] = df['Id'].astype('int8')
df_scaled['Wilderness_Area1'] = df['Wilderness_Area1'].astype('int8')
df_scaled['Wilderness_Area2'] = df['Wilderness_Area2'].astype('int8')
df_scaled['Wilderness_Area3'] = df['Wilderness_Area3'].astype('int8')
df_scaled['Wilderness_Area4'] = df['Wilderness_Area4'].astype('int8')
df_scaled['Cover_Type'] = df['Cover_Type'].astype('int8')

df_test_copy_scaled['Id'] = df_test_copy['Id'].astype('int8')
df_test_copy_scaled['Wilderness_Area1'] = df_test_copy['Wilderness_Area1'].astype('int8')
df_test_copy_scaled['Wilderness_Area2'] = df_test_copy['Wilderness_Area2'].astype('int8')
df_test_copy_scaled['Wilderness_Area3'] = df_test_copy['Wilderness_Area3'].astype('int8')
df_test_copy_scaled['Wilderness_Area4'] = df_test_copy['Wilderness_Area4'].astype('int8')


# In[ ]:


for i in range(8):
    print(df[df['Cover_Type']==i].shape[0])


# In[ ]:


X = df_scaled.drop(['Cover_Type'], axis='columns')
y = df_scaled['Cover_Type']

max_samples = y.value_counts().iat[0]
classes = y.unique().tolist()
sampling_strategy = dict((clas, max_samples) for clas in classes)

sampler = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=random_state)

x_columns = X.columns.tolist()
X, y = sampler.fit_resample(X, y)
X = pd.DataFrame(X, columns=x_columns)
y = pd.Series(y)


df_scaled_ups = X
df_scaled_ups['Cover_Type'] = y


# In[ ]:


df_scaled_ups.head()


# In[ ]:


for i in range(8):
    print(df_scaled_ups[df_scaled_ups['Cover_Type']==i].shape[0])


# In[ ]:


# define X and y

feature_cols = [col for col in list(df_scaled_ups.columns) if col not in {'Id', 'Cover_Type'}] # i.e. all the columns except the Cover_Type and Id
X = df_scaled_ups[feature_cols]
y = df_scaled_ups['Cover_Type']

X.shape, y.shape


# In[ ]:


# split X and y into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state)


# In[ ]:


importances = feature_importances(clf_feat_imp, X_train, y_train)


# In[ ]:


selected_features = select(importances, 0.0003)

X = X[selected_features]


# In[ ]:


# X.head()


# In[ ]:


# split X and y into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


# In[ ]:


# remove outliers

iso_clf = IsolationForest(n_estimators=100, 
                          behaviour='new', 
                          max_samples=100,
                          contamination=0.01,
                          verbose=0,
                          random_state=random_state,
                          n_jobs=-1)

outliers = []
for cl in [1, 2, 3, 4, 5, 6, 7]:
    y_cl = y_train[y_train == cl]

    if not y_cl.empty:
        X_cl = X_train.loc[y_cl.index]

        iso_clf = iso_clf.fit(X_cl, y_cl)
        pred = iso_clf.predict(X_cl)
        outliers += y_cl[pred == -1].index.tolist()
    
if outliers:
    X_train = X_train.drop(outliers, axis='index')
    y_train = y_train.drop(outliers)


# In[ ]:


# training the models

stack = stack.fit(X_train, y_train)


# In[ ]:


# test the model

prediction = stack.predict(X_test)

pred_All = metrics.accuracy_score(y_test, prediction)

print(metrics.accuracy_score(y_test, prediction))


# In[ ]:


# # define X and y for the prediction
# #feature_cols_testData = [col for col in list(df_test_copy_w1.columns[1:]) if col not in {'Id','Cover_Type'}] # i.e. all the columns except the Cover_Type and Id

try:
    X_testData = df_test_copy_scaled[selected_features]
    print('selected_features taken')
except:
    X_testData = df_test_copy_scaled[feature_cols]
    print('feature_cols taken')


# In[ ]:


Cover_Type_prediction_wAll = stack.predict(X_testData)


# In[ ]:


df_test_copy['Cover_Type'] = Cover_Type_prediction_wAll
#df_test_copy_w4.head()


# In[ ]:


df_test_copy.head()


# In[ ]:


df_submission=df_test_copy[['Id','Cover_Type']].copy()
df_test_copy_all = df_test_copy[['Id','Cover_Type']].copy()


# In[ ]:


df_test_copy_comp = df_test_copy.copy()
df_test_copy_comp['Cover_type_all'] = df_test_copy_comp['Cover_Type']
df_test_copy_comp.drop(['Cover_Type'], axis=1, inplace=True)

df_test_copy_comp.head()


# In[ ]:


# for the non-keras models, summarising the model accuracy
pred_w1, pred_w2, pred_w3, pred_w4, pred_All


# In[ ]:





# In[ ]:


# frames = [df_test_copy_w1,df_test_copy_w2,df_test_copy_w3,df_test_copy_w4]
# df_submission = pd.concat(frames)
# df_submission=df_submission[['Id','Cover_Type']].copy()


# In[ ]:


frames = [df_test_copy_w1,df_test_copy_w2,df_test_copy_w3,df_test_copy_w4]
df_submission = pd.concat(frames)

# will need this to determine which models are giving the same answer
df_test_copy_1to4 = df_submission.copy()
df_test_copy_1to4['Cover_type_1to4'] = df_test_copy_1to4['Cover_Type']
df_test_copy_1to4.drop(['Cover_Type'], axis=1, inplace=True)

# df_test_copy_1to4.head()


# In[ ]:


# df_submission=df_submission[['Id','Cover_Type']].copy()


# In[ ]:


#df_submission['Id'] = df_submission.index
# df_submission.head()


# In[ ]:


# commented out because already submitted
#df_submission.to_csv('submission20191014_1.csv', index=False)


# In[ ]:


# also for the whole forest approach
df_submission = df_test_copy_comp[['Id','Cover_type_all']].copy()
df_submission = df_submission.rename({'Cover_type_all': 'Cover_type'}, axis=1) 


# In[ ]:


# this is the answers for the entire forest model
df_submission.to_csv(sub_2, index=False)


# In[ ]:





# In[ ]:


#
# bringing the results for all areas and individual areas together
#


# In[ ]:


df_test_copy_comp['Cover_type_1to4'] = df_test_copy_1to4['Cover_type_1to4']
df_test_copy_comp['Cover_type_keras'] = df_test_copy_keras['Cover_type_keras']
# df_test_copy_comp.head()


# In[ ]:


#
# using keras on the entire forest
# this is the fourth model
#


# In[ ]:


#Define Keras sequential classifier
clf_krs = keras.Sequential()


# In[ ]:


# #input layer
# clf_krs.add(layers.Dense(128,activation='relu',input_shape=(len(selected_features),)))

# #hidden layer 1
# clf_krs.add(layers.Dense(64,activation='relu'))

# #hidden layer 2
# clf_krs.add(layers.Dense(32,activation='relu'))

# #output layer
# clf_krs.add(layers.Dense(8,activation='softmax'))


# In[ ]:


# Adding the input layer and the first hidden layer
clf_krs.add(layers.Dense(units = 1024, kernel_initializer='Orthogonal', activation = 'relu', input_dim=(len(selected_features))))
clf_krs.add(layers.Dropout(rate = 0.1))

# Adding the second hidden layer
clf_krs.add(layers.Dense(units = 512, kernel_initializer= 'Orthogonal', activation = 'relu'))
clf_krs.add(layers.Dropout(rate = 0.1))

# Adding the third hidden layer
clf_krs.add(layers.Dense(units = 256, kernel_initializer= 'Orthogonal', activation = 'relu'))
clf_krs.add(layers.Dropout(rate = 0.1))

# Adding the fourth hidden layer
clf_krs.add(layers.Dense(units = 128, kernel_initializer= 'Orthogonal', activation = 'relu'))
clf_krs.add(layers.Dropout(rate = 0.1))

# Adding the fifth hidden layer
clf_krs.add(layers.Dense(units = 64, kernel_initializer= 'Orthogonal', activation = 'relu'))
clf_krs.add(layers.Dropout(rate = 0.1))

# Adding the sixth hidden layer
clf_krs.add(layers.Dense(units = 32, kernel_initializer= 'Orthogonal', activation = 'relu'))
clf_krs.add(layers.Dropout(rate = 0.1))

# Adding the seventh hidden layer
clf_krs.add(layers.Dense(units = 16, kernel_initializer= 'Orthogonal', activation = 'relu'))
clf_krs.add(layers.Dropout(rate = 0.1))

# Adding the output layer
clf_krs.add(layers.Dense(units = 8, kernel_initializer= 'Orthogonal', activation = 'softmax'))


# In[ ]:


#summary report of classifier we have just built
print("Summary report of Keras classifier:") 
clf_krs.summary()


# In[ ]:


clf_krs.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])


# In[ ]:


keras_result = clf_krs.fit(df_scaled_ups[selected_features],
                           df_scaled_ups['Cover_Type'],
                           epochs=1000,
                           batch_size=1000)


# In[ ]:


# accuracy of training model
np.mean(keras_result.history['accuracy'])


# In[ ]:


# using from the second model

try:
    X_testData = df_test_copy_scaled[selected_features]
    print('selected_features taken')
except:
    X_testData = df_test_copy_scaled[feature_cols]
    print('feature_cols taken')


# In[ ]:


Cover_Type_prediction_keras = clf_krs.predict_classes(X_testData)


# In[ ]:


df_test_copy_comp['Cover_type_keras_all'] = Cover_Type_prediction_keras


# In[ ]:


df_test_copy_comp.head()


# In[ ]:


df_submission = df_test_copy_comp[['Id','Cover_type_keras_all']].copy()
df_submission = df_submission.rename({'Cover_type_keras_all': 'Cover_type'}, axis=1) 


# In[ ]:


# generated the fourth mdoel csv
df_submission.to_csv(sub_2a, index=False)


# In[ ]:





# In[ ]:


# comparing all the models to see which give the same answer, that will be the new training data
# the keras models dont work as well as i hoped, so dropping them from the final comparison

# df_new_training = df_test_copy_comp[(df_test_copy_comp['Cover_type_all'] == df_test_copy_comp['Cover_type_1to4']) & 
#                                    (df_test_copy_comp['Cover_type_all'] == df_test_copy_comp['Cover_type_keras_all']) & 
#                                    (df_test_copy_comp['Cover_type_all'] == df_test_copy_comp['Cover_type_keras'])]

# df_new_test = df_test_copy_comp[~((df_test_copy_comp['Cover_type_all'] == df_test_copy_comp['Cover_type_1to4']) & 
#                                    (df_test_copy_comp['Cover_type_all'] == df_test_copy_comp['Cover_type_keras_all']) & 
#                                    (df_test_copy_comp['Cover_type_all'] == df_test_copy_comp['Cover_type_keras']))]


df_new_training = df_test_copy_comp[(df_test_copy_comp['Cover_type_all'] == df_test_copy_comp['Cover_type_1to4'])]

df_new_test = df_test_copy_comp[~((df_test_copy_comp['Cover_type_all'] == df_test_copy_comp['Cover_type_1to4']))]


# In[ ]:


df_new_training.sample(5)


# In[ ]:


df_new_test.sample(5)


# In[ ]:


len(df_new_training),len(df_new_test)


# In[ ]:





# In[ ]:


# now lets redo the wildnerness areas one by one again
# this will be the fifth model
# and it should be the best - generally gives me a 0.005 to 0.01 better score


# In[ ]:


# split the test data into wilderness areas

df_new_test_w1 = df_new_test[df_new_test['Wilderness_Area1']==1]
df_new_test_w2 = df_new_test[df_new_test['Wilderness_Area2']==1]
df_new_test_w3 = df_new_test[df_new_test['Wilderness_Area3']==1]
df_new_test_w4 = df_new_test[df_new_test['Wilderness_Area4']==1]


# In[ ]:


# need to define the stacks, can change later for each area etc

random_state = 0

ab_clf = AdaBoostClassifier(n_estimators=150,
                            base_estimator=DecisionTreeClassifier(
                                min_samples_leaf=2,
                                random_state=random_state),
                            random_state=random_state)

et_clf = ExtraTreesClassifier(max_depth=None,
                              n_estimators=150,
                              n_jobs=-1,
                              random_state=random_state)

lg_clf = LGBMClassifier(n_estimators=150,
                         num_class=8,
                         num_leaves=25,
                         learning_rate=5,
                         min_child_samples=20,
                         bagging_fraction=.3,
                         bagging_freq=1,
                         reg_lambda = 10**4.5,
                         reg_alpha = 1,
                         feature_fraction=.2,
                         num_boost_round=4000,
                         max_depth=-1,
                         n_jobs=4,
                         silent=-1,
                         verbose=-1)

lda_clf = LinearDiscriminantAnalysis()

knn_clf = KNeighborsClassifier(n_jobs=-1, n_neighbors =1)

gnb_clf = GaussianNB()

svc_clf = SVC(random_state=random_state,
              probability=True,
              verbose=True)

bg_clf = BaggingClassifier(n_estimators=150,
                           verbose=0,
                           random_state=random_state)

gb_clf = GradientBoostingClassifier(n_estimators=150,
                              min_samples_leaf=100,
                              verbose=0,
                              random_state=random_state)

# xgb_clf = XGBClassifier(n_estimators = 500,
#                        learning_rate = 0.1,
#                        max_depth = 200,
#                        objective = 'binary:logistic',
#                        random_state=random_state,
#                        n_jobs = -1)

cb_clf = CatBoostClassifier(n_estimators = 150,
                           max_depth = None,
                           learning_rate = 0.3,
                           random_state=random_state,
                           cat_features = None,
                           verbose = False)

rf_clf = RandomForestClassifier(n_estimators=150,
                                max_depth = None,
                                verbose=0,
                                random_state=random_state)

hg_clf = HistGradientBoostingClassifier(max_iter = 150,
                                        max_depth = 25,
                                        random_state = 0)

stack_w1 = StackingCVClassifier(classifiers=[hg_clf, knn_clf, rf_clf, et_clf],
                             meta_classifier=rf_clf,
                             cv=5,
                             stratify=True,
                             shuffle=True,
                             use_probas=True,
                             use_features_in_secondary=True,
                             verbose=1,
                             random_state=random_state,
                             n_jobs=-1)

stack_w2 = StackingCVClassifier(classifiers=[gnb_clf],
                             meta_classifier=rf_clf,
                             cv=5,
                             stratify=True,
                             shuffle=True,
                             use_probas=True,
                             use_features_in_secondary=True,
                             verbose=1,
                             random_state=random_state,
                             n_jobs=-1)

stack_w3 = StackingCVClassifier(classifiers=[et_clf],
                             meta_classifier=rf_clf,
                             cv=5,
                             stratify=True,
                             shuffle=True,
                             use_probas=True,
                             use_features_in_secondary=True,
                             verbose=1,
                             random_state=random_state,
                             n_jobs=-1)

stack_w4 = StackingCVClassifier(classifiers=[hg_clf],
                             meta_classifier=rf_clf,
                             cv=5,
                             stratify=True,
                             shuffle=True,
                             use_probas=True,
                             use_features_in_secondary=True,
                             verbose=1,
                             random_state=random_state,
                             n_jobs=-1)


# In[ ]:


# wilderness area 1


# In[ ]:


# define X and y

feature_cols = [col for col in list(df_new_training.columns) if col not in {'Id', 'Cover_type_all', 'Cover_type_1to4','Cover_type_keras'}] # i.e. all the columns except the Cover_Type and Id
X = df_new_training[df_new_training['Wilderness_Area1']==1][feature_cols]
y = df_new_training[df_new_training['Wilderness_Area1']==1]['Cover_type_all']

X.shape, y.shape


# In[ ]:


X.head()


# In[ ]:


y.head()


# In[ ]:


# split X and y into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


# In[ ]:


# remove outliers

iso_clf = IsolationForest(n_estimators=100, 
                          behaviour='new', 
                          max_samples=100,
                          contamination=0.01,
                          verbose=0,
                          random_state=random_state,
                          n_jobs=-1)

outliers = []
for cl in [1, 2, 3, 4, 5, 6, 7]:
    y_cl = y_train[y_train == cl]

    if not y_cl.empty:
        X_cl = X_train.loc[y_cl.index]

        iso_clf = iso_clf.fit(X_cl, y_cl)
        pred = iso_clf.predict(X_cl)
        outliers += y_cl[pred == -1].index.tolist()
    
if outliers:
    X_train = X_train.drop(outliers, axis='index')
    y_train = y_train.drop(outliers)


# In[ ]:


# training the models

stack_w1 = stack_w1.fit(X_train, y_train)


# In[ ]:


# test the model

prediction = stack_w1.predict(X_test)

print(metrics.accuracy_score(y_test, prediction))


# In[ ]:


# use all the columns this time

X_testData = df_new_test_w1[feature_cols]


# In[ ]:


Cover_Type_prediction_w1_new = stack_w1.predict(X_testData)


# In[ ]:


df_new_test_w1['Cover_Type'] = Cover_Type_prediction_w1_new


# In[ ]:


# df_new_test_w1.head()


# In[ ]:


# wilderness area 2


# In[ ]:


# define X and y

feature_cols = [col for col in list(df_new_training.columns) if col not in {'Id', 'Cover_type_all', 'Cover_type_1to4','Cover_type_keras'}] # i.e. all the columns except the Cover_Type and Id
X = df_new_training[df_new_training['Wilderness_Area2']==1][feature_cols]
y = df_new_training[df_new_training['Wilderness_Area2']==1]['Cover_type_all']

X.shape, y.shape


# In[ ]:


X.head()


# In[ ]:


# split X and y into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


# In[ ]:


# remove outliers

iso_clf = IsolationForest(n_estimators=100, 
                          behaviour='new', 
                          max_samples=100,
                          contamination=0.01,
                          verbose=0,
                          random_state=random_state,
                          n_jobs=-1)

outliers = []
for cl in [1, 2, 3, 4, 5, 6, 7]:
    y_cl = y_train[y_train == cl]

    if not y_cl.empty:
        X_cl = X_train.loc[y_cl.index]

        iso_clf = iso_clf.fit(X_cl, y_cl)
        pred = iso_clf.predict(X_cl)
        outliers += y_cl[pred == -1].index.tolist()
    
if outliers:
    X_train = X_train.drop(outliers, axis='index')
    y_train = y_train.drop(outliers)


# In[ ]:


# training the models

stack_w2 = stack_w2.fit(X_train, y_train)


# In[ ]:


# test the model

prediction = stack_w2.predict(X_test)

print(metrics.accuracy_score(y_test, prediction))


# In[ ]:


X_testData = df_new_test_w2[feature_cols]


# In[ ]:


Cover_Type_prediction_w2_new = stack_w2.predict(X_testData)


# In[ ]:


df_new_test_w2['Cover_Type'] = Cover_Type_prediction_w2_new


# In[ ]:


# df_new_test_w2.head()


# In[ ]:


# wilderness area 3


# In[ ]:


# define X and y

feature_cols = [col for col in list(df_new_training.columns) if col not in {'Id', 'Cover_type_all', 'Cover_type_1to4','Cover_type_keras'}] # i.e. all the columns except the Cover_Type and Id
X = df_new_training[df_new_training['Wilderness_Area3']==1][feature_cols]
y = df_new_training[df_new_training['Wilderness_Area3']==1]['Cover_type_all']

X.shape, y.shape


# In[ ]:


X.head()


# In[ ]:


# split X and y into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


# In[ ]:


# remove outliers

iso_clf = IsolationForest(n_estimators=100, 
                          behaviour='new', 
                          max_samples=100,
                          contamination=0.01,
                          verbose=0,
                          random_state=random_state,
                          n_jobs=-1)

outliers = []
for cl in [1, 2, 3, 4, 5, 6, 7]:
    y_cl = y_train[y_train == cl]

    if not y_cl.empty:
        X_cl = X_train.loc[y_cl.index]

        iso_clf = iso_clf.fit(X_cl, y_cl)
        pred = iso_clf.predict(X_cl)
        outliers += y_cl[pred == -1].index.tolist()
    
if outliers:
    X_train = X_train.drop(outliers, axis='index')
    y_train = y_train.drop(outliers)


# In[ ]:


# training the models

stack_w3 = stack_w3.fit(X_train, y_train)


# In[ ]:


# test the model

prediction = stack_w3.predict(X_test)

print(metrics.accuracy_score(y_test, prediction))


# In[ ]:


X_testData = df_new_test_w3[feature_cols]    


# In[ ]:


Cover_Type_prediction_w3_new = stack_w3.predict(X_testData)


# In[ ]:


df_new_test_w3['Cover_Type'] = Cover_Type_prediction_w3_new


# In[ ]:


# df_new_test_w3.head()


# In[ ]:


# wilderness area 4


# In[ ]:


# define X and y

feature_cols = [col for col in list(df_new_training.columns) if col not in {'Id', 'Cover_type_all', 'Cover_type_1to4','Cover_type_keras'}] # i.e. all the columns except the Cover_Type and Id
X = df_new_training[df_new_training['Wilderness_Area4']==1][feature_cols]
y = df_new_training[df_new_training['Wilderness_Area4']==1]['Cover_type_all']

X.shape, y.shape


# In[ ]:


X.head()


# In[ ]:


# split X and y into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


# In[ ]:


# remove outliers

iso_clf = IsolationForest(n_estimators=100, 
                          behaviour='new', 
                          max_samples=100,
                          contamination=0.01,
                          verbose=0,
                          random_state=random_state,
                          n_jobs=-1)

outliers = []
for cl in [1, 2, 3, 4, 5, 6, 7]:
    y_cl = y_train[y_train == cl]

    if not y_cl.empty:
        X_cl = X_train.loc[y_cl.index]

        iso_clf = iso_clf.fit(X_cl, y_cl)
        pred = iso_clf.predict(X_cl)
        outliers += y_cl[pred == -1].index.tolist()
    
if outliers:
    X_train = X_train.drop(outliers, axis='index')
    y_train = y_train.drop(outliers)


# In[ ]:


# training the models

stack_w4 = stack_w4.fit(X_train, y_train)


# In[ ]:


# test the model

prediction = stack_w4.predict(X_test)

print(metrics.accuracy_score(y_test, prediction))


# In[ ]:


X_testData = df_new_test_w4[feature_cols]    


# In[ ]:


Cover_Type_prediction_w4_new = stack_w4.predict(X_testData)


# In[ ]:


df_new_test_w4['Cover_Type'] = Cover_Type_prediction_w4_new


# In[ ]:


# df_new_test_w4.head()


# In[ ]:





# In[ ]:


df_new_training['Cover_Type'] = df_new_training['Cover_type_1to4']


# In[ ]:


frames = [df_new_training, df_new_test_w1, df_new_test_w2, df_new_test_w3, df_new_test_w4]
df_submission = pd.concat(frames)


# In[ ]:


df_submission=df_submission[['Id','Cover_Type']].copy()


# In[ ]:


len(df_submission)


# In[ ]:


# this is the fifth model
df_submission.to_csv(sub_3, index=False)


# In[ ]:





# In[ ]:


# as i went through this i did find than some wilderness areas seem to work work best with the first model
# so selecting those to combine for the final submission
# combined submission taking wilderness areas from different parts
# i had to use the training model accuracy as a guide - so always a risk i actually make things worse!

use_w1 = 1
use_w2 = 0
use_w3 = 0
use_w4 = 0


# In[ ]:


# create wildnerness areas from the final model

frames = [df_new_training, df_new_test_w1, df_new_test_w2, df_new_test_w3, df_new_test_w4]
df_final = pd.concat(frames)

df_final_w1 = df_final[df_final['Wilderness_Area1']==1]
df_final_w2 = df_final[df_final['Wilderness_Area2']==1]
df_final_w3 = df_final[df_final['Wilderness_Area3']==1]
df_final_w4 = df_final[df_final['Wilderness_Area4']==1]


# In[ ]:


if use_w1 == 1:
    df_use_w1 = df_test_copy_w1
else:
    df_use_w1 = df_final_w1
    
if use_w2 == 1:
    df_use_w2 = df_test_copy_w2
else:
    df_use_w2 = df_final_w2
    
if use_w3 == 1:
    df_use_w3 = df_test_copy_w3
else:
    df_use_w3 = df_final_w3
    
if use_w4 == 1:
    df_use_w4 = df_test_copy_w4
else:
    df_use_w4 = df_final_w4
    
frames_use = [df_use_w1, df_use_w2, df_use_w3, df_use_w4]
df_use = pd.concat(frames_use)


# In[ ]:


df_submission=df_use[['Id','Cover_Type']].copy()

df_submission.to_csv(sub_4, index=False)


# In[ ]:




