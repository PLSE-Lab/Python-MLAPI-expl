# %% [markdown]
# # Machine Learning 2020 Course Projects
# 
# ## Project Schedule
# 
# In this project, you will solve a real-life problem with a dataset. The project will be separated into two phases:
# 
# 27th May - 10th June: We will give you a training set with target values and a testing set without target. You predict the target of the testing set by trying different machine learning models and submit your best result to us and we will evaluate your results first time at the end of phase 1.
# 
# 9th June - 24th June: Students stand high in the leader board will briefly explain  their submission in a proseminar. We will also release some general advice to improve the result. You try to improve your prediction and submit final results in the end. We will again ask random group to present and show their implementation.
# The project shall be finished by a team of two people. Please find your teammate and REGISTER via [here](https://docs.google.com/forms/d/e/1FAIpQLSf4uAQwBkTbN12E0akQdxfXLgUQLObAVDRjqJHcNAUFwvRTsg/alreadyresponded).
# 
# The submission and evaluation is processed by [Kaggle](https://www.kaggle.com/t/b3dc81e90d32436d93d2b509c98d0d71).  In order to submit, you need to create an account, please use your team name in the `team tag` on the [kaggle page](https://www.kaggle.com/t/b3dc81e90d32436d93d2b509c98d0d71). Two people can submit as a team in Kaggle.
# 
# You can submit and test your result on the test set 2 times a day, you will be able to upload your predicted value in a CSV file and your result will be shown on a leaderboard. We collect data for grading at 22:00 on the **last day of each phase**. Please secure your best results before this time.
# 
# 

# %% [markdown]
# ## Project Description
# 
# Car insurance companies are always trying to come up with a fair insurance plan for customers. They would like to offer a lower price to the careful and safe driver while the careless drivers who file claims in the past will pay more. In addition, more safe drivers mean that the company will spend less in operation. However, for new customers, it is difficult for the company to know who the safe driver is. As a result, if a company offers a low price, it bears a high risk of cost. If not, the company loses competitiveness and encourage new customers to choose its competitors.
# 
# 
# Your task is to create a machine learning model to mitigate this problem by identifying the safe drivers in new customers based on their profiles. The company then offers them a low price to boost safe customer acquirement and reduce risks of costs. We provide you with a dataset (train_set.csv) regarding the profile (columns starting with ps_*) of customers. You will be asked to predict whether a customer will file a claim (`target`) in the next year with the test_set.csv 
# 
# You can find the dataset in the `project` folders in the jupyter hub. We also upload dataset to Kaggle and will test your result and offer you a leaderboard in Kaggle:
# https://www.kaggle.com/t/b3dc81e90d32436d93d2b509c98d0d71

# %% [markdown]
# ### Notes:
# We tried various classifier e.g. linear SVM, Neural Networks, Decision Trees at first but then decided to use the Random Forest Classifier for this project because it had the best results and was easy to use.
# 
# Our score on kaggle at the end of phase one was 0.5103.
# 
# After we got the notebooks with the hints we tried various things like undersamplung, replacing missing values or scaling our data.
# 
# The best result we achieved after various attempts has been 0.5164 locally and 0.51093 on the kaggle leaderboard.
# 
# The presentation of Hawar & Pablo during the last proseminar gave us some new valuable ideas. Especially the pipeline and the scaling they did.
# 
# So we created a pipeline and executed a big RandomizedSearchCV on the new (and now much smaller data). The best estimator of this search instantly achieved a score of 0.52570 on kaggle which is a big improvement comparedto our previous results.
# 
# Lastly, we did some further hyperparameter tuning with some grid searches to further improve our result. Our best kaggle result is 0.52605.
# 
# ## Important!!
# For validation, the final classifier we used to get our best prediction can be found at the very bottom of this notebook. (heading Best setup)

# %% [code]
# Quick load dataset and check
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm, trange
from matplotlib.colors import ListedColormap
from IPython.display import set_matplotlib_formats
from sklearn import cluster, datasets
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import SMOTENC
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import f1_score, make_scorer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, make_scorer


# %% [code]
cat_features = ['ps_ind_02_cat', 'ps_ind_04_cat', 'ps_ind_05_cat', 'ps_car_01_cat', 'ps_car_02_cat', 'ps_car_03_cat', 'ps_car_04_cat', 'ps_car_05_cat', 'ps_car_06_cat', 'ps_car_07_cat', 'ps_car_08_cat', 'ps_car_09_cat', 'ps_car_10_cat', 'ps_car_11_cat']
bin_features = ['ps_ind_06_bin', 'ps_ind_07_bin', 'ps_ind_08_bin', 'ps_ind_09_bin', 'ps_ind_10_bin', 'ps_ind_11_bin', 'ps_ind_12_bin', 'ps_ind_13_bin', 'ps_ind_16_bin', 'ps_ind_17_bin', 'ps_ind_18_bin']
calc_features = ['ps_calc_01', 'ps_calc_02', 'ps_calc_03', 'ps_calc_04', 'ps_calc_05', 'ps_calc_06', 'ps_calc_07', 'ps_calc_08', 'ps_calc_09', 'ps_calc_10', 'ps_calc_11', 'ps_calc_12', 'ps_calc_13', 'ps_calc_14','ps_calc_15_bin', 'ps_calc_16_bin', 'ps_calc_17_bin', 'ps_calc_18_bin', 'ps_calc_19_bin', 'ps_calc_20_bin' ]
cont_features = ['ps_ind_15', 'ps_ind_03', 'ps_car_15', 'ps_ind_14', 'ps_car_13', 'ps_reg_03', 'ps_reg_01', 'ps_car_12', 'ps_car_14', 'ps_ind_01', 'ps_reg_02', 'ps_car_11']

def split(data_train):
    """
    Splits a dataset from sklearn into train and test sets.
    
    :param: dataset: sklearn dataset (data, labels) (2-tuple of numpy arrays)
    :returns: x_train, x_test, y_train, y_test (4-tuple of numpy arrays)
    """

    # Get data and labels
    fea_col = data_train.columns[2:]
    Y = data_train['target']
    X = data_train[fea_col]

    # Reshape Y to [num_points, 1]
    Y = np.expand_dims(Y, axis=1)

    # Split the data into train and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=123)
    
    print("Shape of data:")
    print(f"X_train: {X_train.shape}, X_test: {X_test.shape}, Y_train: {Y_train.shape}, Y_test: {Y_test.shape}")
    return X_train, X_test, Y_train, Y_test

# %% [code]
filename = "train_set.csv"
data_train = pd.read_csv(filename)
filename = "test_set.csv"
data_test = pd.read_csv(filename)

# %% [code]
data_train.drop(calc_features, axis=1, inplace=True)
data_test.drop(calc_features, axis=1, inplace=True)

# %% [markdown]
# ### Split Dataset

# %% [code]
# Function for splitting into train and test sets
x_train, x_test, y_train, y_test = split(data_train)

# %% [code]
from sklearn.preprocessing import MinMaxScaler
#data_train.drop(calc_features, axis=1, inplace=True)
scaler = StandardScaler().fit(data_train[cont_features])
data_train[cont_features] = scaler.transform(data_train[cont_features])
def one_hot(df, cols):
    for each in cols:
        dummies = pd.get_dummies(df[each], prefix=each, drop_first=False)
        df = pd.concat([df, dummies], axis=1)
    return df

data_train=one_hot(data_train,cat_features)
fea_col = data_train.columns[2:]

X = data_train[fea_col]

x_train, x_test, y_train, y_test = split(data_train)

# %% [markdown]
# ### Balancing

# %% [code]
fea_col = data_train.columns[2:]
Y = y_train
X = x_train[fea_col]

# Reshape Y to [num_points, 1]
#oversampling
#smote_nc=SMOTE()
#X_resampled,y_resampled=smote_nc.fit_resample(X,Y)
#y_resampled = np.expand_dims(y_resampled, axis=1)

smote_nc=SMOTE()
X_resampled,y_resampled=smote_nc.fit_resample(X,Y)
y_resampled = np.expand_dims(y_resampled, axis=1)
print(X_resampled.shape)
x_train=X_resampled

y_train=y_resampled

# %% [markdown]
# ### Random Search with pipeline:

# %% [code]
pipe=Pipeline([('fill',SimpleImputer(missing_values=-1, strategy='mean')),
              ('center',StandardScaler()),
              ('min_max',MinMaxScaler()),
              ('model',RandomForestClassifier())])

# %% [code]
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
class_weight = ['balanced',{0:1,1:5},{0:1,1:10},{0:1,1:15},{0:1,1:20},{0:1,1:25},{0:1,1:12},{0:1,1:17}]
# Create the random grid
random_grid = {'model__n_estimators': n_estimators,
               'model__max_features': max_features,
               'model__max_depth': max_depth,
               'model__min_samples_split': min_samples_split,
               'model__min_samples_leaf': min_samples_leaf,
               'model__bootstrap': bootstrap,
               'model__class_weight':class_weight}
print(random_grid)

# %% [code]
y_train=np.ravel(y_train)
rf = RandomForestClassifier()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(pipe, param_distributions = random_grid,
                               n_iter = 10, cv = 2, verbose=10, random_state=42, n_jobs = 4,
                               scoring='f1_macro')
# Fit the random search model
rf_random.fit(x_train, y_train)

# %% [code]
rf_random.best_params_

# %% [code]
final=rf_random.best_estimator_
predictions=final.predict(x_test)
print(f1_score(y_test, np.array(predictions), average='macro'))

# %% [markdown]
# ### Gridsearch wit pipeline:

# %% [code]
gridpipe=Pipeline([('fill',SimpleImputer(missing_values=-1, strategy='mean')),
              ('center',StandardScaler()),
              ('min_max',MinMaxScaler()),
              ('model',RandomForestClassifier(n_estimators=600,min_samples_split=2,min_samples_leaf=4,
                            max_features='sqrt',max_depth=10,class_weight={0:1,1:22},bootstrap=True ))])
parameters={
    #'model__class_weight':[{0:1,1:10},{0:1,1:12},{0:1,1:14},{0:1,1:16},{0:1,1:18},{0:1,1:20},{0:1,1:22},{0:1,1:24},{0:1,1:26},{0:1,1:28},{0:1,1:30}]
    #'model__max_depth':[10,15,20,25,30,35,40,45,50],
    #'model__min_samples_leaf': [1, 2, 4],  
    'model__min_samples_split': [2,5,7,10,15],
    #'model__max_features': ['auto', 'sqrt'],
}
clf = GridSearchCV(gridpipe, parameters,verbose=10, n_jobs=4,scoring='f1_macro',  cv=2)
clf.fit(x_train, y_train)

# %% [code]
clf.best_params_

# %% [code]
pipeBest=Pipeline([('fill',SimpleImputer(missing_values=-1, strategy='mean')),
              ('center',StandardScaler()),
              ('min_max',MinMaxScaler()),
              ('model',RandomForestClassifier(n_estimators=600,min_samples_split=2,min_samples_leaf=4,
                            max_features='sqrt',max_depth=10,class_weight={0:1,1:20},bootstrap=True ))])

# %% [code]
y_train=np.ravel(y_train)
pipeBest.fit(x_train, y_train)

# %% [code]
predictions=pipeBest.predict(x_test)
print(f1_score(y_test, np.array(predictions), average='macro'))

# %% [markdown]
# ## Best result of all searches
# {'model__n_estimators': 600,
#  'model__min_samples_split': 2,
#  'model__min_samples_leaf': 4,
#  'model__max_features': 'sqrt',
#  'model__max_depth': 10,
#  'model__class_weight': {0: 1, 1: 20},
#  'model__bootstrap': True}

# %% [markdown]
# # Best setup:

# %% [code]
#load data
filename = "train_set.csv"
data_train = pd.read_csv(filename)
filename = "test_set.csv"
data_test = pd.read_csv(filename)
#drop calc features
data_train.drop(calc_features, axis=1, inplace=True)
data_test.drop(calc_features, axis=1, inplace=True)
x_train, x_test, y_train, y_test = split(data_train)
pipeBest=Pipeline([('fill',SimpleImputer(missing_values=-1, strategy='mean')),
              ('center',StandardScaler()),
              ('min_max',MinMaxScaler()),
              ('model',(n_estimators=600,min_samples_split=2,min_samples_leaf=4,
                            max_features='sqrt',max_depth=10,class_weight={0:1,1:20},bootstrap=True ))])

y_train=np.ravel(y_train)
pipeBest.fit(x_train, y_train)
predictions=pipeBest.predict(x_test)
print(f1_score(y_test, np.array(predictions), average='macro'))

# %% [markdown]
# ### Submission
# 
# Please only submit the csv files with predicted outcome with its id and target [here](https://www.kaggle.com/t/b3dc81e90d32436d93d2b509c98d0d71). Your column should only contain `0` and `1`.

# %% [code]
fea_col = data_test.columns[1:]
data_test_X = data_test[fea_col].values
print(data_test_X.shape)
y_target = pipeBest.predict(data_test_X)
print(y_target.shape)

# %% [code]
data_test_X = data_test.drop(columns=['id']).values
#y_target=np.array(predictions)

# %% [code]
data_out = pd.DataFrame(data_test['id'].copy())
print(y_target.shape)
data_out.insert(1, "target", y_target.astype(int), True) 
data_out.to_csv('submission.csv',index=False)