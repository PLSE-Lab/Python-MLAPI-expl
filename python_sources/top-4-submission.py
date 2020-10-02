#!/usr/bin/env python
# coding: utf-8

# # Earthquake damage prediction
# 
# 
# The result of this prediction was submitted for the competition hosted at https://www.drivendata.org/competitions/57/nepal-earthquake.
# 
# As of 10/07, it was ranked 112.

# In[ ]:


from IPython.display import Image
Image(url = 'https://scx2.b-cdn.net/gfx/news/2016/mountingtens.jpg')


# Credit: Roger Bilham/CIRES

# ### In this kernel an overall view of the Nepal Earthquake dataset is provided.
# 
# ### Models are trained using Random Forest Classification and XGBoost Classification methods. The f1_score is maximised by removing outliers and tuning xgboost hyper-parameters. 

# In[ ]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import scipy
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


train_labels = pd.read_csv('../input/richters-predictor-modeling-earthquake-damage/train_labels.csv')
test = pd.read_csv('../input/richters-predictor-modeling-earthquake-damage/test_values.csv')
train = pd.read_csv('../input/richters-predictor-modeling-earthquake-damage/train_values.csv')


# The damage_grade column, which is the variable we're trying to predict, comes in a different file:'train_labels' Fortunately, it has the same length and indexes as the 'train_values' file

# In[ ]:


train_labels.head()


# In[ ]:


train_labels.info()


# In[ ]:


train.head()


# In[ ]:


train.info()


# Moreover, there are no null values. Let's just check that all Building IDs from the train_labels dataset match those of train_values

# In[ ]:


res = train_labels.building_id.equals(train.building_id)
print("Statement 'all building IDs match is'", res)


# Thus, the damage grade column can simply be added on to the train values data frame

# In[ ]:


train['damage_grade'] = train_labels.damage_grade


# In[ ]:


train.describe()


# The mean damage grade is ~2.38, with a standard deviation fo 0.6, indicating that most buildings were severly damaged

# In[ ]:


fig = plt.subplots(figsize = (9,5))
sns.countplot(train.damage_grade)
plt.show()


# In[ ]:


pd.value_counts(train.damage_grade)


# The majority, >50% buildings were labeled with damage_grade = 2, while only ~25,000, <10% buildings were labeled as mildly damaged (damage_grade = 1)

# Let's have a look at the superstrctures

# In[ ]:


train.iloc[:,[i for i in range(15, 26)]].head()


# We can see just from the first 5 rows, that some buildings have superstructures from multiple materials. We'll leave the columns as such for now, however there could be some merit in trying to group them together in some way

# In[ ]:


train.iloc[:,[i for i in range(28, 39)]].head()


# In[ ]:


train.has_secondary_use.mean()


# Let's see whether there are buildings with multiple secondary uses by summing up the means and see whether it's equal to the has_secondary_use mean

# In[ ]:


total = 0
for i in range(29, 39):
    col = train.columns[i]
    total+=train[col].mean()
print(f'The sum of means of the secondary_use columns is {total}')


# This is slightly larger than the previous mean, so there is a very small number of buildings with multiple secondary uses

# In[ ]:


plt.hist(train.age)
plt.show()


# In[ ]:


plt.hist(train.age,range=(0,175), bins = 15)
plt.show()


# Most buildings are aged 0-50, there is a small cluster of ancient outliers that were labeled as being 995 years old

# In[ ]:


sns.barplot('damage_grade', 'age', data = train)


# Younger buildings seem more likely to have been less likely affected; however there's no mean age difference between medium and severly affected buildings

# In[ ]:


sns.barplot('damage_grade', 'land_surface_condition', data= train)


# Some small correlation between land surface condition and damage grade

# In[ ]:


train.drop('building_id', inplace = True, axis = 1) #this column isn't needed


# Let's examine superstructure info

# In[ ]:


#14- 24 are columns containing superstructure info
superstructure_cols = []
for i in range(14, 25):
    superstructure_cols.append(train.columns[i])


# In[ ]:


corr = train[superstructure_cols+['damage_grade']].corr()


# In[ ]:


sns.heatmap(corr)


# Mud mortar stone homes were the worse, being strongly correlated with a higher damage grade, while cement mortar brick were least damaged

# In[ ]:


sns.barplot('damage_grade', 'has_superstructure_adobe_mud', data=train,)


# In[ ]:


#pearsonr correlation implies normal distribution
#The p-value roughly indicates the probability of an uncorrelated system producing datasets that have a
#Pearson correlation at least as extreme as the one computed from these datasets
scipy.stats.pearsonr(train.damage_grade, train.has_superstructure_mud_mortar_stone)


# In[ ]:


scipy.stats.pearsonr(train.damage_grade, train.has_superstructure_cement_mortar_brick)


# Some of the superstrcture values are strongly positively or negatively correlated between them. For example, buildings made from timber are also largely composed of bamboo. 

# In[ ]:


superstructure_cols = []
for i in range(14, 25):
    superstructure_cols.append(train.columns[i])


# In[ ]:


secondary_use = []
for i in range(27, 37):
    secondary_use.append(train.columns[i])


# In[ ]:


corr = train[secondary_use +['damage_grade']].corr()


# In[ ]:


sns.heatmap(corr)


# Close to no relation between seconday use and damage grade

# In[ ]:


additional_num_data = []
for i in range(7):
    additional_num_data.append(train.columns[i])
additional_num_data.append(train.columns[26])


# In[ ]:


corr = train[additional_num_data+['damage_grade']].corr()


# In[ ]:


sns.heatmap(corr)


# Let's see what accuracy we can get with the raw data, and then we will try to modify it and decide on the best models
# 
# First let's change the object type columns to int

# In[ ]:


train.dtypes.value_counts()


# In[ ]:


print('Object data types:\n')
#we'll use the function later, without wanting to print anything
def get_obj(train, p = False):
    obj_types = []
    for column in train.columns:
        if train[column].dtype == 'object': 
            if p: print(column)
            obj_types.append(column)
    return obj_types
obj_types = get_obj(train, True)


# In[ ]:


def transform_to_int(train, obj_types):
    #Assign dictionaries with current values and replacements for each column
    d_lsc = {'n':0, 'o':1, 't':2}
    d_ft = {'h':0, 'i':1, 'r':2, 'u':3, 'w':4}
    d_rt = {'n':0, 'q':1, 'x':2}
    d_gft = {'f':0, 'm':1, 'v':2, 'x':3, 'z':4}
    d_oft = {'j':0, 'q':1, 's':2, 'x':3}
    d_pos = {'j':0, 'o':1, 's':2, 't':3}
    d_pc = {'a':0, 'c':1, 'd':2, 'f':3, 'm':4, 'n':5, 'o':6, 'q':7, 's':8, 'u':9}
    d_los = {'a':0, 'r':1, 'v':2, 'w':3}
    #Each positional index in replacements corresponds to the column in obj_types
    replacements = [d_lsc, d_ft, d_rt, d_gft, d_oft, d_pos, d_pc, d_los]

    #Replace using lambda Series.map(lambda)
    for i,col in enumerate(obj_types):
        train[col] = train[col].map(lambda a: replacements[i][a]).astype('int64')
transform_to_int(train, obj_types)


# In[ ]:


train.dtypes.value_counts()


# All columns are now integer types

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier


# In[ ]:


#separate column to be predicted from the rest
y = train.pop('damage_grade') 
x = train.copy()


# Let's start by running a RandomForestClassifier on the data

# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(x, y)

rcf = RandomForestClassifier()
model = rcf.fit(x_train, y_train)

model.score(x_test, y_test)


# In[ ]:


y_pred = model.predict(x_test)


# F1 score will be used to assess the model accuracy for this competiton
# F1 score combines accuracy and precision. As there are 3 possible labels, micro averaged F1 score is used.

# In[ ]:


f1_score(y_test, y_pred,average='micro')


# Get confusion matrix

# In[ ]:


def get_conf_matrix(y_test, y_pred):    
    data = confusion_matrix(y_test, y_pred) #get confusion matrix
    cm = pd.DataFrame(data, columns=np.unique(y_test), index = np.unique(y_test)) #build the confusion matrix as a dataframe table
    cm.index.name = 'Observed'
    cm.columns.name = 'Predicted'
    plt.figure(figsize = (10,7))
    sns.heatmap(cm, annot=True, fmt="d", annot_kws={"size": 12}) #plot a heatmap
    plt.title("Confusion Matrix")
    plt.show()
get_conf_matrix(y_test, y_pred)


# The model is overestimating the buildings with damage_grade = 2 : 37,201 observed and 42,243 predicted: 5,042 buildings more, or ~13.5%.
# 
# It underestimates the other two damage ranks. Damage_grade = 1 is predicted the worst, out of 6,170 observed, only 4,625 were predicted, 25% less. Moreover, the model correctly predicted only 3,015 buildings, having an accuracy of <50% for damage grade=1, as opposed to an accuracy of 30,724/37,201 * 100 ~= 82.5% 
# 
# While this doesn't come as a big surprise given the damage grades distribution, it indicates that focusing on refining damage_grade = 1 predictions may lead to significant f1 score improvement

# The feature importance dataframe helps us focus on relevant variables

# In[ ]:


importance = pd.DataFrame({"Feature":list(x), "Importance": rcf.feature_importances_}) # build a dataframe with features and their importance
importance = importance.sort_values(by="Importance", ascending=False) #sort by importance
importance


# The geographic location has the highest impact

# Before we commence cleaning the data, let's have a look at the f1 scores given by some other models:

# In[ ]:


#import the fitting methods to try
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier


classifiers = [
    KNeighborsClassifier(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    GradientBoostingClassifier()]

def model_and_test(X, y, classifiers):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.1, random_state = 42)
    for model in classifiers:
        this_model = model.__class__.__name__ #get the name of the classifier
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        
        print(f'{this_model} f1 score:')
        score = f1_score(y_test, y_pred,average='micro')
        print(f'{score:.4f}')
        print('\n')


# In[ ]:


model_and_test(x, y, classifiers)


# Random Forest Classifier outputs the best prediction, followed by knn  

# One way to improve model accuracy is to remove outliers. Plotting boxplots for the most important parameters could help visualize how these look like

# In[ ]:


boxplot_cols=["geo_level_3_id","geo_level_2_id","geo_level_1_id","age", "area_percentage", "height_percentage"]
q=1
plt.figure(figsize=(20,20))
for j in boxplot_cols:
    plt.subplot(3,3,q)
    ax=sns.boxplot(train[j].dropna())
    plt.xlabel(j)
    q+=1
plt.show()


# No outliers for the geolocation data, however there are some for the age and normalized area/height of the building footprint

# Z scores can be used instead to filter data, with the risk however of removing some useful data points. To avoid losing useful data, the importance data-frame will be used in order to only drop rows which contain outliers for relevant variables.

# In[ ]:


def remove_outliers(df, col_cutoff = 0.01, z_score = 3.5): #define a function to get rid of all outliers of the most important columns
    important_cols = importance[importance.Importance>col_cutoff]['Feature'].tolist() #get all columns with importance > 0.01.  
    df_new = df.copy() #init the new df
    for col in important_cols: df_new = df_new[np.abs(scipy.stats.zscore(df_new[col]))<z_score] #removing all rows where a z-score is >3
    return df_new


# In[ ]:


df = pd.concat([x, y], axis = 1)


# In[ ]:


df_new = remove_outliers(df)


# In[ ]:


y = df_new.pop('damage_grade')
x=df_new


# In[ ]:


sns.countplot(y)


# The current settings are not very good, they are removing too many columns with damage_grade = 1, so a milder method of removing outliers is required

# Given the size of our dataset, ~ 260,000 samples, considering all variables with z scores > 3, as outliers, corresponding to 0.27% percentile, might be removing some useful data. 
# 
# A z score of 3.5, corresponding with the 0.0465% could also be good enough to remove outliers, while preserving more samples. This way, the original distrbituion between damage grades may be better preserved too.

# Before moving on with outlier removal, let's look at a generally better method of predicting tabular data: XGBoost.
# 
# XGradientBoost is an ensemble learning method built on top of Decision Trees, imporving upon other Gradient boosting methods.
# More details can be found here https://xgboost.readthedocs.io/en/latest/python/python_intro.html

# In[ ]:


import xgboost as xgb


# In[ ]:


#Bring up original data
def get_original():
    df = pd.read_csv('../input/richters-predictor-modeling-earthquake-damage/train_values.csv')
    df.drop('building_id', axis =1, inplace=True)
    obj_types = get_obj(df)
    transform_to_int(df, obj_types)
    df['damage_grade'] = train_labels.damage_grade

    return df
df = get_original()

# a function that will later be used to divide dataframe into x(independent variables) and y(dependent variable)
def get_xy(df):
    y = df.pop('damage_grade')
    x= df
    return x, y


# In[ ]:


y = df.damage_grade
x = df.drop('damage_grade', axis = 1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.1, random_state = 42)


# In[ ]:


pd.value_counts(y_test) #to confirm that the original proportion of damage grades is preserved


# In[ ]:


def test_model(model, removing = False, col_cutoff = 0.01, z_score = 3.5):
    df_train = pd.concat([x_train, y_train], axis = 1) #combine them together, so outliers are simultaneously removed from both
    if removing: df_train = remove_outliers(df_train, col_cutoff, z_score) 
    x, y =get_xy(df_train)
    model.fit(x, y)

    y_pred = model.predict(x_test)
    print(f1_score(y_test, y_pred, average='micro'))
test_model(xgb.XGBRFClassifier())


# In[ ]:


models = [xgb.XGBRFClassifier(), xgb.XGBClassifier()]
for model in models:
    print(model.__class__.__name__, 'score:',end =' ')
    test_model(model, True)


# Not only is no improvement observed following outlier removal, but the score decreased by ~0.001.
# 
# However, XGBClassifier is giving a much better score, even better than rand forest classifier, so it will be the chosen model for this submission.
# Let's test a few general ways of removing outliers:

# I'll be commenting out some of the code, as it takes quite some time to run the. The outputs are copied in the kernel

# In[ ]:


xgbc = xgb.XGBClassifier()


# In[ ]:


'''
xgbc = xgb.XGBClassifier() #init xgbc 
for a in [0.01, 0.02, 0.05]:
    for b in [2.5, 3, 3.5]:
        print('removing outliers on columns with importance >,',a,'on z scores >',b,'. Score =', end=' ')
        test_model(xgbc, True, a, b) '''


# removing outliers on columns with importance >, 0.01 on z scores > 2.5 . Score = 0.6931813821418978
# 
# removing outliers on columns with importance >, 0.01 on z scores > 3 . Score = 0.7232262768120948
# 
# removing outliers on columns with importance >, 0.01 on z scores > 3.5 . Score = 0.7284831740915544
# 
# removing outliers on columns with importance >, 0.02 on z scores > 2.5 . Score = 0.7151682590844557
# 
# removing outliers on columns with importance >, 0.02 on z scores > 3 . Score = 0.7249913664095775
# 
# removing outliers on columns with importance >, 0.02 on z scores > 3.5 . Score = 0.7272936571889029
# 
# removing outliers on columns with importance >, 0.05 on z scores > 2.5 . Score = 0.72460765127969
# 
# removing outliers on columns with importance >, 0.05 on z scores > 3 . Score = 0.7252599670004989
# 
# removing outliers on columns with importance >, 0.05 on z scores > 3.5 . Score = 0.7273704002148804

# In[ ]:


'''for b in [2.5, 3, 3.5]:
    print('removing outliers on columns with importance > 0.1,','on z scores >',b,'. Score =', end=' ')
    test_model(xgbc, True, 0.1, b)'''


# removing outliers on columns with importance > 0.1, on z scores > 2.5 . Score = 0.7273320287018918
# 
# removing outliers on columns with importance > 0.1, on z scores > 3 . Score = 0.7285982886305207
# 
# removing outliers on columns with importance > 0.1, on z scores > 3.5 . Score = 0.727600629292813

# In[ ]:


print('No outlier removal score:', end = ' ')
test_model(xgbc, False)


# Currently the best result is achieved when leaving all the data in. However, it is likely that following hyper-parameter tuning, a superior result can be achieved by dropping some outliers. Since dropping outliers on columns with importance >0.1, using a z score of 3 achieved the best result, that is what will be used. 
# 
# It will only drop on outliers from area and height percentage, ensuring no important rows are dropped, and the original distribution between damage grades is maintained. 

# In[ ]:


x, y = get_xy(df)


# In[ ]:


df_train = pd.concat([x, y], axis = 1) #combine them together, so outliers are simultaneously removed from both x and y
df_train = remove_outliers(df_train, 0.1, 3)
x, y =get_xy(df_train)


# In[ ]:


xgbc = xgb.XGBRFClassifier()


# Hyper-parameter tuning is a crucial part of ML if we want to find the best model.
# First, testing is performed to find an optimal max depth and estimator numbers.
# 
# Max depth represents how deep the decision trees in the model will be, while number of estimators indicates how many weak lernears(decision trees) are to be used.
# 
# Generally, the larger these are, the better the accuracy of the model is, however going too deep may lead to overfitting and consuming too much computational power for a negligible score increase in return.

# In[ ]:


parameters = {'max_depth' : [5, 10, 20, 40]} #first looking for an optimal max_depth


# In[ ]:


from sklearn.model_selection import GridSearchCV
#grid search cv tries all the parameters individually using cross validation, default set to 5 folds
grid_search = GridSearchCV(xgbc, parameters, scoring="f1_micro", n_jobs=-1, verbose=3)
# grid_result = grid_search.fit(x, label_encoded_y)


# In[ ]:


def plot_score(grid_result, parameters, name):    
    means = grid_result.cv_results_['mean_test_score'] #get the means of the scores from the 5 folds
    stds = grid_result.cv_results_['std_test_score'] #standard error of scores for plotting error bars

    # plot scores vs parameter
    plt.errorbar(parameters[name], means, yerr=stds)
    pyplot.xlabel(name)
    pyplot.ylabel('f1 score')
#plot_score(grid_result,parameters, 'max_depth')


# In[ ]:


Image(url = 'https://imgur.com/16Lya4M.png')


# The best result at max_depth = 20.

# In[ ]:


xgbc = xgb.XGBRFClassifier(max_depth = 20)


# In[ ]:


n_estimators = [50, 100, 150, 200]
param2 = {'n_estimators':n_estimators}


# In[ ]:


grid_search_estimators = GridSearchCV(xgbc, param2, scoring="f1_micro", n_jobs=-1, verbose=3)
# grid_result_estimators = grid_search.fit(x, label_encoded_y)


# In[ ]:


# plot_score(grid_result_estimators,param2, 'n_estimators')


# In[ ]:


Image(url = 'https://i.imgur.com/FwgVGgk.png')


# Best result at n_estimators = 150
# 
# We could try to further narrow down the optimal max-depth and n_estimators, however it is taking too long. 
# 
# While we have optimised two important parameters, there are many more xgboost hyper-parameters that can optimised. 
# In order to test many at once, randomized search cross-validation is employed. It randomly tests a given number of iterations from the given params and finds the ones outputting the best score.

# In[ ]:


xgbc = xgb.XGBClassifier(max_depth = 20, n_estimators = 150)


# In[ ]:


params={
 "learning_rate"    : [0.1, 0.2, 0.3] ,
 "min_child_weight" : [ 3, 5, 7 ],
 "gamma"            : [ 0.0, 0.05, 0.1, 0.2 , 0.3],
 "colsample_bylevel" :[0.2, 0.5, 0.8, 1.0],
 "colsample_bynode": [0.2, 0.5, 0.8, 1.0],
 "subsample": [0.2, 0.5, 0.8, 1.0],
 "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ]
    }


# In[ ]:


from sklearn.model_selection import RandomizedSearchCV
rand_search = RandomizedSearchCV(xgbc,param_distributions=params ,n_iter=10,scoring='f1_micro',n_jobs=-1, verbose = 3)


# In[ ]:


# rand_res = rand_search.fit(x, y)


# In[ ]:


# best_params = rand_res.best_params_


# In[ ]:


best_params = {'subsample': 0.8,
 'min_child_weight': 5,
 'learning_rate': 0.1,
 'gamma': 0.05,
 'colsample_bytree': 0.3,
 'colsample_bynode': 0.8,
 'colsample_bylevel': 0.8}


# A ~0.02 f1 score improvement has been achieved. It is now time to fit the test data and submit it.

# In[ ]:


xgbc = xgb.XGBClassifier( min_child_weight= 5, learning_rate= 0.1, gamma= 0.05, subsample= 0.8,colsample_bytree= 0.3, colsample_bynode= 0.8,
 colsample_bylevel= 0.8, max_depth = 20, n_estimators = 150)

# xgbc.fit(x, y) #final model


# In[ ]:


def submit_model(model, file_name): #I defined a function because I was submitting multiple models
    test = pd.read_csv('../input/richters-predictor-modeling-earthquake-damage/test_values.csv') #get the test csv into a dataframe
    submission_ids = test.pop('building_id') #get the building ids
    transform_to_int(test, get_obj(test)) #transform obj_types to int to predict damage grades
    submission_predictions = model.predict(test) #predict
    subbmission = pd.DataFrame({'building_id':submission_ids, 'damage_grade':submission_predictions}) #save buildings_ids and predicted damage grades to a data frame
    subbmission.to_csv(file_name, index = False) #save as a csv file


# In[ ]:


# submit_model(xgbc, 'submission_xgb4.csv')
#0.7477 score


# In order to further improve the model, some of the following steps can be taken:
# <ul>
#     <li> Assesing the relation between outliers of specific variables such as age or height percentage and damage grade</li>
#     <li> Grouping some of the variables together to create one which correlates more strongly with damage grade. Some candidates to consider include:
#         <ol>  <li> superstructure parameters </li>
#             <li> secondary_use parameters </li>
#             <li> other possibly correlated variables, such as ground_floor_type and superstructure used </li>
#         </ol>  
#     </li>
#     <li> Further tuning the hyper-parameters; this may however come at a significant computational cost </li>
# </ul>

# I hope you enjoyed this project !!
# 
# I'd be more than happy to respond to any comments/suggestions you may have !
