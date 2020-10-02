#!/usr/bin/env python
# coding: utf-8

# <h1>Udacity Capstone Project</h1>
# 
# 
# This investigation hopes to use the Kaggle dataset to investigate social and economic aspects of student's lives and see if their final grades can be predicted based on these.

# <h2> Investigating Dataset</h2>
# Within this dataset, there are two csv files. Each file represents a different class, one being a maths class, the other being Portuguese. The file names are 
# 
# 1. student-mat.csv
# 
# 2. student-por.csv
# 
# Both files have the same columns so we can have a sneak peak at this below

# In[ ]:


import pandas as pd

maths = pd.read_csv('../input/student-mat.csv') 
portug = pd.read_csv('../input/student-por.csv')

print(maths.head())


# In[ ]:


sample = maths.loc[0,:]
print(sample)


# We can see we have two files with 33 columns however the meanings of the eery columns are not overly clear from looking at one row of the dataset. Below we have a full mapping of the columns with a more clear explanation.

# <h3> File Schemas </h3>
# 
# <ul>
#     <li>school: Student's school (binary: 'GP' - Gabriel Pereira or 'MS' - Mousinho da Silveira)</li>
# 
# <li>sex:Student's sex (binary: 'F' - female or 'M' - male)</li>
# 
# <li>age: Student's age (numeric: from 15 to 22)</li>
# 
# <li>address: Student's home address type (binary: 'U' - urban or 'R' - rural)</li>
# 
# <li>famsize: Family size (binary: 'LE3' - less or equal to 3 or 'GT3' - greater than 3)</li>
# 
# <li>Pstatus: Parent's cohabitation status (binary: 'T' - living together or 'A' - living apart)</li>
# 
# <li>Medu: Mother's education (numeric: 0 - none, 1 - primary education (4th grade), 2 - 5th to 9th grade, 3 - secondary education, or 4 - higher education)</li>
# 
# <li>Fedu: Father's education (numeric: 0 - none, 1 - primary education (4th grade), 2 - 5th to 9th grade, 3 - secondary education, or 4 - higher education)</li>
# 
# <li>Mjob: Mother's job (nominal: 'teacher', 'health' care related, civil 'services' (e.g. administrative or police), 'at_home' or 'other')</li>
# 
# <li>Fjob: Father's job (nominal: 'teacher', 'health' care related, civil 'services' (e.g. administrative or police), 'at_home' or 'other')</li>
# 
# <li>reason: Reason to choose this school (nominal: close to 'home', school 'reputation', 'course' preference or 'other')</li>
# 
# <li>guardian: Student's guardian (nominal: 'mother', 'father' or 'other')</li>
# 
# <li>traveltime: Home to school travel time (numeric: 1 - <15 min., 2 - 15 to 30 min., 3 - 30 min. to 1 hour, or 4 - >1 hour)</li>
# 
# <li>studytime: Weekly study time (numeric: 1 - <2 hours, 2 - 2 to 5 hours, 3 - 5 to 10 hours, or 4 - >10 hours)</li>
# 
# <li>failures: Number of past class failures (numeric: n if 1<=n<3, else 4)</li>
# 
# <li>schoolsup:Extra educational support (binary: yes or no)</li>
# 
# <li>famsup: Family educational support (binary: yes or no)</li>
# 
# <li>paid: Extra paid classes within the course subject (Math or Portuguese) (binary: yes or no)</li>
# 
# <li>activities: Extra-curricular activities (binary: yes or no)</li>
# 
# <li>nursery: Attended nursery school (binary: yes or no)</li>
# 
# <li>higher" Wants to take higher education (binary: yes or no)</li>
# 
# <li>internet: Internet access at home (binary: yes or no)</li>
# 
# <li>romantic: With a romantic relationship (binary: yes or no)</li>
# 
# <li>famrel: Quality of family relationships (numeric: from 1 - very bad to 5 - excellent)</li>
# 
# <li>freetime: Free time after school (numeric: from 1 - very low to 5 - very high)</li>
# 
# <li>goout: Going out with friends (numeric: from 1 - very low to 5 - very high)</li>
# 
# <li>Dalc: Workday alcohol consumption (numeric: from 1 - very low to 5 - very high)</li>
# 
# <li>Walc: Weekend alcohol consumption (numeric: from 1 - very low to 5 - very high)</li>
# 
# <li>health: Current health status (numeric: from 1 - very bad to 5 - very good)</li>
# 
# <li>absences: Number of school absences (numeric: from 0 to 93)</li>
# 
# <li>G1: First period grade (numeric: from 0 to 20)</li>
# 
# <li>G2: Second period grade (numeric: from 0 to 20)</li>
# 
# <li>G3: Final grade (numeric: from 0 to 20, output target)</li>
# </ul>

# <h3>Joining Datasets together</h3>
# For the rest of this notebook we will combine the datasets for the two classes together and work on one combined dataset and investigate the features in the dataset

# In[ ]:


totalDataSet = pd.concat([maths,portug])


# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
lst = ['school','sex','address','Dalc']
fig = plt.figure(figsize=(10, 20))
plt.rcParams.update({'font.size': 15})

for x,y in enumerate(lst):
    plt.subplot(len(lst),1,1+x) 
    plt.xlabel(y)
    plt.ylabel("G3")
    totalDataSet.groupby(y)['G3'].mean().plot(kind='bar')


# The above code allows us to dynamically look at different attributes and let us have an insight into how they might affect the final grade. For example we can see that there is little difference in final grades between males and females. However we can see that the GP school seems to have a better average score than MS. One other tool which might give us a better insight would be a heat map. 

# In[ ]:


import seaborn as sns
sns.set()
sns.heatmap(totalDataSet.corr(),linewidths=.5)


# The heat map shows the correlation between each feature in the dataset. It can be seen how there are extremely high correlations between the final grades and the grades given in first and second perion. The next features which seem to be most correlated are the level of education which the student's parents have attained.

# In[ ]:


ave = sum(totalDataSet.G3)/float(len(totalDataSet))
totalDataSet['average'] = ['above average' if i > ave else 'under average' for i in totalDataSet.G3]
sns.swarmplot(x=totalDataSet.Dalc, y =totalDataSet.G3, hue = totalDataSet.average)
totalDataSet.drop('average',axis=1);


# In[ ]:


ax2 = pd.value_counts(totalDataSet['Dalc']).sort_values(ascending=False).plot.bar()
ax2.set_xlabel('Number of Weekdays spent Drinking')
ax2.set_ylabel('Number of Students')


# We can see the number of students that drink on a weekday drops dramatically after 1. This is a good sign as it can be seen from the scatter plot that the number of below average students seem to outweigh the number of above average as the student drinks more and more during the week

# # Data PreProcessing
# We will look at the dataset and investigate some preprocessing techniques and see which ones will be most appropriate for the data. 
# 
# First we will separate the feature we are trying to predict, G3, with the rest of the features. In order to make this a bit more challenging for the algorithms we will also drop G2 and G1. It will be interesting to see if final grades can be predicted without these intermediate grades.

# In[ ]:


outcomes = totalDataSet['G3']
features_raw = totalDataSet.drop(['G3','G2','G1'],axis=1)


# In[ ]:


features_raw.hist(alpha=0.5, figsize=(16, 10))


# We can see there are a few features which seem to have a skewed distribution. For example 
#     1. Dalc
#     2. absences
#     3. failures
#     4. traveltime
# These are some good choices which could use a logarithimic distribution applied to them

# In[ ]:


skewed=['Dalc','Walc','absences','failures','traveltime']
features_log_transformed = pd.DataFrame(data=features_raw)
features_log_transformed[skewed] = features_raw[skewed].apply(lambda x:np.log(x+1))

features_log_transformed[skewed].hist(alpha=0.5, figsize=(16, 10))


# ## Normalizing Numerical Features
# It is also good practice to perform some scaling on numerical features. This will ensure that each feature is treated equally when performing supervised learning algortihms on the data.

# In[ ]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
numerical = ['age','Medu','Fedu','traveltime','studytime','failures','famrel','freetime','goout','Dalc','Walc','health','absences']
features_log_minmax_transform = pd.DataFrame(data=features_log_transformed)
features_log_minmax_transform[numerical] = scaler.fit_transform(features_log_transformed[numerical])

display(features_log_minmax_transform.head(n=5))


# Even though most of the numerical feature were already within a range of 1-5 it was good to scale all features so they can be on the same range

# ## One Hot Encoding
# Most algorithms expect numerical values to process. However it can be clearly seen that there are non numerical features. Pandas can be used to encode these to numerical values

# In[ ]:


features_final = pd.get_dummies(features_log_minmax_transform)
encoded = list(features_final.columns)
print("{} total features after one-hot encoding".format(len(encoded)))
print(encoded)


# ## Training the model
# Finally we can start getting to the good stuff
# We obviously can't train the model on the whole dataset so we will use SkLearns cross validation implementation to split the data into training and testing sets

# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(features_final,outcomes, test_size = 0.2, random_state=42)

print("Training set has {} samples".format(X_train.shape[0]))
print("Testing set has {} samples".format(X_test.shape[0]))


# ## Base Model
# Linear Regression will be used as a benchmark model and we will use other algorithms in an attempt to outperform this

# ### Linear Regression
# Training and Testing the model

# In[ ]:


from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train,y_train)


# Now to test the model and see how it performed!

# In[ ]:


y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error

train_rms = sqrt(mean_squared_error(y_train, y_train_pred))
test_rms = sqrt(mean_squared_error(y_test, y_test_pred))


print("The Root mean Squared Error for the training set is", train_rms)
print("The Root mean Squared Error for the testing set is ", test_rms)


mae_train = mean_absolute_error(y_train, y_train_pred)
mae_test = mean_absolute_error(y_test, y_test_pred)
print('Mean Absolute Error for Training Set: %f' % mae_train)
print('Mean Absolute Error for Testing Set: %f' % mae_test)

print("Cross val score for training set",cross_val_score(model, X_train, y_train, cv=5).mean())
print("Cross val score for testing set",cross_val_score(model, X_test, y_test, cv=5).mean())


# These results suggest that a students grade can be predicted with approximately an error of 1.8-1.9 away from the actual grade which gives us a margin of Error of about 10-40% margin of Error. This is quite a large swing for an error but for the larger error it is still quite close to the actualy score, the error is just relative and shows that the algorithm performs admirably when predicting results 

# ## Testing other algorithms
# In this section we will test out some other algorithms. Once we have found what is believed to be the optimal solution, a grid search will be used on the hyper parameters in the hope that the optimal solution will be found
# The algorithms to be tested will include
# 
#     1.XgBoost
#     
#     2.LightGBM
#     
#     3.SVM
#     
# We can create a function to help reduce the code needed 

# In[ ]:


def model_Creator_Tester(name,model,X_train,X_test,y_train,y_test):
    print(name)
    model.fit(X_train,y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    from sklearn.metrics import mean_squared_error
    from math import sqrt
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import mean_absolute_error

    train_rms = sqrt(mean_squared_error(y_train, y_train_pred))
    test_rms = sqrt(mean_squared_error(y_test, y_test_pred))


    print("The Root mean Squared Error for the training set is", train_rms)
    print("The Root mean Squared Error for the testing set is ", test_rms)


    mae_train = mean_absolute_error(y_train, y_train_pred)
    mae_test = mean_absolute_error(y_test, y_test_pred)
    print('Mean Absolute Error for Training Set: %f' % mae_train)
    print('Mean Absolute Error for Testing Set: %f' % mae_test)
    return train_rms,test_rms,mae_train,mae_test;
    #print("Cross val score for training set",cross_val_score(model, X_train, y_train, cv=5).mean())
    #print("Cross val score for testing set",cross_val_score(model, X_test, y_test, cv=5).mean())


# In[ ]:


from sklearn.svm import SVC
from xgboost import XGBClassifier
import lightgbm as lgb
names = ["Linear_Regression","XGB","SVM","LGB"]
models = [LinearRegression(),XGBClassifier(),SVC(gamma='auto'),lgb.LGBMRegressor()]
results = {}
for x,y in zip(names,models):
    print("\n",y,"\n")
    results[x]=model_Creator_Tester(x,y,X_train,X_test,y_train,y_test)


# In[ ]:


def color_gradient ( val, beg_rgb, end_rgb, val_min = 0, val_max = 1):
    val_scale = (1.0 * val - val_min) / (val_max - val_min)
    return ( beg_rgb[0] + val_scale * (end_rgb[0] - beg_rgb[0]),
             beg_rgb[1] + val_scale * (end_rgb[1] - beg_rgb[1]),
             beg_rgb[2] + val_scale * (end_rgb[2] - beg_rgb[2]))


# In[ ]:


#print(results)
def print_results(results):
    titles = ["Root Mean Square for Training Set","Root Mean Square for Testing Set","Mean Absolute Error for Training Set","Mean Absolute Error for Testing Set"]
    fig = plt.figure(figsize=(10, 10))
    plt.rcParams.update({'font.size': 10})
    grad_beg, grad_end = ( 0.1, 0.1, 0.1), (1, 1, 0)
    for i,k in enumerate(results):
        tempVals = []
        for j in results.keys():
            #print(i,j)
            #print(results[j][i])
            tempVals.append(results[j][i])
        print(tempVals)
        print(results.keys())
        plt.subplot(len(titles)/2.,len(titles)/2.,1+i)
        col_list = [ color_gradient( val,
                                 grad_beg,
                                 grad_end,
                                 min( tempVals),
                                 max(tempVals)) for val in tempVals]

        plt.bar(results.keys(),tempVals,color = col_list)
        plt.title(titles[i])


# In[ ]:


print_results(results)


# The results here are quite interesting. The darkest bars have the lowest errors, which seems to be LGB in every scenario except for MAE on the training set. In this case XGB seems to outperform LGB. It could be said that may have overfitted the training set. It seems to be that Support Vector Machine is the worst performing algorithm on this dataset

# ## Optimising Hyper Paramaters
# In the results we have seen above this is using the most vanilla of algortihms. The next stage of this is to use GridSearch to try and optimise the hyperparamaters for each algorithm and see how much more we can improve performance

# In[ ]:


from sklearn.model_selection import GridSearchCV
params={'Linear_Regression':{'fit_intercept':[True,False], 'normalize':[True,False], 'copy_X':[True,False]},
        'XGB':{
                'boster':['gbtree'],
                'eta':[0.05,0.1,0.25,0.5,0.8],
                'gamma':[0.05,0.1,0.25,0.5,0.8],
                #'reg_alpha': [0.05,0.1,0.25,0.5,0.8],
                #'reg_lambda': [0.05,0.1,0.25,0.5,0.8],
                'max_depth':[3,6,10],
                'subsample':[0.1,0.25,0.5,0.8]
        },
        'SVM':{'C': [0.001, 0.01, 0.1, 1, 10], 'gamma' : [0.001, 0.01, 0.1, 1],'kernel':['rbf','linear']},
       'LGB':{'boosting_type': ['gbdt'],
                'num_leaves': [20,50,80],
                'learning_rate': [0.05,0.1,0.25,0.5,0.8],
                'subsample_for_bin': [10,100,500],
                'min_child_samples': [20,50,100],
                'reg_alpha': [0.05,0.1,0.25,0.5,0.8],
                'reg_lambda': [0.05,0.1,0.25,0.5,0.8]
             }
        }

names = ["Linear_Regression","SVM","LGB","XGB"]

models = [LinearRegression(),SVC(),lgb.LGBMRegressor(),XGBClassifier()]


# In[ ]:


def grid_model_Creator_Tester(name,model,X_train,X_test,y_train,y_test):
    print(name)
    model.fit(X_train,y_train)
    best_model = model.best_params_
    print(best_model)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    from sklearn.metrics import mean_squared_error
    from math import sqrt
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import mean_absolute_error

    train_rms = sqrt(mean_squared_error(y_train, y_train_pred))
    test_rms = sqrt(mean_squared_error(y_test, y_test_pred))


    print("The Root mean Squared Error for the training set is", train_rms)
    print("The Root mean Squared Error for the testing set is ", test_rms)


    mae_train = mean_absolute_error(y_train, y_train_pred)
    mae_test = mean_absolute_error(y_test, y_test_pred)
    print('Mean Absolute Error for Training Set: %f' % mae_train)
    print('Mean Absolute Error for Testing Set: %f' % mae_test)
    return train_rms,test_rms,mae_train,mae_test;


# In[ ]:


results = {}
for x,y in zip(names,models):
    results[x]=grid_model_Creator_Tester(x,GridSearchCV(y,params[x]),X_train,X_test,y_train,y_test)
print_results(results)


# Looking at the errors for the four algorithms, it can be seen that LGB has the lowest error on both testing sets. It could be said that XGB seems to overfit the datasets as the training error for that is very low while the testing error is much higher!
