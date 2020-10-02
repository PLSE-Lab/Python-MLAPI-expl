#!/usr/bin/env python
# coding: utf-8

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

# ## Project Description
# 
# Car insurance companies are always trying to come up with a fair insurance plan for customers. They would like to offer a lower price to the careful and safe driver while the careless drivers who file claims in the past will pay more. In addition, more safe drivers mean that the company will spend less in operation. However, for new customers, it is difficult for the company to know who the safe driver is. As a result, if a company offers a low price, it bears a high risk of cost. If not, the company loses competitiveness and encourage new customers to choose its competitors.
# 
# 
# Your task is to create a machine learning model to mitigate this problem by identifying the safe drivers in new customers based on their profiles. The company then offers them a low price to boost safe customer acquirement and reduce risks of costs. We provide you with a dataset (train_set.csv) regarding the profile (columns starting with ps_*) of customers. You will be asked to predict whether a customer will file a claim (`target`) in the next year with the test_set.csv 
# 
# You can find the dataset in the `project` folders in the jupyter hub. We also upload dataset to Kaggle and will test your result and offer you a leaderboard in Kaggle:
# https://www.kaggle.com/t/426d97d4138b49b2802c2ee0461a18ac

# ## Phase 1: 26th May - 9th June
# 
# ### Data Description
# 
# In order to take a look at the data, you can use the `describe()` method. As you can see in the result, each row has a unique `id`. `Target` $\in \{0, 1\}$ is whether a user will file a claim in his insurance period. The rest of the 57 columns are features regarding customers' profiles. You might also notice that some of the features have minimum values of `-1`. This indicates that the actual value is missing or inaccessible.
# 

# In[ ]:


# Quick load dataset and check
import pandas as pd
import matplotlib as plt


# In[ ]:


filename = "train_set.csv"
data_train = pd.read_csv(filename)
filename = "test_set.csv"
data_test = pd.read_csv(filename)


# In[ ]:


data_train.describe()


# The prefix, e.g. `ind` and `calc`, indicate the feature belongs to similiar groupings. The postfix `bin` indicates binary features and `cat` indicates categorical features. The features without postfix are ordinal or continuous. Similarly, you can check the statistics for testing data:

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')

train_missing_count = (data_train == -1).sum()
plt.rcParams['figure.figsize'] = (20,10)
train_missing_count.plot.bar()


# We can drop features with many missing values. In this case, features 'ps_car_03_cat', 'ps_car_05_cat' have by far the most missing values so it is a good idea to drop them. 

# In[ ]:


# Drop columns with many -1 values
vars_to_drop = ['ps_car_03_cat', 'ps_car_05_cat']
data_train.drop(vars_to_drop, inplace=True, axis=1)


# For binary and categorical features we can replace missing data with the most frequent value of the feature. For other features continuous or ordinal, replace missing data with mean value of the feature.

# In[ ]:


from sklearn.impute import SimpleImputer

mean_imputer = SimpleImputer(missing_values=-1, strategy='mean')
mode_imputer = SimpleImputer(missing_values=-1, strategy='most_frequent')

data_train['ps_reg_03'] = mean_imputer.fit_transform(data_train[['ps_reg_03']]).ravel()
data_train['ps_car_12'] = mean_imputer.fit_transform(data_train[['ps_car_12']]).ravel()
data_train['ps_car_14'] = mean_imputer.fit_transform(data_train[['ps_car_14']]).ravel()
data_train['ps_car_11'] = mode_imputer.fit_transform(data_train[['ps_car_11']]).ravel()


# In[ ]:


## Select target and features
fea_col = data_train.columns[2:]
data_Y = data_train['target']
data_X = data_train[fea_col]


# ### Split data_train into training and validation set

# In[ ]:


from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(data_X, data_Y, test_size = 0.1, shuffle=True, random_state=42)


# ### Resample data

# We used either one of the two following cells for oversampling

# In[ ]:


from sklearn.utils import resample

train = pd.concat([x_train, y_train], axis=1)
class_0 = train[train['target'] == 0]
class_1 = train[train['target'] == 1]

class_1_upsampled = resample(class_1, 
                          replace=True,
                          n_samples=len(class_0),
                          random_state=42)

upsampled = pd.concat([class_0, class_1_upsampled])

x_train = upsampled.drop('target', axis=1)
y_train = upsampled.target


# In[ ]:


from imblearn.over_sampling import SMOTE

sm = SMOTE(sampling_strategy='minority', random_state=42)
x_train_over, y_train_over = sm.fit_sample(x_train, y_train)


# ### Hyperparameter Tuning with GridSearchCV

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

rf = RandomForestClassifier()

param_grid = {'n_estimators': [50],
              'n_jobs': [-1],
              'class_weight': ['balanced'],
              'min_samples_leaf': [40, 50, 60], 
              'min_samples_split': [50, 80, 110, 140]
              }

clf = GridSearchCV(rf, param_grid, cv=3, scoring='f1_macro', verbose=10)
clf.fit(x_train, y_train)


# In[ ]:


clf.best_params_


# ### Train Model (RandomForestClassifier)

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import VotingClassifier

rf_clf = RandomForestClassifier(n_estimators=50, 
                                n_jobs=-1,
                                class_weight='balanced',
                                min_samples_leaf=50, 
                                min_samples_split=50,
                                random_state=42,
                                verbose=5)

rf_clf.fit(x_train, y_train)
y_pred = rf_clf.predict(x_val)


# ### Get prediction statistics

# In[ ]:


sum(y_pred==y_val)/len(y_val)


# In[ ]:


from sklearn.metrics import confusion_matrix

confusion_matrix(y_true=y_val, y_pred=y_pred)


# In[ ]:


from sklearn.metrics import plot_confusion_matrix

plot_confusion_matrix(rf_clf, x_val, y_val)


# In[ ]:


from sklearn.metrics import f1_score

f1_score(y_val, y_pred, average='macro')


# ### Submission
# 
# Please only submit the csv files with predicted outcome with its id and target [here](https://www.kaggle.com/t/b3dc81e90d32436d93d2b509c98d0d71). Your column should only contain `0` and `1`.

# In[ ]:


data_test_X = data_test.drop(columns=['id', 'ps_car_03_cat', 'ps_car_05_cat'])
y_target = rf_clf.predict(data_test_X)
y_target0 = y_target[y_target == 0]
y_target1 = y_target[y_target == 1]
print(len(data_test_X), len(y_target0), len(y_target1))
sum(y_target==0)


# In[ ]:


data_out = pd.DataFrame(data_test['id'].copy())
data_out.insert(1, "target", y_target, True) 
data_out.to_csv('submission.csv',index=False)


# In[ ]:


data_out


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




