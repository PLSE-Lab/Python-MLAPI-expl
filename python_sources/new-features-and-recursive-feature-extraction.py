#!/usr/bin/env python
# coding: utf-8

# **Extra Tree Classification Baseline Kernel**
# 
# This kernel is a simple exercise in using basic techniques in machine learning. It is meant as a jumping off point to develop more complex and better models to increase your LB score.

# **Load Data and Develop Features**
# 
# There are many kernels that do EDA so I won't get into that here. So, let's just develop some new features. We will have a few similar to those developed in other kernels plus some newer ones. 
# 
# Since we don't know if points are in the same direction or in opposite directions or even right angles to each other we can just generate a few combinations via adding and subtracting some fo the values. We can also expand on the shadiness as well, whether or not that makes much difference. I will not add any soil features though other kernels have some success with combining soil features.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.feature_selection import RFECV

from scipy import stats

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

# set up dataset
number_classes = 7
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')

known = train_df['Cover_Type']
train_df = train_df.drop(['Cover_Type','Id'],axis=1)

Id_test = test_df['Id']
test_df = test_df.drop(['Id'],axis=1)

####################### Test data #############################################
train_df['HF1'] = train_df['Horizontal_Distance_To_Hydrology']+train_df['Horizontal_Distance_To_Fire_Points']
train_df['HF2'] = (train_df['Horizontal_Distance_To_Hydrology']-train_df['Horizontal_Distance_To_Fire_Points'])
train_df['HR1'] = (train_df['Horizontal_Distance_To_Hydrology']+train_df['Horizontal_Distance_To_Roadways'])
train_df['HR2'] = (train_df['Horizontal_Distance_To_Hydrology']-train_df['Horizontal_Distance_To_Roadways'])
train_df['FR1'] = (train_df['Horizontal_Distance_To_Fire_Points']+train_df['Horizontal_Distance_To_Roadways'])
train_df['FR2'] = (train_df['Horizontal_Distance_To_Fire_Points']-train_df['Horizontal_Distance_To_Roadways'])
train_df['EV1'] = train_df.Elevation+train_df.Vertical_Distance_To_Hydrology
train_df['EV2'] = train_df.Elevation-train_df.Vertical_Distance_To_Hydrology
train_df['Mean_HF1'] = train_df.HF1/2
train_df['Mean_HF2'] = train_df.HF2/2
train_df['Mean_HR1'] = train_df.HR1/2
train_df['Mean_HR2'] = train_df.HR2/2
train_df['Mean_FR1'] = train_df.FR1/2
train_df['Mean_FR2'] = train_df.FR2/2

train_df['slope_hyd'] = (train_df['Horizontal_Distance_To_Hydrology']**2+train_df['Vertical_Distance_To_Hydrology']**2)**0.5
train_df.slope_hyd=train_df.slope_hyd.map(lambda x: 0 if np.isinf(x) else x) # remove infinite value if any

#Mean distance to Amenities 
train_df['Mean_Amenities']=(train_df.Horizontal_Distance_To_Fire_Points + train_df.Horizontal_Distance_To_Hydrology + train_df.Horizontal_Distance_To_Roadways) / 3 
#Mean Distance to Fire and Water 
train_df['Mean_Fire_Hyd1']=(train_df.Horizontal_Distance_To_Fire_Points + train_df.Horizontal_Distance_To_Hydrology) / 2
train_df['Mean_Fire_Hyd2']=(train_df.Horizontal_Distance_To_Fire_Points - train_df.Horizontal_Distance_To_Hydrology) / 2

#Shadiness
train_df['Shadiness_morn_noon'] = train_df.Hillshade_9am/(train_df.Hillshade_Noon+1)
train_df['Shadiness_noon_3pm'] = train_df.Hillshade_Noon/(train_df.Hillshade_3pm+1)
train_df['Shadiness_morn_3'] = train_df.Hillshade_9am/(train_df.Hillshade_3pm+1)
train_df['Shadiness_morn_avg'] = (train_df.Hillshade_9am+train_df.Hillshade_Noon)/2
train_df['Shadiness_afernoon'] = (train_df.Hillshade_Noon+train_df.Hillshade_3pm)/2
train_df['Shadiness_total_mean'] = (train_df.Hillshade_9am+train_df.Hillshade_Noon+train_df.Hillshade_3pm)/3


# Apply the same features to the test data as well.

# In[ ]:


test_df['HF1'] = test_df['Horizontal_Distance_To_Hydrology']+test_df['Horizontal_Distance_To_Fire_Points']
test_df['HF2'] = (test_df['Horizontal_Distance_To_Hydrology']-test_df['Horizontal_Distance_To_Fire_Points'])
test_df['HR1'] = (test_df['Horizontal_Distance_To_Hydrology']+test_df['Horizontal_Distance_To_Roadways'])
test_df['HR2'] = (test_df['Horizontal_Distance_To_Hydrology']-test_df['Horizontal_Distance_To_Roadways'])
test_df['FR1'] = (test_df['Horizontal_Distance_To_Fire_Points']+test_df['Horizontal_Distance_To_Roadways'])
test_df['FR2'] = (test_df['Horizontal_Distance_To_Fire_Points']-test_df['Horizontal_Distance_To_Roadways'])
test_df['EV1'] = test_df.Elevation+test_df.Vertical_Distance_To_Hydrology
test_df['EV2'] = test_df.Elevation-test_df.Vertical_Distance_To_Hydrology
test_df['Mean_HF1'] = test_df.HF1/2
test_df['Mean_HF2'] = test_df.HF2/2
test_df['Mean_HR1'] = test_df.HR1/2
test_df['Mean_HR2'] = test_df.HR2/2
test_df['Mean_FR1'] = test_df.FR1/2
test_df['Mean_FR2'] = test_df.FR2/2

test_df['slope_hyd'] = (test_df['Horizontal_Distance_To_Hydrology']**2+test_df['Vertical_Distance_To_Hydrology']**2)**0.5
test_df.slope_hyd=test_df.slope_hyd.map(lambda x: 0 if np.isinf(x) else x) # remove infinite value if any

#Mean distance to Amenities 
test_df['Mean_Amenities']=(test_df.Horizontal_Distance_To_Fire_Points + test_df.Horizontal_Distance_To_Hydrology + test_df.Horizontal_Distance_To_Roadways) / 3 
#Mean Distance to Fire and Water 
test_df['Mean_Fire_Hyd1']=(test_df.Horizontal_Distance_To_Fire_Points + test_df.Horizontal_Distance_To_Hydrology) / 2
test_df['Mean_Fire_Hyd2']=(test_df.Horizontal_Distance_To_Fire_Points + test_df.Horizontal_Distance_To_Hydrology) / 2

#Shadiness
test_df['Shadiness_morn_noon'] = test_df.Hillshade_9am/(test_df.Hillshade_Noon+1)
test_df['Shadiness_noon_3pm'] = test_df.Hillshade_Noon/(test_df.Hillshade_3pm+1)
test_df['Shadiness_morn_3'] = test_df.Hillshade_9am/(test_df.Hillshade_3pm+1)
test_df['Shadiness_morn_avg'] = (test_df.Hillshade_9am+test_df.Hillshade_Noon)/2
test_df['Shadiness_afernoon'] = (test_df.Hillshade_Noon+test_df.Hillshade_3pm)/2
test_df['Shadiness_total_mean'] = (test_df.Hillshade_9am+test_df.Hillshade_Noon+test_df.Hillshade_3pm)/3

print('Total number of features : %d' % (train_df.shape)[1])


# **Feature Selection**
# 
# The let's do feature selection. I haven't gone through features to check correlation or checked for other things like that. We could try things like PCA but I think that is over kill and would require scaling. Therefore, I will use Recursive Feature Extraction with Cross-Validation (RFECV) using an ExtraTree Classifier estimator. You can play with this and it likely can be improveed but it is a start. RFECV will get rid of low variance features as well which is a bonus. (So soil7 and soil15 for example). This takes a bit to run so don't worry if it takes 10 minutes.

# In[ ]:


selector = RFECV(estimator=ExtraTreesClassifier(n_estimators=200, criterion='entropy', min_samples_split=3, n_jobs=-1),step=1, cv=5)
selector = selector.fit(train_df,known)
print('Optimal number of features : %d' % selector.n_features_)
X_train = selector.transform(train_df)
X_test = selector.transform(test_df)
#print(train_df.columns[selector.support_].values) #Uncomment if you want to see what features were selected


# **Run ExtraTree Model**
# 
# Great - now we have the best features to use. The first thing we should do is set up a k-fold to iterate over when developing the models. So, let's use a stratified k-fold. I use stratified here to keep the training groups with approximately the same groupings of cover. I used 9 folds but you could experiment with that if you like.
# 
# You will notice that at the end of the for loop, I create a prediction on the test data for each model. This generates 'k' sets of data that I can use for some basic ensembling on the test data.

# In[ ]:


strat_kfold = StratifiedKFold(n_splits=5)
p_corr = []
y_true = []
pred_ens = []

for train_index, cv_index in strat_kfold.split(X_train, known):

    Xtrain, X_cv = (np.array(X_train))[train_index], (np.array(X_train))[cv_index]
    Ytrain, y_cv = (np.array(known))[train_index], (np.array(known))[cv_index]
    
    etr = ExtraTreesClassifier(n_estimators=700, criterion='entropy', min_samples_split=3, n_jobs=-1)
    trained_model = etr.fit(Xtrain, Ytrain)
    
    pred_te = (trained_model.predict(X_cv))    
    
    y_true = np.hstack((y_true,y_cv))
    pred_ens = np.hstack((pred_ens,pred_te))
   
    predictions = (trained_model.predict(np.array(X_test)))     
    p_corr.append(predictions)


# **Results**
# 
# So, how did our models work out? Let's look at the confusion matrix. You can see we do a good job of predicting everything in general. The accuracy shows a reasonable value that turns out close to the LB score, so we know we aren't over fitting anything horribly.
# 
# One thing to note is that cover 1 and cover 2 are often "confused" - most of the errors in cover 1 are confused for cover 2 and vice versa. That means, to really improve the predictions, we need to address this issue.

# In[ ]:


print('\n5-fold Confusion Matrix')
print(confusion_matrix(y_true,pred_ens))

print('\n5-fold Accuracy:')
print(accuracy_score(y_true,pred_ens))


# **Make a Submission File**
# 
# So, let's prep a submission. Since we have 'k' predictions, we can ensemble those in some way. I use the most simplest of methods. I just grab the mode value of the 7 predictions. This works fine if there is a clear majority of course, but if there is a tie, mode fails. Other methods could be to use VotingClassifier for instance - or a blending/stacking method as well. It works fine here though for a basic baseline.

# In[ ]:


q = (stats.mode(p_corr[:],axis=0)).mode

sub = pd.DataFrame()
sub['Id'] = Id_test
sub['Cover_Type'] = q.reshape(-1)
sub.head()
sub.to_csv('my_submission.csv', index=False)

