#!/usr/bin/env python
# coding: utf-8

# # Importing

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[ ]:


train = pd.read_csv('/kaggle/input/forest-cover-type-prediction/train.csv')
test = pd.read_csv('/kaggle/input/forest-cover-type-prediction/test.csv')
test_id = test.Id


def write_sub(pred,postfix):
    '''
    
    Writes submission file
    
    pred    - predicted values
    postfix - description of a file
    
    '''
    
    sub = pd.DataFrame({'Id':test_id,'Cover_Type':pred})
    file_name = '/kaggle/working/'+'sub_'+postfix+'.csv'
    sub.to_csv(file_name,index = False)
    print(file_name,' is ready, please submit')


# # EDA

# In[ ]:


train.head()


# In[ ]:


print('(TRAIN) No of cols: {}\n(TRAIN) No of rows: {}'.format(len(train.columns),len(train.index)))
print('(TEST)  No of cols: {}\n(TEST)  No of rows: {}'.format(len(test.columns),len(test.index)))


# Our test set is much biger than train set. Hope it works fine.
# 
# We have 55 features, 40 columns  are just one hot encoded feature of soil type

# In[ ]:


# Alternativly just use train.info()

for col in train.columns:
    print('(TRAIN) Column {} is {} type'.format(train[col].name,train[col].dtype))
    
for col in test.columns:
    print('(TEST) Column {} is {} type'.format(test[col].name,test[col].dtype))


# In[ ]:


print('No of cols with NaN\nTraining set: {}\nTest set: {}\n'.format(len(train.isna().sum()[train.isna().sum() != 0]),
                                                                     len(test.isna().sum()[test.isna().sum() != 0])))


# Luckily all features are of int type.
# 
# Therefore we can avoide different types of encodings.
# 
# Additionaly we have not got missing values
# 
# Let`s check closer these values

# In[ ]:


# let`s use pair plot to have an overal idea about data.

from pandas.plotting import scatter_matrix

scatter_matrix(train.loc[:,'Elevation':'Horizontal_Distance_To_Fire_Points'],
               c=train.Cover_Type,
               alpha = 0.2,
               hist_kwds = {'bins':100},
               figsize = (20,20));


# Imidiatly we observe than Elevation is the most promise feature, it separates different cover types pretty efficently.
# Also that is true for Distance to Road feature
# 
# Next lets check correlation between numerical features

# In[ ]:


cols = ['Elevation', 'Aspect', 'Slope',
        'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology',
        'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon',
        'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points','Cover_Type']

cm = train[cols].corr()
fig, ax = plt.subplots(figsize=(10,5));
ax.matshow(cm);
plt.xticks(range(cm.shape[1]), cm.columns, fontsize=10, rotation=90);
plt.yticks(range(cm.shape[1]), cm.columns, fontsize=10, rotation=0);


# From above correlation matrix we observe:
# 
# 1) Strong correlation between HillShade_3pm and Hillshade_noon
# 
# 2) Strong correlation between Vertical_Distance_To_Roadways and Hosontal_Distance_To_Roadways
# 
# 3) Some   correlation between Elevation - (Horizontal_Distance_To_Hydrology/Horizontal_distance_to_Roadwasys/ Horizontal_distance_to_fire_points)
# 
# Therefore thos features must be removed to avoid problems during fitting, alternativly we can combine correlated features to derive new features. In tis way we don`t loos information

# # Features

# In[ ]:


def model_input(data):
    
    
    '''
       
       Transforms dataframe to a form
       we would like to have it in our model
       
       also makes life easier since we can transform train and test set to the same format
       
    '''
    
    data = data.drop(['Id'],axis = 1)
    # the most reliable way yo combine distances

    # Better feature engineering can be found here
    # http://nbviewer.ipython.org/github/aguschin/kaggle/blob/master/forestCoverType_featuresEngineering.ipynb
    
    data['Distance_to_Hydrology'] = np.sqrt(data['Horizontal_Distance_To_Hydrology']**2 +                                              data['Vertical_Distance_To_Hydrology']**2)

    data['Elevation-VDH'] = abs(data['Elevation'] - data['Vertical_Distance_To_Hydrology'])
    data['Elevation-HDF'] = abs(data['Elevation'] - data['Horizontal_Distance_To_Fire_Points'])
    data['Elevation-HDH'] = abs(data['Elevation'] - data['Horizontal_Distance_To_Hydrology'])

    data['Elevation+VDH'] = abs(data['Elevation'] + data['Vertical_Distance_To_Hydrology'])
    data['Elevation+HDF'] = abs(data['Elevation'] + data['Horizontal_Distance_To_Fire_Points'])
    data['Elevation+HDH'] = abs(data['Elevation'] + data['Horizontal_Distance_To_Hydrology'])

    data['HDF+VDH'] = abs(data['Horizontal_Distance_To_Fire_Points'] - data['Vertical_Distance_To_Hydrology'])
    data['HDF+HDH'] = abs(data['Horizontal_Distance_To_Fire_Points'] - data['Horizontal_Distance_To_Hydrology'])

    data['HDF+VDH'] = abs(data['Horizontal_Distance_To_Fire_Points'] + data['Vertical_Distance_To_Hydrology'])
    data['HDF+HDH'] = abs(data['Horizontal_Distance_To_Fire_Points'] + data['Horizontal_Distance_To_Hydrology'])

    # From pair plot you can see that Vertical_Distance_To_Hydrology has negative values, therefore we can derive new feature
    # if Vertical_Distance_To_Hydrology is positive we encode it as 1 if it is negative as 0

    data['Higherwater'] = data['Vertical_Distance_To_Hydrology'].apply(lambda x: 1 if x>0 else 0)


    data['Hillshade_Noon_3pm'] = (data['Hillshade_Noon'] + data['Hillshade_3pm'])/2


    # Drop values we used for feature engineering
    data = data.drop(['Horizontal_Distance_To_Hydrology','Vertical_Distance_To_Hydrology','Elevation'],axis=1)
    data = data.drop(['Hillshade_Noon','Hillshade_3pm'],axis=1)
    return data


# In[ ]:


train = model_input(train)
test = model_input(test)


# # Modeling

# In[ ]:


from sklearn.model_selection import GridSearchCV, KFold,train_test_split, cross_val_score

X = train.drop(['Cover_Type'],axis = 1)
y = train['Cover_Type']


#-----------------------------------------------------------------------------------------
# from sklearn.preprocessing import LabelEncoder,LabelBinarizer

# remove binarizer and train wit xgb lgb
# LabelBinarizer is cool staff
# However it is almost incombatable with powerfull classfiers as XGB og LGB
# In the end I decided to remove it.
# But keep in mind it is usefull

#lb = LabelBinarizer()
#y = lb.fit_transform(y)

#-----------------------------------------------------------------------------------------


# In[ ]:


from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import MinMaxScaler,StandardScaler, RobustScaler
from sklearn.preprocessing import OneHotEncoder

#---------------------------------------------------------------------------------
# I have observed that scaling of numerica' values doesn`t work so good here
# therefore i would suggest to avoid minmax,standard or robust scalling
# Just treat values as it is
# The similar obsiravtion can be found here :
#---------------------------------------------------------------------------------
# https://www.kaggle.com/sharmasanthosh/exploratory-study-of-ml-algorithms
#---------------------------------------------------------------------------------
# last figure

#mms_cols = ['Elevation', 'Aspect', 'Slope','Distance_to_Hydrology',
            #'Horizontal_Distance_To_Roadways','Horizontal_Distance_To_Fire_Points',
            
            #'Elevation-VDH','Elevation-HDF','Elevation-HDH',]

#sca_cols = ['Hillshade_9am','Hillshade_Noon_3pm']

#---------------------------------------------------------------------------------=

# Nevertheless we still need to encode Higherwater feature
ohe_cols = ['Higherwater']

# I like to use this transformer

trans = make_column_transformer(
    #(RobustScaler(),mms_cols),
    #(StandardScaler(),sca_cols),
    (OneHotEncoder(),ohe_cols),
    remainder = 'passthrough'
)


# Let`s create 5 splits. we will test them with cross_val_score
from sklearn.model_selection import cross_val_score
kfolds = KFold(n_splits = 5, random_state=42, shuffle = True)      


# ## First look at classifiers
# 
# In this part I will just check different classifiers with more or less default hyperparametes to have a rough idea about score that can be improved later on.
# 
# Also with following code we may have an idea how feature engineering improves accuracy, without computitionaly expeansive gridseachCV

# ### 1. KNeighborsClassifier

# In[ ]:


# The simpliest one

from sklearn.neighbors import KNeighborsClassifier

pipe_knn = make_pipeline(trans,
                         KNeighborsClassifier(3))

print ('Mean Accuracy: ',cross_val_score(pipe_knn,X,y,cv = kfolds).mean(),
       'STD:           ',cross_val_score(pipe_knn,X,y,cv = kfolds).std())

# Let`s also fit the model and check the score with Late Submission

pred = KNeighborsClassifier(3).fit(X,y).predict(test)
write_sub(pred,'def_knn')


# Late Submission Score: 0.66435

# ### 2. RandomForestClassifier

# In[ ]:


# Because I like this classifier

from sklearn.ensemble import RandomForestClassifier

pipe_rfc = make_pipeline(trans,
                     RandomForestClassifier(n_estimators = 100, max_depth = 25))

print ('Mean Accuracy: ',cross_val_score(pipe_rfc,X,y,cv = kfolds).mean(),
       'STD:           ',cross_val_score(pipe_rfc,X,y,cv = kfolds).std())


# Let`s also fit the model and check the score with Late Submission
pred = RandomForestClassifier(n_estimators = 100, max_depth = 25).fit(X,y).predict(test)
write_sub(pred,'def_rfr')


# Late Submission Score: 0.75692

# ### 3. ExtraTreesClassifier

# In[ ]:


# To compare with random forest

from sklearn.ensemble import ExtraTreesClassifier

pipe_etc = make_pipeline(trans,
                         ExtraTreesClassifier(n_estimators = 100, max_depth = 25))

print ('Mean Accuracy: ',cross_val_score(pipe_etc,X,y,cv = kfolds).mean(),
       'STD:           ',cross_val_score(pipe_etc,X,y,cv = kfolds).std())

# Let`s also fit the model and check the score with Late Submission
pred = ExtraTreesClassifier(n_estimators = 100, max_depth = 25).fit(X,y).predict(test)
write_sub(pred,'def_etc')


# Late Submission Score: 0.77821

# ### 4. XGBClassifier

# In[ ]:


from xgboost import XGBClassifier

pipe_xgb = make_pipeline(trans,
                         XGBClassifier(n_estimators = 100, max_depth = 25))

print ('Mean Accuracy: ',cross_val_score(pipe_xgb,X,y,cv = kfolds).mean(),
       'STD:           ',cross_val_score(pipe_xgb,X,y,cv = kfolds).std())
pred = XGBClassifier(n_estimators = 100, max_depth = 25).fit(X,y).predict(test)
write_sub(pred,'def_xgb')


# Late Submission Score: 0.77824

# ### 5. VotingClassifier

# In[ ]:


from sklearn.ensemble import VotingClassifier

pipe_vc = VotingClassifier( estimators = [('rfc',RandomForestClassifier(n_estimators = 100, max_depth = 25)),
                                          ('etc',ExtraTreesClassifier(n_estimators = 100, max_depth = 25)),
                                          ('xgb',XGBClassifier(n_estimators = 100, max_depth = 25))],
                            voting = 'hard',
                            n_jobs=-1
                           )

# voting = 'soft'    is recomended with well tuned classifiers

print ('Mean Accuracy: ',cross_val_score(pipe_vc,X,y,cv = kfolds).mean(),
       'STD:           ',cross_val_score(pipe_vc,X,y,cv = kfolds).std())

# Let`s also fit the model and check the score with Late Submission
pred = pipe_vc.fit(X,y).predict(test)
write_sub(pred,'def_vc')


# Late Submission Score: 0.77766

# ## Summary
# 
# Let`s just summarise a bit what scores we have achieved with almost default hyperparameters so far.
# 
# Unfortunately Voting strategy has not improved the score significantly
# 
# Eventhough cross_val score has slightly increase but submission score has decreased.
# 
# Anyways, lets try to tune RandomForest and Extratrees with gridsearchCV

# In[ ]:


lb_score = [0.66435,0.75692,0.77821,0.77824,0.77766]
cross_acc_mean = [0.8113095238095239,0.8724206349206349,0.8791666666666667,0.8818121693121693,0.8836640211640212]
cross_acc_std  = [0.007234736333478586,0.0090432088016599,0.007320087176315713,0.00722868772751064,0.005549253164921434]
clf = ['KNN','RFC','EXT','XGB','Voting']
x = np.arange(len(clf))

fig, ax = plt.subplots()
ax.bar(x - 0.2, lb_score, 0.4,label='LB score')
ax.bar(x + 0.2, cross_acc_mean, 0.4,yerr = cross_acc_std, label='Cross val score')

ax.set_ylabel('Scores')
ax.set_title('LB and CV Scores')
ax.set_xticks(x)
ax.set_xticklabels(clf)
ax.legend(loc = 'lower right')


# # GridSearchCV
# 

# ### 1. RandomForestClassifier

# In[ ]:


param_rfc = {'randomforestclassifier__n_estimators':[100,300,500,700],
             'randomforestclassifier__max_depth':[20,50,70]
            }

search_rfc = GridSearchCV(pipe_rfc,param_rfc,cv = kfolds, scoring = 'accuracy')
search_rfc.fit(X,y)

print('Best score: ',search_rfc.best_score_)
print('Best param: ',search_rfc.best_params_)

model_rfc = search_rfc.best_estimator_


# In[ ]:


# Let`s also fit the model and check the score with Late Submission
pred = model_rfc.fit(X,y).predict(test)
write_sub(pred,'adj_rfc')


# Late Submission Score: 0.76138

# ### 2. ExtraTreesClassifier

# In[ ]:


param_etc = {'extratreesclassifier__n_estimators':[100,300,500,700],
             'extratreesclassifier__max_depth':[50,70,100]
            }

#Best param:  {'extratreesclassifier__max_depth': 70, 'extratreesclassifier__n_estimators': 300}

search_etc = GridSearchCV(pipe_etc,param_etc,cv = kfolds, scoring = 'accuracy')
search_etc.fit(X,y)

print('Best score: ',search_etc.best_score_)
print('Best param: ',search_etc.best_params_)

model_etc = search_etc.best_estimator_


# Late Submission Score: 0.78583

# In[ ]:


# Let`s also fit the model and check the score with Late Submission
pred = model_etc.fit(X,y).predict(test)
write_sub(pred,'adj_etc')


#  ### 3. XGB

# In[ ]:


param_xgb = {'xgbclassifier__n_estimators':[100,300,500,700],
             'xgbclassifier__max_depth':[50,70,100]
            }
# Best param:  {'xgbclassifier__max_depth': 50, 'xgbclassifier__n_estimators': 100}

search_xgb = GridSearchCV(pipe_xgb,param_xgb,cv = kfolds, scoring = 'accuracy')
search_xgb.fit(X,y)

print('Best score: ',search_xgb.best_score_)
print('Best param: ',search_xgb.best_params_)

model_xgb = search_xgb.best_estimator_


# In[ ]:


pred = model_xgb.fit(X,y).predict(test)
write_sub(pred,'adj_xgb')


# Late Submission Score: 0.77907

# ### 4. VotingClassifier tuned

# In[ ]:


from sklearn.ensemble import VotingClassifier

pipe_vc = VotingClassifier(estimators = [ ('rfc',RandomForestClassifier(n_estimators = 100, max_depth = 25)),
                                          ('etc',model_etc),
                                          ('xgb',model_xgb)],
                            voting = 'soft',
                            n_jobs=-1
                           )

print ('Mean Accuracy: ',cross_val_score(pipe_vc,X,y,cv = kfolds).mean(),
       'STD:           ',cross_val_score(pipe_vc,X,y,cv = kfolds).std())


# In[ ]:


pred = pipe_vc.fit(X,y).predict(test)
write_sub(pred,'adj_vc')


# In[ ]:





# In[ ]:





# In[ ]:


pipe_vc.fit(X_train,lb.inverse_transform(y_train))
print('Accuracy on TRAIN: ',pipe_vc.score(X_train,lb.inverse_transform(y_train)))
print('Accuracy on VAL  : ',pipe_vc.score(X_val,lb.inverse_transform(y_val)))

from sklearn.metrics import confusion_matrix

confusion_matrix(lb.inverse_transform(y_val),pipe_vc.predict(X_val))


# In[ ]:


pipe_vc.fit(X,lb.inverse_transform(y))


# In[ ]:


pred = pipe_vc.predict(test)


# In[ ]:


pred


# In[ ]:


sub = pd.DataFrame({'Id':test_id,'Cover_Type':pred})


sub.to_csv('sub.csv',index = False)


# In[ ]:




