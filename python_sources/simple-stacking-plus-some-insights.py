#!/usr/bin/env python
# coding: utf-8

# # Roosevelt National Forest classification

# In my latest kernel https://www.kaggle.com/obiaf88/stacking-classifiers-with-gridsearch I stacked together different classifiers in order to build a better model for Roosevel Forest classification. 
# Kernel had an accuracy score of 86% in the testing dataset but only a 76,6% as public score.  
# For sure part of the difference is related to overfitting but is this the only reason for the difference in the accuracy score? 
# In this kernel I want to investigate deeper inside the results to understand if there may be others reasons.
# 

# In[ ]:


import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 70)
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,GradientBoostingClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold

import seaborn as sns
from sklearn.utils import class_weight
import warnings
warnings.simplefilter(action='ignore')

train = pd.read_csv(r'../input/learn-together/train.csv')
test = pd.read_csv(r'../input/learn-together/test.csv')


# ### Features engineering

# In[ ]:


cover_type = {1:'Spruce/Fir', 2:'Lodgepole Pine',3:'Ponderosa Pine',4:'Cottonwood/Willow',5:'Aspen',6:'Douglas-fir',7:'Krummholz'}
train['Cover_type_description'] = train['Cover_Type'].map(cover_type)

# I put together train and test to work simultaneously on both of them
combined_data = [train, test]

def distance(a,b):
    return np.sqrt(np.power(a,2)+np.power(b,2))

extremely_stony = [1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1]
stony = [0,1,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
rubbly = [0,0,1,1,1,0,0,0,0,1,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
other = [0,0,0,0,0,0,1,1,0,0,0,0,0,1,0,1,1,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0]

for data in combined_data:

    data['mean_Hillshade'] = (data['Hillshade_9am']+ data['Hillshade_Noon']+data['Hillshade_3pm'])/3
    data['Distance_to_hidrology'] = distance(data['Horizontal_Distance_To_Hydrology'],data['Vertical_Distance_To_Hydrology'])
    data['Distance_hydrology_roads'] =distance(data['Vertical_Distance_To_Hydrology'], data['Horizontal_Distance_To_Roadways'])
    data['Distance_hydrology_fire'] = distance(data['Vertical_Distance_To_Hydrology'], data['Horizontal_Distance_To_Fire_Points'])
    data['extremely_stony_level'] = data[[col for col in data.columns if col.startswith("Soil")]]@extremely_stony
    data['stony'] = data[[col for col in data.columns if col.startswith("Soil")]]@stony
    data['rubbly'] = data[[col for col in data.columns if col.startswith("Soil")]]@rubbly
    data['other'] = data[[col for col in data.columns if col.startswith("Soil")]]@other
    data['Hillshade_noon_3pm'] = data['Hillshade_Noon']- data['Hillshade_3pm']
    data['Hillshade_3pm_9am'] = data['Hillshade_3pm']- data['Hillshade_9am']
    data['Hillshade_9am_noon'] = data['Hillshade_9am']- data['Hillshade_Noon']
    data['Up_the_water'] = data['Vertical_Distance_To_Hydrology'] > 0
    data['Horizontal_plus_vertical_distance_to_hydrology'] = data['Horizontal_Distance_To_Hydrology'] + data['Vertical_Distance_To_Hydrology']
    data['Total_horizontal_distance'] = data['Horizontal_Distance_To_Hydrology']+ data['Horizontal_Distance_To_Roadways']+ data['Horizontal_Distance_To_Fire_Points']
    data['Elevation_of_hydrology'] = data['Elevation']+ data['Vertical_Distance_To_Hydrology']
    data['Elevation_of_hydrology2'] = data['Elevation']- data['Vertical_Distance_To_Hydrology']
    data['Distance_to_firepoints plus Distance_to_roads'] = data['Horizontal_Distance_To_Fire_Points']+ data['Horizontal_Distance_To_Roadways']
    data['Distance_to_roads plus distance_to_hydrology'] = data['Horizontal_Distance_To_Roadways'] + data['Horizontal_Distance_To_Hydrology']
    data['Distance_to_firepoints minus Distance_to_roads'] = data['Horizontal_Distance_To_Fire_Points']- data['Horizontal_Distance_To_Roadways']
    data['Distance_to_roads minus distance_to_hydrology'] = data['Horizontal_Distance_To_Roadways'] - data['Horizontal_Distance_To_Hydrology']
    data['Elevation_plus_slope'] = data['Elevation']+ data['Slope']
    data['Elevation_plus_aspect'] =data['Elevation']+ data['Aspect']
    data['Elevation_Aspect_Slope'] = data['Elevation']+ data['Aspect']+ data['Slope']
    data['Slope_plus_aspect'] = data['Slope']+ data['Aspect']
    data['Hillshade9_plus_hillshadenoon_plus_hillshade3'] = data['Hillshade_9am'] + data['Hillshade_Noon']+ data['Hillshade_3pm']
    data['Aspen'] = data['Soil_Type11']+data['Soil_Type13']+data['Soil_Type18']+data['Soil_Type19']+data['Soil_Type26']+data['Soil_Type30']
    data['Cottonwood/Willow'] = data['Soil_Type1']+data['Soil_Type3']+data['Soil_Type14']+data['Soil_Type17']
    data['Douglas-fir'] = data['Soil_Type5']+data['Soil_Type10']+data['Soil_Type16']
    data['Krummholz'] = data['Soil_Type35']+data['Soil_Type36']+data['Soil_Type37']+data['Soil_Type38']+data['Soil_Type39']+data['Soil_Type40']
    data['Lodgepole Pine'] = data['Soil_Type8']+data['Soil_Type9']+data['Soil_Type12']+data['Soil_Type20']+data['Soil_Type25']+data['Soil_Type28']+data['Soil_Type29']+data['Soil_Type32']+data['Soil_Type33']+data['Soil_Type34']
    data['Ponderosa Pine'] = data['Soil_Type2']+data['Soil_Type4']+data['Soil_Type6']
    data['Spruce/Fir'] = data['Soil_Type21']+data['Soil_Type22']+data['Soil_Type23']+data['Soil_Type24']+data['Soil_Type27']+data['Soil_Type31']
    data['Spruce/Fir2'] = data['Soil_Type19']+data['Soil_Type21']+data['Soil_Type22']+data['Soil_Type23']+data['Soil_Type24']+data['Soil_Type27']+data['Soil_Type31']+data['Soil_Type35']+data['Soil_Type38']+data['Soil_Type39']+data['Soil_Type40']
    data['Lodgepole Pine2'] = data['Soil_Type2']+data['Soil_Type3']+data['Soil_Type4']+data['Soil_Type6']+data['Soil_Type8']+data['Soil_Type9']+data['Soil_Type10']+data['Soil_Type11']+data['Soil_Type12']+data['Soil_Type13']+data['Soil_Type14']+data['Soil_Type16']+data['Soil_Type17']+data['Soil_Type18']+data['Soil_Type20']+data['Soil_Type25']+data['Soil_Type26']+data['Soil_Type28']+data['Soil_Type29']+data['Soil_Type30']+data['Soil_Type32']+data['Soil_Type33']+data['Soil_Type34']+data['Soil_Type36']
    data['Aspen_elev'] = data['Aspen'] * data['Elevation']
    data['Cottonwood/Willow_elev'] = data['Cottonwood/Willow'] * data['Elevation']
    data['Douglas-fir_elev'] = data['Douglas-fir'] * data['Elevation']
    data['Krummholz_elev'] = data['Krummholz'] * data['Elevation']
    data['Lodgepole Pine_elev'] = data['Lodgepole Pine'] * data['Elevation']
    data['Ponderosa Pine_elev'] =data['Ponderosa Pine'] * data['Elevation']
    data['Spruce/Fir_elev'] = data['Spruce/Fir'] * data['Elevation']
    data['Aspen_elev2'] = data['Aspen'] * data['Elevation_of_hydrology2']
    data['Cottonwood/Willow_elev2'] = data['Cottonwood/Willow'] * data['Elevation_of_hydrology2']
    data['Douglas-fir_elev2'] = data['Douglas-fir'] * data['Elevation_of_hydrology2']
    data['Krummholz_elev2'] = data['Krummholz'] * data['Elevation_of_hydrology2']
    data['Lodgepole Pine_elev2'] = data['Lodgepole Pine'] * data['Elevation_of_hydrology2']
    data['Ponderosa Pine_elev2'] =data['Ponderosa Pine'] * data['Elevation_of_hydrology2']
    data['Spruce/Fir_elev2'] = data['Spruce/Fir'] * data['Elevation_of_hydrology2']

soil_columns = [col for col in train.columns if col.startswith('Soil')]

# we can drop soil columns because we have group each soil according to their stoyness
for data in combined_data:
    data.drop(columns = soil_columns, inplace=True)

# for categorical value (up the water) we can encode them into dummy values  
columns_to_encode = ['Up_the_water']
        
train = pd.get_dummies(train,columns = columns_to_encode)
test = pd.get_dummies(test,columns = columns_to_encode)

# once we have encoded the columns we can drop them from both training and testing dataset

for data in combined_data:
    data.drop(columns= columns_to_encode, inplace=True)


# ## Model building - Random Forest classifier

# In[ ]:


target = 'Cover_Type'
features = [ col for col in train.columns if col not in ['Id','Cover_Type','Cover_type_description']]

X = train[features]
y = train[target]

# distribution of cover types in training dataset
ax = sns.countplot(x = train['Cover_type_description'])
ax.set_title("Distribution of covert type in training dataset")
plt.xticks(rotation=90)
plt.show()


# As we can see cover types are perfectly distribuited in training dataset, so it means that dataset is balanced. This is something that we need to keep in mind in the next parts of the kernel.
# 
# As the dataset is perfectly balanced we can reduce test_size to 20%; this will allow us to have more data to train the model with.
# 

# In[ ]:


X_train, X_test,y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

random_forest = RandomForestClassifier(n_estimators = 5, criterion ='entropy', max_depth =35)
random_forest.fit(X_train,y_train)

y_pred = random_forest.predict(X_test)
print("Accuracy random forest :", accuracy_score(y_test, y_pred))


# Accuracy of the model is not bad, but what is the accuracy for different cover types? 
# Are there any differences? 
# Let's see what is accuracy among cover types.

# In[ ]:


# we can put predictions in a dataframe along with the true values of cover types from testing dataset
prediction = pd.DataFrame({'y_test':y_test, 'y_pred':y_pred})
prediction['Cover_type_y_test'] = prediction['y_test'].map(cover_type)
cover_type_desc = ['Spruce/Fir','Lodgepole Pine','Ponderosa Pine','Cottonwood/Willow','Aspen','Douglas-fir','Krummholz']

for cover in cover_type_desc:
    print("Accuracy score for {} is: ".format(cover), accuracy_score(prediction[prediction['Cover_type_y_test']== cover]['y_test'],prediction[prediction['Cover_type_y_test']== cover]['y_pred']))


# As we can see accuracy is really different among cover types! 
# In particular we can see that Spruce/Fir and Lodgepole Pine have an accuracy which is by far lower than the others cover types.  This is really important. 
# Now let's look at confusion matrix.

# In[ ]:


confusion_matrix = pd.DataFrame(confusion_matrix(y_test, y_pred), index = cover_type_desc, columns= cover_type_desc)

print("Confusion matrix for whole dataset :\n", confusion_matrix)


# From confusion matrix we have a confirmation that the classification between Spruce/Fir and Lodgepole Pine is critical for the accuracy of the model.
# What are the most important features for the classification? Let's plot the top 10.

# In[ ]:


feat_importances = pd.Series(random_forest.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.show()


# All features related to elevation are really important for forests classification

# In[ ]:


#relationship for training dataset
columns = ['Elevation', 'Aspect','Slope', 'mean_Hillshade']
#for cover_type in list_of_cover_type:
sns.pairplot(train, hue = "Cover_type_description", vars = columns)
plt.legend()
plt.show()


# From the pairplot above we can see that elevation can differentiate between cover types pretty well but there are two cover types which are ovelapping: Spruce/Fir and Lodgepole Pine. They grow at the same elevation so, in order to differentiate between them, we should introduce new fetures in the dataset which are different for the cover types or use the existing ones and create new useful ones with feautures engineering.

# Now we can apply the model to test dataset and put the predictions in a dataframe

# In[ ]:


X_sub = test[features]
y_pred_sub = random_forest.predict(X_sub)
   
prediction_test = pd.DataFrame( {'prediction_test': y_pred_sub}, index=None)
prediction_test['Cover_type_description'] = prediction_test['prediction_test'].map(cover_type)


# Let's take a look at the distribution of cover types in the testing dataframe.

# In[ ]:


ax = sns.countplot(x = prediction_test['Cover_type_description'])
ax.set_title("Distribution of covert type in testing dataset prediction")
plt.xticks(rotation=90)
plt.show()


# As we can see the testing dataset is really imbalanced, unlike training dataset. 
# The most important cover types are Spuce/Fir and Lodgepole Pine which are the cover types that are more difficult to differentiate and have a lower accuracy.
# **This is the reason why our model, beside overfitting, is perfoming so bad in the testing dataset compared to training dataset!**
# Cover types counts and weights in the testing dataset are the following.
# 

# In[ ]:


cover_type_count =prediction_test.groupby(prediction_test['prediction_test'], as_index=True).agg('count')
covert_type_weights = prediction_test.groupby(prediction_test['prediction_test'],as_index=True).agg('count')/len(prediction_test)

print("Cover type count :\n")
print(cover_type_count)

print("Cover type weights:\n")
print(covert_type_weights)


# For Spuce/Fir and Lodgepole Pine weights respectively 37% and 39% of the whole testing dataset (this percentages may be slightly different according to the model but the takeaway is that testing dataset is inbalanced).
# 
# How can we overcome this?
# 
# We can build a random forest with weights in which each weight is the weight of covert type in the testing dataset. Then we can create a xgboost model and stack together the two models to have a better classification

# In[ ]:


#RANDOM FOREST

sample_weights = {1: 0.372046, 2:0.398975,3:0.064180, 4:0.003768,5:0.059755,6:0.043810,7:0.057465}
    
forest_weights = RandomForestClassifier(n_estimators = 200, criterion ='entropy', max_depth =50,class_weight=sample_weights)
forest_weights.fit(X_train, y_train)
            
y_pred = forest_weights.predict(X_test)
        
print("Accuracy random forest with weights:", accuracy_score(y_test, y_pred))


# Using random forest with weights there is a slight improvement in the accuracy of the model becasuse we don't weight each cover type  the same (balanced weight) but we weight more Spruce/Fir and Lodgepole Pine.

# In[ ]:


#XGBOOST
from xgboost import XGBClassifier 
xgbclassifier = XGBClassifier()
xgbclassifier = XGBClassifier(n_estimators=200, random_state=0,learning_rate=0.03)
xgbclassifier.fit(X_train, y_train)
y_pred = xgbclassifier.predict(X_test)

print("Accuracy xgboost:", accuracy_score(y_test, y_pred))


# In[ ]:


#extra tree classifier
extra_tree = ExtraTreesClassifier(n_estimators=200, max_depth= 30,criterion='entropy',bootstrap = True,class_weight=sample_weights)
extra_tree.fit(X_train, y_train)
y_pred = extra_tree.predict(X_test)

print("Accuracy extra tree:", accuracy_score(y_test, y_pred))


# In[ ]:


#gradient boosting
gradient_boosting = GradientBoostingClassifier(n_estimators=200,max_depth= 20)
gradient_boosting.fit(X_train, y_train)
y_pred = gradient_boosting.predict(X_test)

print("Accuracy gradient boosting:", accuracy_score(y_test, y_pred))


# In[ ]:


#STACKING MODELS TOGETHER
from vecstack import stacking

classifiers = [forest_weights,xgbclassifier,extra_tree,gradient_boosting]
X_train, X_test = train_test_split(train, test_size=0.2, random_state=5)
y_train = X_train[target]
y_test = X_test[target]

S_train, S_test = stacking(classifiers,X_train[features], y_train, test[features],regression=False,mode='oof_pred_bag',needs_proba=False,save_dir=None,metric=accuracy_score,n_folds=3,stratified=True,shuffle=True,random_state=0,verbose=2)
model = RandomForestClassifier(n_estimators = 300, criterion ='entropy', max_depth =35,class_weight=sample_weights)
model = model.fit(S_train, y_train)
y_pred = model.predict(S_test)


# In[ ]:



#submission
sub = pd.DataFrame({'Id': test.Id, 'Cover_Type': y_pred})
sub.to_csv('submission.csv', index = False)


# If you have found this kernel useful, please upvote!
