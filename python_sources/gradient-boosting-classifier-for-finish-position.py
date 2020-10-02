#!/usr/bin/env python
# coding: utf-8

# # Predicting Greyhound Finishing Position, 1st to 6th.
# ## 1. Description.
# ### There are 2,000 races in the dataset. Crayford 380 metre races only. The dataset can be used to predict the race winner or the finishing position of each greyhound. This notebook will be used to create a classification model of greyhound finish postion.
# ### Research Question:
# ### Can the model outperform the market in predicting the finish position of greyhounds in competitive six-runner races, 1st to 6th?

# ## 1.1 Loading Packages and Data

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing
from sklearn.ensemble import  GradientBoostingClassifier # classifier


# In[ ]:


df = pd.read_csv("../input/greyhound-racing-uk-predict-finish-position/data_final.csv")


# ## 1.2 Exploring the dataset

# In[ ]:


df.head()


# In[ ]:


df.info()


# ## 1.3 Building the classification model, predicting finish position 1st to 6th.
# >  I will be creating a multiclass classifier, using sklearn's Gradient Boosting Classifier. The target variable is 'Finished', 1 to 6 (1st to 6th).  
# >  The variable 'Race_ID' (this is for Identification only) will not be used. 'Winner', which is the binary classification target variable, 1 to 0 (Win/lose), will also be removed as this is clearly cheating.
# >  
# 

# ## 1.3.1 The features / predictor variables

# In[ ]:


# Features
features = ['Trap', 'BSP', 'Time_380', 'Finish_Recent', 'Finish_All', 'Stay_380',            'Races_All','Odds_Recent','Odds_380', 'Distance_Places_All', 'Dist_By',            'Races_380', 'Odds','Last_Run','Early_Time_380', 'Early_Recent' ,            'Distance_All', 'Wins_380', 'Grade_380','Finish_380','Early_380',            'Distance_Recent', 'Public_Estimate','Wide_380', 'Favourite']
# Target
target = ['Finished']


# In[ ]:


df[features].corr()


# ### There are three odds related features, 'BSP','Odds' & 'Public_Estimate'.
# > These are highly correlated, see correlation matrix below.
# > Betfair Starting Price (BSP) is most highly correlated with the target 'Finished', this feature will be kept.
# > I will remove the other two odds features, 'Odds' and 'Public_Estimate' from the list of features to use in my prediction model.

# In[ ]:


df[['BSP','Odds','Public_Estimate','Finished']].corr()
features.remove('Odds')
features.remove('Public_Estimate')
print(features)
print("\nThere are now",len(features),"features remaining.")


# ### 1.3.2 Splitting the data, train & test.

# In[ ]:


train=df.sample(frac=0.80,random_state=10) #random state is a seed value
test=df.drop(train.index)


# In[ ]:


# train_X, train_y
train_X = train[features]
train_y = train[target]

# test_X, test_y
test_X = test[features]
test_y = test[target]


# ### 1.3.3 Train the model

# In[ ]:


# Create model
model = GradientBoostingClassifier(n_estimators = 10, max_features = None, min_samples_split = 2)
model.fit(train_X, train_y.values.ravel())


# In[ ]:


# evaluate the model on TRAINING DATA
accuracy = model.score(train_X, train_y)
print('    Training Model Accuracy:    ' + str(round(accuracy*100,2)) + '%')


# ### 1.3.5 Test the Model

# In[ ]:


# evaluate the model on Test data
accuracy = model.score(test_X, test_y)
print('    Test Model Accuracy:  ' + str(round(accuracy*100,2)) + '%')


# ### 1.3.6 Calculate Market Predictions of Greyhound Finishing Position
# > Does the model beat the market in predicting finish positon?
# > Yes, sligthly.

# In[ ]:


# evaluate the market on Test data
# the feature 'Public_Estimate' gives the market prediction of finish position for each greyhound.
market_data = list(zip(test['Public_Estimate'], test['Finished']))
total = len(list(market_data))
count=0
for val in market_data:
    if val[0] == val[1]:
        count+=1
print('    Test Market Accuracy:      ' + str(round(count/total,3)*100) + '%')  # - - test   


# ### 2. Conclusion

# #### The model outperformed the market in predicting greyhound finishing position, slightly. 
# #### Model Accuracy = 22.12%
# #### Market Accuracy = 20.9%
