#!/usr/bin/env python
# coding: utf-8

# # Titanic: Machine Learning from Disaster

# This is my solution for this Kaggle competition. To keep the kernel short I have removed the EDA, because that was already presented in other kernels. 
# 
# I use a simple XGBoost model. I use feature_selection from scikit-learn to improve the accuracy of the model.

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectFromModel

from sklearn.preprocessing import LabelEncoder

import xgboost as xgb


# In[ ]:


import os
print(os.listdir("../input"))


# In[ ]:


df_train=pd.read_csv("../input/train.csv")
df_test=pd.read_csv('../input/test.csv')


# 
# ### Missing values

# In[ ]:


print('Size of the training set',df_train.shape)
print('Size of the test set',df_test.shape)
missing =  pd.DataFrame(data = {'Missing in training set': df_train.isnull().sum(), 
                        'Missing in test set': df_test.isnull().sum()})
missing


# ### Feature engineering

# Concatenate the train and test sets and save the target variable in y.

# In[ ]:


s = df_train.shape[0]
y = df_train['Survived']
df_train = df_train.drop(columns = 'Survived')

X = pd.concat([df_train, df_test], ignore_index=True)


# Extract the title Mr., Mrs. etc. from the name. I have kept the titles Mr., Mrs., Master. and Miss. The other titles I have renamed according to the gender either as Mr. or Mrs. (first I have checked that there are no children in these categories). 

# In[ ]:


X['Title'] = X['Name'].apply(lambda x: x.partition(',')[-1].split()[0])

X['Title'][(X['Title'] != 'Mr.') & (X['Title'] != 'Mrs.')  &
           (X['Title'] != 'Master.') & (X['Title'] != 'Miss.') & (X['Sex'] == 'male')] = 'Mr.'
X['Title'][(X['Title'] != 'Mr.') & (X['Title'] != 'Mrs.')  & 
           (X['Title'] != 'Master.') & (X['Title'] != 'Miss.') & (X['Sex'] == 'female')] = 'Mrs.'


# Next I have addressed the missing values in the Cabin feature. First I have extracted the deck from the as being the first character in the Cabin feature. Then I have looked at groups of people that have bought the tickets together and made the assumption that people who travel together most likely stay close to each other on the same deck (I have checked and in the available data this is true except for one case). Unfortunately the wast majority of the Cabin information is missing so I have managed to recover only a handful of values. I have assigned the deck value N to the rest of the missing values. (After some analysis I have decided to completely remove the Cabin feature)

# In[ ]:


ind=X['Cabin'][X['Cabin'].notnull()].index.values
X['Cabin'][ind] = X['Cabin'][ind].apply(lambda x:  x[0])

groups = X.groupby('Ticket').count()
groups_with_cabin = groups[groups['Cabin'] != 0]
groups_with_cabin['Diff'] = groups_with_cabin['Fare'] - groups_with_cabin['Cabin']

ind=groups_with_cabin[(groups_with_cabin['Diff']>0)].index
cabins = X[(X['Ticket'].isin(ind))].sort_values('Ticket')
cabins = cabins.groupby('Ticket')['Cabin'].transform(lambda x: x.fillna(x.mode()[0]))
X.set_value(cabins.index,'Cabin', cabins.values)
X['Cabin'].fillna('N',inplace=True);


# As most adult males did not survive but young boys did, I have calculated the average age of children and adults, by grouping the age according to title, Master. for children, Mr. for adults.  For female passengers it is more complicated as the title 'Miss.' belongs to both girls and unmarried women. Here I made the assumption that the if a Miss. was traveling with her parents it was a child if not it was an adult. This is approximately true, but there were also adults traveling with there parents and children traveling with companions, who were not listed in the ParCh feature. 
# 
# Next I have filled the remaining missing values with the average age of male and female passengers. 

# In[ ]:


boy = np.round(X['Age'][(X['Name'] == 'Master.')].mean(),decimals = 1)
X['Age'][(X['Name'] == 'Master.')] = X['Age'][(X['Name'] == 'Master.')].fillna(boy)

girl = np.round(X['Age'][(X['Parch'] > 0) & (X['Name'] == 'Miss.')].mean(),decimals = 1)
X['Age'][(X['Parch'] > 0) & (X['Name'] == 'Miss.')] = X['Age'][(X['Parch'] > 0) & (X['Name'] == 'Miss.')].fillna(girl)
       
unmarried_female = np.round(X['Age'][(X['Parch'] == 0) & (X['Name'] == 'Miss.')].mean(),decimals = 1)
X['Age'][(X['Parch'] == 0) & (X['Name'] == 'Miss.')] = X['Age'][(X['Parch'] == 0) & (X['Name'] == 'Miss.')].fillna(unmarried_female)
        
#Fill the remaining missing values with the average age of male and female pessangers
mean_age = X[(X['Name'] != 'Miss.') & (X['Name'] != 'Master.')].groupby('Sex').mean()
X['Age'][(X['Sex'] == 'female')] = X['Age'][(X['Sex'] == 'female')].fillna(np.round(mean_age['Age']['female'], decimals = 1))
X['Age'][(X['Sex'] == 'male')] = X['Age'][(X['Sex'] == 'male')].fillna(np.round(mean_age['Age']['male'], decimals = 1))


# I have filled the missing values in Embarked by the most popular value and the missing values in Fare as the average as a function of passenger class. 

# In[ ]:


X['Embarked'].fillna(X['Embarked'].value_counts().idxmax(),inplace=True)


# In[ ]:


X['Fare'] = X.groupby('Pclass')['Fare'].transform(lambda x: x.fillna(x.mean()))


# Now remove the features that I do not need.

# In[ ]:


X.drop(columns=['PassengerId', 'Name','Cabin', 'Ticket'], inplace=True)


# In[ ]:


X.head()


# There is one last transformation needed before we can train our model. There are a couple of features that contain string values and usually ML models cannot work with them. We need to transform them in numeric values either by label encoding or one-hot-encoding (OHE). As I am planning to use a tree based model (XGboost) label encoding is very well suited for the job. (I have also tried OHE but it gave me a smaller accuracy).  

# In[ ]:


def label_encoding(df):
    obj = [var for var in df.columns if df[var].dtype=='object']
    for c in obj:
        le=LabelEncoder()
        df[c]=le.fit_transform(df[c])
    return df


# In[ ]:


X = label_encoding(X)
X.head(3)


# As a last step before the modeling we split the dataset into the training and the test set. 

# In[ ]:


X_train = X.iloc[:s, :]
X_test = X.iloc[s:, :]


# ### Modeling

# I have used the XGBoost classifier to obtain the feature importance as assigned by this model. Then the features are sorted by importance and the model is trained and cross-validated on smaller and smaller feature number of features. 

# In[ ]:


param = {'max_depth': 4, 'learning_rate': 0.01, 'n_estimators': 500, 'objective': 'binary:logistic',
        'reg_alpha': 0, 'reg_lambda': 0, 'seed': 1}

xgb_model = xgb.XGBClassifier(**param)

# fit model on all training data
xgb_model.fit(X_train, y)
sc = cross_val_score(xgb_model, X_train, y, cv = 10)
print("Accuracy: %.2f%%" % (sc.mean() * 100.0))

# Fit model using each importance as a threshold
thresholds = np.sort(xgb_model.feature_importances_)
for thresh in thresholds:
    # select features using threshold
    selection = SelectFromModel(xgb_model, threshold=thresh, prefit = True)
    select_X_train = selection.transform(X_train)
    # train model
    selection_model =xgb.XGBClassifier(**param)
    sc = cross_val_score(selection_model, select_X_train, y, cv = 10) 
    print("Thresh=%.3f, number of features=%d, Accuracy: %.2f%%" % (thresh, select_X_train.shape[1], sc.mean()*100.0))


# We observe that the maximum accuracy is not obtained using all the features, but using the 5 most important features.  
# 
# First I stor the feature names and the importance given to them by XGBoost in a dataframe and sort it in a descending order, so the most important feature is on top. Then I select the number of features that give the highest accuracy, 5.

# In[ ]:


feat_imp = pd.DataFrame(data = {'Feature': X_train.columns, 'Importance': xgb_model.feature_importances_})
feat_imp.sort_values(by=['Importance'], ascending = False, inplace = True)
selected_features = feat_imp['Feature'][:5].values
print(selected_features)
feat_imp


# Take only the selected features from the train and test sets and train the model again, then predict the survival in the test set.

# In[ ]:


X_train = X_train[selected_features]
X_test = X_test[selected_features] 


# In[ ]:


xgb_model.fit(X_train, y)
survival = xgb_model.predict(X_test)


# In[ ]:


submission = pd.DataFrame(data = {'PassengerId': df_test['PassengerId'], 'Survived': survival})
submission.head(5)


# In[ ]:


submission.to_csv('mysubmission.csv', index = False)


# In[ ]:




