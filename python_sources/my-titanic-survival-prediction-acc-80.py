#!/usr/bin/env python
# coding: utf-8

# # #Exploring the titanic Dataset
# * Introduction
# * Load and check the dataset
# * Feature Engineering
# * Finding which is the best model
# * Model creation
# * checking accuracy
# * Hyper parameter Tunning
# * Submission
# 
# 
# 
# 
# 

# #  Introduction
#      This is my first kaggle notebook,In this notebook we going to predict the survival of the people in the titanic disaster.
# 

# # Load and check the Dataset
# Importing necessary Libraries for our model
# 

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno
import numpy as np
from sklearn.preprocessing import OneHotEncoder
# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)


# In[ ]:


dataset=pd.read_csv("../input/titanic/train.csv")
dataset.head()


# # Exploritary Data Analysis

# In[ ]:


missing_data=missingno.matrix(dataset,figsize = (10,10))
missing_data


# In[ ]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Pclass',data=dataset,palette='rainbow')


# * In the above graph ) represent passanger not survived,One represent passanger survived.The passanger who where in the first class has a high survival rate.

# In[ ]:


sns.boxplot(x='Pclass',y='Age',data=dataset)


# # Feature Engineering

# * Filling values in the missing area

# In[ ]:


dataset['Sex'] = np.where(dataset['Sex'] == 'female', 1, 0)


# 
# * We have to fill the missing age data with the help of box plot in EDA

# In[ ]:


def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):

        if Pclass == 1:
            return 37

        elif Pclass == 2:
            return 29

        else:
            return 24

    else:
        return Age
    
dataset['Age'] =dataset[['Age','Pclass']].apply(impute_age,axis=1)


# In[ ]:


dataset.Age.isnull().sum()


# In[ ]:



df_sex_one_hot = pd.get_dummies(dataset['Embarked'], 
                                prefix='Embarked')


# Combine the one hot encoded columns with df_con_enc
dataset= pd.concat([dataset,df_sex_one_hot],axis=1)
                       

# Drop the original categorical columns (because now they've been one hot encoded)
dataset=dataset.drop(['Embarked'], axis=1)


# In[ ]:


dataset.isnull().sum()


# * In Cabin there are Lots of missing  so we remove the cabin column,Then we dont need Ticket and Name  this are not very important for our model.

# In[ ]:


dataset=dataset.drop(['Cabin','Ticket','Name'],axis=1)


# In[ ]:


dataset.isnull().sum()
dataset.shape


# In[ ]:


cs=dataset
cs.to_csv('../formulated_train.csv', index=False)
print('modified train CSV is ready!')


# * We finish the handling  the train dataset now we have to do the same thing to the test data

# In[ ]:


test_dataset=pd.read_csv("../input/titanic/test.csv")
test_dataset.head()


# In[ ]:


missing_data=missingno.matrix(test_dataset,figsize = (10,10))
missing_data


# In[ ]:


test_dataset['Sex'] = np.where(test_dataset['Sex'] == 'female', 1, 0)


def impute_age1(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):

        if Pclass == 1:
            return 37

        elif Pclass == 2:
            return 29

        else:
            return 24

    else:
        return Age
    
test_dataset['Age'] =test_dataset[['Age','Pclass']].apply(impute_age1,axis=1)


# In[ ]:


# One hot encode the columns in the test data frame (like X_train)
test_embarked_one_hot = pd.get_dummies(test_dataset['Embarked'], 
                                       prefix='Embarked')


# In[ ]:


# Combine the test one hot encoded columns with test
test_dataset = pd.concat([test_dataset, 
                  test_embarked_one_hot], axis=1)


# In[ ]:


test_dataset=test_dataset.drop(['Embarked','Name','Cabin','Ticket'], axis=1)
test_dataset['Fare'].fillna(np.mean(test_dataset['Fare']), inplace=True)


# In[ ]:


print(test_dataset.isnull().sum())

test_dataset.shape


# In[ ]:


#train_df=pd.concat([df['SalePrice'],train_df],axis=1)
csv=test_dataset
csv.to_csv('../formulated_test.csv', index=False)
print('modified test CSV is ready!')


# * Finally our Feature engineering part was over next step we have to develope our model

# # Finding which is the best model
# 
# * Idid some cross validation then I choose Decission tree

# # Model creation
# 

# In[ ]:


df=cs #cs is a formulated train dataset
x=df.drop(['Survived'],axis=1)

y=df['Survived']


# In[ ]:


from sklearn.model_selection import train_test_split 
from sklearn import tree

# Decision Tree Classifier

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

                                           
model = tree.DecisionTreeClassifier()
model.fit(x_train,y_train)
 

predictions = model.predict(x_test)
predictions


# # checking accuracy

# In[ ]:



from sklearn.metrics import confusion_matrix

accuracy=confusion_matrix(y_test,predictions)

print("confusion_matrix:",accuracy)

from sklearn.metrics import accuracy_score

accuracy=accuracy_score(y_test,predictions)
print("accuracy_score:",accuracy)


# * we have to reduce the type 1and type 2 error for that we need hypertunning our model

# In[ ]:



from sklearn.model_selection import RandomizedSearchCV

from scipy.stats import randint

est=tree.DecisionTreeClassifier()
rf_p_dist={'max_depth':[3,5,10,None],
           
           'max_features':randint(1,10),
           'criterion':['entropy','gini'],
          
           'min_samples_leaf':randint(1,4),
           
        
    
        }
def hypertunning_rscv(est,p_distr,nbr_iter,x,y):
    rdmsearch=RandomizedSearchCV(est,param_distributions=p_distr,
                                 n_jobs=-1,n_iter=nbr_iter,cv=9)
    rdmsearch.fit(x,y)
    ht_params=rdmsearch.best_params_
    ht_score=rdmsearch.best_score_
    return ht_params,ht_score


rf_parameters,rf_ht_score=hypertunning_rscv(est,rf_p_dist,40,x,y)

rf_parameters


# In[ ]:


# after hyper parameter tunning

from sklearn.model_selection import train_test_split 
from sklearn import tree

# Decision Tree Classifier

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

                                           
model = tree.DecisionTreeClassifier(criterion= 'gini',
                                             max_depth= 3,
                                             max_features= 9,
                                             min_samples_leaf= 1,
                                             splitter='best')
model.fit(x_train,y_train)
 # Cross Validation 

predictions = model.predict(x_test)
#print("preddicted value",pcriterion='gini'redictions)
#print(len(predictions))

from sklearn.metrics import confusion_matrix

accuracy=confusion_matrix(y_test,predictions)

print("confusion_matrix:",accuracy)

from sklearn.metrics import accuracy_score

accuracy=accuracy_score(y_test,predictions)
accuracy


# * after the hyperparameter tunning false negative little bit reduced and the accuracy was increased

# In[ ]:


test_df=csv


prediction = model.predict(test_df)
prediction


# In[ ]:


# Create a submisison dataframe and append the relevant columns
submission = pd.DataFrame()
submission['PassengerId'] = test_df['PassengerId']
submission['Survived'] = prediction # our model predictions on the test dataset
submission.head()


# Let's convert our submission dataframe 'Survived' column to ints
submission['Survived'] = submission['Survived'].astype(int)
print('Converted Survived column to integers.')


# In[ ]:





# Are our test and submission dataframes the same length?
if len(submission) == len(test_df):
    print("Submission dataframe is the same length as test ({} rows).".format(len(submission)))
else:
    print("Dataframes mismatched, won't be able to submit to Kaggle.")

# Convert submisison dataframe to csv for submission to csv 
# for Kaggle submisison
submission.to_csv('../submission1.csv', index=False)
print('Submission CSV is ready!')

