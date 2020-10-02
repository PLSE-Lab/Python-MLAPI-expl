#!/usr/bin/env python
# coding: utf-8

# # Step - 1 : Frame The Problem
# 
# ## TODO:
# * Try using PClass as a numerical and categorical feature within the same model
# * Three variable layers 
#     * list of all features including synthetic like PClass as a separate features, 
#     * list of different algos
#     * list of different voting schemas
#     * pick one which provides the best prediction for particular passenger
# 
# 
# 

# # Step - 2 : Obtain the Data

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as ms

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


data = pd.read_csv('../input/train.csv')


# # Step - 3 : Analyse the Data

# In[ ]:


data.head()


# In[ ]:


data.info()


# In[ ]:


data.describe()


# In[ ]:


ms.matrix(data)


# # Step - 4 : Feature Engineering

# ## convert categorical values
# 
# ### Pclass 
# Pclass could be as one feature with a number of a class or it could be a different features for each distinct value
# 
# ### Sex
# Sex need to be a boolean value female
# 
# ### Embarked
# Embarked need to be a one feature for each embarked distinct value.
# Note: research impact of removing one particular of embarked value because it is calculable from all others
# 
# ## Syntetic features
# FamilySize of relative abourd including passanger.
# IsAlone indicates no relatives on board

# In[ ]:


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    # Categorical 
    pclass = pd.get_dummies(df['Pclass'], prefix = 'Pclass')
    embarked = pd.get_dummies(df['Embarked'], prefix = 'embarked')
    sex = pd.get_dummies(df['Sex'], prefix = 'Sex')   
    
    # Syntetic
    family = df[['SibSp', 'Parch']].copy()

    # Family size of relative abourd including passanger
    family['FamilySize'] = df['SibSp'] + df['Parch'] + 1

    # IsAlone indicates no relatives on board
    family['IsAlone'] = 1  
    family['IsAlone'].loc[family['FamilySize'] > 1] = 0 

    family.drop(['SibSp','Parch'],axis=1,inplace=True)
    
    # concat
    df = pd.concat([df, pclass, sex, embarked, family], axis=1)

    return df


# In[ ]:


data = feature_engineering(data)


# In[ ]:


ms.matrix(data)


# ## Display corelations between features

# In[ ]:


sns.heatmap(data.corr(),cmap='coolwarm')
plt.title('data.corr()')


# # Step - 5 : Model Selection

# ## Train Test Split

# In[ ]:


X = data.drop('Survived',axis=1)
Y = data['Survived']


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.20)


# ## Building a model

# ### LogisticRegression

# data contains features which could not be used with specific models (e.g. null values are not acceptable for LogisticRegression.
# Defining a function which cleans data for specific model

# In[ ]:


def clean_for_logistic_regression(df_X: pd.DataFrame, df_Y: pd.DataFrame, drop_rows: bool = False):
    '''
    Keep only numerical features.
    When the whole row is removed from df_X, than corresponded row is also removed from df_Y
    Impute age based on median of corresponded Pclass (improvements possible based on title in the name)
    Impute Fare as median of existing fares (improvements possible based on Pclass, title, Embark)
    Impute Embarked based on the most common value - mode

    Keep only these features:
        Age	SibSp	Parch 
        Fare
        Pclass_1	Pclass_2	Pclass_3	Sex_female	Sex_male	
        embarked_C	embarked_Q	embarked_S
        FamilySize	IsAlone
    '''
    features = ['Age', 
                'Sex_female', 
                'Sex_male', 
                'SibSp', 
                'Parch', 
                'FamilySize', 
                'IsAlone',
                'Fare', 
                'Pclass', 
                'Pclass_1', 
                'Pclass_2', 
                'Pclass_3',
                'embarked_C', 
                'embarked_Q', 
                'embarked_S'
               ]
    
    df_X = df_X.copy()
    
    pclass_1_median_age = df_X['Age'].loc[df_X['Pclass_1'] == 1].median()
    pclass_2_median_age = df_X['Age'].loc[df_X['Pclass_2'] == 1].median()     
    pclass_3_median_age = df_X['Age'].loc[df_X['Pclass_3'] == 1].median() 
        
    def impute_age(cols):
        Age = cols[0]
        Pclass = cols[1]

        if pd.isnull(Age):
            if Pclass == 1:
                return pclass_1_median_age
            elif Pclass == 2:
                return pclass_2_median_age
            else:
                return pclass_3_median_age
        else:
            return Age
    
    df_X['Age'] = df_X[['Age', 'Pclass']].apply(impute_age, axis=1)
    
    df_X['Embarked'].fillna(df_X['Embarked'].mode()[0], inplace = True)
    df_X['Fare'].fillna(df_X['Fare'].median(), inplace = True)
    
    df_X['SibSp'].fillna(0, inplace = True)
    df_X['Parch'].fillna(0, inplace = True)
    
    df_X = df_X[features]
    
    return df_X, df_Y    


# In[ ]:


X_train_lr, Y_train_lr = clean_for_logistic_regression(X_train, Y_train)
X_test_lr, Y_test_lr = clean_for_logistic_regression(X_test, Y_test)


# In[ ]:


from sklearn.linear_model import LogisticRegression

model = LogisticRegression(solver='lbfgs', max_iter=1000)
model.fit(X_train_lr,Y_train_lr)


# In[ ]:


Y_hat_test_lr = model.predict(X_test_lr)


# # Step - 6 : Evaluation

# ## Coefficients 
# The most biggest impact for survival were being female, first class and embarked from Cherbourg 

# In[ ]:


import operator

coef = list(zip(X_test_lr.columns.values, model.coef_[0]))
sorted(coef, key=operator.itemgetter(1), reverse=True)


# In[ ]:


## Confusion Matrix


# In[ ]:


from sklearn.metrics import confusion_matrix, classification_report


# In[ ]:


print(confusion_matrix(Y_test_lr, Y_hat_test_lr))


# In[ ]:


print(classification_report(Y_test_lr,Y_hat_test_lr))


# # Step - 7 : Predict on New Cases

# In[ ]:


prediction_data = pd.read_csv('../input/test.csv')


# In[ ]:


prediction_data.head()


# In[ ]:


ms.matrix(prediction_data)


# In[ ]:


prediction_data = feature_engineering(prediction_data)


# In[ ]:


ms.matrix(prediction_data)


# In[ ]:


X_prediction_lr, Y_prediction_lr = clean_for_logistic_regression(prediction_data, None)


# In[ ]:


ms.matrix(X_prediction_lr)


# In[ ]:


Y_prediction_lr = model.predict(X_prediction_lr)


# In[ ]:


prediction_data['Survived'] = Y_prediction_lr
submit = prediction_data[['PassengerId','Survived']]

submit.head()


# In[ ]:


submit.to_csv("../working/submit.csv", index=False)

