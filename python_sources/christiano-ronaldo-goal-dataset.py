#!/usr/bin/env python
# coding: utf-8

# This dataset was part of competetion problem I had faced ,This was my first model I have built from scratch so please mind the errors. 
# 
# 

# Understanding uptill now :
# 
# We have few rows with missing is_goal data which is our test data.
# We have to predict the probaability for the missing rows is_goal column with our machine learnign model
# 
# The data we will use for train is the rows which are having any particular value of is_goal present.
# 
# Now once we have segregated the train data and test data.
# 
# 
# We will divide our data into X, y i.e input and output set, 
# -->So we will put the output Y column as is_goal and for 
# -->Other columns will be our input X.
# 
# Now as the data we have in X constitutes of the columns with different data types , so we would require to do some exploratory data analysis before we could make use of the same in our model.
# 
# 
# The first thing is that : the columns pertaining to numerical value have a lot of Null(NAN) values present in them, we tend to use filler functions to treat them first like FillNa() to fill the null values with the mean average or some by default value or we can simply drop them altogether but that might lead to loss of data.
# 
# Note : We can also use SimpleImputer to transform our data .
# 
# question for myself?
# We are doing above steps in the hope of having better performance on our system i,e lesser mean absolute error, mean squared error or whichever evaluation metric we are using .
# or is it a necessity to implicate our Data into Ml Model that is it won't be working with NAN values or  throwing.
# 
# okay so once we have treated the null values with our imputer/fillna for numerical or i.e non object data type then we can work over our categorical features .
# 
# 
# As the categorical variable data also possesss significant amout of business information it would very much apt to bring our categorical into numerical so that they can inculcated to be included into our model.
# 
# We have tried using label encoder for our data which will encode each of the term with a numerical value and i have seen mean absolute error for my model hasn't reduced much after using it.
# 
# 
# For using label encoder I have filtered categorical variables and the variables with the data set similar in train , validation and test are good to be taken for label encoding.
# 
# 

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

data = pd.read_csv('../input/data.csv',index_col=0)

data['shot_id_number'] = data.index + 1  # shot id number has null value

sample = pd.read_csv('../input/sample_submission.csv',index_col=0)

data_train_test_combined =  data[data['is_goal'].notnull()]
test_file = data[data['is_goal'].isnull()] #.dropna(axis=0,subset=['shot_id_number'])


# In[ ]:


data_train_test_combined.dropna(axis=0,subset=['is_goal'],inplace=True) #Not required
y = data_train_test_combined.is_goal
X = data_train_test_combined.drop(['is_goal'],axis =1 ) #.select_dtypes(exclude =['object'])


# In[ ]:


train_X,test_X,train_y,test_y = train_test_split(X,y,test_size=0.25)
X_test = test_file.drop(['is_goal'],axis =1)


# In[ ]:


from sklearn.impute import SimpleImputer
my_imputer = SimpleImputer()
imputed_X_train = my_imputer.fit_transform(train_X.select_dtypes(exclude='object'))
imputed_X_valid = my_imputer.transform(test_X.select_dtypes(exclude='object'))
imputed_final_test = my_imputer.transform(X_test.select_dtypes(exclude='object'))


# In[ ]:


# pd.DataFrame(imputed_X_train,columns=train_X.select_dtypes(exclude ='object').columns) #imputer return numpy ndarray
# pd.DataFrame(imputed_X_train)

# pd.DataFrame(imputed_X_train,columns=train_X.select_dtypes(exclude ='object').columns)


# In[ ]:


# All categorical columns
object_cols = [col for col in train_X.columns if train_X[col].dtype == "object"]

# Columns that can be safely label encoded
good_label_cols = [col for col in object_cols if 
                   set(train_X[col]) == set(X_test[col])]
        
# Problematic columns that will be dropped from the dataset
bad_label_cols =  list (set(object_cols) - set(good_label_cols))
        
print('Categorical columns that will be label encoded:', good_label_cols)
print('\nCategorical columns that will be dropped from the dataset:', bad_label_cols)


# In[ ]:


# Label Encoding 
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
# Drop categorical columns that will not be encoded
label_X_train = train_X.drop(bad_label_cols, axis=1)
label_X_valid = test_X.drop(bad_label_cols, axis=1)
label_final_X_test = X_test.drop(bad_label_cols,axis=1)

# Apply label encoder 
label_encoder = LabelEncoder()


for col in set(good_label_cols):
    label_X_train[col] = label_encoder.fit_transform(label_X_train[col].astype(str))
    label_X_valid[col] = label_encoder.transform(label_X_valid[col].astype(str))
    label_final_X_test[col] = label_encoder.transform(label_final_X_test[col].astype(str))


# In[ ]:


Numerical = pd.DataFrame(imputed_X_train,columns=train_X.select_dtypes(exclude ='object').columns)
Numerical_V = pd.DataFrame(imputed_X_valid,columns=test_X.select_dtypes(exclude='object').columns)
Numerical_T = pd.DataFrame(imputed_final_test,columns=X_test.select_dtypes(exclude='object').columns)
Numerical_T.columns


# In[ ]:


#pd.concat(label_X_train,pd.DataFrame(imputed_X_train,columns=train_X.select_dtypes(exclude ='object').columns))
Categorical = label_X_train[good_label_cols]
Categorical_V = label_X_valid[good_label_cols]
Categorical_T = label_final_X_test[good_label_cols]
Categorical_T.columns


# In[ ]:


#model_train_x = pd.concat([label_X_train[good_label_cols],imputed_X_train],ignore_index=True,sort=False)
# pd.concat([data1, data2], ignore_index=True, sort =False)
Model_train_X = pd.concat([Numerical.reset_index() ,Categorical.reset_index() ],ignore_index=True,sort=False,axis=1) # Number of rows were increasing
Model_valid_X = pd.concat([Numerical_V.reset_index(),Categorical_V.reset_index()],ignore_index=True,sort=False,axis=1) 
Model_test_X = pd.concat([Numerical_T.reset_index(),Categorical_T.reset_index()],ignore_index=True,sort=False,axis=1)


# Not used the feature engineering but for the next 2 section of code I have plotted feature engineering.

# In[ ]:


#Model_train_X.shape
#feature engineering
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
sel = SelectFromModel(RandomForestClassifier(n_estimators = 100))
sel.fit(Model_train_X, train_y)
sel.get_support()
selected_feat= Model_train_X.columns[(sel.get_support())]
len(selected_feat)
print(selected_feat)



# In[ ]:


def plot_feature_importances(clf, X_train, y_train=None, 
                             top_n=10, figsize=(8,8), print_table=False, title="Feature Importances"):
    '''
    plot feature importances of a tree-based sklearn estimator
    
    Note: X_train and y_train are pandas DataFrames
    
    Note: Scikit-plot is a lovely package but I sometimes have issues
              1. flexibility/extendibility
              2. complicated models/datasets
          But for many situations Scikit-plot is the way to go
          see https://scikit-plot.readthedocs.io/en/latest/Quickstart.html
    
    Parameters
    ----------
        clf         (sklearn estimator) if not fitted, this routine will fit it
        
        X_train     (pandas DataFrame)
        
        y_train     (pandas DataFrame)  optional
                                        required only if clf has not already been fitted 
        
        top_n       (int)               Plot the top_n most-important features
                                        Default: 10
                                        
        figsize     ((int,int))         The physical size of the plot
                                        Default: (8,8)
        
        print_table (boolean)           If True, print out the table of feature importances
                                        Default: False
        
    Returns
    -------
        the pandas dataframe with the features and their importance
        
    Author
    ------
        George Fisher
    '''
    
    __name__ = "plot_feature_importances"
    
    import pandas as pd
    import numpy  as np
    import matplotlib.pyplot as plt
    
    from xgboost.core     import XGBoostError
    from lightgbm.sklearn import LightGBMError
    
    try: 
        if not hasattr(clf, 'feature_importances_'):
            clf.fit(X_train.values, y_train.values.ravel())

            if not hasattr(clf, 'feature_importances_'):
                raise AttributeError("{} does not have feature_importances_ attribute".
                                    format(clf.__class__.__name__))
                
    except (XGBoostError, LightGBMError, ValueError):
        clf.fit(X_train.values, y_train.values.ravel())
            
    feat_imp = pd.DataFrame({'importance':clf.feature_importances_})    
    feat_imp['feature'] = X_train.columns
    feat_imp.sort_values(by='importance', ascending=False, inplace=True)
    feat_imp = feat_imp.iloc[:top_n]
    
    feat_imp.sort_values(by='importance', inplace=True)
    feat_imp = feat_imp.set_index('feature', drop=True)
    feat_imp.plot.barh(title=title, figsize=figsize)
    plt.xlabel('Feature Importance Score')
    plt.show()
    
    if print_table:
        from IPython.display import display
        print("Top {} features in descending order of importance".format(top_n))
        display(feat_imp.sort_values(by='importance', ascending=False))
        
    return feat_imp

import pandas as pd
# X_train = pd.DataFrame(X)
# y_train = pd.DataFrame(y)

from xgboost              import XGBClassifier
from sklearn.ensemble     import ExtraTreesClassifier
from sklearn.tree         import ExtraTreeClassifier
from sklearn.tree         import DecisionTreeClassifier
from sklearn.ensemble     import GradientBoostingClassifier
from sklearn.ensemble     import BaggingClassifier
from sklearn.ensemble     import AdaBoostClassifier
from sklearn.ensemble     import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm             import LGBMClassifier


clfs = [XGBClassifier(),              LGBMClassifier(), 
        ExtraTreesClassifier(),       ExtraTreeClassifier(),
        BaggingClassifier(),          DecisionTreeClassifier(),
        GradientBoostingClassifier(), LogisticRegression(),
        AdaBoostClassifier(),         RandomForestClassifier()]

for clf in clfs:
    try:
        _ = plot_feature_importances(clf, Model_train_X, train_y, top_n=Model_train_X.shape[1], title=clf.__class__.__name__)
    except AttributeError as e:
        print(e)


# In[ ]:


my_model = XGBRegressor(n_estimators=1000)
my_model.fit(Model_train_X,train_y)
from sklearn.metrics import mean_absolute_error
mean_absolute_error(my_model.predict(Model_valid_X),test_y)


# I had tried impelementing the model alone on the categorical variable where performance is also more or less the same.

# In[ ]:


my_model = XGBRegressor(n_estimators=1000)
my_model.fit(label_X_train[good_label_cols],train_y)
from sklearn.metrics import mean_absolute_error
mean_absolute_error(my_model.predict(label_X_valid[good_label_cols]),test_y)
#ans = my_model.predict(label_final_X_test[good_label_cols])
# ans[:5]


# In[ ]:


test_preds = my_model.predict(label_final_X_test[good_label_cols])
output = pd.DataFrame({'shot_id_number': X_test['shot_id_number'],
                       'is_goal': test_preds})
output.loc[output['is_goal'] < 0,'is_goal'] = 0
output.to_csv('submission.csv',index=False)


# In[ ]:


ans = pd.read_csv('submission.csv')


# In[ ]:




ans[ans['is_goal'] < 0] 


# In[ ]:


ans.describe()


# This was the first model I had built but the performance was too low so tried on the top with label encoders but again the performance had been just satisfactory uptill now. 

# If someone has any decent suggestions over here which can lead to significant improvement in the model performance are also welcome
