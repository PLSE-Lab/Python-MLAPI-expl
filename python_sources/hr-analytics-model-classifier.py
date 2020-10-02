#!/usr/bin/env python
# coding: utf-8

# **Aim**:            
#     Company needs your help in identifying the eligible candidates at a particular checkpoint 
#     so that they can expedite the entire

# In[ ]:


# import libraries for data exploration
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %config InlineBackend.figure_Format='retina'
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


# let's load the dataset
hr = pd.read_csv(r'train_LZdllcl (1).csv')
bckup = hr.copy()
hr.head()


# In[ ]:


hr.info() # basic descr


# Our dataset contains 54808 entries and 14 features

# ### Attribute information

# **Dependent variables**:       
# 
#     Employee id - Unique ID for employee
#     Department - Department of employee
#     Region - Region of employment (unordered)
#     Education - Education Level
#     Gender - Gender of Employee
#     Recruitment Channel - Channel of recruitment for employee
#     No of trainings - no of other trainings completed in previous year on soft skills, technical skills etc.
#     Age - Age of Employee
#     Previous year rating - Employee Rating for the previous year
#     Length of service - Length of service in years
#     KPIs met >80% ? - if Percent of KPIs(Key performance Indicators) >80% then 1 else 0
#     Awards won? - if awards won during previous year then 1 else 0
#     Avg training score - Average score in current training evaluations
# 
# **Target variable**:               
# 
#     Is promoted ? - (Target) Recommended for promotion

# ### Data Type      
# **Object**
# 
#     - department              
#     - region                  
#     - education               
#     - gender                  
#     - recruitment_channel 
#     
# **Numeric**              
#     - employee_id
#     - no_of_trainings
#     - age
#     - previous_year_rating    
#     - length_of_service       
#     - KPIs_met >80%           
#     - awards_won?             
#     - avg_training_score      
#     - is_promoted             

# ### Feature category        
# **Categorical**       
#     - department              
#     - region                  
#     - education               
#     - gender                  
#     - recruitment_channel 
#     - no_of_trainings
#     - age
#     - previous_year_rating 
#     - KPIs_met >80%
#     - awards_won?
#     - is_promoted 
#     - 
# **Continuous**    
#     - length_of_service
#     - avg_training_score
#     

# ## Exploratory Data Analysis

# Let's drop unnecessary features.

# In[ ]:


hr = hr.drop('employee_id', axis=1)


# ### Let's check the balance of the dataset

# In[ ]:


hr['is_promoted'].value_counts()


# In[ ]:


sns.countplot(hr['is_promoted'])


# As we can figure out, only few candidates will be considered for promotion, and that makes sense. But, in order to get 
# accurate results, we may need to handle this imbalance data.

# ### Handling missing values

# In[ ]:


hr.isnull().sum()


# we have missing values in 'Region' and 'Previous year rating' column

# In[ ]:


hr['education'].value_counts()


# In[ ]:


hr['previous_year_rating'].value_counts()


# In[ ]:


labels = hr['is_promoted'].copy()
hr = hr.drop('is_promoted', axis=1)


# In[ ]:


# let's impute the missing values with mode and median value for now.


# In[ ]:


from sklearn.impute import SimpleImputer


# In[ ]:


imputer = SimpleImputer(strategy='most_frequent')


# In[ ]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


# In[ ]:


num_pipline = Pipeline([  # create pipelines for feature transformations
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])


# In[ ]:


cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder())
])


# In[ ]:


num_attribs = list(hr.select_dtypes(include=np.number))
cat_attribs = list(hr.select_dtypes(exclude=np.number))


# In[ ]:


full_pipeline = ColumnTransformer([
    ('num_attribs', num_pipline, num_attribs),
    ('cat_attribs', cat_pipeline, cat_attribs)
])


# In[ ]:


hr_prepared = full_pipeline.fit_transform(hr)


# In[ ]:


hr_prepared.shape


# After one hot encoding, our dimensions has increased from 14 to 58.

# In[ ]:


# hr_prepared = pd.DataFrame(hr_prepared, columns=list(hr))


# In[ ]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(hr_prepared, labels, test_size=0.2, random_state=42)


# In[ ]:


X_train.shape, X_test.shape, y_train.shape, y_test.shape


# ### Model building

# **Logistic Regression**

# **Without treating imbalance**

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, f1_score # choosing confusion matrix and F1 score


# In[ ]:


log_reg = LogisticRegression()

log_reg.fit(X_train, y_train)


# In[ ]:


predicted = log_reg.predict(X_test)


# In[ ]:


print(confusion_matrix(y_test, predicted))


# We could see that, the model couldn't classify more than half of the data is missclassified in true negative region.

# In[ ]:


print(classification_report(y_test, predicted))


# In[ ]:


f1_score(y_test, predicted)


# Our classification results are poor for predicting, whether a customer has got promotion.

# **Imbalance handling**

# Since under sampling leads to loss of data, let's try first with over sampling method SMOTE.

# In[ ]:


from imblearn.over_sampling import SMOTE


# In[ ]:


sm = SMOTE()


# In[ ]:


X, y = sm.fit_resample(hr_prepared, labels)


# **After handling imbalance**

# In[ ]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[ ]:


new_model = LogisticRegression()
new_predict = new_model.fit(X_train, y_train).predict(X_test)


# In[ ]:


print(confusion_matrix(y_test, new_predict))


# In[ ]:


print(classification_report(y_test, new_predict))


# In[ ]:


f1_score(y_test, new_predict)


# As we can see, the f score has greatly improved after handling imbalance, from 30% to 80%. Let's see we can improve
# the score with other models.

# **Cat Boost**

# We used one hot encoding to encode the categorical features in the data, let's use Cat boost algorithm, which has in built 
# mechanism to handle categorical features.

# In[ ]:


hr.isnull().sum()


# In[ ]:


cat_hr = bckup.copy()

education_mode = cat_hr['education'].mode()[0]
pyr_median = cat_hr['previous_year_rating'].median()

cat_hr['education'].fillna(education_mode, inplace=True)
cat_hr['previous_year_rating'].fillna(pyr_median, inplace=True)


# In[ ]:


# cat_hr = bckup.dropna(how='any')


# In[ ]:


cat_hr.isnull().sum().sum()


# In[ ]:


cat_hr.drop('employee_id', axis=1, inplace=True)
# cat_hr.head()


# In[ ]:


from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTENC


# To resample data with categorical features, we are using SMOTENC, which is designed to handle both categorical and
# numerical features.

# In[ ]:


sm = SMOTENC(categorical_features=[0, 1, 2, 3, 4]) # categorical feature column index are given as input


# In[ ]:


X = cat_hr.drop('is_promoted', axis=1)
y = cat_hr['is_promoted'].copy()

X, y = sm.fit_resample(X, y)


# In[ ]:


cat_hr.columns


# In[ ]:


X = pd.DataFrame(X, columns=['department', 'region', 'education', 'gender', 'recruitment_channel',
       'no_of_trainings', 'age', 'previous_year_rating', 'length_of_service',
       'KPIs_met >80%', 'awards_won?', 'avg_training_score'])


# In[ ]:


np.unique(y, return_counts=True)


# Now, our data is balanced.

# In[ ]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[ ]:


X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[ ]:


new_cbr = CatBoostClassifier(verbose=400, eval_metric='F1')


# In[ ]:


new_cbr.fit(X_train, y_train,cat_features=[0,1,2,3,4],eval_set=(X_test, y_test),plot=True) #index of cat. features are mentioned


# Our accuracy has greatly increased from 80% to 96%, using Cat boost.

# In[ ]:


test_data = pd.read_csv(r'test_2umaH9m.csv') 
# test_data.head()


# In[ ]:


# test_data_prepared = full_pipeline.fit_transform(test_data)


# In[ ]:


test_data.isnull().sum()


# ### Prepare test data

# In[ ]:


test_data['education'].fillna(test_data['education'].mode()[0], inplace=True)
test_data['previous_year_rating'].fillna(test_data['previous_year_rating'].median(), inplace=True)


# In[ ]:


final = test_data.drop('employee_id', axis=1)


# In[ ]:


cbr_predicted = new_cbr.predict(final)

cbr_predicted = pd.DataFrame(cbr_predicted, columns=['is_promoted'])

df = pd.concat([test_data['employee_id'], cbr_predicted], axis=1)


# ### Final prediction

# In[ ]:


df.to_csv(r'C:\Users\gokul\Downloads\results.csv')


# In[ ]:




