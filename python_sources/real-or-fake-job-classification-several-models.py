#!/usr/bin/env python
# coding: utf-8

# # Real of Fake Job Descriptions using Several Models
# 
# Thanks for taking the time to look at my kernel. I am relatively new to ML and data science, so all criticism is welcome.
# 
# Currently I only have a Logistic Regression Model, but I plan to add more!
# I tried to build a pipeline to do all my tasks, but I couldn't get it to work (so I commented it out). It should still provide the benefit of pipelines
# in terms of readability so you can better understand what I did.

# In[ ]:


import numpy as np
import pandas as pd

import os

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        file_path = dirname +'/'+ filename

# Read file
df = pd.read_csv(file_path)

df.head()


# # Doing some Preprocessing

# In[ ]:


df.isnull().sum()


# Going through the dataframe, I notice some missing values and some useless columns that won't help with predicting the target.
# 
# Remove useless columns:
# 

# In[ ]:


df = df.drop(
        ['job_id', 'title', 'location', 'department', 'salary_range', 'company_profile', 'description', 'requirements',
         'benefits'], axis=1).sort_index()


# Personally, I really wanted to get a system going to clean up the location column (because I was thinking of one hot encoding the country, state, and city, but
# I was not sure if that would help the model or not. 

# Set my target variable

# In[ ]:


y = df['fraudulent']

y


# In[ ]:


X = df.drop(['fraudulent'], axis=1)

X.head()


# I was going to have a beautifully organized pipeline going with the plan of imputing all the missing numericals and
# one hot encoding all my categoricals.

# In[ ]:


# Different features for my model
numerical_features = ['telecommuting', 'has_company_logo', 'has_questions']
label_features = ['employment_type', 'required_experience', 'required_education', 'industry', 'function']


# As stated before, we saw some nan's in columns that consisted of strings, so this is how I imputed them:
# - Sidenote, what methods do you guys prefer for imputering? I looked at sklearn_pandas.CategoricalImputer, but it didn't work. So I did it manually

# In[ ]:


for feature in label_features:
    X[feature].replace(np.nan, X[feature].mode()[0], regex=True, inplace=True)
    
X.head()


# Here is my pipeline stuff if you are curious

# In[ ]:


'''
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median'))])

    categorical_transformer = Pipeline(steps=[
        ('cat_imputer', OneHotEncoder())])

    preprocessing = ColumnTransformer(transformers=[
        ('numerical', numeric_transformer, numerical_features),
        ('categorical', categorical_transformer, label_features)
    ])

    log_reg = Pipeline(steps=[
        ('preprocessing', preprocessing),
        ('scaler', StandardScaler(with_mean=False)),
        ('log', LogisticRegression())
    ])
    '''


# Manually doing what I intended for the pipeline to do:

# # Logistic Regression

# In[ ]:


from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_absolute_error
from imblearn.under_sampling import RandomUnderSampler
from sklearn.decomposition import PCA


# In[ ]:


imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

imputer.fit_transform(X[numerical_features])

c_t = make_column_transformer((OneHotEncoder(), label_features), remainder='passthrough')

big_X = c_t.fit_transform(X).toarray()

sc = StandardScaler()

big_X = sc.fit_transform(big_X)

pca = PCA()

big_X = pca.fit_transform(big_X)

rus = RandomUnderSampler()

undersampled_x, y = rus.fit_resample(big_X,y)

x_train, x_test, y_train, y_test = train_test_split(undersampled_x, y, test_size=0.2, random_state=0)

log_reg = LogisticRegression()

log_reg.fit(x_train, y_train)

y_pred = log_reg.predict(x_test)

print(confusion_matrix(y_test, y_pred))
print(f'Prediction score: {log_reg.score(x_test, y_test) * 100:.2f}%')
print(f'MAE from Logistic Regression: {mean_absolute_error(y_test, y_pred) * 100:.2f}%')


# # K-Nearest Neighbors with Grid Search and Cross Validation
# To optimize K, I used GridSearchCV neighbors 1-30 and used cross validation kfold=5 to reduce variablity.
# 
# Import what we need:

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV


# Now let's scale X and to prepare for fitting

# In[ ]:


knn = KNeighborsClassifier()

grid = GridSearchCV(knn, param_grid={'n_neighbors':range(1,31)}, scoring='accuracy')

grid.fit(undersampled_x,y)

for i in range(0, len(grid.cv_results_['mean_test_score'])):
    print('N_Neighbors {}: {} '.format(i+1, grid.cv_results_['mean_test_score'][i]*100))


# OK scores. I used GridSearch and Cross Validation to help my model scores out a bit. There wasn't much variance though.

# # Random Forests Classification with Cross Validation
# Let's import what we need:

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


# In[ ]:


rf = RandomForestClassifier(bootstrap=True)

rf.fit(undersampled_x,y)

# Cross validation of 5 folds
score = cross_val_score(rf, undersampled_x, y)

print(f'Prediction score: {np.mean(score) * 100:.2f}%')


# So far, this is the best scoring model I have created for this problem. Before cross validation, I attempted random forests and scored 92%. I was suspicious of this result, so I added CV and my scores shot down to ~82% on average. While this seems "bad" in a way,
# compared to the other models it is not! This model does not shoot down under 80%, while my other models' scores vary from 76%-83%.  
# I suspect that the nature of the Random Forest being able to generalize more data is the reason why it scores higher than the others. What do you think about this claim?

# Overall, I think the undersampling hurt my scores significantly. Does anyone have any suggestions? 
# 
# Thanks for reading and give me your thoughts!
