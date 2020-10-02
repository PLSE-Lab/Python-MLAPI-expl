#!/usr/bin/env python
# coding: utf-8

# **Predicting cab booking cancellations**
# 
# In this kernal, I am going to predict cab booking cancellations. Ensemble learning methods like Bagging Classifier and Voting Classifiers are used to boost the accuracy of predictions. If you are new to the words "Ensemble Learning", it's advisable to visit this link -  [Ensemble Learning Sklearn](http://scikit-learn.org/stable/modules/ensemble.html). 
# 
# **Contents**
# 1. Import Libraries (Includes ensemble learners).
# 2. Load the dataset and understanding the dataset.
# 3. Data Cleaning.
# 4. Feature Importance.
# 5. Feature Heatmap.
# 6. Ensemble Learning Methods
# 7. Predictions.
# 

# **1. Import Libraries**
# 
# * Import the libraries needed to work on the dataset.

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, mean_squared_error
from sklearn.linear_model import LogisticRegression


# **2. Load Dataset and undertanding the dataset**
# 
# * Load the dataset and look at the shape and null values. Null values are needed for basic data cleaning.

# In[ ]:


df = pd.read_csv('../input/Kaggle_YourCabs_training.csv')
print (df.columns)

print (df.shape)
df.isnull().sum()


# **3. Data Cleaning**
# 
# * Drop the columns that are not contributing the predictions. These are to be choosen on the basic observations and null value observations.

# In[ ]:


df['from_area_id'] = df['from_area_id'].fillna(value = np.mean(df['from_area_id']))


# In[ ]:


df = df.drop(['package_id', 'to_city_id', 'from_city_id', 'from_date', 'to_date', 'from_lat', 
              'from_long', 'to_lat', 'to_long', 'to_area_id', 'id', 'booking_created'], 
             axis = 1)

df.isnull().sum()


# In[ ]:


X = df[['Car_Cancellation']]
y = df[['vehicle_model_id', 'travel_type_id', 'online_booking', 
        'mobile_site_booking']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)


# **4. Feature Importance**
# 
# * Feature Importance is calculated using Random Forest Classifier. It can be clearly observed that "Cost of Error" is out performer.

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')

rf = RandomForestClassifier()
rf.fit(y.values, X.values.ravel())

importance = rf.feature_importances_
importance = pd.DataFrame(importance, index = y.columns, columns=['Importance'])

feats = {}
for feature, importance in zip(y.columns,rf.feature_importances_):
    feats[feature] = importance
    
print (feats)
importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 
                                                                            'Gini-importance'})
importances.sort_values(by='Gini-importance').plot(kind='bar', rot=45)


# **5. Feature HeatMap**
# 
# * Correlation of features on each other can be observed using seaborn. It could be observed that most of the features are independent of others making it easy to predict.

# In[ ]:


y_cols = y.columns.tolist()
corr = df[y_cols].corr()

sns.heatmap(corr)


# **6. Ensemble Learning Models**
# 
# * Logistic Regression, Random Forest Classifier, Bagging Classifier and Support Vector Machine are used vote with Voting Classifier, hard voting is made. This model is used to fit and predict data in the following sections.

# In[ ]:


lr = LogisticRegression()
rf = RandomForestClassifier(n_estimators=10)
bg = BaggingClassifier(DecisionTreeClassifier(), max_samples = 0.5)
clf = SVC(kernel = 'linear')

evc = VotingClassifier(estimators =[('lr', lr),('rf', rf),('bg', bg),('clf', clf)], 
                       voting = 'hard')


# **7. Making Predictions and Testing**
# 
# * The model obtained in the above section is used to fit and predict data.
# * Mean Square error and Confusion matrix are provided to give error magnitudes.

# In[ ]:


evc.fit(y_train, X_train)

predicted_data = evc.predict(y_test)
print ('Score of the Model:')
print (evc.score(y_test, X_test))
print ('Confusion Matrix:')
print (confusion_matrix(X_test, predicted_data))

