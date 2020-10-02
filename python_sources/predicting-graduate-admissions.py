#!/usr/bin/env python
# coding: utf-8

# ## Predicting Graduate Admissions
# 
# The dataset contains records of 500 students and their chance of admit.
# 
# The following features are given in the dataset:
# 
# 1. GRE Score (out of 340) - The student's GRE Score
# 
# 
# 2. TOEFL Score (out of 120) - The student's TOEFL score
# 
# 
# 3. University rating (1 to 5) - The student's undergrad university rating
# 
# 
# 4. SOP (1 to 5) - The student's SOP rating
# 
# 
# 5. LOR (1 to 5) - The student's LOR rating
# 
# 
# 6. CGPA (out of 10) - The student's undergrad CGPA
# 
# 
# 7. Research (0 or 1) - If the student has any research experience or not
# 
# 
# 8. Chance of Admit (0 to 1) - The probability of the student getting an admit

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_style('whitegrid')


# In[ ]:


data = pd.read_csv('../input/Admission_Predict_Ver1.1.csv')
data.drop('Serial No.', axis=1, inplace=True)


# In[ ]:


data.head()


# In[ ]:


data.describe()


# In[ ]:


plt.hist(data['GRE Score'], bins=25)
plt.show()


# The majority of students score between 310 and 330. This is pretty common since anything lower than 310 reduces the chances of the student getting an admit and anything above 330 will greatly increase the chances of getting an admit.

# In[ ]:


plt.scatter(data['GRE Score'], data['TOEFL Score'])
plt.show()


# The general trend is the student getting better score in GRE also tend to perform better in TOEFL

# In[ ]:


plt.scatter(data['GRE Score'], data['CGPA'])
plt.show()


# Students with higher CGPA have a higher score in GRE.

# In[ ]:


sns.pairplot(data)


# In[ ]:


corr = data.corr()
sns.heatmap(corr, annot=True)


# From the above two graphs, we can see that there is a high correlation between CGPA and the chances of getting an admit. Intrestingly, having researh experience is not that important.

# ## Modeling

# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler


# In[ ]:


X = data.drop('Chance of Admit ', axis=1).copy()
y = data['Chance of Admit '].copy()


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# - Using Linear regression

# In[ ]:


lr = LinearRegression()


# In[ ]:


lr.fit(X_train, y_train)


# In[ ]:


lr_pred = lr.predict(X_test)
r2_score(y_test, lr_pred)


# - Using Random forest

# In[ ]:


rfr = RandomForestRegressor(n_estimators=100)


# In[ ]:


rfr.fit(X_train, y_train)


# In[ ]:


rfr_pred = rfr.predict(X_test)
r2_score(y_test, rfr_pred)


# Checking the importances of the different parameters

# In[ ]:


imp = pd.DataFrame(sorted(zip(rfr.feature_importances_, X_train.columns), reverse=True), columns=['Importance', 'Feature'])
plt.figure(figsize=(12,6))
sns.barplot(imp['Feature'], imp['Importance'])


# The model must be tuned so that the model does not depend too much on just one feature

# In[ ]:


scalerX = MinMaxScaler(feature_range=(0, 1))
X_train[X_train.columns] = scalerX.fit_transform(X_train[X_train.columns])
X_test[X_test.columns] = scalerX.transform(X_test[X_test.columns])


# In[ ]:


rfr = RandomForestRegressor(n_jobs = -1)
param_grid = {'n_estimators': [200, 500, 800, 1000], 
                    'max_depth': [4, 5, 6, 7], 
                    'min_samples_split': [2, 3, 4, 5],
                    'max_features': [1,2,3,4,5,6,7]}
rfr_grid = GridSearchCV(estimator = rfr, param_grid = param_grid, cv = 3,n_jobs = -1)
rfr_grid.fit(X_train,y_train)


# In[ ]:


rfr_grid.best_params_


# In[ ]:


rfr.set_params(**rfr_grid.best_params_)


# In[ ]:


rfr.fit(X_train, y_train)


# In[ ]:


rfr_pred = rfr.predict(X_test)
r2_score(y_test, rfr_pred)


# In[ ]:


imp = pd.DataFrame(sorted(zip(rfr.feature_importances_, X_train.columns), reverse=True), columns=['Importance', 'Feature'])
plt.figure(figsize=(12,6))
sns.barplot(imp['Feature'], imp['Importance'])


# ## Conclusion
# 
# It is interesting to see that the prebability of getting an admit depends majorly on the student's CGPA, GRE and TOEFL scores. SOP and LOR have relatively lower importance and the research experience has the lowest importance of all.
