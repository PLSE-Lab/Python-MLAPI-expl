#!/usr/bin/env python
# coding: utf-8

# ### Hello everyone! In this kernel, I will show that we underestimate linear regression in vain.

# ### This is a beginner's guide.

# In[ ]:


import pandas as pd

import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import cross_val_score


# In[ ]:


df = pd.read_csv('../input/graduate-admissions/Admission_Predict_Ver1.1.csv')


# In[ ]:


df.head()


# First of all, let's preprocess our data a bit.

# In[ ]:


df = df.drop(columns=['Serial No.']) # because it's absolutely doesn't important


# In[ ]:


df = df.rename(columns={
    'GRE Score': 'GRE',
    'TOEFL Score': 'TOEFL',
    'University Rating': 'UR',
    'Chance of Admit ': 'Chance',
    'LOR ': 'LOR'
})


# In[ ]:


df.head()


# That's better! Now let's see the correlation table of features.

# In[ ]:


df.corr()


# In[ ]:


sns.heatmap(df.corr());


# We can see that CGPA is the most correlated feature. Let's check the graph.

# In[ ]:


sns.lineplot(x="CGPA", y="Chance", data=df);


# If you look at all the dependencies, you will see that basically there is a linear relationship between the features. This suggests the use of linear regression.

# In[ ]:


sns.lineplot(x="TOEFL", y="GRE", data=df);


# In[ ]:


sns.lineplot(x="CGPA", y="GRE", data=df);


# In[ ]:


sns.lineplot(x="SOP", y="LOR", data=df);


# Now let's check how good RandomForestRegressor is compared to linear regression, as RandomForestRegressor shows himself very well in other tasks.

# In[ ]:


X, y = df.drop(columns=['Chance']), df['Chance']


# In[ ]:


rfr = RandomForestRegressor(random_state=42)
lr = LinearRegression()


# In[ ]:


params = {
    'n_estimators': range(10, 51, 10),
    'max_depth': range(1, 13, 2),
    'min_samples_leaf': range(1, 8),
    'min_samples_split': range(2, 10, 2)
}


# In[ ]:


search = GridSearchCV(rfr, params, cv=10, n_jobs=-1)


# In[ ]:


search.fit(X, y)


# In[ ]:


search.best_params_


# In[ ]:


best_rfr = search.best_estimator_


# Also you can see that CGPA is very important feature! Hmm...

# In[ ]:


imp = pd.DataFrame(best_rfr.feature_importances_, index=X.columns, columns=['importance'])
imp.sort_values('importance').plot(kind='barh', figsize=(12, 8))


# And now the moment of truth. Let's see the result.

# In[ ]:


cross_val_score(best_rfr, X, y, cv=10).mean()


# In[ ]:


cross_val_score(lr, X, y, cv=10).mean()


# The **conclusion** is: always look at the dependencies between the features! **We underestimate linear regression!**
# 
# Thank you for attention!
# Good luck
