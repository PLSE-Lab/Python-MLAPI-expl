#!/usr/bin/env python
# coding: utf-8

# Import statements

# In[ ]:


import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
import dill


# Reading the Version 1.1 of the dataset
# 
# The Serial No. can be set as the index as it does not have any other purpose
# 
# Renaming the columns which had extra white spaces

# In[ ]:


data = pd.read_csv('../input/Admission_Predict_Ver1.1.csv').set_index('Serial No.')
data = data.rename(columns={'Chance of Admit ': 'Chance of Admit', 'LOR ': 'LOR'})


# In[ ]:


print('The dataset has {} rows.'.format(len(data)))
print('The dataset has {} columns'.format(data.columns))


# In[ ]:


print(data.info())


# In[ ]:


print(data.isnull().any())


# None of the values have any Null values

# Checking the correlation between all the features

# In[ ]:


sns.heatmap(data.corr(), annot=True)


# The Chance of Admit strongly correlates with CGPA, GRE Score and TOEFL Score. It has a weaker correlation with University Rating, SOP and LOR. The Research has the least correlation.

# Let us look at individual features and their properties

# The Research feature is a binary value and will not show any significant results in the PairPlot. It can be dropped

# In[ ]:


sns.pairplot(data.drop(columns='Research'))


# The PairPlot gives a lot of information in one plot. Let us disect it by each feature.
#     1. As GRE Score increases, the TOEFL Scores also increases and even the CGPA Score increases. This means that students who score good GRE Scores also score good TOEFL Scores and have better CGPAs.
#     2. GRE Score has a distribution with a mean near 320 and majority of students score between 310 to 330 which is indicated by the peaks in the graph.
#     3. Students from higher University Rating have higher GRE Scores, TOEFL Scores and CGPA.
#     4. Students have better LOR and SOP rating when they are from higher rated universities.
#     5. We can see a similar trend for relation between GRE Scores, TOEFL Scores, CGPA Scores, LOR Ratings and SOP Ratings. They all increases together, i.e., as students with better SOPs also tend to have better LORs, better GRE, TOEFL and CGPA Scores and are from higher rated universities and they have a higher chance of admit.

# Splitting the data into training and testing set

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(data.drop(columns='Chance of Admit'), data['Chance of Admit'], test_size=0.2)


# In[ ]:


X_train.describe()


# The ranges of all the features are very distinct and should be scaled correctly before making any predictions

# In[ ]:


dfm = X_train.melt(var_name='columns')
g = sns.FacetGrid(dfm, col='columns')
g = (g.map(sns.distplot, 'value'))


# In[ ]:


scaler = StandardScaler().fit(X_train)
X_train_norm = scaler.transform(X_train)
X_test_norm = scaler.transform(X_test)
X_train = pd.DataFrame(X_train_norm, columns=X_train.columns)
X_test = pd.DataFrame(X_test_norm, columns=X_test.columns)


# In[ ]:


dfm = X_train.melt(var_name='columns')
g = sns.FacetGrid(dfm, col='columns')
g = (g.map(sns.distplot, 'value'))


# We can see that all the values have 0 mean and have a normal distribution. This enables the model to make better predictions.

# Creating a Grid Search for Random Forest Regressor

# In[ ]:


gridsearch = GridSearchCV(estimator=RandomForestRegressor(),
                          param_grid={
                              'n_estimators': [50, 100, 250, 300],
                              'max_depth': [None, 100, 200, 300, 400]
                          },
                          cv=3,
                          return_train_score=False,
                          scoring='r2')
gridsearch.fit(X=X_train, y=y_train)
pd.DataFrame(gridsearch.cv_results_).set_index('rank_test_score').sort_index()


# Taking a look at the Grid Search results, we can see that the Random Forest Regressor with a maximum depth of 200 and 300 estimators gave the best results. But, when we look more closely, we can see that the regressor with maximum depth as None and number of estimators as 50 give almost the same result (difference is 0.001465) but takes less than half the time to make the predictions. If time is of concern, this model should be picked.

# Because Random Forest is "random", the results and the inference above might change when this Notebook is run when making the commit.

# Creating a pipeline

# In[ ]:


pipe = make_pipeline(scaler, gridsearch)


# This pipeline can be used to deploy this model. It contains the complete model which scales the input data and runs it through the Random Forest to get a prediction.

# Testing out the predictions of the pipeline and the model trained above.

# In[ ]:


print('Original model: ' + str(gridsearch.predict(X=scaler.transform(data.drop(columns='Chance of Admit').iloc[0].values.reshape(1, -1)))[0]))
print('Pipeline model: ' + str(pipe.predict(X=data.drop(columns='Chance of Admit').iloc[0].values.reshape(1, -1))[0]))


# The predictions are the same. We can now export the pipeline.

# In[ ]:


with open('rf_v1.pkl', 'wb') as f:
    dill.dump(pipe, f)


# In[ ]:


with open('rf_v1.pkl', 'rb') as f:
    model = dill.load(f)
    print(model.predict(X=data.drop(columns='Chance of Admit').iloc[0].values.reshape(1, -1))[0])


# We can see that the model is exported correctly and when imported, gives the same prediction for the same data.
