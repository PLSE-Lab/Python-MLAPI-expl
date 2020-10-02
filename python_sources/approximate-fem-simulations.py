#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # Data
# Load Data and initialize environment

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import PowerTransformer
from sklearn.pipeline import make_pipeline
import xgboost as xgb

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

n_jobs = -2
cv=4

df_1000 = pd.read_csv("/kaggle/input/fem-simulations/1000randoms.csv")

test_size = 0.25
random_state = 42
features = ['ecc', 'N', 'gammaG', 'Esoil', 'Econc', 'Dbot', 'H1', 'H2', 'H3']
label = ['Mr_t', 'Mt_t', 'Mr_c', 'Mt_c']

X = df_1000[features]
y = df_1000[label]


# # Holdout
# Split Data

# In[ ]:


# take only the first label ('Mr_t'), for demonstration
X_train, X_test, y_train, y_test = train_test_split(
        X, y.iloc[:,0],
        test_size = test_size,
        random_state = random_state
    )


# # Configuration
# Configuration of Machine Learning Method and Optimization Methodology

# In[ ]:



pipe = make_pipeline(
    PowerTransformer(),
    xgb.XGBRegressor()
)

param_grid = {
    pipe.steps[-1][0] + '__' + 'max_depth': np.arange(2,6),
    pipe.steps[-1][0] + '__' + 'n_estimators': np.arange(100,1001,100)
}


# # Train and Optimize

# In[ ]:


model = GridSearchCV(
        pipe,
        param_grid = param_grid,
        n_jobs = n_jobs,
        cv = cv,
        verbose = 1
    )
model.fit(X_train, y_train)
print('--> best params:', model.best_params_)


# # Predict and Evaluate Performance

# In[ ]:


y_hat = model.predict(X_test)

results = pd.DataFrame(
            {
                'R-Squared TRAIN': [r2_score(y_train, model.predict(X_train))],
                'R-squared TEST': [r2_score(y_test, y_hat)],
                'MAE TEST': [mean_absolute_error(y_test, y_hat)],
                'MSE TEST': [mean_squared_error(y_test, y_hat)],
                'RMSE TEST': [np.sqrt(mean_squared_error(y_test, y_hat))]
            },
            index=[pipe.steps[-1][0]]
        ).T
results


# # Comment
# The performance of the model is significantly good for a Machine Learning project.
# This may result from the nature of the problem: The FEM-Simulation produces a model itself. So this Model also contains a artificial "behaviour" or "status" of the object of investigation. Therefore it might be very easy for the ML-method to imitate this FEM-model.
# 
# Besides: I have just done something similar for my masters thesis and therefore have looked out for different solutions how this problem can be handled. This was of course not only about finding any ML-Method that performs good, because this will always be something mainstream like XGB ;) But I also searched for ways to express and handle uncertainty of these results. And also tried to find out how many samples will actually be necessary to produce stable results. This may become interesting depending on the number of objects one will have to create FEM-Models on. And since those can be comutational expensive, you might also want to reduce the number of FEM-simulations.

# # Additional Performance Test
# While the Model was trained with cross validation and the holdout / splitting approach, there is also another possibility in this specific case. The thousand samples can be used for training and then test the model on the bigger (5000 samples). The assumption here is that all the samples are made with the same FEM-model. So that some of the dataset with 1000 samples might be included in the 5000 samples, but even though: using the 5000 samples dataset for testing will at least have 4000 unknown labels. So lets do that:

# In[ ]:


df_5184 = pd.read_csv("/kaggle/input/fem-simulations/5184doe.csv")

X_additional = df_5184[features]
y_additional = df_5184[label]
y_additional = y_additional.iloc[:,0]

y_hat_additional = model.predict(X_additional)

results = pd.DataFrame(
            {
                'R-squared': [r2_score(y_additional, y_hat_additional)],
                'MAE': [mean_absolute_error(y_additional, y_hat_additional)],
                'MSE': [mean_squared_error(y_additional, y_hat_additional)],
                'RMSE': [np.sqrt(mean_squared_error(y_additional, y_hat_additional))]
            },
            index=['additional ' + pipe.steps[-1][0]]
        ).T
results


# Compared to the previous they are significantly worse. But still, tey are good quite good. You can compare it to the span (6.28) of the values and determine the proportional loss (4%):

# In[ ]:


span = max(y_additional) - min(y_additional)
rmse = np.sqrt(mean_squared_error(y_additional, y_hat_additional))
rmse_prop = 100 * (rmse / span)

print("span: ", span)
print("RMSE: ", np.sqrt(mean_squared_error(y_additional, y_hat_additional)))
print("therefore, the proportional RMSE is: {:.2f}%".format(rmse_prop))

