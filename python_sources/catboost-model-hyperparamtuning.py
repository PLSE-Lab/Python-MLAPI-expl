#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Model Evaluation
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import shap

# Hyperparam Tuning
from skopt import dummy_minimize
from skopt import gp_minimize

# Models used for the analysis
import catboost
from catboost import Pool

# Version check
print(f'catboost_version: {catboost.__version__}')


# In[ ]:


path = '../input/covid19/Kaggle_Sirio_Libanes_ICU_Prediction.xlsx'
data = pd.read_excel(path)
data.shape


# In[ ]:


# Corr matrix
corr = data.corr()

# Select only variables with +.20 possitive corr, and -.20 negative corr!
teste = corr[(corr['ICU'] >= 0.20)|(corr['ICU'] <= -0.20)]
teste.index.values


# In[ ]:


# window var added with the top 62 corr vars
top_62_vars = ["AGE_ABOVE65", "ALBUMIN_MEDIAN", "ALBUMIN_MEAN", "ALBUMIN_MIN",
       "ALBUMIN_MAX", "BE_ARTERIAL_MEDIAN", "BE_ARTERIAL_MEAN",
       "BE_ARTERIAL_MIN", "BE_ARTERIAL_MAX", "BE_VENOUS_MEDIAN",
       "BE_VENOUS_MEAN", "BE_VENOUS_MIN", "BE_VENOUS_MAX",
       "HEMATOCRITE_MEDIAN", "HEMATOCRITE_MEAN", "HEMATOCRITE_MIN",
       "HEMATOCRITE_MAX", "HEMOGLOBIN_MEDIAN", "HEMOGLOBIN_MEAN",
       "HEMOGLOBIN_MIN", "HEMOGLOBIN_MAX", "LACTATE_MEDIAN", "LACTATE_MEAN",
       "LACTATE_MIN", "LACTATE_MAX", "LEUKOCYTES_MEDIAN", "LEUKOCYTES_MEAN",
       "LEUKOCYTES_MIN", "LEUKOCYTES_MAX", "NEUTROPHILES_MEDIAN",
       "NEUTROPHILES_MEAN", "NEUTROPHILES_MIN", "NEUTROPHILES_MAX",
       "UREA_MEDIAN", "UREA_MEAN", "UREA_MIN", "UREA_MAX",
       "BLOODPRESSURE_DIASTOLIC_MEAN", "RESPIRATORY_RATE_MEAN",
       "BLOODPRESSURE_DIASTOLIC_MEDIAN", "RESPIRATORY_RATE_MEDIAN",
       "BLOODPRESSURE_DIASTOLIC_MIN", "HEART_RATE_MIN", "TEMPERATURE_MIN",
       "OXYGEN_SATURATION_MIN", "BLOODPRESSURE_SISTOLIC_MAX", "HEART_RATE_MAX",
       "RESPIRATORY_RATE_MAX", "OXYGEN_SATURATION_MAX",
       "BLOODPRESSURE_DIASTOLIC_DIFF", "BLOODPRESSURE_SISTOLIC_DIFF",
       "HEART_RATE_DIFF", "RESPIRATORY_RATE_DIFF", "TEMPERATURE_DIFF",
       "OXYGEN_SATURATION_DIFF", "BLOODPRESSURE_DIASTOLIC_DIFF_REL",
       "BLOODPRESSURE_SISTOLIC_DIFF_REL", "HEART_RATE_DIFF_REL",
       "RESPIRATORY_RATE_DIFF_REL", "TEMPERATURE_DIFF_REL",
       "OXYGEN_SATURATION_DIFF_REL", "ICU", "WINDOW"]

top_vars = data.loc[:,top_62_vars]


# In[ ]:


x = top_vars.drop('ICU', axis=1)
y = top_vars['ICU']

X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.2, 
                                                  random_state=0)


# In[ ]:


# selectin all the features execept for the y variable
features = [feat for feat in list(top_vars) 
            if feat != 'ICU']

categorical_features1 = np.where(top_vars[features].dtypes != np.float)[0]

clf = catboost.CatBoostClassifier(
    iterations=10,
    random_seed=301,
    learning_rate=0.3,
    custom_loss=['AUC', 'Accuracy']
)

clf.fit(X_train, y_train, 
        cat_features=categorical_features1,
        eval_set=(X_val, y_val), 
        verbose=1,
        use_best_model=True, 
        plot=True
)


# In[ ]:


preds = clf.predict(X_val)
print(confusion_matrix(y_val, preds))
print(accuracy_score(preds, y_val) * 100)


# In[ ]:


# same for the full data
features= [feat for feat in list(data) 
            if feat != 'ICU']

categorical_features = np.where(data[features].dtypes != np.float)[0]

# Select the full raw data for split
x_f_data = data[features]
y_f_data = data[['ICU']]

X_f_train, X_f_val, y_f_train, y_f_val = train_test_split(
                                                  x_f_data, y_f_data, 
                                                  test_size=0.2, 
                                                  random_state=0)


# In[ ]:


model_f = catboost.CatBoostClassifier(
    iterations=100,
    random_seed=42,
    learning_rate=0.2, 
    custom_loss=['AUC', 'Accuracy']
)

model_f.fit(X_f_train, y_f_train, 
        cat_features=categorical_features,
        eval_set=(X_f_val, y_f_val), 
        verbose=100,
        plot=True
)
print('CatBoost model is fitted: ' + str(clf_f.is_fitted()))
print('CatBoost model parameters:')
print(clf_f.get_params())


# In[ ]:


preds_f = model_f.predict(X_f_val)
print(confusion_matrix(y_f_val, preds_f))
print(accuracy_score(preds_f, y_f_val) * 100)


# In[ ]:


# Tip: for model hyper Parameter tuning you can use '?your model',
# it shows all the params that it uses
get_ipython().run_line_magic('pinfo', 'catboost.CatBoostClassifier')


# In[ ]:


def train_model(params):
    learning_rate = params[0]
    subsample = params[1]
    min_child_samples = params[2]
    depth = params[3]
    
    print(params, '\n')
    
    model = catboost.CatBoostClassifier(learning_rate=learning_rate, depth=depth,
                                        min_child_samples=min_child_samples,
                                        subsample=subsample, random_seed=42,
                                        custom_loss=['AUC', 'Accuracy'])
    
    
    model.fit(X_f_train, y_f_train,
             cat_features=categorical_features,
             eval_set=(X_f_val, y_f_val), 
             verbose=100
             #plot=True,
             )
    
    predh_f = model.predict(X_f_val)
    
    return -accuracy_score(predh_f, y_f_val)

# space of hyperparamtuning
space = [(1e-3, 1e-1, 'log-uniform'), # Learning_rate
         (0.05, 1.0), #Subsample
         (1, 100),    #Min_child_samples
         (4, 10)      #Depth
        ]

result = dummy_minimize(train_model, space, 
                        random_state=1, verbose=1,
                        n_calls=10)


# In[ ]:


result.x
# [0.09871192514273254, 0.935929491371726, 10, 7] - acc(92.47%)


# In[ ]:


# Bayesian Optimization
results_gp = gp_minimize(train_model, space, random_state=1, n_calls=15, n_random_starts=5)


# In[ ]:


results_gp.x
# [0.07431528396574379, 0.8539953708517165, 32, 7] - acc(91.68%)


# In[ ]:


shap_values = model_f.get_feature_importance(Pool(X_f_val, label=y_f_val,
                                                cat_features=categorical_features),
                                                type="ShapValues")

expected_value = shap_values[0, 1]
shap_values = shap_values[:,:-1]

shap.initjs()
shap.summary_plot(shap_values, X_f_val)


# <h3>Summary:</h3>
# <p>With the top62 vars the catmodel performed: <b>84.67532467532467</b></p>
# <p>[258  15]</p>
# <p>[44   68]</p>
# 
# <p> and with all the data the model performed: <b>91.42857142857143</b> </p>
# <p>[259  14]</p>
# <p>[19  93]</p>
# 
# <p> Bayesian Optimazation + full data the model performed: <b>91.68%</b></p>
# <p> params: [0.07431528396574379, 0.8539953708517165, 32, 7] </p>
# <p>[262  11]</p>
# <p>[ 21  91]</p>
# 
# <p> Random search + full data the model performed: <b>92.47%</b> </p>
# <p> params:[0.09871192514273254, 0.935929491371726, 10, 7] </p>
# 
# 
# <p>
# in this case the random search showed better results, but I believe that running the Baysean optimization for more iterations can generate better results than <b>92.47</b>
# </p>
# 
# 
