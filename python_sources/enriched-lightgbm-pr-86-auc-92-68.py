#!/usr/bin/env python
# coding: utf-8

# ## Abstract
# 
# 
# 
# ### **Methods**
# > - **Data preparation**: Mean imputation, really minor feature engineering.
# 
# > - **Data split**: Stratified 70/30 split.
# 
# > - **Models**: 3x LightGBM (major), Randomforest (baseline).
# 
# > - **Evaluation**: Results were extracted as the mean of 100 executions, all varying the dataset splits while preserving stratification.
# 
# > - **Metrics**: Precision-recall auc and ROC auc as primary metrics, precision, recall and accuracy as secondary.
# 
# > - **Hyperparameter tunning**: Nevergrad, gaussian mixture, lagrangian relaxation and manual inputs.
#     
# 
# ### **Results**
# > - Accuracy:   0.8796885813148784
# - Precision:  0.8321628618100186
# - Recall:     0.6916774193548385
# - ROC Auc:    0.9268143064134833
# - PR Auc:     0.8611576142950069
# 
# ### **Conclusion**
# 
# > The model is only as relevant as it the way it is integrated into practice. To that extent there are models with higher precision or higher recall with the same tier of overall performance, which could allow for different benefits, such as optimized resource allocation and screening aid respectively, though the present models are far more consistent in precision than recall.
# 
# > The variable composition makes these models very acessible which could have an enormous impact in the lens of public health and 150 million people that rely on SUS for their healthcare needs.
# For as much as machine learning may drive progress in the fight against the pandemic, its implementation is just as challenging.
# 
# 
# 
# Minor disclaimer: Still a work in progress... I'll work on interpretability and some pathways to implementation in the future.

# In[ ]:


import shap
import numpy  as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt

from sklearn.impute   import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics  import accuracy_score, auc, roc_curve, precision_recall_curve, roc_auc_score, precision_score, recall_score, average_precision_score
from sklearn.model_selection import train_test_split, StratifiedKFold

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost  import XGBClassifier

plt.style.use('ggplot')


# ## Auxiliar Functions

# In[ ]:


def evaluate(model, testing_set_x, testing_set_y):
    predictions = model.predict_proba(testing_set_x)
    
    accuracy  = accuracy_score(testing_set_y, predictions[:,1] >= 0.5)
    roc_auc   = roc_auc_score(testing_set_y, predictions[:,1])
    precision = precision_score(testing_set_y, predictions[:,1] >= 0.5)
    recall    = recall_score(testing_set_y, predictions[:,1] >= 0.5)
    pr_auc    = average_precision_score(testing_set_y, predictions[:,1])
    
    result = pd.DataFrame([[accuracy, precision, recall, roc_auc, pr_auc]], columns=['Accuracy', 'Precision', 'Recall', 'ROC_auc','PR_auc'])
    return(result)


# In[ ]:


def run_experiment(df, model_class, n = 100, **kwargs):
    results = pd.DataFrame(columns=['Accuracy', 'Precision', 'Recall', 'ROC_auc','PR_auc'])
    for i in range(n):
        # Compose dataset
        train_x, test_x = train_test_split(df.drop('PATIENT_VISIT_IDENTIFIER', axis=1),
                               test_size = 0.3,
                               stratify  = df['ICU'],
                               random_state = i
                                )
        
        train_y = train_x.pop('ICU')
        test_y  = test_x.pop('ICU')
        
        # Train Model
        model = model_class(**kwargs)
        model.fit(train_x, train_y)
         
        # Evaluate results
        current_result = evaluate(model, test_x, test_y)
        results = results.append(current_result)
        
    return(results.reset_index(drop=True))


# In[ ]:


def print_results(df, plot = True, extras = False, color='dodgerblue'):
    print('|||||||||||||||||||||||||||||||||||||||||||||||||||||||')
    print('[ Experiment Results ]')
    print('Accuracy:   {}'.format(df.Accuracy.mean()))
    print('Precision:  {}'.format(df.Precision.mean()))
    print('Recall:     {}'.format(df.Recall.mean()))
    print('ROC Auc:    {}'.format(df.ROC_auc.mean()))
    print('PR Auc:     {}'.format(df.PR_auc.mean()))
    print('|||||||||||||||||||||||||||||||||||||||||||||||||||||||')
    
    if plot:
        fig = px.box(df.melt(var_name='metric'),
                       y = 'metric',
                       x = 'value',
                       title = 'Distribution of Metric Values Across 100 Runs',
                       color_discrete_sequence=[color]
                      )

        fig.update_xaxes(title='Metric')
        fig.update_yaxes(title='Value')

        fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 00)',
                           'paper_bgcolor': 'rgba(240, 240, 240, 100)'})
        fig.show()
        
        
    if extras:
        print('Also, the maximum results were:')
        print('    Accuracy:   {}'.format(df.Accuracy.max()))
        print('    Precision:  {}'.format(df.Precision.max()))
        print('    Recall:     {}'.format(df.Recall.max()))
        print('    ROC Auc:    {}'.format(df.ROC_auc.max()))
        print('    PR Auc:     {}'.format(df.PR_auc.max()))


# ## Data Preparation
# 
# Nothing fancy, basicaly turned the 'categorical' variables into numerics.
# 
# #### Missingness Imputation
# There is considerable missingness in the dataset, which is to be expected as not every patient made the same set of exams.
# With that said, I tested mean, median and raw imputation (-2 for every np.nan), and mean imputation came out on top.
# 
# 
# #### Missingness as a feature
# 
# > "Our bugs are just additional features" - Dark Souls franchise ([reference](https://www.youtube.com/watch?v=hWbZZslsKqc&t=196))
# 
# The very missingness in the data sheds lights on how many procedures our patient went through.
# This could be indicative of complications, or at the very least indicate that the values were imputed rather than measured, so we'll make a feature out of missingness within each row.

# In[ ]:


# Read data
raw_data = pd.read_excel('../input/covid19/Kaggle_Sirio_Libanes_ICU_Prediction.xlsx')
raw_data.sample(5)

# Data Preparation
raw_data['AGE_PERCENTIL'] = raw_data['AGE_PERCENTIL'].str.replace('Above ','').str.extract(r'(.+?)th')
raw_data['WINDOW'] = raw_data['WINDOW'].str.replace('ABOVE_12','12-more').str.extract(r'(.+?)-')

# Missingness as features
raw_data['row_missingness'] = raw_data.isnull().sum(axis=1)

# Mean imputation
mean_impute  = SimpleImputer(strategy='mean')
imputed_data = mean_impute.fit_transform(raw_data)
imputed_data = pd.DataFrame(imputed_data, columns = raw_data.columns)


# In[ ]:


# Check imbalance
raw_data['ICU'].value_counts()


# #### Data Split
# For splitting data under class imbalance as we have here, I made sure to stratify by the labels in order to ensure consistent results at the end.
# The `ICU` label has an imbalance in the realms of 30/70, where a **baseline accuracy is set at 73.25%**.

# ### Random Forest Baseline
# 
# As part of a methodological choice, I chose to create a random forest as a baseline for the next models.
# Rather than blindly going for the usual suspects (Catboost, Lightgbm, DNNs).
# 
# The not greedy pattern of bagging makes the results very consistent with little effort.
# As far as my opinion is concerned, Random forests are great baselines, but hardly ever the best final models - yet the goal here is to have a strong foundation of what improving the score will look like.
# 
# 

# In[ ]:


rf_optimal = {
              'n_estimators':2100,
              'max_depth':27,
              'max_features':0.15,
              'max_samples':0.5363991145732665,
              'min_samples_split':2,
              'min_samples_leaf':4,
              'n_jobs':-1,
              'random_state':451,
            }


# #### Model Evaluation
# As the data presented is not particularly large, and the imbalanced exarcebates this issue, I'll use multiple runs of the same experiment in order to increase the trustworthyness of the results.
# Furthermore, I'll also make sure that every set of data is stratified, as to increase the consistency of training and evaluation.
# 
# This is very much recommeded, both by small size, but also due to the stakes of our model: **human lives**.

# In[ ]:


rf_experiment = run_experiment(imputed_data, model_class = RandomForestClassifier, **rf_optimal)
print_results(rf_experiment, color = '#3F3F3F')


# ### Hyperparameter Tunning
# 
# All parameters were obtained by a mix of optimization and manual input.
# I used facebook's nevergrad library and a few others in order to create distinct, yet performatic models.

# In[ ]:


optimal_1 = {'learning_rate': 0.02956340635276464,
 'n_estimators': 3831,
 'num_leaves': 101,
 'max_depth': 28,
 'max_bin': 211,
 'bagging_freq': 9,
 'bagging_fraction': 0.9292245982209768,
 'feature_fraction': 0.95,
 'lambda_l1': 2.50667180728151,
 'lambda_l2': 4.010110517090694,
 'drop_rate': 0.5917712341785191,
 'min_child_samples': 15,
 'min_child_weight': 3,
 'min_split_gain': 0.0,
 'scale_pos_weight': 0.283126887443018,
 'boosting_type': 'gbdt',
 'bagging_seed': 42,
 'metric': 'auc',
 'verbosity': -1,
 'random_state': 451,
 'max_drop': 50}


# In[ ]:


optimal_2 = {'learning_rate': 0.05744913989406643,
 'n_estimators': 2067,
 'num_leaves': 8,
 'max_depth': 27,
 'max_bin': 384,
 'bagging_freq': 5,
 'bagging_fraction': 0.7038650070406707,
 'feature_fraction': 0.4806588217742334,
 'lambda_l1': 2.841137907985995,
 'lambda_l2': 5.983397074528167,
 'drop_rate': 0.490746746058113,
 'min_child_samples': 3,
 'min_child_weight': 0,
 'min_split_gain': 0.0,
 'scale_pos_weight': 9.91024410907254,
 'boosting_type': 'gbdt',
 'bagging_seed': 42,
 'metric': 'auc',
 'verbosity': -1,
 'random_state': 451,
 'max_drop': 50}


# In[ ]:


optimal_3 = {'learning_rate': 0.05744913989406643,
 'n_estimators': 2067,
 'num_leaves': 8,
 'max_depth': 27,
 'max_bin': 384,
 'bagging_freq': 5,
 'bagging_fraction': 0.7038650070406707,
 'feature_fraction': 0.4806588217742334,
 'lambda_l1': 2.841137907985995,
 'lambda_l2': 5.983397074528167,
 'drop_rate': 0.490746746058113,
 'min_child_samples': 3,
 'min_child_weight': 0,
 'min_split_gain': 0.0,
 'scale_pos_weight': 0.91024410907254,
 'boosting_type': 'gbdt',
 'bagging_seed': 42,
 'metric': 'auc',
 'verbosity': -1,
 'random_state': 451,
 'max_drop': 50}


# In[ ]:


# Model #1
lgbm_experiment_1 = run_experiment(imputed_data, model_class = LGBMClassifier, **optimal_1)
print_results(lgbm_experiment_1, color = '#8400E8')


# In[ ]:


# Model #2
lgbm_experiment_2 = run_experiment(imputed_data, model_class = LGBMClassifier, **optimal_2)
print_results(lgbm_experiment_2, color = '#00E800')


# In[ ]:


# Model #3
lgbm_experiment_3 = run_experiment(imputed_data, model_class = LGBMClassifier, **optimal_3)
print_results(lgbm_experiment_3, extras=True, color = '#00A4E8')


# ### Model Interpretation
# 
# Let's make use of [SHAP](https://www.nature.com/articles/s42256-019-0138-9.epdf?shared_access_token=RCYPTVkiECUmc0CccSMgXtRgN0jAjWel9jnR3ZoTv0O81kV8DqPb2VXSseRmof0Pl8YSOZy4FHz5vMc3xsxcX6uT10EzEoWo7B-nZQAHJJvBYhQJTT1LnJmpsa48nlgUWrMkThFrEIvZstjQ7Xdc5g%3D%3D) as a way to leverage understanding from our model.

# In[ ]:


# Lets train a single model first
train_x, test_x = train_test_split(imputed_data.drop('PATIENT_VISIT_IDENTIFIER', axis=1),
                                   test_size = 0.3,
                                   stratify  = imputed_data['ICU'],
                                   random_state = 451
                                  )
        
train_y = train_x.pop('ICU')
test_y  = test_x.pop('ICU')


model = LGBMClassifier(**optimal_3)
model.fit(train_x, train_y)


# In[ ]:


# Extract shap values
explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(train_x)


# In[ ]:


# Average feature contribution
plt.title('Average Feature Contribution for each Class')
shap.summary_plot(shap_values, train_x, plot_type="bar")


# > From this plot we can see the most important features as per their SHAP values, and the top 20 features show a rather balanced pattern in contribution for each class.
# 
# > The importance of `AGE_PERCENTIL` is somewhat unexpected. While the age may relate to many phenomena it is hardly ever powerful in predicting anything specific. A feature so broad having such impact could be indactive of selection bias within the dataset.
# 
# > It is just a suspicion, but we have to factor the average patient profile of this hospital, as well as the possibility that media broadcasting certain age groups as risk groups could cause them to change their behaviour during that time, and such temporal shift of behaviour could affect the dataset.

# In[ ]:


# Granular feature contribution plot
plt.title('Feature Contribution According to Value')
shap.summary_plot(shap_values[1], train_x, plot_size = (15,10))


# > This plot has a gigantic wealth of knowledge.
# 
# > From the first two features, we can see that respiratory rate is key, and the mean is possibly being blinded by extremes when measuring, causing max to be more assertive.
# 
# > The `OTHER` feature carries some relevant signal and it is therefore recommended to unpack whatever is inside that feature.
# 
# > There is more to extract from this plot, and this is still only the top 20 features of the 230.

# In[ ]:


shap.initjs()
shap.force_plot(explainer.expected_value[1], shap_values[1][0:50], train_x[0:50])


# > Here we have a sample of 50 rows and by using the cursor we can have a rough idea of why a particular region was classified in such way. The default settings "sample order by similarity" groups similar rows.
# 
# > While there is no definite range where the model always outputs a class, every region associated with positive COVID-19 diagnosis has respiratory variables driving such decision. The fact that the model has no definitive regions is a good sign, as the variables are not specific and other diseases could cause similar values.

# ### Future Updates
# 
# - Add more models;
# - Add feature engineering.

# ### Closing Remarks
# 
# So there we have it, our best Lightgbm achieves Precision-Recall Auc of **0.86** and ROC Auc of **0.92**.
# Also, the models present different patterns of average precision and recall, which allows for flexibility depending on the task in which the model will be deployed to help.
# 
# One such example would be to use the recall model as a screening aid and the precision model for resource allocation (staff, exams, etc).
# It is also worth noting that the we should not exclude the possibility of a selection bias in the dataset composition, as specific hospitals may catter to specific patients as well.
# 
# If there's any question just let me know, Cheers :)

# ![](https://i.redd.it/8xj5uz79yon41.jpg)
# "*We are waves on the same ocean, leafs from the same tree, flowers from the same garden.*" - Chinese doctors upon disembarking on Italy
