#!/usr/bin/env python
# coding: utf-8

# # NLP Disasters Prediction
# ## Benchmarking BERT Variations
# 
# ![NLP](https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcTWL5PJwJNJ8t8FJzL3b0qvXwfpC1jbDVG3Q59ue1xgN87mxfK7)
# 
# The purpose of this notebook is to benchmark BERT variations, checking the validation and submission scores for each one of them and try ensemble the predictions in order to get a better classifier.
# 
# * **1.** We start by running the models: BERT, ROBERTA, XLNET and ALBERT using the combination of Hugging Face + KTrain wrapper. We can notice how the KTrain tool can reduce significantly the necessary lines of code to create a model.
# * **2.** Then, we evaluate the ROC curves and metrics for each one of these models.
# * **3.** Finally, we try to stack the models by taking averages and comparing different scores with the pure estimators
# * **4.** Submit the models and evaluate the submission metrics
# * **5.** Take conclusions
# 
# **Note:** In my previous [NLP Disasters Predictions notebook](https://www.kaggle.com/guidant/disasternlp-benchmarking-tfhub-bert-variations), I tried to improve the BERT model by adding new features and autocorrecting the input texts. These tries were not succcessful but, as we can see here, we will improve the BERT result this time. I am writting a new notebook because the idea and the libraries that I will use are both, different from the ones of the previous try.

# # Part 0. Before starting: Install ktrain and import libraries
# 
# We start by installing the ktrain library and by importing everything that we will use. Notice that, in this version of the docker, it's necessary to update the scikit learn before installing the ktrain tool.

# In[ ]:


get_ipython().run_cell_magic('capture', '', '!pip install -U scikit-learn')


# In[ ]:


get_ipython().run_cell_magic('capture', '', '!pip install ktrain')


# In[ ]:


import numpy as np
import pandas as pd
import tensorflow as tf

import ktrain
import plotnine
import seaborn as sns

from ktrain import text
from transformers import *
from plotnine import *
from plotnine.options import figure_size

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_curve, auc, accuracy_score, f1_score

from pylab import rcParams


# Importing the inputs and taking a look:

# In[ ]:


input_dir = '/kaggle/input/nlp-getting-started/'
checkpoint_dir = '/kaggle/input/checkpoint-nlp-disaster/'
output_dir = '/kaggle/working/'

df_fit = pd.read_csv(input_dir + 'train.csv')
df_sub = pd.read_csv(input_dir + 'test.csv')

df_fit.head()


# Ok then, we are ready to start!
# 
# # 1. Training Different BERT Variations
# 
# ![SSS](https://faculty.elgin.edu/dkernler/statistics/ch01/images/strata-sample.gif)
# 
# We start by getting a stratified sample for training and testing. In our case, stratification means: getting different sets with the same percentage of zeroes and ones in the target columns (i.e: taking samples with the a similar variation and distribution of different characteristics):

# In[ ]:


sss = StratifiedShuffleSplit(n_splits=1, random_state=42)
train_index, valid_index = next(sss.split(df_fit, df_fit['target']))

df_trn = df_fit.iloc[train_index, :]
df_tst = df_fit.iloc[valid_index, :]


# Here we have a for loop to get the training, testing and submission results for each model in order to generate our benchmarks. The models_list is the list of the models that will be trained and if it has no elements, we load the results of the previous execution. It's not possible to run all the models in a loop like that because of memory limitations but I ran each model in different sections, saving all the results:

# In[ ]:


benchmark_dict = dict()

# model_list = ['bert-base-uncased', 'xlnet-large-cased', 'albert-base-v2', 'roberta-base']
model_list = []

if len(model_list) > 0:

    nrow_trn, nrow_tst, nrow_sub = df_trn.shape[0], df_tst.shape[0], df_sub.shape[0]
    dict_benchmark = {'Model': [], 'Type': [], 'Id': [], 'Prediction': [], 'Target': []}
    dict_submission = {'Model': [], 'Id': [], 'Proba': []}

    for curr_model in model_list:
    
        curr_benchmark = dict()

        t = text.Transformer(curr_model, maxlen=140, classes=[0, 1])
        trn = t.preprocess_train(df_trn['text'].tolist(), df_trn['target'].tolist())
        val = t.preprocess_train(df_tst['text'].tolist(), df_tst['target'].tolist())

        model = t.get_classifier()
        learner = ktrain.get_learner(model, train_data=trn, val_data=val, batch_size=16)
        learner.fit_onecycle(2e-6, 3)
        predictor = ktrain.get_predictor(model, t)
    
        curr_predict_trn = predictor.predict_proba(df_trn['text'].tolist())[:, 1].tolist()
        curr_predict_tst = predictor.predict_proba(df_tst['text'].tolist())[:, 1].tolist()
        curr_predict_sub = predictor.predict_proba(df_sub['text'].tolist())[:, 1].tolist()
    
        dict_benchmark['Model'] += (nrow_trn + nrow_tst) * [curr_model]
        dict_benchmark['Type'] += (nrow_trn * ['Train']) + (nrow_tst * ['Test'])
        dict_benchmark['Id'] += df_trn['id'].tolist() + df_tst['id'].tolist()
        dict_benchmark['Prediction'] += curr_predict_trn + curr_predict_tst
        dict_benchmark['Target'] += df_trn['target'].tolist() + df_tst['target'].tolist()
    
        dict_submission['Model'] += nrow_sub * [curr_model]
        dict_submission['Id'] += df_sub['id'].tolist()
        dict_submission['Proba'] += curr_predict_sub
    
    df_benchmark, df_submission = pd.DataFrame(dict_benchmark), pd.DataFrame(dict_submission)
    df_benchmark.to_csv(output_dir + 'benchmark.csv'), df_submission.to_csv('submission.csv')
    
else:
    
    benchmark_list, submission_list = [], []
    for model_name in ['albert', 'bert', 'roberta', 'xlnet']:
        benchmark_list.append(pd.read_csv(checkpoint_dir + 'benchmark_' + model_name + '.csv'))
        submission_list.append(pd.read_csv(checkpoint_dir + 'submission_' + model_name + '.csv'))
        
    df_benchmark = pd.concat(benchmark_list, ignore_index=True)
    df_submission = pd.concat(submission_list, ignore_index=True)


# So, here we have the dataset with all the predictions and targets:

# In[ ]:


df_benchmark = df_benchmark.loc[:, ['Model', 'Type', 'Id', 'Prediction', 'Target']]
df_benchmark.head()


# # 2. Evaluating the BERT Variations
# 
# ## 2.1. Comparing ROC Curves
# 
# Let's start by comparing the ROC curves of each model:

# In[ ]:


def get_df_roc(df_in, models_list=None):
    df_roc = pd.DataFrame({'Model': [], 'FalsePositiveRate': [], 'TruePositiveRate': []})
    if models_list is None:
        models_list = set(df_in['Model'].tolist())
    for curr_model in models_list:
        fpr_list, tpr_list, _ = roc_curve(df_in.loc[df_in['Model'] == curr_model, :]['Target'].tolist(), 
                                          df_in.loc[df_in['Model'] == curr_model, :]['Prediction'].tolist())
        model_list = [curr_model] * len(fpr_list)
    
        df2append = pd.DataFrame({
            'Model': model_list,
            'FalsePositiveRate': fpr_list,
            'TruePositiveRate': tpr_list
        })
        df_roc = pd.concat([df_roc, df2append], ignore_index=True)
    return df_roc

df_roc = get_df_roc(df_benchmark)
df_roc.head()


# Using the facets layer to compare all the curves:

# In[ ]:


def plot_roc_curves(df_in, models_list=None, size_x=10, size_y=7):
    plotnine.options.figure_size = (size_x, size_y)
    return ggplot(get_df_roc(df_in, models_list), aes(x='FalsePositiveRate', y='TruePositiveRate', color='Model', group='Model')) +        geom_line() + facet_wrap('~ Model', nrow=2) + theme(legend_position='none') +        theme(text=element_text(size=14)) + ggtitle('Comparing ROC curves:')
    
plot_roc_curves(df_benchmark)


# We can see that the ROBERTA and BERT models seems to have a better behaviour. 
# 
# ## 2.2. Comparing Scores
# 
# We should confirm it by taking a look at different types of scores. We will check the **AUC (Area Under ROC Curve)**, the **Accuracy** and the **F1-Score**:

# In[ ]:


def get_benchmark_scores(df_in, models_list=set(df_benchmark['Model'].tolist())):

    dict_metrics = dict(Model=[], Metric=[], Value=[])

    for curr_model in models_list:
        for metric in ['Accuracy', 'AUC', 'F1-Score']:
        
            filter_cond = df_in['Model'] == curr_model
            filter_cond &= df_in['Type'] == 'Test'
            
            sorted_df = df_in.loc[filter_cond, ['Id', 'Prediction', 'Target']].sort_values(by='Id')
            predictions = sorted_df['Prediction'].tolist()
            targets = sorted_df['Target'].tolist()
        
            if metric in ['Accuracy', 'F1-Score']:
                predictions = [1 if X >= 0.5 else 0 for X in predictions]
            
            if metric == 'Accuracy':
                score = accuracy_score(targets, predictions)
            elif metric == 'F1-Score':
                score = accuracy_score(targets, predictions)
            elif metric == 'AUC':
                fpr, tpr, _ = roc_curve(targets, predictions)
                score = auc(fpr, tpr)
            
            dict_metrics['Model'].append(curr_model)
            dict_metrics['Metric'].append(metric)
            dict_metrics['Value'].append(score)
        
    return pd.DataFrame(dict_metrics)

df_metrics = get_benchmark_scores(df_benchmark)
df_metrics.head()


# Plotting the table results in the form of a comparative grid, where the models represent the columns and the rows represent the scores:

# In[ ]:


def plot_benchmark_scores(df_in, models_list=set(df_benchmark['Model'].tolist())):
    
    return ggplot(get_benchmark_scores(df_in, models_list), aes(x='Model', y='Value', fill='Model')) +        geom_hline(aes(yintercept='Value'), linetype='dashed') +        geom_point(size=5) + facet_grid('Metric~Model',scales='free_x') +        ggtitle('Comparing Scores for Different Models')

plot_benchmark_scores(df_benchmark)


# Again, we can see that the best results are obtained for the BERT and the ROBERTA model. **BUT** the Roberta Model seems to be better than the original BERT! Let's start to plan our model stacking.
# 
# # 3. Stacking the Models
# 
# ## 3.1. Comparing Correlations
# 
# Comparing the correlations between the outputs of each pair of models over the validation set:

# In[ ]:


dict_corr = dict(Model1=[], Model2=[], Correlation=[])
models_list = set(df_benchmark['Model'].tolist())

for model1 in models_list:
    for model2 in models_list:
        
        dict_corr['Model1'].append(model1)
        dict_corr['Model2'].append(model2)
        
        filter_cond1 = df_benchmark['Model'] == model1
        filter_cond1 &= df_benchmark['Type'] == 'Test'
        
        filter_cond2 = df_benchmark['Model'] == model2
        filter_cond2 &= df_benchmark['Type'] == 'Test'
        
        pred1 = df_benchmark.loc[filter_cond1]['Prediction'].tolist()
        pred2 = df_benchmark.loc[filter_cond2]['Prediction'].tolist()
        
        dict_corr['Correlation'].append(np.corrcoef(np.array(pred1), np.array(pred2))[0, 1])
        
dict_model_bias = {
    'albert-base-v2': 'ALBERT',
    'bert-base-uncased': 'BERT',
    'roberta-base': 'ROBERTA',
    'xlnet-large-cased': 'XLNET'
}
        
df_corr = pd.DataFrame(dict_corr)
df_corr['Model1'] = df_corr['Model1'].apply(lambda X: dict_model_bias[X])
df_corr['Model2'] = df_corr['Model2'].apply(lambda X: dict_model_bias[X])

df_corr.head()


# In[ ]:


df_corr['X'] = 0
df_corr['Y'] = 0

ggplot(df_corr, aes(x='X', y='Y', fill='Correlation')) +    geom_point(aes(size='Correlation')) + ggtitle('Correlation Among Models') + facet_grid('Model1~Model2') +    theme(text=element_text(size=10), axis_text=element_blank()) + xlab('Model 1') + ylab('Model 2') +    scale_size_continuous([10, 20])


# We can see that there is a strong correlation between the best $2$ models, which makes sense: both of them has really high scores and consequently both models will generate similar predictions. Two stacking possibilities will be explored here:
# 
# 1. Taking a simple average between the BERT and the ROBERTA model (the best ones) and
# 2. Taking a complete average, along all estimators
# 
# ## 3.2. Evaluating Ensemble Models
# 
# So, let's compare the $2$ ensemble models proposed in the last section with the best pure models (BERT and Roberta). We will use the functions that we already defined in the previous sections:

# In[ ]:


def get_df_ensemble(df_in, name_new_model):
    
    df_ensemble = df_in.groupby('Id').mean()
    df_ensemble['Id'] = df_ensemble.index

    train_ids = list(set(df_in[df_in['Type'] == 'Train']['Id'].tolist()))
    test_ids = list(set(df_in[df_in['Type'] == 'Test']['Id'].tolist()))

    df_ensemble['Id'] = df_ensemble.index
    df_ensemble['Model'] = name_new_model
    df_ensemble['Type'] = 'NA'
    df_ensemble.loc[df_ensemble['Id'].isin(train_ids), 'Type'] = 'Train'
    df_ensemble.loc[df_ensemble['Id'].isin(test_ids), 'Type'] = 'Test'
    df_ensemble = df_ensemble.loc[:, ['Model', 'Type', 'Id', 'Prediction', 'Target']]
    
    return df_ensemble

df_ensemble_all = get_df_ensemble(df_benchmark, 'ensemble_all')
best_models = ['bert-base-uncased', 'roberta-base']
df_ensemble_best = get_df_ensemble(df_benchmark.loc[df_benchmark['Model'].isin(best_models), :], 'ensemble_best')

df_benchmark_improved = pd.concat([df_benchmark, df_ensemble_all, df_ensemble_best], ignore_index=True)
plot_benchmark_scores(df_benchmark_improved, models_list=best_models + ['ensemble_best', 'ensemble_all'])


# So, ensembling all the models generates results that are almost equal to a ROBERTA Neural Network. Let's check the ROC Curves...we could explose all the models but I want to not show many useless graphics with the price of losing interpretability.

# In[ ]:


plot_roc_curves(df_benchmark_improved, models_list=['bert-base-uncased', 'roberta-base', 'ensemble_all', 'ensemble_best'])


# # 4. Preparing / Evaluating Submissions
# 
# Let's submit $4$ different tries:
# 
# 1. The BERT model
# 2. The ROBERTA model
# 3. The average of the outputs of all used models and
# 4. The average of the outputs of the BERT and ROBERTA estimators
# 
# Then, I will show and compare the submission results for each one of them.

# In[ ]:


df_submission = df_submission.loc[:, ['Model', 'Id', 'Proba']]
df_submission.head()


# In[ ]:


df_submission_ensemble_all = df_submission.groupby('Id').mean()
df_submission_ensemble_all['Id'] = df_submission_ensemble_all.index
df_submission_ensemble_all['Model'] = 'ensemble_all'
df_submission_ensemble_all = df_submission_ensemble_all.loc[:, ['Model', 'Id', 'Proba']]

df_submission_ensemble_best = df_submission.loc[df_submission['Model'].isin(best_models), :].groupby('Id').mean()
df_submission_ensemble_best['Id'] = df_submission_ensemble_best.index
df_submission_ensemble_best['Model'] = 'ensemble_best'
df_submission_ensemble_best = df_submission_ensemble_best.loc[:, ['Model', 'Id', 'Proba']]

df_submission_enhanced = pd.concat([df_submission, df_submission_ensemble_all, df_submission_ensemble_best], ignore_index=True)
df_submission_enhanced.tail()


# In[ ]:


def export_model(model_name):
    df_export = df_submission_enhanced.loc[df_submission_enhanced['Model'] == model_name, :]
    df_export.loc[:, 'Target'] = df_export['Proba'].apply(lambda X: 1 if X >= 0.5 else 0)
    df_export = df_export.loc[:, ['Id', 'Target']]
    df_export.rename({'Id': 'id', 'Target': 'target'}, inplace=True)
    df_export.to_csv(output_dir + model_name + '.csv', index=False)


# Exporting the different submissions:

# In[ ]:


export_model('ensemble_all')
export_model('ensemble_best')
export_model('bert-base-uncased')
export_model('roberta-base')


# And here I noted the submission results for each try. Again, we can see that the ROBERTA and the Ensemble Model involving all the estimators generate the same final scores. For that reason, in our benchmark, I will consider that the best model is the average of all the possibilities since, in the real life, stacked models tend to generate less cases of overfitting:

# In[ ]:


df_submission_results = pd.DataFrame({
    'Model': ['Roberta', 'Bert', 'Ensemble All', 'Ensemble Best'],
    'Value': [0.83026, 0.82310, 0.83026, 0.82822],
    'X': [0] * 4
})

ggplot(df_submission_results, aes(x='X', y='Value', fill='Model')) + geom_hline(aes(yintercept='Value'), linetype='dashed') +    geom_point(stat='identity', color='black', size=10) + facet_grid('~Model') +    xlab('Model') + ggtitle('Benchmarking Submissions') + ylab('F1 Kaggle Final Score') +    theme(legend_position='none', axis_text_x=element_blank(), text=element_text(size=12))


# # 5. Conclusions
# 
# * The BERT Mmodel and the ROBERTA model are the best single estimators BUT
# * Taking the average of all the Neural Networks is better than taking the average just between the $2$ models
# * It happens because the BERT and the ROBERTA models are strongly correlated AND
# * We can also conclude that our champions are: ROBERTA or FULL AVERAGE MODEL
# * Then: we get the FULL AVERAGE MODEL because, in real life, it will be more robust, being harder to get overfitted

# # Versions Log
# 
# * **V1 to V9**: Tests
# * **V10**: First version
# * **V11, V12**: Some corrections + Addition of Ensemble Models ROC curves
