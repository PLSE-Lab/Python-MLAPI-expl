#!/usr/bin/env python
# coding: utf-8

# settings > packages > github user/repo > dromosys/open-solution-home-credit 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sys
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


#!find /opt/conda/lib/python3.6 -name *.yaml


# In[ ]:


sys.version


# In[ ]:


np.version.version, pd.__version__


# In[ ]:


get_ipython().system('find /opt/conda/lib/python3.6/site-packages/opensolutionhomecredit*')


# In[ ]:


get_ipython().system('cp /opt/conda/lib/python3.6/site-packages/src/kaggle.yaml neptune.yaml')


# In[ ]:


#from src.pipeline_manager import PipelineManager

#pipeline_manager = PipelineManager()

from src.pipeline_manager import *


# In[ ]:


dev_mode = True
submit_predictions = True
pipeline_name = 'lightGBM'
model_level = 'first'


# In[ ]:


from src.utils import read_params
from deepsense import neptune
ctx = neptune.Context()
params = read_params(ctx, fallback_file='neptune.yaml')


# In[ ]:


import src.pipeline_config as cfg


# In[ ]:


cfg.DEV_SAMPLE_SIZE = 10000


# In[ ]:


from src.pipeline_manager import _read_data
from src.pipeline_manager import _get_fold_generator
from src.pipeline_manager import _fold_fit_evaluate_predict_loop
from src.pipeline_manager import _aggregate_test_prediction

def train_evaluate_predict_cv(pipeline_name, model_level, dev_mode, submit_predictions):
    if bool(params.clean_experiment_directory_before_training) and os.path.isdir(params.experiment_directory):
        logger.info('Cleaning experiment_directory...')
        shutil.rmtree(params.experiment_directory)

    tables = _read_data(dev_mode, read_train=True, read_test=True)

    target_values = tables.application_train[cfg.TARGET_COLUMNS].values.reshape(-1)
    fold_generator = _get_fold_generator(target_values)

    fold_scores, out_of_fold_train_predictions, out_of_fold_test_predictions = [], [], []
    for fold_id, (train_idx, valid_idx) in enumerate(fold_generator):
        (train_data_split,valid_data_split) = tables.application_train.iloc[train_idx], tables.application_train.iloc[valid_idx]

        logger.info('Started fold {}'.format(fold_id))
        logger.info('Target mean in train: {}'.format(train_data_split[cfg.TARGET_COLUMNS].mean()))
        logger.info('Target mean in valid: {}'.format(valid_data_split[cfg.TARGET_COLUMNS].mean()))
        logger.info('Train shape: {}'.format(train_data_split.shape))
        logger.info('Valid shape: {}'.format(valid_data_split.shape))

        score, out_of_fold_prediction, test_prediction = _fold_fit_evaluate_predict_loop(train_data_split,valid_data_split,tables,fold_id, pipeline_name, model_level='first')

        logger.info('Fold {} ROC_AUC {}'.format(fold_id, score))
        ctx.channel_send('Fold {} ROC_AUC'.format(fold_id), 0, score)

        out_of_fold_train_predictions.append(out_of_fold_prediction)
        out_of_fold_test_predictions.append(test_prediction)
        fold_scores.append(score)

    out_of_fold_train_predictions = pd.concat(out_of_fold_train_predictions, axis=0)
    out_of_fold_test_predictions = pd.concat(out_of_fold_test_predictions, axis=0)

    test_prediction_aggregated = _aggregate_test_prediction(out_of_fold_test_predictions)
    score_mean, score_std = np.mean(fold_scores), np.std(fold_scores)

    logger.info('ROC_AUC mean {}, ROC_AUC std {}'.format(score_mean, score_std))
    ctx.channel_send('ROC_AUC', 0, score_mean)
    ctx.channel_send('ROC_AUC STD', 0, score_std)

    logger.info('Saving predictions')
    out_of_fold_train_predictions.to_csv(os.path.join(params.experiment_directory,'{}_out_of_fold_train_predictions.csv'.format(pipeline_name)),index=None)
    out_of_fold_test_predictions.to_csv(os.path.join(params.experiment_directory,'{}_out_of_fold_test_predictions.csv'.format(pipeline_name)),index=None)
    test_aggregated_file_path = os.path.join(params.experiment_directory,'{}_test_predictions_{}.csv'.format(pipeline_name,params.aggregation_method))
    test_prediction_aggregated.to_csv(test_aggregated_file_path, index=None)

    if not dev_mode:
        logger.info('verifying submission...')
        sample_submission = pd.read_csv(params.sample_submission_filepath)
        verify_submission(test_prediction_aggregated, sample_submission)

        if submit_predictions and params.kaggle_api:
            make_submission(test_aggregated_file_path)


# In[ ]:


import warnings
warnings.simplefilter(action = "ignore", category = RuntimeWarning)


# In[ ]:


#%prun 
train_evaluate_predict_cv(pipeline_name, model_level, dev_mode, submit_predictions)


# In[ ]:


get_ipython().system('find /kaggle/working/result -name *.csv')


# In[ ]:


get_ipython().system('cp /kaggle/working/result/lightGBM_test_predictions_rank_mean.csv .')


# In[ ]:


get_ipython().system('rm -rf /kaggle/working/result/')


# In[ ]:





# In[ ]:





# In[ ]:




