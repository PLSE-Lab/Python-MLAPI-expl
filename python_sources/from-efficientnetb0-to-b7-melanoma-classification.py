#!/usr/bin/env python
# coding: utf-8

# # From EfficientNetB0 to B7 - Melanoma Classification with Tensorflow/Keras
# 
# This kernel presents the results of training the EfficientNet models, from B0 to B7, using TPUs (yeah, it took a looooooong time to train it all).
# 
# And the kernel I used to train all these models is [here](https://www.kaggle.com/fredericods/efficientnetbi-melanoma-classification-with-tf). 
# 
# The main idea is to analyze the influence of the model size on its performance. And obviously, practice computer vision and deep learning skills. :)
# 
# The performance of the trained networks here is still far behind the top-score solutions, but there is a lot of room for improvement: tuning the learning rate, larger input size, larger networks, more complex augmentations, using metadata, ensembling etc.
# 
# Feel free to criticize and suggest!
# 
# **Training/Modeling Highlights:**
# - Image input size: 256 x 256
# - Fine tuning EfficientNet with only a dense layer on the top
# - Stratified Group 4-Fold Validation: imbalanced target distribution + not have same patient on train and validation set
# - Augmentations: random flip left-right and random flip up-down
# - Learning Rate Scheduler: adopting a learning rate ramp-up because fine-tuning a pre-trained model
# - Adam optimizer
# - Loss function: Binary Cross-Entropy Loss with label_smoothing
# - Epochs: 10
# - Model checkpoint: saving when best validation loss is achieved
# 
# **References:**
# - https://www.kaggle.com/reighns/groupkfold-efficientbnet-and-augmentations
# - https://www.kaggle.com/jakubwasikowski/stratified-group-k-fold-cross-validation
# - https://www.kaggle.com/cdeotte/rotation-augmentation-gpu-tpu-0-96
# - https://www.kaggle.com/ajaykumar7778/melanoma-tpu-efficientnet-b5-dense-head
# - https://www.kaggle.com/khoongweihao/siim-isic-multiple-model-training-stacking

# ## 1) Importing libraries and dataset

# In[ ]:


import os
from os import listdir
from os.path import isfile, join

import numpy as np
import pandas as pd

from plotly.subplots import make_subplots
import plotly.graph_objects as go
        
DATA_PATH = '/kaggle/input/efficientbx-melanoma-classification-with-tf'

# Importing Data
history_files = [f for f in listdir(DATA_PATH) if isfile(join(DATA_PATH, f)) and f.split('_')[0] == 'history']
submit_files = [f for f in listdir(DATA_PATH) if isfile(join(DATA_PATH, f)) and f.split('_')[0] == 'submit']


# In[ ]:


# Preprocessing Data for visualization (Performance x Model)
list_results = []
for file_name_i in history_files:
    model_name_i = file_name_i[8:22]
    fold_name_i = file_name_i[23:29]
    df_i = pd.read_csv(os.path.join(DATA_PATH, file_name_i), index_col=0)
    auc_i = df_i[df_i.val_loss == df_i.val_loss.min()]['val_auc'].iloc[0]
    loss_i = df_i.val_loss.min()
    list_results.append([model_name_i, fold_name_i, auc_i, loss_i])
df_results = pd.DataFrame(list_results, columns = ['Model', 'Fold', 'AUC', 'Loss'])

for model_name_i in df_results.Model.unique():
    mean_auc_i = df_results.AUC[df_results.Model == model_name_i].mean()
    mean_loss_i = df_results.Loss[df_results.Model == model_name_i].mean()
    df_i = pd.DataFrame([[model_name_i, 'mean_fold', mean_auc_i, mean_loss_i]], columns = ['Model', 'Fold', 'AUC', 'Loss'])
    df_results = pd.concat([df_results, df_i])
    
df_results = df_results.sort_values(by = ['Model', 'Fold'], ascending = True).reset_index().drop(columns = ['index'])


# In[ ]:


# Preprocessing Data for visualization (Training History)
model_names = ['EfficientNetB' + str(i) for i in range(8)]
fold_names = ['fold_' + str(i) for i in range(4)]
dict_df_history = {}
for model_i in model_names:
    dict_df_history[model_i] = {}
    for fold_i in fold_names:
        file_name_i = 'history_' + model_i + '_' + fold_i + '.csv'
        df_history_i = pd.read_csv(os.path.join(DATA_PATH, file_name_i))
        df_history_i=df_history_i.rename(columns = {
            'Unnamed: 0': 'Epoch',
            'loss': 'Train Loss',
            'auc': 'Train AUC',
            'val_loss': 'Valid Loss',
            'val_auc': 'Valid AUC',
            'lr': 'Learning Rate'
        }
                           )
        dict_df_history[model_i][fold_i] = df_history_i


# ## 2) Visualizing Results

# In[ ]:


from plotly.subplots import make_subplots
import plotly.graph_objects as go

df_results_mean_fold = df_results[df_results.Fold=='mean_fold']

fig = make_subplots(rows=2, cols=1)

fig.append_trace(
    go.Scatter(x=df_results_mean_fold.Model, y=df_results_mean_fold.AUC, name='AUC'),
    row=1, col=1
)

fig.append_trace(
    go.Scatter(x=df_results_mean_fold.Model, y=df_results_mean_fold.Loss, name='Loss'),
    row=2, col=1
)


fig.update_layout(height=800, width=600, title_text="Performance x Model")
fig.update_yaxes(title_text="AUC", row=1, col=1)
fig.update_yaxes(title_text="Loss", row=2, col=1)

for trace in fig['data']: 
    trace['showlegend'] = False
        
fig.show()


# ### Model Training History

# In[ ]:


fig = make_subplots(rows=2, cols=2, subplot_titles=('Fold 0', 'Fold 1', 'Fold 2', 'Fold 3'))

# Add Traces
for model_name_i in model_names:
    for i, fold_i in enumerate(fold_names):
        for set_i in ['Train', 'Valid']:
            for metric_i in ['AUC', 'Loss']:
                index_dict = {
                    'fold_0': [1,1],
                    'fold_1': [1,2],
                    'fold_2': [2,1],
                    'fold_3': [2,2]
                }
                color_dict = {
                    'Train AUC': {'color': '#dba053'},
                    'Valid AUC': {'color': '#6e53db'},
                    'Train Loss': {'color': '#4cd489'},
                    'Valid Loss': {'color': '#bf2e2e'}
                }
                fig.add_trace(
                    go.Scatter(
                        x=dict_df_history[model_name_i][fold_i]['Epoch'],
                        y=dict_df_history[model_name_i][fold_i][set_i + ' ' + metric_i],
                        name=set_i + ' ' + metric_i,
                        #name=model_name_i + ' ' + fold_i + ' ' + set_i + ' ' + metric_i,
                        #name=model_name_i[-2:] + fold_i[-1] + set_i + metric_i,
                        line=color_dict[set_i + ' ' + metric_i],
                        legendgroup=model_name_i[-2:] + ' ' + metric_i,
                        showlegend=(i==0)
                    ),
                    row=index_dict[fold_i][0], col=index_dict[fold_i][1]
                )

list_buttons = []
for i, model_name_i in enumerate(model_names):
    visible_list_aux_i = [False]*128
    visible_list_aux_i[i*16:i*16+16] = [True]*16
    button_i = dict(
        label=model_name_i[-2:],
        method="update",
        args=[{"visible": visible_list_aux_i}])
    list_buttons.append(button_i)                
                
fig.update_layout(
    updatemenus=[
        dict(
            active=0,
            buttons=list_buttons,
            direction="right",
            pad={"r": 10, "t": 10},
            showactive=True,
            x=0.00,
            xanchor="left",
            y=1.25,
            yanchor="top",
            type = 'buttons'
        )
    ])

fig.show()


# ## 3) Submit Files

# In[ ]:


sample_submit = pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/sample_submission.csv')
for model_name_i in model_names:
    target_list = []
    for fold_i in fold_names:
        target_i = pd.read_csv(os.path.join(DATA_PATH, 'submit_' + model_name_i + '_' + fold_i + '.csv'))['target'].values
        target_list.append(target_i)
    df_submit_i = sample_submit.copy()
    df_submit_i['target'] = np.mean(target_list, axis=0)
    df_submit_i.to_csv('submit_' + model_name_i + '_mean_fold.csv', index=False)
    print(model_name_i + ' - Mean Fold submit file saved')

