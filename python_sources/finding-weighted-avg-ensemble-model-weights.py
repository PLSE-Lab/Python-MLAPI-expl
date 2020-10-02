#!/usr/bin/env python
# coding: utf-8

# # An experiment to find Best Weighted avg ensemble model weights coefficient using Brute Force (Generic way)
# 
# #### During multi model we come across a situation to use mix of model to offset the weakness of one model over other using stacking. For this we need right blend of coefficients for each model.
# #### It is a tedious job to find those blending coefficients. 
# ### One way to generate them using Brute force technic (Try all combination).
# 
# #### For this I developed a generic utility whihc takes consolidated columns and few other parameter to go thru all possible combination and list best weighted coefficients.
# 
# Source code available [here](https://github.com/KeshavShetty/kesh-utils/tree/master/KUtils/common) and PyPi package [here](https://pypi.org/project/kesh-utils/)
# 
# The generic method is KUtils.common.blend_util() with arguments
# - df # The consilidated dataframe
# - actual_target_column_name, 
# - columns_to_blend - # List of columns to use while blend, - Not fixed you can use any combination like 2, 3,... n columns which exist in consolidated data
# - starting_weight=1,
# - max_weight=10,
# - step_weight=1,
# - minimize_loss='rmse', # other option 'mae'
# - verbose=False 
# 
# The parameter <b><u>starting_weight=1, max_weight=10, step_weight=1 acts something like range(1,10,10)</u></b> including upper limit.
# Utility will go thru all possible combinations and calculate MAE or RMSE between new blend probabability vs actual target
# 
# > For this demo I used a consolidated probability dataset which I generated for some other binary classification which contains below columns
# 
# - row_id # Unique rowid or original dataset
# - Actual # Actual target column from original datset (0,1)
# - gnb_proba # Probability generated from Gaussian Naive Bayes 
# - dt_le_proba # Probability generated from Decision Tree
# - rf_proba 	# Probability generated thru Random Forest
# - xgb_proba # Probability generated thru XGBoost

# In[ ]:


# Lets start
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


# Install the Library (Refer: https://pypi.org/project/kesh-utils/ )
get_ipython().system('pip install kesh-utils')


# In[ ]:


get_ipython().system('pip install statsmodels==0.10.0rc2 --pre  # Statsmodel has sme problem with factorial in latest lib')


# In[ ]:


# Ignore the warnings if any
import warnings  
warnings.filterwarnings('ignore')


# In[ ]:


# Load the dataset 
consolidated_proba_df = pd.read_csv('../input/consolidated_proba.csv')


# In[ ]:


# Quick check the consolidate dataset
consolidated_proba_df.head(10)


# # In action

# In[ ]:


# Load the util
from KUtils.common import blend_util


# In[ ]:


# Let run and find best weighted avg coefficients for columns 'gnb_proba', 'dt_le_proba', 'rf_proba', 'xgb_proba'

best_blend_df = blend_util.find_best_stacking_blend(
    consolidated_proba_df, 
    actual_target_column_name='Actual', 
    columns_to_blend=['gnb_proba', 'dt_le_proba', 'rf_proba', 'xgb_proba'],
    starting_weight=1,
    max_weight=10,
    step_weight=1,
    minimize_loss='rmse', # other option mae
    verbose=False
)


# In[ ]:


# The best blend df gets appended as and when new best options are found. So all best blend options are at the end of the dataframe (Sort it if you want)
best_blend_df.tail()


# # Looking at RMSE you can say 16th row will have lowest error and coeeficients found are
# - for gnb_proba 1
# - for dt_le_proba 1
# - for rf_proba 10
# - for xgb_proba 6

# In[ ]:


# Create new probablity based new model blend coefficients. Total weight 18
consolidated_proba_df['final_blended_proba'] = (consolidated_proba_df['gnb_proba']*1 + 
                                                consolidated_proba_df['dt_le_proba']*1 + consolidated_proba_df['rf_proba']*10 + consolidated_proba_df['xgb_proba']*6)/18


# In[ ]:


consolidated_proba_df.head(10)


# # Extra
# ### Lets check the efficiency of blending over individual model using Sensitivity-Specificity Cutoff

# In[ ]:


import matplotlib.pyplot as plt
from KUtils.logistic_regression import auto_logistic_regression as autoglm

from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, make_scorer, recall_score, precision_score


# ### Plots of Gaussian Naive Bayes

# In[ ]:


# First(0) column contains prob for 0-class and second(1) contains prob for 1-class
gnb_pred_df = pd.DataFrame({'Actual':consolidated_proba_df['Actual'], 'Probability':consolidated_proba_df['gnb_proba']}) 
return_dictionary = autoglm.calculateGLMKpis(gnb_pred_df, cutoff_by='Sensitivity-Specificity', include_cutoff_df_in_return=True)
cutoff_df = return_dictionary['cutoff_df']


# In[ ]:


cutoff_df.plot.line(x='Probability', y=['Accuracy','Sensitivity','Specificity'])
plt.show()


# In[ ]:


cutoff_df.plot.line(x='Probability', y=['Precision','Recall'])
plt.show()


# In[ ]:


prob_column='gnb_proba'
prob_cutoff = 0.04
consolidated_proba_df['predicted'] = consolidated_proba_df[prob_column].map(lambda x: 1 if x > prob_cutoff else 0)

local_confusion_matrix = metrics.confusion_matrix(consolidated_proba_df['Actual'], consolidated_proba_df['predicted'] )
        
accuracy = metrics.accuracy_score(consolidated_proba_df['Actual'], consolidated_proba_df['predicted'])
precision = metrics.precision_score(consolidated_proba_df['Actual'], consolidated_proba_df['predicted'])
recall = metrics.recall_score(consolidated_proba_df['Actual'], consolidated_proba_df['predicted'])
f1_score = metrics.f1_score(consolidated_proba_df['Actual'], consolidated_proba_df['predicted'])
roc_auc = metrics.roc_auc_score(consolidated_proba_df['Actual'], consolidated_proba_df['predicted'])

print(" Accuracy {0:.3f}, \n Precision {1:.3f}, \n Recall {2:.3f}, \n f1_score {3:.3f}, \n roc_auc {4:.3f}".format(
    accuracy, precision,recall,f1_score,roc_auc))


# ### Plots of Random Forest

# In[ ]:


# First(0) column contains prob for 0-class and second(1) contains prob for 1-class
rf_pred_df = pd.DataFrame({'Actual':consolidated_proba_df['Actual'], 'Probability':consolidated_proba_df['rf_proba']}) 
return_dictionary = autoglm.calculateGLMKpis(rf_pred_df, cutoff_by='Sensitivity-Specificity', include_cutoff_df_in_return=True)
cutoff_df = return_dictionary['cutoff_df']


# In[ ]:


cutoff_df.plot.line(x='Probability', y=['Accuracy','Sensitivity','Specificity'])
plt.show()


# In[ ]:


cutoff_df.plot.line(x='Probability', y=['Precision','Recall'])
plt.show()


# In[ ]:


prob_column='rf_proba'
prob_cutoff = 0.3
consolidated_proba_df['predicted'] = consolidated_proba_df[prob_column].map(lambda x: 1 if x > prob_cutoff else 0)

local_confusion_matrix = metrics.confusion_matrix(consolidated_proba_df['Actual'], consolidated_proba_df['predicted'] )
        
accuracy = metrics.accuracy_score(consolidated_proba_df['Actual'], consolidated_proba_df['predicted'])
precision = metrics.precision_score(consolidated_proba_df['Actual'], consolidated_proba_df['predicted'])
recall = metrics.recall_score(consolidated_proba_df['Actual'], consolidated_proba_df['predicted'])
f1_score = metrics.f1_score(consolidated_proba_df['Actual'], consolidated_proba_df['predicted'])
roc_auc = metrics.roc_auc_score(consolidated_proba_df['Actual'], consolidated_proba_df['predicted'])

print(" Accuracy {0:.3f}, \n Precision {1:.3f}, \n Recall {2:.3f}, \n f1_score {3:.3f}, \n roc_auc {4:.3f}".format(
    accuracy, precision,recall,f1_score,roc_auc))


# ### Plots of XGB

# In[ ]:


# First(0) column contains prob for 0-class and second(1) contains prob for 1-class
xgb_pred_df = pd.DataFrame({'Actual':consolidated_proba_df['Actual'], 'Probability':consolidated_proba_df['xgb_proba']}) 
return_dictionary = autoglm.calculateGLMKpis(xgb_pred_df, cutoff_by='Sensitivity-Specificity', include_cutoff_df_in_return=True)
cutoff_df = return_dictionary['cutoff_df']


# In[ ]:


cutoff_df.plot.line(x='Probability', y=['Accuracy','Sensitivity','Specificity'])
plt.show()


# In[ ]:


cutoff_df.plot.line(x='Probability', y=['Precision','Recall'])
plt.show()


# In[ ]:


prob_column='xgb_proba'
prob_cutoff = 0.2
consolidated_proba_df['predicted'] = consolidated_proba_df[prob_column].map(lambda x: 1 if x > prob_cutoff else 0)

local_confusion_matrix = metrics.confusion_matrix(consolidated_proba_df['Actual'], consolidated_proba_df['predicted'] )
        
accuracy = metrics.accuracy_score(consolidated_proba_df['Actual'], consolidated_proba_df['predicted'])
precision = metrics.precision_score(consolidated_proba_df['Actual'], consolidated_proba_df['predicted'])
recall = metrics.recall_score(consolidated_proba_df['Actual'], consolidated_proba_df['predicted'])
f1_score = metrics.f1_score(consolidated_proba_df['Actual'], consolidated_proba_df['predicted'])
roc_auc = metrics.roc_auc_score(consolidated_proba_df['Actual'], consolidated_proba_df['predicted'])

print(" Accuracy {0:.3f}, \n Precision {1:.3f}, \n Recall {2:.3f}, \n f1_score {3:.3f}, \n roc_auc {4:.3f}".format(
    accuracy, precision,recall,f1_score,roc_auc))


# ### Plots of new blend

# In[ ]:


# First(0) column contains prob for 0-class and second(1) contains prob for 1-class
final_blend_pred_df = pd.DataFrame({'Actual':consolidated_proba_df['Actual'], 'Probability':consolidated_proba_df['final_blended_proba']}) 
return_dictionary = autoglm.calculateGLMKpis(final_blend_pred_df, cutoff_by='Sensitivity-Specificity', include_cutoff_df_in_return=True)
cutoff_df = return_dictionary['cutoff_df']


# In[ ]:


cutoff_df.plot.line(x='Probability', y=['Accuracy','Sensitivity','Specificity'])
plt.show()


# In[ ]:


cutoff_df.plot.line(x='Probability', y=['Precision','Recall'])
plt.show()


# In[ ]:


prob_column='final_blended_proba'
prob_cutoff = 0.28
consolidated_proba_df['predicted'] = consolidated_proba_df[prob_column].map(lambda x: 1 if x > prob_cutoff else 0)

local_confusion_matrix = metrics.confusion_matrix(consolidated_proba_df['Actual'], consolidated_proba_df['predicted'] )
        
accuracy = metrics.accuracy_score(consolidated_proba_df['Actual'], consolidated_proba_df['predicted'])
precision = metrics.precision_score(consolidated_proba_df['Actual'], consolidated_proba_df['predicted'])
recall = metrics.recall_score(consolidated_proba_df['Actual'], consolidated_proba_df['predicted'])
f1_score = metrics.f1_score(consolidated_proba_df['Actual'], consolidated_proba_df['predicted'])
roc_auc = metrics.roc_auc_score(consolidated_proba_df['Actual'], consolidated_proba_df['predicted'])

print(" Accuracy {0:.3f}, \n Precision {1:.3f}, \n Recall {2:.3f}, \n f1_score {3:.3f}, \n roc_auc {4:.3f}".format(
    accuracy, precision,recall,f1_score,roc_auc))


# ### With blending we achived better Accuracy as well as other KPis like F1 and ROC improved
# 
# #### You can experiement with different combinations for parameter columns_to_blend
# 
# ### Upvote if you liked the Kernel. Leave comments if any. 

# In[ ]:




