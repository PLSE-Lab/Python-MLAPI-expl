#!/usr/bin/env python
# coding: utf-8

# # Task: Predict which customer group is worth targeting
# 
# The goal of this task is to determine which group of customers are worth targeting given the following dataset. The data is highly ananomized so there isn't a good possibility to apply domain expertise. My intial goal for this task were to try out the stacked classifier from SK learn (new in version 0.22.1).
# 
# Thanks to:
# * Thanisis https://www.kaggle.com/tsiaras for the dataset https://www.kaggle.com/tsiaras/predicting-profitable-customer-segments and also for the task

# In[ ]:


pip install --upgrade scikit-learn


# In[ ]:


import sklearn
print(sklearn.__version__)


# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

pd.set_option("display.max_columns", 80)
pd.set_option("display.max_rows", 100)


# In[ ]:


data = pd.read_csv('/kaggle/input/predicting-profitable-customer-segments/customerTargeting.csv')
data.head()


# # Data:
# The columns starting with 'g1_' represent info about the first group
# the columns starting with 'g2_' represent infor about the second group
# the columns with 'c_' are features representing a comparison of the two groups
# 
# 'target' is the value we are trying to predict. A value of 0 indicates that neither group was profitable after the marketing campaign, a value of  1 indicates group 1 was more profitable and a value of 2 indicates that a value of 2 was more profitable

# In[ ]:


data.info() #no data missing and all values are already integers/floats


# In[ ]:


data.describe()


# In[ ]:


corr_df = data.corr()
corr_df #unhide the output to see the full set of corrolation coeficients


# In[ ]:


import matplotlib
import seaborn as sns
from matplotlib.pyplot import figure
figure(num=None, figsize=(12, 12))

sns.heatmap(corr_df)


# In[ ]:


sns.countplot(data['target'])


# In[ ]:


data['target'].value_counts()


# In[ ]:


def enc_0(num):
    if num == 0:
        return 1
    else:
        return 0

def enc_1(num):
    if num == 1:
        return 1
    else:
        return 0

def enc_2(num):
    if num == 2:
        return 1
    else:
        return 0

data['target_0'] = pd.Series([enc_0(x) for x in data.target], index=data.index)
data['target_1'] = pd.Series([enc_1(x) for x in data.target], index=data.index)
data['target_2'] = pd.Series([enc_2(x) for x in data.target], index=data.index)

## One hot encoding the outcomes didn't help any of the models I tried
## But there is one possibility with this I left unexplored, see my final thoughts for more details


# In[ ]:


corr_df = data.corr()
corr_df = corr_df.sort_values(by=['target'])
pd.DataFrame(corr_df['target'])


# I had experimented with removing c_15, c_9, and c_7, because of their really low corrolation coeficients. After testing I discovered they had a not insignifigant effect of the capture rate of 0 for the target.

# In[ ]:


# There are other methods of generating this in less code
# I just want my mistakes to be a little easier to spot
# the d_x refers to the delta between g1_x and g2_x
data['d_1'] = data['g1_1'] - data['g2_1']
data['d_2'] = data['g1_2'] - data['g2_2']
data['d_3'] = data['g1_3'] - data['g2_3']
data['d_4'] = data['g1_4'] - data['g2_4']
data['d_5'] = data['g1_5'] - data['g2_5']
data['d_6'] = data['g1_6'] - data['g2_6']
data['d_7'] = data['g1_7'] - data['g2_7']
data['d_8'] = data['g1_8'] - data['g2_8']
data['d_9'] = data['g1_9'] - data['g2_9']
data['d_10'] = data['g1_10'] - data['g2_10']
data['d_11'] = data['g1_11'] - data['g2_11']
data['d_12'] = data['g1_12'] - data['g2_12']
data['d_13'] = data['g1_13'] - data['g2_13']
data['d_14'] = data['g1_14'] - data['g2_14']
data['d_15'] = data['g1_15'] - data['g2_15']
data['d_16'] = data['g1_16'] - data['g2_16']
data['d_17'] = data['g1_17'] - data['g2_17']
data['d_18'] = data['g1_18'] - data['g2_18']
data['d_19'] = data['g1_19'] - data['g2_19']
data['d_20'] = data['g1_20'] - data['g2_20']
data['d_21'] = data['g1_21'] - data['g2_21']


# In[ ]:


features = ['c_1', 'c_2', 'c_3', 'c_4', 'c_5', 'c_6','c_7', 'c_8', 'c_9', 'c_10', 'c_11', 'c_12', 'c_13',
            'c_14', 'c_15', 'c_16', 'c_17', 'c_18', 'c_19', 'c_20', 'c_21', 'c_22', 'c_23', 'c_24', 'c_25',
            'c_26', 'c_27', 'c_28', 'd_1', 'd_2', 'd_3', 'd_4', 'd_5', 'd_6', 'd_7', 'd_8', 'd_9', 'd_10',
            'd_11', 'd_12', 'd_13', 'd_14', 'd_15', 'd_16', 'd_17', 'd_18', 'd_19', 'd_20', 'd_21']#,
            #'g1_1', 'g1_5', 'g2_12', 'g2_21', 'g2_1', 'g2_13', 'g2_11']


# I had expirimented with many different features to include, ultimately the delta variable set proved to add the most value. I commented out some of the variables that had boosted the models accuracy but had an unintended side affect (described below).

# In[ ]:


X = data[features]
y = data['target'] #[['target_0', 'target_1', 'target_2']] # one hot encoding did not improve performance


# In[ ]:


from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split


from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score


# In[ ]:


train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 1, test_size=.2)


# In[ ]:


hgb = HistGradientBoostingClassifier()
hgb.fit(train_X, train_y)
ls_preds = hgb.predict(val_X)
print('Score of Histogram-based Gradient Boosting Regression Tree: ', accuracy_score(val_y, ls_preds))


# In[ ]:


from sklearn.metrics import confusion_matrix

confuse = confusion_matrix(val_y, ls_preds)
confuse = pd.DataFrame(confuse)
figure(num=None, figsize=(7, 7))
sns.heatmap(confuse, linewidths=1,annot=True, fmt='.5g', annot_kws={"size": 12},cmap="YlGnBu", 
            yticklabels=['0 True','1 True','2 True'], xticklabels=['0 Predicted','1 Predicted','2 Predicted'])


# The above model is my final model. I had some other models that were more accurate but this model did a better job of catching True negitives. through feature engineering, hyperparameter tuning, and feature selection I was able to get slightly higher accuracy in other models but each of them wouldn't predict a 0 state (which remember meant that neither group was profitable to market to).
# 
# # Accuracy
# This model is only 56% accurate. The model had to detect one of 3 distinct outcomes, if each of those outcomes were equally likely then deciding at random would give us an accuracy of 33%. Since group one being more profitable is the most common outcome, a model can be 46% accurate just by always predicting target to be 1. 
# 
# Below there are several other models hidden, which had decent enough performance to be worth a look, feel free to unhide them if your curious. The other visible model below is the one I initially set forth to test out, the Stacked Classifier, but I'm not hugely thrilled with its performance.

# In[ ]:


from sklearn.ensemble import ExtraTreesClassifier
etc = ExtraTreesClassifier(n_estimators=90, random_state=0)
etc.fit(train_X, train_y)
ls_preds = etc.predict(val_X)
print('Score of Etremely Random Forest: ', accuracy_score(val_y, ls_preds))


# In[ ]:


from sklearn.metrics import confusion_matrix

confuse = confusion_matrix(val_y, ls_preds)
confuse = pd.DataFrame(confuse)
figure(num=None, figsize=(7, 7))
sns.heatmap(confuse, linewidths=1,annot=True, fmt='.5g', annot_kws={"size": 12},cmap="YlGnBu", 
            yticklabels=['0 True','1 True','2 True'], xticklabels=['0 Predicted','1 Predicted','2 Predicted'])


# In[ ]:


from sklearn.ensemble import BaggingClassifier
baggin = BaggingClassifier(n_estimators=90)
baggin.fit(train_X, train_y)
ls_preds = baggin.predict(val_X)
print('Score of Histogram-based Gradient Boosting Regression Tree: ', accuracy_score(val_y, ls_preds))


# In[ ]:


confuse = confusion_matrix(val_y, ls_preds)
confuse = pd.DataFrame(confuse)
figure(num=None, figsize=(7, 7))
sns.heatmap(confuse, linewidths=1,annot=True, fmt='.5g', annot_kws={"size": 12},cmap="YlGnBu", 
            yticklabels=['0 True','1 True','2 True'], xticklabels=['0 Predicted','1 Predicted','2 Predicted'])


# In[ ]:


import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier

estimators = [
    ('rf', ExtraTreesClassifier(n_estimators=90, random_state=0)),
    ('hgb', RandomForestClassifier(n_estimators=40)),
    ('ada', HistGradientBoostingClassifier()),
    ('bag', BaggingClassifier(n_estimators=70))
    ]
clf = StackingClassifier(estimators=estimators, final_estimator=AdaBoostClassifier(n_estimators=40))
clf.fit(train_X, train_y)
ls_preds = clf.predict(val_X)
print('Score of stacked: ', accuracy_score(val_y, ls_preds))


# In[ ]:


confuse = confusion_matrix(val_y, ls_preds)
confuse = pd.DataFrame(confuse)
figure(num=None, figsize=(7, 7))
sns.heatmap(confuse, linewidths=1,annot=True, fmt='.5g', annot_kws={"size": 12},cmap="YlGnBu", 
            yticklabels=['0 True','1 True','2 True'], xticklabels=['0 Predicted','1 Predicted','2 Predicted'])


# # Final Thoughts:
# 
# I set out to try out the StackingClassifier and I can say I was successful with that. There is still room for improvement in the model, and I imagine more signal to tease out of the dataset. I look forward to see what others try with this task, I'm really curious what I may not have found. 
# 
# Each of the models that I created did rather poorly at predicting an outcome of 0 (where neither group was profitable). One possibility that I did not explore was the option of training a seperate model (probably a decision tree) and tune it to just detect 0 or not 0 outcome. Including such model in the stack could have been useful.
# 
# I'll also acknowledge that I'm not overly experienced with model stacking/blending so I may have overlooked some better model designs. 
# 
# Currently I'm challenging myself to complete one Kaggle Task per week. If you have any constructive feedback or want to collaborate with me on a future task feel free to message me or leave a comment on this notebook.
# 
# As always upvotes are appreciated.
