#!/usr/bin/env python
# coding: utf-8

# In this little kernel I will showcase how to use a basic Random Forest classifier to tackle the TalkingData problem. Also, as a small bonus, I will throw in some basic validation as well, plotting the AUC score of our classifier. Let's get started by reading the train and test datasets.

# In[ ]:


import gc
import time
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['OMP_NUM_THREADS'] = '4'

path = '../input/'
dtypes = {
        'ip'            : 'uint32',
        'app'           : 'uint16',
        'device'        : 'uint16',
        'os'            : 'uint16',
        'channel'       : 'uint16',
        'is_attributed' : 'uint8',
        'click_id'      : 'uint32'
        }

print('started loading')

train_columns = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'is_attributed']
train_df = pd.read_csv(path+"train.csv", dtype=dtypes, skiprows = range(1, 131886954), usecols=['ip','app','device','os', 'channel', 'click_time', 'is_attributed'])
test_df = pd.read_csv(path+"test.csv", dtype=dtypes, usecols=['ip','app','device','os', 'channel', 'click_time', 'click_id'])

print('finished loading')


# The preprocessing step comes from [this excellent kernel](https://www.kaggle.com/alexanderkireev/deep-learning-support-9663).

# In[ ]:


def prep_data(d):
    print('hour, day, wday....')
    d['hour'] = pd.to_datetime(d.click_time).dt.hour.astype('uint8')
    d['day'] = pd.to_datetime(d.click_time).dt.day.astype('uint8')
    d['wday']  = pd.to_datetime(d.click_time).dt.dayofweek.astype('uint8')
    print('grouping by ip-day-hour combination....')
    gp = d[['ip','day','hour','channel']].groupby(by=['ip','day','hour'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'qty'})
    d = d.merge(gp, on=['ip','day','hour'], how='left')
    del gp; gc.collect()
    print('group by ip-app combination....')
    gp = d[['ip','app', 'channel']].groupby(by=['ip', 'app'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_app_count'})
    d = d.merge(gp, on=['ip','app'], how='left')
    del gp; gc.collect()
    print('group by ip-app-os combination....')
    gp = d[['ip','app', 'os', 'channel']].groupby(by=['ip', 'app', 'os'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_app_os_count'})
    d = d.merge(gp, on=['ip','app', 'os'], how='left')
    del gp; gc.collect()
    print("vars and data type....")
    d['qty'] = d['qty'].astype('uint16')
    d['ip_app_count'] = d['ip_app_count'].astype('uint16')
    d['ip_app_os_count'] = d['ip_app_os_count'].astype('uint16')
    print("label encoding....")
    from sklearn.preprocessing import LabelEncoder
    d[['app','device','os', 'channel', 'hour', 'day', 'wday']].apply(LabelEncoder().fit_transform)
    print('dropping')
    d.drop(['click_time', 'ip'], 1, inplace=True)
    
    return d


train_df = prep_data(train_df)
test_df = prep_data(test_df)
print("finished")


# Now we will split the training data into training and validation.

# In[ ]:


RANDOM_SEED = 1
import random
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

X_train, X_test = train_test_split(train_df, test_size=0.1, random_state=RANDOM_SEED)

y_train = X_train['is_attributed']
X_train = X_train.drop(['is_attributed'], axis=1)
y_test = X_test['is_attributed']
X_test = X_test.drop(['is_attributed'], axis=1)


# Let's now train our Random Forest Classifier (where `max_depth` shows the maximum depth of each tree).

# In[ ]:


from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(max_depth=9, random_state=0)
rf.fit(X_train, y_train)


# Since we do not want the classifier to predict 0 or 1, but we want to have output in the range [0, 1], we will use `predict_proba` for the predictions, which returns the probability of the item being assigned to each class.

# In[ ]:


predictions = rf.predict_proba(X_test)
predictions


# As we can see `predictions` is a 2D array of `(class1_prob, class2_prob)`, where `class1_prob = 1 - class2_prob`. This is not the desirable format. We want a single value for `is_attributed`. The first class corresponds to `0` and the second to `1`. So, we need to convert `predictions` to a proper 1D submission by subtracting from the whole 1 the first value of each probability prediction. That way we will end up with an 1D array of [0, 1] probability values.

# In[ ]:


def convert_preds(raw_preds):
    preds = []
    for p in raw_preds:
        preds.append(1 - p[0])
    return preds


# In[ ]:


val_preds = convert_preds(predictions)


# In[ ]:


max(val_preds)


# Let's now calculate and plot the AUC score of the validation set. The code for this can be [here](https://www.kaggle.com/antmarakis/calculating-and-plotting-auc-score).

# In[ ]:


import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix, precision_recall_curve, auc,
                             roc_curve, recall_score, classification_report, f1_score,
                             precision_recall_fscore_support)

fpr, tpr, thresholds = roc_curve(y_test, val_preds)
roc_auc = auc(fpr, tpr)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, label='AUC = %0.4f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.001, 1])
plt.ylim([0, 1.001])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show();


# Finally, let's predict items from the test dataset:

# In[ ]:


ids = test_df['click_id']
test_df.drop('click_id', axis=1, inplace=True)
predictions = rf.predict_proba(test_df)


# In[ ]:


sub_preds = convert_preds(predictions)


# In[ ]:


sub = pd.DataFrame()
sub['click_id'] = ids
sub['is_attributed'] = sub_preds
sub.to_csv('rf_sub.csv', index=False)

