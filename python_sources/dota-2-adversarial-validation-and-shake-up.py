#!/usr/bin/env python
# coding: utf-8

# In this Kernel I'd like to share a couple of ideas which I found on Kernels myself:
#  - adversarial validation
#  - shakeup simulation
# 
# Originals where I've first seen these ideas implemented:
#  - [Quora adversarial validation](https://www.kaggle.com/tunguz/quora-adversarial-validation) by Bojan Tunguz
#  - [Quora shake-up simulation](https://www.kaggle.com/c/quora-insincere-questions-classification/discussion/78952#463246) by some other master
#  
# ##  Adversarial validation
#  
# The idea of adversarial validation is to forget targets at all, concatenate training and test features, and label train instances with 1s and test instances - with 0s. Then we perform cross-validation, and finally see whether a model can distinguish train from test (if in such classification setting scores are pretty high) or not. 
#  - In the first case we can identify features (via weights/feature importances) which help to distinguish train from test. Such features are "suspicious" - their distribution has most likely changes in the test set as compared to the training set. These features deserve special investigation (see [this example](https://www.kaggle.com/tunguz/ms-malware-adversarial-validation), again by Bojan Tunguz, from that funny [MS malware competition](https://www.kaggle.com/c/microsoft-malware-prediction/))
#  - In the second case the model can not distinguish train from test just based on the provided features, that's a good sign

# In[ ]:


import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


PATH_TO_DATA = '../input/'

df_train_features = pd.read_csv(os.path.join(PATH_TO_DATA, 
                                             'train_features.csv'), 
                                    index_col='match_id_hash')
df_test_features = pd.read_csv(os.path.join(PATH_TO_DATA, 'test_features.csv'), 
                                   index_col='match_id_hash')


# We concatenate training and test sets, only features, no targets whatsoever.

# In[ ]:


df = pd.concat([df_train_features, df_test_features])


# New target is just a binary feature indicating training instances:

# In[ ]:


is_train = [1] * len(df_train_features) + [0] * len(df_test_features)


# Grab just basic RF model and perform 5-fold cross-validation.

# In[ ]:


model = RandomForestClassifier(n_estimators=100, n_jobs=4, random_state=17)


# In[ ]:


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=17)


# In[ ]:


get_ipython().run_cell_magic('time', '', "adv_validation_scores = cross_val_score(model, df, is_train, cv=skf, n_jobs=4,\n                                       scoring='roc_auc')")


# We see "nice" AUCs around 51-52%. So the model doesn't distinguish train and test. That's good! So distributions of basic features that we've passed into the model are pretty much the same in training and test sets. 

# In[ ]:


adv_validation_scores


# Feature importance doesn't surprise at all. However, in case of high adversarial validation scores do explore it carefully. 

# In[ ]:


model.fit(df, is_train)


# In[ ]:


import eli5


# In[ ]:


eli5.show_weights(estimator=model, feature_names=list(df.columns))


# ## Shake-up simulation
# 
# Oh, it's just a fancy name for repeated validation where holdout size is equal to the test size in our competition. You split the training set (now with original targets, forget about adversarial validation) randomly multiple times (`sklearn.model_selection.StratifiedShuffleSplit`), setting test_size to be 10k in our case, and collect statistics about the distribution of these holdout scores. In such a way you can sort of estimate the degree of a possible shake-up (i.e., variation in private LB scores). 

# In[ ]:


from sklearn.model_selection import StratifiedShuffleSplit


# In[ ]:


skf = StratifiedShuffleSplit(n_splits=150, test_size=10000, random_state=1)


# Now we actually need original targets.

# In[ ]:


df_train_targets = pd.read_csv(os.path.join(PATH_TO_DATA, 
                                            'train_targets.csv'), 
                                   index_col='match_id_hash')


# In[ ]:


get_ipython().run_cell_magic('time', '', "scores = cross_val_score(model, df_train_features, df_train_targets['radiant_win'], \n                         cv=skf, n_jobs=4, scoring='roc_auc')")


# In[ ]:


scores = pd.Series(scores)
mean = scores.mean()
lower, upper = mean - 2 * scores.std(), mean + 2 * scores.std()


# In[ ]:


from matplotlib import pyplot as plt
import seaborn as sns


# Plot the distribution of validation scores.

# In[ ]:


sns.distplot(scores, bins=10);
plt.vlines(x=[mean], ymin=0, ymax=110, 
           label='mean', linestyles='dashed');
plt.vlines(x=[lower, upper], ymin=0, ymax=110, 
           color='red', label='+/- 2 std',
          linestyles='dashed');
plt.legend();


# Looks like variations up to +/- 0.01 AUC are expected.
# 
# So here we've covered two interesting validation hacks. Maybe they are not extremely useful in the DotA 2 winner prediction competition, but at least the techniques are interesting and it's good to know them.
