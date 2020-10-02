#!/usr/bin/env python
# coding: utf-8

# # Comparison of Train and Test Sets
# In this notebook, I will investigate whether the data in the train and test sets are comparable or not. If not, it might be difficult to train a model that performs  well on both train and test data. Even when the performance in cross-validation is good on the training data, we might have a suboptimal model on the test data.

# In[ ]:


# Import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# ### Check data overlap
# We can check whether the train and test sets overlap, by checking their Ids.

# In[ ]:


# Load train and test sets
train = pd.read_csv("/kaggle/input/learn-together/train.csv")
test = pd.read_csv("/kaggle/input/learn-together/test.csv")

# Check that Ids are unique and distinct between train and test set
print('Id in train set is unique.') if train.Id.nunique() == train.shape[0] else print('Id is not unique in train set')
print('Train and test sets are distinct.') if len(np.intersect1d(train.Id.values, test.Id.values))== 0 else print('Ids in train and test overlap')


# ### Compare interval columns
# Here we will compare the distribution of the interval columns.
# As we can see below, the train and test set do not have similar distributions.

# In[ ]:


def compare_datasets(feature):
    fig, ax = plt.subplots(figsize=(6,4))
    sns.distplot(test[feature], ax=ax, kde=True, hist=False, label='test', kde_kws={'color': 'b', 'lw': 2})
    sns.distplot(train[feature], ax=ax, kde=True, hist=False, label='train', kde_kws={'color': 'g', 'lw': 2})
    plt.title('Comparison of ' + feature + ' in train and test set')
    plt.legend();

interval_cols = ['Elevation',
 'Aspect',
 'Slope',
 'Horizontal_Distance_To_Hydrology',
 'Vertical_Distance_To_Hydrology',
 'Horizontal_Distance_To_Roadways',
 'Hillshade_9am',
 'Hillshade_Noon',
 'Hillshade_3pm',
 'Horizontal_Distance_To_Fire_Points']

for c in interval_cols:
    compare_datasets(c)


# ### Compare boolean variables
# For the boolean variables, we will compare the proportion between the train and test sets.
# There are substantial differences in mean for some boolean variables. For instance, the proportion for Wilderness_Areas 1 and 4 are very different. Additionally, Soil_Types 10 and 29 differ a lot between the train and test set. 

# In[ ]:


def compare_bools_traintest(feature_list):
    train_feat = train[feature_list].mean()
    train_feat.name = 'Train'
    test_feat = test[feature_list].mean()
    test_feat.name = 'Test'
    print(pd.concat([train_feat, test_feat], axis=1))

wilderness_cols = [c for c in train.columns if c.startswith('Wilderness')]
soil_cols = [c for c in train.columns if c.startswith('Soil')]

compare_bools_traintest(wilderness_cols)
compare_bools_traintest(soil_cols)


# # Conclusion 
# The data in the train set is not completely representative of the test data on which submissions will be evaluated. Therefore, it might be difficult to train a model that performs equally well on the test set. 
