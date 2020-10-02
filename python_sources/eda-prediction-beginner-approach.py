#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set()
from sklearn.impute import SimpleImputer


# ### Loading datasets

# In[ ]:


train = pd.read_csv("../input/training_set_metadata.csv")
test = pd.read_csv("../input/test_set_metadata.csv")
sub_sample = pd.read_csv("../input/sample_submission.csv")


# In[ ]:


print(train.shape)
train.head()


# In[ ]:


print(test.shape)
test.head()


# In[ ]:


sub_sample.head()


# __It seems, submission format requires One Hot Encoded `target` class with `object_id`__.

# In[ ]:


# but class 99 missing here
sorted(train.target.unique())


# In[ ]:


# checking null/missing values in train dataset
train.isnull().sum()


# __`distmod` feature contains `NaN` values__.

# In[ ]:


# impute missing values by replacing them with mean values of that feature
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
train["distmod"] = imp.fit_transform(np.array(train["distmod"]).reshape(-1,1))


# In[ ]:


train.isnull().sum().any()


# In[ ]:


# checking null values in test dataset
test.isnull().sum()


# __`hostgal_specz` and `distmod`  feature contains `NaN` values__.

# In[ ]:


# different approach to deal with null values
test.distmod.fillna(test["distmod"].mean(), inplace=True)
test.hostgal_specz.fillna(test.hostgal_specz.mean(), inplace=True)


# In[ ]:


test.isnull().sum().any()


# In[ ]:


# checking correlation between features
plt.figure(figsize=(12,9))
sns.heatmap(train.corr(), annot=True, fmt=".1f", cmap="RdYlBu")


# __No highly correlated feature present in the train dataset__.

# ### Data Visualization

# In[ ]:


# lets look at the target distribution visually
plt.figure(figsize=(15,6))
train.target.value_counts().sort_index().plot.bar()


# In[ ]:


colors = np.random.rand(train.shape[0])
area = (25 * np.random.rand(train.shape[0]))**2
        
plt.subplots(figsize=(15,6))
plt.scatter(train.distmod, train.hostgal_specz, s = area, c = colors, alpha = 0.5)
plt.xlabel("distmod")
plt.ylabel("host galaxy spectroscopic redshift")
plt.show()


# In[ ]:


colors = np.random.rand(train.shape[0])
area = (25 * np.random.rand(train.shape[0]))**2
        
plt.subplots(figsize=(15,6))
plt.scatter(train.distmod, train.hostgal_photoz, s = area, c = colors, alpha = 0.5)
plt.xlabel("distmod")
plt.ylabel("host galaxy photometric redshift")
plt.show()


# In[ ]:


colors = np.random.rand(train.shape[0])
area = (25 * np.random.rand(train.shape[0]))**2
        
plt.subplots(figsize=(15,6))
plt.scatter(train.distmod, train.hostgal_photoz_err, s = area, c = colors, alpha = 0.5)
plt.xlabel("distmod")
plt.ylabel("host galaxy photometric redshift error")
plt.show()


# In[ ]:


fig, ax = plt.subplots(figsize=(18,8))

for target_class in train.target.unique():
    used_class = train[train.target == target_class]
    
    colors = np.random.rand(len(used_class))
    area = (25 * np.random.rand(len(used_class)))**2
    
    ax.scatter(x = used_class.gal_l, y = used_class.gal_b, alpha = 0.5, s = area, c = colors)

plt.xlabel("galactical longitude")
plt.ylabel("galactical latitude")


# In[ ]:


# adding 99 target class manually
targets = np.hstack([np.unique(train['target']), [99]])


# In[ ]:


targets


# In[ ]:


target_map = {j : i for i, j in enumerate(targets)}


# In[ ]:


target_map


# In[ ]:


target_ids = [target_map[i] for i in train['target']]
train["target_id"] = target_ids # adding a new feature to the train dataset


# In[ ]:


train.head()


# __Below code taken from this [kernel](https://www.kaggle.com/kyleboone/naive-benchmark-galactic-vs-extragalactic)__.

# In[ ]:


# Build the flat probability arrays for both the galactic and extragalactic groups
galactic_cut = train['hostgal_specz'] == 0


# In[ ]:


print(galactic_cut[:5])


# In[ ]:


galactic_data = train[galactic_cut]


# In[ ]:


galactic_data.head()


# In[ ]:


extragalactic_data = train[~galactic_cut]
galactic_classes = np.unique(galactic_data['target_id'])


# In[ ]:


extragalactic_data.head()


# In[ ]:


galactic_classes


# In[ ]:


extragalactic_classes = np.unique(extragalactic_data['target_id'])


# In[ ]:


extragalactic_classes


# In[ ]:


# Add class 99 (id=14) to both groups.
galactic_classes = np.append(galactic_classes, 14)
extragalactic_classes = np.append(extragalactic_classes, 14)


# In[ ]:


galactic_probabilities = np.zeros(15)


# In[ ]:


galactic_probabilities


# In[ ]:


galactic_probabilities[galactic_classes] = 1. / len(galactic_classes)


# In[ ]:


galactic_probabilities[galactic_classes]


# In[ ]:


extragalactic_probabilities = np.zeros(15)
extragalactic_probabilities[extragalactic_classes] = 1. / len(extragalactic_classes)


# In[ ]:


extragalactic_probabilities[extragalactic_classes]


# In[ ]:


# Apply this prediction to a table
def do_prediction(table):
    probs = []
    for index, row in tqdm.tqdm(table.iterrows(), total=len(table)):
        if row['hostgal_photoz'] == 0:
            prob = galactic_probabilities
        else:
            prob = extragalactic_probabilities
        probs.append(prob)
    return np.array(probs)


# In[ ]:


pred = do_prediction(train)


# In[ ]:


pred


# In[ ]:


test_pred = do_prediction(test)


# In[ ]:


col_names = ['class_%d' % i for i in targets]
submission_df = pd.DataFrame(data = test_pred, columns = col_names)


# In[ ]:


submission_df.insert(0, "object_id", test["object_id"].values)


# In[ ]:


submission_df.head()


# In[ ]:


submission_df.to_csv("submission.csv", index=False)


# In[ ]:




