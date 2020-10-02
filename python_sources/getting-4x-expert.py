#!/usr/bin/env python
# coding: utf-8

# ## Hi everyone from Kaggle !

# In this notebook, I will try to find out how many people have achieved the expert level (`tier=2`) in all 4 categories, that is to say, **Discussion, Kernels, Datasets and Competitions**.

# For **Discussion, Kernels, and Competitions** it's actually quite easy, but we will see that it gets a lot trickier with **Datasets**, since we are missing some information (we don't have, for each dataset id, the list of all its version ids, what prevents us from matching a dataset with all its upvotes, only with the upvotes on the latest version). So please, Kaggle Team, add this information !

# In[ ]:


import numpy as np
import pandas as pd 
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        if 'User' in filename or 'Data' in filename:
            print(os.path.join(dirname, filename))


# Let's first take a look at the users achievements :

# In[ ]:


achievements = pd.read_csv('/kaggle/input/meta-kaggle/UserAchievements.csv')
achievements.head()


# In[ ]:


np.unique(achievements.AchievementType), sorted(np.unique(achievements.Tier))


# Now, let's take only `UserId` that are at least expert (`Tier>=2`) in all 3 categories present in `achievements` : 

# In[ ]:


from copy import deepcopy as dc
def get_users_all_level(tier=0):
    ach = dc(achievements)
    if tier < 5: #in this case we remove kaggle team
        ach = ach.loc[ach.Tier < 5]
    d = ach.loc[(ach.Tier>=tier) & (ach.AchievementType=='Discussion'), 'UserId'].values.tolist()
    c = ach.loc[(ach.Tier>=tier) & (ach.AchievementType=='Competitions'), 'UserId'].values.tolist()
    s = ach.loc[(ach.Tier>=tier) & (ach.AchievementType=='Scripts'), 'UserId'].values.tolist()
    return [user for user in c if user in d and user in s]


# Look at how many users achieved a given level in all 3 categories Discussion, Competition and Kernels. We are interested in people who are at least 3x expert.

# In[ ]:


print(f'We have {len(get_users_all_level(tier=2))} experts in 3, {len(get_users_all_level(tier=3))} masters in 3 and only {len(get_users_all_level(tier=4))} grandmasters in 3 !')
interest_users = get_users_all_level(tier=2)


# There are 254 users who are expert or more in all 3 categories Competitions, Discussion and Script !

# Now let's look for those among them who are also Dataset experts. That's more tricky. In order to do this, we will look at the datasets published by those users. To get Dataset expert you need at least 3 bronze medals in datasets, that means at least 3 datasets with at least 5 non-novice and non-self votes.

# To filter these `interest_users`, we first need to get the legitimate upvoters on their datasets : non-novice users.

# In[ ]:


users = pd.read_csv('/kaggle/input/meta-kaggle/Users.csv')
users.columns = ['UserId', 'UserName', 'DisplayName', 'RegisterDate', 'PerformanceTier']
users = users.loc[users.PerformanceTier>0]
users.sample(5)


# Let's take a look at who these `interest_users` are :

# In[ ]:


users.loc[users.UserId.isin(interest_users)].sample(10)


# Looking for me...

# In[ ]:


users.loc[users.UserName=='louise2001']


# In[ ]:


my_userid = users.loc[users.UserName=='louise2001', 'UserId'].values[0]
my_userid in interest_users


# Here I am !

# In[ ]:


total = pd.read_csv('/kaggle/input/meta-kaggle/Users.csv').shape[0]
novices = total - users.shape[0]
contributors = users.loc[users.PerformanceTier==1].shape[0]
experts = users.loc[users.PerformanceTier==2].shape[0]
masters = users.loc[users.PerformanceTier==3].shape[0]
grandmasters  = users.loc[users.PerformanceTier==4].shape[0]
kaggle_team = users.loc[users.PerformanceTier==5].shape[0]
print(f'Out of a total of {total} registered Kaggle users, we have :')
print(f'{novices} novices ({round(novices/total*100,3)}%)')
print(f'{contributors} contributors ({round(contributors/total*100,3)}%)')
print(f'{experts} experts ({round(experts/total*100,3)}%)')
print(f'{masters} masters ({round(masters/total*100,3)}%)')
print(f'{grandmasters} grandmasters ({round(grandmasters/total*100,3)}%)')
print(f'{kaggle_team} Kaggle Team ({round(kaggle_team/total*100,3)}%)')


# We can now use this to count only votes cast by non-novices :

# In[ ]:


data_votes = pd.read_csv('/kaggle/input/meta-kaggle/DatasetVotes.csv')
data_votes = data_votes.loc[data_votes.UserId.isin(users.UserId.values)]
data_votes.sample(5)


# Here, you can notice the core of the problem : we have the `DatasetVersionId` of each vote, not the `DatasetId`, which will make it impossible to match exactly.

# Let's now take a look at the datasets. We are only interested in datasets with at least 5 votes, and created by our interest_users :

# In[ ]:


datasets = pd.read_csv('/kaggle/input/meta-kaggle/Datasets.csv')
datasets.columns = ['DatasetId', 'CreatorUserId', 'OwnerUserId', 'OwnerOrganizationId',
       'CurrentDatasetVersionId', 'CurrentDatasourceVersionId', 'ForumId',
       'Type', 'CreationDate', 'ReviewDate', 'FeatureDate', 'LastActivityDate',
       'TotalViews', 'TotalDownloads', 'TotalVotes', 'TotalKernels']
datasets = datasets.loc[(datasets['TotalVotes']>=5) & (datasets['CreatorUserId'].isin(interest_users)), ['DatasetId', 'CreatorUserId',
       'CurrentDatasetVersionId', 'CurrentDatasourceVersionId', 'CreationDate', 'TotalVotes']]
datasets.sample(5)


# What we can already do is filter out all `interest_users` that don't have at least 3 datasets with at least 5 votes each, since they can't be Datasets expert. Note : that doesn't guarantee that those who remain actually are dataset experts, since there could be votes cast by themselves or by novices, but that's a first step.

# In[ ]:


interest_users[:] = [user for user in interest_users if datasets.loc[datasets.CreatorUserId==user].shape[0]>=3]
len(interest_users)


# Wow, we only have 53 candidates left ! Let's refilter `datasets` on those users, so that it's easier to process in the following steps.

# In[ ]:


datasets = datasets.loc[datasets.CreatorUserId.isin(interest_users)]
datasets


# Sadly, we don't have, for each `DatasetId`, the list of all its `DatasetVersionId`. Therefore we can only match votes that have been made on the latest version of a dataset... We might miss a lot of people, unfortunately.

# In[ ]:


datasets = datasets.merge(data_votes, left_on='CurrentDatasetVersionId', right_on = 'DatasetVersionId')
datasets


# Remove self votes : 

# In[ ]:


datasets = datasets.loc[datasets.CreatorUserId != datasets.UserId]
datasets


# We just have to substract datasets that have at least 5 votes :

# In[ ]:


size = datasets.groupby(['DatasetId']).size()
size = size[size>=5]
datasets = datasets.loc[datasets['DatasetId'].isin(size.index)]
dataset_experts = np.unique(datasets.CreatorUserId.values)


# In[ ]:


interest_users[:] = [user for user in interest_users if user in dataset_experts]
len(interest_users)


# So in the end, we get only 51 of them... note that it could be very underestimated since we cannot have, for each `DatasetId`, the list of all its `DatasetVersionId`, therefore we can only match votes cast on the latest version of a dataset.

# In[ ]:


my_userid in interest_users


# Here I am !

# # Thanks for reading, hope you found it interesting !
