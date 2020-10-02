#!/usr/bin/env python
# coding: utf-8

# In this notebook I correct the model-free benchmark from [ryches](https://www.kaggle.com/ryches/model-free-benchmark). I corrected rushing in opponent's field so that it is not always 50 yards but the distance from the scrimmage line to touchdown. I've also corrected the distribution so that it is always 100 or less.

# Note: the score of this notebook is better than 0.01417. If it's still 0.01417 than I just haven't run the latest iteration yet because I'm out of submissions for the day. If you submit this notebook you'll get a better score.

# You can use this notebook as post-processing and get a better score. Non-tree-based models will give you non-0/1 values for rows like Yards-99 which pretty much never happen

# In[ ]:


import pandas as pd
import numpy as np


# In[ ]:


train = pd.read_csv("../input/nfl-big-data-bowl-2020/train.csv")


# In[ ]:


train = train.drop_duplicates(subset="PlayId")


# Checking out what the yardage cumulative distribution function looks like from all of our data

# In[ ]:


dist = train["Yards"].hist(density = True, cumulative = True, bins = 200)


# We can split the dataset into times when in the opponent half and when in own half. This will help us a tiny bit in accurately plotting the distribution. More will be explained later

# In[ ]:


train_own = train[train["FieldPosition"] == train["PossessionTeam"]]
train_other = train[train["FieldPosition"] != train["PossessionTeam"]]


# In[ ]:


len(train_own),len(train_other)


# In[ ]:


import matplotlib.pyplot as plt
own_cdf = np.histogram(train_own["Yards"], bins=199,
                 range=(-99,100), density=True)[0].cumsum()
other_cdf = np.histogram(train_other["Yards"], bins=199,
                 range=(-99,100), density=True)[0].cumsum()


# In[ ]:


dist = train_own["Yards"].hist(density = True, cumulative = True, bins = 200)


# In[ ]:


dist = train_other["Yards"].hist(density = True, cumulative = True, bins = 200)


# You'll notice something very interesting when we print these two cdf's. When the rushing team is in their own half there is the possibility of them rushing all 100 yards of the field. But when they start a play from their opponents half they can only achieve a maximum of 50 yards on a play. This is reflected by the cdf maxing out at yard-50 and then being 1's for the rest of the yards. 
# 
# We can use these two different distributions and just apply them to all of the plays and it turns out this is a decent benchmark.
# 
# We can extend this even further by knowing that when the ball is in the opponents half the yard line determines the maximum number of yards the rushing team can travel. 

# cdf's for both are very similar. The difference between the two is negligble because 50+ yard rushes are extremely rare anyway

# In[ ]:


own_cdf


# In[ ]:


other_cdf


# In[ ]:


y_train = train["Yards"].values


# In[ ]:


y_ans = np.zeros((len(train),199))

for i,p in enumerate(y_train):
    for j in range(199):
        if j-99>=p:
            y_ans[i][j]=1.0


# In[ ]:


print("validation score own half:",np.sum(np.power(own_cdf-y_ans,2))/(199*(len(train))))
print("validation score other half:",np.sum(np.power(other_cdf-y_ans,2))/(199*(len(train))))


# We can see that our validation score is pretty good for both own half and other half distributions applied to the whole dataset. 

# In[ ]:


import matplotlib.pyplot as plt
plt.plot(own_cdf)
plt.plot(other_cdf)


# In[ ]:


from kaggle.competitions import nflrush
env = nflrush.make_env()
for (test_df, sample_prediction_df) in env.iter_test():
    if test_df["FieldPosition"].iloc[0] != test_df["PossessionTeam"].iloc[0]:
        #when they are in the opponents half
        cdf = np.copy(other_cdf)
        cdf[-(100 - (test_df["YardLine"].iloc[0])):] = 1
        sample_prediction_df.iloc[0, :] = cdf
#         print('a',-test_df["YardLine"].iloc[0])
#         print(sample_prediction_df.iloc[0, 80:199].tolist())
    else:
        #when they are in their own half
        cdf = np.copy(own_cdf)
        cdf[-(100 - (test_df["YardLine"].iloc[0]+50)):] = 1
        if (99-14) > -(100 - (test_df["YardLine"].iloc[0]+50)):
            cdf[:-(100 - (test_df["YardLine"].iloc[0]+50)) - 100] = 0
        sample_prediction_df.iloc[0, :] = cdf
#         print('b',-test_df["YardLine"].iloc[0])
#         print(sample_prediction_df.iloc[0, 80:199].tolist())
    env.predict(sample_prediction_df)

env.write_submission_file()


# In[ ]:




