#!/usr/bin/env python
# coding: utf-8

# The goal is to see if it's more likely to have a higher loyalty score if the card holder keeps using the payment system in the new transaction files in new merchants, because if they do and intuitively it means they like the service.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

from matplotlib import pyplot as plt

# Any results you write to the current directory are saved as output.


# In[ ]:


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
hist_tran = pd.read_csv("../input/historical_transactions.csv")
new_tran = pd.read_csv("../input/new_merchant_transactions.csv")

print("loading done")


# In[ ]:


overlap_card_train_hist = set(train["card_id"]) & set(hist_tran["card_id"])
overlap_card_train_new = set(train["card_id"]) & set(new_tran["card_id"])

overlap_card_test_hist = set(test["card_id"]) & set(hist_tran["card_id"])
overlap_card_test_new = set(test["card_id"]) & set(new_tran["card_id"])

print(len(overlap_card_train_hist), len(overlap_card_train_new), len(train["card_id"]))
print(len(overlap_card_test_hist), len(overlap_card_test_new), len(test["card_id"]))


# It seems the spliting of train and test are solely based on card id hashing. Both appear 100% in historical transactions and both have some missing entries in the new transaction file meaning they didn't use it in new merchants.

# Now let's see if the assumption is true - that card holders continuously use the service in new merchants should tend to have higher loyalty scores.

# In[ ]:


train["card_in_new"] = pd.Series([i in overlap_card_train_new for i in train["card_id"]], index=train.index)

scores_not_in_new = train[train["card_in_new"] == False]["target"]
scores_not_in_new.describe()


# In[ ]:


scores_in_new = train[train["card_in_new"] == True]["target"]
scores_in_new.describe()


# Now this doesn't make sense - the original expectation is for scores_in_new to be in general > scores_not_in_new, but the mean of scores_in_new is actually smaller. Let's plot the data.

# In[ ]:



from matplotlib import pyplot as plt
plt.figure(figsize=(12, 5))
plt.hist(scores_not_in_new.values, bins=200)
plt.title('Histogram target counts for scores_not_in_new')
plt.xlabel('Count')
plt.ylabel('Target')
plt.show()


# In[ ]:


plt.figure(figsize=(12, 5))
plt.hist(scores_in_new.values, bins=200)
plt.title('Histogram target counts for scores_in_new')
plt.xlabel('Count')
plt.ylabel('Target')
plt.show()


# It seems we have outliers, let's try removing them

# In[ ]:


train_cleaned = train[train["target"] > -30]

scores_not_in_new_cleaned = train_cleaned[train_cleaned["card_in_new"] == False]["target"]
print(scores_not_in_new_cleaned.describe())

scores_in_new_cleaned = train_cleaned[train_cleaned["card_in_new"] == True]["target"]
print(scores_in_new_cleaned.describe())


# In[ ]:



from matplotlib import pyplot as plt
plt.figure(figsize=(12, 5))
plt.hist(scores_not_in_new_cleaned.values, bins=200)
plt.title('Histogram target counts for scores_not_in_new_cleaned')
plt.xlabel('Count')
plt.ylabel('Target')
plt.show()


# In[ ]:



from matplotlib import pyplot as plt
plt.figure(figsize=(12, 5))
plt.hist(scores_in_new_cleaned.values, bins=200)
plt.title('Histogram target counts for scores_in_new_cleaned')
plt.xlabel('Count')
plt.ylabel('Target')
plt.show()


# Now we see an interesting pattern. For the card holders which appear in both hist and new transaction history, there's a gap in "target=0", whereas for the other group (card holder not appeared in new transaction history), there's a huge spike in target=0. 
# 
# Let's understand this more quantitatively.

# In[ ]:



print(
    len(scores_in_new_cleaned[scores_in_new_cleaned == 0.0]),
    len(scores_in_new_cleaned),
    len(scores_in_new_cleaned[scores_in_new_cleaned == 0.0]) / len(scores_in_new_cleaned),
)
print(
    len(scores_not_in_new_cleaned[scores_not_in_new_cleaned == 0.0]),
    len(scores_not_in_new_cleaned),
    len(scores_not_in_new_cleaned[scores_not_in_new_cleaned == 0.0]) / len(scores_not_in_new_cleaned)
)


# The difference is really big. 7% in the disppearing group have 0 score, whereas 0.06% in overlap group has 0 score. This leads to my guess that:
# 1. loyalty score = 0 is likely a default value, which we assign when there's not enough information for us to determine the loyalty score
# 2. And as long as we have **some** information, there loyalty score will very rarely become 0 (it might be very small value but really rarely 0)

# I'm also intrigued by where those 117 come from - do we apply some kind of thresholds like, on the number of new merchants being developed in new transaction?

# In[ ]:


card_id_to_new_merchants = {}
for _, row in new_tran.iterrows():
    card_id_to_new_merchants.setdefault(row.card_id, set())
    card_id_to_new_merchants[row.card_id].add(row.merchant_id)
    


# In[ ]:


train_cleaned["new_merchant_count"] = pd.Series(
    [len(card_id_to_new_merchants.get(c, set())) for c in train_cleaned["card_id"]],
    index=train_cleaned.index)


# In[ ]:


train_cleaned_in_new = train_cleaned[train_cleaned["card_in_new"] == True]
train_cleaned_in_new_and_zero_score = train_cleaned_in_new[train_cleaned_in_new["target"] == 0]
train_cleaned_in_new_and_zero_score["new_merchant_count"].describe()


# In[ ]:


plt.figure(figsize=(12, 5))
plt.hist(train_cleaned_in_new_and_zero_score["new_merchant_count"].values, bins=200)
plt.title('Histogram new merchant counts for train_cleaned_in_new_and_zero_score')
plt.xlabel('Count')
plt.ylabel('New merchant count')
plt.show()


# In[ ]:


train_cleaned_in_new_and_nonzero_score = train_cleaned_in_new[train_cleaned_in_new["target"] != 0]
print(train_cleaned_in_new_and_nonzero_score["new_merchant_count"].describe())

plt.figure(figsize=(12, 5))
plt.hist(train_cleaned_in_new_and_nonzero_score["new_merchant_count"].values, bins=200)
plt.title('Histogram new merchant counts for train_cleaned_in_new_and_nonzero_score')
plt.xlabel('Count')
plt.ylabel('New merchant count')
plt.show()


# Well, my guess for the underlying rule being "IF newMerchantCount < k THEN target = 0" is clearly wrong, because both groups have cards with new merchant counts being 1. Still, we can clearly see higher new merchant count being a huge factor contributing to nonzero loyalty score. So my guess wasn't completely irrelevant.
# 
# Finally let's also try to see the relationship between new merchant count and loyalty score 

# In[ ]:


plt.scatter(
    train_cleaned_in_new["new_merchant_count"].values,
    train_cleaned_in_new["target"].values
)


# In[ ]:


max_merchant_bucket = train_cleaned_in_new["new_merchant_count"].max()
expectations = [
    train_cleaned_in_new[train_cleaned_in_new["new_merchant_count"] == i]["target"].mean()
     for i in range(max_merchant_bucket + 1)
]
expectations = [
    -5 if np.isnan(v) else v
    for v in expectations
]
# plt.plot(
#     range(max_merchant_bucket + 1),
#     [train_cleaned_in_new[train_cleaned_in_new["new_merchant_count"] == i].mean()
#      for i in range(max_merchant_bucket + 1)]
# )


# In[ ]:


plt.scatter(
    range(max_merchant_bucket + 1),
    expectations
)


# Very interesting, it looks like a normal distribution but it's tricky to interpret. It seems to indicate that the loyalty score's variance will reduce as the number of merchants increases (which makes a lot of sense), but the expectation of the score doesn't necessarily increase (even slightly negatively correlated from the way it seems). 
# 
# One explanation could be that some people maybe using the payment service a lot, and they discover a lot of underlying problems (but they can't stop using it for some reason, maybe it's too hard to swtich, maybe there's no other alternative, maybe they are lazy, whatever) which cause the loyalty drop. Nonetheless, aggregation of the new/hist transactions separately and trying to find some difference between the two seem to be a good direction when doing feature engineering.
