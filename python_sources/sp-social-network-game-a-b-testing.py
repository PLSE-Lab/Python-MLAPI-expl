#!/usr/bin/env python
# coding: utf-8

# clicks.csv has the following columns:
# 
# user_id: a unique id for each visitor to the FarmBurg site
# 
# ab_test_group: either A, B, or C depending on which group the visitor was assigned to
# 
# click_day: only filled in if the user clicked on a link to purchase

# # Determine whether or not there is a significant difference in the percent of users who purchased the upgrade package among groups A, B, and C.

# In[4]:


import pandas as pd

df = pd.read_csv('../input/clicks.csv')
df.head(20)


# In[5]:


df["is_purchase"] = df["click_day"].apply(lambda x: "Purchase" if pd.notnull(x) else "No Purchase")
purchase_counts = df.groupby(["group", "is_purchase"]).user_id.count().reset_index()
print(purchase_counts)


# ## Chi Square Test

# In[15]:


from scipy.stats import chi2_contingency

# contingency = [[A_purchases, A_not_purchases],
#                [B_purchases, B_not_purchases],
#                [C_purchases, C_not_purchases]]

contingency = [[316, 1350],
               [183, 1483],
               [83, 1583]]

pvalue = chi2_contingency(contingency)[1]

f = lambda x: True if x <= 0.05 else False

is_significant = f(pvalue)

print(is_significant)


# # Does each price level (\$0.99, \$1.99, \$4.99) allow us to make enough money so that we can exceed the target sales goal?

# In[6]:


# calculate the percent of visitors who would need to purchase the upgrade package at each price point ($0.99, $1.99, $4.99) in order to generate sales target of $1,000 per week.

num_visits = len(df)

p_clicks_099 = (1000 / 0.99) / num_visits
p_clicks_199 = (1000 / 1.99) / num_visits
p_clicks_499 = (1000 / 4.99) / num_visits


# # See if the percent of Group A that purchased an upgrade package is significantly greater than p_clicks_099 (the percent of visitors who need to buy an upgrade package at \$0.99 in order to make our target of \$1,000).

# ## We should use a binomial test on each group to see if the observed purchase rate is significantly greater than what we need in order to generate at least \$1,000 per week.
# 
# ## What price should we charge for the upgrade package? 

# In[13]:


from scipy.stats import binom_test

x = 316
n = 1350 + 316
p = p_clicks_099

pvalueA = binom_test(x, n, p)

x = 183
n = 1483 + 183
p = p_clicks_199

pvalueB = binom_test(x, n, p)

x = 83
n = 1583 + 83
p = p_clicks_499

pvalueC = binom_test(x, n, p)

print(pvalueA, pvalueB, pvalueC)
print("Intuitively we may want to choose $0.99 as it is likely to be accepted by users.\nSurprisingly, $4.99 should be our choice.")

