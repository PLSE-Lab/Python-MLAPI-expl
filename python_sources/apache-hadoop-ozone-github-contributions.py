#!/usr/bin/env python
# coding: utf-8

# In[41]:


import pandas as pd
import numpy as np
import os.path
from os import path

pd.set_option('display.max_rows', 500)
if path.exists("/kaggle/input/apache-hadoop-ozone-contributions"):
   os.chdir("/kaggle/input/apache-hadoop-ozone-contributions")


# In[42]:


participants = pd.read_csv("github/participant.csv")


prs = pd.read_csv("github/prs.csv")
prs.createdAt = pd.to_datetime(prs.createdAt)
prs.updatedAt = pd.to_datetime(prs.updatedAt)
prs.closedAt = pd.to_datetime(prs.closedAt)

comments = pd.read_csv("github/comments.csv")
comments.createdAt = pd.to_datetime(comments.createdAt)

roles = comments[["author","role"]].drop_duplicates()
roles = roles.rename(columns={"author":"login"})
roles = roles.set_index("login")

reviews = pd.read_csv("github/reviews.csv")
reviews.updated = pd.to_datetime(reviews.updated)
reviews["updated_month"] = reviews.updated.dt.to_period("M")

allprs = prs
# reviews.head()
# comments.head()
# roles.head()
# prs.head()
# participants.head()


# In[43]:


# #Enable this section for date filter
# startDate = "2020-06-01"
# endDate = "2020-06-30"

# prs = prs[prs["createdAt"] > startDate]
# prs = prs[prs["createdAt"] < endDate]

# comments = comments[comments["createdAt"] > startDate]
# comments = comments[comments["createdAt"] < endDate]

# reviews = reviews[reviews["updated"] > startDate]
# reviews = reviews[reviews["updated"] > endDate]


# In[44]:


#Caclculate the combination of commented OR reviewer issue by a specific user

contribution_comments = comments[["number","author"]].drop_duplicates().rename(columns={"author":"login"})
contribution_comments.insert(2,"type","comment")

contribution_reviews = reviews[["number","login"]].drop_duplicates()
contribution_reviews["type"] = "review"

commented_or_reviewed_pr = pd.concat([contribution_comments,contribution_reviews])[["number","login"]].drop_duplicates()

commented_or_reviewed_pr = commented_or_reviewed_pr.set_index("number").join(allprs[["number","author"]].set_index("number"))
commented_or_reviewed_pr = commented_or_reviewed_pr[commented_or_reviewed_pr["author"] != commented_or_reviewed_pr["login"]][["login"]].reset_index()
commented_or_reviewed_pr = commented_or_reviewed_pr.groupby("login").count().rename(columns={"number":"commented_or_reviewed_pr"})


# # Github Contributions per user
# 
# Note: only events from apache/hadoop-ozone repository are included. Earlier PRs/comments are not here.

# In[45]:


p = prs[["number","author"]].drop_duplicates()
pr_per_login = p.groupby("author").count().sort_values("number", ascending=False)
pr_per_login = pr_per_login.rename(columns={"author":"login"})
pr_per_login = pr_per_login.reset_index().rename(columns={"author":"login","number":"created_pr"}).set_index("login")
pr_per_login

#Comments only on PRs where the author is different
comments_filtered = comments.set_index("number").join(prs[["number","author"]].set_index("number"),rsuffix="pr")
comments_filtered = comments_filtered[comments_filtered["author"] != comments_filtered["authorpr"]]
commented_pr = comments_filtered.reset_index()[["author","number","createdAt"]].groupby(["author","number"]).count().reset_index()[["number","author"]].groupby("author").count()

commented_prs = comments_filtered.reset_index()[["author","number","createdAt"]]    .groupby(["author","number"]).count()    .reset_index()[["number","author"]]    .groupby("author").count()    .reset_index().rename(columns={"author":"login","number":"prs_commented"})    .set_index("login")

comments_all_per_login = comments[["author","createdAt","role"]].groupby(["author","role"]).count().sort_values("createdAt",ascending=False)
comments_all_per_login = comments_all_per_login.reset_index().rename(columns={"author":"login","createdAt":"comments_all"}).set_index("login")

comments_other_per_login = comments_filtered[["author","createdAt"]].groupby(["author"]).count().sort_values("createdAt",ascending=False)
comments_other_per_login = comments_other_per_login.reset_index().rename(columns={"author":"login","createdAt":"comments_for_others"}).set_index("login")


p = reviews[["number","login"]].drop_duplicates()
review_per_login = p.groupby("login").count()
review_per_login = review_per_login.rename(columns={"author":"login", "number":"reviewed_pr"})

activity = comments_all_per_login.rename(columns={"createdAt":"comments"}).reset_index()
activity = activity.rename(columns={"author":"login"}).set_index("login")
activity = activity   .join(comments_other_per_login)   .join(review_per_login)   .join(pr_per_login)   .join(commented_prs)   .join(commented_or_reviewed_pr)   .sort_values("created_pr",ascending=False)

activity["ratio"] = activity["commented_or_reviewed_pr"] / activity["created_pr"]
activity.sort_values("created_pr", ascending=False)


# # Bus factor (number of contributors responsible for the 50% of the prs)

# In[46]:



t = activity.sort_values("created_pr", ascending=False)["created_pr"].cumsum()
t[t<prs.shape[0]*0.5]


# ## People with created PRs > reviewed/commented PRS

# In[47]:


a = activity[(activity["commented_or_reviewed_pr"] <= activity["created_pr"]) | activity["commented_or_reviewed_pr"].isnull()].sort_values("created_pr",ascending=False)
a[~a["created_pr"].isnull()]
a.sort_values(["role","commented_or_reviewed_pr"])


# # Number of individual contributors per month
# 
# Number of different Github users who either created PR, commented PR, added review to a PR
# 
# Note: only events from apache/hadoop-ozone repository are included. Earlier PRs/comments are not here.

# In[50]:


pr_author_monthly = prs[["number","author","createdAt"]].groupby([prs.createdAt.dt.to_period("M"),"author"]).count().drop(["createdAt","number"],axis=1).reset_index().rename(columns={"createdAt":"month"})

coment_author_monthly = comments[["author","createdAt"]].groupby([comments.createdAt.dt.to_period("M"),"author"]).count().drop(["createdAt"],axis=1).reset_index().rename(columns={"createdAt":"month"})
reviews_author_monthly = reviews[["login","updated"]].groupby([reviews.updated.dt.to_period("M"),"login"]).count().drop(["updated"],axis=1).reset_index().rename(columns={"updated":"month","login":"author"})
coment_author_monthly
reviews_author_monthly
pd.concat([pr_author_monthly,coment_author_monthly,reviews_author_monthly]).drop_duplicates().groupby("month").count()


# # Number of PRs closed/created per month

# In[49]:


closed = prs[["number","title","closedAt"]].groupby(prs.closedAt.dt.to_period("M")).count().drop(["title","closedAt"],axis=1).reset_index().rename({"closedAt":"month","number":"closed"},axis=1).set_index("month")
created = prs[["number","title","createdAt"]].groupby(prs.createdAt.dt.to_period("M")).count().drop(["title","createdAt"],axis=1).reset_index().rename({"createdAt":"month","number":"created"},axis=1).set_index("month")
monthly_pr = created.join(closed)
monthly_pr["increment"] = monthly_pr["created"] - monthly_pr["closed"]
monthly_pr


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




