#!/usr/bin/env python
# coding: utf-8

# # A Minimalist Baseline Submission
# Stripped down to essentials, these 19 lines are all you need to break into the top 50% on the leaderboard. I'll start by presenting the bare code for you to figure out on your own, and then explain it in detail down below.

# In[ ]:


import pandas as pd
from sklearn.linear_model import LogisticRegression

Xy_labelled = pd.read_csv("../input/cat-in-the-dat/train.csv", index_col="id")
X_labelled = Xy_labelled.drop(columns=["target"])
y_labelled = Xy_labelled["target"]
X_test = pd.read_csv("../input/cat-in-the-dat/test.csv", index_col="id")

target_mean = y_labelled.mean()
for col in X_labelled.columns:
    stats = Xy_labelled.groupby(col)["target"].agg(["sum", "count"])
    likelihoods = ((stats["sum"] + (800 * target_mean)) / (stats["count"] + 800)).to_dict()
    X_labelled[col] = X_labelled[col].map(likelihoods)
    X_test[col] = X_test[col].map(lambda v: likelihoods.get(v, target_mean))

model = LogisticRegression(solver="saga", n_jobs=-1, random_state=0).fit(X_labelled, y_labelled)
proba_predictions = model.predict_proba(X_test)[:, 1]
output = pd.DataFrame({"id": X_test.index, "target": proba_predictions})
output.to_csv("submission.csv", index=False)


# ## The code, explained
# ```
# import pandas as pd
# from sklearn.linear_model import LogisticRegression
# ```
# Pandas provides all of the methods we need to load and pre-process the data, and `LogisticRegression` will handle all the machine-learning we need.
# ```
# Xy_labelled = pd.read_csv("../input/cat-in-the-dat/train.csv", index_col="id")
# X_labelled = Xy_labelled.drop(columns=["target"])
# y_labelled = Xy_labelled["target"]
# X_test = pd.read_csv("../input/cat-in-the-dat/test.csv", index_col="id")
# ```
# Here we read in the data and separate out the target values from the features. There are no missing values for this particular data set, so no imputation is required.
# ```
# target_mean = y_labelled.mean()
# for col in X_labelled.columns:
#     stats = Xy_labelled.groupby(col)["target"].agg(["sum", "count"])
#     likelihoods = ((stats["sum"] + (800 * target_mean)) / (stats["count"] + 800)).to_dict()
#     X_labelled[col] = X_labelled[col].map(likelihoods)
#     X_test[col] = X_test[col].map(lambda v: likelihoods.get(v, target_mean))
# ```
# This is a basic implementation of "target encoding". We use the constant `800` as a smoothing factor, providing a weighted average between the group target likelihood and the global target likelihood. (We chose this particular value based upon out-of-band testing. Any value between 500 and 1000 will work just fine.)
# 
# Note that because there are values in the test set that are not in the training data, we must supply a default value for the test set, and the global target likelihood is the obvious choice.
# ```
# model = LogisticRegression(solver="saga", n_jobs=-1, random_state=0).fit(X_labelled, y_labelled)
# proba_predictions = model.predict_proba(X_test)[:, 1]
# ```
# We apply LogisticRegression without any parameter tuning. Other solvers would likely perform just as well, but SAGA is the newest and documented to be fast for large datasets. Because we are evaluated on "area under the ROC curve", we need to submit "proba" probabilities rather than target values.
# ```
# output = pd.DataFrame({"id": X_test.index, "target": proba_predictions})
# output.to_csv("submission.csv", index=False)
# ```
# Finally, we write out our predictions, submit, and profit from a barely-above-average LB score.
# 
