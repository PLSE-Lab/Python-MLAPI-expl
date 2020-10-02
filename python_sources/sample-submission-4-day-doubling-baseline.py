#!/usr/bin/env python
# coding: utf-8

# # Sample Submission: 4-Day Doubling Baseline
# 
# This is for the COVID-19 forecasting challenge, Local-CA Week 1 edition.
# 
# This will walk through making a sample submission based on the dataset.
# 
# It makes the obviously flawed assumption that COVID-19 cases/fatalities double indefinitely every 4 days from the most recent data points.
# 
# It will apply the most recent data point (prior to each of the public/private evaluation windows), and propagate those forward through time.

# In[ ]:


# Read in the data

import numpy as np
import pandas as pd

train = pd.read_csv("../input/covid19-local-us-ca-forecasting-challenge-week-1/ca_train.csv")
test  = pd.read_csv("../input/covid19-local-us-ca-forecasting-challenge-week-1/ca_test.csv")


# # Public Leaderboard Predictions
# 
# We should be only using data up until 2020-03-11 for the public leaderboard.

# In[ ]:


public_leaderboard_start_date = "2020-03-12"
last_public_leaderboard_train_date = "2020-03-11"
public_leaderboard_end_date  = "2020-03-26"

submission = test[["ForecastId"]]
submission.insert(1, "ConfirmedCases", 0)
submission.insert(2, "Fatalities", 0)

cases  = train[train["Date"]==last_public_leaderboard_train_date]["ConfirmedCases"].values[0] * (2**(1/4))
deaths = train[train["Date"]==last_public_leaderboard_train_date]["Fatalities"].values[0] * (2**(1/4))

for i in list(submission.index)[:15]:
    cases = cases * (2**(1/4))
    deaths = deaths * (2**(1/4))
    submission.loc[i, "ConfirmedCases"] = cases
    submission.loc[i, "Fatalities"] = deaths

submission


# # Final Evaluation Prediction
# 
# For this, we'll use the most recent data point to predict forward into the future period (corresponding to after submissions close for this competition).

# In[ ]:


last_train_date = max(train["Date"])

cases  = train[train["Date"]==last_public_leaderboard_train_date]["ConfirmedCases"].values[0]
deaths = train[train["Date"]==last_public_leaderboard_train_date]["Fatalities"].values[0]

for i in submission.index:
    if test.loc[i, "Date"]>last_train_date: # Apply growth rule
        cases  = cases  * (2**(1/4))
        deaths = deaths * (2**(1/4))
    if test.loc[i, "Date"]>public_leaderboard_end_date: # Update submission value
        submission.loc[i, "ConfirmedCases"] = cases
        submission.loc[i, "Fatalities"] = deaths

submission


# In[ ]:


submission.to_csv("submission.csv", index=False)

