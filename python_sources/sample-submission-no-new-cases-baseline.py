#!/usr/bin/env python
# coding: utf-8

# # Sample Submission: No New Cases Baseline
# 
# This is for the COVID-19 forecasting challenge, Local-CA Week 1 edition.
# 
# This will walk through making a sample submission based on the dataset.
# 
# It makes the obviously flawed assumption that there's no subsequent spread of COVID-19 or cases/fatalities reported based on it.
# 
# It will apply the most recent data point (prior to each of the public/private evaluation windows), and propagate those forward through time.

# In[ ]:


# Read in the data

import pandas as pd

train = pd.read_csv("../input/covid19-local-us-ca-forecasting-challenge-week-1/ca_train.csv")
test  = pd.read_csv("../input/covid19-local-us-ca-forecasting-challenge-week-1/ca_test.csv")


# # Public Leaderboard Predictions
# 
# In order to maintain a meaningful public leaderboard, we should be only using data up until 2020-03-11.

# In[ ]:


public_leaderboard_start_date = "2020-03-12"
last_public_leaderboard_train_date = "2020-03-11"
public_leaderboard_end_date  = "2020-03-26"

submission = test[["ForecastId"]]
submission.insert(1, "ConfirmedCases", 0)
submission.insert(2, "Fatalities", 0)

submission.loc[test["Date"]<=public_leaderboard_end_date, "ConfirmedCases"] = train[train["Date"]==last_public_leaderboard_train_date]["ConfirmedCases"].values[0]
submission.loc[test["Date"]<=public_leaderboard_end_date, "Fatalities"] = train[train["Date"]==last_public_leaderboard_train_date]["Fatalities"].values[0]
submission


# # Final Evaluation Prediction
# 
# For this, we'll use the most recent data point to predict forward into the future period (corresponding to after submissions close for this competition).

# In[ ]:


last_train_date = max(train["Date"])

submission.loc[test["Date"]>public_leaderboard_end_date, "ConfirmedCases"] = train[train["Date"]==last_train_date]["ConfirmedCases"].values[0]
submission.loc[test["Date"]>public_leaderboard_end_date, "Fatalities"] = train[train["Date"]==last_train_date]["Fatalities"].values[0]
submission


# In[ ]:


submission.to_csv("submission.csv", index=False)

