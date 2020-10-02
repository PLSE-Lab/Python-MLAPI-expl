#!/usr/bin/env python
# coding: utf-8

# # Sample Submission: No New Cases Baseline
# 
# This is for the COVID-19 forecasting challenge, Global Week 1 edition.
# 
# This will walk through making a sample submission based on the dataset.
# 
# It makes the obviously flawed assumption that there's no subsequent spread of COVID-19 or cases/fatalities reported based on it.
# 
# It will apply the most recent data point (prior to each of the public/private evaluation windows), and propagate those forward through time.

# In[ ]:


# Read in the data

import numpy as np
import pandas as pd

train = pd.read_csv("../input/covid19-global-forecasting-challenge-week-1/train.csv")
test  = pd.read_csv("../input/covid19-global-forecasting-challenge-week-1/test.csv")


# # Setup Submission File

# In[ ]:


submission = test[["ForecastId"]]
submission.insert(1, "ConfirmedCases", 0)
submission.insert(2, "Fatalities", 0)


# # Pull List of Locations We're Predicting

# In[ ]:


locations = list(set([(test.loc[i, "Province/State"], test.loc[i, "Country/Region"]) for i in test.index]))
locations


# # Public Leaderboard Predictions
# 
# We should be only using data up until 2020-03-11 for the public leaderboard.

# In[ ]:


public_leaderboard_start_date = "2020-03-12"
last_public_leaderboard_train_date = "2020-03-11"
public_leaderboard_end_date  = "2020-03-26"

for loc in locations:
    if type(loc[0]) is float and np.isnan(loc[0]):
        confirmed=train[((train["Country/Region"]==loc[1]) & (train["Date"]==last_public_leaderboard_train_date))]["ConfirmedCases"].values[0]
        deaths=train[((train["Country/Region"]==loc[1]) & (train["Date"]==last_public_leaderboard_train_date))]["Fatalities"].values[0]
        submission.loc[((test["Country/Region"]==loc[1]) & (test["Date"]<=public_leaderboard_end_date)), "ConfirmedCases"] = confirmed
        submission.loc[((test["Country/Region"]==loc[1]) & (test["Date"]<=public_leaderboard_end_date)), "Fatalities"] = deaths
    else:
        confirmed=train[((train["Province/State"]==loc[0]) & (train["Country/Region"]==loc[1]) & (train["Date"]==last_public_leaderboard_train_date))]["ConfirmedCases"].values[0]
        deaths=train[((train["Province/State"]==loc[0]) & (train["Country/Region"]==loc[1]) & (train["Date"]==last_public_leaderboard_train_date))]["Fatalities"].values[0]
        submission.loc[((test["Country/Region"]==loc[1]) & (test["Date"]<=public_leaderboard_end_date)), "ConfirmedCases"] = confirmed
        submission.loc[((test["Country/Region"]==loc[1]) & (test["Date"]<=public_leaderboard_end_date)), "Fatalities"] = deaths

submission


# # Final Evaluation Prediction
# 
# For this, we'll use the most recent data point to predict forward into the future period (corresponding to after submissions close for this competition).

# In[ ]:


last_train_date = max(train["Date"])

for loc in locations:
    if type(loc[0]) is float and np.isnan(loc[0]):
        confirmed=train[((train["Country/Region"]==loc[1]) & (train["Date"]==last_train_date))]["ConfirmedCases"].values[0]
        deaths=train[((train["Country/Region"]==loc[1]) & (train["Date"]==last_train_date))]["Fatalities"].values[0]
        submission.loc[((test["Country/Region"]==loc[1]) & (test["Date"]>public_leaderboard_end_date)), "ConfirmedCases"] = confirmed
        submission.loc[((test["Country/Region"]==loc[1]) & (test["Date"]>public_leaderboard_end_date)), "Fatalities"] = deaths
    else:
        confirmed=train[((train["Province/State"]==loc[0]) & (train["Country/Region"]==loc[1]) & (train["Date"]==last_train_date))]["ConfirmedCases"].values[0]
        deaths=train[((train["Province/State"]==loc[0]) & (train["Country/Region"]==loc[1]) & (train["Date"]==last_train_date))]["Fatalities"].values[0]
        submission.loc[((test["Country/Region"]==loc[1]) & (test["Date"]>public_leaderboard_end_date)), "ConfirmedCases"] = confirmed
        submission.loc[((test["Country/Region"]==loc[1]) & (test["Date"]>public_leaderboard_end_date)), "Fatalities"] = deaths

submission


# In[ ]:


submission.to_csv("submission.csv", index=False)

