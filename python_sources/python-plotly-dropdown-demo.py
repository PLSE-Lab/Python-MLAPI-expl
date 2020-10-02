#!/usr/bin/env python
# coding: utf-8

# It's surprisingly easy to use a dropdown in Plotly to change a graph, helping make a more interactive dashboard in a Python notebook. Here's a quick demo on COVID19 submission data (which has a more complete [dasbhoard here](https://www.kaggle.com/benhamner/covid19-forecasting-submissions-dashboard))

# In[ ]:


import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go


# I'm hiding the code below, which creates the dataframe I'm using to plot and isn't relevant for the demo

# In[ ]:


submission_path = "../input/covid19-global-forecasting-submissions/week_1"
submission_files = os.listdir(submission_path)

submissions_list = []

for f in submission_files:
    submission = pd.read_csv(os.path.join(submission_path, f))
    submission.insert(0, "SubmissionId", int(f[:-4]))
    submissions_list.append(submission)

submissions = pd.concat(submissions_list, ignore_index=True, sort=False)
# submissions

# Read in solution/test files
wk1_solution = pd.read_csv("../input/covid-19-forecasting-ongoing-data-updates/wk1_solution.csv")
wk1_test = pd.read_csv("../input/covid19-forecasting-week-one-launch-data/test.csv")
wk1_test = wk1_test.rename(columns={"Province/State": "Province_State", "Country/Region": "Country_Region"})

# Merge submissions to solution and test files
wk1 = submissions.merge(wk1_test, on="ForecastId", how="left")
wk1 = wk1.merge(wk1_solution, on="ForecastId", how="left", suffixes=("", "Actual"))

# Add ranks and scores to the submissions
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 
wk1.loc[:,"FatalitiesSLE"] = (np.log(1+wk1["Fatalities"])-np.log(1+wk1["FatalitiesActual"]))**2
wk1.loc[:,"ConfirmedCasesSLE"] = (np.log(1+wk1["ConfirmedCases"])-np.log(1+wk1["ConfirmedCasesActual"]))**2
wk1_scores = wk1[wk1["Usage"]=="Private"][["SubmissionId", "FatalitiesSLE", "ConfirmedCasesSLE"]].groupby("SubmissionId").mean().reset_index()
wk1_scores.loc[:, "FatalatiesRMSLE"] = np.sqrt(wk1_scores["FatalitiesSLE"])
wk1_scores.loc[:, "ConfirmedCasesRMSLE"] = np.sqrt(wk1_scores["ConfirmedCasesSLE"])
wk1_scores.loc[:, "OverallRMSLE"] = (wk1_scores["FatalatiesRMSLE"]+wk1_scores["ConfirmedCasesRMSLE"])/2.0
wk1_scores = wk1_scores.sort_values(by="OverallRMSLE")
wk1 = wk1.merge(wk1_scores[["SubmissionId", "FatalatiesRMSLE", "ConfirmedCasesRMSLE", "OverallRMSLE"]], on="SubmissionId")
wk1_ranks = wk1[["SubmissionId", "OverallRMSLE"]].drop_duplicates().sort_values(by="OverallRMSLE")
wk1_ranks["Rank"] = list(range(1, wk1_ranks.shape[0]+1))
wk1 = wk1.merge(wk1_ranks[["SubmissionId", "Rank"]], on="SubmissionId", how="inner")


# The code below creates our figure! It adds two traces to the figure per state (a boxplot and a line), and sets up the dropdown menu. The menu itself works by showing only the traces corresponding to the selected state (via the "visible" attribute), and hiding all the remaining traces.

# In[ ]:


states = sorted(set(wk1[wk1["Country_Region"]=="US"]["Province_State"]))

fig=go.Figure()

region_plot_names = []
buttons=[]

default_state = "California"

for region_name in states:
    region = wk1[(wk1["Province_State"]==region_name) & (wk1["Usage"]!="Public") & (wk1["Rank"]<=25)]
    region_actual = region[(region["SubmissionId"]==region["SubmissionId"][region.index[1]]) & (region["Usage"]=="Private")]
    
    # We have two traces we're plotting per state: a boxplot of the submission quartiles, and a line with the current data to-date
    fig.add_trace(go.Box(x=region["Date"], y=region["Fatalities"], visible=(region_name==default_state)))
    fig.add_trace(go.Scatter(x=region_actual["Date"], y=region_actual["FatalitiesActual"], line={}, visible=(region_name==default_state)))
  
    region_plot_names.extend([region_name]*2)
    
for region_name in states:
    buttons.append(dict(method='update',
                        label=region_name,
                        args = [{'visible': [region_name==r for r in region_plot_names]}]))

# Add dropdown menus to the figure
fig.update_layout(showlegend=False, updatemenus=[{"buttons": buttons, "direction": "down", "active": states.index(default_state), "showactive": True, "x": 0.5, "y": 1.15}])
fig.show()


# In[ ]:




