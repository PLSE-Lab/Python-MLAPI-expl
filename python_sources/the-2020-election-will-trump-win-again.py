#!/usr/bin/env python
# coding: utf-8

# # General Election Poll Analysis

# In the spirit of the election season, I will be analyzing US presidential election polling data from 2016 and 2020.  I will be answering many questions, including who might win in 2020.  
# 
# This notebook will be updated until the conclusion of the 2020 election.

# ## Importing and Cleaning the Data

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random as r
from datetime import datetime
from scipy import stats


# Three unique datasets will be utilized in this analysis:
# 
# * 2016 polling data (from Kaggle)
# * 2020 polling data (from FiveThirtyEight)
# * 2016 Election Results by State and County (from Data World)

# ### 2016 Polling Data (Trump vs. Clinton)

# In[ ]:


data_2016 = pd.read_csv("../input/2016-election-polls/presidential_polls.csv")
data_2016 = data_2016[["startdate", "enddate", "state", "pollster", "grade", "samplesize", "population", "adjpoll_clinton", "adjpoll_trump"]]
trump_clinton = data_2016.rename(columns = {"startdate": "start_date", "enddate": "end_date", "grade":"fte_grade", "samplesize":"sample_size", "adjpoll_clinton":"Clinton", "adjpoll_trump":"Trump"})
trump_clinton["start_date"] = pd.to_datetime(trump_clinton["start_date"])
trump_clinton["end_date"] = pd.to_datetime(trump_clinton["end_date"])
trump_clinton = trump_clinton.sort_values(by = ["end_date", "start_date"]) #Arranging the polls from most to least recent
trump_clinton["dem_lead"] = trump_clinton["Clinton"] - trump_clinton["Trump"] #lead of the democratic candidate (negative if they are losing)


# In[ ]:


trump_clinton.head()


# After renaming some columns, this data is already in the format we want.  Only the important columns are extracted.

# In[ ]:


trump_clinton.info()


# The missing values in the "grade" column represent polls with an unknown accuracy.

# ### 2020 Polling Data

# In[ ]:


data_2020 = pd.read_csv("../input/2020-general-election-polls/president_polls.csv")
data_2020 = data_2020[["poll_id","start_date", "end_date", "state", "pollster", "fte_grade", "sample_size", "population", "answer", "pct"]]


# In[ ]:


data_2020.head()


# In[ ]:


data_2020.info()


# Again, some missing values exist in the "grade" column.  The missing values in the "state" column represent national polls:

# In[ ]:


data_2020["state"] = data_2020.state.fillna("U.S.")


# We will have to clean this data by spreading the candidates into their own separate columns, and will also remove irrelevant candidates.  First, it will be necessary to separate different individual matchups involving Trump into different data tables.

# In[ ]:


def trump_opponent(data_2020, opp):
    trump_vs = data_2020[(data_2020["answer"] == opp) | (data_2020["answer"] == "Trump")]
    trump_vs = trump_vs.pivot_table(values = "pct", index = ["poll_id", "start_date", "end_date", "state", "pollster", "fte_grade", "sample_size", "population"], columns = "answer")
    trump_vs = trump_vs.dropna(axis = 0, how = "any") #Drops the Trump polls against any opponent that isn't our opp parameter
    trump_vs = trump_vs.reset_index().drop(columns = ["poll_id"])
    trump_vs["start_date"] = pd.to_datetime(trump_vs["start_date"])
    trump_vs["end_date"] = pd.to_datetime(trump_vs["end_date"]) 
    trump_vs["dem_lead"] = trump_vs[opp] - trump_vs["Trump"] 
    trump_vs = trump_vs.sort_values(by = ["end_date", "start_date"]) #Arranging the polls from most to least recent
    return trump_vs


# #### Trump vs. Biden

# In[ ]:


trump_biden = trump_opponent(data_2020, "Biden")

trump_biden.head()


# #### Trump vs. Sanders

# In[ ]:


trump_sanders = trump_opponent(data_2020, "Sanders")

trump_sanders.head()


# #### Trump vs. Warren

# In[ ]:


trump_warren = trump_opponent(data_2020, "Warren")

trump_warren.head()


# #### Trump vs. Buttigieg

# In[ ]:


trump_buttigieg = trump_opponent(data_2020, "Buttigieg")

trump_buttigieg.head()


# Now, we have 2020 polling data in the exact same format as our 2016 polling data.

# ### 2016 Election Results

# Finally, let's take a look at the results of the 2016 election.  We will get the percentage of Trump and Clinton voters for each state (not including third-party candidates), as well as manually add a row displaying the national results.

# In[ ]:


results_2016 = pd.read_csv("../input/2020-general-election-polls/nytimes_presidential_elections_2016_results_county.csv")
results_2016 = results_2016.groupby("State").sum()[["Clinton", "Trump"]]
results_2016.loc["U.S."] = [65853514, 62984828] #Adding a row for the national result
results_2016["Clinton_pct"] = 100 * results_2016["Clinton"] / (results_2016["Clinton"] + results_2016["Trump"])
results_2016["Trump_pct"] = 100 * results_2016["Trump"] / (results_2016["Clinton"] + results_2016["Trump"]) #percentages
results_2016["dem_lead"] = results_2016["Clinton_pct"] - results_2016["Trump_pct"]
results_2016["index"] = list(range(0,50))
results_2016["state"] = results_2016.index
results_2016 = results_2016.set_index("index")


# In[ ]:


results_2016.head()


# In[ ]:


results_2016.info()


# Note that there may be slight inaccuracies in comparison to the actual results because we do not have data for independant candidates.  Therefore, the Trump and Clinton percentages add up to 100% which is slightly misleading.

# ## What Happened in 2016? (The Polling Averages)

# This section will analyze both national and battleground state polls in 2016 by comparing them to the actual results to determine how we should treat the 2020 polls this far into the election cycle.  We will compare the averages of all polls, historically reliable polls only, and polls of likely voters for each state.

# In[ ]:


def trump_vs_clinton(trump_clinton, state, results_2016, reliable = False, likely_voters = False):
    
    #getting polls for the specified state / U.S. and filtering if necessary
    match_up = trump_clinton
    match_up = match_up[match_up["state"] == state]
    
    if reliable == True:
        match_up = match_up[match_up["fte_grade"].isin(["A+", "A", "A-"])]
    if likely_voters == True:
        match_up = match_up[match_up["population"] == "lv"]
    
    #Accounting for repeated polls which have the same end date
    
    match_up = match_up.groupby(["end_date", "pollster", "fte_grade", "population"]).mean().reset_index()
    match_up.index = match_up["end_date"]
    
    #A rolling average of democrat lead/deficit in the past 14 days
    
    if state == "U.S.":
        match_up["average_lead"] = match_up["dem_lead"].rolling("14D", min_periods = 0).mean()
    else:
        match_up["average_lead"] = match_up["dem_lead"].rolling("30D", min_periods = 0).mean()
    
    #Plotting the time series
    
    polls_vs_final =  [match_up.iloc[-1]["average_lead"], results_2016[results_2016["state"] == state].iloc[0]["dem_lead"]]
    polls_df = pd.DataFrame(polls_vs_final)
    polls_df[1] = ["Final/Current Polling Average", "Actual Results"]
    
    plt.subplots(figsize = (9,6))
    plt.subplot(1,2,1)
    plt.plot(match_up["end_date"], match_up["average_lead"])
    plt.xlabel("Date")
    plt.xticks(rotation = 90)
    plt.ylabel("Lead or Deficit vs. Trump (%)")
    plt.title(f"Trump vs. Clinton in {state}")  
    plt.subplot(1,2,2)
    plt.bar(polls_df[1], polls_df[0])
    plt.xticks(rotation = 90)
    plt.title("Poll Accuracy Chart")
    plt.show()
    
    if reliable == True:
        return f"Percentage Points of Trump Underestimation in {state} From Historically Reliable Polls: {round(polls_vs_final[0] - polls_vs_final[1], 2)}%"
    if likely_voters == True:
        return f"Percentage Points of Trump Underestimation in {state} From Polls of Likely Voters: {round(polls_vs_final[0] - polls_vs_final[1], 2)}%"
    return f"Percentage Points of Trump Underestimation in {state} From All Polls: {round(polls_vs_final[0] - polls_vs_final[1], 2)}%"


# ### Nationally

# In[ ]:


trump_vs_clinton(trump_clinton, "U.S.", results_2016)


# In[ ]:


trump_vs_clinton(trump_clinton, "U.S.", results_2016, reliable = True)


# In[ ]:


trump_vs_clinton(trump_clinton, "U.S.", results_2016, likely_voters = True)


# In 2016, the national polling was right on the money, as the average of all polls the last two weeks before election day only underestimated Trump's chances by 0.03%.  We can also see that the time of the election was the time of one of Clinton's weakest leads at just over 2%.  

# Of course, the US presidential election is decided by the electoral college, not the national popular vote, so it would be helpful to look at battleground states (states that may go either way and decide the election) and other states where Trump performed better than expected.

# ### Florida

# In[ ]:


trump_vs_clinton(trump_clinton, "Florida", results_2016)


# In[ ]:


trump_vs_clinton(trump_clinton, "Florida", results_2016, reliable = True)


# In[ ]:


trump_vs_clinton(trump_clinton, "Florida", results_2016, likely_voters = True)


# The polling averages only barely underestimated Trump's chances in Florida, with historically reliable polls being least accurate by a slight margin and polls of likely voters being the most accurate.  Only the historically reliable polls had Clinton winning Florida, though we can see her numbers went down quickly in those polls right before the election.

# ### Pennsylvania

# In[ ]:


trump_vs_clinton(trump_clinton, "Pennsylvania", results_2016)


# In[ ]:


trump_vs_clinton(trump_clinton, "Pennsylvania", results_2016, reliable = True)


# In[ ]:


trump_vs_clinton(trump_clinton, "Pennsylvania", results_2016, likely_voters = True)


# Trump's chances in Pennsylvania were underestimated by huge margins, especially by historically reliable polls, despite Clinton having mostly positive momentum towards the end.

# ### Ohio

# In[ ]:


trump_vs_clinton(trump_clinton, "Ohio", results_2016)


# In[ ]:


trump_vs_clinton(trump_clinton, "Ohio", results_2016, reliable = True)


# In[ ]:


trump_vs_clinton(trump_clinton, "Ohio", results_2016, likely_voters = True)


# Trump was already projected to win Ohio by a small margin, but he ended up winning by over 8%! Again, the historically reliable polls did the worst job of predictions.  We can see, however, that Clinton had a lot of negative momentum towards the end of the Ohio polling.

# ### Michigan

# In[ ]:


trump_vs_clinton(trump_clinton, "Michigan", results_2016)


# In[ ]:


trump_vs_clinton(trump_clinton, "Michigan", results_2016, reliable = True)


# In[ ]:


trump_vs_clinton(trump_clinton, "Michigan", results_2016, likely_voters = True)


# Michigan is yet another state that polls failed to predict accurately, though the historically reliable polls did a better job here than the averages of all polls.

# ### Wisconsin

# In[ ]:


trump_vs_clinton(trump_clinton, "Wisconsin", results_2016)


# In[ ]:


trump_vs_clinton(trump_clinton, "Wisconsin", results_2016, reliable = True)


# In[ ]:


trump_vs_clinton(trump_clinton, "Wisconsin", results_2016, likely_voters = True)


# Wisconsin was one of the most unexpected states that Trump won, though the historically reliable polls did much better here that the average of all polls.  Interestingly, however, Clinton had positive momentum in the historically reliable polls and negative momentum in polls of likely voters, as well as the average of all polls.

# ### Minnesota

# In[ ]:


trump_vs_clinton(trump_clinton, "Minnesota", results_2016)


# In[ ]:


trump_vs_clinton(trump_clinton, "Minnesota", results_2016, reliable = True)


# In[ ]:


trump_vs_clinton(trump_clinton, "Minnesota", results_2016, likely_voters = True)


# Clinton did manage to win Minnesota, but by much closer margins than expected.  Again, the historically reliable polls did better here and there was negative momentum for Clinton across the board.

# Since the national polls estimated the results of the popular vote perfectly, yet polls underestimated Trump's chances in battleground states, there must have been some states which overestimated Trump's chances.  Note that these states don't play a huge factor in the election because they always decisively vote in a certain direction, so **we have a smaller sample size of polls**.  Let's take a look at the polls of the most populated states in the country.

# ### California

# In[ ]:


trump_vs_clinton(trump_clinton, "California", results_2016)


# In[ ]:


trump_vs_clinton(trump_clinton, "California", results_2016, reliable = True)


# In[ ]:


trump_vs_clinton(trump_clinton, "California", results_2016, likely_voters = True)


# Trump's chances were overestimated in this state by the average of all polls, yet interestingly underestimated by historically reliable polls.

# ### Illinois

# In[ ]:


trump_vs_clinton(trump_clinton, "Illinois", results_2016)


# In[ ]:


trump_vs_clinton(trump_clinton, "Illinois", results_2016, reliable = True)


# In[ ]:


trump_vs_clinton(trump_clinton, "Illinois", results_2016, likely_voters = True)


# Similar patterns exist here as the California polls.  Again, note that the sample size of polls is very small.

# ### New York

# In[ ]:


trump_vs_clinton(trump_clinton, "New York", results_2016)


# In[ ]:


trump_vs_clinton(trump_clinton, "New York", results_2016, reliable = True)


# In[ ]:


trump_vs_clinton(trump_clinton, "New York", results_2016, likely_voters = True)


# Trump was overestimated across the board in New York polls.

# ### Texas

# In[ ]:


trump_vs_clinton(trump_clinton, "Texas", results_2016)


# In[ ]:


trump_vs_clinton(trump_clinton, "Texas", results_2016, reliable = True)


# In[ ]:


trump_vs_clinton(trump_clinton, "Texas", results_2016, likely_voters = True)


# Though Trump carried Texas pretty easily, he was overestimated by 7.29% in historically reliable polls! Clinton also seemed to pick up positive momentum towards the end of polling in this state.  As we will later see, Texas could be a closer race in 2020, so this information is important.

# ## What Happened in 2016? (The Best Pollsters)

# Now, which individual pollsters did the best job calling the election nationally and in each battleground state the two weeks leading up to the election? Furthermore, did pollsters who released more polls predict the national and battleground states more accurately?

# In[ ]:


def get_best_pollsters(trump_clinton, state, results_2016):
    
    #Getting polls only from the final two weeks in a given state
    
    final_polls = trump_clinton[(trump_clinton["end_date"] >= "2016-10-17") & (trump_clinton["state"] == state)]
    
    #Getting a variable to represent the true results in each state
    
    final_polls = pd.merge(final_polls, results_2016, on = "state", how = "inner")
    
    #Getting the average results in the state for each pollster and the difference from actual results
    
    by_pollster = final_polls.groupby(["pollster", "dem_lead_y"]).mean().reset_index()[["pollster", "dem_lead_x", "dem_lead_y"]]
    by_pollster["trump_underestimation"] = by_pollster["dem_lead_x"] - by_pollster["dem_lead_y"]
    graph_pollsters = by_pollster.sort_values("trump_underestimation", ascending = False)
    
    #Getting the number of polls from each pollster
    
    num_polls = final_polls.groupby("pollster").size().reset_index()
    
    #Finally, plotting our results
    
    plt.subplots(figsize = (10,10))
    sns.barplot(x = graph_pollsters["trump_underestimation"], y = graph_pollsters["pollster"])
    plt.xlabel("Percentage Point Difference From Actual Result (more positive number = more Trump underestimation)")
    plt.ylabel(None)
    plt.show()
    
    #Table of most to least accurate pollsters in terms of magnitude of inaccuracy
    
    best_pollsters = by_pollster
    best_pollsters["trump_underestimation"] = abs(by_pollster["dem_lead_x"] - by_pollster["dem_lead_y"])
    best_pollsters = pd.merge(best_pollsters, num_polls, on = "pollster") 
    best_pollsters["Number of Polls"] = best_pollsters[0]
    best_pollsters["Pct Pts Inaccuracy"] = best_pollsters["trump_underestimation"]
    best_pollsters["Pollster"] = best_pollsters["pollster"]
    best_pollsters = best_pollsters.sort_values("trump_underestimation")[["Pollster", "Pct Pts Inaccuracy", "Number of Polls"]]
    
    #Linear Regression of polls released the final two weeks vs final inaccuracy
    
    sns.lmplot(data = best_pollsters, x = "Number of Polls", y = "Pct Pts Inaccuracy")
    plt.title(f"Number of Polls Released vs. Inaccuracy in {state}")
    print(stats.linregress(best_pollsters["Number of Polls"], best_pollsters["Pct Pts Inaccuracy"]))


# ### Nationally

# In[ ]:


get_best_pollsters(trump_clinton, "U.S.", results_2016)


# The Times-Picayune/Lucid, which was tied for releasing the most polls in the final two weeks leading up to the election, got an average that predicted the exact popular vote.  
# 
# There existed a moderate correlation between number of polls released the final two weeks before the election and inaccuracy (r = -0.44).  The probability that this occurred by chance is 2.07% (p = 0.0207).

# ### Florida

# In[ ]:


get_best_pollsters(trump_clinton, "Florida", results_2016)


# Rasumussen predicted the Florida results very precisely, and released more polls than any other pollster.  We can also see a balance of Trump underestimation and overestimation, which makes sense due to our findings in Florida from before.  Also, it would be best to never listen to Saint Leo's polling!
# 
# There existed a moderate correlation between number of polls released the final two weeks before the election and inaccuracy (r = -0.31).  The probability that this occurred by chance is 16.5%.

# ### Pennsylvania

# In[ ]:


get_best_pollsters(trump_clinton, "Pennsylvania", results_2016)


# Google Consumer Surveys was the only pollster that overestimated Trump's chances in this state, and only did so by a very small margin.
# 
# There existed a small correlation between number of polls released the final two weeks before the election and inaccuracy (r = -0.26).  The probability that this occurred by chance is 44.7%.

# ### Ohio

# In[ ]:


get_best_pollsters(trump_clinton, "Ohio", results_2016)


# Again, Google Consumer Surveys gave the best prediction for Ohio, though SurveyMonkey was a very close second.
# 
# There existed a small correlation between number of polls released the final two weeks before the election and inaccuracy (r = -0.20).  The probability that this occurred by chance is 66.1%.

# ### Michigan

# In[ ]:


get_best_pollsters(trump_clinton, "Michigan", results_2016)


# Another state where every pollster underestimated Trump's chances, though SurveyMonkey was very close.
# 
# There existed a moderate correlation between number of polls released the final two weeks before the election and inaccuracy (r = -0.42).  The probability that this occurred by chance is 34.5%.

# ### Wisconsin

# In[ ]:


get_best_pollsters(trump_clinton, "Wisconsin", results_2016)


# This time, Google Consumer Surveys was way off and SurveyMonkey was right on the money as they also performed the most polls.
# 
# There existed a moderate correlation between number of polls released the final two weeks before the election and inaccuracy (r = -0.35).  The probability that this occurred by chance is 39.4%.

# ### Minnesota

# In[ ]:


get_best_pollsters(trump_clinton, "Minnesota", results_2016)


# Another good prediction by Google Consumer Surveys.
# 
# There existed a small positive correlation between number of polls released the final two weeks before the election and inaccuracy (r = 0.29).  The probability that this occurred by chance is 63.6%.

# ## Conclusions

# Now, from the polling information above, what did we learn? How might we use this to interpret the 2020 polls if we assume voter and polling behavior follows similar patterns?

# * We should listen to national polls, which were right on the money, a lot more closely than state polls.  Although the electoral college (state by state results) is most important in American elections, there are still less paths to win the presidency the less support a candidate acquires nationally.
# * In every battleground state we looked at, with the exception of Florida, Trump was underestimated in average polling by 4.5-6%, beyond margin of error for the average poll.  There is incredibly small probability that this occurred by chance, so this most likely means that these polls failed to be truly random or they did not take into account some outside factor such as voter enthusiasm.  However, the polls of likely voters barely deviated from the average of all polls.  Either the polling of "likely voters" is not at all accurate itself, or voter enthusiasm was not a huge factor.
# * In every populated state we looked, Trump was overestimated in the average of all polls, though this carried little effect in the electoral college process as these states voted the same direction anyway.  As New York, California, and Illinois are solid blue states, a simple explanation could be that Trump voters didn't see their ballot as worthwhile in these regions.  However, Texas, a consistent red state, is a different story.  Either Trump voters were complacent in Texas, or there was strong turnout for Clinton.
# * In Florida, Pennsylvania, and Ohio, the average of all polls did better than the polls that have been historically reliable.  In Michigan, Wisconsin, and Minnesota, three bordering northern states, the historically reliable polls were more accurate.  We should keep this in mind when analyzing 2020 polls.
# * In the final polling days leading up to the election, Clinton had negative momemtum nationally and in most states.  Keep in mind there were an additional few days between the end of the final polling period and voting day, so this negative momentum likely continued which caused polls to underestimate Trump's chances.
# 
# * The Times-Picayune/Lucid, ABC News/Washington Post, and YouGov predicted the national popular vote most accurately.
# * Google Consumer Surveys and SurveyMonkey were most consistently accurate in the battleground states.
# * SurveyMonkey was a very accurate pollster in Florida, Ohio, Michigan, and Wisconsin
# * Google Consumer Surveys was a very strong pollster in Pennsylvania, Ohio, and Minnesota
# 
# * Nationally, the number of polls a pollster released the final two weeks before the election predicted the final accuracy incredibly well.  We should trust pollsters who release more national polls before the election.
# * In battleground states, the number of polls a pollster released the final two weeks before the election also predicted the final accuracy quite well.  Of course, the sample size is smaller for state polls than national polls

# ## What Will Happen in 2020? (COMING SOON)
