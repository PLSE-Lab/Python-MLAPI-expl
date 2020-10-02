#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np


# ### Import the cricket dataset

# In[ ]:


t20_data = pd.read_csv("../input/T20_matches_ball_by_ball_data.csv", 
                       parse_dates=["date"], low_memory=False)


# ### Check for missing values in the dataset

# In[ ]:


t20_data.isnull().sum()


# ### Add total runs column
# The total_runs column represent the total runs scored by the team 

# In[ ]:


t20_data["runs_plus_extras"] = t20_data["Run_Scored"] + t20_data["Extras"]
t20_data["total_runs"] = t20_data.groupby(["Match_Id", "Batting_Team"])["runs_plus_extras"].cumsum()


# ### Add wickets lost column
# The wickets column respresent the number of wickets lost by the team

# In[ ]:


t20_data["wicket_fall"] = t20_data["Dismissed"].isnull().map({True: 0, False: 1})
t20_data["wickets"] = t20_data.groupby(["Match_Id", "Batting_Team"])["wicket_fall"].cumsum()


# ### Batting scoreboard

# #### Aggregating runs scored and balls faced by each batsman, for a given match (represented by match id)

# In[ ]:


runs_scored = t20_data.groupby(["Match_Id", "Batting_Team", "Striker"], as_index=False)["Run_Scored"].sum()
balls_faced = t20_data.groupby(["Match_Id", "Batting_Team", "Striker"], as_index=False)["Run_Scored"].count()
balls_faced.columns = ["Match_Id", "Batting_Team", "Striker", "Balls"]
batting_scoreboard = pd.merge(runs_scored, balls_faced, 
                              on=["Match_Id", "Batting_Team", "Striker"], how="left")


# #### Capturing the dismissal (how a batsman got out) information from the t20 dataframe

# In[ ]:


t20_dismissal = t20_data[["Match_Id", "Batting_Team", "Striker", "Dismissal"]]
t20_dismissal["concat_key"] = t20_dismissal["Match_Id"].map(str) + ":" + t20_dismissal["Striker"]
t20_dismissal = t20_dismissal.drop_duplicates(subset=["concat_key"], keep="last")
t20_dismissal = t20_dismissal.drop(labels="concat_key", axis = 1)
t20_dismissal = t20_dismissal.sort_values(["Match_Id", "Batting_Team"])
t20_dismissal.Dismissal.fillna("not out", inplace=True)


# #### Merging the dismissal information to batting scoreboard dataframe

# In[ ]:


batting_scoreboard = pd.merge(batting_scoreboard, t20_dismissal, 
                              on=["Match_Id", "Batting_Team", "Striker"], how="left")
batting_scoreboard.head()


# ### Batsman Statistics
# #### Reference: https://en.wikipedia.org/wiki/Cricket_statistics
# - Innings: The number of innings in which the batsman actually batted.
# - Not outs: The number of times the batsman was not out at the conclusion of an innings they batted in.1
# - Runs: The number of runs scored.
# - Highest score: The highest score ever made by the batsman.
# - Batting average: The total number of runs divided by the total number of innings in which the batsman was out.
# - Centuries: The number of innings in which the batsman scored one hundred runs or more.
# - Half-centuries: The number of innings in which the batsman scored fifty to ninety-nine runs (centuries do not count as half-centuries as well).
# - Balls faced: The total number of balls received, including no balls but not including wides.
# - Strike rate: The average number of runs scored per 100 balls faced. (SR = [100 * Runs]/BF)

# #### Get a unique list of batsman from the scoreboard dataframe

# In[ ]:


batsman_statistics = pd.DataFrame({"Batsman": batting_scoreboard.Striker.unique()})


# #### Compute "Innings" information for each batsman from the scoreboard dataframe

# In[ ]:


Innings = pd.DataFrame(batting_scoreboard.Striker.value_counts())
Innings.reset_index(inplace=True)
Innings.columns = ["Batsman", "Innings"]


# #### Compute "Not outs" information for each batsman from the scoreboard dataframe

# In[ ]:


Not_out = batting_scoreboard.Dismissal == "not out"
batting_scoreboard["Not_out"] = Not_out.map({True: 1, False: 0})
Not_out = pd.DataFrame(batting_scoreboard.groupby(["Striker"])["Not_out"].sum())
Not_out.reset_index(inplace=True)
Not_out.columns = ["Batsman", "Not_out"]


# #### Compute "Balls" information for each batsman from the scoreboard dataframe

# In[ ]:



Balls = pd.DataFrame(batting_scoreboard.groupby(["Striker"])["Balls"].sum())
Balls.reset_index(inplace=True)
Balls.columns = ["Batsman", "Balls"]


# #### Compute "Runs" information for each batsman from the scoreboard dataframe

# In[ ]:


Run_Scored = pd.DataFrame(batting_scoreboard.groupby(["Striker"])["Run_Scored"].sum())
Run_Scored.reset_index(inplace=True)
Run_Scored.columns = ["Batsman", "Run_Scored"]


# #### Compute "Highest score" information for each batsman from the scoreboard dataframe

# In[ ]:


Highest_Score = pd.DataFrame(batting_scoreboard.groupby(["Striker"])["Run_Scored"].max())
Highest_Score.reset_index(inplace=True)
Highest_Score.columns = ["Batsman", "Highest_Score"]


# #### Compute "Centuries " information for each batsman from the scoreboard dataframe

# In[ ]:


Centuries = pd.DataFrame(batting_scoreboard.loc[batting_scoreboard.Run_Scored >= 100,].groupby(["Striker"])["Run_Scored"].count())
Centuries.reset_index(inplace=True)
Centuries.columns = ["Batsman", "Centuries"]


# #### Compute "Half-Centuries " information for each batsman from the scoreboard dataframe

# In[ ]:


Half_Centuries = pd.DataFrame(batting_scoreboard.loc[(batting_scoreboard.Run_Scored >= 50) & 
                                                     (batting_scoreboard.Run_Scored < 100),].groupby(["Striker"])["Run_Scored"].count())
Half_Centuries.reset_index(inplace=True)
Half_Centuries.columns = ["Batsman", "Half_Centuries"]


# #### Merge all the metric to the batsman statitics dataframe     

# In[ ]:


batsman_statistics = pd.merge(batsman_statistics, Innings, on=["Batsman"], how="left")
batsman_statistics = pd.merge(batsman_statistics, Not_out, on=["Batsman"], how="left")
batsman_statistics = pd.merge(batsman_statistics, Balls, on=["Batsman"], how="left")
batsman_statistics = pd.merge(batsman_statistics, Run_Scored, on=["Batsman"], how="left")
batsman_statistics = pd.merge(batsman_statistics, Highest_Score, on=["Batsman"], how="left")

batsman_statistics = pd.merge(batsman_statistics, Centuries, on=["Batsman"], how="left")
batsman_statistics.Centuries.fillna(0, inplace=True)
batsman_statistics.Centuries = batsman_statistics.Centuries.astype("int")

batsman_statistics = pd.merge(batsman_statistics, Half_Centuries, on=["Batsman"], how="left")
batsman_statistics.Half_Centuries.fillna(0, inplace=True)
batsman_statistics.Half_Centuries = batsman_statistics.Half_Centuries.astype("int")


# #### Compute "Batting average" for each batsman from the scoreboard dataframe

# In[ ]:


batsman_statistics["Batting_Average"] = batsman_statistics.Run_Scored / (batsman_statistics.Innings - batsman_statistics.Not_out)
batsman_statistics.loc[batsman_statistics["Batting_Average"] == np.inf, "Batting_Average"] = 0
batsman_statistics.loc[batsman_statistics["Batting_Average"].isnull(), "Batting_Average"] = 0


# #### Compute "Strike rate for each batsman from the scoreboard dataframe

# In[ ]:


batsman_statistics["Strike_Rate"] = (batsman_statistics.Run_Scored * 100) / batsman_statistics.Balls


# In[ ]:


batsman_statistics = batsman_statistics.round({"Batting_Average": 2, "Strike_Rate": 2})


# In[ ]:


batsman_statistics.head()


# In[ ]:




