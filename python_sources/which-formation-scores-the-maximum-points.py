#!/usr/bin/env python
# coding: utf-8

# ## Linear Programming optimization for the best team to pick from the dataset, given a budget and other player position based constraints. We want to maximize the points earned by this configuration. Note that we don't really use the features yet, but once more data becomes available when the Premier League starts, and the API data gets updated for each gameweek, then we can start predictive modeling.
# 
# ## The idea for this comes from this wonderful blog post http://www.philipkalinda.com/ds9.html. I'll keep adding to it, but this is the core idea and application.

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

from pulp import * # Python package for Linear Programming
import re
import ast

import warnings
warnings.filterwarnings('ignore')
# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv('../input/FPL_2018_19_Wk0.csv')
df = df[df['Points']>50] # For a first pass, only pick the players who scored >50 points last season
df.reset_index(inplace=True,drop=True)
df.head()


# In[ ]:


df.shape


# In[ ]:


# Create the decision variables.. all the players (so 371 for this initial try)
def create_dec_var(df):
    decision_variables = []
    
    for rownum, row in df.iterrows():
        variable = str('x' + str(rownum))
        variable = pulp.LpVariable(str(variable), lowBound = 0, upBound = 1, cat= 'Integer')
        decision_variables.append(variable)
                                  
    return decision_variables

# This is what we want to maximize (objective function)
def total_points(df,lst,prob):
    total_points = ""
    for rownum, row in df.iterrows():
        for i, player in enumerate(lst):
            if rownum == i:
                formula = row['Points']*player
                total_points += formula

    prob += total_points
    
    return prob

# Add constraint for cash
def cash(df,lst,prob,avail_cash):
    total_paid = ""
    for rownum, row in df.iterrows():
        for i, player in enumerate(lst):
            if rownum == i:
                formula = row['Cost']*player
                total_paid += formula
    prob += (total_paid <= avail_cash), "Cash"
    
    return prob

# Add constraint for number of goalkeepers
def team_gkp(df,lst,prob,avail_gk):
    total_gk = ""
    for rownum, row in df.iterrows():
        for i, player in enumerate(lst):
            if rownum == i:
                if row['Position'] == 'GKP':
                    formula = 1*player
                    total_gk += formula

    prob += (total_gk == avail_gk), "GK"
    
    return prob

# Add constraint for number of defenders
def team_def(df,lst,prob,avail_def):
    total_def = ""
    for rownum, row in df.iterrows():
        for i, player in enumerate(lst):
            if rownum == i:
                if row['Position'] == 'DEF':
                    formula = 1*player
                    total_def += formula

    prob += (total_def == avail_def), "DEF"
    
    return prob

# Add constraint for number of midfielders
def team_mid(df,lst,prob,avail_mid):
    total_mid = ""
    for rownum, row in df.iterrows():
        for i, player in enumerate(lst):
            if rownum == i:
                if row['Position'] == 'MID':
                    formula = 1*player
                    total_mid += formula

    prob += (total_mid == avail_mid), "MID"
    
    return prob

# Add constraint for number of forwards
def team_fwd(df,lst,prob,avail_fwd):
    total_fwd = ""
    for rownum, row in df.iterrows():
        for i, player in enumerate(lst):
            if rownum == i:
                if row['Position'] == 'FWD':
                    formula = 1*player
                    total_fwd += formula

    prob += (total_fwd == avail_fwd), "FWD"
    
    return prob


# In[ ]:


# Assemble the whole problem data
def find_prob(df,ca,gk,de,mi,fw):
    
    prob = pulp.LpProblem('FantasyTeam', pulp.LpMaximize)
    lst = create_dec_var(df)
    
    prob = total_points(df,lst,prob)
    prob = cash(df,lst,prob,ca)
    prob = team_gkp(df,lst,prob,gk)
    prob = team_def(df,lst,prob,de)
    prob = team_mid(df,lst,prob,mi)
    prob = team_fwd(df,lst,prob,fw)
    
    return prob


# In[ ]:


# Solve the problem
def LP_optimize(df, prob):
    prob.writeLP('FantasyTeam.lp')
    
    optimization_result = prob.solve()
    assert optimization_result == pulp.LpStatusOptimal

#     print("Status:", LpStatus[prob.status])
#     print("Optimal value:", pulp.value(prob.objective))


# In[ ]:


# Find the optimal team
def df_decision(df,prob):
    variable_name = []
    variable_value = []

    for v in prob.variables():
        variable_name.append(v.name)
        variable_value.append(v.varValue)

    df_vals = pd.DataFrame({'variable': variable_name, 'value': variable_value})
    for rownum, row in df_vals.iterrows():
        value = re.findall(r'(\d+)', row['variable'])
        df_vals.loc[rownum, 'variable'] = int(value[0])

    df_vals = df_vals.sort_index(by='variable')

    #append results
    for rownum, row in df.iterrows():
        for results_rownum, results_row in df_vals.iterrows():
            if rownum == results_row['variable']:
                df.loc[rownum, 'Decision'] = results_row['value']

    return df


# ## First we want to see how it does for picking all 15 players on the team, with a maximum possible budget of 1000 (2 GK, 5 DEF, 5 FWD, 3 FWD)

# In[ ]:


prob = find_prob(df,1000,2,5,5,3)
LP_optimize(df,prob)


# In[ ]:


df_final = df_decision(df,prob)
print(df_final[df_final['Decision']==1.0].Cost.sum(), df_final[df_final['Decision']==1.0].Points.sum())


# In[ ]:


# The final 15
df_final[df_final['Decision']==1.0]


# ### Now to see which formation maximizes the points earned by the starting XI.. Looking at the roster for the current season, the minimum price for a GK and a DEF are 40 each, and 45 for MID and FWD respectively. That means the maximum Cash we can use to pick a starting XI is 830. 

# ## 7 possible formations::
# 1. 3-4-3
# 2. 3-5-2
# 3. 4-3-3
# 4. 4-4-2
# 5. 4-5-1
# 6. 5-3-2
# 7. 5-4-1

# In[ ]:


prob343 = find_prob(df,830,1,3,4,3)
prob352 = find_prob(df,830,1,3,5,2)
prob433 = find_prob(df,830,1,4,3,3)
prob442 = find_prob(df,830,1,4,4,2)
prob451 = find_prob(df,830,1,4,5,1)
prob532 = find_prob(df,830,1,5,3,2)
prob541 = find_prob(df,830,1,5,4,1)


# In[ ]:


def prob_formations(df,prob):
    LP_optimize(df,prob)
    df_final = df_decision(df,prob)
    
    print(df_final[df_final['Decision']==1.0]['Points'].sum())
    
    return(df_final[df_final['Decision']==1.0])


# In[ ]:


# 1. 3-4-3
prob_formations(df,prob343)


# In[ ]:


# 2. 3-5-2
prob_formations(df,prob352)


# In[ ]:


# 3. 4-3-3
prob_formations(df,prob433)


# In[ ]:


# 4. 4-4-2
prob_formations(df,prob442)


# In[ ]:


# 5. 4-5-1
prob_formations(df,prob451)


# In[ ]:


# 6. 5-3-2
prob_formations(df,prob532)


# In[ ]:


# 7. 5-4-1
prob_formations(df,prob541)


# ## So we see that, given Mo Salah's classification as a Midfielder, it makes the most sense to stack your defence or midfield (4-5-1 or 5-4-1) to get the maximum points tally in a given Gameweek. I'll add more analysis (and definitely visualizations) over time. 

# In[ ]:




