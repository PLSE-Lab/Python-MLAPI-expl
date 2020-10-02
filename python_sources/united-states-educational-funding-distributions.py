#!/usr/bin/env python
# coding: utf-8

# ## United States Educational Funding
# _By Nick "Lines Lines Lines" Brooks_
# 
# Hey. Here is a brief look at spending in the United States between 1992 and 2015. This is a casual project, so I'll keep adding things over time.
# 

# In[ ]:


# General
import pandas as pd
import numpy as np

# Viz
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Warnings
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


df = pd.read_csv('../input/elsect_summary.csv')
df.columns = df.columns.str.capitalize()


# In[ ]:


df.head()


# In[ ]:


# Extracting Missing Count and Unique Count by Column
unique_count = []
for x in df.columns:
    unique_count.append([x,len(df[x].unique()),df[x].isnull().sum()])
    
print("Dataframe Dimension: {} Rows, {} Columns".format(*df.shape))
pd.DataFrame(unique_count, columns=["Column","Unique","Missing"]).set_index("Column").T


# ## Distribution:
# 
# I like to start by getting an idea for the distributions. Helps put everything else in perspective.

# In[ ]:


def multi_distplot(df, columns, log=False):
    f, ax = plt.subplots(3,3,figsize=(12, 12), sharey=False)
    for row,data in enumerate([columns[0:3],columns[3:6],columns[6:9]]):
        for i,y in enumerate(data):
            sns.kdeplot(df[y],shade=True,ax=ax[row,i])
            if log is True:
                ax[row,i].set_xscale('log')
            ax[row,i].set_title("Distribution of {}".format(y))
            ax[row,0].set_ylabel("Density")
            ax[row,i].set_xlabel("USD in Log Scale")
    plt.tight_layout(pad=0)


# In[ ]:


multi_distplot(df=df, columns=df.columns[2:11], log=True)
plt.show()


# In[ ]:


def state_multi_distplot(df, statevar, columns, log=False):
    f, ax = plt.subplots(3,3,figsize=(12, 12), sharey=False)
    for row,data in enumerate([columns[0:3],columns[3:6],columns[6:9]]):
        for i,y in enumerate(data):
            for state in df[statevar][df[statevar].notnull()].unique():
                sns.kdeplot(df[y][df[statevar] == state],shade=False,ax=ax[row,i], label=state)
            if log is True:
                ax[row,i].set_xscale('log')
            ax[row,i].set_title("Distribution of {}".format(y))
            ax[row,0].set_ylabel("Density")
            ax[row,i].set_xlabel("USD in Log Scale")
            ax[row,i].legend_.remove()
    plt.tight_layout(pad=0)


# In[ ]:


state_multi_distplot(df=df, statevar= "State",columns=df.columns[2:11], log=True)
plt.show()


# Some oddballs in sight! Since I want to dig deeper, I'm going to have to find a way to deal with all these meddeling states.

# ## TimeSeries
# Lets look at variables over time. Since there are too many states, I will have to find a meaningful way to set them apart, and carry on the analysis with a smaller group.

# In[ ]:


# State List
states_list = df["State"][df["State"].notnull()].unique()

# Aggregate the States for General Time-Series
df_mean = (df.mean()/1000).reset_index()
df_mean.columns =["Variable","Mean"]

# Since these variables operate at a different level of magnitude, split them.
highmag = list(df_mean["Variable"][(df_mean["Mean"] > 2000) & (df_mean["Variable"] != "Year")])
lowmag = [x for x in list(df_mean["Variable"][df_mean["Variable"] != "Year"]) if x not in highmag]
print("High Magnitude Variables:\n",highmag)
print("\nLow Magnitude Variables:\n",lowmag)


# In[ ]:


f, ax = plt.subplots(1,2, figsize=[11,4])
for i,x in enumerate([lowmag,highmag]):
    (df[x+["Year"]].groupby(["Year"]).mean()/1000).plot(ax=ax[i])
    #ax[i].legend(fontsize='large', loc='center left',bbox_to_anchor=(1, 0.5))
    ax[i].set_title("Aggregated Revenue and Expenditure over Time")
    ax[i].set_xlabel("Year")
ax[0].set_ylabel("USD in Thousands")
plt.tight_layout(pad=0)
plt.show()


# **Thoughs:** <br>
# Total Revenue and Total Expenditure track each other closely. On average, seems like assets more or less equal liabilities. I would be curious to look deeper into this, finding states with the largest difference.
# 
# **Trend Interpretation:** <br>
# Interesting to see the revenue spike and bust during the 2008 recession. Other than that, the trend seems steady in an upwards direction.
# 
# ***
# ### **Volatility:** <Br>

# In[ ]:


# State/Year Aggregator
def aggr_year_state(df, var, n=7):
    temp = (df[[var,"Year","State"]].groupby(["Year", "State"]).mean()/1000).reset_index().set_index("Year")
    temp = temp[temp.notnull()]
    volatile = temp.groupby("State").std().sort_values(by=var, ascending=False).reset_index().loc[:n,"State"]
    return temp, volatile

# Volatility Plot
def volatile_subplots(df, variables):
    f, ax = plt.subplots(len(variables), figsize=[10,9], sharex=True)
    for i,var in enumerate(variables):
        temp, volatile= aggr_year_state(df,var)
        for state in volatile:
            ax[i].plot(temp[var][temp["State"] == state].diff(), label=state)
        ax[i].legend(fontsize='large', loc='center left',bbox_to_anchor=(1, 0.5))
        ax[i].set_title("Yearly Difference for {} - Top 7 Volatile States".format(var))

        ax[i].set_ylabel("Difference in Thousands USD")
    ax[len(variables)-1].set_xlabel("Year")
    plt.tight_layout(pad=0)


# In[ ]:


volatile_subplots(df=df, variables= ["Total_revenue", "Total_expenditure", "Enroll"])
plt.show()


# **Interpretation:** <br>
# Cool. Although I chose the states based on Standard Deviation, the spending magnitude surely visually favors the big states here. I wonder what would happen if I normalize these amounts. 

# ***
# ### **Revenue and Expenditure: Benefitter and Contributors on a State Level** <br>

# In[ ]:


df["Exp-Rev"] = (df["Total_expenditure"] - df["Total_revenue"])/df["Total_revenue"]


# **METRIC:** <br>
# This metric is built to quantify how big the gap is between Revenue and Expenditure, while normalizing the value to compare it between states.

# In[ ]:


sns.distplot(df["Exp-Rev"])
plt.show()


# In[ ]:


def state_year_plot(df, var,states):
    f, ax = plt.subplots(1, figsize=[8,5])
    for state in states:
        ax.plot(df[var][df["State"] == state], label=state)
    ax.legend(fontsize='large', loc='center left',bbox_to_anchor=(1, 0.5))
    ax.set_xlabel("Year")
    ax.set_ylabel("Difference Scaled by Total Revenue")
    ax.set_title("Difference Between Total Expenditure and Revenue by State")
        


# In[ ]:


# New DF
benefit = round(df[["Exp-Rev","Year","State"]].groupby(["Year","State"]).mean().mul(100),2).reset_index().set_index("Year")
# Top / Bottom States
sort_exp = benefit.groupby("State").mean().sort_values(by="Exp-Rev", ascending=False).reset_index()
top = list(sort_exp["State"].head())
bot = list(sort_exp["State"].tail())
# Plot
state_year_plot(df=benefit, var="Exp-Rev", states= top+bot)


# While Hawaii overspent by 20% of its Revenue in 2004-2008, Indiana had a surplus of 20% in 2004. Ok.

# ***
# _by Nick_
