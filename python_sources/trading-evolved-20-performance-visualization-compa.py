#!/usr/bin/env python
# coding: utf-8

# # 20 Performance Visualization and Combinations

# The following code should run successfully on a local computer, but not here. The code is correct. Outputs are hidden.

# # Model comparison

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


import pandas as pd

base_path = '../Backtests/'

bm = 'SPXTR'
bm_name = 'S&P 500 Total Return Index'


strat_names = {
    "core_trend" : "Core Trend Strategy",    
    "time_return" : "Time Return Strategy",
    "counter_trend" : "Counter Trend Strategy",
    "curve_trading" : "Curve Trading Strategy",
    "equity_momentum" : "Equity Momentum Strategy",
}

strat = 'curve_trading'
strat_name = strat_names[strat]

df = pd.read_csv(base_path + strat + '.csv', index_col=0, parse_dates=True, names=[strat] )
df[bm_name] = pd.read_csv(base_path + bm + '.csv', index_col=0, parse_dates=[0] )
df = df.loc[:'2018-12-31'].dropna()
print("Fetched: {}".format(strat_name))


# In[ ]:


import matplotlib
import matplotlib.pyplot as plt
yr_periods = 252

# Format for book display
font = {'family' : 'eurostile',
        'weight' : 'normal',
        'size'   : 16}
matplotlib.rc('font', **font)

def equity_graph(df):
    df = df / df.iloc[0]
    df['Correlation'] = df[strat].pct_change().rolling(window=int(yr_periods / 1)).corr(df[bm_name].pct_change())
    
    df['Drawdown'] = (df[strat] / df[strat].cummax()) - 1
    
    df.fillna(0, inplace=True) # make sure no NA values are in there
    fig = plt.figure(figsize=(15, 12))

    # First chart
    ax = fig.add_subplot(311)
    ax.set_title('Strategy Comparisons')
    ax.semilogy(df[strat], '-',label=strat_name, color='black')
    ax.semilogy(df[bm_name] , '--', color='grey')
    ax.legend()
    
    # Second chart
    ax = fig.add_subplot(312)
    ax.fill_between(df.index, df['Drawdown'], label='Drawdown', color='black')
    ax.legend()

    # Third chart
    ax = fig.add_subplot(313)
    ax.fill_between(df.index,df['Correlation'], label='12M Rolling Correlation', color='grey')
    ax.legend()

equity_graph(df)


# In[ ]:


import empyrical as em
from IPython.core.display import display, HTML

monthly_data = em.aggregate_returns(df[strat].pct_change(),'monthly')
yearly_data = em.aggregate_returns(df[strat].pct_change(),'yearly')

table = """
<table class='table table-hover table-condensed table-striped'>
<thead>
<tr>
<th style="text-align:right">Year</th>
<th style="text-align:right">Jan</th>
<th style="text-align:right">Feb</th>
<th style="text-align:right">Mar</th>
<th style="text-align:right">Apr</th>
<th style="text-align:right">May</th>
<th style="text-align:right">Jun</th>
<th style="text-align:right">Jul</th>
<th style="text-align:right">Aug</th>
<th style="text-align:right">Sep</th>
<th style="text-align:right">Oct</th>
<th style="text-align:right">Nov</th>
<th style="text-align:right">Dec</th>
<th style="text-align:right">Year</th>
</tr>
</thead>
<tbody>
<tr>"""

first_year = True
first_month = True
yr = 0
mnth = 0
for m, val in monthly_data.iteritems():
    yr = m[0]
    mnth = m[1]

    if(first_month):
        table += "<td align='right'><b>{}</b></td>\n".format(yr)
        first_month = False

    if(first_year): # pad empty months for first year if sim doesn't start in January
        first_year = False
        if(mnth > 1):
            for i in range(1, mnth):
                table += "<td align='right'>-</td>\n"

    table += "<td align='right'>{:+.1f}</td>\n".format(val * 100)

    if(mnth==12): # check for dec, add yearly
        table += "<td align='right'><b>{:+.1f}</b></td>\n".format(yearly_data[yr] * 100)     
        table += '</tr>\n <tr> \n'    
        first_month = True

# add padding for empty months and last year's value
if(mnth != 12):
    for i in range(mnth+1, 13):
        table += "<td align='right'>-</td>\n"
        if(i==12):
            table += "<td align='right'><b>{:+.1f}</b></td>\n".format(
                yearly_data[yr] * 100
            ) 
            table += '</tr>\n <tr> \n'
table += '</tr>\n </tbody> \n </table>'

display(HTML(table))


# In[ ]:


def holding_period_map(df):
    yr = em.aggregate_returns(df[strat].pct_change(), 'yearly')
    df = pd.DataFrame(columns=range(1,len(yr)+1), index=yr.index)

    yr_start = 0
    
    table = "<table class='table table-hover table-condensed table-striped'>"
    table += "<tr><th>Years</th>"
    
    for i in range(len(yr)):
        table += "<th>{}</th>".format(i+1)
    table += "</tr>"

    for the_year, value in yr.iteritems(): # Iterates years
        table += "<tr><th>{}</th>".format(the_year) # New table row
        
        for yrs_held in (range(1, len(yr)+1)): # Iterates yrs held 
            if yrs_held   <= len(yr[yr_start:yr_start + yrs_held]):
                ret = em.annual_return(yr[yr_start:yr_start + yrs_held], 'yearly' )
                table += "<td>{:+.0f}</td>".format(ret * 100)
        table += "</tr>"    
        yr_start+=1
    return table

table = holding_period_map(df)
display(HTML(table))


# # Combined Models

# In[ ]:


import pandas as pd
import numpy as np

base_path = '../Backtests/'


# In[ ]:


# Rebalance on percent divergence
class PercentRebalance(object):
    def __init__(self, percent_target):
        self.rebalance_count = 0
        self.percent_target = percent_target
        
    def rebalance(self, row, weights, date):
        total = row.sum()
        rebalanced = row
        rebalanced = np.multiply(total, weights)
        if np.any(np.abs((row-rebalanced)/rebalanced) > (self.percent_target/100.0)):
            self.rebalance_count = self.rebalance_count + 1
            return rebalanced
        else:
            return row

# Rebalance on calendar
class MonthRebalance(object):
    def __init__(self, months):
        self.month_to_rebalance = months
        self.rebalance_count = 0
        self.last_rebalance_month = 0

    def rebalance(self, row, weights, date):
        current_month = date.month

        if self.last_rebalance_month != current_month:
            total = row.sum()
            rebalanced = np.multiply(weights, total)
            self.rebalance_count = self.rebalance_count + 1
            self.last_rebalance_month = date.month
            return rebalanced
        else:
            return row


# In[ ]:



# Calculate the rebalanced combination
def calc_rebalanced_returns(returns, rebalancer, weights):
    returns = returns.copy() + 1
    
    # create a numpy ndarray to hold the cumulative returns
    cumulative = np.zeros(returns.shape)
    cumulative[0] = np.array(weights)

    # also convert returns to an ndarray for faster access
    rets = returns.values

    # using ndarrays all of the multiplicaion is now handled by numpy
    for i in range(1, len(cumulative) ):
        np.multiply(cumulative[i-1], rets[i], out=cumulative[i])
        cumulative[i] = rebalancer.rebalance(cumulative[i], weights, returns.index[i])

    # convert the cumulative returns back into a dataframe
    cumulativeDF = pd.DataFrame(cumulative, index=returns.index, columns=returns.columns)

    # finding out how many times rebalancing happens is an interesting exercise
    print ("Rebalanced {} times".format(rebalancer.rebalance_count))

    # turn the cumulative values back into daily returns
    rr = cumulativeDF.pct_change() + 1
    rebalanced_return = rr.dot(weights) - 1
    return rebalanced_return

def get_strat(strat):
    df = pd.read_csv(base_path + strat + '.csv', index_col=0, parse_dates=True, names=[strat] )
    return df


# In[ ]:


# Use monthly rebalancer, one month interval
rebalancer = MonthRebalance(1)

# Define strategies and weights
portfolio = {
    'core_trend': 0.25,
    'counter_trend': 0.25,
    'curve_trading': 0.25,
    'time_return': 0.25,
}

# Read all the files into one DataFrame
df = pd.concat(
        [
            pd.read_csv('{}{}.csv'.format(
                        base_path,
                        strat
                        ), 
                        index_col=0,
                        parse_dates=True,
                        names=[strat]
                       ).pct_change().dropna()
            for strat in list(portfolio.keys())
        ], axis=1
)

# Calculate the combined portfolio
df['Combined'] = calc_rebalanced_returns(
    df, 
    rebalancer, 
    weights=list(portfolio.values())
    )

df.dropna(inplace=True)



# In[ ]:


# Make Graph
import matplotlib 
import matplotlib.pyplot as plt

include_combined = True
include_benchmark = True
benchmark = 'SPXTR'

if include_benchmark:
    returns[benchmark] = get_strat(benchmark).pct_change()

#returns = returns['2003-1-1':]
normalized = (returns+1).cumprod()

font = {'family' : 'eurostile',
        'weight' : 'normal',
        'size'   : 16}

matplotlib.rc('font', **font)


fig = plt.figure(figsize=(15, 8))

# First chart
ax = fig.add_subplot(111)
ax.set_title('Strategy Comparisons')

dashstyles = ['-','--','-.','.-.', '-']
i = 0
for strat in normalized:
    if strat == 'Combined':
        if not include_combined:
            continue
        clr = 'black'
        dash = '-'
        width = 5
    elif strat == benchmark:
        if not include_benchmark:
            continue
        clr = 'black'
        dash = '-'
        width = 2
    #elif strat == 'equity_momentum':
    #    continue

    else:
        clr = 'grey'
        dash = dashstyles[i]
        width = i + 1
        i += 1
    ax.semilogy(normalized[strat], dash, label=strat, color=clr, linewidth=width)
    
    ax.legend()


# In[ ]:


df.to_clipboard()


# In[ ]:


portfolio = {
    'x': 1,
    'y': 2,
    'z': 3
    
}
#print(portfolio.values())

x = np.array(list(portfolio.keys()))
             
print(x)

