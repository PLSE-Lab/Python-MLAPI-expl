# Importing pandas
import pandas as pd

# Importing matplotlib and setting aesthetics for plotting later.
import matplotlib.pyplot as plt
%matplotlib inline
%config InlineBackend.figure_format = 'svg' 
plt.style.use('fivethirtyeight')

#FILLTERING DATA 
# Reading datasets/coinmarketcap_06122017.csv into pandas
dec6 = pd.read_csv('../input/coin-market/coinmarketcap_06012018.csv')

# Selecting the 'id' and the 'market_cap_usd' columns
market_cap_raw = dec6[['id','market_cap_usd']]

# Counting the number of values
print(market_cap_raw.count())

# Filtering out rows without a market capitalization
cap = market_cap_raw.query('market_cap_usd > 0')

# Counting the number of values again
print(cap.count())

#COMPARING BITCOIN TO THE MARKET
#Declaring these now for later use in the plots
TOP_CAP_TITLE = 'Top 10 market capitalization'
TOP_CAP_YLABEL = '% of total cap'

# Selecting the first 10 rows and setting the index
cap10 = cap.head(10).set_index('id')

# Calculating market_cap_perc
cap10 = cap10.assign(market_cap_perc = lambda x: (x.market_cap_usd / cap.market_cap_usd.sum()) * 100)

# Plotting the barplot with the title defined above 
ax = cap10.market_cap_perc.head(10).plot(kind = 'bar', title = TOP_CAP_TITLE)
                                         
# Annotating the y axis with the label defined above
ax.set_ylabel(TOP_CAP_YLABEL)

# Colors for the bar plot
COLORS = ['orange', 'green', 'orange', 'cyan', 'cyan', 'blue', 'silver', 'orange', 'red', 'green']

# Plotting market_cap_usd as before but adding the colors and scaling the y-axis  
ax = cap10.market_cap_perc.head(10).plot(kind = 'bar', title = TOP_CAP_TITLE, color = COLORS, logy = True)

# Annotating the y axis with 'USD'
ax.set_ylabel('USD')

# Final touch! Removing the xlabel as it is not very informative
ax.set_xlabel('')

#VOLATILITY OF CRYTOCURRENCIES
# Selecting the id, percent_change_24h and percent_change_7d columns
volatility = dec6[['id','percent_change_24h','percent_change_7d']]

# Setting the index to 'id' and dropping all NaN rows
volatility = volatility.set_index('id').dropna()

# Sorting the DataFrame by percent_change_24h in ascending order
volatility = volatility.sort_values(by=['percent_change_24h'])

# Checking the first few rows
print(volatility.head())

#Defining a function with 2 parameters, the series to plot and the title
def top10_subplot(volatility_series, title):
    # Making the subplot and the figure for two side by side plots
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 6))
    
    # Plotting with pandas the barchart for the top 10 losers
    ax = volatility_series[:10].plot.bar(ax=axes[0], color='darkred')
    
    # Setting the figure's main title to the text passed as parameter
    fig.suptitle(t = title)
    # ... YOUR CODE FOR TASK 7 ...
    
    # Setting the ylabel to '% change'
    ax.set_ylabel('% change')
    # ... YOUR CODE FOR TASK 7 ...
    
    # Same as above, but for the top 10 winners
    ax = volatility_series[-10:].plot.bar(ax=axes[1], color='darkblue')
    
    # Returning this for good practice, might use later
    return fig, ax

DTITLE = "24 hours top losers and winners"

# Calling the function above with the 24 hours period series and title DTITLE  
fig, ax = top10_subplot(volatility.percent_change_24h, DTITLE)

# Sorting in ascending order
volatility7d = volatility[['percent_change_7d']].sort_values(by=['percent_change_7d'])

WTITLE = "Weekly top losers and winners"

# Calling the top10_subplot function
fig, ax = top10_subplot(volatility7d, WTITLE)

#MARKET CAPITALIZATION
# Selecting everything bigger than 10 billion 
largecaps = cap.query('market_cap_usd>10000000000')

# Printing out largecaps
print(largecaps)

# Making a nice function for counting different marketcaps from the
# "cap" DataFrame. Returns an int.
def capcount(query_string):
    return cap.query(query_string).count().id

# Labels for the plot
LABELS = ["biggish", "micro", "nano"]

# Biggish Crypto
biggish = capcount('(market_cap_usd>10000000000) or (market_cap_usd>2000000000 and market_cap_usd<10000000000) or (market_cap_usd>300000000 and market_cap_usd<2000000000)')

# Micro Crypto
micro = capcount('market_cap_usd>50000000 and market_cap_usd<300000000')

# Nano Crypto
nano = capcount('market_cap_usd<50000000')

# Making a list with the 3 counts
values = [biggish,micro,nano]

# Plotting them with matplotlib
plt.bar(x=LABELS, height=values)