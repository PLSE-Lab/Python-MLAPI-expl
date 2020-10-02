#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('tar -xjf ../input/trades.tar.bz')


# In[ ]:


get_ipython().system('wget https://raw.githubusercontent.com/matplotlib/mpl_finance/master/mpl_finance.py')


# In[ ]:


import matplotlib.pyplot as plt
from mpl_finance import candlestick_ohlc
import numpy as np
import re
parser = re.compile('[A-Za-z0-9-]* \d+.\d+.\d+ \d+:\d+:\d+.\d+ (\d+) price (\d+.?\d+?) dir (BUY|SELL) amount (\d+)')


# In[ ]:


lines = list(filter(None, open('trades.txt').read().splitlines()))
print('Lines: ', len(lines))


# Convert data to NP matrix
# 
# __(time, price multiplied by 100, side, amount)__

# In[ ]:


def lines_to_np_matrix(raw_data):
    def line_to_np_row(line):
        a,b,c,d = parser.search(line).groups()
        return [int(a), int(float(b)*100), +1 if c=='BUY' else -1, int(d)]
    return np.vstack(list(map(line_to_np_row, raw_data)))


# In[ ]:


data = lines_to_np_matrix(lines)


# In[ ]:


lines = [] # free space


# Transform trades to candles
# 
# __(time, open, high, low, close)__
# 
# and list of trades for each candle

# In[ ]:


def data_to_candles(data, period = 60*1000*1000): #60*1000*1000 = 1min
    highs,lows,opens,closes = [data[0][1]],[data[0][1]],[data[0][1]],[data[0][1]]
    trades = [[]]
    times = [data[0][0]//period]
    for entry in data:
        time = entry[0]//period
        price = entry[1]
        quantity = entry[3]*entry[2]
        if time == times[-1]:
            trades[-1].append((price, quantity))
            highs[-1] = max(highs[-1], price)
            lows[-1] = min(lows[-1], price)
            closes[-1] = price
        else:
            trades.append([(price, quantity)])
            closes.append(price)
            opens.append(price)
            highs.append(price)
            lows.append(price)
            times.append(time)
    return np.vstack([times, opens, highs, lows, closes]).T, trades

def extract_consequtive_parts(candles, trades):
    # extracts consequtive trade sessions from data
    parts = []
    prev = 0
    for i in range(1, len(candles)):
        if candles[i][0] != candles[i-1][0]+1:
            parts.append(list(range(prev,i)))
            prev = i
    parts.append(list(range(prev,len(candles))))
    return parts


# In[ ]:


candles, trades = data_to_candles(data)
parts = extract_consequtive_parts(candles, trades)


# Visualization

# In[ ]:


def graph_data(candles):

    fig = plt.figure()
    ax1 = plt.subplot2grid((1,1), (0,0))
    time, openp, highp, lowp, closep = candles.T
    ohlc = zip(time, openp/100, highp/100, lowp/100, closep/100)
    candlestick_ohlc(ax1, ohlc, width=0.4, colorup='#77d879', colordown='#db3f3f')
    for label in ax1.xaxis.get_ticklabels():
        label.set_rotation(45)
    ax1.grid(True)
    plt.ylabel('Price')
    plt.subplots_adjust(left=0.09, bottom=0.20, right=0.94, top=0.90, wspace=0.2, hspace=0)
    plt.show()


# In[ ]:


graph_data(candles[parts[0]])


# In[ ]:


import random

# Model returns list of trades
# trade = (minute_id, price, side)
# +1 = BUY, -1 = SELL

def model_random(candles_all, trades_all, parts):
    # Buy random
    my_trades = []
    for part in parts:
        part = part[:-10] # without last 10 mins
        trades = [trades_all[i] for i in part] # trades by minute inside one part
        for minute, trades_in_minute in zip(part, trades):
            wanted_price = None
            wanted_position = None
            for trade in trades_in_minute:
                if wanted_price:
                    if trade[0] == wanted_price and trade[1]*wanted_position > 0:
                        my_trades.append((minute, trade[0], trade[1]/abs(trade[1])))
                        break
                elif random.random() < 0.0001:
                    wanted_price = trade[0]
                    wanted_position = trade[1]
    return my_trades

def model_0(candles_all, trades_all, parts):
    # Buy if there is 1000 contracts in one order
    my_trades = []
    for part in parts:
        part = part[:-10] # without last 10 mins
        trades = [trades_all[i] for i in part] # trades by minute inside one part
        for minute, trades_in_minute in zip(part, trades):
            wanted_price = None
            wanted_position = None
            for trade in trades_in_minute:
                if wanted_price:
                    if trade[0] == wanted_price and trade[1]*wanted_position > 0:
                        my_trades.append((minute, trade[0], trade[1]/abs(trade[1])))
                        break
                elif abs(trade[1]) >= 1000:
                    wanted_price = trade[0]
                    wanted_position = trade[1]
    return my_trades

def model_1(candles_all, trades_all, parts):
    # Buy if current candle has changed for more than 10 (close-open)
    my_trades = []
    for part in parts:
        part = part[:-10] # without last 10 mins
        trades = [trades_all[i] for i in part] # trades by minute inside one part
        for minute, trades_in_minute in zip(part, trades):
            wanted_price = None
            wanted_position = None
            if candles_all[minute][4] - candles_all[minute][1] >= 10:
                wanted_price = candles_all[minute][4]
                wanted_position = -1
            if candles_all[minute][4] - candles_all[minute][1] <= -10:
                wanted_price = candles_all[minute][4]
                wanted_position = +1
            if wanted_price is None:
                continue
            for trade in trades_in_minute:
                if trade[0] == wanted_price and trade[1]*wanted_position > 0:
                    my_trades.append((minute, trade[0], trade[1]/abs(trade[1])))
                    break
    return my_trades

def model_2(candles_all, trades_all, parts):
    # Buy if there is 1500 contracts in one order
    my_trades = []
    for part in parts:
        part = part[:-10] # without last 10 mins
        trades = [trades_all[i] for i in part] # trades by minute inside one part
        for minute, trades_in_minute in zip(part, trades):
            wanted_price = None
            wanted_position = None
            for trade in trades_in_minute:
                if wanted_price:
                    if trade[0] == wanted_price and trade[1]*wanted_position > 0:
                        my_trades.append((minute, trade[0], trade[1]/abs(trade[1])))
                        break
                elif abs(trade[1]) >= 1500:
                    wanted_price = trade[0]
                    wanted_position = trade[1]
    return my_trades


# In[ ]:


def model_profit(model, candles, trades, parts):
    model_trades = model(candles, trades, parts)
    summary_profit = np.zeros(11)
    summary_square_profit = np.zeros(11)
    print('Total trades made: ', len(model_trades))
    for trade in model_trades:
        for i in range(1, 11):
            summary_profit[i] += (candles[trade[0]+i][4] - trade[1])*trade[2]
            summary_square_profit[i] += ((candles[trade[0]+i][4] - trade[1])*trade[2])**2
    return summary_profit/len(model_trades), summary_square_profit/len(model_trades)


# In[ ]:


sr, sqr = model_profit(model_random, candles, trades, parts)
s0, sq0 = model_profit(model_0, candles, trades, parts)
s1, sq1 = model_profit(model_1, candles, trades, parts)
s2, sq2 = model_profit(model_2, candles, trades, parts)


# In[ ]:


fee = 0.01/100*6666
plt.plot(sr[1:]-fee, label='random')
plt.plot(s0[1:]-fee, label='1000+')
plt.plot(s1[1:]-fee, label='1500+')
plt.plot(s2[1:]-fee, label='candle change 10+')
plt.legend(loc='lower right')
plt.axhline(0, color='black')
plt.title('Profit in cents per trade per one contract')
plt.xlabel('Minutes wait')
plt.show()


# In[ ]:


fee = 0.01/100*6666
plt.plot(sqr[1:]-fee, label='random')
plt.plot(sq0[1:]-fee, label='1000+')
plt.plot(sq1[1:]-fee, label='1500+')
plt.plot(sq2[1:]-fee, label='candle change 10+')
plt.legend(loc='lower right')
plt.axhline(0, color='black')
plt.title('Average square')
plt.xlabel('Minutes wait')
plt.show()


# In[ ]:


|

