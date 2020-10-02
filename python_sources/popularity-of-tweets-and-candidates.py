import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import re as re
from pylab import *
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


def tweet_about(df, regex, name):
    df['temp'] = df['text'].apply(lambda x: re.match('.*' + regex+ '.*',x))
    df.loc[df['temp'].apply(lambda x: x is not None), name] = 1
    del df['temp']

def plot_cand(dat, x, y, cand_name, extra_lab = ''):
    df = dat[dat['handle'] == cand_name]
    x_vals = np.unique(df[x].values)
    plt.plot(x_vals, df[y], label=cand_name + ' ' +extra_lab)
    

data = pd.read_csv('../input/tweets.csv')
data = data[data['is_retweet'] == False]
#data = data.dropna(subset = data['text'])
data['text'] = data['text'].apply(lambda x: x.lower())
data['weight'] = data['retweet_count'] + data['favorite_count'] + data['is_retweet']
data.sort_values(by = 'weight', ascending = False)
data['stupid_temp'] = data['text'].apply(lambda x: re.match('.*stupid.*',x))
data.loc[data['stupid_temp'].apply(lambda x: x is not None), 'stupid'] = 1

tweet_about(data, '.*wi+n+.*', 'win')
tweet_about(data, '.*(hilary|clinton).*', 'hilary')
tweet_about(data, '.*(donald|trump).*', 'trump')
tweet_about(data, '.*rally.*', 'rally')
tweet_about(data, '.*america.*', 'america')
tweets_person = [data['hilary'].sum(), data['trump'].sum()]
                
candidates = ['Hillary Clinton','Donald Trump']
plt.bar([0,1],tweets_person, align = 'center')
plt.xticks([0,1],candidates)
plt.title("Tweets about each candidate")
fName = 'tweets_about_candidates'
plt.savefig(fName, type = 'png')

#Group data according to candidate
cand_handles = ['HillaryClinton','realDonaldTrump']
cand_tweets = data.groupby('handle')

#find the most favorited tweets
fav_tweets = []
for df in cand_tweets:
    fav_df = df[1][['handle','text','retweet_count','favorite_count']]
    name = str(fav_df['handle'].unique()).replace("[","").replace("]","")
    fav_df = fav_df.sort_values(by = 'favorite_count', ascending = False)
    fav_5 = fav_df[['handle','text']].head()
    fav_5 = fav_5['handle'].str.cat(fav_5['text'], sep = ': ')
    fav_tweets.append(fav_5)
#print(fav_5
fav_hil = fav_tweets[0].values
fav_trump = fav_tweets[1].values
fav_tweets = pd.DataFrame([fav_hil,fav_trump]).transpose()
fav_tweets.columns = ['Hillary','Trump']
fav_tweets.to_csv('Favorite Tweets.csv')
print(fav_tweets)

#Aggregate all the variables for each candidate
tweet_stat_sum = cand_tweets.aggregate(np.sum)
tweet_stat_count = cand_tweets.aggregate(np.count_nonzero)

#Number of tweets about opposing candidate
tweets_about_opp = [tweet_stat_sum['trump']['HillaryClinton'], tweet_stat_sum['hilary']['realDonaldTrump']]
total_tweets = tweet_stat_count['text']
percent_about_opp = tweets_about_opp / total_tweets.apply(lambda x: float(x))
cla()
clf()
plt.bar([0,1], percent_about_opp, align = 'center')
plt.xticks([0,1], candidates)
plt.title("Percentage of Tweets About Opponent")
fName2 = 'pct_tweets_about_opp'
plt.savefig(fName2, type = 'png')

data['time'] = pd.to_datetime(data['time'])
data['month'] = data['time'].apply(lambda x: x.month)
monthly_cand_data = data.groupby(['handle', 'month'], as_index = False)

monthly_cand_tweet_stat = monthly_cand_data.aggregate(np.sum)
trump_monthly_retweets = monthly_cand_tweet_stat['retweet_count']
months = ['Jan', 'Feb', 'Mar','Apr','May','Jun','Jul','Aug','Sep']

cla()
clf()
for cand in cand_handles:
    plot_cand(monthly_cand_tweet_stat, 'month','favorite_count',cand, 'favorites')
    plot_cand(monthly_cand_tweet_stat, 'month','retweet_count',cand, 'retweets')
plt.legend(loc = 'best')
plt.xticks(range(1,10), months)
plt.title("Favorites and Retweets by Candidate")
plt.tight_layout()
fName3 = 'retweets_mo'
plt.savefig(fName3, type = 'png')
