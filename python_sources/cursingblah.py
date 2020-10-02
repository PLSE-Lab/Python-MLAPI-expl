from __future__ import division
import sqlite3, time, csv, re, random
import numpy

sql_conn = sqlite3.connect('../input/database.sqlite')

testdata = sql_conn.execute("SELECT score, body FROM May2015 WHERE body LIKE '%fuck%' LIMIT 10000")

upword = ['gold', 'awesome', 'love', 'mother', 'wasps', 'seat', 'often', 'son', 'motherfucker', 'inbox', 'dark', 'reek', 'finally', 'door', 'one', 'imagine', 'mom', 'star', 'bronn', 'told', 'until', 'friends', 'holy', 'those', 'later', 'thank']
downword = ['downvotes', 'downvote', 'sub', 'downvoted', 'downvoting', 'pathetic', 'shut', 'fans', 'fat', 'post', 'lol', 'internet', 'should', 'cares', 'joke', 'off', 'maybe', 'gun', 'dont', 'faggot', 'players', 'vote', 'women', 'stupid', 'racist', 'loser', 'games', 'haha', 'retarded', 'pussy']

inconclusive = 0
upvoted = 0
downvoted = 0
upchance = 0
inconchance = 0
downchance = 0

for data in testdata:
    k = 0
    chance = 0
    for word in upword:
        if word in data[1]:
            chance += 30 - k
        k += 1
    k = 0
    for word in downword:
        if word in data[1]:
            chance -= 30-k
    if chance > 30:
        upchance += 1
    elif chance < -30:
        downchance += 1
    else:
        inconchance += 1
        
    if data[0] > 75:
        upvoted += 1
    elif data[0] < 0:
        downvoted += 1
    else:
        inconclusive += 1
        
print (upchance, inconchance, downchance)
print (upvoted, inconclusive, downvoted)

