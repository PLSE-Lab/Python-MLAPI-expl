from __future__ import division
import sqlite3, time, csv, re, random
import numpy

sql_conn = sqlite3.connect('../input/database.sqlite')

testdata = sql_conn.execute("SELECT score, body FROM May2015 LIMIT 100000")
upword = ['always', 'door', 'interesting', 'necessarily', 'boyfriend', 'favorite', 'secret', 'card', 'app', 'size', 'switch', 'reminds', 'theyll', 'meeting', 'older', 'forget', 'realized', 'powerful', 'definitely', 'imagine', 'heart', 'may', 'fully', 'peace', 'house', 'early', 'often', 'peter', 'universe', 'those', 'turn', 'episode', 'sharing', 'super', 'space', 'best', 'north', 'example', 'agent', 'touch', 'math', 'water', 'party', 'hanging', 'catch', 'wearing', 'itll', 'friends', 'wine', 'feeling', 'several', 'ate', 'official', 'helmet', 'character', 'saw', 'thought', 'tyler', 'armor', 'gym', 'drink', 'budget', 'genuinely', 'keeps', 'nick', 'entertainment', 'key', 'also', 'iron', 'four', 'station', 'apply', 'england', 'open', 'vice', 'enjoyed', 'eventually', 'steel', 'beer', 'office', 'opponent', 'spiderman', 'few', 'political', 'future', 'anus', 'dinosaur', 'japan', 'draft', 'constantly', 'notes', 'campaign', 'program', 'phase', 'huge', 'under', 'itself']
downword = ['downvotes', 'downvote', 'edit', 'downvoting', 'downvoted', 'comment', 'butthurt', 'deleted', 'americans', 'karma', 'women', 'cares', 'arsenal', 'post', 'stupid', 'lol', 'idiot', 'bunch', 'moron', 'shut', 'lame', 'racist', 'sarcasm', 'retarded', 'le', 'fat', 'society', 'asshole', 'disagree', 'idiots', 'proof', 'waste', 'shitty', 'dumb', 'rapist', 'cavs', 'funny', 'retard', 'chick', 'classy', 'arent', 'dumbass', 'loldota', 'nigger', 'cunt', 'reddit', 'wow', 'typical', 'pathetic', 'grammar', 'cheating', 'truth', 'deserve', 'circlejerk', 'faggot', 'sub', 'sir', 'considering', 'xd', 'sjw', 'not', 'simply', 'trash', 'upvote', 'posting', 'yall', 'amd', 'please', 'jerk', 'yay', 'smh', 'shit', 'clip', 'gross', 'should', 'becomes', 'comments', 'zip', 'repost', 'garbage', 'losers', 'votes', 'liberals', 'you', 'flame', 'ugly', 'joke', 'nope', 'guess', 'whoosh', 'internet', 'rofl', 'sorry', 'fool', 'btw', 'subreddit', 'afford', 'fact', 'neckbeards', 'girl']

inconclusive = 0
upvoted = 0
downvoted = 0
upchance = 0
downchance = 0
inconchance = 0
uphit = 0
upmiss = 0
upincon = 0
downhit = 0
downmiss = 0
downincon = 0

i = 0
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
            chance -= 30 - k
    if chance > 30:
        upchance += 1
        if data[0] > 50:
            uphit += 1
        elif data[0] < 0:
            upmiss+= 1
        else:
            upincon += 1
    elif chance < -150:
        downchance += 1
        if data[0] < 0:
            downhit += 1
        elif data[0] > 50:
            downmiss += 1
        else:
            downincon += 1
    else:
        inconchance += 1
        
    if data[0] > 50:
        upvoted += 1
    elif data[0] < 0:
        downvoted += 1
    else:
        inconclusive += 1
        
    i += 1
    
print (upchance, inconchance, downchance)
print (upvoted, inconclusive, downvoted)
print (uphit, upincon, upmiss)
print (downhit, downincon, downmiss)