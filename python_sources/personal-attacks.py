from __future__ import division
import sqlite3, time, csv, re, random
import numpy
import re;
sql_conn = sqlite3.connect('../input/database.sqlite')

testdata = sql_conn.execute("SELECT subreddit, body, author FROM May2015 LIMIT 10000000")
youre1 = "y?o?u'?r?e? *(are)? *(a|an)? *(butthurt|stupid|idiot|moron|lame|retarded|asshole|idiots|shitty|dumb|retard|dumbass|nigger|cunt|pathetic|circlejerk|faggot|trash|jerk|shit|garbage|losers|ugly|neckbeard|douche|bitch|scum)"
totalComments = dict();
curses = dict()
totalCurses = 0;
cursesPerComment = dict();
users = dict();
usersCursed = dict();
totalUsers = 0;
for data in testdata:
    if users.get(data[2], 0) == 0:
        totalUsers += 1;
    users[data[2]] = users.get(data[2], 0)+1;
    found = False;
    current = 0;
    totalComments[data[0]] = totalComments.get(data[0], 0)+ 1;
    if (re.match(youre1, data[1]) != None):
        if found == False:
            curses[data[0]]= curses.get(data[0], 0)+ 1;
            found = True;
            current += 1;
            usersCursed[data[2]] = usersCursed.get(data[2], 0)+1;
        else:
            current += 1;
    if found:
        totalCurses += 1;
        cursesPerComment[current] = cursesPerComment.get(current, 0)+ 1;
    
percentage = dict();
for subreddit in curses:
    percentage[subreddit] = curses[subreddit]/totalComments[subreddit];
for subreddit in sorted(percentage, key=percentage.get, reverse=True):
    if(totalComments[subreddit] > 1000):
        print (subreddit, (percentage[subreddit]), totalComments[subreddit])

for number in cursesPerComment:
    print("%s Curse/s per comment: %s", number, cursesPerComment[number]);
    
print ("Total unique commenters: ", totalUsers);
counter = 0;
for user in sorted(usersCursed, key=usersCursed.get, reverse=True):
    counter += 1;
    print (user, (usersCursed[user]/users[user]), users[user])
    if counter ==101:
        break;

