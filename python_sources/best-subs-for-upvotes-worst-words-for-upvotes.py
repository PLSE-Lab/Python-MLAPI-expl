#!/usr/bin/env python
# coding: utf-8

# What subreddits are you most likely to get upvoted in? What subreddits are you most likely to get downvoted in? What subreddits have the best and worst upvote/downvote ratios?

# In[ ]:


import sqlite3
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from operator import itemgetter
from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
print('Querying DB...\n')
sql_conn = sqlite3.connect('../input/database.sqlite')

#select all posts that have a score greater than 10
scoreBiggerThan10 = sql_conn.execute("SELECT subreddit, score FROM May2015                                 WHERE score > 10                                 LIMIT " + str(500000000))

#select all posts that have a score of 1 or less
scoreLessThan1 = sql_conn.execute("SELECT subreddit, score FROM May2015                                 WHERE score >= 1                                 LIMIT " + str(500000000))

upvoteSubreddits = defaultdict(int)
for item in scoreBiggerThan10:
    upvoteSubreddits[item[0]] += 1
    
downvoteSubreddits = defaultdict(int)
for item in scoreLessThan1:
    downvoteSubreddits[item[0]] += 1


#only select subreddits with at least 5000 posts with a score greater than 10 upvotes
d = {k: v for k, v in upvoteSubreddits.items() if v > 5000}
sorted_x = sorted(d.items(), key=itemgetter(1))
labels, values = zip(*sorted_x)
indexes = np.arange(len(labels))
plt.figure(figsize=(20,25))
plt.barh(indexes[:50], list(values[-50:]), align="center")
plt.yticks(range(50), list(labels[-50:]), fontsize=15)
plt.xlabel("# of posts with at least 10 upvotes")
plt.ylabel("Sub name")
plt.xlim([0,200000])
plt.ylim([0,50])
plt.title("Subreddits where you're likely to be upvoted")
plt.show()


#only select subreddits with at least 5000 posts with a score of 1 or fewer upvotes
d = {k: v for k, v in downvoteSubreddits.items() if v > 5000}
sorted_x2 = sorted(d.items(), key=itemgetter(1))
labels, values = zip(*sorted_x2)
indexes = np.arange(len(labels))
plt.figure(figsize=(20,25))
plt.barh(indexes[:50], list(values[-50:]), align="center")
plt.yticks(range(50), list(labels[-50:]), fontsize=15)
plt.xlabel("# of posts with no upvotes")
plt.ylabel("Sub name")
plt.xlim([0,500000])
plt.ylim([0,50])
plt.title("Subreddits where you're unlikely to be upvoted")
plt.show()


#had to convert the sorted_x & sorted_x2 tuples into dicts to use with
#the common_entries function
sorted_x = dict((x, y) for x, y in sorted_x)
sorted_x2 = dict((x, y) for x, y in sorted_x2)

#convert two sorted dictionaries into a list of 3-tuples that are composed of 
#(a subreddit name, the number of 10+ upvoted posts, the number of <1 upvoted posts)
def common_entries(*dcts):
    for i in set(dcts[0]).intersection(*dcts[1:]):
        yield (i,) + tuple(d[i] for d in dcts)
common_entry_list = list(common_entries(sorted_x, sorted_x2))
#convert common_entry_list into new list of 2-tuples that contain the
#(subreddit name, ratio of upvoted posts/downvoted posts)
final_tuple = []
for entry in common_entry_list:
    final_tuple += [(entry[0], entry[1]/entry[2])]

    
#reconvert list of tuples into a dictionary for the final graph
d = dict((x, y) for x, y in final_tuple)
sorted_x_final = sorted(d.items(), key=itemgetter(1))
labels, values = zip(*sorted_x_final)
indexes = np.arange(len(labels))

plt.figure(figsize=(20,25))
plt.barh(indexes[:50], list(values[-50:]), align="center")
plt.yticks(range(50), list(labels[-50:]), fontsize=15)
plt.xlabel("Ratio of upvotes to non-upvotes")
plt.ylabel("Sub name")
plt.xlim([0,0.4])
plt.ylim([0,50])
plt.title("Subreddits with a high upvote/non-upvote ratio")
plt.show()


sorted_x_final = sorted(d.items(), key=itemgetter(1), reverse=True)
labels, values = zip(*sorted_x_final)
indexes = np.arange(len(labels))

plt.figure(figsize=(20,25))
plt.barh(indexes[:50], list(values[-50:]), align="center")
plt.yticks(range(50), list(labels[-50:]), fontsize=15)
plt.xlabel("Ratio of upvotes to non-upvotes")
plt.ylabel("Sub name")
plt.xlim([0,0.4])
plt.ylim([0,50])
plt.title("Subreddits with a low upvote/non-upvote ratio")
plt.show()


# What specific words are likely to result in downvotes? I determine this by tallying the associated score of each post for each individual word. Unfortunately the data only has the total score for each post. In the database, `ups` is the same as the net `score` value and the `downs` value is always 0.

# In[ ]:


import sqlite3, re, random
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import operator
from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
print('Querying DB...\n')
sql_conn = sqlite3.connect('../input/database.sqlite')

#select all posts that have a negative score and were not deleted
downvotedPosts = sql_conn.execute("SELECT body, score FROM May2015                                 WHERE score < 0 AND May2015.body NOT LIKE '%[deleted]%'                                LIMIT " + str(10000000))

#make dictionary containing postID, body, and score
downvotedPostDict = defaultdict(tuple)
for count, item in enumerate(downvotedPosts):
    #print(item[0], item[1])
    downvotedPostDict[count] += (item[0], item[1])

#remove most commonly used articles that don't tell us much
words_to_ignore = ['and','the','of','a','in','to','it','i','that','for','with','on','this','they','at','but','from','by','is','are','be','if','was','as','or','so']
#In Python, searching a set is much faster than searching a list, so convert the stop words to a set
stops = set(words_to_ignore) 

wordDict = defaultdict(int)

#process text
for post in downvotedPostDict:
    postdata = downvotedPostDict.get(post)
    #Remove non-letters        
    letters_only = re.sub("[^a-zA-Z]", " ", postdata[0]) 
    #Convert to lower case, split into individual words
    words = letters_only.lower().split()                             
    #Remove stop words and word fragments
    meaningful_words = [w for w in words if not w in stops and len(w)>2]   
    #for each word in body text, add the post's score to that words total score
    for word in meaningful_words: 
        wordDict[word] += postdata[1]
#print(wordDict)

#print the top 300 most negative words
sorted_neg = sorted(wordDict.items(), key=operator.itemgetter(1))
#print(sorted_neg[:300])


########### Same thing but for posts of any score, for comparison
#select all posts that were not deleted
downvotedPosts = sql_conn.execute("SELECT body, score FROM May2015                                 WHERE score > 0 AND May2015.body NOT LIKE '%[deleted]%'                                LIMIT " + str(10000000))

#make dictionary containing postID, body, and score
downvotedPostDict = defaultdict(tuple)
for count, item in enumerate(downvotedPosts):
    #print(item[0], item[1])
    downvotedPostDict[count] += (item[0], item[1])

wordDict = defaultdict(int)

#process text
for post in downvotedPostDict:
    postdata = downvotedPostDict.get(post)
    # Remove non-letters        
    letters_only = re.sub("[^a-zA-Z]", " ", postdata[0]) 
    #Convert to lower case, split into individual words
    words = letters_only.lower().split()                             
    #Remove stop words and word fragments
    meaningful_words = [w for w in words if not w in stops and len(w)>2]   
    #for each word in body text, add the post's score to that words total score
    for word in meaningful_words: 
        wordDict[word] += postdata[1]
#print(wordDict)

#get the top 300 most positive words
sorted_pos = sorted(wordDict.items(), key=operator.itemgetter(1), reverse=True)
#print(sorted_pos[:300])


#########Comparing the two lists

neg_word_list = [word[0] for word in sorted_neg[:300]]
pos_word_list = [word[0] for word in sorted_pos[:300]]

#only keep the top 300 negative words that also do not appear in the top 300 words
dissimilar_list = [word for word in neg_word_list if word not in pos_word_list]

print(dissimilar_list)


# I'd say the results match my expectations. The slurs and cursewords are likely to be downvoted on reddit. 'Hate' and 'kill' in one's post indicate a negative sentiment. 'Seriously', 'completely', and 'literally' can be used to express dissatisfaction in a more passive way (eg "That's seriously what you think?"). Posts concerning 'men', 'women', 'white', 'black', 'opinion', 'sex', 'god'. or the 'police' are divisive and tend to attract downvotes (but probably also a lot of upvotes). I have a hunch that 'pay', 'support', and 'kids' are related to paying child support, another divisive issue. The "oppositional" words ('stop', 'aren', 'dont', 'rather') imply that at least the person the user is responding too is already primed to downvote them. 'Internet', 'sub', and 'thread' are a byproduct of the place this data was gathered. Be sure to comment if you have theories for any of the rest of the words in the dissimilar list!
