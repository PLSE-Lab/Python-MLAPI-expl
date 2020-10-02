import sqlite3
import random
import pandas as pd
import numpy as np
from math import sqrt

# I foolow the recomandation described in the blog post http://i83.co.uk/blog/netflix-recommendation-engine-python/

sql_conn = sqlite3.connect('../input/database.sqlite')


df = pd.read_sql("""
    WITH topSub AS (
        SELECT
            subreddit
        FROM May2015
        GROUP BY subreddit
        ORDER BY count(*) DESC
        LIMIT 20
    )
    SELECT
      author
      , subreddit
      , count(*) as nbComment
    FROM May2015
    WHERE
       subreddit NOT IN topsub 
    AND
       author IN (
            SELECT
                author
            FROM
                May2015
            WHERE
                subreddit NOT IN topsub
            GROUP BY author
            HAVING count(DISTINCT subreddit)  > 15
            AND count(DISTINCT parent_id) < 3000
            LIMIT 50000)
    GROUP BY author, subreddit""", sql_conn)
    
def retro_dictify(frame):
    d = {}
    for row in frame.values:
        here = d
        for elem in row[:-2]:
            if elem not in here:
                here[elem] = {}
            here = here[elem]
        here[row[-2]] = row[-1]
    return d

#create a train set and test set
testDict = {}
dataDict = retro_dictify(df)
testSize = int(pow(len(dataDict),0.5))
rows = random.sample(dataDict.keys(), testSize)
for row in rows:
    testDict[row] = dataDict[row]
    del dataDict[row]

# Returns the Pearson correlation coefficient for p1 and p2
def sim_pearson(prefs,p1,p2):
	# Get the list of mutually rated items
	si={}
	for item in prefs[p1]:
		if item in prefs[p2]: si[item]=1
 
	# Find the number of elements
	n= float(len(si))
 
	# If they have no ratings in common, return 0
	if n==0: return 0
 
	# Add up all the preferences
	sum1=sum([prefs[p1][it] for it in si])
	sum2=sum([prefs[p2][it] for it in si])
 
	# Sum up the squares
	sum1Sq=sum([pow(prefs[p1][it],2) for it in si])
	sum2Sq=sum([pow(prefs[p2][it],2) for it in si])
 
	# Sum up the products
	pSum=sum([prefs[p1][it]*prefs[p2][it] for it in si])
 
	# Calculate Pearson score
	num=pSum-(sum1*sum2/n)
	den=sqrt((sum1Sq-pow(sum1,2)/n)*(sum2Sq-pow(sum2,2)/n))
	if den==0: return 0
 
	r=num/den
	return r
	
# Returns the best matches for person from the prefs dictionary
# Number of results and similarity function are optional params
def topMatches(prefs,person,n=10,similarity=sim_pearson):
	scores=[(similarity(prefs,person,other),other)
		for other in prefs if other!=person]
 
	# Sort the list so highest scores appear at the top
	scores.sort()
	scores.reverse()
	return scores[0:n]
	
# Gets recommendations for a person by using a weighted average of every other user's rankings
def getRecommendations(prefs,person,similarity=sim_pearson):
	totals={}
	simSums={}
	for other in prefs:
		# don't compare to self
		if other==person: continue
		sim=similarity(prefs,person,other)
 
		# ignore scores of zero or lower
		if sim<=0:continue
		for item in prefs[other]:
 
			# only score movies I haven't seen yet
			if item not in prefs[person] or prefs[person][item]==0:
				# Similarity * score
				totals.setdefault(item,0)
				totals[item]+=prefs[other][item]*sim
				#Sum of similarities
				simSums.setdefault(item,0)
				simSums[item]+=sim
 
		# Create the normalised list
		rankings=[(total/simSums[item],item) for item,total in totals.items()]
 
		# Return the sorted list
		rankings.sort()
		rankings.reverse()
		return rankings[0:10]

def reset(percent=50):
    return random.randrange(100) < percent

columns=('precision', 'recall', 'RMSE', 'MAE', 'ARHR')
i = 0
evaluations = pd.DataFrame(columns=columns)
for author in testDict:
    test = {}
    test[author] = testDict[author]
    y = {}
    
    for sub in test[author]:
        if reset(40) and 3*len(y) < len(test[author]):
            y[sub] = test[author][sub]
    
    for sub in y:
        del test[author][sub]
        
    dataDict[author] = test[author]
    recomandations = getRecommendations(dataDict,author,similarity=sim_pearson)

    precision = 0
    recall = 0
    RMSE = 0
    MAE = 0
    ARHR = 0
    err = []
    
    print("\n" + author)
    print(test[author])
    print("Y:" + str(y))
    print("recomandations:" + str(recomandations))
        
    if recomandations:
        for rank in range(0,len(recomandations)):
            if recomandations[rank][1] in y.keys():
                err.append(recomandations[rank][0] - y[recomandations[rank][1]])
                ARHR += 1/(rank+1)
                
        precision = len(err) / len(recomandations)
        recall = len(err) / len(y)
        RMSE = pow(np.mean([pow(x,2) for x in err]),.5) if err else 0 
        MAE = pow(np.mean([abs(x) for x in err]),.5) if err else 0 
        ARHR = ARHR/len(err) if err else 0 
    
    evaluations.loc[i] = [precision, recall, RMSE, MAE, ARHR]
    print(evaluations.loc[i])
    i += 1
    del dataDict[author]
