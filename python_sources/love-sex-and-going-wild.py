'''
In this script we explore usage of certain keywords in popular subreddits dedicated to relationships and intergender dynamics.
Currently participating subreddits: Relationships, Sex, GoneWild, TwoXChromosomes, TheRedPill.
The last two were included for their controversial nature and steady supply of reddit drama.
(c) 2015 Anton Matrosov
'''


import sqlite3, numpy


subreddits = ['india']

# With '% ...%' clause we search for words beginning with given keywords 
# For example, 'divorc' will result in divorce, divorces, divorced, divorcing and so on
# 'Sex' will result in sex, sexy, sexual and much more, but NOT in unisex
# It's far from perfect, but still can give pretty plausible results

keywords =  {
                'left':         ['leftism', 'leftist'],
                'right':       ['right', 'right-wing'],
                'liberal':          ['liberal'],
                'conservative':     ['conservative', 'tradition', 'culture'],
                'politics':         ['political','agenda']
            }
            
            
sql = sqlite3.connect ('../input/database.sqlite').cursor()



# Creates query string to find all comments containing any given keyword within given subreddit
# The comments are represented by their scores, since that's what we are interested in

def query (subreddit = '', keywords = []):
    
    base = "SELECT score FROM May2015"
    sub = ""
    keys = ""
    
    if subreddit != '':
        sub = "subreddit = '{}' ".format(subreddit)
        
    if len(keywords) > 0:
        keys = "(body LIKE "
        i = 0
        for key in keywords:
            keys += "'% {}%'".format(key)
            if i < len(keywords) - 1:
                keys += " OR body LIKE "
            else:
                keys += ")"
            i += 1
            
    if subreddit != '' or len(keywords) > 0:
        base += " WHERE "
        
    if subreddit != '' and len(keywords) > 0:
        sub += "AND "
            
    return base + sub + keys



# Calculates statistical data for comments containing given keywords within given subreddit
    
def calculate (subreddit = '', keywords = []):
    
    allData = sql.execute (query (subreddit)).fetchall()
    keyData = sql.execute (query (subreddit, keywords)).fetchall()
    
    frac = len (keyData) / len (allData)
    mean = numpy.mean (keyData)
    std = numpy.std (keyData)
    
    return {'frac': frac, 'mean': mean, 'std': std}
    
    

# Main routine
# Calculating statistical data for all 5 subreddits and 7 keywords

def main ():
    
    for subreddit in subreddits:
        print ("Exploring r/{subreddit}:".format(subreddit = subreddit))
        for key, words in keywords.items():
            result = calculate (subreddit, words)
            line = "{p:.1f}% of comments contain '{key}' or its derivatives | score mean: {mean:.1f}, score std: {std:.1f}"
            print (line.format(p = result['frac'] * 100, key = key, mean = result['mean'], std = result['std']))
        print ()
        
    print ("The next step is to build 'keyword portraits' for these subreddits and compile score histograms")
    


# Entry point    
    
main ()
    
    