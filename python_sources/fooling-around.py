# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import sqlite3, numpy


subreddits = ['relationships', 'sex', 'gonewild', 'TwoXChromosomes', 'TheRedPill', 'GWCouples', 'gwcumsluts', 'polyamory']

# With '% ...%' clause we search for words beginning with given keywords 
# For example, 'divorc' will result in divorce, divorces, divorced, divorcing and so on
# 'Sex' will result in sex, sexy, sexual and much more, but NOT in unisex
# It's far from perfect, but still can give pretty plausible results

keywords =  {
                'love':         ['love', 'loving'],
                'dating':       ['date', 'dating'],
                'sex':          ['sex'],
                'marriage':     ['marriage', 'marry', 'marrie'],
                'cheating':     ['cheat'],
                'divorce':      ['divorc'],
                'money':        ['money', 'rich', 'wealth'],
                'women':        ['women', 'girl', 'woman', 'female'],
                'men'  :        ['men', 'boy', 'man', 'male'],
                'whore':        ['whor', 'slut'],
                'pig'  :        ['pig'],
                'pussy':        ['pussy', 'vagina'],
                'anal' :        ['anal', 'assh'],
                'skinny':       ['skinny', 'thin', 'lean', 'athletic'],
                'fat'   :       ['fat', 'chubby', 'overweight'],
                'cow'   :       ['cow'],
                'penis' :       ['penis', 'dick', 'cock', 'balls'],
                'cunt'  :       ['cunt'],
                'beautiful' :   ['beautiful', 'pretty', 'gorge', 'stunn'],
                'sexy'      :   ['sexy', 'hot'],
                'handsome' :    ['handsome'],
                'porn'    :     ['porn'],
                'butt'  : ['butt', 'ass', 'tush'],
                'boobs'  : ['tit', 'boob', 'knocker', 'rack'],
                'Sarah'  : ['Sarah'],
                'Chris'  : ['Chris'],
                'Rape'   : ['rape'],
                'Swinging' : ['swinging', 'open', 'play'],
                'cum'   : ['cum', 'jiz', 'splo'],
                'masturbation' : [ 'masturb'],
                'monogamy' : ['monogamy', 'non-monogamy', 'nonmonogamy', 'non monogamy']
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

def sort(a):
    return a['mean']


def main ():
    
    for subreddit in subreddits:
        print ("Exploring r/{subreddit}:".format(subreddit = subreddit))
        
        results = []
        for key, words in keywords.items():
            result = calculate (subreddit, words)
            result['key'] = key
            results.append(result)
            
        sorted_results = sorted(results, key=sort)
        
        for result in sorted_results:
            line = "{p:.1f}% of comments contain '{key}' or its derivatives | score mean: {mean:.1f}, score std: {std:.1f}"
            print (line.format(p = result['frac'] * 100, key = result['key'], mean = result['mean'], std = result['std']))
        print ()
    


# Entry point    
    
main ()
    