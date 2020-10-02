from collections import Counter
import sqlite3, time, csv, re, random


sql_conn = sqlite3.connect('../input/database.sqlite')

link_regex = r'^\s*\[(.+?)\]\((.+?)\)\s*$'
def grab_link(body):
    '''
    Test And Grab one-liner links
    '''
    m = re.search(link_regex, body)
    if m is None:
        return None, None
    if m.group(1).lower() == m.group(2).lower():
        return None, None
    if len( m.group(1) ) > 25:
        return None, None
    # group 1 is link text, group 2 is url    
    return (m.group(1), m.group(2))
    
    
limit = 10000000
gilded = sql_conn.execute("SELECT subreddit, body, gilded, score FROM May2015 \
                            ORDER BY score DESC \
                            LIMIT " + str(limit))
                                

links = []
textCounts = Counter()
for post in gilded:
    text, url = grab_link(post[1])
    if text:
        textCounts[text] += 1
        links.append((text, url, post[3]))
        
        
print("Top")
print("=========================")
        
for (t,u,s) in links[1:20]:
    print("%s %d" % (t, s))
    print(u)
    
print("Top Texts")
print("=========================")
print( "\n".join( [a + " " + str(b) for (a,b) in textCounts.most_common(100)] ) )
