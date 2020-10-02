import sqlite3
import pandas as pd
import networkx as nx
from networkx import bipartite

sql_conn = sqlite3.connect('../input/database.sqlite')

# take a random sample of 5,000,000 posts
sql_cmd = 'SELECT subreddit, author FROM May2015 WHERE author != "[deleted]" ORDER BY Random() LIMIT 5000000'
c = sql_conn.cursor()

c.execute(sql_cmd)

subreddits = set()
B = nx.Graph() # bipartite graph on subreddits |_| authors

for subreddit, author in c:
    B.add_edge(subreddit, author)
    subreddits.add(subreddit)

# calculate Jaccard similarity graph on subreddits    
G = bipartite.overlap_weighted_projected_graph(B, subreddits, jaccard=True)
nx.write_gpickle(G,'jaccard_graph_5million.pkl')
    
#data['body'] = data.body.apply(lambda s: s.encode('ascii','ignore'))
#data.to_csv('sc.csv')