import random
from collections import defaultdict
random.seed(42)
import pandas as pd
import sqlite3

# Get our data
import sqlite3
con = sqlite3.connect('../input/database.sqlite')
sample = pd.read_sql_query("""
SELECT *
FROM Papers""", con)
s = " ".join(sample["Title"].values.flatten()).split()

# Build our model
markov = defaultdict(lambda:defaultdict(int))
chains = [(s[i],s[i+1]) for i in range(len(s)-1)]
for chain in chains:
  markov[chain[0]][chain[1]] += 1

for title in range(100):
  # Generate 
  out = []
  base = random.choice(s)
  for i in range(random.randint(5,12)):
    out.append( base ) 
    choices = markov[base]
    if len(choices) > 0: 
      # weighted random choice
      base = random.choice([k for k in choices for dummy in range(choices[k])])
    else:
      # continue with a random token
      base = random.choice(s)
		
  # Output
  print(" ".join(out))
