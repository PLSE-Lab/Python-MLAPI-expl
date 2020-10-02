#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import sympy

import threading
import time
import subprocess


# In[ ]:


get_ipython().run_cell_magic('bash', '-e', "if ! [[ -f ./linkern ]]; then\n  wget http://www.math.uwaterloo.ca/tsp/concorde/downloads/codes/src/co031219.tgz\n  echo 'c3650a59c8d57e0a00e81c1288b994a99c5aa03e5d96a314834c2d8f9505c724  co031219.tgz' | sha256sum -c\n  tar xf co031219.tgz\n  (cd concorde && CFLAGS='-O3 -march=native -mtune=native -fPIC' ./configure)\n  (cd concorde/LINKERN && make -j && cp linkern ../../)\n  rm -rf concorde co031219.tgz\nfi")


# In[ ]:


def write_tsp(cities, filename, name='traveling-santa-2018-prime-paths'):
    with open(filename, 'w') as f:
        f.write('NAME : %s\n' % name)
        f.write('COMMENT : %s\n' % name)
        f.write('TYPE : TSP\n')
        f.write('DIMENSION : %d\n' % len(cities))
        f.write('EDGE_WEIGHT_TYPE : EUC_2D\n')
        f.write('NODE_COORD_SECTION\n')
        for row in cities.itertuples():
            f.write('%d %.11f %.11f\n' % (row.Index+1, row.X, row.Y))
        f.write('EOF\n')
        
def read_tour(filename):
    tour = open(filename).read().split()[1:]
    tour = list(map(int, tour))
    if tour[-1] == 0: tour.pop()
    return tour

def score_tour(tour):
    df = cities.reindex(tour + [0]).reset_index()
    primes = list(sympy.primerange(0, len(cities)))
    df['prime'] = df.CityId.isin(primes).astype(int)
    df['dist'] = np.hypot(df.X - df.X.shift(-1), df.Y - df.Y.shift(-1))
    df['penalty'] = df['dist'][9::10] * (1 - df['prime'][9::10]) * 0.1

    #Chippy's hint
    sum_dist = df['dist'].sum()
    s= sum_dist + df['penalty'].sum()
    df = df.iloc[::-1]
    df['penalty'] = df['dist'][9::10] * (1 - df['prime'][9::10]) * 0.1
    
    r= sum_dist + df['penalty'].sum()
    rev = False
    if r < s:
        rev = True
        s = r

    return s , rev
        

def write_submission(tour, filename):
    assert set(tour) == set(range(len(tour)))
    pd.DataFrame({'Path': list(tour) + [0]}).to_csv(filename, index=False)
    
class LinkernBatch(object):

    def __init__(self, seconds=60):
        self.output = None
        self.error = None
        self.running = False
        self.seconds = seconds
        thread = threading.Thread(target=self.run, args=())
        thread.daemon = True
        thread.start()  
        
            
    def run(self):
        self.running = True
        try:
            print("start batch")
            bash= "nice ./linkern -s 42 -S linkern.tour -R 1000000000 -t {} ./cities1k.tsp >linkern.log".format(self.seconds)
            process = subprocess.Popen(bash.split(), stdout=subprocess.PIPE)
            self.output, self.error = process.communicate()
        except:
            None
        self.running = False            
        
        print("end batch error:",self.error ) 


# In[ ]:


cities = pd.read_csv('../input/cities.csv', index_col=['CityId'])
cities1k = cities * 1000        
        
write_tsp(cities1k, 'cities1k.tsp')

linkern = LinkernBatch(seconds=20000)

start_time = time.time()
best_score = 9999999
best_tour = None
time.sleep(1)

print ("start monitoring")
while linkern.running:
    try :
        tour = read_tour('linkern.tour')
        if len(list(tour))== 197769:
            score, rev  = score_tour(tour)
            if rev :
                tour = tour [::-1]
            elapsed_time = time.time() - start_time
            if score < best_score : 
                best_score = score
                best_tour = tour 
                print("best_score: {}, elapsed time: {}, rev: {}".format(best_score,int(elapsed_time), rev))
    except:
        None
    time.sleep(1)
print ("end monitoring")
    
last_tour = read_tour('linkern.tour')
last_score, rev = score_tour(tour)   
print("last_score: {}, rev: {}".format(last_score, rev))

if not best_tour is None:   
    write_submission(best_tour, 'submission.csv')
    print("best_score: {}".format(best_score))

