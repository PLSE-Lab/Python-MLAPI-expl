#!/usr/bin/env python
# coding: utf-8

# This notebook shows how to build and run Clusterig Travel Sanat solver, by runnig Concorde and LKH on each cluster to find best path in each cluster.

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import sympy


# ## Build concorde

# In[ ]:


get_ipython().run_cell_magic('bash', '-e', "if ! [[ -f ./linkern ]]; then\n  wget http://www.math.uwaterloo.ca/tsp/concorde/downloads/codes/src/co031219.tgz\n  echo 'c3650a59c8d57e0a00e81c1288b994a99c5aa03e5d96a314834c2d8f9505c724  co031219.tgz' | sha256sum -c\n  tar xf co031219.tgz\n  (cd concorde && CFLAGS='-Ofast -march=native -mtune=native -fPIC' ./configure)\n  (cd concorde/LINKERN && make -j && cp linkern ../../)\n  rm -rf concorde co031219.tgz\nfi")


# # Build LKH

# In[ ]:


# %%bash -e
# wget http://akira.ruc.dk/~keld/research/LKH/LKH-2.0.9.tgz
# tar xvfz LKH-2.0.9.tgz
# cd LKH-2.0.9
# make


# # Write TSP for Concorde

# In[ ]:


def write_tsp(cities, filename, dim, name='traveling-santa-2018-prime-paths'):
    with open(filename, 'w') as f:
        f.write('NAME : %s\n' % name)
        f.write('COMMENT : %s\n' % name)
        f.write('TYPE : TSP\n')
        f.write('DIMENSION : %d\n' % dim)
        f.write('EDGE_WEIGHT_TYPE : EUC_2D\n')
        f.write('NODE_COORD_SECTION\n')
        for row in cities.itertuples():
            f.write('%d %.11f %.11f\n' % (row.idx, row.X, row.Y))
        f.write('EOF\n')


# # Write TSP for LKH

# In[ ]:


# def write_tsp1(cities, filename,dim, name='traveling-santa-2018-prime-paths'):
#     with open("../working/LKH-2.0.9/{0}".format(filename), 'w') as f:
#         f.write('NAME : %s\n' % name)
#         f.write('COMMENT : %s\n' % name)
#         f.write('TYPE : TSP\n')
#         f.write('DIMENSION : %d\n' % dim)
#         f.write('EDGE_WEIGHT_TYPE : EUC_2D\n')
#         f.write('NODE_COORD_SECTION\n')
#         for row in cities.itertuples():
#             f.write('%d %.11f %.11f\n' % (row.idx, row.X, row.Y))
#         f.write('EOF\n')


# In[ ]:


# def write_parameters(filename):
#     parameters = [
#     ("PROBLEM_FILE", "{0}.tsp\n".format(filename)),
#     ("OUTPUT_TOUR_FILE", "{0}_sol.csv\n".format(filename)),
#     ("SEED", 2018),
#     ('CANDIDATE_SET_TYPE', 'POPMUSIC'), #'NEAREST-NEIGHBOR', 'ALPHA'),
#     ('INITIAL_PERIOD', 1000),
#     ('MAX_TRIALS', 1000),
#     ]
#     with open("../working/LKH-2.0.9/{0}.par".format(filename), 'w') as f:
#         for param, value in parameters:
#             f.write("{} = {}\n".format(param, value))
#     #print("Parameters saved as", filename)


# In[ ]:


cities = pd.read_csv('../input/cities.csv')
cities['idx'] = cities.index + 1 
cities.head()


# In[ ]:


def plot_tour(tour, tg, cmap=mpl.cm.gist_rainbow):
    fig, ax = plt.subplots(figsize=(25, 25))
    ind = tour
    plt.plot(tg.X[ind], tg.Y[ind], linewidth=1)


# # Scale input

# In[ ]:


cities1k = cities
cities1k.X = cities.X * 1000
cities1k.Y = cities.Y * 1000


# # Clustering cities by Kmeans into 36 cluster

# In[ ]:


# Kmeans
from sklearn.cluster import MiniBatchKMeans,Birch
coords = np.vstack((cities1k.X.values,cities1k.Y.values)).T
sample_ind = np.random.permutation(len(coords))
kmeans = MiniBatchKMeans(n_clusters = 36, batch_size = 50).fit(coords[sample_ind])
cities1k.loc[:, 'kmeans']   = kmeans.predict(cities1k[['X', 'Y']])


# # Clustering cities by GMM into 36 cluster

# In[ ]:


# GMM
from sklearn.mixture import GaussianMixture
mclusterer = GaussianMixture(n_components=36, tol=0.01, random_state=66, verbose=1)
cities['gmm'] = mclusterer.fit_predict(cities[['X', 'Y']].values)
nmax = cities.gmm.max()
print("{} clusters".format(nmax+1))


# # Plot Clustering Results on Santa cities

# In[ ]:


plt.style.use('seaborn')
fig, ax = plt.subplots(ncols=1, nrows=2,figsize=(15, 5))
plt.subplot(1,2,1)
plt.scatter(cities1k.X.values, cities1k.Y.values,c=cities1k.gmm.values,s=0.3, cmap='nipy_spectral', alpha=0.9)
plt.subplot(1,2,2)
plt.scatter(cities1k.X.values, cities1k.Y.values,c=cities1k.kmeans.values,s=0.3, cmap='nipy_spectral', alpha=0.9)
plt.show()


# # Prepare .tsp files for Concorde & LKH and plot each cluster seperatly - Kmeans

# In[ ]:


# Concorde on Kmeans
plt.style.use('seaborn')
fig, ax = plt.subplots(ncols=6, nrows=6,figsize=(15, 15))
cmap=mpl.cm.nipy_spectral

for i in range(cities1k.kmeans.max()+1):
    citiesk = cities1k[cities1k.kmeans == i]
    citiesk = citiesk.reset_index(drop=True)
    citiesk['idx'] = citiesk.index + 1
    dim = len(citiesk)
    #citiesk.to_csv('citieskm{0}.csv'.format(i),index=False)
    write_tsp(citiesk, 'citieskm{0}.tsp'.format(i),dim)
#     write_tsp1(citiesk, 'citieskm{0}.tsp'.format(i),dim)
#     write_parameters('citieskm{0}'.format(i))
    plt.subplot(6,6,i+1)
    plt.scatter(citiesk.X.values, citiesk.Y.values,s=0.5,color=cmap(i), alpha=0.99)  
    plt.title(i)
    plt.xticks([])
    plt.yticks([])


# # Prepare .tsp files for Concorde & LKH and plot each cluster seperatly - GMM

# In[ ]:


# Concorde on GMM
plt.style.use('seaborn')
fig, ax = plt.subplots(ncols=6, nrows=6,figsize=(15, 15))
cmap=mpl.cm.nipy_spectral

for i in range(cities1k.kmeans.max()+1):
    citiesk = cities1k[cities1k.gmm == i].reset_index()
    citiesk['idx'] = citiesk.index + 1
    dim = len(citiesk)
    #citiesk.to_csv('citiesgmm{0}.csv'.format(i),index=False)
    write_tsp(citiesk, 'citiesgmm{0}.tsp'.format(i),dim)
#     write_tsp1(citiesk, 'citiesgmm{0}.tsp'.format(i),dim)
#     write_parameters('citiesgmm{0}'.format(i))
    plt.subplot(6,6,i+1)
    plt.scatter(citiesk.X.values, citiesk.Y.values,s=0.5,color=cmap(i), alpha=0.99)  
    plt.title(i)
    plt.xticks([])
    plt.yticks([])


# In[ ]:


#!cd LKH-2.0.9 && ls


# In[ ]:


print(cities1k.kmeans.max())
print(cities1k.gmm.max())


# # Run Concorde on each Kmeans clusters 

# In[ ]:


get_ipython().run_cell_magic('bash', '', 'for i in {0..36}\n    do\n    echo $i\n    time ./linkern -K 1 -s 42 -S linkernkm$i.tour -R 999999999 -t 3 ./citieskm$i.tsp >linkernkm$i.log\n    done')


# # Run Concorde on each GMM clusters 

# In[ ]:


get_ipython().run_cell_magic('bash', '', 'for i in {0..36}\n    do\n    echo $i\n    time ./linkern -K 1 -s 42 -S linkerngmm$i.tour -R 999999999 -t 3 ./citiesgmm$i.tsp >linkerngmm$i.log\n    done')


# In[ ]:


#!cat ./LKH-2.0.9/citiesgmm4_sol.csv


# # Run LKH on each GMM clusters 

# In[ ]:


# %%bash
# cd ./LKH-2.0.9
# for i in {0..64}
#     do
#     echo $i
#     timeout 20s ./LKH citieskm$i.par
#     done


# In[ ]:


def from_file(filename):  # from linkern's output or csv
    seq = [int(x) for x in open(filename).read().split()[1:]]
    return (seq if seq[-1] == 0 else (seq + [0]))


# In[ ]:


# def read_tour(filename):
#     tour = []
#     for line in open(filename).readlines():
#         line = line.replace('\n', '')
#         try:
#             tour.append(int(line) - 1)
#         except ValueError as e:
#             pass  # skip if not a city id (int)
#     return tour[:-1]


# # Plot Concord solution for every Kmeans cluster

# In[ ]:


plt.style.use('seaborn')
fig, ax = plt.subplots(ncols=1, nrows=1,figsize=(13, 9))
cmap=mpl.cm.nipy_spectral

for i in range(cities1k.kmeans.max()+1):
    citiesk = cities1k[cities1k.kmeans == i].reset_index()
    citiesk['idx'] = citiesk.index + 1
    tour = from_file('linkernkm{0}.tour'.format(i))
 #   plt.subplot(10,10,i+1)
    plt.plot(citiesk.X[tour], citiesk.Y[tour], linewidth=1)  
 #   plt.title(i)
 #   plt.xticks([])
 #   plt.yticks([])
plt.show()


# # Plot Concord solution for every GMM cluster

# In[ ]:


plt.style.use('seaborn')
fig, ax = plt.subplots(ncols=1, nrows=1,figsize=(13, 9))
cmap=mpl.cm.nipy_spectral

for i in range(cities1k.gmm.max()+1):
    citiesk = cities1k[cities1k.gmm == i].reset_index()
    citiesk['idx'] = citiesk.index + 1
    tour = from_file('linkerngmm{0}.tour'.format(i))
 #   plt.subplot(10,10,i+1)
    plt.plot(citiesk.X[tour], citiesk.Y[tour], linewidth=1)  
 #   plt.title(i)
 #   plt.xticks([])
 #   plt.yticks([])
plt.show()


# # Plot LKH solution for every GMM cluster

# In[ ]:


# plt.style.use('seaborn')
# fig, ax = plt.subplots(ncols=1, nrows=1,figsize=(13, 9))
# cmap=mpl.cm.nipy_spectral

# for i in range(cities1k.gmm.max()+1):
#     citiesk = cities1k[cities1k.gmm == i].reset_index()
#     citiesk['idx'] = citiesk.index + 1    
#     tour = read_tour('../working/LKH-2.0.9/citiesgmm{0}_sol.csv'.format(i))
#  #   plt.subplot(10,10,i+1)
#     plt.plot(citiesk.X[tour], citiesk.Y[tour], linewidth=1)  
#  #   plt.title(i)
#  #   plt.xticks([])
#  #   plt.yticks([])
# plt.show()


# In[ ]:


#@staticmethod
def score(cities, tour):
    penalized = ~cities.CityId.isin(sympy.primerange(0, len(cities)))
    df = cities.reindex(tour)
    dist = np.hypot(df.X.diff(-1), df.Y.diff(-1))
    penalty = 0.1 * dist[9::10] * penalized[tour[9::10]]
    return dist.sum() + penalty.sum()


# # Show sum of all Concord solution's scores for every Kmeans cluster

# In[ ]:


scoretotal = 0
for i in range(cities1k.kmeans.max()+1):
    citiesk = cities1k[cities1k.kmeans == i].reset_index()
    citiesk['idx'] = citiesk.index + 1
    tour = from_file('linkernkm{0}.tour'.format(i))
    scorei = score(citiesk,tour)
    scorei = scorei/1000
    scoretotal = scorei + scoretotal
print(scoretotal)


# # Show sum of all Concord solution's scores for every GMM cluster

# In[ ]:


scoretotal = 0
for i in range(cities1k.kmeans.max()+1):
    citiesk = cities1k[cities1k.kmeans == i].reset_index()
    citiesk['idx'] = citiesk.index + 1
    tour = from_file('linkerngmm{0}.tour'.format(i))
    scorei = score(citiesk,tour)
    scorei = scorei/1000
    scoretotal = scorei + scoretotal
print(scoretotal)


# # Show sum of all LKH solution's scores for every GMM cluster

# In[ ]:


# scoretotal = 0
# for i in range(cities1k.gmm.max()+1):
#     citiesk = cities1k[cities1k.kmeans == i].reset_index()
#     citiesk['idx'] = citiesk.index + 1
#     tour = read_tour('../working/LKH-2.0.9/citiesgmm{0}_sol.csv'.format(i))
#     scorei = score(citiesk,tour)
#     scorei = scorei/1000
#     scoretotal = scorei + scoretotal
# print(scoretotal)


# In[ ]:


# tour.to_csv('submission.csv')
# tour.score()

