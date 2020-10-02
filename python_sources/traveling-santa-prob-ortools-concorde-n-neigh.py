#!/usr/bin/env python
# coding: utf-8

# # Intro

# This uses the dataset from [Traveling Santa competition](https://www.kaggle.com/c/traveling-santa-2018-prime-paths)
# 
# **Overview**
# 
# We're given a set of coordinates and asked to find the optimal path through them. There is one twist, the scorer will impose a 10% penalty on every 10th jump if the cityId is not a prime number.
# 
# **Why This Problem is Interesting**
# 
# First, we don't see a lot of [Traveling Salesmen](https://en.wikipedia.org/wiki/Travelling_salesman_problem) problems on Kaggle, so it's good to see one. It's also got almost 200K points, so it's not going to fit into many TSP solvers out of the box. The jump penalty is a good twist, will likely make the winner but is not so harsh that you'll fail horribly if you ignore it.
# 
# **Solution Approach**
# 
# Fist we create a baseline solution using a greedy nearest neighbor algorithm. This will kind of suck. Then we'll dobetter by breaking the problem into subproblems for [Google OR-Tools](https://developers.google.com/optimization/routing/routing_options) to solve. This will be better, but not optimial. Last we will use [pyconcorde](https://github.com/jvkersch/pyconcorde) out of the box to find our best solution.

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#load-data" data-toc-modified-id="load-data-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>load data</a></span></li><li><span><a href="#scoring-routine" data-toc-modified-id="scoring-routine-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>scoring routine</a></span></li><li><span><a href="#Greedy-nearest-neighbor" data-toc-modified-id="Greedy-nearest-neighbor-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Greedy nearest neighbor</a></span></li><li><span><a href="#Google-OR-Tools" data-toc-modified-id="Google-OR-Tools-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Google OR-Tools</a></span><ul class="toc-item"><li><span><a href="#Get-Subproblems" data-toc-modified-id="Get-Subproblems-4.1"><span class="toc-item-num">4.1&nbsp;&nbsp;</span>Get Subproblems</a></span></li><li><span><a href="#OR-Tools-helpers" data-toc-modified-id="OR-Tools-helpers-4.2"><span class="toc-item-num">4.2&nbsp;&nbsp;</span>OR Tools helpers</a></span></li><li><span><a href="#Find-Route-through-Clusters" data-toc-modified-id="Find-Route-through-Clusters-4.3"><span class="toc-item-num">4.3&nbsp;&nbsp;</span>Find Route through Clusters</a></span></li><li><span><a href="#Solve-the-problem" data-toc-modified-id="Solve-the-problem-4.4"><span class="toc-item-num">4.4&nbsp;&nbsp;</span>Solve the problem</a></span></li></ul></li><li><span><a href="#Concorde" data-toc-modified-id="Concorde-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Concorde</a></span></li></ul></div>

# ## load data

# In[ ]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore",category=DeprecationWarning)


# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from scipy.spatial.distance import cdist, euclidean


# In[ ]:


df_cities = pd.read_csv('../input/cities.csv')
df_cities.tail()


# ## scoring routine 

# In[ ]:


def isPrime(n):
    #from https://stackoverflow.com/questions/4114167/checking-if-a-number-is-a-prime-number-in-python
    from itertools import count, islice
    return n > 1 and all(n%i for i in islice(count(2), int(np.sqrt(n)-1)))

def total_distance(path):
    #initialize counters
    prev_city_num = path[0]
    total_distance = 0
    step_num = 1
    for city_num in path[1:]:
        #get distance
        new_city = (df_cities.X[city_num], df_cities.Y[city_num])
        prev_city = (df_cities.X[prev_city_num], df_cities.Y[prev_city_num])
        distance = euclidean(new_city, prev_city)
        
        #check for 10% penalty
        if step_num % 10 == 0 and not isPrime(prev_city_num):
            distance = distance * 1.1
        
        total_distance += distance
        
        #increment counters
        prev_city_num = city_num
        step_num = step_num + 1
    return total_distance


# ## Greedy nearest neighbor

# Our baseline algorithm is just a greedy nearest neighbor alog. Starting with origin, we'll find the nearest neighbor and go to it. Then we repeat until we run out of points, when we return to the origin from wherever we end up.
# 
# This isn't expected to be competitive because we'll likely end up in many dead ends and have to make large jumps, especially at the end.

# In[ ]:


get_ipython().run_cell_magic('time', '', 'def nearest_neighbour(cities):\n    ids = cities.CityId.values[1:]\n    xy = np.array([cities.X.values, cities.Y.values]).T[1:]\n    path = [0,]\n    while len(ids) > 0:\n        last_x, last_y = cities.X[path[-1]], cities.Y[path[-1]]\n        last_point = [[last_x, last_y]]\n        nearest_index = cdist(last_point, xy, metric=\'euclidean\').argmin()\n        \n        path.append(ids[nearest_index])\n        ids = np.delete(ids, nearest_index, axis=0)\n        xy = np.delete(xy, nearest_index, axis=0)\n    path.append(0)\n    return path\n\nnnpath = nearest_neighbour(df_cities)\n\nprint(f"Total Distance: {total_distance(nnpath):.1f}")\ndf_path = pd.DataFrame({\'CityId\':nnpath}).merge(df_cities,how = \'left\')\nfig, ax = plt.subplots(figsize=(10,7))\nax.plot(df_path[\'X\'], df_path[\'Y\'])')


# ## Google OR-Tools

# This is too big of a problem to just fit into OR-tools routing engine. So we're going to cluster the points and solve the route through the clusters, then solve the routes through individual clusters. This solution improves marginally on the work of [jpmiller](https://www.kaggle.com/jpmiller/google-or-tools-w-clusters)

# ### Get Subproblems

# In[ ]:


from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture

n_clusters = 445
#more clusters == worse solution, fewer resources
#my experience:
#   - 1000 clusters = 1593976
#   - 445  clusters = 1585990
#   - 100  clusters = 1577067

num_iterations_per_solve = 250
#higher is gives a better solution, but I've set it low so this kernel solves quickly. 
#500 is a pretty good value in my exp


# In[ ]:


get_ipython().run_cell_magic('time', '', 'cities = df_cities.copy()\nmclusterer = GaussianMixture(n_components=n_clusters)\ncities[\'cluster\'] = mclusterer.fit_predict(cities[[\'X\', \'Y\']].values)\ncities[\'cluster_\'] = cities[\'cluster\'].astype(str) + "_"\n\nplt.figure(figsize=(10,7))\nclusters = sns.scatterplot(x=cities.X, y=cities.Y, alpha = 0.1, marker=\'.\', hue=cities.cluster_, legend=False)')


# In[ ]:


#plot number of points in each cluster
plt.suptitle('Points Per Cluster')
ax = cities.groupby('cluster')['CityId'].count().hist()


# ### OR Tools helpers

# In[ ]:


from scipy.spatial.distance import pdist, squareform
from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2

#%% functions
def create_mat(df):
    mat = pdist(locations, 'euclidean')
    return squareform(mat)

def create_distance_callback(dist_matrix):
    def distance_callback(from_node, to_node):
      return int(dist_matrix[from_node][to_node])
    return distance_callback

def optimize(df, startnode=None, stopnode=None):     
    num_nodes = df.shape[0]
    dist_matrix = create_mat(df)
    routemodel = pywrapcp.RoutingModel(num_nodes, 1, [startnode], [stopnode])
    
    dist_callback = create_distance_callback(dist_matrix)
    routemodel.SetArcCostEvaluatorOfAllVehicles(dist_callback)
    search_parameters = pywrapcp.RoutingModel.DefaultSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    search_parameters.solution_limit = num_iterations_per_solve
    
    assignment = routemodel.SolveWithParameters(search_parameters)
    # print(f"Solved points:{num_nodes}, distance:{assignment.ObjectiveValue()}")
    return routemodel, assignment
    
def get_route(df, startnode, stopnode): 
    routing, assignment = optimize(df, int(startnode), int(stopnode))
    index = routing.Start(0)
    route = []
    while not routing.IsEnd(index):
        node_index= routing.IndexToNode(index)
        route.append(node_index)
        index = assignment.Value(routing.NextVar(index))      
    route.append(routing.IndexToNode(index))
    return route


# ### Find Route through Clusters

# In[ ]:


get_ipython().run_cell_magic('time', '', "nnode = int(cities.loc[0, 'cluster'])\ncenter_df = cities.groupby('cluster')['X', 'Y'].agg('mean').reset_index()\nlocations = center_df[['X', 'Y']]\nsegment = get_route(locations, nnode, nnode)\n\nfig, ax = plt.subplots()\nclusters = sns.scatterplot(x=cities.X, y=cities.Y, alpha = 0.1, marker='.', hue=cities.cluster_, legend=False, ax=ax)\ncenters = sns.scatterplot(x=center_df.X, y=center_df.Y, marker='x', ax=ax)\n\nordered_clusters = center_df.loc[segment].reset_index(drop=True)\ncenter_route = ax.plot(ordered_clusters.X, ordered_clusters.Y)")


# In[ ]:


#get entries and exits for each cluster
last_exit_cityId = 0
entry_cityIds = []
exit_cityIds = []
for i,m in enumerate(ordered_clusters.cluster, 0):
    cluster = cities.loc[cities.cluster==m].reset_index(drop=True)
    if i < len(ordered_clusters)-1:
        next_center = ordered_clusters.loc[i+1, ['X', 'Y']]
    else:
        next_center = ordered_clusters.loc[0, ['X', 'Y']]
    
    #cluster entry is based on the nearest neighbor to the exit fo last cluster
    last_exit = cities.loc[last_exit_cityId, ['X', 'Y']]
    entry = cdist([last_exit], cluster[['X','Y']], metric='euclidean').argmin()
    entry_cityID = cluster.iloc[entry].CityId
    entry_cityIds.append(entry_cityID)
    
    #cluster exit is based on nearest neighbor to center of next cluster
    exit = cdist([next_center], cluster[['X','Y']], metric='euclidean').argmin()
    exit_cityID = cluster.iloc[exit].CityId
    exit_cityIds.append(exit_cityID)
    
    last_exit_cityId = exit_cityID

ordered_clusters['entry_cityId'] = entry_cityIds
ordered_clusters['exit_cityId'] = exit_cityIds
ordered_clusters.head()   


# ### Solve the problem

# In[ ]:


get_ipython().run_cell_magic('time', '', 'seglist = []\n#total_cities = cities.shape[0]\ncities[\'cluster_index\'] = cities.groupby(\'cluster\').cumcount()\n\nfor i,m in enumerate(ordered_clusters.cluster):\n    if i % 25 == 0: print(f"finished {i} clusters of {ordered_clusters.shape[0]-1}")\n    district = cities[cities.cluster == m]\n    \n    clstart = ordered_clusters.loc[i, \'entry_cityId\']\n    nnode = district.loc[clstart, \'cluster_index\']\n    clstop = ordered_clusters.loc[i, \'exit_cityId\']\n    pnode = district.loc[clstop, \'cluster_index\']\n    locations = district[[\'X\', \'Y\']].values\n    \n    segnodes = get_route(locations, nnode, pnode) #output is type list\n    ord_district =  district.iloc[segnodes]\n    segment = ord_district.index.tolist()\n    seglist.append(segment)\n\nseglist.append([0])\nortools_path = np.concatenate(seglist)')


# In[ ]:


print(f"Total Distance: {total_distance(ortools_path):.1f}")
df_ortools = df_cities.loc[ortools_path].drop_duplicates()
#df_ortools = pd.DataFrame({'CityId':ortools_path}).merge(df_cities,how = 'left')
fig, ax = plt.subplots(figsize=(10,7))
ax = ax.plot(df_ortools['X'], df_ortools['Y'])


# ## Concorde

# This concorde is pretty quick and beats our other solutions handily.

# In[ ]:


from concorde.tsp import TSPSolver
from concorde.tests.data_utils import get_dataset_path


# In[ ]:


get_ipython().run_cell_magic('time', '', 'start_cities = pd.read_csv(\'../input/cities.csv\') # 1533418.5\nsolver = TSPSolver.from_data(\n    start_cities.X,\n    start_cities.Y,\n    norm="EUC_2D"\n)\ntour_data = solver.solve(time_bound = 60.0, verbose = True)')


# In[ ]:


tour = start_cities.loc[tour_data.tour]
tour = tour.append({'CityId':0, 'X':316.836739, 'Y': 2202.340707}, ignore_index=True)


# In[ ]:


print(f"Total Distance: {total_distance(tour.CityId.tolist()):.1f}")
fig, ax = plt.subplots(figsize=(10,7))
ax = ax.plot(tour.X,tour.Y)


# In[ ]:


sub_df = pd.DataFrame(tour_data.tour,columns=['Path']).drop_duplicates()
sub_df=sub_df.append({'Path':0}, ignore_index=True)
sub_df.to_csv("submission.csv", index=False)


# In[ ]:




