#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from IPython.display import Image
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# # Use TextbookSampleCode Classes

# In[2]:


class Graph:
    """Representation of a simple graph using an adjacency map."""

    # ------------------------- nested Vertex class -------------------------
    class Vertex:
        """Lightweight vertex structure for a graph."""
        __slots__ = '_element'

        def __init__(self, x):
            """Do not call constructor directly. Use Graph's insert_vertex(x)."""
            self._element = x

        def element(self):
            """Return element associated with this vertex."""
            return self._element

        def __hash__(self):  # will allow vertex to be a map/set key
            return hash(id(self))

        def __str__(self):
            return str(self._element)

    # ------------------------- nested Edge class -------------------------
    class Edge:
        """Lightweight edge structure for a graph."""
        __slots__ = '_origin', '_destination', '_element'

        def __init__(self, u, v, x):
            """Do not call constructor directly. Use Graph's insert_edge(u,v,x)."""
            self._origin = u
            self._destination = v
            self._element = x

        def endpoints(self):
            """Return (u,v) tuple for vertices u and v."""
            return (self._origin, self._destination)

        def opposite(self, v):
            """Return the vertex that is opposite v on this edge."""
            if not isinstance(v, Graph.Vertex):
                raise TypeError('v must be a Vertex')
            return self._destination if v is self._origin else self._origin
            raise ValueError('v not incident to edge')

        def element(self):
            """Return element associated with this edge."""
            return self._element

        def __hash__(self):  # will allow edge to be a map/set key
            return hash((self._origin, self._destination))

        def __str__(self):
            return '({0},{1},{2})'.format(self._origin, self._destination, self._element)

    # ------------------------- Graph methods -------------------------
    def __init__(self, directed=False):
        """Create an empty graph (undirected, by default).
    
        Graph is directed if optional paramter is set to True.
        """
        self._outgoing = {}
        # only create second map for directed graph; use alias for undirected
        self._incoming = {} if directed else self._outgoing
    
    def _validate_vertex(self, v):
        """Verify that v is a Vertex of this graph."""
        if not isinstance(v, self.Vertex):
            raise TypeError('Vertex expected')
        if v not in self._outgoing:
            raise ValueError('Vertex does not belong to this graph.')

    def is_directed(self):
        """Return True if this is a directed graph; False if undirected.
    
        Property is based on the original declaration of the graph, not its contents.
        """
        return self._incoming is not self._outgoing  # directed if maps are distinct

    def vertex_count(self):
        """Return the number of vertices in the graph."""
        return len(self._outgoing)

    def vertices(self):
        """Return an iteration of all vertices of the graph."""
        return self._outgoing.keys()

    def edge_count(self):
        """Return the number of edges in the graph."""
        total = sum(len(self._outgoing[v]) for v in self._outgoing)
        # for undirected graphs, make sure not to double-count edges
        return total if self.is_directed() else total // 2

    def edges(self):
        """Return a set of all edges of the graph."""
        result = set()  # avoid double-reporting edges of undirected graph
        for secondary_map in self._outgoing.values():
            result.update(secondary_map.values())  # add edges to resulting set
        return result

    def get_edge(self, u, v):
        """Return the edge from u to v, or None if not adjacent."""
        self._validate_vertex(u)
        self._validate_vertex(v)
        return self._outgoing[u].get(v)  # returns None if v not adjacent

    def degree(self, v, outgoing=True):
        """Return number of (outgoing) edges incident to vertex v in the graph.
    
        If graph is directed, optional parameter used to count incoming edges.
        """
        self._validate_vertex(v)
        adj = self._outgoing if outgoing else self._incoming
        return len(adj[v])

    def incident_edges(self, v, outgoing=True):
        """Return all (outgoing) edges incident to vertex v in the graph.
    
        If graph is directed, optional parameter used to request incoming edges.
        """
        self._validate_vertex(v)
        adj = self._outgoing if outgoing else self._incoming
        for edge in adj[v].values():
            yield edge

    def insert_vertex(self, x=None):
        """Insert and return a new Vertex with element x."""
        v = self.Vertex(x)
        self._outgoing[v] = {}
        if self.is_directed():
            self._incoming[v] = {}  # need distinct map for incoming edges
        return v

    def insert_edge(self, u, v, x=None):
        """Insert and return a new Edge from u to v with auxiliary element x.
    
        Raise a ValueError if u and v are not vertices of the graph.
        Raise a ValueError if u and v are already adjacent.
        """
        if self.get_edge(u, v) is not None:  # includes error checking
            raise ValueError('u and v are already adjacent')
        e = self.Edge(u, v, x)
        self._outgoing[u][v] = e
        self._incoming[v][u] = e


# # Question 1

# In[ ]:


def graph_from_edgelist(E, directed=False):
   """Make a graph instance based on a sequence of edge tuples.

   Edges can be either of from (origin,destination) or
   (origin,destination,element). Vertex set is presume to be those
   incident to at least one edge.

   vertex labels are assumed to be hashable.
   """
   g = Graph(directed)
   V = set()
   for e in E:
       V.add(e[0])
       V.add(e[1])

   verts = {}  # map from vertex label to Vertex instance
   for v in V:
       verts[v] = g.insert_vertex(v)

   for e in E:
       src = e[0]
       dest = e[1]
       element = e[2] if len(e) > 2 else None
       g.insert_edge(verts[src], verts[dest], element)

   return g


# In[ ]:


E = (('U','V',1), ('U','W',2),
     ('V','X',3), ('V','W',4),
     ('W','X',5), ('W','Y',6),
     ('X','Y',7), ('X','Z',8)
    )
graph = graph_from_edgelist(E, False)

vertices = []
edges = []
print('Vertices (Undirected Graph):', '\n')
for vertice in graph._outgoing:
    vertices.append(vertice)
    print(vertice)
print('\n')    

for vertice in graph._outgoing:
    print('Vertice --', vertice)
    print('Adjacent Vertices (Origin, Desination, Path Number')
    incidents = graph.incident_edges(vertice)
    for incident in incidents:
        edges.append(incident)
        print(incident)


# In[3]:


Image("../input/q1.jpg")


# In[ ]:



def BFS(g, s, discovered):
    """Perform BFS of the undiscovered portion of Graph g starting at Vertex s.

    discovered is a dictionary mapping each vertex to the edge that was used to
    discover it during the BFS (s should be mapped to None prior to the call).
    Newly discovered vertices will be added to the dictionary as a result.
    """
    level = [s]  # first level includes only s
    while len(level) > 0:
        next_level = []  # prepare to gather newly found vertices
        for u in level:
            for e in g.incident_edges(u):  # for every outgoing edge from u
                v = e.opposite(u)
                if v not in discovered:  # v is an unvisited vertex
                    discovered[v] = e  # e is the tree edge that discovered v
                    next_level.append(v)  # v will be further considered in next pass
        level = next_level  # relabel 'next' level to become current
        
            
            
def BFS_complete(g):
    """Perform BFS for entire graph and return forest as a dictionary.

    Result maps each vertex v to the edge that was used to discover it.
    (vertices that are roots of a BFS tree are mapped to None).
    """
    forest = {}
    for u in g.vertices():
        if u not in forest:
            forest[u] = None  # u will be a root of a tree
            BFS(g, u, forest)
    return forest


# # Question 2

# In[ ]:


bfs = BFS_complete(graph)


# In[ ]:


for key, value in bfs.items():
    print('Found:', key, 'along the', value, 'path')


# In[ ]:


def DFS(g, u, discovered):
    """Perform DFS of the undiscovered portion of Graph g starting at Vertex u.

    discovered is a dictionary mapping each vertex to the edge that was used to
    discover it during the DFS. (u should be "discovered" prior to the call.)
    Newly discovered vertices will be added to the dictionary as a result.
    """
    for e in g.incident_edges(u):  # for every outgoing edge from u
        v = e.opposite(u)
        if v not in discovered:  # v is an unvisited vertex
            discovered[v] = e  # e is the tree edge that discovered v
            DFS(g, v, discovered)  # recursively explore from v


def construct_path(u, v, discovered):
    """
    Return a list of vertices comprising the directed path from u to v,
    or an empty list if v is not reachable from u.

    discovered is a dictionary resulting from a previous call to DFS started at u.
    """
    path = []  # empty path by default
    if v in discovered:
        # we build list from v to u and then reverse it at the end
        path.append(v)
        walk = v
        while walk is not u:
            e = discovered[walk]  # find edge leading to walk
            parent = e.opposite(walk)
            path.append(parent)
            walk = parent
        path.reverse()  # reorient path from u to v
    return path


def DFS_complete(g):
    """Perform DFS for entire graph and return forest as a dictionary.

    Result maps each vertex v to the edge that was used to discover it.
    (Vertices that are roots of a DFS tree are mapped to None.)
    """
    forest = {}
    for u in g.vertices():
        if u not in forest:
            forest[u] = None  # u will be the root of a tree
            DFS(g, u, forest)
    return forest


# # Question 3

# In[ ]:


dfs = DFS_complete(graph)


# In[ ]:


for key, value in dfs.items():
    print('Found:', key, 'along the', value, 'path')


# # Question 4a

# In[ ]:


E2 = (('1','2','Path1'), ('1','3','Path2'), ('1','4','Path3'), 
     ('2','3','Path4'), ('2','4','Path5'),
     ('3','4','Path6'), 
     ('4','6','Path7'),
     ('6','5','Path8'), ('6','7','Path9'),
     ('5','7','Path10'), ('5','8','Path11'),
     ('7','8','Path12')
    )
graph2 = graph_from_edgelist(E2, False)

vertices = []
edges = []
print('Vertices (Undirected Graph):', '\n')
for vertice in graph2._outgoing:
    vertices.append(vertice)
    print(vertice)
print('\n')    

for vertice in graph2._outgoing:
    print('Vertice --', vertice)
    print('Adjacent Vertices (Origin, Desination, Path Number')
    incidents = graph2.incident_edges(vertice)
    for incident in incidents:
        edges.append(incident)
        print(incident)


# # Question 4b

# In[ ]:


bfs2 = BFS_complete(graph2)


# In[ ]:


for key, value in bfs2.items():
    print('Found:', key, 'along the', value)


# # Question 4c

# In[ ]:


dfs2 = DFS_complete(graph2)


# In[ ]:


for key, value in dfs2.items():
    print('Found:', key, 'along the', value)

