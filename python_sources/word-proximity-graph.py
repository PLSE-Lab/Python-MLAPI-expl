#from sets import Set
import igraph
import nltk
import sqlite3
import sys

#reload(sys)
#sys.setdefaultencoding('utf8')

def process_words(words):
    nodes = list(set(words))
    edges = []

    previous_word = None
    for word in words:
        if previous_word:
            edges.append([previous_word, word])
        previous_word = word

    return nodes, edges

def create_graph_from_nodes_and_edges(nodes, edges):
    print("%d nodes" % len(nodes))
    print("%d edges" % len(edges))

    graph = igraph.Graph(directed=True)
    graph.add_vertices(len(nodes))
    graph.vs["name"] = [node for node in nodes]
    graph.add_edges([[nodes.index(v1), nodes.index(v2)] for v1, v2 in edges])
    return graph

def my_union(g1, g2):
    (nodes1, edges1) = g1
    (nodes2, edges2) = g2
    nodes = list(set.union(set(nodes1), set(nodes2)))
    edges = edges1 + edges2
    return nodes, edges

conn = sqlite3.connect('../input/database.sqlite')
words = set()

cur = conn.execute('select count(*) from Emails')
total_count = cur.fetchone()[0]

cur = conn.execute('select ExtractedBodyText from Emails')
nodes = []
edges = []
graph = (nodes, edges)

count = 1
for row in cur.fetchall():
    print("Processing email #%d of %d" % (count, total_count))
    body = row[0]
    tokens = nltk.word_tokenize(body)
    sentence_graph = process_words(tokens)
    graph = my_union(graph, sentence_graph)
    count += 1

g = create_graph_from_nodes_and_edges(graph[0], graph[1])

igraph.save(g, "emails.graphml.gz", format="graphmlz")
