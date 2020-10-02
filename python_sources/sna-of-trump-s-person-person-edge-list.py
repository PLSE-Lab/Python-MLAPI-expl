#!/usr/bin/env python
# coding: utf-8

# # Overview
# This notebook describes a social network analysis of Trump's "Person-Person" edge list.  The findings for three different centrality measures (betweenness, closeness, and PageRank) provide evidence of a lack of cohesion within Donald Trump's circle, but this is likely the result of issues with the BuzzFeed's data set's collection methodology.  Since the data set focuses on individuals' relationships with Trump, but does not provide much insight into the relationships between the non-Trump individuals themselves, the centrality measurements offer little room for comparing the influence of particular individuals.  Further details are provided in the explanations below.

# In[ ]:


import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns


# ###Disclaimer on Edge Analysis
# While the current data set allows for the creation of a directed graph, the current state of the "Connection" attribute is such that there is an inconsistency in which persons are the target and source nodes. In some circumstances, "Person A" may be the source node, but in others rows, "Person A" may actually be the target node. Furthermore, the "Connection" field does not provide a predictable list of categories.  In my view, the utility of the data set would be greatly increased if a finite set of relationships is defined and added to the CSVs.
# 
# Given the amount of effort involved in establishing such a schema, this analysis relies on undirected edges (at least for the time being).
# 
# ##High-level Graph Attributes
# There are 264 relationships described in the Person_Person.csv file, with 232 unique individuals.  A random individual in Trump's social network has an average of two contacts in the rest of the social network.  117 of these relationships (or approx. 44%) are with Donald Trump himself.  The graph's density indicates that the individuals have very little connection with one another.  Given the way this data has been collected with a focus on the individuals' direct relationship with Donald Trump, the density is not particularly surprising.

# In[ ]:


person_df = pd.read_csv('../input/Person_Person.csv',
                        encoding='ISO-8859-1')
person_G = nx.from_pandas_dataframe(person_df,
                                    source='Person A',
                                    target='Person B',
                                    edge_attr='Connection')
print(nx.info(person_G))
print('Density %f' % (nx.density(person_G)))


# In[ ]:


person_df[(person_df['Person A'] == 'DONALD J. TRUMP') | 
          (person_df['Person B'] == 'DONALD J. TRUMP')].info()


# #### Function for Drawing Graphs

# In[ ]:


pos = nx.spring_layout(person_G) # Needs to created ahead of time for a consistent graph layout

def draw_graph(dataframe, metric):
    node_sizes = dataframe[metric].add(.5).multiply(50).tolist()
    nodes = dataframe.index.values.tolist()
    edges = nx.to_edgelist(person_G)
    metric_G = nx.Graph()
    metric_G.add_nodes_from(nodes)
    metric_G.add_edges_from(edges)
    labels = {}
    color_map = {}
    for node in nodes[:25]:
        labels[node] = node
    plt.figure(1, figsize=(20,20))
    plt.title('Trump\'s Person-Person Network (%s)' % (metric))
    nx.draw(metric_G,
            pos=pos,
            node_size=node_sizes, 
            node_color='#747587',
            with_labels=False)
    nx.draw_networkx_nodes(metric_G,
                           pos=pos,
                           nodelist=nodes[:25],
                           node_color='#873737',
                           nodesize=node_sizes)
    nx.draw_networkx_nodes(metric_G,
                           pos=pos,
                           nodelist=nodes[25:],
                           node_color='#747587',
                           nodesize=node_sizes)
    nx.draw_networkx_edges(metric_G,
                           pos=pos,
                           edgelist=edges,
                           arrows=False)
    tmp = nx.draw_networkx_labels(metric_G,
                                  pos=pos,
                                  labels=labels,
                                  font_size=16,
                                  font_color='#000000',
                                  font_weight='bold')


# ##Betweenness Centrality
# Betweenness centrality measures the extent to which a particular individual acts as a "broker" to other nodes in the social network. The higher an individual's betweenness score, the greater their ability to impede information flow throughout the network.  Mathematically, the an individual's score is the ratio of the number of shortest paths, containing the individual in question, between all possible pairs of nodes, versus the number of all shortest paths between node pairs in the network. If an individual has a betweenness score of 0, then it means that they have no role in the flow of information.  On the other hand, a betweenness score of 1 indicates that all information must pass through that individual.
# 
# ### Donald Trump is #1 for Betweenness Centrality
# Donald Trump's centrality score (0.798) is 6x larger than the second highest person (Jared Kushner: 0.156).  Trump dominates the flow of information among his relationships. Given that this network was developed on the basis of people's ties to Donald Trump, his position at the top is not surprising.  However, the sheer difference between the first and second highest betweenness scores may suggest a lack of collaboration and interaction within his social network.
# 
# ### Jared Kushner is #2
# Jared Kushner also has a noticeable lead on other members of Trump's circle with it being 3x larger than the third highest account.  Although not a massive score to at all suggest that Kushner controls access to Trump, it does support the belief that he acts as a trusted advisor given his relative control of the information flow.  It should be noted, however, that Paul Manafort's position as third highest raises questions about the age and collection methods for this data.
# 
# ### Amusing Rankings
# Establishment Republicans Mike Pence and Reince Preibus rank at #14 and #15 (respectively), suggesting a lack of influence on Donald Trump.  Also, Vladimir Putin (with a very low score 0f 0.008) ranks two spots above Kellyanne Conway (#23 and #25, respectively).  If this data is up-to-date and accurate, Conway's nearly total lack of information brokerage power would be especially odd given her role as Trump's campaign manager.

# In[ ]:


person_betweenness = pd.Series(nx.betweenness_centrality(person_G), name='Betweenness')
person_person_df = pd.Series.to_frame(person_betweenness)
person_person_df['Closeness'] = pd.Series(nx.closeness_centrality(person_G))
person_person_df['PageRank'] = pd.Series(nx.pagerank_scipy(person_G))
desc_betweenness = person_person_df.sort_values('Betweenness', ascending=False)
print('Top Highest Betweenness Centrality Persons')
desc_betweenness.head(25)


# In[ ]:


draw_graph(desc_betweenness, 'Betweenness')


# ##Closeness Centrality
# Closeness centrality provides information as to how easily an individual can access all other individuals in the social network.  If an individual has a closeness score of 1, then he or she has direct access to all other individuals.  Someone with a score of 0 cannot access anyone else.
# 
# ### Donald Trump is #1 for Closeness Centrality
# Again, given the way that this data was collected, Donald Trump's position with the highest closeness score is not surprising.
# 
# ### Trump Family Dominates the Top 5
# Jared Kushner also makes another appearance in second, but with the rest of Trump's "public facing" family also present.  Paul Manafort makes another appearance at #5 (again, raising questions about the collection methods used to create this data set).
# 
# 
# ### All are Equal (Almost)
# Unlike in the case of information brokerage (as described in the betweenness centrality section above), everyone essentially has equal access to everyone else.  Everyone even has a relatively close score to Trump's.  However, taking into account the earlier density metric for the graph (0.009), it is clear that this is not because everyone knows each other.  Rather, it reflects Donald Trump's domination of this social network in so far as their connections to one another derive from their associations with Trump.

# In[ ]:


desc_closeness = person_person_df.sort_values('Closeness', ascending=False)
print('Top Highest Closeness Centrality Persons')
desc_closeness.head(25)


# In[ ]:


draw_graph(desc_closeness, 'Closeness')


# ##PageRank Centrality
# PageRank centrality provides a measurement for influence itself. The more relationships an individual has with other important individuals, and the more selective those other important individuals are with their relationships, then the greater the influence of that particular individual. The method itself was invented by the cofounders of Google to rank their search engine's results, but it can likewise be applied to social networks.  Just as a particular webpage can provide the best information based on its usage by other websites, so can a particular individual yield great influence based on other individuals' reliance upon him or her for information.
# 
# ### Donald Trump is (again) #1
# As before, this is unsurprising given the manner in which this data was collected with a focus on Trump as the nexus for all relationships in this social network.
# 
# ### Jared Kushner is (again) #2.  Family is important
# Jared Kushner is in the second spot with a PageRank score that is 8x lower than Donald Trump.  Keeping in mind the collection method that was used in creating the edge list, it does further support the idea that Kushner is an important advisor within Trump's circle.  Notably, his PageRank score is nearly 3x times larger than that of Mike Pence (making his appearance at #8, below Jeff Sessions and Stephen Miller, but three spots above Vladimir Putin).  While Paul Manafort is #3 in the rankings (again, raising questions as to the age of this data set), Ivanka and Donald Trump Jr. also act as influential nodes.  Also of interest, is the approximate equality of Donald Trump Jr.'s influence with that of Stephen Miller.

# In[ ]:


desc_pagerank = person_person_df.sort_values('PageRank', ascending=False)
print('Top Highest PageRank Persons')
desc_pagerank.head(25)


# In[ ]:


draw_graph(desc_pagerank, 'PageRank')


# ## A Final Reminder
# 
# [*The TrumpWorld map is based on a wide-ranging search of financial disclosures, news stories, and other records.*][1] - BuzzFeed
# 
# When analyzing this data set, it is important to remember that the data was collected based on an individual's relationship with Donald Trump himself.  It seems likely that the collection method does not emphasize Trump's network participants with one another.  It also means that the nodes and edges in the network are defined by what is publicly available.  There is certainly the possibility of unknown influential persons and unknown relationships between people in Trump's circle.  In turn, this means that centrality metrics underestimate the influence of certain individuals (besides Trump himself) since their social networks are relatively undeveloped.
# 
# ### Trump's Person-Person Network w/o Trump
# Emphasizing the network's dependence on Trump for its existence, if we remove Donald Trump from his own social network, we see how little contact the different members have with one another.  With Trump's absence, an individual has an average of a relationship with only one other person.  Given that they had an average of two relationships when Trump is in the network, this implies that one of those relationships is with the man himself.  Furthermore, the density of the graph is cut in half (from approx. 0.01 to 0.006).  While this could be evidence of the lack of unity within Trump's circle, I suspect that this is more likely a result of flaws with the data set's collection methods.  By contrast, removing Jared Kushner, the second most central figure according to all our centrality metrics, barely impacts the network's original organization.
# 
#   [1]: https://www.buzzfeed.com/johntemplon/help-us-map-trumpworld?utm_term=.jnBwJz60n#.seBqB54dg

# In[ ]:


edited_G = person_G.copy()
edited_G.remove_node('DONALD J. TRUMP')
print('High-level Metrics in Graph w/ Trump')
print('='*10)
print(nx.info(person_G))
print('Density %f\n' % (nx.density(person_G)))
print('High-level Metrics in Graph w/o Trump')
print('='*10)
print(nx.info(edited_G))
print('Density %f\n' % (nx.density(edited_G)))
kushner_G = person_G.copy()
kushner_G.remove_node('JARED KUSHNER')
print('High-level Metrics in Graph w/o Jared Kushner')
print('='*10)
print(nx.info(kushner_G))
print('Density %f' % (nx.density(kushner_G)))

