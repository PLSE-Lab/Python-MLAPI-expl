#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import plotly.plotly as py
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)

import cvxpy as cvx
import networkx as nx
import community
from community import community_louvain
from community.community_louvain import best_partition

plt.style.use('fivethirtyeight')
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['figure.titleweight'] = 'bold' 

df=pd.read_csv('../input/israeli_elections_results_1996_to_2015.csv',encoding='iso-8859-1')


# # Preprocessing Functions:
# Selecting two election rounds (by year), 
# keeping only settlements that appeared in both years,
# dropping empty rows and columns,
# and accounting for relative change in settlement sizes (i.e population growth)..

# In[ ]:


def read_and_prepare_data(df,election_year):
    
    votes=df[df['year']==int(election_year)]
    votes=votes.drop(columns='year')
    votes=votes.drop(votes.columns[range(3,8)],axis=1)
    votes=votes.drop(votes.columns[range(0,2)],axis=1)
    votes=votes.drop(votes.columns[votes.sum()==0],axis=1) #clearing empty columns
    votes=votes[np.sort(votes.columns)]
    votes=(votes[(votes.sum(axis=1)>0)]) #clearing empty rows
    votes=votes.add_suffix(election_year)            
    return votes

def load_and_join_data(df,x_label,y_label):
    x_data=read_and_prepare_data(df,x_label)
    y_data=read_and_prepare_data(df,y_label)
    x_data=x_data.groupby('settlement_name_english'+x_label).sum()
    y_data=y_data.groupby('settlement_name_english'+y_label).sum()
    M=x_data.shape[1]
    data_joint=pd.merge(x_data,y_data, how='inner', left_index=True, right_index=True)
    x_data=data_joint[data_joint.columns[range(0,M)]]
    y_data=data_joint[data_joint.columns[M:]]
    x_data=x_data.div(x_data.sum(axis=1),axis=0)
    x_data=x_data.mul(y_data.sum(axis=1),axis=0)
    return x_data,y_data


# # Keeping only major parties
# Dropping all parties below a given threshold, and accounting for how much data is lost.

# In[ ]:


def major_parties(votes,threshold,election_year,verbose):
    if 'settlement_name_english'+election_year in votes.columns:
        votes=votes.drop('settlement_name_english'+election_year,axis=1)
    party_is_major=(votes.sum(axis=0)/sum(votes.sum(axis=0)))>threshold
    major_party_votes=np.sum(votes.values[:,party_is_major],axis=0)
    votes_in_major_parties=np.int(100*np.round(np.sum(major_party_votes)/np.sum(votes.values),2))
    if verbose:
        print(str(votes_in_major_parties)+'% of the '+election_year+' votes are in major parties')
    major_party_votes=major_party_votes/sum(major_party_votes) #rescaling to ignore dropped data
    major_party_titles=[party_is_major.index.values[party_is_major==True][n][:-4] for n in range(0,sum(party_is_major))]
    return party_is_major,major_party_votes, major_party_titles


# # Partitioning the parties into communities
# Here we partition the parties into communities (similar to ([this previous kernel](http://www.kaggle.com/itamarmushkin/partitioning-the-parties))),  in order to color parties by community.

# In[ ]:


def correlation_communities(votes, party_threshold, link_threshold, community_colors):
    
    votes=votes.select_dtypes(include=[np.number])
    relative_votes=votes.div(votes.sum(axis=1), axis=0)
    party_titles=relative_votes.columns.values
    party_is_major=((votes.sum(axis=0)/sum(votes.sum(axis=0)))>party_threshold)
    major_parties=party_titles[party_is_major]
    relative_votes=relative_votes[major_parties]
    
    C=np.corrcoef(relative_votes,rowvar=0)
    A=1*(C>link_threshold)
    G=nx.Graph(A)
    G=nx.relabel_nodes(G,dict(zip(G.nodes(),major_parties)))
    communities=best_partition(G)
    node_coloring=[community_colors[communities[node]] for node in sorted(G.nodes())]
    return communities,node_coloring


# # Solving the vote flow model
# In the heart of this analysis is a very simple model:
# For every two consecutive election cycles, we build a simple linear model, in which every voter for party i in the former cycle has a probability p_ij to vote for party j in the latter cycle.
# We solve this model with simple constraints (0=<p_ij=<1, \sigma_j p_ij=1, etc. )
# This model gives us the vote flow: 
#     For example, If party i recieved 10% of the votes in the t-th election, party j recieved 5% of the votes in the t+1-th election, and p_ij =30%, then 3% of the votes for party j were at the expense of party i.
#     The other 2% are from other parties - including possibly party j itself, if it was present at the t-th election.

# In[ ]:


def solve_transfer_coefficients(x_data,y_data,alt_scale,verbose):
    C=cvx.Variable([x_data.shape[1],y_data.shape[1]])
    constraints=[0<=C, C<=1, cvx.sum(C,axis=1)==1]
    
    objective=cvx.Minimize(cvx.sum_squares((x_data.values*C)-y_data.values))
    prob=cvx.Problem(objective, constraints)
    prob.solve(solver='ECOS')
    if verbose:
        print (prob.status)
    if (prob.status!='optimal') and (prob.status!='optimal_inaccurate'): #Just a numeric thing, to rescale the objective function for the computation to succeed
        objective=cvx.Minimize(alt_scale*cvx.sum_squares((x_data.values*C)-y_data.values))
        prob=cvx.Problem(objective, constraints)
        prob.solve()
        if verbose:
            print(prob.status+'(with alt_scale)')
    C_mat=C.value

    if verbose:
        print(C_mat.min()) #should be above 0
        print(C_mat.max()) #should be below 1
        print(C_mat.sum(axis=1).min()) #should be close to 1
        print(C_mat.sum(axis=1).max()) #should be close to 1
    
    if verbose:
        misplaced_votes=np.sum(np.abs(np.matmul(x_data.values,C_mat)-y_data.values))
        properly_placed_votes=np.int(100*np.round(1-misplaced_votes/np.sum(y_data.values),2))
        print('Transfer model properly account for '+str(properly_placed_votes)+'% of the votes on the local level') #this counts the overall error per settlement
    
    return C_mat


# # Calculation (and plot generation)
# 
# Thresholds were chosen to keep relevant parties and relations, while keeping the figure relatively clear.
# 
# The community partition is generated automatically, but to keep the coloring somewhat coherent between years, I've chosen the coloring manually.
# 
# The plot is interactive - you can move the parties around. I've sorted them according to community, just for visibility's sake.

# In[ ]:


election_labels=['1996', '1999', '2003', '2006', '2009', '2013', '2015']
party_threshold=0.019
link_threshold=0.01
transfer_threshold=0.005
alt_scale=1e-3

community_colors={0:'black',1:'red',2:'blue',3:'green',4:'brown'}
community_colorses={'1996': community_colors}
community_colorses['1999']={0: 'black', 1:'red', 2:'blue', 3:'green'}
community_colorses['1996']=community_colors
community_colorses['2003']=community_colors
community_colorses['2006']=community_colors
community_colorses['2009']=community_colors
community_colorses['2013']=community_colors
community_colorses['2015']={0: 'black', 1:'blue', 2:'red', 3:'green', 4:'brown'}


data_trace = dict(
    type='sankey',
    orientation = "v",
    node = dict(
        pad = 15,
        thickness = 20,
        line = dict(
            color = "black",
            width = 0.5
        ),
        label=[],
        color=[]
    ),
    link = dict(
        source=[],
        target=[],
        value=[]
    )
)

layout =  dict(
    title = "Basic Sankey Diagram",
    width = 1180,
    height = 1544,
    font = dict(
        size = 14
    )
)

for election_index in range(0,len(election_labels)-1):
    x_label=election_labels[election_index]
    y_label=election_labels[election_index+1]
    [x_data,y_data]=load_and_join_data(df,x_label, y_label)
    [major_x,major_party_votes_x,major_party_titles_x]=major_parties(x_data,party_threshold,election_year=x_label,verbose=False)
    major_party_titles_x=[party+'_'+x_label for party in major_party_titles_x]
    [major_y,major_party_votes_y,major_party_titles_y]=major_parties(y_data,party_threshold,election_year=y_label,verbose=False)
    major_party_titles_y=[party+'_'+y_label for party in major_party_titles_y]

    [comms_x,colors_x]=correlation_communities(x_data,party_threshold=party_threshold, link_threshold=link_threshold, community_colors=community_colorses[x_label])
    [comms_y,colors_y]=correlation_communities(y_data,party_threshold=party_threshold, link_threshold=link_threshold, community_colors=community_colorses[y_label])
    
    C_mat=solve_transfer_coefficients(x_data[x_data.columns[major_x]],y_data[y_data.columns[major_y]],alt_scale,verbose=False)
    vote_transfers=np.matmul(np.diag(major_party_votes_x),C_mat)
    links=np.where(vote_transfers>transfer_threshold)

    major_parties_error=np.sum(np.abs(np.matmul(major_party_votes_x,C_mat)-major_party_votes_y))
    major_parties_correct_votes=np.int(100*np.round(1-major_parties_error,2))
    print('Transfer model properly accounts for '+str(major_parties_correct_votes)+'% of the votes for on a national level '+'from '+str(x_label)+' to '+str(y_label))
    
    data_trace['node']['color']=data_trace['node']['color']+colors_x #at the end we need to add the last election
    data_trace['node']['label']=data_trace['node']['label']+major_party_titles_x
    
    if y_label==election_labels[-1]:
        data_trace['node']['color']=data_trace['node']['color']+colors_y
        data_trace['node']['label']=data_trace['node']['label']+major_party_titles_y
    
    if len(data_trace['link']['source'])==0:
        data_trace['link']['source']=links[0]
    else:
        data_trace['link']['source']=np.append(data_trace['link']['source'],links[0]+np.round(np.max(data_trace['link']['source']))+1)
    data_trace['link']['target']=np.append(data_trace['link']['target'],links[1]+np.round(np.max(data_trace['link']['source']))+1)    
    data_trace['link']['value']=np.append(data_trace['link']['value'],vote_transfers[links[0],links[1]])


# In[ ]:


fig = dict(data=[data_trace], layout=layout)
iplot(fig,validate=False)


# # Results and Conclusions
# 
# ## Community Partitioning
# (Going into depth into the 2015 elections in [this Kernel](http://https://www.kaggle.com/itamarmushkin/partitioning-the-parties))  
# As we can see, the community partitioning is mostly consistent between years, even if not entirely so.  
# The parties are always partitioned into Arab parties (black), "Left"/"Center-Left" (red), and "Right" (blue). Interestingly, the Socialist "One Nation" party, led by a Jewish union leader, is consistently classified as an Arab party (i.e same community as distinctly Arab parties).  
# The right is sometimes divided between "secular Right" (Likud etc., in blue) and "Religious right" (Shas etc. in green), up to a few abberations ("The Jewish Home" is its own community in 2015, and so is Yisrael Beitenu in 2009). 
# 
# ## Shifts in Votes
# (going in depth into the 2013 and 2015 elections in [this Kernel](http://https://www.kaggle.com/itamarmushkin/shift-in-votes-from-2013-to-2015))
# 
# Visibly, **most votes shift are between parties in the same community**.  
# It is most evident with the Arab parties, where no votes shift from any of the Jewish party communities to or from it. Similarly, United Torah Judaism (an Ultra-Orthodox party) seems almost entirely disjoint from the rest of the parties.  
# It it also a **trend that intensifies** - we see less and less shifts between red and blue parties over the years. When we do, it's through parties which are considered "centrist" (Third Way, Center Party, Kulanu - etc).
# 
# Many of the votes shift simply refer to forlmal merging and splitting of parties (e.g. Likud Beitenu = Likud + Yisrael Beitenu, or Zionist Union = Labour Party + Hatnuta).  
# However, in the "Center-Left", we see examples of new parties overtaking old ones, which are indistinguishable from formal merging/splitting - but are not!  
# For example, looking at Kadima (2009), we see that both Yesh Atid (2013) and Hatnua (2013) look like formal splits from it - but Hatnua really is, while Yesh Atis really isn't!
