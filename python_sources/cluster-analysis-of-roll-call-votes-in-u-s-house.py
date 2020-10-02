#!/usr/bin/env python
# coding: utf-8

# # Cluster Analysis of Roll Call Votes in U.S. House of Representatives
# 
# ## Imports:

# In[ ]:


#import modules
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn import metrics
from ast import literal_eval
import pandas as pd
import numpy as np
from collections import defaultdict


# ## Read Data and Preprocessing

# In[ ]:


#read members, rollcall_votes and rollcall_info files to dataframe
rollcall_votes = pd.read_csv('/kaggle/input/house-of-representatives-congress-116/house_rollcall_votes_116.csv', index_col = 0)
members = pd.read_csv('/kaggle/input/house-of-representatives-congress-116/house_members_116.csv', index_col = 0)
rollcall_info = pd.read_csv('/kaggle/input/house-of-representatives-congress-116/house_rollcall_info_116.csv', index_col = 0)
bills = pd.read_csv('/kaggle/input/house-of-representatives-congress-116/house_legislation_116.csv', index_col = 0, parse_dates=["date_introduced"])

#deletes votes related to initial quorum and the election of speaker of the house from dataframes
del rollcall_votes['2019:002']
del rollcall_votes['2019:001']
rollcall_info.drop(labels =['2019:001', '2019:002'], inplace = True)

#drop members who have died, left congress or are non-voting members. Also remove speaker of the house who traditionally does not vote 
to_drop =['G000582', 'R000600', 'N000147', 'P000610', 'S001177', 'S001204', 'P000197', 'J000255', 'H001087','C001092', 'C000984', 'D000614','M001179']
rollcall_votes.drop(labels = to_drop, inplace = True)
members.drop(to_drop, inplace = True)
#fill nan values with 0
rollcall_votes.fillna(0, inplace= True)

#replace vote strings with integers representing votes: yes votes to 1, no vote to -1, and abstentions to 0

replace = {'No': -1, 'Aye': 1, 'Yea': 1, 'Nay': -1, 'Not Voting': 0, 'Present': 0}
rollcall_votes.replace(replace, inplace= True)


#converts values of relevant columns to lists of strings rather than strings
bills.cosponsors=bills.cosponsors.apply(literal_eval)
bills.subjects = bills.subjects.apply(literal_eval)
bills.committees = bills.committees.apply(literal_eval)
bills.related_bills = bills.related_bills.apply(literal_eval)


#list of motions related to passage of bills and agreement to resolutions
motions_to_agree_or_pass =['On Agreeing to the Resolution', 
'On Passage', 
'On Motion to Suspend the Rules and Pass', 
'On Motion to Suspend the Rules and Agree',
'On Motion to Suspend the Rules and Agree, as Amended',
'On Motion to Suspend the Rules and Pass, as Amended',
'On Passage, Objections of the President to the Contrary Notwithstanding']

#list of motions related to recommitting bills or resolutions to committee
motion_to_recommit = ['On Motion to Recommit with Instructions','On Motion to Recommit', 'On Motion to Commit with Instructions' ]

#list of motions related to amending bill or resolution
amendment = ['On Agreeing to the Amendment', 'On Motion to Suspend the Rules and Concur in the Senate Amendment',
            'On Motion to Concur in the Senate Amendment', 'On Motion to Concur in the Senate Amendment with an Amendment']

#check unique values in rollcall_votes 
unique = []
for i, row in rollcall_votes.iterrows():
    for x in pd.unique(rollcall_votes.loc[i]):
        if x not in unique:
            unique.append(x)
print(unique)


# <code>rollcall_votes</code> : includes the voting record of each Representative indexed by name_id. Columns represent individual rollcall votes and are named in the form year:rollcall#. Values are 1 for yes vote, -1 for no vote, and 0 for abstention. 
# 
# <code>rollcall_info</code> : dataframe representing information related to each rollcall vote. Index values are the same as column names of rollcall_votes.
# 
# <code>members</code> : dataframe indexed by name_id includes information related to each Representative.
# 
# <code>bills</code> : bills indexed by bill_id includes information related to all bills and resolutions introduced in the House.

# In[ ]:


rollcall_votes.head()


# In[ ]:


rollcall_info.head()


# In[ ]:


bills.head()


# In[ ]:


members.head()


# ## Visual EDA of rollcall_votes using T-SNE

# In[ ]:


#filter rollcall votes to only look at votes which pertain to the passage of bills or agreement to resolutions
rollcalls_of_interest = [index for index, row in rollcall_info.iterrows() if  (row['question'] in motions_to_agree_or_pass) ]
rollcalls_of_interest =  rollcall_votes[rollcalls_of_interest]

#Use K-Means to find clusters for n=2
n = 2
x = rollcalls_of_interest.loc[members.index].values
model = KMeans(n_clusters = n, random_state = 42, algorithm = 'auto')
model.fit(x)
labels = model.predict(x)
members['cluster'] =labels

#create model
model = TSNE(learning_rate = 10)

#fit model
transformed = model.fit_transform(x)
xs = transformed[:,0]
ys = transformed[:,1]

df_trans = pd.DataFrame({'xs':xs, 'ys':ys})

#determine centroid locations of t-sne clusters
cpx =[]
cpy=[]

for i in range(n):
    xi = df_trans.loc[(labels ==i)]['xs'].mean()
    yi = df_trans.loc[(labels ==i)]['ys'].mean()
    cpx.append(xi)
    cpy.append(yi)

#create plots
plt.scatter(df_trans.loc[members.current_party.values =='Democratic']['xs'], df_trans.loc[members.current_party.values =='Democratic']['ys'], c= 'tab:green')
plt.scatter(df_trans.loc[members.current_party.values =='Republican']['xs'], df_trans.loc[members.current_party.values =='Republican']['ys'], c= 'tab:blue')
plt.legend(loc ='best', labels = ['Democratic', 'Republican'])

#annotate clusters centroids for each species
for i in range(n):
    plt.annotate(s = str(i), xy = (cpx[i], cpy[i]), xytext = (-4,-4), textcoords = 'offset points')

plt.title('T-SNE: Rollcall Votes in House')
plt.show()


# The plot above, shows two major clusters related to rollcall votes in the House. Cluster 0 highly associated with memebrs of the Democratic Party and cluster 1 hihgly associated with membes of the Republican Party.

# ## T-SNE of House Votes Filtered by Policy Area

# ### Filter and Subet Data

# In[ ]:


#rollcalls_of_interest: list of votes which pertain to passage of bills or agreement to resolutions pertaining to Immigration
bills_by_policy_area = [bill for bill, row in bills.iterrows() if row.policy_area == 'Immigration']
rollcalls_of_interest = [index for index, row in rollcall_info.iterrows() if  ((row['question'] in motions_to_agree_or_pass) and 
                                                                               (row['bill_id'] in bills_by_policy_area)) ]

#rollcalls_by_subject: df repesenting subset of rollcall_votes with index in rollcalls_of_interest
rollcalls_by_subject = rollcall_votes[rollcalls_of_interest]


# In[ ]:


rollcall_info.loc[rollcalls_of_interest].bill_id.unique()


# The unique bills voted by rollcall pertaining to immigration

# ### Determine Optimal Number of Clusters using Silhoutte Score

# In[ ]:


#import modules
from sklearn.cluster import KMeans
from sklearn import metrics

#instantaiate lists
xp=[]
yp=[]
x = rollcalls_by_subject.loc[members.index].values

# fit KMeans model for n_clusters 2:30
for i in range(2,30):
    model = KMeans(n_clusters = i, random_state = 42)
    model.fit(x)
    labels = model.predict(x)
    score =metrics.silhouette_score(x, labels, metric='euclidean')
    xp.append(i)
    yp.append(score)
plt.plot(xp,yp)
plt.title('Kmeans Clustering')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.show()


# ### Plot T-SNE with Annotated Cluster Centroids

# In[ ]:


n = 28
model = KMeans(n_clusters = n, random_state = 42, algorithm = 'auto')
model.fit(x)
labels = model.predict(x)
members['cluster'] =labels

#create model
model = TSNE(learning_rate = 10)

#fit model
transformed = model.fit_transform(rollcalls_by_subject.loc[members.index].values)
xs = transformed[:,0]
ys = transformed[:,1]

df_trans = pd.DataFrame({'xs':xs, 'ys':ys})

#determine centroid locations of t-sne clusters
cpx =[]
cpy=[]

for i in range(n):
    xi = df_trans.loc[(labels ==i)]['xs'].mean()
    yi = df_trans.loc[(labels ==i)]['ys'].mean()
    cpx.append(xi)
    cpy.append(yi)

#create plots
plt.scatter(df_trans.loc[members.current_party.values =='Democratic']['xs'], df_trans.loc[members.current_party.values =='Democratic']['ys'], c= 'tab:green')
plt.scatter(df_trans.loc[members.current_party.values =='Republican']['xs'], df_trans.loc[members.current_party.values =='Republican']['ys'], c= 'tab:blue')
plt.legend(loc ='best', labels = ['Democratic', 'Republican'])

#annotate clusters centroids for each species
for i in range(n):
    plt.annotate(s = str(i), xy = (cpx[i], cpy[i]), xytext = (-4,-4), textcoords = 'offset points')

plt.title('T-SNE: Rollcall On Immigration')
plt.show()


# We can visually assess three major clusters: 0, 1 and 2. The plot indicates a division in the Republican Party related to Immigration.

# In[ ]:


dic = defaultdict()
for i in range(n):
    dic[i] = len(members.loc[labels == i])

#count within each cluster in descending order
pd.DataFrame.from_dict(dic, orient = 'index', columns = ['count']).sort_values('count', ascending = False).head()


# Cluster counts sorted in descending order shows that cluster 2 has 41 members indicating a fairly large split in the Republican party in votes pertaining to immigration. 

# In[ ]:


#members of cluster 2
members.loc[labels == 2].head(10)


# Shows some of the members of cluster 2. We might hypothesize that cluster 2 represents the more conservative tea-party faction of the Republican Party. 

# In[ ]:



for i in members.cluster.sort_values().unique():
    if 'df' in locals():
        df1  =pd.DataFrame(rollcall_votes.loc[members.loc[labels == i].index].mean().loc[rollcalls_of_interest], columns =[i])

        df = df.merge(df1, how = 'outer', left_index =True, right_index =True)
    else:
        
        df =pd.DataFrame(rollcall_votes.loc[members.loc[labels == i].index].mean().loc[rollcalls_of_interest], columns =[i])
        

#rollcall_votes.loc[members.loc[labels == 0].index].mean().loc[rollcalls_of_interest]
df.merge(rollcall_info.bill_id, how ='left',left_index =True, right_index =True).reset_index().set_index('bill_id')[['index',1,0, 2]].head(20)


# Shows the mean vote within each cluster indexed by bill_id. The major disagreement between clusters 0 and 2 was related to the passage H.R. 1044. Clusters 1 and 0 both overwhelmingly voted for the bill, whereas cluster 2 overwhelmingly voted against the bill. 

# In[ ]:


bills.loc['H.R.1044'].title


# Inspection of the congressional record however, indicates that the disagreement might have been more concerning the way the bill was passed through 'suspension of the rules' rather than a core disagreement with the intention of the bill. For instance, in the congressional record Doug Collins of Georgia states that he agrees with the bill in priciple but does not believe it is ready for passage without the vetting required under normal House procedure. 
