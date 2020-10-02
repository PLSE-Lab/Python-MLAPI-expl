#!/usr/bin/env python
# coding: utf-8

# <br>
# # <font color="red">!! A Hybrid Approach to Connect Donors to Projects !! </font>
# 
# # <font color="red">Using Graphs, Word Embeddings, Context & DeepNets</font>
# 
# <br>
# ## <font color="red">1. Problem Statement </font>
# 
# Majority of the donors on DonorsChoose.org are first time donors. If DonorsChoose.org can motivate even a fraction of those donors to make another donation, that could have a huge impact on the number of classroom requests fulfilled. DonorsChoose.org needs to connect donors with the projects that most inspire them. The idea is to pair up donors to the classroom requests that will most motivate them to make an additional gift.
# 
# ## <font color="red">1.1 Breakdown of the Problem Statement </font>
# 1. The problem statement has two verticals - Donors and Projects  
# 2. The target action among the two verticals is to find right connections betwen them.  
# 3. The main idea is to find the relevant donors for any given new project.   
# 4. Send targeted emails to the recommended donors.    
# 
# ## <font color="red">1.2 Hypothesis Generation  </font>
# 
# Hypothesis Generation is the process to undestand the relationships, possible answeres to the problem statement without looking at the data. So before exploring, lets try to answer the key question - What makes a donor donate to a given project? The possible hypothesis could be following - 
# 
# 1. **Altruism**  - Many generous donors might feel that they find it really important to help others. Thus, they make donations.   [Recently](https://www.donorschoose.org/project/have-a-seat-and-lets-move/3270204/), I also made a donation on donorschoose and the feeling was great. 
# 
# 2. **External Influence**  - Some donors might get influenced from others and makes the donation after watching other people donate.  There is also a [study](https://onlinelibrary.wiley.com/doi/full/10.1111/ecoj.12114) which was conducted  on measuring Peer Effects in Charitable Giving. 
# 
# 3. **Social Cause** - Many donors might feel that they give because their donations matter to someone they know and care about. This may be the schools from which they studied, or their teachers from their childhood days.  
# 
# 4. **Marketing Influence** - Donors might have find it somewhere about the classroom requests, so they tend to donate.  
# 
# ## <font color="red">1.3 Exploratory Data Analysis </font>
# 
# In my previous kernel on donorschoose, I performed an In-depth exploration in order to understand the dataset which included geography analysis, time series trends, distributions of important variables of the dataset, text based analysis, and comparisons.   
# 
# Link : https://www.kaggle.com/shivamb/a-visual-and-insightful-journey-donorschoose
# 
# 
# ![](https://i.imgur.com/ml7uzlr.png)
# 
# 

# # <font color="Red">2. Approach - Connecting Donors to Classroom Requests</font>
# 
# 
# Based on the problem statement, insights from eda, business intution, hypothesis generation, and thought process, the overview of my approach is shown in the following figure and is described afterwards:  
# 
# <br>
# 
# ![DonorsChoose%20%281%29.png](https://i.imgur.com/htGl6pf.jpg)
# 
# <br>
# 
# The main ideas behind the appraoch are:
# 
# 1. Connect donors whose features are relevant to the given project  (project - donor similarity)  
# 2. Connect donors of the projects whose features are contextually similar to the given project (project - project similarity)   
# 3. Find similar donors using donor - donor similarity   (donor - donor similarity)    
# 4. Cluster the donors on the basis of their contextual details (context aware recommendations)  
# 5. Find the donors who are likely to make donation in a given project on the basis of their past behaviours (Donors who donated in these projects also donated in these other projects )
# 
# In the implementation part, I am making use of following: 
# 
# 1. A Graph based approach to represent the donor profiles, project profiles, connections among the donors, connection among the projects. 
# 2. Training a deep neural network using the past donor project interactions to generate the donor-project relevance score. 
# 3. Ensemble of the two parts are given as the final results      
# 
# **Part A: Graph Based Recommendations**
#     
# In this approach, the information of the donors and the projects is saved individually in separate graph like structures. Every node of these graphs repersents either a donor or a project, the edges between them represents how similar the two nodes are. The process flow is defined below:  
#     
# >    1. First, the projects and donors are profiled individually based on their past data. In this step, additional features for the donors and projects are also created using aggregations, feature engineering techniques, and word embedding vectors.   
# >    2. There are two types of features in the dataset - text features and non-text features. From my learnings from the [previous](https://www.kaggle.com/c/donorschoose-application-screening) donorschoose competition and insights from eda, I realize that text features plays an imporant role in describing the context of the projects. Using the pre-trained models, the word embeddings vectors of text features are then created.   
# >    3. The complete information is then represented in the form of graphs in which every nodes repersents either a donor or a project.  
# >    4. To create the edges among the nodes graphs, similarity algorithms such as cosine similairty, jaccard similarirty, and pearson correlation are run. The edges are then created among the donor - donor nodes, project - project nodes.   
# >    5. To find the relevant donors for a given project, its edges are created with the donors graph using the same similarity algorithm. The most strong connections represents first set of suggested donors for a project.  
# >    6. This set is further extended by finding the donors of the similar projects using projects graph.  Also, other similar donors are added as the recommended set.  
# >    7. Using the non text features, the final recommendations are filtered. Those features act as the filters to the final set.  
# 
# **Context Aware**  
# >   One of the important vertical of the donorschoose data is the donors. It is worth the effort to enrich the additional details or context about the donors. Using external datasets, donor profiles can be enriched with more features which can be used in the overall recommendation model. This idea is inspired from the main intution behind **context aware recommendations**.     The more details about this section are given in section 5.1  
#     
#  
# **Part B: Recommendations using DeepNets**
# 
# The key idea behind this part is evaluating the past behaviours of donors to measure if a given donor - project combination is significant or not. All the past project donor interactions are represented in the form of a matrix in which every row repersent a project, every column a donor and every cell value represnt the interaction score. In this model, Donation amount is choosen as the proxy for the interaction score.  The process flow is defined below:  
# 
# > 1. In this approach, the past interactions of donors and projects are represnted in the form of embedding layers.    
# > 2. The embedding layers for donors and the projects are concatenated together. The resultant vector is then passed from multiple multi-layer perceptrons layers.  
# > 3. In the model training process, the embedding parameters are learned which gives a latent representation of every project donor interaction. The learned model then tries to predict the interaction score for a given donor - project combination.   
# > 4. Final recommendations of the donors for a project are generated by predicting the interaction scores for every combination.  Donors having higher score are recommended.  
# 
# **Final Results : finding the donors of a given project ** 
# 
# The final results are given as the combination of results obtained in part A and part B.  
# 
# In this whole workflow, some initial filters are applied on the dataset such as considering only the projects which are Fully Funded and the donors which are not teachers. 
# 
# <br><hr>
# # <font color="red"> 3. Dataset Preparation   </font>
# 
# To begin with, Import all the rquired python modules to be used in the entire framework.  

# In[32]:


# keras modules 
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from keras.layers import Embedding, Reshape, Dropout, Dense, Input, Concatenate
from keras.models import Sequential, Model

# sklearn modules 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans 
from sklearn import preprocessing 
import matplotlib.pyplot as plt

# other python utilities 
from collections import Counter 
from tqdm import tqdm 
import pandas as pd 
import numpy as np 
import warnings, math
import gc 

# ignore the warnings 
warnings.filterwarnings('ignore')


# ## <font color="red">3.1 Load Dataset</font> 
# Load the dataset files into pandas dataframe. I am using four datasets - donors, donations, projects, schools.

# In[33]:


path = "../input/io/"
donors_df = pd.read_csv(path+"Donors.csv")
donations_df = pd.read_csv(path+"Donations.csv")
schools_df = pd.read_csv(path+"Schools.csv")
projects_df = pd.read_csv(path+"Projects.csv")


# ## <font color="red">3.2 Merge Datasets</font>
# 
# Merge the input datasets together so that all the features can be used for the project and user profiling. Also, create additional features in the projects data related to Posted Year and Posted Month.  

# In[34]:


## Merge donations and donors 
donation_donor = donations_df.merge(donors_df, on='Donor ID', how='inner')

## Merge projects and schools 
projects_df = projects_df.merge(schools_df, on='School ID', how='inner')

## Create some additional features in projects data
projects_df['cost'] = projects_df['Project Cost']                      .apply(lambda x : float(str(x).replace("$","").replace(",","")))
projects_df['Posted Date'] = pd.to_datetime(projects_df['Project Posted Date'])
projects_df['Posted Year'] = projects_df['Posted Date'].dt.year
projects_df['Posted Month'] = projects_df['Posted Date'].dt.month

## Merge projects and donations (and donors)
master_df = projects_df.merge(donation_donor, on='Project ID', how='inner')

## Delete unusued datasets and clear the memory
del donation_donor, schools_df
gc.collect()


# # <font color="red">4. Projects Profiling</font> 
# 
# Profiling of projects is a pretty straightforward task because every row in the projects dataset repersents the features of a particular project. Every row in this data is itself a profile of a particular project. For example - 
# 
#     1. Project A : 
#         - Project Title : Need Flexible Seating  
#         - Project Category : Supplies    
#         - Project State : Texas  
#         - Posted Year : 2017  
#         - Project Cost : 1000
#         - ...  
#         
#         
#     2. Project B : 
#         - Project Title : Need Chromebooks  
#         - Project Category : Technology  
#         - Project State : Florida  
#         - Posted Year : 2016  
#         - Project Cost : 500
#         - ...
# 
# Every feature repersents some information about the project. One of the important feature about projects are the text features such as project essay. To use project essays as feature of the project, I will use pre-trained word embedding vectors from fastText.  
# 
# ## <font color="red">4.1 Create Word Embedding Vectors</font>  
# 
# Word embeddings are the form of representing words and documents using a dense vector representation. The position of a word within the vector space is learned from text and is based on the words that surround the word when it is used. Word embeddings can be trained using the input corpus itself or can be generated using pre-trained word embeddings such as Glove, FastText, and Word2Vec. Any one of them can be used as transfer learning. I am using fastText English Word Vectors from Common Crawl as the pre-trained vectors. These vectors contain 2 million word vectors trained on 600B tokens. 
# 
# Source of the fastText vectors - https://www.kaggle.com/facebook/fatsttext-common-crawl/data  
# 

# In[35]:


## Create a smaller version of data so that it runs on kaggle kernel
## keep only fully funded projects
projects_mini = projects_df[projects_df['Project Current Status'] == "Fully Funded"]

## Set rows = -1 to run on complete dataset, To run in kaggle kernel, I am setting to a smaller number 
rows = 5000

## keep only the projects of 2017, quarter 3, take small sample, (so that it runs on kaggle kernels)
if rows != -1:
    projects_mini = projects_mini[(projects_mini['Posted Year'] == 2017) &
                                  (projects_mini['Posted Month'] > 9)]
    projects_mini = projects_mini.reset_index()[:rows]

## replace the missing values and obtain project essay values 
projects_mini['Project Essay'] = projects_mini['Project Essay'].fillna(" ")
xtrain = projects_mini['Project Essay'].values


# First of all load the pre-trained word vectors from fastText Common Crawl into the python dictionary.  This dictionary will be used to get the word embedding vector of a given word in the project essay. 

# In[36]:


EMBEDDING_FILE = '../input/fatsttext-common-crawl/crawl-300d-2M/crawl-300d-2M.vec'

embeddings_index = {}
f = open(EMBEDDING_FILE, encoding="utf8")
count = 0
for line in tqdm(f):
    count += 1
    ## Remove this if condition to read 2M rows 
    if count == 500000: 
        break
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()


# Now the word embedding vectors are loaded, lets convert the project essay values to document vectors. For this purpose, I have created a function which can be used to compute the average word embedding vector of an entire document.

# In[37]:


def generate_doc_vectors(s):
    words = str(s).lower().split() 
    words = [w for w in words if w.isalpha()]
    M = []
    for w in words:
        if w in embeddings_index:
            M.append(embeddings_index[w])
    v = np.array(M).sum(axis=0)
    if type(v) != np.ndarray:
        return np.zeros(300)
    return v / np.sqrt((v ** 2).sum())

xtrain_embeddings = [generate_doc_vectors(x) for x in tqdm(xtrain)]

del xtrain
gc.collect()


# Before using the word embeddings, lets normalize them using Standard Scaler

# In[38]:


scl = preprocessing.StandardScaler()
xtrain_embeddings = np.array(xtrain_embeddings)
xtrain_embeddings = scl.fit_transform(xtrain_embeddings)


# Lets take a look at the project profiles and some of their features 

# In[39]:


projects_mini[['Project ID', 'Project Title', 'Project Subject Category Tree', 'Project Resource Category', 'Project Cost']].head()


# Lets take a look at the word embedding vectors for every project

# In[40]:


print ("Project ID: " + projects_mini['Project ID'].iloc(0)[0])
print ("Project Vector: ")
print (xtrain_embeddings)


# # <font color="red">5. Donors Profiling </font>
# 
# Now, lets profile the donors using their past donations. The aggregated results from the past donations made by the donors will indicate the characteristics of the donors. However, there exists a cold start with the donors who have made only single donation in the past. Basically which means that their profiles will not depict a true picture about their characteristics. 
# 
# Cold Start Problem : Donors with Single Donation
# 
# So as we start aggregating the profiles, we will end be creating more and more better profiles if the number of donations by a particular donor is higher while it will be poor if the donations are lesser. 
# 
# <table align="center">
#     <tr><td>Donor</td><td> Number of Projects Donated</td><td>Profiling</td></tr>
#     <tr><td>A</td><td>1</td><td>Less Accurate Profiling</td></tr>
#     <tr><td>B</td><td>2 - 5</td><td>Average Profiling</td></tr>
#     <tr><td>C</td><td>5 - 10</td><td>Good Profiling</td></tr>
#     <tr><td>D</td><td>10 - 100</td><td>Very Good Profiling</td></tr>
#     <tr><td>E</td><td>100+</td><td>Excellent Profiling</td></tr>
# </table>
# 
# 
# 
# 
# In my approach I am handelling this part using two ways, one is to use external datasets to add more context in the profiles and second is to simply ignore the fact that number of donations are less and use the features as it is. The intution about second approach is that atleast there will be initial level understanding about the donors. 
# 
# 1. Enrichment using external context 
# 2. Use the first project of the donor  
# 
# ## <font color="red">5.1 Context Enrichment</font>
# 
# The first part is somewhat close to **context aware recommendation** engines in which the model uses additional context to measure the similarity of a new donor with the other donors in order to create the donor profiles. 
# 
# In this technique, donor profiles can be enriched with some external data which will be used as features to find similar donors. In the given donorschoose dataset, very few details about the context / persona of donors is given. Example of context are shown in the following image. 
# 
# So basically in this section, we will create additional features which can be added to the donors profile. 
# 
# <br><br>
# 
# ![context-aware.png](https://i.imgur.com/dMU0his.jpg)
# 
# 
# <br><br>
# 
# Context can be defined in any terms such as the social media information of the donor, the credit card or the banking transactions of the donor, the demographics details of the donor, or the details about the area from where they belong. 
# 
# The main idea of using context is to understand better about what context of donors are likely to donate. This data can help to establish some examples such as :  
# 
# 1. Donors which belong to metro cities having high unemployment rate are probably similar.  
# 2. Donors which belong to areas where population is higher tends to donate for health related projects  
# 3. Donors which belong to areas where median income is higher they tend to donate in rural areas  
# 4. Donors which belong to cities where median home value is lower, they tend to donate for seating projects   
# 
# 
# To implement this part I made use of the external data about donor areas and enriched details such as populatiom, population density, median home value, household income etc about their areas. 
# 
# Source of the dataset : https://www.unitedstateszipcodes.org/
# 
# In this kernel, I am using only a smaller data about the donor areas. But I have posted a [gist](https://gist.github.com/shivam5992/7d346da49930b1f4726f0700366a1dd2) in order to get data for the entire donorschoose donors data. 

# In[41]:


# slicing the users dataset so that it runs in kaggle kernel memory 
users = master_df[(master_df['Donor Is Teacher'] == 'No') & 
                  (master_df['Donor State'] == 'Texas') & 
                  (master_df['Posted Year'] == 2017)].reset_index()

users1 = users[:1000]
del master_df
gc.collect()


# #### <font color="red"> Load external data </font>

# In[42]:


## load the external dataset
external_context = pd.read_csv('../input/external-context/area_context_texas.csv')
features = list(set(external_context.columns) - set(['id', 'zipcode']))
agg_doc = {}
for feat in features:
    agg_doc[feat] = 'mean'

area_context = external_context.groupby('id').agg(agg_doc).reset_index().rename(columns = {'id' : 'Donor Zip'})
area_context = area_context[area_context['Housing Units'] != 0]
area_context.head()


# #### <font color="red">Clustering the external data </font>
# 
# The idea here is that we need to create additional feature which reflects the context of the donor area (or something else). One technique which can help here is the clustering of external data and use the clusters as the features. Lets do that using K-means clustering.  
# 
# Find the right K for K-means clustering of the data using Elbow method. 

# In[43]:


features = list(set(area_context.columns) - set(['Donor Zip']))
inretia = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    kmeans.fit(area_context[features])
    inretia.append(kmeans.inertia_)
plt.plot(range(1,11),inretia)
plt.title('Finding the Optimal Number of Clusters')
plt.xlabel('Number of clusters')
plt.xlabel('Kmeans Inretia')
plt.show()


# From the graph, we can see that this data can be clustered better with K = 3

# In[44]:


# apply kmeans clustering 
kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=0) 
area_context['area_context_cluster'] = kmeans.fit_predict(area_context[features])

# merge with the donors data
users1['Donor Zip'] = users1['Donor Zip'].astype(str)
area_context['Donor Zip'] = area_context['Donor Zip'].astype(str)

users1 = users1.merge(area_context[['Donor Zip', 'area_context_cluster']], on="Donor Zip", how="left")
area_context[['Donor Zip', 'area_context_cluster']].head(10)


# So basically we have now generated additional feature which can be added to the donors profiles and can be used to find similar users, as we will see in the next sections. In the similar manner, with more additional data, more context clusters can be created and added to the donors profile. 
# 
# - The second approach:
# 
# While it is true that donors having single project donation will not have enough data to create a good profile, but atleast they have some data which can be used to understand the preferences / likes / behaviour of the donor. So in thie approach, I am not removing the donors having single donation. 
# 
# ## <font color="red"> 5.2 Feature Engineering - Donor Profiles</font>
# 
# Lets create the word embedding vectors for all the projects in which donors have donated. Later on we will use these vector and aggregate them for obtaining average project vector for donor. 

# In[45]:


users1['Project Essay'] = users1['Project Essay'].fillna(" ")
utrain = users1['Project Essay'].values

utrain_embeddings = [generate_doc_vectors(x) for x in tqdm(utrain)]
utrain_embeddings = np.array(utrain_embeddings)
utrain_embeddings = scl.fit_transform(utrain_embeddings)

del utrain
gc.collect()


# ## <font color="red">5.3 Aggregate Donors Data - Create Donors Profiles </font>
# 
# Lets aggregate the users data and obtain the features related to donors based on their past donations. I am using the following features: 
# 
# 1. Donation Amount : Min, Max, Median, Mean  
# 2. Project Cost : Min, Max, Median, Mean  
# 3. Number of Projects Donated  
# 4. Project Subject Category Distribution  
# 5. Project Grade Level Distribution  
# 6. Project Essays - Average Word Embedding Vector  

# In[46]:


## handle few missing values 
users1['Project Type'] = users1['Project Type'].fillna("Teacher Led")
users1['Project Subject Category Tree'] = users1['Project Subject Category Tree'].fillna(" ")
users1['area_context_cluster'] = users1['area_context_cluster'].astype(str)

## aggregate the donors and their past donations in order to create their donor - profiles
user_profile = users1.groupby('Donor ID').agg({'Donation Amount' : ['min', 'max', 'mean', 'median'],
                                               'cost' : ['min', 'max', 'mean', 'median'], 
                                      'Project Subject Category Tree' : lambda x: ", ".join(x), 
                                      'Project ID' : lambda x: ",".join(x), 
                                      'School Metro Type' : lambda x: ",".join(x), 
                                      'Project Title' : lambda x: ",".join(x), 
                                      'area_context_cluster' : lambda x: ",".join(x), 
                                      'School Percentage Free Lunch' : 'mean',
                                      'Project Grade Level Category' : lambda x : ",".join(x),
                                      'Project Type' : 'count'}
                                    ).reset_index().rename(columns={'Project Type' : "Projects Funded"})


# In[47]:


## flatten the features of every donor

def get_map_features(long_text):
    a = long_text.split(",")
    a = [_.strip() for _ in a]
    mapp = dict(Counter(a))
    return mapp
    
user_profile['Category_Map'] = user_profile['Project Subject Category Tree']['<lambda>'].apply(get_map_features)
user_profile['Projects_Funded'] = user_profile['Project ID']['<lambda>'].apply(get_map_features)
user_profile['GradeLevel_Map'] = user_profile['Project Grade Level Category']['<lambda>'].apply(get_map_features)
user_profile['AreaContext_Map'] = user_profile['area_context_cluster']['<lambda>'].apply(get_map_features)
user_profile['SchoolMetroType_Map'] = user_profile['School Metro Type']['<lambda>'].apply(get_map_features)
user_profile = user_profile.drop(['Project Grade Level Category', 'Project Subject Category Tree',  'School Metro Type', 'Project ID', 'area_context_cluster'], axis=1)

user_profile.head()


# ## <font color="red">5.4 Create Donor Feature Vector </font>
# 
# Take the average of project vectors in which users have donated 

# In[48]:


def get_average_vector(project_ids):
    ids = list(project_ids.keys())
    
    donor_proj_vec = []
    for idd in ids:        
        unique_proj_ids = users1[users1['Project ID'] == idd].index.tolist()[0]
        donor_proj_vec.append(utrain_embeddings[unique_proj_ids])
    proj_vec = np.array(donor_proj_vec).mean(axis=0)
    return proj_vec 

user_profile['project_vectors'] = user_profile['Projects_Funded'].apply(lambda x : get_average_vector(x))
user_profile[['Donor ID', 'project_vectors']].head(10)


# So, now we have created the nodes for the donors graph and the projects graph. Every node repersents the information about a particular donor / project. In the next section, lets create the interaction matrices which will help to create the edges between the nodes in the donors / projects graphs. 
# 
# # <font color="red">6. Creating the Interaction Matrices</font>
# 
# Interaction matrices repersents the relationships between the any two nodes of the graphs. For example, consider a n * n project - project matrix in which every row repersents a project, every column repersent other project and the value at cell i, j repersent the interaction score of the two projects.  
# 
# One way to measure the interaction score is to compute the similarity among the two nodes. For example, if two projects has higher number of similar and correlated features, the similarity score for them will be higher, and so is the interaction score. A number of different techniques can be used to measure the similarity among the two nodes / vectors such as Jaccard Similarity, Ngram Similarity, Cosine Similarity etc. In this kernel I am using cosine similarity. 
# 
# ## <font color="red">6.1 Project - Project Edges </font>
# 
# ### <font color="red">6.1.1 Create Project Interactions </font>  
# 
# Lets first compute the project interaction scores using cosine similarities. Save the results in a variable. 
# 
# **Note -** At this point, while computing the similarity, I am currently using text features to measure the similarities. Other non text features will be used later to filter the recommendations and predictions in the next sections.

# In[49]:


# compute the project project interactions 
project_interactions = linear_kernel(xtrain_embeddings, xtrain_embeddings)


# ### <font color="red">6.1.2 Find top similar nodes  </font>  
# 
# Next, iterate for every node (project) and compute its top similar nodes

# In[50]:


# create the edges of one node with other most similar nodes  
project_edges = {}
for idx, row in projects_mini.iterrows():
    similar_indices = project_interactions[idx].argsort()[:-100:-1]
    similar_items = [(project_interactions[idx][i], projects_mini['Project ID'][i]) for i in similar_indices]
    project_edges[row['Project ID']] = similar_items[:20]


# ### <font color="red">6.1.3 Write a function to obtain similar nodes </font>  
# 
# Lets write a function which can be used to obtain the most similar project, similarity or the interaction score along with the project features. 

# In[51]:


def get_project(id):
    return projects_mini.loc[projects_mini['Project ID'] == id]['Project Title'].tolist()[0]

def similar_projects(project_id, num):
    print("Project: " + get_project(project_id))
    print("")
    print("Similar Projects: ")
    print("")
    recs = project_edges[project_id][1:num]
    for rec in recs:
        print(get_project(rec[1]) + " (score:" + str(rec[0]) + ")")


# Lets obtain some projects and their similar projects 

# In[52]:


similar_projects(project_id="a0446d393feaadbeb32cd5c3b2b36d45", num=10)


# In[53]:


similar_projects(project_id="83b4f3fbe743cb12ae2be7347ef03ecb", num=10)


# ## <font color="red">6.2 Donor - Donor Edges </font>  
# 
# ### <font color="red">6.2.1 Create Donor Interactions </font>  
# 
# Similar to project project interactions, we can compute the donor - donor interactions using their profiles and the context.

# In[54]:


user_embeddings = user_profile['project_vectors'].values

user_embeddings_matrix = np.zeros(shape=(user_embeddings.shape[0], 300))
for i,embedding in enumerate(user_embeddings):
    user_embeddings_matrix[i] = embedding

donors_interactions = linear_kernel(user_embeddings_matrix, user_embeddings_matrix)


# ### <font color="red">6.2.2 Find top similar nodes</font>  
# 
# Next, iterate for every node (donor) and compute its top similar nodes

# In[55]:


user_edges = {}
for idx, row in user_profile.iterrows():
    similar_indices = donors_interactions[idx].argsort()[:-10:-1]

    similar_items = [(float(donors_interactions[idx][i]), list(user_profile['Donor ID'])[i]) for i in similar_indices]
    user_edges[row['Donor ID'][0]] = similar_items[1:]


# ### <font color="red">6.2.3 Write a function to obtain the similar nodes </font>  
# 
# Lets write a function which can be used to obtain the most similar donors, similarity or the interaction score along with the donor features. 

# In[56]:


def get_donor(id):
    return user_profile.loc[user_profile['Donor ID'] == id]['Donor ID'].tolist()[0]

def similar_users(donor_id, num):
    print("Donor: " + get_donor(donor_id))
    print ("Projects: " + str(user_profile[user_profile['Donor ID'] == donor_id]['Project Title']['<lambda>'].iloc(0)[0]))

    print("")
    print("Similar Donors: ")
    print("")    
    recs = user_edges[donor_id][:num]
    for rec in recs:
        print("DonorID: " + get_donor(rec[1]) +" | Score: "+ str(rec[0]) )
        print ("Projects: " + str(user_profile[user_profile['Donor ID'] == rec[1]]['Project Title']['<lambda>'].iloc(0)[0]))
        print   ("")


# Lets view some donors and their similar donors

# In[57]:


similar_users(donor_id="fee882faa77bc6691bd24d4d5abd5733", num=5)


# In[58]:


similar_users(donor_id="d52242e9d5006fb97fcdb5565982f0ad", num=5)


# So we can see that model is picking the results having similar content features. 
# 
# Donors having similar features as of the given project are recommended. Example - donors from the projects **"Books Forever", "For the Love of Reading",  "We like Buddy Books and We Cannot Lie"** are suggested as one part of the set, This is because the features of the donors who donated in these projects have matched with the features of the given project.  
# 
# In case of Project titled **Learning through Art**, similar projects also had features related to arts / learning etc. for example - **Printmaking Production**, **Art & Crafts Materials for PreK** etc. For project related to food and health category, other similar projects were also picked from the same categories such as breakfast options, food options, heatlhy styles etc. 
# 
# ## <font color="red">7. Building the Graphs </font>
# 
# In the previous sections, we have created the nodes for donors and projects, interaction matrices for donors and projects. Now we can create the graph like structures to store this information. One plus point of using graphs is that it can help to find the connections really quick. 
# 
# ## <font color="red">7.1 Donors Graph </font>
# 
# Write a class for creating Donors Graph. The main functions that I will use are :
# 
#     _create_node : function to add a new node in the graph  
# 
#     _create_edges : function to create the edges among the nodes  
# 
#     _view_graph : function to view the graph  

# In[59]:


class DonorsGraph():
    """
    Class to create the graph for donors and save their information in different nodes.
    """
    
    
    def __init__(self, graph_name):
        self.graph = {}
        self.graph_name = graph_name
    
    # function to add new nodes in the graph
    def _create_node(self, node_id, node_properties):
        self.graph[node_id] = node_properties 
    
    # function to view the nodes in the graph
    def _view_nodes(self):
        return self.graph
    
    # function to create edges
    def _create_edges(self, node_id, node_edges):
        if node_id in self.graph:
            self.graph[node_id]['edges'] = node_edges


# ### <font color="red">7.1.1 Add the Donors Nodes </font>  
# 
# Initialize the donors graph and iterate in donor proflies to add the nodes with their properties

# In[60]:


## initialize the donors graph
dg = DonorsGraph(graph_name = 'donor')

## iterate in donor profiles and add the nodes
for idx, row in user_profile.iterrows():
    node_id = row['Donor ID'].tolist()[0]
    node_properties = dict(row)
    dg._create_node(node_id, node_properties)


# Lets view how the features of a node looks like.
# 
# Aggregated project vectors:

# In[61]:


node = dg._view_nodes()['12d74c3cd5f21ed4b17c781da828d076']
node[('project_vectors','')][0:50]


# Aggregated other features

# In[62]:


del node[('project_vectors','')]
node


# ### <font color="red">7.1.2 Add the Edges among Donors Nodes </font>  
# 
# Using interaction matrices created earlier, create the edges among donor nodes 

# In[63]:


def get_donor(id):
    return user_profile.loc[user_profile['Donor ID'] == id]['Donor ID'].tolist()[0]

def get_similar_donors(donor_id, num):
    # improve this algorithm - > currently only text, add other features as well 
    recs = user_edges[donor_id][:num]    
    return recs 

for idx, row in user_profile.iterrows():
    node_id = row['Donor ID'].tolist()[0]
    node_edges = get_similar_donors(donor_id=node_id, num=5)
    dg._create_edges(node_id, node_edges)


# Lets view the updated nodes

# In[64]:


dg._view_nodes()['00b3c149822c79e4fca9be0bea5c900c']['edges']


# ## <font color="red">7.2 Projects Graph </font>
# 
# Write a class for creating Projects Graph. The main functions are similar to donors graph :
# 
#     _create_node : function to add a new node in the graph  
# 
#     _create_edges : function to create the edges among the nodes  
# 
#     _view_graph : function to view the graph  

# In[65]:


class ProjectsGraph():
    def __init__(self, graph_name):
        self.graph = {}
        self.graph_name = graph_name
        
    def _create_node(self, node_id, node_properties):
        self.graph[node_id] = node_properties 
    
    def _view_nodes(self):
        return self.graph
    
    def _create_edges(self, node_id, node_edges):
        if node_id in self.graph:
            self.graph[node_id]['edges'] = node_edges


# ### <font color="red">7.2.1 Add the Project Nodes </font>  
# 
# Initialize the project graph and iterate in project proflies to add the nodes with their properties

# In[66]:


pg = ProjectsGraph(graph_name = 'projects')

for idx, row in projects_mini.iterrows():
    node_id = row['Project ID']
    node_properties = dict(row)
    del node_properties['Project Essay']
    del node_properties['Project Need Statement'] 
    del node_properties['Project Short Description']
    pg._create_node(node_id, node_properties)


# Lets view a node 

# In[67]:


pg._view_nodes()['83b4f3fbe743cb12ae2be7347ef03ecb']


# ### <font color="red">7.2.2 Add the Edges among Project Nodes </font>  
# 
# Using interaction matrices created earlier, create the edges among project nodes 

# In[68]:


def get_similar_projects(project_id, num):
    recs = project_edges[project_id][:num]
    return recs 

for idx, row in projects_mini.iterrows():
    node_id = row['Project ID']
    node_edges = get_similar_projects(project_id=node_id, num=5)
    pg._create_edges(node_id, node_edges)


# Lets view the edges of a node

# In[69]:


pg._view_nodes()['83b4f3fbe743cb12ae2be7347ef03ecb']['edges']


# Lets view the graphs
# 
# ### <font color="red">7.2.2.1 Donors Graph</font>

# In[70]:


from IPython.core.display import display, HTML, Javascript
import IPython.display
import json

nodes = []
links = []

nodes.append({'id' : 'Donors', 'group' : 2, 'size' : 20 })
for key, val in list(dg._view_nodes().items())[:50]:
    if len(val['edges']) == 0:
        continue
    nodes.append({'id' : key, 'group' : 2, 'size' : 15})
    links.append({"source" : "Donors", "target" : key, "value" : 10})
    
    for node in val['edges']:
        nodes.append({'id' : node[1], 'group' : 2, 'size' : 12})
        
        sv = np.log(node[0])
        ew = 10
        if sv > 6:
            ew = 100
        elif sv > 5:
            ew = 20
        elif sv > 4:
            ew = 15
        else:
            ew = 10
                    
        links.append({"source": key, "target": node[1], "value": ew})
doc = {'nodes' : nodes, 'links' : links}
with open("donorg.json", "w") as fout:
    fout.write(json.dumps(doc))
    
    

nodes = []
links = []
nodes.append({'id' : 'Projects', 'group' : 0, 'size' : 20, "title" : "Projects" })
for key, val in list(pg._view_nodes().items())[:75]:
    if len(val['edges']) == 0:
        continue

    nodes.append({'id' : key, 'group' : 0, 'size' : 15})
    links.append({"source" : "Projects","title" : val['Project Title'], "target" : key, "value" : 10})
    for node in val['edges']:
        title = projects_mini[projects_mini['Project ID'] == node[1]]['Project Title'].iloc(0)[0]
        nodes.append({'id' : node[1], 'group' : 2, 'size' : 12, "title": title})
        links.append({"source": key, "target": node[1], "value": 8})
doc = {'nodes' : nodes, 'links' : links}
with open("projectg.json", "w") as fout:
    fout.write(json.dumps(doc))


# In[71]:


html7="""<!DOCTYPE html>
<meta charset="utf-8">
<style>

.links line {
  stroke: #999;
  stroke-opacity: 0.8;
}
.node text {
  pointer-events: none;
  font: 10px sans-serif;
}

.tooldiv {
    display: inline-block;
    width: 120px;
    background-color: white;
    color: #000;
    text-align: center;
    padding: 5px 0;
    border-radius: 6px;
    z-index: 1;
}
.nodes circle {
  stroke: #fff;
  stroke-width: 1.5px;
}

div.tooltip {	
    position: absolute;			
    text-align: center;			
    width: 250px;					
    height: 40px;					
    padding: 2px;				
    font: 12px sans-serif;		
    background: lightsteelblue;	
    border: 0px;		
    border-radius: 8px;			
    pointer-events: none;			
}

</style>
<svg id="donorg" width="860" height="760"></svg>"""

js7="""require.config({
    paths: {
        d3: "https://d3js.org/d3.v4.min"
     }
 });
 
require(["d3"], function(d3) {// Dimensions of sunburst.
 
var svg = d3.select("#donorg"),
    width = +svg.attr("width"),
    height = +svg.attr("height");

var color = d3.scaleOrdinal(d3.schemeCategory20);

var simulation = d3.forceSimulation()

    .force("link", d3.forceLink().id(function(d) { return d.id; }).distance(20).strength(1))
    .force("charge", d3.forceManyBody().strength(-155))
    .force("center", d3.forceCenter(width / 2, height / 2));

d3.json("donorg.json", function(error, graph) {
  if (error) throw error;

  var link = svg.append("g")
      .attr("class", "links")
    .selectAll("line")
    .data(graph.links)
    .enter().append("line")
      .attr("stroke-width", function(d) { return Math.sqrt(d.value); });

// Define the div for the tooltip
var div = d3.select("body").append("div")	
    .attr("class", "tooltip")				
    .style("opacity", 0);

  var node = svg.append("g")
      .attr("class", "nodes")
    .selectAll("circle")
    .data(graph.nodes)
    .enter().append("circle")
      .attr("r", function(d) {return d.size})
      .attr("fill", function(d) { return color(d.group); })
      .call(d3.drag()
          .on("start", dragstarted)
          .on("drag", dragged)
          .on("end", dragended)).on("mouseover", function(d) {		
            div.transition()		
                .duration(200)		
                .style("opacity", .9);		
            div.html(d.id)
                .style("left", (d3.event.pageX) + "px")		
                .style("top", (d3.event.pageY - 28) + "px");	
            })					
        .on("mouseout", function(d) {		
            div.transition()		
                .duration(500)		
                .style("opacity", 0);	
        });
          
    

  simulation
      .nodes(graph.nodes)
      .on("tick", ticked);
      

  simulation.force("link")
      .links(graph.links);

  function ticked() {
    link
        .attr("x1", function(d) { return d.source.x; })
        .attr("y1", function(d) { return d.source.y; })
        .attr("x2", function(d) { return d.target.x; })
        .attr("y2", function(d) { return d.target.y; });

    node
        .attr("cx", function(d) { return d.x; })
        .attr("cy", function(d) { return d.y; });
  }
});

function dragstarted(d) {
  if (!d3.event.active) simulation.alphaTarget(0.3).restart();
  d.fx = d.x;
  d.fy = d.y;
}

function dragged(d) {
  d.fx = d3.event.x;
  d.fy = d3.event.y;
}

function dragended(d) {
  if (!d3.event.active) simulation.alphaTarget(0);
  d.fx = null;
  d.fy = null;
}
 });
"""

h = display(HTML(html7))
j = IPython.display.Javascript(js7)
IPython.display.display_javascript(j)


# ### <font color="red">7.2.2.2 Projects Graph </font>

# In[72]:


html8="""<!DOCTYPE html>
<meta charset="utf-8">
<style>

.links line {
  stroke: #999;
  stroke-opacity: 0.8;
}
.node text {
  pointer-events: none;
  font: 10px sans-serif;
}

.tooldiv {
    display: inline-block;
    width: 120px;
    background-color: white;
    color: #000;
    text-align: center;
    padding: 5px 0;
    border-radius: 6px;
    z-index: 1;
}
.nodes circle {
  stroke: #fff;
  stroke-width: 1.5px;
}

div.tooltip {	
    position: absolute;			
    text-align: center;			
    width: 250px;					
    height: 40px;					
    padding: 2px;				
    font: 12px sans-serif;		
    background: lightsteelblue;	
    border: 0px;		
    border-radius: 8px;			
    pointer-events: none;			
}

</style>
<svg id="projectg" width="860" height="760"></svg>"""

js8="""
 
require(["d3"], function(d3) {// Dimensions of sunburst.
 
var svg = d3.select("#projectg"),
    width = +svg.attr("width"),
    height = +svg.attr("height");

var color = d3.scaleOrdinal(d3.schemeCategory20);

var simulation = d3.forceSimulation()

    .force("link", d3.forceLink().id(function(d) { return d.id; }).distance(20).strength(1))
    .force("charge", d3.forceManyBody().strength(-155))
    .force("center", d3.forceCenter(width / 2, height / 2));

d3.json("projectg.json", function(error, graph) {
  if (error) throw error;

  var link = svg.append("g")
      .attr("class", "links")
    .selectAll("line")
    .data(graph.links)
    .enter().append("line")
      .attr("stroke-width", function(d) { return Math.sqrt(d.value); });

// Define the div for the tooltip
var div = d3.select("body").append("div")	
    .attr("class", "tooltip")				
    .style("opacity", 0);

  var node = svg.append("g")
      .attr("class", "nodes")
    .selectAll("circle")
    .data(graph.nodes)
    .enter().append("circle")
      .attr("r", function(d) {return d.size})
      .attr("fill", function(d) { return color(d.group); })
      .call(d3.drag()
          .on("start", dragstarted)
          .on("drag", dragged)
          .on("end", dragended)).on("mouseover", function(d) {		
            div.transition()		
                .duration(200)		
                .style("opacity", .9);		
            div.html(d.title)
                .style("left", (d3.event.pageX) + "px")		
                .style("top", (d3.event.pageY - 28) + "px");	
            })					
        .on("mouseout", function(d) {		
            div.transition()		
                .duration(500)		
                .style("opacity", 0);	
        });
          
    

  simulation
      .nodes(graph.nodes)
      .on("tick", ticked);
      

  simulation.force("link")
      .links(graph.links);

  function ticked() {
    link
        .attr("x1", function(d) { return d.source.x; })
        .attr("y1", function(d) { return d.source.y; })
        .attr("x2", function(d) { return d.target.x; })
        .attr("y2", function(d) { return d.target.y; });

    node
        .attr("cx", function(d) { return d.x; })
        .attr("cy", function(d) { return d.y; });
  }
});

function dragstarted(d) {
  if (!d3.event.active) simulation.alphaTarget(0.3).restart();
  d.fx = d.x;
  d.fy = d.y;
}

function dragged(d) {
  d.fx = d3.event.x;
  d.fy = d3.event.y;
}

function dragended(d) {
  if (!d3.event.active) simulation.alphaTarget(0);
  d.fx = null;
  d.fy = null;
}
 });
"""

h = display(HTML(html8))
j = IPython.display.Javascript(js8)
IPython.display.display_javascript(j)


# Note - The centeral nodes are added so that graph remains intact. Also, similar type of graphs can be created with different geographies. Example - Graphs for different states. 
# 
# # <font color="red">8. Connecting Donors to Projects </font>
# 
# In this section, we will find the right donors of the projects using two methods - Graph Based Approach and Deep Neural Networks Approach. 
# 
# ## <font color="red">8.1 Graph based approach </font>
# 
# The main flow of generating recommendations using graphs is the following: 
# 
# >    1. Create the project vector for the new project. This project vector acts as a new node which can be added to the graphs in order to find its connections. 
# >    2. To find the most relevant donors of this project, find the edges of this new project node from the donors graph. This will give the first set of recommended donors.  
# >    3. Extend the recommended donors set by Finding the most similar projects of this new node from the projects graph and obtaining their donors.  
# >    4. Also from the donors graph, Find the similar donors of all the donors obtained from donors.  
# >    5. Finally, Make use of non text features to filter out the recommended donors.  

# In[73]:


def connect_project_donors(project_id):
        
    # get the project index
    proj_row = projects_mini[projects_mini['Project ID'] == project_id]
    proj_ind = proj_row.index

    # get the project vector
    proj_vector = xtrain_embeddings[proj_ind]
    
    # match the vector with the user vectors 
    cossim_proj_user = linear_kernel(proj_vector, user_embeddings_matrix)
    reverse_matrix = cossim_proj_user.T
    reverse_matrix = np.array([x[0] for x in reverse_matrix])
    similar_indices = reverse_matrix.argsort()[::-1]
    
    # filter the recommendations
    projects_similarity = []
    recommendations = []
    top_users = [(reverse_matrix[i], user_profile['Donor ID'][i]) for i in similar_indices[:10]]
    for x in top_users:
        user_id = x[1]
        user_row= user_profile[user_profile['Donor ID'] == user_id]
        
        ## to get the appropriate recommendations, filter them using other features 
        cat_count = 0
        
        ## Making use of Non Text Features to filter the recommendations
        subject_categories = proj_row['Project Subject Category Tree'].iloc(0)[0]
        for sub_cat in subject_categories.split(","):
            if sub_cat.strip() in user_row['Category_Map'].iloc(0)[0]:
                cat_count += user_row['Category_Map'].iloc(0)[0][sub_cat.strip()]

        grade_category = proj_row['Project Grade Level Category'].iloc(0)[0]
        if grade_category in user_row['Category_Map'].iloc(0)[0]:
            cat_count += user_row['Category_Map'].iloc(0)[0][grade_category]

        metro_type = proj_row['School Metro Type'].iloc(0)[0]
        if metro_type in user_row['SchoolMetroType_Map'].iloc(0)[0]:
            cat_count += user_row['SchoolMetroType_Map'].iloc(0)[0][metro_type]
        
        x = list(x)
        x.append(cat_count)
        recommendations.append(x)
        
        ## Find similar donors
        donor_nodes = dg._view_nodes()
        if x[1] in donor_nodes:
            recommendations.extend(donor_nodes[x[1]]['edges'])

    ## Find Similar Projects 
    project_nodes = pg._view_nodes()
    if project_id in project_nodes:
        projects_similarity.extend(project_nodes[project_id]['edges'])    

    return projects_similarity, recommendations
    
def get_recommended_donors(project_id):
    # Find the recommended donors and the similar projects for the given project ID 
    sim_projs, recommendations = connect_project_donors(project_id)

    # filter the donors who have already donated in the project
    current_donors = donations_df[donations_df['Project ID'] == project_id]['Donor ID'].tolist()

    # Add the donors of similar projects in the recommendation
    for simproj in sim_projs:
        recommendations.extend(connect_project_donors(simproj[1])[1])
    
    ######## Create final recommended donors dataframe 
    # 1. Most relevant donors for a project 
    # 2. Similar donors of the relevant donors 
    # 3. Donors of the similar project 
    
    recommended_df = pd.DataFrame()
    recommended_df['Donors'] = [x[1] for x in recommendations]
    recommended_df['Score'] = [x[0] for x in recommendations]
    recommended_df = recommended_df.sort_values('Score', ascending = False)
    recommended_df = recommended_df.drop_duplicates()

    recommended_df = recommended_df[~recommended_df['Donors'].isin(current_donors)]
    return recommended_df


# **Note -** In the above implementation, I used non text features to filter the recommendations, these can be used again according to the use-case. One can define what sort of filters they need to apply on the recommendations set.  
# 
# ### Lets view some results for the given projects 

# In[74]:


def _get_results(project_id):
    proj = projects_mini[projects_mini['Project ID'] == project_id]
    print ("Project ID: " + project_id )
    print ("Project Title: " + proj['Project Title'].iloc(0)[0])
    print ("")

    print ("Recommended Donors: ")
    recs = get_recommended_donors(project_id)
    donated_projects = []
    for i, row in recs.head(10).iterrows():
        donor_id = row['Donors']
        print (donor_id +" | "+ str(row['Score']))
        donor_projs = user_profile[user_profile['Donor ID'] == donor_id]['Project Title']['<lambda>'].iloc(0)[0]
        donor_projs = donor_projs.split(",")
        for donor_proj in donor_projs:
            if donor_proj not in donated_projects:
                donated_projects.append(donor_proj)
    print ("")
    print ("Previous Projects of the Recommended Donors: ")
    for proj in donated_projects:
        print ("-> " + proj)

project_id = "d20ec20b8e4de165476dfd0a68da0072"
_get_results(project_id)


# In[75]:


project_id = "ad51c22f0d31c7dc3103294bdd7fc9c1"
_get_results(project_id)


# So these were the recommendations which mainly used the content features of the projects and donors in order to establish right relationships among the two verticals.
# 
# # <font color="red">9.2 Connecting Donors - Projects using DeepNets</font>
# 
# "Donors who donated in these projects also donated in these other projects." 
# 
# In the previous part, A graph based content similarity matching techniques were used to match the donors for the projects. In this section, we will use a deep neural network to identify the donors having similar behaviours. This approach is very close to collaborative filtering approach. 
# 
# > **Question** - Are there any particular donors you want to target through this competition. Like first time donors or repeat donors or both?
# 
# > **Thomas Vo (from donorschoose.org)** - The short answer is no, we're not targeting a specific subset. This is because they are both important to our business, they behave very differently, and we'd like to improve our odds for both subsets. I will say though, that whereas we have systems in place to accommodate repeat donors, we are really not sure what to do with new donors. It's a sample size of one and the projects are ephemeral, which makes it especially difficult to figure out which projects would entice these donors.
# 
# 
# While it is true that this approach may not work very well for first time donors but it can work very well for the donors having multiple donation. One way to ensemble this approach for first time donors can be to find the similarities among the donors based on content and context which belong to the first time donors set. Following figure depicts how this approach works.
# 
# <br>
# 
# ![deepcolab.png](https://i.imgur.com/PpopDBB.png)
# 
# <br>
# 
# 
# The main idea behind this approach is to identify donors based on the similarity in their behaviours. In the donorschoose usecase the behavious can be indirectly represented by the donation amounts from donors in the projects in which they have donated in the past. 
# 
# Generating outcomes using DeepNets can be thought of as a regression problem where the model tries to predict the interaction scores for each project for each donor. The donors having high interaction scores with respect to a project are treated as the recommendations. In this approach, the model learns the donor and project embeddings as the features and predicts what could be the outcome of a donor - project combination. 
# 
# 
# ### Process Flow of Deep Net Model
# 
# - In the deepnets model, the embedding vectors (different from word embeddings) for the donors and one for the projects are created. 
# - These embedding vectors are merged together to give a single vector which repersents the latent infromation about a donor-project pair. 
# 
# Please note that these embeddings are different from the word embeddings. These embeddings are a repersentation of the behaviour of the donor with multiple projects and the interactions of multiple donors on a single project. For example, as shown  
# 
# ![donor-pro.png](https://i.imgur.com/1WOmUBk.png)
# 
# - The merged vector becomes the independent variable of the dataset. The target variable of this dataset is the logarithmic of donation amount or interaction score 
# 
# ![DeepNEt.png](https://i.imgur.com/ogdVSaj.jpg)
# 
# - During the training process, the embeddings parameters are learned which gives a latent representation of the donors / projects.  
# - The learned model is used to predict if a new donor-project combination has higher interaction score. If it is higher than the donor is reocmmended for the project. 
# 
# ### References used 
# 
# 1. [Neural Item Embedding For Collaborative Filtering] https://arxiv.org/pdf/1603.04259.pdf
# 2. [Neural Collaborative Filtering] https://www.comp.nus.edu.sg/~xiangnan/papers/ncf.pdf
# 3. [Fast.ai] http://www.fast.ai/
# 
# 
# ## <font color="red">9.2.1 Creating the Donor-Project-Interaction Matrix</font>
# 
# Create the interaction data frame - Project, Donors, InteractionScore2

# In[76]:


## create the interaction data frames
interactions = users[['Project ID', 'Donor ID', 'Donation Amount']]
interactions['interaction_score_2'] = np.log(interactions['Donation Amount'])


# ## <font color="red">9.2.2 PreProcessing</font>
# 
# For the model implementation in Keras, I mapped all the donor ids and project ids to an integer between 0 and either the total number of donors or the total number of projects.

# In[77]:


unique_donors = list(interactions['Donor ID'].value_counts().index)
donor_map = {}
for i, user in enumerate(unique_donors):
    donor_map[user] = i+1

unique_projs = list(interactions['Project ID'].value_counts().index)
proj_map = {}
for i, proj in enumerate(unique_projs):
    proj_map[proj] = i+1

tags = {'donors' : donor_map, 'projects' : proj_map}


# Once the Id maps have been created, now convert the actual project and donor ids in the integer forms

# In[78]:


def getID(val, tag):
    return tags[tag][val]
     
interactions['proj_id'] = interactions['Donor ID'].apply(lambda x : getID(x, 'donors'))
interactions['user_id'] = interactions['Project ID'].apply(lambda x : getID(x, 'projects'))


# Obtain the maximum number of donors and projects present in the dataset. This will be used to define the network architecture

# In[79]:


# remove the duplicate entries in the dataset 
max_userid = interactions['user_id'].drop_duplicates().max() + 1
max_movieid = interactions['proj_id'].drop_duplicates().max() + 1


# Make a random shuffling on the dataset

# In[80]:


shuffled_interactions = interactions.sample(frac=1., random_state=153)
PastDonors = shuffled_interactions['user_id'].values
PastProjects = shuffled_interactions['proj_id'].values
Interactions = shuffled_interactions['interaction_score_2'].values


# ## <font color="red">9.2.3 Create Model Architecture</font>
# 
# Lets create the model architecture. As discussed in the above sections, the deep neural network learns the embeddings of donors and projects.  
# 
# There are essentially three parts of this model.  
# 
# 1. Donor Embedding Architecture  
# 2. Project Embedding Architecture  
# 3. Multi Layer Perceptron Architecture   
# 

# In[81]:


def create_model(n_donors, m_projects, embedding_size):
    
    # add input layers for donors and projects
    donor_id_input = Input(shape=[1], name='donor')
    project_id_input = Input(shape=[1], name='project')

    # create donor and project embedding layers 
    donor_embedding = Embedding(output_dim=embedding_size, input_dim=n_donors,
                               input_length=1, name='donor_embedding')(donor_id_input)
    project_embedding = Embedding(output_dim=embedding_size, input_dim=m_projects,
                               input_length=1, name='project_embedding')(project_id_input)
    
    # perform reshaping on donor and project vectors 
    donor_vecs = Reshape([embedding_size])(donor_embedding)
    project_vecs = Reshape([embedding_size])(project_embedding)
    
    # concatenate the donor and project embedding vectors 
    input_vecs = Concatenate()([donor_vecs, project_vecs])
    
    # add a dense layer
    x = Dense(128, activation='relu')(input_vecs)
    
    # add the output layer
    y = Dense(1)(x)
    
    # create the model using inputs and outputs 
    model = Model(inputs=[donor_id_input, project_id_input], outputs=y)
    
    # compile the model, add optimizer function and loss function
    model.compile(optimizer='adam', loss='mse')  
    return model


# Obtain the model

# In[82]:


embedding_size = 10
model = create_model(max_userid, max_movieid, embedding_size)


# Lets view the model summary

# In[83]:


model.summary()


# Create a function which can be used to generate the interaction scores 

# In[84]:


def rate(model, user_id, item_id):
    return model.predict([np.array([user_id]), np.array([item_id])])[0][0]

def predict_rating(movieid, userid):
    return rate(model, movieid - 1, userid - 1)


# ## <font color="red">9.2.4 Train the model </font>
# 
# Train the model with early stopping callback

# In[85]:


## with more data, nb_epooch can also be increased
history = model.fit([PastDonors, PastProjects], Interactions, nb_epoch=2, validation_split=.20)


# In[86]:


min_val_loss, idx = min((val, idx) for (idx, val) in enumerate(history.history['val_loss']))
print ('Minimum RMSE at epoch', '{:d}'.format(idx+1), '=', '{:.4f}'.format(math.sqrt(min_val_loss)))


# ## <font color="red">9.2.5 Generate the predictions for a project </font>
# 
# Generate the predictions, before that perform some basic preprocessing.

# In[87]:


Past_Donors = users[['Donor ID', 'Donor City']]
Past_Projects = users[['Project ID', 'Project Title']]

Past_Donors = Past_Donors.drop_duplicates()
Past_Projects = Past_Projects.drop_duplicates()

Past_Donors['user_id'] = Past_Donors['Donor ID'].apply(lambda x : getID(x, 'donors'))
Past_Projects['proj_id'] = Past_Projects['Project ID'].apply(lambda x : getID(x, 'projects'))

## for this sample run, get common IDs from content based and 
## collaborative approaches to get the results together

list1 = list(projects_mini['Project ID'].values)
list2 = list(Past_Projects['Project ID'].values)
common_ids = list(set(list1).intersection(list2))


# ### <font color="red">Sample Predictions - 1</font>

# In[88]:


idd = proj_map['ad51c22f0d31c7dc3103294bdd7fc9c1']

user_ratings = interactions[interactions['proj_id'] == idd][['user_id', 'proj_id', 'interaction_score_2']]
user_ratings['predicted_amt'] = user_ratings.apply(lambda x: predict_rating(idd, x['user_id']), axis=1)
recommendations = interactions[interactions['user_id'].isin(user_ratings['user_id']) == False][['user_id']].drop_duplicates()
recommendations['predicted_amt'] = recommendations.apply(lambda x: predict_rating(idd, x['user_id']), axis=1)
recommendations['predicted_amt'] = np.exp(recommendations['predicted_amt'])
recommendations.sort_values(by='predicted_amt', ascending=False).merge(Past_Donors, on='user_id', how='inner', suffixes=['_u', '_m']).head(10)


# ### <font color="red">Final Results : combine the results of two approaches</font>

# In[89]:


project_id = "ad51c22f0d31c7dc3103294bdd7fc9c1"


# #### <font color="red">1. Recommendations using Content+Context Similarity</font>

# In[90]:


_get_results(project_id)


# #### <font color="red">2. Recommendations using Behavioural Similarity</font>

# In[91]:


title = projects_mini[projects_mini['Project ID'] == project_id]['Project Title'].iloc(0)[0]
print ("Project ID: " + project_id )
print ("Project Title: " + title)
print ("")
    
idd = proj_map[project_id]
user_ratings = interactions[interactions['proj_id'] == idd][['user_id', 'proj_id', 'interaction_score_2']]
user_ratings['predicted_amt'] = user_ratings.apply(lambda x: predict_rating(idd, x['user_id']), axis=1)
recommendations = interactions[interactions['user_id'].isin(user_ratings['user_id']) == False][['user_id']].drop_duplicates()
recommendations['predicted_amt'] = recommendations.apply(lambda x: predict_rating(idd, x['user_id']), axis=1)
recommendations['predicted_amt'] = np.exp(recommendations['predicted_amt'])
recs = recommendations.sort_values(by='predicted_amt', ascending=False).merge(Past_Donors, on='user_id', how='inner', suffixes=['_u', '_m']).head(10)

past_projs = []
print ("Donors based on Behavioural Similarity: ")
for i, row in recs.head(5).iterrows():
    print (row['Donor ID'] +" | Donated Amount: "+ str(row['predicted_amt']))
    dons = donations_df[donations_df['Donor ID'] == row['Donor ID']]
    for i,x in dons.head(3).iterrows():
        projs = projects_df[projects_df['Project ID'] == x['Project ID']]
        title = projs['Project Title'].iloc(0)[0]
        cost = projs['Project Cost'].iloc(0)[0]
        txt =title + " ($" + str(cost) + ")"
        past_projs.append(txt)

print ("")
print ("Past Projects and Donations: ")
for proj_ in past_projs:
    print (proj_)


# So in the final results we can see that donors are recommended from different characteristics. 
# 1. Donros whose features are similar to the given project are recommended. 
# 2. Donors of the projects which are contextually similar to a given project (for example - **Diverse Books for Dynamic Students**, **Thesaurus' Expand Vocabulary For English Language Learners!**, and **Teaching My Special Friends How To Communicate!** are contextually similar to **Native American Book Project**) are recommended. 
# 3. Donors having similar donors features are recommended.  
# 4. Donors having similar behaviours are recommended using their past interactions on projects.  

# ### <font color="red">Evaluation Metrics </font>  
# 
# After deployment of this complete model, several evaluation metrics can be used to check if the model is making significant changes or not. Following are the differet ideas: 
# 
# 1. Quality of Recommendations  - It is often helpful to manually evaluate if the generated outcomes align with the problem statement and the business logic. By looking at the quality of recommendations, one can check how good the recommendation model is working.  
# 2. Marketing Metrics   
#     2.1 Open Rates : How many donors opened the email.  
#     2.2 Clickthrough Rate : Total number of unique clicks per 100 delivered emails  
#     2.3 Unsubscrive Rate : How many donors unsubscribed after receiving the emails.  
#     2.4 Number of Donors making the donations after receiving the emails.  
# 3. Model Metrics  
#     3.1 Root Mean Squared Error : In the Deep Net which was used for generating the predicted donation amount for a given project - donor combincation, RMSE can help to evaluate the model. As in my plementation, we saw that after t epoochs the model RMSE was equal to 0.5. This can be further improved using more data. 
#     
# ### <font color="red">End Notes</font>
# 
# So in this kernel I implemented different ideas to tackle the problem statement. I took different ideas from a number of research papers,  articles and my thought process.  
# 
# Also, it is important to note that this kernel mainly showcases "The Hybrid approach to connect donors to projects" and how to implement it. The results / accuracies of the output might not seem decent in some examples because it is not run on entire dataset.  However, if entire dataset of donors, projects, and external data is taken and the models are trained with more number of epoochs the results will improve further. 
# 
# The references are given below: 
# 
# 1. Content Based Recommendations (http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.130.8327&rep=rep1&type=pdf)
# 2. Graph Based Collaborative Ranking (http://www.ra.ethz.ch/cdstore/www10/papers/pdf/p519.pdf)  
# 3. Text Similarity with Word Embeddings (https://staff.fnwi.uva.nl/m.derijke/wp-content/papercite-data/pdf/kenter-short-2015.pdf)
# 4. Word Embeddings for Text Similarity (http://ad-publications.informatik.uni-freiburg.de/theses/Bachelor_Jon_Ezeiza_2017.pdf)  
# 5. Context Aware Recommendations Overview (https://www.slideshare.net/irecsys/contextaware-recommendation-a-quick-view)
# 5. Neural Collaborative Filtering (https://www.comp.nus.edu.sg/~xiangnan/papers/ncf.pdf)  
# 6. My Previous Research Paper on - Topic Modelling Driven Content Based Recommendation Engine (https://www.sciencedirect.com/science/article/pii/S1877050917326960)  
# 

# In[ ]:





# In[ ]:




