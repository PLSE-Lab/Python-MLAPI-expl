#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from lightfm import LightFM
from lightfm.data import Dataset
import plotly as py
import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly.offline import *
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
init_notebook_mode(connected=True)
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


def data_preprocessing():
    ratings = pd.read_csv("../input/ratings.csv")
    issue_features = pd.read_csv("../input/issues.csv",usecols=['id','issue_id','work_id','average_rating','ratings_1','ratings_2','ratings_3'])
    issue_features.rename(index=str, columns={'ratings_1':'i_feat1','ratings_2':'i_feat2','ratings_3':'i_feat3'}, inplace= True)
    empdf=pd.read_csv('../input/employee.csv')
    fdf=pd.merge(empdf,ratings,on='user_id')
    fdf.drop(['rating_4','rating_5'],axis=1,inplace=True)
    fdf.rename(index=str, columns={'rating_1':'emp_feat1','rating_2':'emp_feat2','rating_3':'emp_feat3'}, inplace= True)
    return pd.merge(fdf,issue_features,on='issue_id')


# In[ ]:


def build_lightfm_dataset(df):
    dataset = Dataset()
    dataset.fit(df['user_id'],
            df['issue_id'],
            ((rows['emp_feat1'], rows['emp_feat1'], rows['emp_feat3']) for _,rows in df.iterrows()),
            ((rows['i_feat1'], rows['i_feat1'], rows['i_feat3']) for _,rows in df.iterrows())
           )
    return dataset


# In[ ]:


def create_interaction_matrix(df):
    '''
    Function to create an interaction matrix dataframe from transactional type interactions
    Required Input -
        - df = Pandas DataFrame containing user-item interactions
        - user_col = column name containing user's identifier
        - item_col = column name containing item's identifier
        - rating col = column name containing user feedback on interaction with a given item
        - norm (optional) = True if a normalization of ratings is needed
        - threshold (required if norm = True) = value above which the rating is favorable
    Expected output - 
        - Pandas dataframe with user-item interactions ready to be fed in a recommendation algorithm
    '''
    interactions = df.groupby(['user_id', 'issue_id'])['rating']             .sum().unstack().reset_index().             fillna(0).set_index('user_id')
    return interactions


# In[ ]:


def build_interaction_matrix(df):
    (interactions, weights) = dataset.build_interactions((rows['user_id'], rows['issue_id']) for _,rows in df.iterrows())
    return interactions, weights


# In[ ]:


def build_feature_matrix():
    empdf=pd.read_csv('../input/employee.csv',index_col='user_id').join(intertest,on='user_id',how='inner').sort_values(by='user_id').iloc[:,:3]
    issuedf=pd.read_csv('../input/issues.csv').join(intertest.transpose(),how='right').iloc[:,4:7]
    issue_feat=csr_matrix(issuedf.values)
    user_feat=csr_matrix(empdf.values)
    return issue_feat,user_feat


# In[ ]:


def runMF(interactions, item_feat, user_feat, n_components=30, loss='warp', k=15, epoch=30,n_jobs = 4):
    '''
    Function to run matrix-factorization algorithm
    Required Input -
        - interactions = dataset create by create_interaction_matrix
        - n_components = number of embeddings you want to create to define Item and user
        - loss = loss function other options are logistic, brp
        - epoch = number of epochs to run 
        - n_jobs = number of cores used for execution 
    Expected Output  -
        Model - Trained model
    '''
    x = csr_matrix(interactions.values)
    model = LightFM(no_components= n_components, loss=loss,k=k)
    model.fit(x,epochs=epoch,num_threads = n_jobs,user_features=user_feat,item_features=item_feat)
    return model


# In[ ]:


def sample_recommendation_user(model, interactions, user_id, user_dict, 
                               item_dict,threshold = 0,nrec_items = 10, show = False):
    '''
    Function to produce user recommendations
    Required Input - 
        - model = Trained matrix factorization model
        - interactions = dataset used for training the model
        - user_id = user ID for which we need to generate recommendation
        - user_dict = Dictionary type input containing interaction_index as key and user_id as value
        - item_dict = Dictionary type input containing item_id as key and item_name as value
        - threshold = value above which the rating is favorable in new interaction matrix
        - nrec_items = Number of output recommendation needed
    Expected Output - 
        - Prints list of items the given user has already bought
        - Prints list of N recommended items  which user hopefully will be interested in
    '''
    issuefeat,userfeat=build_feature_matrix()
    n_users, n_items = interactions.shape
    user_x = user_dict[user_id]
    scores = pd.Series(model.predict(user_x,np.arange(n_items),user_features=userfeat,item_features=issuefeat))
    scores.index = interactions.columns
    scores = list(pd.Series(scores.sort_values(ascending=False).index))
    
    known_items = list(pd.Series(interactions.loc[user_id,:]                                  [interactions.loc[user_id,:] > threshold].index) 								 .sort_values(ascending=False))
    
    scores = [x for x in scores if x not in known_items]
    return_score_list = scores[0:nrec_items]
    known_items = list(pd.Series(known_items).apply(lambda x: item_dict[x]))
    scores = list(pd.Series(return_score_list).apply(lambda x: item_dict[x]))
    if show == True:
        print("Known Likes:")
        counter = 1
        for i in known_items:
            print(str(counter) + '- ' + i)
            counter+=1

        print("\n Recommended Items:")
        counter = 1
        for i in scores:
            print(str(counter) + '- ' + i)
            counter+=1
    return return_score_list


# In[ ]:


def sample_recommendation_item(model,interactions,item_id,user_dict,item_dict,number_of_user,itemfeat):
    '''
    Funnction to produce a list of top N interested users for a given item
    Required Input -
        - model = Trained matrix factorization model
        - interactions = dataset used for training the model
        - item_id = item ID for which we need to generate recommended users
        - user_dict =  Dictionary type input containing interaction_index as key and user_id as value
        - item_dict = Dictionary type input containing item_id as key and item_name as value
        - number_of_user = Number of users needed as an output
    Expected Output -
        - user_list = List of recommended users 
    '''
    n_users, n_items = interactions.shape
    x = np.array(interactions.columns)
    scores = pd.Series(model.predict(np.arange(interactions.shape[0]),user_features=userfe,item_features=csr_matrix(np.broadcast_to(itemfeat,(interactions.shape[0],len(itemfeat)))),item_ids=np.repeat(1,interactions.shape[0])))
    user_list = list(scores.sort_values(ascending=False).head(number_of_user).index)
    return user_list 


# In[ ]:





# In[ ]:


def create_user_dict(interactions):
    '''
    Function to create a user dictionary based on their index and number in interaction dataset
    Required Input - 
        interactions - dataset create by create_interaction_matrix
    Expected Output -
        user_dict - Dictionary type output containing interaction_index as key and user_id as value
    '''
    user_id = list(interactions.index)
    user_dict = {}
    counter = 0 
    for i in user_id:
        user_dict[i] = counter
        counter += 1
    return user_dict


# In[ ]:


def generate_user_similarity_matrix(model,interactions,num_emp):
    _,userfe=build_feature_matrix()
    _,latusr=modelnew.get_user_representations(userfe)
    spsr=pd.DataFrame(cosine_similarity(csr_matrix(latusr[:num_emp,:])))
    spsr.columns=intertest.index[:num_emp]
    spsr.index=intertest.index[:num_emp]
    return spsr


# In[ ]:


def generate_item_similarity_matrix(model,interactions,num_item):
    issuefe,_=build_feature_matrix()
    _,latitm=modelnew.get_item_representations(issuefe)
    spsr=pd.DataFrame(cosine_similarity(csr_matrix(latitm[:num_item,:])))
    spsr.columns=interactions.columns[:num_item]
    spsr.index=interactions.columns[:num_item]
    return spsr


# In[ ]:


def create_item_dict(interactions):
    '''
    Function to create an item dictionary based on their item_id and item name
    Required Input - 
        - df = Pandas dataframe with Item information
        - id_col = Column name containing unique identifier for an item
        - name_col = Column name containing name of the item
    Expected Output -
        item_dict = Dictionary type output containing item_id as key and item_name as value
    '''
    issue_id = list(interactions.columns)
    issue_dict = {}
    counter = 0 
    for i in issue_id:
        issue_dict[i] = counter
        counter += 1
    return issue_dict


# In[ ]:


def create_heatmap_emp(empdf,title='None Provided'):
    data = [
    go.Heatmap(
        z=empdf.values[:,:],
        x=empdf.columns[:],
        y=empdf.columns[:],
        colorscale='Viridis',
    )
    ]

    layout = go.Layout(
    title=title,
    xaxis = dict(ticks='', nticks=len(empdf.columns)),
    yaxis = dict(ticks='',nticks=len(empdf.columns) )
    )

    fig = go.Figure(data=data, layout=layout)
    py.offline.iplot(fig, filename='basic-heatmap')


# In[ ]:


df1=data_preprocessing()
intertest= create_interaction_matrix(df1)
userdict=create_user_dict(intertest)
itemdict=create_item_dict(intertest)
issuefe,userfe=build_feature_matrix()
modelnew=runMF(interactions=intertest, user_feat=userfe, item_feat= issuefe)
emp=314
print("Recommended issue for employee %d :"%emp)
print(sample_recommendation_user(modelnew,intertest,user_id=emp,user_dict=userdict,item_dict=itemdict))
itemno=1
print("Recommended employee for issue %d :"%itemno)
print(sample_recommendation_item(modelnew,intertest,1,userdict,itemdict,10,[5,2,3]))
usersim=generate_user_similarity_matrix(modelnew,intertest,10)
create_heatmap_emp(usersim,"employee similarity")
itemsim=generate_item_similarity_matrix(modelnew,intertest,10)
create_heatmap_emp(itemsim,"issue similarity")

