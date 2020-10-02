#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from IPython.core.display import HTML

path = "../input/"

versions = pd.read_csv(path+"KernelVersions.csv")
kernels = pd.read_csv(path+"Kernels.csv")
users = pd.read_csv(path+"Users.csv")

def pressence_check(title, tokens):
    present = False
    for token in tokens:
        words = token.split()
        if all(wrd.lower().strip() in title.lower() for wrd in words):
            present = True
    return present 
    
def get_kernels(tokens, n):
    versions['isRel'] = versions['Title'].apply(lambda x : pressence_check(x, tokens))
    relevant = versions[versions['isRel'] == 1]
    relevant = relevant.groupby('KernelId').agg({'TotalVotes' : 'sum', 'Title' : lambda x : "#".join(x).split("#")[0]})
    results = relevant.reset_index().sort_values('TotalVotes', ascending = False).head(n)
    results = results.rename(columns={'KernelId' : 'Id', 'TotalVotes': 'Votes'})
    results = results.merge(kernels, on="Id").sort_values('TotalVotes', ascending = False)
    results = results.merge(users.rename(columns={'Id':"AuthorUserId"}), on='AuthorUserId')
    return results[['Title', 'CurrentUrlSlug', 'TotalViews', 'TotalComments', 'TotalVotes', "DisplayName","UserName"]]


def best_kernels(tokens, n = 10):
    response = get_kernels(tokens, n)     
    hs = """<style>
                .rendered_html tr {font-size: 12px; text-align: left}
            </style>
            <h3><font color="#1768ea">"""+tokens[0].title()+"""</font></h3>
            <table>
            <th>
                <td><b>Kernel Title</b></td>
                <td><b>Author</b></td>
                <td><b>Total Views</b></td>
                <td><b>Total Comments</b></td>
                <td><b>Total Votes</b></td>
            </th>"""
    for i, row in response.iterrows():
        url = "https://www.kaggle.com/"+row['UserName']+"/"+row['CurrentUrlSlug']
        aurl= "https://www.kaggle.com/"+row['UserName']
        hs += """<tr>
                    <td>"""+str(i+1)+"""</td>
                    <td><a href="""+url+""" target="_blank"><b>"""  + row['Title'] + """</b></a></td>
                    <td><a href="""+aurl+""" target="_blank">"""  + row['DisplayName'] + """</a></td>
                    <td>"""+str(row['TotalViews'])+"""</td>
                    <td>"""+str(row['TotalComments'])+"""</td>
                    <td>"""+str(row['TotalVotes'])+"""</td>
                    </tr>"""
    hs += "</table>"
    display(HTML(hs))


# # Data Science Glossary on Kaggle
# 
# Kaggle is the place to do data science projects. There are so many algorithms and concepts to learn. Kaggle Kernels are one of the best resources on internet to understand the practical implementation of algorithms. However there are almost 200,000 kernels published on kaggle and sometimes it becomes diffcult to search for the right implementation. 
# 
# Recently, Kaggle team updated the [Meta Kaggle](https://www.kaggle.com/kaggle/meta-kaggle) database and I am using it to collate the best kernels by different topics, and finally creating a glossary of machine learning, natural language processing algorithms shared on kaggle kernels. One can use this kernel as the one place to find other great kernels shared by great authors. Hope you like this kernel. 
# 
#   
# 
# ## 1. Regression Algorithms
# 

# In[ ]:


tokens = ["linear regression"]
best_kernels(tokens, 10)


# In[ ]:


tokens = ['logistic regression', "logistic"]
best_kernels(tokens, 10)


# In[ ]:


tokens = ['Stepwise regression']
best_kernels(tokens, 10)


# In[ ]:


tokens = ['polynomial regression']
best_kernels(tokens, 10)


# In[ ]:


tokens = ['multivariate regression']
best_kernels(tokens, 10)


# ## 2. Regularization Algorithms

# In[ ]:


tokens = ['Ridge']
best_kernels(tokens, 10)


# In[ ]:


tokens = ['Lasso']
best_kernels(tokens, 10)


# In[ ]:


tokens = ['ElasticNet']
best_kernels(tokens, 10)


# ## 3. Tree Based Models

# In[ ]:


tokens = ['Decision Tree']
best_kernels(tokens, 10)


# In[ ]:


tokens = ['random forest']
best_kernels(tokens, 10)


# In[ ]:


tokens = ['lightgbm', 'light gbm', 'lgb']
best_kernels(tokens, 10)


# In[ ]:


tokens = ['xgboost', 'xgb']
best_kernels(tokens, 10)


# In[ ]:


tokens = ['catboost']
best_kernels(tokens, 10)


# In[ ]:


tokens = ['adaboost']
best_kernels(tokens, 10)


# ## 4. Neural Networks and Deep Learning

# In[ ]:


tokens = ['neural network']
best_kernels(tokens, 10)


# In[ ]:


tokens = ['backpropagation']
best_kernels(tokens, 10)


# In[ ]:


tokens = ['autoencoder']
best_kernels(tokens, 10)


# In[ ]:


tokens = ['deep learning']
best_kernels(tokens, 10)


# In[ ]:


tokens = ['convolutional neural networks', 'cnn']
best_kernels(tokens, 10)


# In[ ]:


tokens = ['lstm']
best_kernels(tokens, 10)


# In[ ]:


tokens = ['gru']
best_kernels(tokens, 10)


# In[ ]:


tokens = ['mxnet']
best_kernels(tokens, 10)


# In[ ]:


tokens = ['resnet']
best_kernels(tokens, 10)


# In[ ]:


tokens = ['Capsule network', 'capsulenet']
best_kernels(tokens, 10)


# ## 5. Clustering Algorithms 

# In[ ]:


tokens = ['kmeans', 'k means']
best_kernels(tokens, 10)


# In[ ]:


tokens = ['hierarchical clustering']
best_kernels(tokens, 10)


# In[ ]:


tokens = ['dbscan']
best_kernels(tokens, 10)


# ## 6. Misc 

# In[ ]:


tokens = ['naive bayes']
best_kernels(tokens, 10)


# In[ ]:


tokens = ['svm']
best_kernels(tokens, 10)


# In[ ]:


tokens = ['ensemble']
best_kernels(tokens, 10)


# In[ ]:


tokens = ['stacking', 'stack']
best_kernels(tokens, 10)


# In[ ]:


tokens = ['feature engineering']
best_kernels(tokens, 10)


# In[ ]:


tokens = ['feature selection']
best_kernels(tokens, 10)


# In[ ]:


tokens = ['cross validation']
best_kernels(tokens, 10)


# In[ ]:


tokens = ['model selection']
best_kernels(tokens, 10)


# In[ ]:


tokens = ['smote']
best_kernels(tokens, 10)


# ## 7. ML Tools

# In[ ]:


tokens = ['scikit']
best_kernels(tokens, 10)


# In[ ]:


tokens = ['tensorflow', 'tensor flow']
best_kernels(tokens, 10)


# In[ ]:


tokens = ['theano']
best_kernels(tokens, 10)


# In[ ]:


tokens = ['keras']
best_kernels(tokens, 10)


# In[ ]:


tokens = ['pytorch']
best_kernels(tokens, 10)


# In[ ]:


tokens = ['vowpal wabbit','vowpalwabbit']
best_kernels(tokens, 10)


# In[ ]:


tokens = ['tensorflow', 'tensor flow']
best_kernels(tokens, 10)


# In[ ]:


tokens = ['eli5']
best_kernels(tokens, 10)


# ## 8. Data Visualization

# In[ ]:


tokens = ['visualization', 'visualisation']
best_kernels(tokens, 10)


# In[ ]:


tokens = ['plotly', 'plot.ly']
best_kernels(tokens, 10)


# In[ ]:


tokens = ['seaborn']
best_kernels(tokens, 10)


# In[ ]:


tokens = ['bokeh']
best_kernels(tokens, 10)


# ## 8. Dimentionality Reduction

# In[ ]:


tokens = ['PCA']
best_kernels(tokens, 10)


# In[ ]:


tokens = ['Tsne', 't-sne']
best_kernels(tokens, 10)


# In[ ]:


tokens = ['Reinforcement', 'Reinforcement Learning']
best_kernels(tokens, 10)


# In[ ]:


tokens = ['Markov', 'Markov Chain']
best_kernels(tokens, 10)


# In[ ]:


tokens = ["Model based learning"]
best_kernels(tokens, 10) 


# In[ ]:


tokens = ["Q-Learning"]
best_kernels(tokens, 10) 


# In[ ]:


tokens = ["Probabilistic"]
best_kernels(tokens, 10) 


# <br>
# Suggest the list of items which can be added to the list. If you liked this kernel, please upvote.  
# 

# In[ ]:


tokens = ["AIMA"]
best_kernels(tokens, 23) 


# In[ ]:


tokens = ["Tutorial"]
best_kernels(tokens, 100) 


# In[ ]:


tokens = ["Generative Adversarial Networks","GAN"]
best_kernels(tokens,10)


# In[ ]:


tokens = ["autoencoder","decoder"]
best_kernels(tokens,10)

