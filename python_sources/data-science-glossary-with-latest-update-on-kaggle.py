#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from IPython.core.display import HTML

path = "../input/"

versions = pd.read_csv(path+"KernelVersions.csv")
kernels = pd.read_csv(path+"Kernels.csv")
users = pd.read_csv(path+"Users.csv")

language_map = {'1' : 'R','5' : 'R', '12' : 'R', '13' : 'R', '15' : 'R', '16' : 'R',
                '2' : 'Python','8' : 'Python', '9' : 'Python', '14' : 'Python'}

def pressence_check(title, tokens, ignore = []):
    present = False
    for token in tokens:
        words = token.split()
        if all(wrd.lower().strip() in title.lower() for wrd in words):
            present = True
    for token in ignore:
        if token in title.lower():
            present = False
    return present 

## check if the latest version of the kernel is about the same topic 
def get_latest(idd):
    latest = versions[versions['KernelId'] == idd].sort_values('VersionNumber', ascending = False).iloc(0)[0]
    return latest['VersionNumber']

def get_kernels(tokens, n, ignore = []):
    versions['isRel'] = versions['Title'].apply(lambda x : pressence_check(x, tokens, ignore))
    relevant = versions[versions['isRel'] == 1]
    results = relevant.groupby('KernelId').agg({'TotalVotes' : 'sum', 
                                                'KernelLanguageId' : 'max', 
                                                'Title' : lambda x : "#".join(x).split("#")[-1],
                                                'VersionNumber' : 'max'})
    results = results.reset_index().sort_values('TotalVotes', ascending = False).head(n)
    results = results.rename(columns={'KernelId' : 'Id', 'TotalVotes': 'Votes'})


    results['latest_version']  = results['Id'].apply(lambda x : get_latest(x))
    results['isLatest'] = results.apply(lambda r : 1 if r['VersionNumber'] == r['latest_version'] else 0, axis=1)
    results = results[results['isLatest'] == 1]

    results = results.merge(kernels, on="Id").sort_values('TotalVotes', ascending = False)
    results = results.merge(users.rename(columns={'Id':"AuthorUserId"}), on='AuthorUserId')
    results['Language'] = results['KernelLanguageId'].apply(lambda x : language_map[str(x)] if str(x) in language_map else "")
    results = results.sort_values("TotalVotes", ascending = False)
    return results[['Title', 'CurrentUrlSlug','Language' ,'TotalViews', 'TotalComments', 'TotalVotes', "DisplayName","UserName"]]


def best_kernels(tokens, n = 10, ignore = []):
    response = get_kernels(tokens, n, ignore)     
    hs = """<style>
                .rendered_html tr {font-size: 12px; text-align: left}
            </style>
            <h3><font color="#1768ea">"""+tokens[0].title()+"""</font></h3>
            <table>
            <th>
                <td><b>Kernel</b></td>
                <td><b>Author</b></td>
                <td><b>Language</b></td>
                <td><b>Views</b></td>
                <td><b>Comments</b></td>
                <td><b>Votes</b></td>
            </th>"""
    for i, row in response.iterrows():
        url = "https://www.kaggle.com/"+row['UserName']+"/"+row['CurrentUrlSlug']
        aurl= "https://www.kaggle.com/"+row['UserName']
        hs += """<tr>
                    <td>"""+str(i+1)+"""</td>
                    <td><a href="""+url+""" target="_blank"><b>"""  + row['Title'] + """</b></a></td>
                    <td><a href="""+aurl+""" target="_blank">"""  + row['DisplayName'] + """</a></td>
                    <td>"""+str(row['Language'])+"""</td>
                    <td>"""+str(row['TotalViews'])+"""</td>
                    <td>"""+str(row['TotalComments'])+"""</td>
                    <td>"""+str(row['TotalVotes'])+"""</td>
                    </tr>"""
    hs += "</table>"
    display(HTML(hs))


# # Data Science Glossary on Kaggle
# 
# Kaggle is the place to do data science projects. There are so many algorithms and concepts to learn. Kaggle Kernels are one of the best resources on internet to understand the practical implementation of algorithms. There are almost 200,000 kernels published on kaggle and sometimes it becomes diffcult to search for the right implementation. I have used the [Meta Kaggle](https://www.kaggle.com/kaggle/meta-kaggle) database to create a glossary of data science models, techniques and tools shared on kaggle kernels. One can use this kernel as the one place to find other great kernels shared by great authors. Hope you like this kernel.  
# 
# 
# ## Contents 
# 
# <ul>
#   <li>1. Regression Algorithms
#     <ul>
#     <li>1.1 Linear Regression</li>
#     <li>1.2 Logistic Regression</li>
#     </ul>
#   </li>
#     <li>2. Regularization Algorithms
#     <ul>
#     <li>2.1 Ridge Regression Regression</li>
#     <li>2.2 Lasso Regression</li>
#     <li>2.3 Elastic Net</li>
#     </ul>
#   </li>
#   </li>
#     <li>3. Tree Based Models
#     <ul>
#     <li>3.1 Decision Tree</li>
#     <li>3.2 Random Forests</li>
#     <li>3.3 Lightgbm</li>
#     <li>3.4 XgBoost</li>
#     <li>3.5 Cat Boost</li>
#     </ul>
#   </li>
# <li>4. Neural Networks and Deep Learning
#     <ul>
#     <li>4.1 Neural Networks</li>
#     <li>4.2 AutoEncoders</li>
#     <li>4.3 DeepLearning</li>
#     <li>4.4 Convolutional Neural Networks</li>
#     <li>4.5 LSTMs</li>
#     <li>4.6 GRUs</li>
#     <li>4.7 MxNet</li>
#     <li>4.8 ResNet</li>
#     <li>4.9 CapsuleNets</li>
#     <li>4.10 VGGs</li>
#     <li>4.11 Inception Nets</li>
#      <li>4.12 Computer Vision</li>
#      <li>4.13 Transfer Learning</li>
#      </ul>
#   </li>
# <li>5. Clustering Algorithms
#     <ul>
#     <li>5.1 K Means Clustering </li>
#     <li>5.2 Hierarchial Clustering</li>
#     <li>5.3 DB Scan</li>
#     <li>5.4 Unsupervised Learning </li>
#     </ul>
#   </li>
#   <li>6. Misc - Models
#     <ul>
#     <li>6.1 K Naive Bayes </li>
#     <li>6.2 SVMs</li>
#     <li>6.3 KNN</li>
#     <li>6.4 Recommendation Engine </li>
#     </ul>
#   </li>
#   <li>7.1 Data Science Techniques - Preprocessing
#     <ul>
#     <li>a. EDA, Exploration </li>
#     <li>b. Feature Engineering </li>
#     <li>c. Feature Selection </li>
#     <li>d. Outlier Treatment</li>
#     <li>e. Anomaly Detection</li>
#     <li>f. SMOTE</li>
#     <li>g. Pipeline</li>
#     <li>g. Missing Values</li>
#     </ul>
#   </li>
#   <li>7.2 Data Science Techniques - Dimentionality Reduction
#     <ul>
#     <li>a. Dataset Decomposition </li>
#     <li>b. PCA </li>
#     <li>c. Tsne </li>
#     </ul>
#   </li>
#   <li>7.3 Data Science Techniques - Post Modelling
#     <ul>
#     <li>a. Cross Validation </li>
#     <li>b. Model Selection </li>
#     <li>c. Model Tuning </li>
#     <li>d. Grid Search </li>
#     </ul>
#   </li>
#   <li>7.4 Data Science Techniques - Ensemblling
#     <ul>
#     <li>a. Ensembling </li>
#     <li>b. Stacking </li>
#     <li>c. Bagging</li>
#     </ul>
#   </li>
#   <li>8. Text Data 
#     <ul>
#     <li>8.1. NLP </li>
#     <li>8.2. Topic Modelling </li>
#     <li>8.3. Word Embeddings </li>
#     </ul>
#   </li>
#  <li>9. Data Science Tools 
#     <ul>
#     <li>9.1 Scikit Learn </li>
#     <li>9.2 TensorFlow </li>
#     <li>9.3 Theano </li>
#     <li>9.4 Kears </li>
#     <li>9.5 PyTorch </li>
#     <li>9.6 Vopal Wabbit </li>
#     <li>9.7 ELI5 </li>
#     <li>9.8 HyperOpt </li>
#     <li>9.9 Pandas </li>
#     <li>9.10 Sql </li>
#     <li>9.11 BigQuery </li>
#     </ul>
#   </li>
# <li>10. Data Visualizations 
#     <ul>
#     <li>10.1. Visualizations </li>
#     <li>10.2. Plotly </li>
#     <li>10.3. Seaborn </li>
#     <li>10.4. D3.Js </li>
#     <li>10.5. Bokeh </li>
#     </ul>
#   </li>
#   <li>11. Time Series  
#     <ul>
#     <li>11.1. Time Series Analysis </li>
#     <li>10.2. ARIMA </li>
#     </ul>
#   </li>
# </ul>
# 
# <br><br>
# 
# ## 1. Regression Algorithms
# 

# In[ ]:


tokens = ["linear regression"]
best_kernels(tokens, 10)


# In[ ]:


tokens = ['logistic regression', "logistic"]
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
best_kernels(tokens, 4)


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


# ## 4. Neural Networks and Deep Learning Models

# In[ ]:


tokens = ['neural network']
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
ignore = ['grupo']
best_kernels(tokens, 10, ignore)


# In[ ]:


tokens = ['mxnet']
best_kernels(tokens, 10)


# In[ ]:


tokens = ['resnet']
best_kernels(tokens, 10)


# In[ ]:


tokens = ['Capsule network', 'capsulenet']
best_kernels(tokens, 5)


# In[ ]:


tokens = ['vgg']
best_kernels(tokens, 5)


# In[ ]:


tokens = ['inception']
best_kernels(tokens, 5)


# In[ ]:


tokens = ['computer vision']
best_kernels(tokens, 5)


# In[ ]:


tokens = ['transfer learning']
best_kernels(tokens, 5)


# In[ ]:


tokens = ['yolo']
best_kernels(tokens, 5)


# ## 5. Clustering Algorithms 

# In[ ]:


tokens = ['kmeans', 'k means']
best_kernels(tokens, 10)


# In[ ]:


tokens = ['hierarchical clustering']
best_kernels(tokens, 3)


# In[ ]:


tokens = ['dbscan']
best_kernels(tokens, 10)


# In[ ]:


tokens = ['unsupervised']
best_kernels(tokens, 10)


# ## 6. Misc - Models 

# In[ ]:


tokens = ['naive bayes']
best_kernels(tokens, 10)


# In[ ]:


tokens = ['svm']
best_kernels(tokens, 10)


# In[ ]:


tokens = ['knn']
best_kernels(tokens, 10)


# In[ ]:


tokens = ['recommendation engine']
best_kernels(tokens, 5)


# ## 7. Important Data Science Techniques

# ### 7.1 Preprocessing

# In[ ]:


tokens = ['EDA', 'exploration', 'exploratory']
best_kernels(tokens, 10)


# In[ ]:


tokens = ['feature engineering']
best_kernels(tokens, 10)


# In[ ]:


tokens = ['feature selection']
best_kernels(tokens, 10)


# In[ ]:


tokens = ['outlier treatment', 'outlier']
best_kernels(tokens, 10)


# In[ ]:


tokens = ['anomaly detection', 'anomaly']
best_kernels(tokens, 8)


# In[ ]:


tokens = ['smote']
best_kernels(tokens, 5)


# In[ ]:


tokens = ['pipeline']
best_kernels(tokens, 10)


# In[ ]:


tokens = ['missing value']
best_kernels(tokens, 10)


# ### 7.2 Dimentionality Reduction

# In[ ]:


tokens = ['dataset decomposition', 'dimentionality reduction']
best_kernels(tokens, 2)


# In[ ]:


tokens = ['PCA']
best_kernels(tokens, 10)


# In[ ]:


tokens = ['Tsne', 't-sne']
best_kernels(tokens, 10)


# ### 7.3 Post Modelling Techniques

# In[ ]:


tokens = ['cross validation']
best_kernels(tokens, 10)


# In[ ]:


tokens = ['model selection']
best_kernels(tokens, 10)


# In[ ]:


tokens = ['model tuning', 'tuning']
best_kernels(tokens, 10)


# In[ ]:


tokens = ['gridsearch', 'grid search']
best_kernels(tokens, 10)


# ### 7.4 Ensemblling

# In[ ]:


tokens = ['ensemble']
best_kernels(tokens, 10)


# In[ ]:


tokens = ['stacking', 'stack']
best_kernels(tokens, 10)


# In[ ]:


tokens = ['bagging']
best_kernels(tokens, 10)


# ## 8. Text Data

# In[ ]:


tokens = ['NLP', 'Natural Language Processing', 'text mining']
best_kernels(tokens, 10)


# In[ ]:


tokens = ['topic modelling']
best_kernels(tokens, 8)


# In[ ]:


tokens = ['word embedding','fasttext', 'glove', 'word2vec']
best_kernels(tokens, 8)


# ## 9. Data Science Tools

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


tokens = ['eli5']
best_kernels(tokens, 10)


# In[ ]:


tokens = ['hyperopt']
best_kernels(tokens, 5)


# In[ ]:


tokens = ['pandas']
best_kernels(tokens, 10)


# In[ ]:


tokens = ['SQL']
best_kernels(tokens, 10)


# In[ ]:


tokens = ['bigquery', 'big query']
best_kernels(tokens, 10)


# In[ ]:


tokens = ['gpu']
best_kernels(tokens, 10)


# ## 10. Data Visualization

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


tokens = ['d3.js']
best_kernels(tokens, 4)


# In[ ]:


tokens = ['bokeh']
best_kernels(tokens, 10)


# ## 11. Time Series

# In[ ]:


tokens = ['time series']
best_kernels(tokens, 10)


# In[ ]:


tokens = ['arima']
best_kernels(tokens, 10)


# ## 12. Some of the Best Tutorials on Kaggle

# In[ ]:


tokens = ['tutorial']
best_kernels(tokens, 10)


# <br>
# Thanks for viewing. Suggest the list of items which can be added to the list. If you liked this kernel, please upvote.  
# 

# Thanks @shivamb and @sudalairajkumar.
