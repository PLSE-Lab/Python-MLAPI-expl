
### we start by importing all the packages we will need during the script
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import networkx as nx # to create the metric database
import os
print(os.listdir("../input"))

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier

import random
random.seed(1)

######################################################################################################
### LOADING THE DATASETS #############################################################################
######################################################################################################


### we indicate the directory and then load all the data in panda dataframes
data_dir = '../input/fake-news-toulouse/data_competition/data_competition' 

relation_users = pd.read_csv(data_dir + "/UserUser.txt", sep="\t", header=None)
relation_users.columns = ["follower", "followed"]

labels_training = pd.read_csv(data_dir + "/labels_training.txt", sep=",")
labels_training.columns = ["news", "label"]

news_users = pd.read_csv(data_dir + "/newsUser.txt", sep="\t", header=None)
news_users.columns = ["news", "user", "times"]


###################################################################################################
######## Create the metrics dataframe according to the results of the Network analysis ############
###################################################################################################

G = nx.DiGraph()

edges = [tuple(x) for x in relation_users.values]
G.add_edges_from(edges)

centrality = nx.degree_centrality(G)
in_centrality = nx.in_degree_centrality(G)
out_centrality = nx.out_degree_centrality(G)
page_rank = nx.pagerank(G)

metrics_user =[centrality, in_centrality, out_centrality, page_rank]
d={}
for k in centrality.keys():
    d[k] = tuple(d[k] for d in metrics_user)
metrics_user=pd.DataFrame.from_dict(d,orient='index',columns=['degree_centrality','in_degree_centrality',
                                                       'out_degree_centrality', 'page_rank'])
metrics_user.index.rename('user')

metrics_user_news = pd.merge(metrics_user,news_users.set_index('user'), left_index=True, right_index=True)
metrics_user_news.index.name= 'user'
metrics_news = metrics_user_news.groupby('news').agg(['sum','mean','max','min'])
metrics_news = metrics_news.drop(columns=[('times', 'mean'),('times', 'max'),('times', 'min')])

metrics_news.columns = pd.Series(metrics_news.columns.tolist()).apply(pd.Series).sum(axis=1)
#array_column = np.column_stack(( metrics_news.iloc[0], metrics_news.iloc[1]))
#colnames = list()
#for i in range(len(array_column)) :
#    colnames.append(array_column[i][0] + "_" + array_column[i][1])


#metrics_news.columns = colnames
#metrics_news = metrics_news.drop([0, 1])
metrics_news.index = range(1, 241)


### we keep here only the metrics of interest for the prediction
metrics_news = metrics_news.drop(['degree_centralitymean', 'degree_centralitymax', 'degree_centralitymin',
                                'in_degree_centralitymean', 'in_degree_centralitymax', 'in_degree_centralitymin',
                                'out_degree_centralitymean', 'out_degree_centralitymax', 'out_degree_centralitymin',
                                'page_rankmean', 'page_rankmax', 'page_rankmin', 'timessum'], axis=1)



######################################################################################################
### CREATE TRAINING AND TO-BE-PREDICTED SETS #########################################################
######################################################################################################

texts_all_train = []
labels_all_train = []
for i in os.listdir(data_dir + "/news/training/") :
    with open(data_dir + "/news/training/"+ i, 'r') as myfile:
        text0=myfile.read().replace('\n', '')
    texts_all_train.append(text0)
    # get if fake or not
    labels_all_train.append(int(labels_training[labels_training["news"] == int(i.split('.')[0])]["label"]))

    
texts_valid = []
for i in os.listdir(data_dir + "/news/test/") :
    with open(data_dir + "/news/test/"+ i, 'r') as myfile:
        text0=myfile.read().replace('\n', '')
    texts_valid.append(text0)



######################################################################################################
### MAKE THE PREDICTIONS #############################################################################
######################################################################################################

### as seen in the Text Mining analysis, we create a pipeline in order to create the TFIDF matrix
text_tfidf = Pipeline([('vect', CountVectorizer(stop_words='english')), ('tfidf', TfidfTransformer())])
tfidf_matrix = text_tfidf.fit_transform(texts_all_train)

### we change the shape of the TFIDF matrix in dense to be able to had to it all the metrics from the web mining analysis
dense = tfidf_matrix.todense()
newCol = metrics_news.iloc[range(0,193)]
allData = np.append(dense, newCol.astype(np.float), 1)

### finally, we can train our classifier that we selected during the Text Mining analysis, the MLP classifier with its specific parameter from the grid search
best_model = MLPClassifier(solver='lbfgs', alpha=0.001, hidden_layer_sizes=3, random_state=1, max_iter=500).fit(allData, labels_all_train)

### now we have to create the TFIDF matrix of the news to be predicted
tfidf_matrix_test = text_tfidf.transform(texts_valid)

### once again, we change the shape of the matrix, add to it the metrics from the Web Mining analysis 
dense_test = tfidf_matrix_test.todense()
newCol_test = metrics_news.iloc[range(193,240)]
allData_test = np.append(dense_test, newCol_test.astype(np.float), 1)

### once we had the explanatory variables, we can do our predictions
predicted_final = best_model.predict(allData_test)


l = os.listdir(data_dir + "/news/test/")[0:47]
id_doc = [int(i.split('.')[0]) for i in l]
res_only_text_mining = pd.DataFrame(
    {'doc': id_doc,
     'class': predicted_final
    })
res_only_text_mining.sort_values(['doc'], ascending=[True])


