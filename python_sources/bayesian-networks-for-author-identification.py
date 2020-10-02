#!/usr/bin/env python
# coding: utf-8

# # Introduction:
# 
# Although we typically represent text as strings -- sequences of words which are, in turn, sequences of characters-- real sentences often possess complicated non-sequential structures. A sentence may be built up of clauses, which are in turn composed of phrases, which are themselves constructed from words. Thus, it often makes more sense to think of sentences as *trees* rather than chains.
# 
# The structure of these trees is dictated by the *grammar* of English -- a collection of rules about how smaller structures can be combined to form larger ones. For example, English's grammar allows an adjective to modify a noun to form a noun phrase. A noun phrase may be combined with a prepositional phrase to form a yet larger noun phrase, which might itself act as the object of a verb in a verb phrase, which might act as the predicate of a clause, and so on.
# 
# An author's 'style' can be viewed as the manner in which he or she uses the grammar of a language to convey a message: how many, and what kinds of phrases does he use? What kinds of sub-phrases are those phrases composed of? What words does she use to build those phrases? And, of course, sometimes authors will violate the "standard"  grammatical rules, combining words and phrases in atypical ways. This is also part of an author's style.
# 
# In this notebook, we will construct a model designed to account for such choices.
# 
# # Model: 
# 
# Below is a syntax tree generated from the first sentence in test.csv using the Stanford natural language parser. 
# 
# ![https://i.imgur.com/UPqK9mR.png](https://i.imgur.com/UPqK9mR.png)
# 
# Any such structure can be transformed into a binary tree -- one in which each node has at most two branches. In natural language processing, this is known as *Chomsky Normal Form*. 
# 
# ![https://i.imgur.com/Sc9qQkD.png](https://i.imgur.com/Sc9qQkD.png)
# 
# We will use trees in this form to construct our classifier.
# 
# The assumptions of our model are as follows
# 
# 1. An author constructs a sentence starting from an S (sentence) node and works his or her way down. 
# 
# 2. With two exceptions, described in 4), each node branches into exactly two child nodes, i.e. each structure within a sentence is composed of two sub-structures.
# 
# 3. The probability that a particular child will be chosen on a given branch for a given parent depends on three things: 
# 
#     a) The content of the parent node (e.g. prepositional phrases often occur within noun phrases).
#     
#     b) The direction of the branch (left or right -- e.g. maybe prepositional phrases are more likely to be found on the right branch of noun phrases).
#     
#     c) The author of the sentence (e.g. when deciding how to fill the right branch of a noun phrase, perhaps Shelley likes to use prepositional phrases, while Poe prefers adverbs)
# 
# 4. There are two ways in which a branch of the tree may terminate:
#     
#     a) There is a class of nodes that corresponds to parts of speech (nouns, verbs, particular punctuation marks, e.g. commas, etc.) Such nodes always have only a single child, corresponding to an individual word or character. Individual words or characters are always 'leaves' of the tree, and have no children themselves.
# 
#     b) There is a special leaf node called 'END', which represents an author's decision to simply not place any child node on a branch. For example, if an author writes a noun phrase consisting of only a single noun, we will represent this as a noun phrase branching into a noun node and an END node.
#     
# 5. The branching process described above continues until every branch of the tree ends in a leaf node.
# 
# In essence, we will be modeling each sentence in our train and test sets as a tree-shaped [Bayesian network](https://en.wikipedia.org/wiki/Bayesian_network). A Bayesian network is a model that represents dependencies between random variables as a graph. Each edge is directed (it points *from* a parent node *to* a child node) and represents the probability distribution of its child node conditioned on its parent node.
# 
# In this case, our random variables are components of a sentence's syntax tree -- words, phrases, etc. We will assume that each such object is chosen randomly from a distribution that depends on the three factors listed in 3. Attached to each branch of the syntax tree is a conditional probability -- the probability P(c|d,p,a) that the branch's child would have been 'c', given that the branch's direction was 'd', its parent was 'p', and the author of the sentence was 'a'. These conditional probabilities, along with the prior probabilities P(a),will form the parameters of our model 
# 
# When given a tree to identify, the probability associated with each author will simply be the product of P(c|d,p,a) over all nodes in the tree, multiplied by the prior probability of a, divided by a normalization constant (more on this in the "Generating Predictions" section).
# 
# Obviously, the process above isn't actually how sentences are written. However, this model automatically takes into account three very important parts of an author's style: sentence structure, sentence length (modeled by an author's propensity for choosing END nodes or part-of-speech nodes versus phrase or clause nodes), and word choice (modeled by the probability distribution of leaf nodes for each part of speech).
# 
# # Code:
# 
# ## Tree generation:
# 
# To start, we will need to generate a tree for every sentence in the train and test datasets. We can do this using Stanford's Natural Language Parser. Although NLTK has methods for interfacing with the parser, I do not believe it can be run on Kaggle, as the methods require access to the parser's jar files, which must be downloaded from [the Stanford Natural Language Processing Groups's page](https://nlp.stanford.edu/software/lex-parser.shtml).
# 
# As such, I have uploaded two pickle files, one for the training data (train_trees.p) and one for the test data (test_trees.p). Each pickle file contains a dictionary with a tree for (almost) every sentence its respective data set. There were a few sentences that were too long to process--the parser's memory requirements scale quadratically with sentence length--so some entries in the dictionaries have values of "None". This amounts to about eight sentences in the training set and about three in the test set.
# 
# The code used to generate the trees is below (commented out). If you wish to run it on your computer, you will have to download the Stanford parser and change the path names at the top of the code accordingly. Be aware that it will take at least several hours to run to completion.

# In[ ]:


'''
import time
import os
import pickle
import pandas as pd
from nltk.internals import find_jars_within_path

pathstring = "C:/Users/Nathaniel/Desktop/stanford-parser-full-2017-06-09"
os.environ["STANFORD_MODELS"] = pathstring + "/stanford-parser-3.8.0-models.jar"
os.environ["STANFORD_PARSER"] = pathstring + "/stanford-parser.jar"
os.environ["JAVAHOME"] = "C:/Program Files/Java/jdk-9.0.1"
os.environ["CLASSPATH"] = pathstring

input_file = 'test.csv'
output_file = "test_trees.p"

from nltk.parse import stanford
import nltk as nl
parser = nl.parse.stanford.StanfordParser(model_path=(pathstring + "/edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz"),java_options='-mx16000m')
parser._classpath = tuple(find_jars_within_path(pathstring))
data = pd.read_csv(input_file)
data['tokens'] = data['text'].map(nl.word_tokenize)
t = time.clock()
treelist = dict()
nones = 0
for i, sent in enumerate(data['text']):
    l = len(data['tokens'][i])
    if l < 200:
        treelist[i] = next(next(parser.raw_parse_sents([sent])))    
    else:
        treelist[i] = None
        nones = nones + 1
        print(l, end=' ')
    print(i, ": ", time.clock() - t)
print(nones)

with open(output_file,"wb") as filehandle:
    pickle.dump(treelist,filehandle)
    
'''


# ## Model Training:
# 
# Now that we have our trees, we'll convert them into Chomsky Normal Form and generate 'count' matrices in the form of pandas DataFrames. The entries in each sentence's matrix count the number of times each parent-child-direction combination occurs within that sentence. So, for example, if there are four occasions in a sentence where a noun phrase contains an adjective on its left branch, there will be a number '4' in the row labeled (NODE, JJ) and column labeled (NP, L). Here, JJ stands for 'Adjective' and 'NP' stands for 'Noun Phrase'. [This page contains links to documentation for all tags used by the Stanford parser.](https://nlp.stanford.edu/software/parser-faq.shtml#c)
# 
# A precise description of the matrix is as follows:
# 
# The columns of the DataFrames are a multi-index, with the top index corresponding to parent nodes, and the bottom index corresponding to directions (Left, Right, and Center for nodes that only have a single child).
# 
# The rows of this dataframes are also a multi-index. The inner index corresponds to child nodes, while the outer index, which exists mostly for convenience, denotes whether or not the inner index corresponds to a leaf.
# 
# The value in the field with row index C, top column index P, and lower column index D denotes the number of times a parent node of type P connects to a child node of type C along direction D. 
# 
# Below is one such matrix, generated from the first sentence in the test dataset -- the same sentence modeled by the two trees above.

# In[ ]:


import pickle
int_outputs = "../input/intermediate-outputs"

with open(int_outputs+"/test_matrices_all.p", "rb") as filehandle:
    test_matrices = pickle.load(filehandle)


# In[ ]:


test_matrices[0]


# A small amount of processing is applied to the tree before generating such a matrix. First, words in the sentence are 'stemmed', i.e. converted to a root form. For example, the word "leaving" is replaced with "leav." This destroys some information, such as tense. However, by only stemming after our sentences have been parsed into trees, we preserve information about a word's part of speech. Stemming helps to keep the size of the matrix managable.
# 
# Second, conversion to Chomsky Normal Form produces many 'composite' tags. An example in the tree above would be VP<\NP-PP>, which represents a portion of a verb phrase consisting of a noun phrase and a prepositional phrase. In generating the matrix above, VP<\NN-PP> is simply replaced with VP++. Again, some information is lost but this helps to keep the size of the matrix managable.
# 
# Such a matrix is generated for every tree in the training and test data. The matrices for the training data are summed together into 3+1 "master" matrices, one for each author plus an even larger matrix that is the sum of the other three, while the matrices for testing data are kept separate. Below is code for generating these matrices.

# In[ ]:


def get_rels_ch3(tree, stemmer=None):
    '''takes a single tree and an optional stemmer, returns a matrix as a pandas DataFrame'''
    tree.chomsky_normal_form()
    tree = nl.ParentedTree.convert(tree)#ParentedTree allows us to access a node's parents
    start_rows = pd.MultiIndex.from_tuples([("LEAF","_END")])
    start_cols = pd.MultiIndex.from_tuples([("ROOT", 'R')])
    child_par = pd.DataFrame([0],index=start_rows, columns=start_cols)#Dataframe to return
    
    for sub in tree.subtrees():
        par = sub.parent()
        if par:
            p_ind = sub.parent_index()
            par_label = par.label()
            child_label = sub.label()
            
            par_label = re.sub(r"(\|[^-]*)|(-[^-]*)","+",par_label)#simplify composite labels
            child_label = re.sub(r"(\|[^-]*)|(-[^-]*)","+",child_label)#like VP<NN-PP> to VP++
            
            new_cols = pd.MultiIndex.from_tuples([(par_label, 'L'), (par_label, 'R')])
            new_row = pd.MultiIndex.from_tuples([("NODE",child_label)])
            new_entry = pd.DataFrame([[1-p_ind, p_ind]],index=new_row, columns=new_cols)
            
            child_par = child_par.add(new_entry, fill_value=0)
            
            if not sub.left_sibling() and not sub.right_sibling():
                child_par.loc[("LEAF","_END"),(par_label,'R')] +=1
        
            if sub.height() == 2:#nltk does not consider leaves to be subtrees
                for leaf in sub.leaves():#so we must handle them separately
                    if stemmer:
                        leaf = stemmer.stem(leaf)
                    leaf = pd.MultiIndex.from_tuples([("LEAF",leaf.lower())])
                    leaf_par = pd.MultiIndex.from_tuples([(child_label, "C")])
                    
    
                    new_leaf = pd.DataFrame([1],index=leaf, columns=leaf_par)
                    child_par = child_par.add(new_leaf, fill_value=0)
            return child_par.to_sparse()


# In[ ]:


def get_counts(column, trees,stem = True):
    '''takes a pandas series, like the author column of the training data, and a list or dictionary of trees.
    the column is used for identifying who each sentence belongs to, and must line up with the list/dict indices
    returns the sum of all matrices generated from the list of trees
    '''
    t1 = time.clock()
    
    stemmer = None
    if stem: stemmer = nl.stem.snowball.SnowballStemmer("english")
        
    start_rows = pd.MultiIndex.from_tuples([("LEAF","_END"), ("LEAF","_OTHER"),("NODE","_OTHER")])
    start_cols = pd.MultiIndex.from_tuples([("_OTHER", 'L'), ("_OTHER", 'R'),("_OTHER", 'C')])
        
    categories = column.unique()
    counts = dict()
    
    for cat in categories: #build an empty matric for each category (author)
        counts[cat] = pd.SparseDataFrame(index=start_rows, columns=start_cols)
    counts['TOTAL'] = pd.SparseDataFrame(index=start_rows, columns=start_cols)
    for i in column.index:
        tree = trees[i]
        if tree:        
            counts[column[i]] = counts[column[i]].add(get_rels_ch3(tree,stemmer), fill_value=0)
        t2 = time.clock()
        print(i,":",t2 - t1)#for gauging the speed of this function
    for i in categories:
        counts['TOTAL'] = counts['TOTAL'].add(counts[i], fill_value=0)
    return counts


# In[ ]:


def get_counts2(column, trees,stem = True): #for getting matrices separately
    t1 = time.clock()
    stemmer = None
    if stem: stemmer = nl.stem.snowball.SnowballStemmer("english")  
    counts = []    
    for i in column.index:
        tree = trees[i]
        if tree:
            counts_i = get_rels_ch3(tree,stemmer)
            
            counts.append(counts_i)
        else:
            counts.append(None)
        t2 = time.clock()
        print(i,":",t2 - t1)
    return counts


# I've uploaded four pickle files with matrices generated directly from sentence trees: train_matrices_master.p contains summed matrices generated from the first 16,000 sentences in the train csv, to be used for training our model. A list of separate matrices from the next 2,000 sentences is stored in train_matrices_calib.p. The remaining sentences from the training csv are stored in train_matrices_holdout.p, to be used for testing. Finally, all matrices from the testing csv are stored in test_matrices_all.p.
# 
# 
# 
# 

# ## Generating Predictions:
# 
# Now that we have all of these matrices, we can now begin to make predictions. From the giant summed matrices for each author, we can directly obtain the conditional probabilities P(c|d,p,a) mentioned earlier: P(c|d,p,a) is simply the entry in row c, column (d,p) of matrix a, divided by the sum of values in that column (subject to some Laplace smoothing).
# 
# The probability associated with the entire tree T, given a, is
# 
# $$P(\textrm{T|a}) = \prod_{c \in T}^{} P(c|d,p,a)$$
# 
# ([This is a property of bayesian networks](https://en.wikipedia.org/wiki/Bayesian_network#Definitions_and_concepts). Our network was defined so as to satisffy the local Markov property conditioned on each author, so it also satisfies the above factorization property.)
# 
# The probability that a particular tree was written by a is thus given by:
# 
# $$P(\textrm{a|T}) =  \frac{P(T|a) P(a)}{P(T)} =   \frac{P(a)}{P(T)}  \prod_{c \in T}^{} P(c|d,p,a)$$
# $$ \propto  P(a) \prod_{c \in T}^{} P(c|d,p,a)$$
# 
# For the purposes of predicting an author, P(T) is merely a constant normalization factor.
# 
# So, to figure out the odds that a particular author wrote a given sentence in the test set, we'll do the following: 
# 
# 1. Generate a matrix for the sentence. We'll call this matrix S.
# 2. Pull up the master matrix A for the author in question.
# 3. For each nonzero value $S_{c,p,d}$ in $S$, look up the corresponding value in A (i.e. the value in the same row and column) and calculate P(c|d,p,a)
# 4. The contribution to the overall probability from this matrix entry is simply P(c|d,p,a) raised to the power $S_{c,p,d}$
# 5. The overall probability that this author wrote this sentence is the product of such contributions over all nonzero $S_{c,p,d}$ (subject to a normalization factor)
# 
# Below is a function that carries out the computations described above.

# In[ ]:


def get_probs(train_dict,test_list,ids,weights,smoothing=0.5):
    t0=time.clock()
    train_cats = {i:v.to_dense().fillna(0) for (i, v) in train_dict.items() if i != "TOTAL"}#training 'master' matrix associated with each category
    train_total = train_dict["TOTAL"].to_dense().fillna(0)
    
    all_rows = train_total.index
    all_columns = train_total.columns
    
    results = pd.DataFrame(index=ids.index, columns=['id']+[i for i in train_cats],dtype=float)
    
    for (n,test_mat) in enumerate(test_list):#for each matrix to be tested in test list
        cat_probs = pd.Series({i:1.0 for i in train_cats})
        if test_mat is not None:
            test_matrix = test_mat.to_dense().fillna(0)
            nonzeros = test_matrix[test_matrix>0].stack().stack()#convert to a single multi-indexed column listing all nonzero entries
            for (j, count) in nonzeros.iteritems():#for each nonzero value in matrix

                row=(j[0],j[1] if (j[0],j[1]) in all_rows else "_OTHER")

                column = (j[3] if (j[3],j[2]) in all_columns else "_OTHER", j[2])
                column_total_nonzeros = train_total[column][train_total[column] >0].size
                if column_total_nonzeros == 0: #if column is empty for all authors, shift to 'other' column
                    column = ("_OTHER", j[2])
                    column_total_nonzeros = train_total[column][train_total[column] >0].size
                
                for (cat,train_matrix) in train_cats.items():
                    numerator = smoothing

                    denominator = (column_total_nonzeros+1)*smoothing

                    if (column in train_matrix.columns and column_total_nonzeros >=2):
                        denominator = denominator + train_matrix.loc[:,column].sum()
                        if (row in train_matrix.index):
                            numerator = numerator +train_matrix.loc[row,column]

                    cat_probs[cat] = cat_probs[cat]*pow(numerator / denominator,count)
                    
                cat_probs = cat_probs/cat_probs.sum()

        cat_probs=cat_probs * weights       
        results.loc[ids.index[n]] = cat_probs/cat_probs.sum()
        print(n,":",time.clock() - t0)
    
    results['id'] = ids
    
    return results


# The parameters of the function above are 
# 
# 1: A dictionary containing matrices as generated by the get_counts function. The indices of this matrix should correspond to the classification categories, with a last matrix labeled 'TOTAL'. So in this case, the indices would be "EAP", "HPL", "MWS", and "TOTAL"
# 
# 2: A list of matrices corresponding to sentences to be classified, as generated by get_counts 2
# 
# 3: A column (pandas Series) of labels with the same length as the list from 2. This is used to create the index of and forms the first column of the output dataframe. For calibrating and holdout testing, we use the corresponding rows of the 'author' column in train.csv. For classifying new data, we use the 'id' column of test.csv
# 
# 4: A series containing counts of the number of sentences from each author in the training data, for calculating the prior probabilities P(a)
# 
# 5: An optional Laplace smoothing parameter. From testing, I find that a value of 0.5 works fairly well.
# 
# At this point, we almost have our full classifier. However, in my tests I found that this function, on its own, tends to be 'overconfident' in its predictions. That is, it tends to produce probabilities that are close to 0 or 1 much more often than probabilities that are close to the author priors. 
# 
# [Naive Bayes classifiers also share this problem of overcondfidence](http://scikit-learn.org/stable/modules/calibration.html). This happens because Naive Bayes classifiers make very strong independence assumptions, which are not always borne out in reality. The result is that information shared between different features is essentially double-counted.
# 
# Likewise, it may be that the indpendence assumptions in our model are the reason it is overconfident. In particular, we assumed that the label of a node depends *only* on its parent, direction, and author.
# 
# Although overconfidence doesn't affect a model's accuracy, it does harm the model's log-loss metric. One solution, which is also used with naive Bayes classifiers, is to run the outputs of the first classifier through a regression model.
# 

# In[ ]:


def probs_regress(train_dict,test_list,ids,weights,smoothing=0.5, regressors={}):
    probs = get_probs(train_dict,test_list,ids,weights,smoothing)
    num_columns = probs.columns.drop('id')
    for i, v in regressors.items():
        probs[i] = regressors[i].predict(probs[i])
    results_sum = probs[num_columns].sum(axis=1)
    probs[num_columns] = probs[num_columns].divide(results_sum, axis=0)
    return probs


# This function has the same parameters as get_probs, with an added parameter called 'regressors'. It is meant to take the form of a dictionary containing one scikit-learn isotonic regressor per classification category (one per author in this case).
# 
# Below is a function for training those regressors. It returns a dictionary as described above.

# In[ ]:


def calibrate_reg(train_dict, calib_set,id_calib, weights, smoothing=0.5):
    
    raw_results = probs_regress(train_dict,calib_set,id_calib,weights,smoothing)
    raw_results_n = pd.DataFrame(index = raw_results.index, columns = ['EAP','HPL','MWS'])

    for i,v in raw_results['id'].iteritems():#generates a dataframe of 'correct' results
        conversion = {'EAP':[1,0,0], 'HPL':[0,1,0], 'MWS':[0,0,1]}
        raw_results_n.loc[i,:] =conversion[v]
    
    regressors = {}
    for i,v in raw_results_n.iteritems():
        ir = sk.isotonic.IsotonicRegression(out_of_bounds = 'clip')
        regressors[i] = ir.fit(raw_results[i], raw_results_n[i])
         
    return regressors


# Finally, putting everything together, we're ready to start classifying. Below, the matrices and regressors used have been extracted from the pickle files uploaded with this kernel. However, example commands are included in comments showing how to generate these objects from the functions above.

# In[ ]:


import pandas as pd
import nltk as nl
import sklearn as sk
from sklearn import isotonic
import pickle
import time
import re
train_data = pd.read_csv('../input/spooky-author-identification/train.csv')
test_data = pd.read_csv('../input/spooky-author-identification/test.csv')
int_outputs = "../input/intermediate-outputs"

#with open(int_outputs+"/train_trees.p", "rb") as filehandle:
#    train_trees = pickle.load(filehandle)
#with open(int_outputs+"/test_trees.p", "rb") as filehandle:
#    test_trees = pickle.load(filehandle)

#train_matrices_all = get_counts(test_data['id'], test_trees)    
with open(int_outputs+"/test_matrices_all.p", "rb") as filehandle:
    test_matrices_all = pickle.load(filehandle)

#train_matrices_master = get_counts(train_data['author'][:16000], train_trees)    
with open(int_outputs+"/train_matrices_master.p", "rb") as filehandle:
    train_matrices_master = pickle.load(filehandle)

#train_matrices_calib = get_counts2(train_data['author'][16000:18000], train_trees)    
with open(int_outputs+"/train_matrices_calib.p", "rb") as filehandle:
    train_matrices_calib  = pickle.load(filehandle)

#train_matrices_holdout = get_counts2(train_data['author'][18000:], train_trees)
with open(int_outputs+"/train_matrices_holdout.p", "rb") as filehandle:
    train_matrices_holdout = pickle.load(filehandle)

#isotonic_regs = calibrate_reg(train_matrices_master, train_matrices_calib + train_matrices_holdout,train_data['author'][16000:], train_data['author'][16000:].value_counts(), smoothing=0.5)
with open(int_outputs+"/isotonic_regs.p", "rb") as filehandle:
    isotonic_regs = pickle.load(filehandle)
    

    
probs = probs_regress(train_dict=train_matrices_master,
                      test_list = test_matrices_all,
                      ids = test_data['id'],
                      weights = train_data['author'][:16000].value_counts(),
                      smoothing=0.5,
                      regressors=isotonic_regs
                     )    
probs.to_csv('submission.csv',index=False)


# In[ ]:


probs


# # Final Notes:
# 
# In my tests, this model's predictions tend to be about 80-85% accurate, with log loss around 0.4. A laplace smoothing parameter of 0.1 instead of 0.5 actually seems to yield a slightly better log loss, at the cost of a small amount of accuracy. 
# 
# Initially, I was worried about this model overfitting the data, due to the very large number of features involved. As such, I created a function for merging 'rare' words or constructs into a label called 'other'. More specifically, it would take in a matrix, and search for rows whose sum fell below a certain threshold. The contents of these rows would be summed together to form an 'other' row, and the original rows would be dropped, and likewise for columns. 
# 
# The intent was that when classifying a sentence containing words or constructs that had never been seen before, the model would consult the 'other' row or column. In essence, the 'other' tag was meant to represent an author's propensity for using extremely uncommon words or structures.
# 
# What I found was that using this function actually decreased the model's performance, with higher 'rarity' thresholds producing worse results. This suggests to me that the model is not overfitting, and that it could possibly handle an even larger feature space.
# 
# Some possibilities might include leaving words unstemmed, being less aggressive when merging composite constructs like VP<\NN-PP>, or possibly even modeling dependencies between left and right children of a node
# 

# In[ ]:


def prep2(train_dict, threshold):
    train_cats = {i:v.to_dense().fillna(0) for (i, v) in train_dict.items() if i != "TOTAL"}
    train_total = train_dict["TOTAL"].to_dense().fillna(0)
    
    row_sums=train_total.sum(axis=1)#sum of rows

    all_oth_rows = train_total.loc[row_sums < threshold,:].drop([("LEAF", "_OTHER"),("NODE", "_OTHER")],axis=0)
    
    all_oth_leaf = all_oth_rows.loc['LEAF']
    all_oth_node = all_oth_rows.loc['NODE']
    
    all_oth_leaf_ind = pd.MultiIndex.from_product([["LEAF"],all_oth_leaf.index])
    all_oth_node_ind = pd.MultiIndex.from_product([["NODE"],all_oth_node.index])
      
    train_total.loc[('LEAF', '_OTHER'),:] = all_oth_leaf.sum(axis=0)
    train_total.loc[('NODE', '_OTHER'),:] = all_oth_node.sum(axis=0)
    
    train_total = train_total.drop(all_oth_leaf_ind, axis=0)
    train_total = train_total.drop(all_oth_node_ind, axis=0)
    
    
    all_oth_l_ind = train_total.columns.intersection(pd.MultiIndex.from_product([all_oth_node.index,["L"]]))
    all_oth_r_ind = train_total.columns.intersection(pd.MultiIndex.from_product([all_oth_node.index,["R"]]))
    all_oth_c_ind = train_total.columns.intersection(pd.MultiIndex.from_product([all_oth_node.index,["C"]]))
    
    all_oth_l = train_total.loc[:,all_oth_l_ind]
    all_oth_r = train_total.loc[:,all_oth_r_ind]
    all_oth_c = train_total.loc[:,all_oth_c_ind]
    
    
    train_total.loc[:,('_OTHER', 'L')] = all_oth_l.sum(axis=1)
    train_total.loc[:,('_OTHER', 'R')] = all_oth_r.sum(axis=1)
    train_total.loc[:,('_OTHER', 'C')] = all_oth_c.sum(axis=1)
    
    train_total = train_total.drop(all_oth_l_ind, axis=1)
    train_total = train_total.drop(all_oth_r_ind, axis=1)
    train_total = train_total.drop(all_oth_c_ind, axis=1)
    
    for (i, matrix) in train_cats.items():
        oth_leaf_ind = matrix.index.intersection(all_oth_leaf_ind)
        oth_node_ind = matrix.index.intersection(all_oth_node_ind)
        
        oth_leaf = matrix.loc[oth_leaf_ind,:]
        oth_node = matrix.loc[oth_node_ind,:]
        
        
        train_cats[i].loc[('LEAF', '_OTHER'),:] = oth_leaf.sum(axis=0)
        train_cats[i].loc[('NODE', '_OTHER'),:] = oth_node.sum(axis=0)
        
        train_cats[i] = train_cats[i].drop(oth_leaf_ind, axis=0)
        train_cats[i] = train_cats[i].drop(oth_node_ind, axis=0)
        
        oth_l_ind = matrix.columns.intersection(all_oth_l_ind)
        oth_r_ind = matrix.columns.intersection(all_oth_r_ind)
        oth_c_ind = matrix.columns.intersection(all_oth_c_ind)

        oth_l = matrix.loc[:, oth_l_ind]
        oth_r = matrix.loc[:, oth_r_ind]
        oth_c = matrix.loc[:, oth_c_ind]

        train_cats[i].loc[:,('_OTHER', 'L')] = oth_l.sum(axis=1)
        train_cats[i].loc[:,('_OTHER', 'R')] = oth_r.sum(axis=1)
        train_cats[i].loc[:,('_OTHER', 'C')] = oth_c.sum(axis=1)
        
        
        train_cats[i] = train_cats[i].drop(columns=oth_l_ind)
        train_cats[i] = train_cats[i].drop(columns=oth_r_ind)
        train_cats[i] = train_cats[i].drop(columns=oth_c_ind)
        
        
    return {'TOTAL':train_total, **train_cats}

