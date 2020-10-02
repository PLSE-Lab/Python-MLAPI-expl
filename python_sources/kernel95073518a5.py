#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import nltk
import string
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from nltk.stem.porter import PorterStemmer
import re
from nltk.corpus import stopwords
from tqdm import tqdm_notebook as tqdm
from prettytable import PrettyTable
import os

from plotly import plotly
import plotly.offline as offline
import plotly.graph_objs as go
offline.init_notebook_mode()


# ## $1.1$ Reading Data

# In[ ]:


#Data taken from Kaggle: https://www.kaggle.com/manasvee1/donorschooseorg-application-screening

project_data = pd.read_csv('../input/donorschooseorg-application-screening/train.csv')
resource_data = pd.read_csv('../input/donorschooseorg-application-screening/resources.csv')


# In[ ]:


# how to replace elements in list python: https://stackoverflow.com/a/2582163/4084039
cols = ['Date' if x=='project_submitted_datetime' else x for x in list(project_data.columns)]

#sort dataframe based on time pandas python: https://stackoverflow.com/a/49702492/4084039
project_data['Date'] = pd.to_datetime(project_data['project_submitted_datetime'])
project_data.drop('project_submitted_datetime', axis=1, inplace=True)
project_data.sort_values(by=['Date'], inplace=True)

# how to reorder columns pandas python: https://stackoverflow.com/a/13148611/4084039
project_data = project_data[cols]

project_data.head(2)


# ## $1.2a$ preprocessing of `project_subject_categories`

# In[ ]:


catogories = list(project_data['project_subject_categories'].values)
# remove special characters from list of strings python: https://stackoverflow.com/a/47301924/4084039
# https://www.geeksforgeeks.org/removing-stop-words-nltk-python/
# https://stackoverflow.com/questions/23669024/how-to-strip-a-specific-word-from-a-string
# https://stackoverflow.com/questions/8270092/remove-all-whitespace-in-a-string-in-python
cat_list = []
for i in catogories:
    temp = ""
    # consider we have text like this "Math & Science, Warmth, Care & Hunger"
    for j in i.split(','): # it will split it in three parts ["Math & Science", "Warmth", "Care & Hunger"]
        if 'The' in j.split(): # this will split each of the catogory based on space "Math & Science"=> "Math","&", "Science"
            j=j.replace('The','') # if we have the words "The" we are going to replace it with ''(i.e removing 'The')
        j = j.replace(' ','') # we are placeing all the ' '(space) with ''(empty) ex:"Math & Science"=>"Math&Science"
        temp+=j.strip()+" " #" abc ".strip() will return "abc", remove the trailing spaces
        temp = temp.replace('&','_') # we are replacing the & value into 
    cat_list.append(temp.strip())
    
project_data['clean_categories'] = cat_list
project_data.drop(['project_subject_categories'], axis=1, inplace=True)


# ## $1.2b$ preprocessing of `project_subject_subcategories`
# 

# In[ ]:


sub_catogories = list(project_data['project_subject_subcategories'].values)
# remove special characters from list of strings python: https://stackoverflow.com/a/47301924/4084039
# https://www.geeksforgeeks.org/removing-stop-words-nltk-python/
# https://stackoverflow.com/questions/23669024/how-to-strip-a-specific-word-from-a-string
# https://stackoverflow.com/questions/8270092/remove-all-whitespace-in-a-string-in-python

sub_cat_list = []
for i in sub_catogories:
    temp = ""
    # consider we have text like this "Math & Science, Warmth, Care & Hunger"
    for j in i.split(','): # it will split it in three parts ["Math & Science", "Warmth", "Care & Hunger"]
        if 'The' in j.split(): # this will split each of the catogory based on space "Math & Science"=> "Math","&", "Science"
            j=j.replace('The','') # if we have the words "The" we are going to replace it with ''(i.e removing 'The')
        j = j.replace(' ','') # we are placeing all the ' '(space) with ''(empty) ex:"Math & Science"=>"Math&Science"
        temp +=j.strip()+" "#" abc ".strip() will return "abc", remove the trailing spaces
        temp = temp.replace('&','_')
    sub_cat_list.append(temp.strip())

project_data['clean_subcategories'] = sub_cat_list
project_data.drop(['project_subject_subcategories'], axis=1, inplace=True)


# ## $1.2c$ preprocessing of `project_grade_category`

# In[ ]:


proj_grade_cat = []

for i in range(len(project_data)):
    pgc = project_data["project_grade_category"][i].replace(" ", "_")
    proj_grade_cat.append(pgc)
    
project_data.drop(['project_grade_category'], axis=1, inplace=True)
project_data["project_grade_category"] = proj_grade_cat


# ## $1.3$ Text preprocessing

# In[ ]:


# merge two column text dataframe: 
project_data["essay"] = project_data["project_essay_1"].map(str) +                        project_data["project_essay_2"].map(str) +                         project_data["project_essay_3"].map(str) +                         project_data["project_essay_4"].map(str)


# In[ ]:


# https://gist.github.com/sebleier/554280
# we are removing the words from the stop words list: 'no', 'nor', 'not'
stopwords= ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",            "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself',             'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',            'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those',             'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',             'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of',             'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',            'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',            'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',            'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very',             's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're',             've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',            "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',            "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't",             'won', "won't", 'wouldn', "wouldn't"]


# In[ ]:


# https://stackoverflow.com/a/47091490/4084039

def decontracted(phrase):
    # specific
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase


# In[ ]:




def getProcessedData(txt_type, working_data):
    preprocessed_data = []
    # tqdm is for printing the status bar
    
    for sentance in tqdm(working_data[txt_type].values):
        sent = decontracted(sentance)
        sent = sent.replace('\\r', ' ')
        sent = sent.replace('\\"', ' ')
        sent = sent.replace('\\n', ' ')
        sent = re.sub('[^A-Za-z0-9]+', ' ', sent)
        # https://gist.github.com/sebleier/554280
        sent = ' '.join(e for e in sent.split() if e.lower() not in stopwords)
        preprocessed_data.append(sent.lower().strip())
        
    return preprocessed_data


# <h2><font color='red'> $1.4$ Preprocessing of `project_title`</font></h2>

# In[ ]:


## Covered Above ...


# ## $1.5$ Preparing data for models
# we are going to consider
# 
#        - school_state : categorical data
#        - clean_categories : categorical data
#        - clean_subcategories : categorical data
#        - project_grade_category : categorical data
#        - teacher_prefix : categorical data
#        
#        - project_title : text data
#        - text : text data
#        - project_resource_summary: text data (optinal)
#        
#        - quantity : numerical (optinal)
#        - teacher_number_of_previously_posted_projects : numerical
#        - price : numerical

# ### $1.5.1$ Vectorizing Categorical data

# In[ ]:


def getCountDict(cat_type):
    count_dict = {}
    info_list = project_data[cat_type]
    project_data.loc[project_data[cat_type].isnull(), cat_type] = 'nan'
    
    for phrase in info_list:
        for data in phrase.split():
            if data not in count_dict: count_dict[data] = 0
            #elif data not in ['nan', np.nan]:
            else:
                count_dict[data] += 1
            
    return dict(sorted(count_dict.items(), key=lambda x: x[1]))


# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer

def getFitCAT_Vectorizer(working_data, cat_type, hstack_features):
    '''
    Fit on only train data.
    '''
    working_data.loc[working_data[cat_type].isnull(), cat_type] = 'nan'
    #print (working_data.keys())
    
    if 1:
        sorted_cat_dict = getCountDict(cat_type)
        print ('Keys...', sorted_cat_dict.keys())
        hstack_features += sorted_cat_dict.keys()
        vectorizer = CountVectorizer(vocabulary=sorted_cat_dict.keys(), lowercase=False, binary=True)
    
    vectorizer.fit(working_data[cat_type].values)
    return vectorizer
    
def getVectorizeCategData(working_data, cat_type, data_type):
    working_data.loc[working_data[cat_type].isnull(), cat_type] = 'nan'
    
    categories_one_hot = vectorizer.transform(working_data[cat_type].values)
    #print(vectorizer.get_feature_names())
    print("Shape of matrix after one hot encodig ",categories_one_hot.shape)
    
    return categories_one_hot


# ### $1.5.2$ Vectorizing Text data

# #### $1.5.2.1$ Bag of words

# In[ ]:


def getFitBOW_Vectorizer(preprocessed_data):
    vectorizer = CountVectorizer(min_df=10)
    vectorizer.fit(preprocessed_data)
    
    return vectorizer

def getBOWVectorizeTxtData(preprocessed_data, vectorizer):
    text_bow = vectorizer.transform(preprocessed_data)
    print("Shape of matrix after one hot encodig ",text_bow.shape)
    
    return text_bow


# #### 1.5.2.2 TFIDF vectorizer

# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer

def getFitTFIDF_Vectorizer(preprocessed_data):
    vectorizer = TfidfVectorizer(min_df=10)
    vectorizer.fit(preprocessed_data)
    return vectorizer

def getTFIDFVectorizeTxtData(preprocessed_data, vectorizer):
    text_tfidf = vectorizer.transform(preprocessed_data)
    print("Shape of matrix after one hot encodig ",text_tfidf.shape)
    return text_tfidf


# ### $1.5.3$ Vectorizing Numerical features
# 

# In[ ]:


price_data = resource_data.groupby('id').agg({'price':'sum', 'quantity':'sum'}).reset_index()
project_data = pd.merge(project_data, price_data, on='id', how='left')


# In[ ]:


from sklearn.preprocessing import Normalizer
import warnings 
warnings.filterwarnings("ignore") 

def getFitNUM_Vectorizer(working_data, num_type):
    '''
    Fit on only train data.
    '''
    
    num_scalar = Normalizer()
    num_scalar.fit(working_data[num_type].values.reshape(-1,1)) # finding the mean and standard deviation of this data
    return num_scalar

def getNUM_Vectors(working_data, num_type, num_scalar):
    # Now standardize the data with above maen and variance.
    num_standardized = num_scalar.transform(working_data[num_type].values.reshape(-1, 1))
    #print(f"Mean : {num_scalar.mean_[0]}, Standard deviation : {np.sqrt(num_scalar.var_[0])}")
    return num_standardized


# ### $1.5.4$ Merging all the above features

# In[ ]:


from scipy.sparse import hstack

def getMergedFeatures(working_data, merge_on):
    valid_cols = []
    for key, value in working_data.items():
        if key in merge_on:
            valid_cols.append(value)
   
    return hstack(tuple(valid_cols))


# # Assignment $4$: Naive Bayes

# <h1>$2.$ Naive Bayes </h1></h1>

# <h2>$2.1$ Splitting data into Train and cross validation(or test): Stratified Sampling</h2>

# In[ ]:


# please write all the code with proper documentation, and proper titles for each subsection
# go through documentations and blogs before you start coding
# first figure out what to do, and then think about how to do.
# reading and understanding error messages will be very much helpfull in debugging your code
# when you plot any graph make sure you use 
    # a. Title, that describes your plot, this will be very helpful to the reader
    # b. Legends if needed
    # c. X-axis label
    # d. Y-axis label
    
from sklearn.model_selection import train_test_split

'''x_pos = project_data[project_data['project_is_approved']==1][:25000]
x_neg =project_data[project_data['project_is_approved']==0][:25000]
X = pd.concat([x_pos, x_neg]).reset_index()

y = X['project_is_approved'][:50000]'''

#Classes of X & project_data have almost same proportion.
X = project_data[:50000]
y = X['project_is_approved']

#Breaking into only train and test as I am gonna use cross-validation.
X_tr, X_test, y_tr, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)


# <h2>$2.2$ Make Data Model Ready: encoding numerical, categorical features</h2>

# In[ ]:


# Ordered dict will be used to ensure one to one correspondence between datapoints features and hstack_features.

from collections import OrderedDict


# In[ ]:


data_dict = {'X_tr':OrderedDict({}), 'X_test': OrderedDict({})}
cols_dict = OrderedDict({'cat_cols': ['school_state','clean_categories', 'clean_subcategories', 'project_grade_category', 'teacher_prefix'],
                 'num_cols': ['price', 'teacher_number_of_previously_posted_projects']
            })
hstack_features = []

for col_type, cols_name in cols_dict.items():
    if col_type == 'cat_cols':
        for cat_type in cols_name:
            print (cat_type)
            vectorizer = getFitCAT_Vectorizer(X_tr, cat_type, hstack_features)
            for data_type, data_part in [('X_tr', X_tr), ('X_test', X_test)]:
                hot_encode = getVectorizeCategData(data_part, cat_type, vectorizer)
                data_dict[data_type][cat_type] = hot_encode
    else:
        for num_type in cols_name:
            vectorizer = getFitNUM_Vectorizer(X_tr, num_type)
            hstack_features.append(num_type)
            for data_type, data_part in [('X_tr', X_tr), ('X_test', X_test)]:
                num_vectors = getNUM_Vectors(data_part, num_type, vectorizer)
                data_dict[data_type][num_type] = num_vectors
        


# <h2>$2.3$ Make Data Model Ready: encoding eassay, and project_title</h2>

# In[ ]:


for col_type in ['essay','project_title']:
    for data_type, data_part in [('X_tr', X_tr), ('X_test', X_test)]:
        preprocessed_data = getProcessedData(col_type, data_part)
        
        if data_type == 'X_tr':
            vectorizer_bog = getFitBOW_Vectorizer(preprocessed_data)
        text_bow = getBOWVectorizeTxtData(preprocessed_data, vectorizer_bog)
        data_dict[data_type]['%s_text_bow'%col_type] = text_bow
        
        if data_type == 'X_tr':
            vectorizer_tfidf = getFitTFIDF_Vectorizer(preprocessed_data)
        text_tfidf = getTFIDFVectorizeTxtData(preprocessed_data, vectorizer_tfidf)
        data_dict[data_type]['%s_text_tfidf'%col_type] = text_tfidf
    
    hstack_features += vectorizer_bog.get_feature_names()


# <h2>$2.4$ Appling NB() on different kind of featurization as mentioned in the instructions</h2>
# 
# <br>Apply Naive Bayes on different kind of featurization as mentioned in the instructions
# <br> For Every model that you work on make sure you do the step 2 and step 3 of instrucations

# In[ ]:


from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_auc_score, auc


# In[ ]:


from sklearn.model_selection import GridSearchCV
import math

def getAUCs(data_pnts_tr, y_tr):
    #Taken and updated from a sample given in Assignment 3.
    
    auc_tr = []
    auc_cv = []
    alphas = [10**i for i in range(-5,4)]
    parameters = {'alpha':alphas}
    
    mnb = MultinomialNB(class_prior=[0.5,0.5])
    clf = GridSearchCV(mnb, parameters, cv=3, scoring='roc_auc')
    clf.fit(data_pnts_tr, y_tr)

    auc_tr = clf.cv_results_['mean_train_score']
    auc_tr_std = clf.cv_results_['std_train_score']
    auc_cv = clf.cv_results_['mean_test_score'] 
    auc_cv_std= clf.cv_results_['std_test_score']

        
    return list(map(lambda x: math.log(x), alphas)), auc_tr, auc_cv

def plotPerformance(Ks, auc_tr, auc_cv, plt_title):
    plt.plot(Ks, auc_tr, label='AUC_Train')
    plt.plot(Ks, auc_cv, label='AUC_Validation')
    
    plt.scatter(Ks, auc_tr, label='Coordinates')
    plt.scatter(Ks, auc_cv, label='Coordinates')
    
    plt.xlabel('Hyperparameter - Log(alpha)')
    plt.ylabel('AUC')
    plt.title("AUC on various Log(alpha) using %s on text features"%plt_title)
    
    plt.legend()


# In[ ]:


from sklearn import metrics

def trainOnOptimalAlpha(optimal_alpha, X_tr, y_tr):
    mnb = MultinomialNB(alpha=optimal_alpha, class_prior=[0.5,0.5])
    mnb.fit(X_tr, y_tr)
    return mnb

def getROC_Data(data_pnts_test, y_test, data_pnts_tr, y_tr, mnb):
    predicted_y_test = mnb.predict_proba(data_pnts_test)[:, 1]
    predicted_y_tr = mnb.predict_proba(data_pnts_tr)[:, 1]
    
    fpr_test, tpr_test, thres_test = roc_curve(y_test, predicted_y_test)
    fpr_tr, tpr_tr, thres_tr = roc_curve(y_tr, predicted_y_tr)
    
    return [fpr_test, tpr_test, thres_test], [fpr_tr, tpr_tr, thres_tr]
    
def makeROC(test_data, train_data, plt_title):
    fpr_tr, tpr_tr, _ = train_data
    fpr_test, tpr_test, _ = test_data
    
    plt.plot(fpr_tr, tpr_tr, label='AUC_Train')
    plt.plot(fpr_test, tpr_test, label='AUC_Test')
    plt.title("ROC Curve using %s on text features"%plt_title)
    
    plt.xlabel('FPT')
    plt.ylabel('TPR')
    plt.legend()
    


# In[ ]:


#Reference: https://stackoverflow.com/questions/19233771/sklearn-plot-confusion-matrix-with-labels/48018785
#fmt='g' reason: https://stackoverflow.com/questions/29647749/seaborn-showing-scientific-notation-in-heatmap-for-3-digit-numbers
from sklearn.metrics import confusion_matrix

def getConfusionMatrix(mnb, data_pnts_test, y_true, plt_title):
    y_pred = mnb.predict(data_pnts_test)
    cm = confusion_matrix(y_true, y_pred) # Predicted values are column wise!
    
    ax = plt.subplot()
    sns.heatmap(cm, annot=True, ax=ax, fmt='g')
    
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('Actual Labels')
    ax.set_title('Confusion Matrix using %s on text features'%plt_title)
    ax.xaxis.set_ticklabels(['0','1'])
    ax.yaxis.set_ticklabels(['0','1'])
    


# ### $2.4.1$ Applying Naive Bayes on BOW,<font color='red'> SET $1$</font>

# In[ ]:


set1_cols = ['school_state','clean_categories', 'clean_subcategories', 'project_grade_category', 'teacher_prefix',
             'price', 'teacher_number_of_previously_posted_projects', 
             'essay_text_bow', 'project_title_text_bow']
plt_title1 = 'BOW'

data_pnts_tr1 = getMergedFeatures(data_dict['X_tr'], set1_cols)
data_pnts_test1 = getMergedFeatures(data_dict['X_test'], set1_cols)

alphas1, auc_tr1, auc_cv1 = getAUCs(data_pnts_tr1, y_tr)
plotPerformance(alphas1, auc_tr1, auc_cv1, plt_title1)

optimal_alpha1 = 1
mnb1 = trainOnOptimalAlpha(optimal_alpha1, data_pnts_tr1, y_tr)
roc_data_test1, roc_data_train1 = getROC_Data(data_pnts_test1, y_test, data_pnts_tr1, y_tr, mnb1)


# In[ ]:


makeROC(roc_data_test1, roc_data_train1, plt_title1)


# In[ ]:


getConfusionMatrix(mnb1, data_pnts_tr1, y_tr, plt_title1)


# In[ ]:


getConfusionMatrix(mnb1, data_pnts_test1, y_test, plt_title1)


# #### $2.4.1.1$ Top $10$ important features of positive class from<font color='red'> SET $1$</font>

# In[ ]:


pos_ftr = mnb1.feature_log_prob_[1, :].argsort()[-10:]
table = PrettyTable()
table.field_names = ["Feature_Name", "Log_Probability"]

for idx in pos_ftr:
    table.add_row([hstack_features[idx], mnb1.feature_log_prob_[0, idx]])

print (table)


# #### $2.4.1.2$ Top $10$ important features of negative class from<font color='red'> SET $1$</font>

# In[ ]:


neg_ftr = mnb1.feature_log_prob_[0, :].argsort()[-10:]
table = PrettyTable()
table.field_names = ["Feature_Name", "Log_Probability"]

for idx in neg_ftr:
    table.add_row([hstack_features[idx], mnb1.feature_log_prob_[0, idx]])

print (table)


# ### $2.4.2$ Applying Naive Bayes on TFIDF,<font color='red'> SET $2$</font>

# In[ ]:


set2_cols = ['school_state','clean_categories', 'clean_subcategories', 'project_grade_category', 'teacher_prefix',
             'price', 'teacher_number_of_previously_posted_projects', 
             'essays_text_tfidf', 'project_title_text_tfidf']
plt_title2 = 'TFIDF'

data_pnts_tr2 = getMergedFeatures(data_dict['X_tr'], set2_cols)
data_pnts_test2 = getMergedFeatures(data_dict['X_test'], set2_cols)

alphas2, auc_tr2, auc_cv2 = getAUCs(data_pnts_tr2, y_tr)
plotPerformance(alphas2, auc_tr2, auc_cv2, plt_title2)
optimal_alpha2 = 1
mnb2 = trainOnOptimalAlpha(optimal_alpha2, data_pnts_tr2, y_tr)
roc_data_test2, roc_data_train2 = getROC_Data(data_pnts_test2, y_test, data_pnts_tr2, y_tr, mnb2)


# In[ ]:


makeROC(roc_data_test2, roc_data_train2, plt_title2)


# In[ ]:


getConfusionMatrix(mnb2, data_pnts_tr2, y_tr, plt_title2)


# In[ ]:


getConfusionMatrix(mnb2, data_pnts_test2, y_test, plt_title2)


# #### $2.4.2.1$ Top $10$ important features of positive class from<font color='red'> SET $2$</font>

# In[ ]:


pos_ftr = mnb2.feature_log_prob_[1, :].argsort()[-10:]
table = PrettyTable()
table.field_names = ["Feature_Name", "Log_Probability"]

for idx in pos_ftr:
    table.add_row([hstack_features[idx], mnb2.feature_log_prob_[0, idx]])

print (table)


# #### $2.4.2.2$ Top $10$ important features of negative class from<font color='red'> SET $2$</font>

# In[ ]:


neg_ftr = mnb2.feature_log_prob_[0, :].argsort()[-10:]
table = PrettyTable()
table.field_names = ["Feature_Name", "Log_Probability"]

for idx in neg_ftr:
    table.add_row([hstack_features[idx], mnb2.feature_log_prob_[0, idx]])

print (table)


# <h1>$3.$ Conclusions</h1>

# In[ ]:


table = PrettyTable()
table.field_names = ["Vectorizer", "Hyper Parameter", "AUC"]

table.add_row(["BOW", optimal_alpha1, auc(roc_data_test1[0],roc_data_test1[1])])
table.add_row(["TFIDF", optimal_alpha2, auc(roc_data_test2[0],roc_data_test2[1])])

print (table)

