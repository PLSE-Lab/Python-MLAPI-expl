#!/usr/bin/env python
# coding: utf-8

# Make my hand dirty with Quora challenge.This is my first competition.Please suggest me to do better and it if helpful then don't forget to up vote and leave you feedback  

# In[ ]:


#Import Initial Packages
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
import gensim
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from textblob import TextBlob
from nltk.stem import WordNetLemmatizer
import re 
from collections import namedtuple
import multiprocessing
import datetime
import os

tokenizer = RegexpTokenizer(r'\w+')
stopwords = stopwords.words("english")
lemmatizer = WordNetLemmatizer()


# **Data Analysis and Natural Language processing**

# In[ ]:


df_train = pd.read_csv("../input/train.csv")


# In[ ]:


df_train.head(3)


# In[ ]:


df_train_set1 = df_train[["qid1","question1"]]
df_train_set2 = df_train[["qid2","question2"]]


# In[ ]:


df_train_set1.columns = ["qid","question"]
df_train_set2.columns =["qid","question"]
df_train_set = pd.concat([df_train_set1,df_train_set2],axis=0)
print("df_train_set_1 :",df_train_set1.shape)
print("df_train_set_2 :",df_train_set1.shape)
print("df_train_set :",df_train_set.shape)


# In[ ]:


df_train_set1.head(3)


# In[ ]:


df_train_set2.head(3)


# In[ ]:


df_train_set.head(3)


# In[ ]:


print(df_train_set1["question"].describe())


# In[ ]:


print(df_train_set2["question"].describe())


# In[ ]:


print(df_train_set["question"].describe())


# In[ ]:


#Lemmatizing words.Ex - consider cats and cat are different word even they both are mostly similar cotext or reffer to similar things
print(lemmatizer.lemmatize("cats"))


# In[ ]:


#Language Processing
def get_processed_text(text=""):
    """
    Remove stopword,lemmatizing the words and remove special character to get important content
    """
    clean_text = re.sub('[^a-zA-Z0-9 \n\.]', ' ', text)
    tokens = tokenizer.tokenize(clean_text)
    tokens = [lemmatizer.lemmatize(token.lower().strip()) for token in tokens
              if token not in stopwords and len(token) >= 2]
    return tokens


# In[ ]:


text = "What is the best phone to buy below 15k"
print ("Original Text : ",text)
processed_text = " ".join(get_processed_text(text))
print ("Processed Text : ",processed_text) #Remove special character(?),english stop words(is,the,by,to,in) 


# In[ ]:


df_train_set.head()


# In[ ]:


#Process and clean up traing set
alldocuments = []
analyzedDocument = namedtuple('AnalyzedDocument', 'words tags')       
keywords = []
for index,record in df_train_set[0:100].iterrows():
    qid = str(record["qid"])
    question = str(record["question"])
    tokens = get_processed_text(question)
    words = tokens
    words_text = " ".join(words)
    words = gensim.utils.simple_preprocess(words_text)
    tags = [qid]
    alldocuments.append(analyzedDocument(words, tags))


# In[ ]:


alldocuments[0:3]


# In[ ]:


def train_and_save_doc2vec_model(alldocuments,document_model="model1",m_iter=100,m_min_count=2,m_size=100,m_window=5):
            print ("Start Time : %s" %(str(datetime.datetime.now())))
            #Train Model
            cores = multiprocessing.cpu_count()
            abs_path = os.getcwd()
            saved_model_name = "doc_2_vec_%s" %(document_model)
            doc_vec_file = "%s" %(saved_model_name)
            if document_model == "model1":
                # PV-DBOW 
                model_1 = gensim.models.Doc2Vec(alldocuments,dm=0,workers=cores,size=m_size, window=m_window,min_count=m_min_count,iter=m_iter,dbow_words=1)
                model_1.save("%s" %(doc_vec_file))
                print ("model training completed : %s" %(doc_vec_file))
            elif document_model == "model2":
                # PV-DBOW 
                model_2 = gensim.models.Doc2Vec(alldocuments,dm=0,workers=cores,size=m_size, window=m_window,min_count=m_min_count,iter=m_iter,dbow_words=0)
                model_2.save("%s" %(doc_vec_file))
                print ("model training completed : %s" %(doc_vec_file))
            elif document_model == "model3":
                # PV-DM w/average
                model_3 = gensim.models.Doc2Vec(alldocuments,dm=1, dm_mean=1,size=m_size, window=m_window,min_count=m_min_count,iter=m_iter)
                model_3.save("%s" %(doc_vec_file))
                print ("model training completed : %s" %(doc_vec_file))

            elif document_model == "model4":
                # PV-DM w/concatenation - window=5 (both sides) approximates paper's 10-word total window size
                model_4 = gensim.models.Doc2Vec(alldocuments,dm=1, dm_concat=1,workers=cores, size=m_size, window=m_window,min_count=m_min_count,iter=m_iter)
                model_4.save("%s" %(doc_vec_file))
                print ("model training completed : %s" %(doc_vec_file))
            print ("Record count %s" %len(alldocuments))
            print ("End Time %s" %(str(datetime.datetime.now())))


# In[ ]:


#Train model
train_and_save_doc2vec_model(alldocuments)


# In[ ]:


ls -ltr


# In[ ]:


def get_question_similarity_score(question1="",question2=""):
    print ("question1 - ",question1)
    print ("question2 - ",question2)
    model_name = "%s" %("doc_2_vec_model1")
    model_saved_file = "%s" %(model_name)
    model = gensim.models.doc2vec.Doc2Vec.load(model_saved_file)
    
    question_token1 = get_processed_text(question1)
    tokenize_text1 = ' '.join(question_token1)
    tokenize_text1 = gensim.utils.simple_preprocess(tokenize_text1)
    infer_vector_of_question1 = model.infer_vector(tokenize_text1)
    
    print("tokenize_text1",tokenize_text1,"infer_vector_of_question1",infer_vector_of_question1)
    
    question_token2 = get_processed_text(question2)
    tokenize_text2 = ' '.join(question_token2)
    tokenize_text2 = gensim.utils.simple_preprocess(tokenize_text2)
    infer_vector_of_question2 = model.infer_vector(tokenize_text2)
    
    print("tokenize_text2",tokenize_text2,"infer_vector_of_question2",infer_vector_of_question2)
    similarity_score = 1
    #similarity_score = model.docvecs.most_similar(infer_vector_of_question1)
    msg= "question : %s model_name : %s " %(question,model_name)
   
    return similarity_score


# In[ ]:


test_id_list = []
similarity_score_list = []
df_test = pd.read_csv("../input/test.csv")
for index,record in df_test[0:10].iterrows():
    test_id = str(record["test_id"])
    question1 = str(record["question1"])
    question2 = str(record["question2"])
    similarity_score = get_question_similarity_score(question1,question2)
    test_id_list.append(test_id)
    similarity_score_list.append(similarity_score)
    
#submission = pd.DataFrame({
#"test_id": test_id_list,
#"is_duplicate": similarity_score_list})
#submission.to_csv('./first_submition.csv', index=False)


# **Final Submition yet to come.Shorty i will update**
