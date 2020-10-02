#!/usr/bin/env python
# coding: utf-8

# # Goal
# COVID-19 Open Research Dataset Challenge has 10 Tasks where each task has several sub tasks, each an insightful question focused on some aspect of Covid-19 pandemic and the virus causing it. In recent times we have seen remarkable progress in Deep-Learning models for machine reading comprehension (MRC) style question answering (Q&A). The MRC style Q&A model takes as input a question and a context and extracts the answer from the context. 
# 

# In[ ]:


pip install pickledb


# In[ ]:


import json
import pandas as pd

def extractData(csvfile, jsonfile):
    data = pd.read_csv(csvfile) # add exception handling incase filepath it wrong or does not exists
    data_with_abstract = data[data.abstract.notnull()]
    with open(jsonfile, 'a+') as json_out_file:
        for index, row in data_with_abstract.iterrows():
            json_data = {}
            json_data["id"] = index
            json_data['articleID'] = row['cord_uid']
            json_data["text"] = row['abstract']
            if json_data["text"] == "Unknown":
                continue
            json.dump(json_data, json_out_file)
            json_out_file.write('\n')
    print("Done !!!")
            
inputfile = "/kaggle/input/CORD-19-research-challenge/metadata.csv"
outputfile = "covid-19.json"

extractData(inputfile, outputfile)


# Now we are going to compute the USE embeddings for all the abstract and store it in a pickle file. We will also store the corresponding abstract in a pickleDB file.

# In[ ]:


import tensorflow_hub as hub
import numpy as np
import pickle
import pickledb
import random

class EmbeddingStore:
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

    def __init__(self, filename):
        self.embedding_file = filename
        self.all_docs_embeddings = self.read_embeddings()
        if self.all_docs_embeddings is None:
            print("all_docs initialized to None")

    def save_embeddings(self):
        f = open(self.embedding_file, 'wb')
        pickle.dump(self.all_docs_embeddings, f)

    def read_embeddings(self):
        try:
            f = open(self.embedding_file, 'rb')
        except FileNotFoundError:
            return None
        try:
            embeddings_ = pickle.load(f)
        except EOFError:
            embeddings_ = None
        return embeddings_

    def add_embeddings(self, new_embeddings_):
        if self.all_docs_embeddings is None:
            combined_embeddings_ = new_embeddings_
        else:
            combined_embeddings_ = np.concatenate((self.all_docs_embeddings, new_embeddings_))
        self.all_docs_embeddings = combined_embeddings_

    def dump(self):
        self.save_embeddings()

    def add_para(self, new_para_):
        new_embeddings_ = self.embed([new_para_])
        self.add_embeddings(new_embeddings_)

    def find_top_k_match(self, query, k):
        #print("Query received = {}".format(query))
        query_embedding_ = self.embed(query)
        #print(query_embedding_.shape)
        #print(query_embedding_[0].shape)
        #print(self.all_docs_embeddings.shape)
        corr = np.inner(query_embedding_[0], self.all_docs_embeddings)
        #print(corr)
        values = np.argpartition(corr, -k)[-k:]
        #print("Top K matches for = {} at {}".format(query, values))
        return values

class ParaStore:

    def __init__(self, docsfilename, embeddingsfile):
        self.docfile = docsfilename
        self.embeddingstore = EmbeddingStore(embeddingsfile)
        self.all_docs = self.read_para()
        self.nos_docs = self.all_docs.totalkeys()
        print("There are {} docs in the store".format(self.nos_docs))

    def save_para(self):
        self.all_docs.dump()

    def read_para(self):
        docs = pickledb.load(self.docfile, False)
        return docs

    # Changed this function to use the aid instead of text
    def already_exists(self, new_para):
        allkeys = self.all_docs.getall()
        for key in allkeys:
            if new_para["aid"] == self.all_docs.get(key)["aid"]:
                print("{} already in the doc store".format(new_para["aid"]))
                return True
        return False

    def add_para(self, new_para):
        text = new_para["text"]
        if not self.already_exists(new_para):
            self.all_docs.set(str(self.nos_docs), new_para)
            self.embeddingstore.add_para(text)
            self.nos_docs = self.nos_docs+1
            if self.nos_docs % 1000 == 0:
                self.dump()
    def dump(self):
        self.all_docs.dump()
        self.embeddingstore.dump()

    def get_para(self, nos):
        return self.all_docs.get(str(nos))

    def get_matching_para(self, query):
        pos = self.embeddingstore.find_top_match(query)
        match = self.get_para(pos)
        return match

    def get_matching_k_para(self, query, k):
        positions = self.embeddingstore.find_top_k_match(query, k)
        matches = {}
        for i in positions:
            match = self.get_para(i)
            matches[str(i)] = match
        return matches

    def filter_k_randomly(self, matches, k):
        kmatches = {}
        #print("Filtering {} from {} matches".format(k, len(matches)))
        if len(matches) <= k:
            return matches
        else:
           indexes = random.sample(matches.keys(), k)
        for i in indexes:
            kmatches[i] = matches[i]
        return kmatches

    def applyFilter(self, filt, text):
        for name in filt:
            if text and name.lower() in text.lower():
                return True
        return False
        
    def get_including_k_para(self, query, filt, k):
        matches = {}
        allkeys = self.all_docs.getall()
        for key in allkeys:
            candidate = self.all_docs.get(key)
            if (query[0] in candidate["text"]) and (self.applyFilter(filt, candidate["text"])):
                matches[key] = candidate
        kmatches = self.filter_k_randomly(matches, k)
        return kmatches


# # Optional
# Import the data i.e. compute the USE embeddings for all the data and store them. Its take considerable time to compute these embeddings. Since it needs to be computed only once, we have computed the embeddings of 37K abstracts and uploaded it as a dataset.

# In[ ]:


def openStore(docfile, embedding_file):
    docstore = ParaStore(docfile, embedding_file)
    return docstore

def importFromDrqa(filename, dbfile, embedfile):
    docstore = openStore(dbfile, embedfile)
    with open(filename) as f:
        line = f.readline()
        while line:
            para = {}
            #print(line)
            data = json.loads(line)
            #print(data["id"])
            para["aid"] = data["articleID"]
            para["text"] = data["text"]
            docstore.add_para(para)
            line = f.readline()
        docstore.dump()
        
filename = "covid-19.json"
pickledbfile = "cord19.db"
embedfile = "cord19.pkl"

importFromDrqa(filename, pickledbfile, embedfile)
print("All Document Imported")


# We are going to use the sub task description to find the abstract that are related to them using cosine similarity.

# In[ ]:


import csv
import json

def openStore(docfile, embedding_file):
    docstore = ParaStore(docfile, embedding_file)
    return docstore

def getTopKMatch(dbfile, embedfile, query, k=1):
    # open the store
    query_vector = [query]
    print("The query vector is {}".format(query_vector))
    docstore = openStore(dbfile, embedfile)
    matches = docstore.get_matching_k_para(query_vector, k)
    return matches

def getTopKStringMatch(dbfile, embedfile, filt, query, k=1):
    # open the store
    query_vector = [query]
    print("The query vector is {}".format(query_vector))
    docstore = openStore(dbfile, embedfile)
    matches = docstore.get_including_k_para(query_vector, filt, k)
    return matches

def createBigRow(med, aids, abstracts):
    print("Creating a big row of {}, {}".format(med,aids))
    big_row = list()
    big_row.append(med)
    for i in aids:
        big_row.append(i)
    for abs in abstracts:
        big_row.append(abs)
    return big_row

def pad2K(data, k):
    if len(data) >= k:
        return data
    else:
        for i in range(len(data), k):
            data.append("None")
    return data

covid19_names = {
    'COVID19',
    'COVID-19',
    '2019-nCoV',
    '2019-nCoV.',
    'coronavirus disease 2019',
    'Corona Virus Disease 2019',
    '2019-novel Coronavirus',
    'SARS-CoV-2',
}

header = ['Question', 'covid-uid 1', 'covid-uid 2', 'covid-uid 3', 'covid-uid 4', 'covid-uid 5', 'Abstract 1', 'Abstract 2', 'Abstract 3', 'Abstract 4', 'Abstract 5']
def evalQueryDetailCSV(filename, outfile, match):
    with open(filename) as f, open(outfile, 'w') as output:
        filewriter = csv.writer(output, delimiter=',')
        line = f.readline()
        filewriter.writerow(header)
        while line:
            line = line.replace("\'", "\"")
            data = json.loads(line)
            q_id = data['id']
            q = data['text']
            med = q
            if match == 'USE':
                topk = getTopKMatch(paraFile, embeddingFile, q, 5)
            else:
                topk = getTopKStringMatch(paraFile, embeddingFile, covid19_names, q, 5)
            aids = []
            abstracts = []
            for key in topk.keys():
                para = topk.get(key)
                aids.append(para["aid"])
                #print(para["aid"])
                abstracts.append(para["text"].encode('utf-8'))
            aids = pad2K(aids, 5)
            abstracts = pad2K(abstracts, 5)
            big_row = createBigRow(med, aids, abstracts)
            filewriter.writerow(big_row)
            line = f.readline()
            
queryFile = "/kaggle/input/taskdescription/task-1-original-q.json"
paraFile = "cord19.db"  # Use precomputed embeddings, this is to save time
embeddingFile = "cord19.pkl" # Use precomputed embeddings, this is to save time


#paraFile = "/kaggle/input/embeddings/cord19.db"  # Use precomputed embeddings, this is to save time
#embeddingFile = "/kaggle/input/embeddings/cord19.pkl" # Use precomputed embeddings, this is to save time


evalQueryDetailCSV(queryFile, "top-5.csv", 'USE')

