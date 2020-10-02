from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

""" to install or update packages and modules: python -m pip install """

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import random
import sys
import seaborn as sns
import spacy  # for lemmatization
import nltk   # for word stemming

""" ################ I/O ################# """
folder=r"C:/Users/admin/Desktop/LEARNING/KAGGLE/Spooky Author comp/"
class_column="author"
class_proxies={"Y": {"EAP":1,"MWS":2,"HPL":3,"unlabeled":-1}}
doc_column="text"

""" ######## Reading & JOINING train data (incl.y) & test data (fake y) for pre-proc ######### """
inname=r'train.csv'
trainset = pd.read_csv(folder+inname,encoding="utf-8")
inname=r'test.csv'
testset = pd.read_csv(folder+inname,encoding="utf-8")

# from sklearn.datasets import load_files
# nlp_loader = load_files(folder+inname)
# # load_files returns a bunch, containing training texts and training labels
# text_train, y_train = nlp_loader.data, nlp_loader.target

testset[class_column]="unlabeled" ### un-labeled TEST samples
print("\nTrain Data dimensions: ",trainset.shape)
print("\nTest Data dimensions: ",testset.shape)
dataset=pd.concat([trainset,testset],ignore_index=True)
print("\nJoined dataset dimensions: ",dataset.shape)
dataset[doc_column] = [doc.replace( "<br />", " ") for doc in dataset[doc_column]] ### clear HTML
""" ########################################################################################## """

""" STEP 1. DROP duplicate rows (double entries) """
print("\nAre there even any duplicates in the current state of the frame?\n")
print(dataset.duplicated(keep=False).any())
if dataset.duplicated(keep=False).any():
    dataset.drop_duplicates(inplace=True)
    print("\nHow about now?\n")
    print(dataset.duplicated(keep=False).any())

""" STEP 2. Cleaned-up data exploration and NA removal """
print("\nData dimensions: ",dataset.shape)
g = dataset.columns.to_series().groupby(dataset.dtypes).groups
print("\nDatatypes of various columns:\n",{k.name: v for k, v in g.items()})
print("\nNulls:")
print(dataset.isnull().sum())
# dataset.XX.fillna(value=dataset.XX.value_counts().idxmax(), inplace=True)
print("Number of doc's in the training set: {}".format(len(trainset)))
print("Samples per class (TRAIN portion):\n{}".format(trainset[class_column].value_counts()))
print("Samples per class (TEST portion j/c):\n{}".format(testset[class_column].value_counts()))

################ CountVectorizer TOKENIZATION approach for Bag-of-Words #################
tokenize=False
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
""" Use default SKLearn or custom, lemmatization-based tokenizer? """
tokenizer="default"
# tokenizer="custom"
# create a custom tokenizer using the SpaCy document processing pipeline
def custom_tokenizer(document):
    lemmatizer = spacy.load('en_core_web_sm')
    # lemmatizer = spacy.load('de_core_news_sm') ### for German
    doc_spacy = lemmatizer(document, disable=['ner','parser'])
    return [token.lemma_ for token in doc_spacy]

max_vocab=5000
min_df=5
max_df=1.0
ngram_range=(1,3)
stop_words="english"
if tokenize:
    if tokenizer=="custom":
        bag = CountVectorizer(  tokenizer=custom_tokenizer, min_df=min_df,max_df=max_df, max_features=max_vocab,
                                ngram_range=ngram_range, stop_words=stop_words).fit(dataset[doc_column])
    else:
        bag = CountVectorizer(  min_df=min_df,max_df=max_df, max_features=max_vocab,
                                ngram_range=ngram_range, stop_words=stop_words).fit(dataset[doc_column])
    vocab = bag.get_feature_names()
    print("Vocabulary size: {}".format(len(vocab)))
    print("First 20 features:\n{}".format(vocab[:20]))
    print("Every 200th feature:\n{}".format(vocab[::200]))
    """ BEWARE: The vectorizer returns a SPARSE matrix, NOT a dense numpy array"""
    """ For use in Pandas convert to array !!!                                 """
    XY=pd.DataFrame(bag.transform(dataset[doc_column]).toarray(), columns=vocab)
    print(len(vocab))
    print(type(vocab))
    print(bag.transform(dataset[doc_column]).toarray().shape)
    print(type(bag.transform(dataset[doc_column]).toarray()))
    # XY[vocab]=bag.transform(dataset[doc_column]).toarray()
else: XY=pd.DataFrame()

################ TF-IDF TOKENIZATION approach for Bag-of-Words #################
"""???"""

# """ STEP #. Lemmatization .OR. Stemming """
# print("Stemming the corpus:")
# stemmer = nltk.stem.PorterStemmer()

XY["doc_length"]=dataset[doc_column].str.len()
XY["N_commas"]=dataset[doc_column].str.count(",")
XY["N_exclams"]=dataset[doc_column].str.count("!")
XY["N_3dots"]=dataset[doc_column].str.count("...")
XY["N_sentences"]=dataset[doc_column].str.count(".")
### ADDING THE TARGET LABEL y ###
class_proxies={"Y":    {"EAP":1,"MWS":2,"HPL":3,"unlabeled":-1}}
print("Label substitution with numbers:\n",class_proxies["Y"])
XY["Y"]=dataset[class_column]
XY.replace(class_proxies, inplace=True)

""" Saving the dataset - ready for model training! """
outname=r"XY.csv"
XY[XY.Y!=-1].to_csv(folder+outname, index=False,encoding="utf-8")
print("\nSaved the dataset to file (%s) - ready for model training!!! \n" % outname)


""" Saving the TEST set - ready for model testing! """
outname=r"XYtest.csv"
XY[XY.Y==-1].drop("Y",1).to_csv(folder+outname, index=False,encoding="utf-8")
print("\nSaved the TEST set to file (%s) - ready for model testing!!! \n" % outname)

print("\n Training Data dimensions: ",XY[XY.Y!=-1].shape)
print("\n Test Data dimensions: ",XY[XY.Y==-1].shape)
# print("\n Training Data sample:\n",XY[XY.Y!=-1].head(10))
