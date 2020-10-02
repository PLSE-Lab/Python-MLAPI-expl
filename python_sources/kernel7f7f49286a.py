import nltk
import numpy as np
import json
import pandas as pd
import ast 
import sklearn
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
import spacy
from nltk import Tree
en_nlp = spacy.load('en')
from nltk.stem.lancaster import LancasterStemmer
st = LancasterStemmer()
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from pprint import pprint

with open("../input/dev-v2.0.json") as file:
    data = json.load(file)

# -----------------------------------------------#
#print(data)
#data = pd.read_csv("train_detect_sent.csv").reset_index(drop=True)
#data.shape

#ast.literal_eval(data["sentences"][0])
#data = data[data["sentences"].apply(lambda x: len(ast.literal_eval(x)))<11].reset_index(drop=True)
       # --------------------------------------#

def create_features(data):
    train = pd.DataFrame()
     
    for k in range(len(data["euclidean_dis"])):
        dis = ast.literal_eval(data["euclidean_dis"][k])
        for i in range(len(dis)):
            train.loc[k, "column_euc_"+"%s"%i] = dis[i]
    
    print("Finished")
    
    for k in range(len(data["cosine_sim"])):
        dis = ast.literal_eval(data["cosine_sim"][k].replace("nan","1"))
        for i in range(len(dis)):
            train.loc[k, "column_cos_"+"%s"%i] = dis[i]
            
    train["target"] = data["target"]
    return train
















