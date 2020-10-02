# This script seeks to predict alcohol content given a beer name

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

def main():
    # Read the beers and only consider rows with ABV defined.
    beers = pd.read_csv("../input/beers.csv")
    beers = beers[pd.notnull(beers.abv)]
    
    # Form features based off of word counts
    vectorizer = CountVectorizer(analyzer="word", max_features=200,\
        stop_words="english")
    features = vectorizer.fit_transform(beers.name)
    targets = beers.abv.astype("float64")
    
    # Split data for testing
    feats_train, feats_test, targets_train, targets_test =\
        train_test_split(features, targets, random_state=23)
    
    # Fit and score
    rgr = RandomForestRegressor()
    rgr.fit(feats_train, targets_train)
    scores = rgr.score(feats_test, targets_test)
    # Prints around 0.35
    print(scores.mean())
    
if __name__ == "__main__":
    main()
