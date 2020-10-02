# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output

from scipy import sparse
from sklearn.feature_extraction import FeatureHasher
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import log_loss
import pickle

##################
# Loading Dataset
##################

print("# Loading Data")
labelCats = pd.read_csv("../input/label_categories.csv") # label_id, category
appLabels = pd.read_csv("../input/app_labels.csv")    # app_id, label_id
appEvents = pd.read_csv("../input/app_events.csv",usecols =["event_id","app_id"]) # 1GB event_id, app_id, is_installed (remove-- is constant), is_active
events = pd.read_csv("../input/events.csv",usecols=["event_id","device_id"])   # 200MB event_id, device_id, timestamp, longitude, latitude
devices = pd.read_csv("../input/phone_brand_device_model.csv")    # device_id, phone_brand, device_model
test = pd.read_csv("../input/gender_age_test.csv")   # device_id
train = pd.read_csv("../input/gender_age_train.csv")  # device_id, gender, age, group
print("# Data Loaded")

#####################
#   Label Categories
#####################
print("# Read Labels Categories")

labelCats["category"] = labelCats["category"].apply(lambda x: " ".join(str(x).replace("-"," ").replace("/"," ").replace("("," ").replace(")"," ").split()))
categories = labelCats.groupby("label_id")["category"].apply(lambda x: " ".join(" ".join(" ".join(["Cat:"+str(i) for i in s.split()]) for s in x).split()))

del labelCats
##################
#   App Labels
##################
print("# Read App Labels")

appLabels["category"] = appLabels["label_id"].map(categories)
categories = appLabels.groupby("app_id")["category"].apply(lambda x: " ".join(str(s) for s in x))

del appLabels
##################
#   App Events
##################
print("# Read App Events")

appEvents["category"] = appEvents["app_id"].map(categories)
categories = appEvents.groupby("event_id")["category"].apply(lambda x: " ".join(str(s) for s in x))

del appEvents
##################
#     Events
##################
print("# Read Events")

events["category"] = events["event_id"].map(categories)
categories = events.groupby("device_id")["category"].apply(lambda x: " ".join(str(s) for s in x))

del events
##################
#  Train and Test
##################
print("# Generate Train and Test")

train["category"] = train["device_id"].map(categories)
test["category"] = test["device_id"].map(categories)

del categories
##################
#   Vectorizer
##################
print("# Vectorizing Train and Test")

def vectorize(train, test, columns, fillMissing):
    data = pd.concat((train, test), axis=0, ignore_index=True)
    data = data[columns].astype(np.str).apply(lambda x: " ".join(s for s in x), axis=1).fillna(fillMissing)
    split_len = len(train)

    # TF-IDF Feature
    vectorizer = TfidfVectorizer(min_df=1)
    data = vectorizer.fit_transform(data)

    train = data[:split_len, :]
    test = data[split_len:, :]
    return train, test


# Group Labels
train, test = vectorize(train,test,["category"],"missing")

pickle.dump(train,open("../generated/categoriesTrain.p","wb"))
pickle.dump(test,open("../generated/categoriesTest.p","wb"))