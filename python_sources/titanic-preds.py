from __future__ import division
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import csv
import re


###  load data

df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")


### clean the data

# age
df["Age"] = df["Age"].fillna(df["Age"].median())
test_df["Age"] = test_df["Age"].fillna(test_df["Age"].median())

# gender
df.loc[df["Sex"] == "male", "Sex"] = 0
df.loc[df["Sex"] == "female", "Sex"] = 1

test_df.loc[test_df["Sex"] == "male", "Sex"] = 0
test_df.loc[test_df["Sex"] == "female", "Sex"] = 1

# embarkation
df["Embarked"] = df["Embarked"].fillna("S")
test_df["Embarked"] = test_df["Embarked"].fillna("S")

emb_df = pd.get_dummies(df['Embarked'])
test_emb_df = pd.get_dummies(test_df['Embarked'])

df = pd.concat([df, emb_df], axis=1)
test_df = pd.concat([test_df, test_emb_df], axis=1)

# cabin
df['Cabin'][df.Cabin.isnull()] = 'U0'
test_df['Cabin'][test_df.Cabin.isnull()] = 'U0'

df = df.fillna(0)
test_df = test_df.fillna(0)

# split feature and target data
tar_df = df["Survived"]
feat_df = df[["Name", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Cabin", "S", "C", "Q"]]
test_df = test_df[["Name", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Cabin", "S", "C", "Q"]]


### feature engineering

title_list = ["sir", "lady", "mssr", "master", "dr", "doctor", "major", "colonel", "mlle", "baroness", "countess", "count",
              "mjr", "col", "the", "rev", "baron", "duke", "duchess", "mr", "mrs", "miss"]


def title_engineering(value):
    return value.split(",")[1].split()[0].lower().replace(".", "")


def cabin_engineering(value):
    return re.compile("([a-zA-Z]+)").search(value).group()


def feature_encoder(value):
    return 1 if value else 0

# name
feat_df["Name"] = feat_df["Name"].apply(title_engineering)
test_df["Name"] = test_df["Name"].apply(title_engineering)

for title in title_list:
    feat_df[title] = feat_df["Name"][feat_df.Name == title]
    test_df[title] = test_df["Name"][test_df.Name == title]
    feat_df[title] = feat_df[title].fillna(0).apply(feature_encoder)
    test_df[title] = test_df[title].fillna(0).apply(feature_encoder)

# family size
feat_df["family_size"] = feat_df["SibSp"] + feat_df["Parch"]
test_df["family_size"] = test_df["SibSp"] + test_df["Parch"]

# fare
feat_df["div_fare"] = feat_df["Fare"] / (feat_df["family_size"] + 1)
test_df["div_fare"] = test_df["Fare"] / (test_df["family_size"] + 1)

# cabin
feat_df["deck"] = pd.factorize(feat_df["Cabin"].apply(cabin_engineering))[0]
test_df["deck"] = pd.factorize(test_df["Cabin"].apply(cabin_engineering))[0]

# remove redundant columns
feat_df = feat_df.drop(["Cabin", "Name", "SibSp", "Parch", "Fare"], axis=1)
test_df = test_df.drop(["Cabin", "Name", "SibSp", "Parch", "Fare"], axis=1)

feature_list = feat_df.columns


### normalization

sc = StandardScaler()

test_df = test_df[feat_df.columns]

for col in ["Age", "Pclass", "deck", "div_fare", "family_size"]:
    feat_df[col] = sc.fit_transform(feat_df[col])
    test_df[col] = sc.fit_transform(test_df[col])



### training & cross validation

y = np.array(tar_df)
X = feat_df.as_matrix()
test_X = test_df.as_matrix()

lr = LogisticRegression()
lr.fit(X, y)

kfold = KFold(X.shape[0], n_folds=10)

print (np.mean(cross_val_score(lr, X, y, cv=kfold)))


### Random Forest & feature importance

# rf = RandomForestClassifier()
# rf.fit(X, y)

# print np.mean(cross_val_score(rf, X, y, cv=kfold))

# feature_importance = rf.feature_importances_

# feature_importance = 100.0 * (feature_importance / feature_importance.max())
# fi_threshold = 3

# important_idx = np.where(feature_importance > fi_threshold)[0]

# important_features = feature_list[important_idx]
# print "\n", important_features.shape[0], "Important features(>", fi_threshold, "% of max importance):\n", \
#         important_features

# sorted_idx = np.argsort(feature_importance[important_idx])[::-1]
# print "\nFeatures sorted by importance (DESC):\n", important_features[sorted_idx]

# # plot feature importance
# pos = np.arange(sorted_idx.shape[0]) + .5
# plt.subplot(1, 2, 2)
# plt.barh(pos, feature_importance[important_idx][sorted_idx[::-1]], align='center')
# plt.yticks(pos, important_features[sorted_idx[::-1]])
# plt.xlabel('Relative Importance')
# plt.title('Variable Importance')
# plt.draw()
# plt.show()


### testing

preds = lr.predict(test_X)
# preds = rf.predict(test_X)



### write results to csv

with open("prediction.csv", "w") as f:
    p_writer = csv.writer(f, delimiter=' ', lineterminator='\n')
    for p in preds:
        p_writer.writerow([p])
