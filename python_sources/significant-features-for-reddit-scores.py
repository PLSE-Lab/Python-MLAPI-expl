"""
Author: Yash Patel
Name: Reddit_Most_Significant.py
Description: Finds the most significant features for when 
trying to achieve the highest Reddit score
"""

import sqlite3
import pandas as pd
import numpy as np
import datetime
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
import matplotlib.pyplot as plt

sql_conn = sqlite3.connect('../input/database.sqlite')
sql_cmd = "SELECT created_utc, subreddit_id, author_flair_css_class, author_flair_text, " \
            "removal_reason, gilded, downs, archived, score, body, distinguished, " \
            "edited, controversiality " \
            "FROM May2015 ORDER BY Random() LIMIT 50"

data = pd.read_sql(sql_cmd, sql_conn)

columns = data.drop(['score'], axis=1).columns
for column in columns:
    uniqueVals = data[column].unique()
    if not isinstance(uniqueVals[0], int):
        mapper = dict(zip(uniqueVals, range(len(uniqueVals))))
        data[column] = data[column].map(mapper).astype(int)

print("Train a Gradient Boosting model")
clf = GradientBoostingClassifier(n_estimators=50, learning_rate=0.005, subsample=0.7,
                                      min_samples_leaf=10, max_depth=7, random_state=11)
# print("Train a Random Forest model")
# clf = RandomForestClassifier(n_estimators=25)

clf.fit(data[columns], data['score'])

# sort importances
indices = np.argsort(clf.feature_importances_)
# plot as bar chart
plt.barh(np.arange(len(columns)), clf.feature_importances_[indices])
plt.yticks(np.arange(len(columns)) + 0.25, np.array(columns)[indices])
_ = plt.xlabel('Relative importance')
plt.savefig("Liberty_Feature_Importance.png")