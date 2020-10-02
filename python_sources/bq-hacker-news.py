# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

#import numpy as np # linear algebra
#import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#import bq_helper as bq # Biq Query

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#hacker_news = bq.BigQueryHelper(active_project= "bigquery-public-data",
#                dataset_name = "hacker_news")

#hacker_news.list_tables()
#hacker_news.table_schema("full")
#hacker_news.head("full")
#hacker_news.head("full", selected_columns="by", num_rows=10)

#query = """SELECT score
#            FROM `bigquery-public-data.hacker_news.full`
#            WHERE type = "job" """

# check how big this query will be
#print(hacker_news.estimate_query_size(query))

import numpy as np
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
Y = np.array([1, 1, 1, 2, 2, 2])
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(X, Y)
#GaussianNB(priors=None)
print(clf.predict([[0.8, 1]]))
#[1]
#clf_pf = GaussianNB()
#clf_pf.partial_fit(X, Y, np.unique(Y))
#GaussianNB(priors=None)
#print(clf_pf.predict([[-0.8, -1]]))

#from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.