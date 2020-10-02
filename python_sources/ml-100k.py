# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
import pandas as pd
pd.set_option('display.max_columns',100)
import matplotlib.pyplot as plt
import seaborn as sns
# Any results you write to the current directory are saved as output.

u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('../input/u.user', sep='|', names=u_cols,encoding='latin-1')
r_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_csv('../input/u.data', sep='\t', names=r_cols)
m_cols = ['movie_id','movie_title','release_date','video_release_date','IMDb_URL','unknown',\
          'Action','Adventure','Animation','Children','Comedy','Crime','Documentary','Drama',\
          'Fantasy','Film_Noir','Horror','Musical','Mystery','Romance','Sci_Fi','Thriller','War','Western']
movies = pd.read_csv('../input/u.item', sep='|', names=m_cols,
                     encoding='latin-1')
movie_ratings = pd.merge(movies, ratings)
df_all = pd.merge(movie_ratings, users)

# The 30 most rated movies
most_rated = df_all.groupby(by = 'movie_title').size().sort_values(ascending=False)[:30]
#lens.movie_title.value_counts()[:30]

movie_stats = df_all.groupby('movie_title').agg({'rating': [np.size, np.mean]}).sort_values([('rating', 'mean')], ascending=False)

rate_atleast_200 = movie_stats['rating']['size'] >= 200
movie_stats[rate_atleast_200].sort_values([('rating', 'mean')],ascending=False)

most_50 = df_all.groupby('movie_title').size().sort_values(ascending=False)[:50]

plt.figure()
sns.distplot(users['age'], hist=True,bins = 30)
plt.title("Distribution of users' ages")
plt.ylabel('count of users')
plt.xlabel('age')

plt.hist(users['age'],bins=30)
plt.title("Distribution of users' ages")
plt.ylabel('count of users')
plt.xlabel('age')

labels = ['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79']
df_all['age_group'] = pd.cut(df_all.age, range(0, 81, 10), right=False, labels=labels)
df_all.groupby('age_group').agg({'rating': [np.size, np.mean]})
df_all.set_index('movie_title', inplace=True)

by_age = df_all.loc[most_50.index].groupby(['movie_title', 'age_group'])
by_age = by_age.rating.mean().unstack(1).fillna(0)

df_all.reset_index(inplace=True)
df_all.set_index('movie_title', inplace=True)

pivoted = df_all.pivot_table(index=['movie_title'],
                           columns=['sex'],
                           values='rating',
                           fill_value=0)
