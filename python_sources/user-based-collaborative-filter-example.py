# %% [markdown]
# > **Written By Mehdi Ozel @Erik Tech Labs 2019**
# **This Script written to **

# %% [code]
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

# %% [markdown]
# **Import Dataset
# **
# * JokeText contain Joke Indexes and Jokes.
# * UserRatings contain Users and User ratings to indexed with jokes
# * Ratings are between -10 to 10
# * There are 100 jokes and 36711 users and ratings.But we will use 5000.

# %% [code]
import pandas as pd
jokes = pd.read_csv("../input/jester-collaborative-filtering-dataset/JokeText.csv")
data = pd.read_csv("../input/jester-collaborative-filtering-dataset/UserRatings1.csv",index_col="JokeId")
data.head()


# %% [markdown]
# **First, we take just 5000 user ratings.Then, we sum all columns for each row which means the sum of ratings of each joke.Then we divide sum of columns to columns size to find mean value of the Joke.Then we find max value of each summed rows to find best joke rated by Users.**

# %% [code]
data = data.iloc[:,:5000]
sums_of_columns = data.sum(axis=1)
columns_size = len(data.columns)
value = sums_of_columns/columns_size
index_of_max = value[value == value.max()].index[0]
print("The best joke is index as {} and value of joke is :{}".format(index_of_max,value.max()))

# %% [markdown]
# **We will use cosine similarity which we imported from sklearn.First, we take transpose data to have a better visual of matrix.Then we find cosine sim of the ratings.Then we find most similiar 10 users.**

# %% [code]
from sklearn.metrics.pairwise import cosine_similarity
data = data.T
Filtering_cosim = cosine_similarity(data,data)

#most_sim_user = sorted(list(enumerate(Filtering_cosim[100])))

most_sim_users = sorted(list(enumerate(Filtering_cosim[8])), key=lambda x: x[1], reverse=True)
most_sim_users = most_sim_users[1:11]
sim_users = [x[0] for x in most_sim_users]
print(sim_users)

# %% [markdown]
# **User based collabrative filtering function**

# %% [code]
candidates_jokes = data.iloc[sim_users,:]

def UBCF(user_num):
    ### finding most similar users among matrix

    most_sim_users = sorted(list(enumerate(Filtering_cosim[user_num])), key=lambda x: x[1], reverse=True)
    most_sim_users = most_sim_users[1:11]

    ### user index and their similairity values 

    sim_users = [x[0] for x in most_sim_users]
    sim_values = [x[1] for x in most_sim_users]

    ### among users having most similar preferences, finding movies having highest average score
    ### however except the movie that original user didn't see

    candidates_jokes = data.iloc[sim_users,:]

    candidates_jokes.mean(axis=0).head()

    mean_score = pd.Series(candidates_jokes.mean(axis=0))
    mean_score = mean_score.sort_values(axis=0, ascending=False)
    
    recom_jokes = list(mean_score.iloc[0:10].keys())
    for i in recom_jokes:
        print("Index Number {} and Joke is {} :".format(i,jokes.iloc[i,:]))
    return(recom_jokes)

# %% [markdown]
# **Write id of the user and this func will suggest you jokes.**

# %% [code]
UBCF(100)