# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

#DIS-FUCKING-CLAIMER: I have no idea what I am doing seeing as my first two years of college so far have all been learning about my white privilege,
#and other pointless GE's. My knowledge revolves around about ten DataCamp tutorials.#



# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory


#first, let's make sure the data-set is in the format of a data frame, using the pandas read_csv method#

my_csv = pd.read_csv("../input/Netflix Shows.csv")

#view as data frame#

netflix_df  = pd.DataFrame(my_csv)

#first entries and last entries#

netflix_df.head()
netflix_df.tail()

#the columns look alright; so let's not combine any of those.#

#let's fill in all missing ratings with a filler of 'PG' and also drop all NA ratings#

netflix_df.rating = netflix_df.rating.dropna('TV-NA')
netflix_df.rating = netflix_df.rating.fillna('TV-PG')

#for all user ratings not filled in, let's get a random number between 60 and 90.#

rand_rating = np.random.randint(60, 90)
netflix_df.user_rating_score.fillna(rand_rating)

#the rating-level column is not really helpful to make any predictions, as most already know what PG, etc. means. drop that column, perhaps#

netflix_df = netflix_df.drop(netflix_df['rating level'])

#perhaps certain years have been more popular to watch or produced better shows than others. let's graph about 10 boxplots to find out.#


sns.boxplot(x = 'releaseyear', y = 'user rating score', data = netflix_df, subplots = True)
plt.show()


#we want to know whether a show's rating level(i.e., PG-13) seems to affect overall user ratings. This time, let's find the mean user rating for
#each show, and also create another seaborn plot. This is a univariate distriubtion. We can also use a violinplot, which is a type used to show this.#

sns.violinplot(x = 'rating level', y = 'user rating score', data = netflix_df, subplots = True)
plt.show()

#subset the data by maturity/ranking level.#

netflix_g_mov = netflix_df[df['rating']== "G"]
netflix_pg_mov = netflix_df[df['rating'] == "PG"]
netflix_pg13_mov = netflix_df[df['rating']== "PG-13"]
netflix_R_mov = netflix_df[df['rating']=="R"]

g_user_rating = np.mean(netflix_g_mov['user rating level'])
pg_user_rating = np.mean(netflix_pg_mov['user rating level'])
pg_13_user_rating = np.mean(netflix_pg13_mov['user rating level'])
r_user_rating = np.mean(netflix_R_mov['user ratng level'])

print('the average user rating based on ranking is': g_user_rating, pg_user_rating, pg_13_user_rating, r_user_rating)


#we're also interested in knowing how many of our shows of the original dataset are of each rating.#
#use numPy#

rankings = [netflix_g_mov, netflix_pg_mov, netflix_pg13_mov, netflix_R_mov]

for r in rankings:
    
    print(np.sum(r))







#What show has the highest rating, and which one has the lowest out of our dataset? there may be several with the lowest/highest.#

max_rating = netflix_df['user_rating_score'].max()
highest_show  = df.title[max_rating]
min_rating = netflix_df['user_rating_score'].min()
lowest_show = df.title[min_rating]

print('Our lowest and highest rated show(s) are:'.format(highest_show, lowest_show))





#small ML sample?#
#The producer behind a new Netflix Original Show wants to know whether this show is a good idea to put out, and whether its user ratings will be high,
#based on TV shows' rating level and rating description of the past. We can't know whether this show will have good ratings or not now, but we can make
#a prediction on it based off a supervised machine learning model.#


#let's choose a simple model to begin with. since ratings is not a categorical variable and we are trying to predict a continuous value, 
#this is a regression problem, not a classification one. let's begin with a simple linear regression.#

from sklearn.linear_model import LinearRegression()
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

array  = netflix_df.values
X  = array[]
 y = array[]


#unsupervised learning#

#an unsupervised learning model regarding our dataset means we are trying to sort the data into any underlying groups, or 'clusters' - AKA
#non-labeled data. perhaps our netflix shows have several underlying ways they 