#!/usr/bin/env python
# coding: utf-8

# The purpose of this contest is to maximize the likelihood to bid on the winner, in tennis tourneys. Let's get data and see what we can do toward this goal..
# 
# # STEP1: Explore
# Load data and print general overview:

# In[1]:


import kaggle
import os
import pandas as pd, numpy as np
from datetime import datetime
import time

df = pd.read_csv('../input/ATP.csv', low_memory=False)


# In[2]:


# what does it look like?
print(df.shape)
df.head()


# Seems we are dealing with a fair quantity of data, with a significant number of variables. But what about the quality? let's get a little closer to the matrix to get the number of NaN occurrences, types of variables..

# In[3]:


df.info()


# The prupose is to maximize likelihood to bet on the winner. For a better understanding and modeling of the "winning pattern", I would make a deep transformation to the data: an observation, for me, should be a match configuration with Plyer1's variables (whether winner or loser), Player2's variables, along with the variables that are not dependent on either players (eg, surface). The predicted target variable will refer to the winner (Player1 or Player2). Then, a classifier can be easily learnt from these.
# For convenience, the "winner" will be denoted P1 (target=0), and the "loser" P2 (target=1).

# In[4]:


# these variables do not seem relevant to me. might be assessed in a further work
df = df.drop(columns=['tourney_id','tourney_name','tourney_date','match_num','winner_entry','loser_entry','winner_id','winner_name','score','loser_id','loser_name'])

# convert numeric varibales to the correct type (csv_read fct does not make auto convert)
col_names_to_convert = ['winner_seed','draw_size','winner_ht','winner_age','winner_rank','winner_rank_points',
                       'loser_seed','loser_ht','loser_age','loser_rank','loser_rank_points','best_of','minutes',
                       'w_ace','w_df','w_svpt','w_1stIn','w_1stWon','w_2ndWon','w_SvGms','w_bpSaved','w_bpFaced',
                       'l_ace','l_df','l_svpt','l_1stIn','l_1stWon','l_2ndWon','l_SvGms','l_bpSaved','l_bpFaced'
                       ]
for col_name in col_names_to_convert:
    df[col_name] = pd.to_numeric(df[col_name], errors='coerce')


# Here is what data look like after transformation

# In[5]:


df.describe().transpose()


# In[6]:


# append a new target variable with the code assigned to winner player (0 when P1 | 1 when P2)
# For this set of data, the winner is always P1, so append 0s to the target variable
df['target'] = np.zeros(df.shape[0], dtype = int)


# In[7]:


# Now we'll generate the second batch of data, ie, by switching P1 and P2. The winner this time will be P2, and the target variable =1
# generate data by switching among P1 and P2 (target will be P2)
df2 = df.copy()
# switch between variables from P1 and those from P2
df2[['winner_seed','winner_hand','winner_ht','winner_ioc','winner_age','winner_rank','winner_rank_points']] = df[['loser_seed','loser_hand','loser_ht','loser_ioc','loser_age','loser_rank','loser_rank_points']]
df2[['loser_seed','loser_hand','loser_ht','loser_ioc','loser_age','loser_rank','loser_rank_points']] = df[['winner_seed','winner_hand','winner_ht','winner_ioc','winner_age','winner_rank','winner_rank_points']]
df2[['w_ace','w_df','w_svpt','w_1stIn','w_1stWon','w_2ndWon','w_SvGms','w_bpSaved','w_bpFaced']] = df[['l_ace','l_df','l_svpt','l_1stIn','l_1stWon','l_2ndWon','l_SvGms','l_bpSaved','l_bpFaced']]
df2[['l_ace','l_df','l_svpt','l_1stIn','l_1stWon','l_2ndWon','l_SvGms','l_bpSaved','l_bpFaced']] = df[['w_ace','w_df','w_svpt','w_1stIn','w_1stWon','w_2ndWon','w_SvGms','w_bpSaved','w_bpFaced']]
df2['target'] = np.ones(df2.shape[0], dtype = int)

df = df.append(df2)


# Let's see what we got

# In[8]:


df.head(2).append(df.tail(2))


# To meet algorithm's expectation, we need to encode the categorical variables, like this:

# In[9]:


from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
df['surface'] = lb.fit_transform(df['surface'].astype(str))
df['tourney_level'] = lb.fit_transform(df['tourney_level'].astype(str))
df['winner_hand'] = lb.fit_transform(df['winner_hand'].astype(str))
df['loser_hand'] = lb.fit_transform(df['loser_hand'].astype(str))
df['round'] = lb.fit_transform(df['round'].astype(str))
df['winner_ioc'] = lb.fit_transform(df['winner_ioc'].astype(str))
df['loser_ioc'] = lb.fit_transform(df['loser_ioc'].astype(str))


# Last, we should not forget about the NaN. As a first approach, let's just fill them with the median value of each column

# In[10]:


# replace nan with 0 and infinity with large values
df = df.fillna(df.median())


# # STEP2: Predict
# 

# Start by splitting train/test subsets

# In[11]:


# subsample for test purpose : TODO: REMOVE FOR FINAL RUN
df = df.sample(100000)

# split train/test subsets (80% train, 20% test)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, :-1], df.target, test_size=.2, random_state=0)


# In[12]:


# import classifiers from sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import PassiveAggressiveClassifier

# set names and prepare the benchmark list
names = ["K Near. Neighb.", "Decision Tree", "Random Forest", "Naive Bayes", "Quad. Dis. Analys", "AdaBoost", 
         "Neural Net" #, "RBF SVM", "Linear SVM", "Ridge Classifier"
        ]

classifiers = [
    KNeighborsClassifier(10),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
    AdaBoostClassifier(),
    MLPClassifier(alpha=1, max_iter=1000)
    # too long run for the test
    #SVC(gamma=2, C=1),
    #SVC(kernel="linear", C=.025),
    #RidgeClassifier(tol=.01, solver="lsqr")
]


# We can now launch the learning step with the selected classifiers, then the test step on the heldout data. Accuracy score is returned for each

# In[ ]:


# init time 
tim = time.time()
print('Learn. model\t\t score\t\t\ttime')
scores = []

for name, clf in zip(names, classifiers):
        print(name, end='')
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        print('\t\t', round(score, 3), '%', '\t\t', round(time.time() - tim, 3))
        scores.append(score)
        tim = time.time()


# In[ ]:


# plot results
import matplotlib.pyplot as plt

plt.rcdefaults()

y_pos = np.arange(len(names))

plt.bar(y_pos, scores, align='center', alpha=0.5)
plt.xticks(y_pos, names, rotation='vertical')
plt.ylabel('Accuracy')
plt.title('Model comparison for ATP prediction')

plt.show()


# # STEP3: Go Further..
# Despite being learnt on a small subset of data, the tested models achived fairly good results and are way superior to the baseline classifier. For example, a simple perceptron achives 76% accuracy, which means, it only misses 1 out of 4 bids!
# 
# 
# However, this remains a humble first dive into the data. Number of things have been bypassed due to time circumstances. Here are some further investigations one could perform for better and hopefully more accurate predictors:
# - Investigate whether a bias exist ude to the large time period (1968--2017). A straightforward technique to lower the effect of the time bias consists of weighting observations wrt their time/date of occurrence (most recent ones are more important).
# - Include the discaded variables, and use variable scaler (to corect the differences in scale) along with more adapted techniques for impacting the missing values.
# - Evaluate and select the most relevant features (feature selection) and explore correlation among them.
# - Fine-tune algorithm's parameters, for example with the grid search interface available from scikit-learn.

# In[ ]:




