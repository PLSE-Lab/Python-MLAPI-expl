#!/usr/bin/env python
# coding: utf-8

# ## Background
# 
# *Wik Hung Pun*
# 
# *6-9-2017*
# 
# Collaborative filtering (CF) is a topic that eluded me in my quest of studying machine learning in the past. This steam games dataset gave me a reason to explore and learn more about CF. After reading some excellent works written by [Ethan Rosenthal](http://blog.ethanrosenthal.com/), [Katherine Bailey](http://katbailey.github.io/post/matrix-factorization-with-tensorflow/), and [Jesse Steinweg-Woods](https://jessesw.com/Rec-System/), I have to say the concept behind collaborative filtering is fairly simple and easy to understand. Although I do not think I can explain the concept nearly as well as these authors, I still would like to share what I have learned with you as a primer for learning collaborative filtering (and reinforce my own learning). For those of you interested in the topic, please do check out the authors I have linked. Now, without further ado...

# # Introduction
# Collaborative filtering (CF) is a technique used for being recommender systems. The goal of CF is to infer the preferences for new items given the known preferences from all the users. [Rosenthal](http://blog.ethanrosenthal.com/2015/11/02/intro-to-collaborative-filtering/) has a great post explaining and implementing item- and user-based CF systems. I'd highly recommend you to read it if you are interested in recommender systems.
# 
# What I would like to focus on is a technique calls **matrix factorization**. [There are a lot of math-y stuff behind it](https://en.wikipedia.org/wiki/Matrix_decomposition), but, for the purpose of this recommender system, we can simply think of it as solving a matrix algebra question via [matrix multiplication](https://en.wikipedia.org/wiki/Matrix_multiplication).
# 
# In this case, if we visualize the transactions as a big matrix, where users are the rows and games are the columns. This big matrix can, in turn, be decomposed into two matrices with dimensions of *users x features* (U), and *features x games* (V). The Steinweg-Woods's image gives a clear depiction of this idea. ![id](https://jessesw.com/images/Rec_images/ALS_Image_Test.png) Once we have the U and V matrices estimated, we can then take the dot product of the two to find the predicted game preferences for each user.

# ## Explicit vs Implicit CF
# Now you have learned basic concept about CF, it is important to point out there are two types of CF: **explicit** and **implicit**. In explicit CF, the values we fill in the users by items matrix were preferences collected *explictly* from users (e.g., thumbs up/down, likes, user ratings, etc.). In contrast, we do not have these direct indicators of preferences from users with implicit CF. Instead, we only have *indirect* indicators such as whether they purchased the product or used the product. For instance, we only know a user bought a game in this dataset and the person might have even played it for a few hours, but we do not know if the user actually liked the game. For all we know, the user may hate the game after playing it! With implicit CF, we try to take account of these possible variations in the model.

# ### Miscellaneous
# Prior to actually getting to the code, it'd be remiss of me to not mention there are actually great packages available of conducting CF analysis. [Rosenthal](http://blog.ethanrosenthal.com/2016/10/19/implicit-mf-part-1/) discussed the strengthes and weaknesses of some of these packages. Since these packages are not available on Kaggle (as fas as I know), I implemented my own here, but you should look for others' implementation if you are working outside of the Kaggle environment.

# #### Code Stuff
# Import packages. Exciting stuff.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import random
from collections import Counter
from sklearn.metrics import roc_curve, auc, average_precision_score


# Taking a look at the dataset. I named the last column as not needed since it does not appear to be associated with anything. The thing to look out for in this dataset is that the purchases and plays are separated into two separate rows. For the purpose of the analysis I will have to convert these into one record. I convert the dataset with following rules:
# 1. If a game is purchased but never played, hours played = 0
# 2. If a game is purchased *and* played, I keep the hours played and remove the purchase record (Playing the game implies it was purchased).

# In[ ]:


path = '../input/steam-200k.csv'
#path = 'steam-200k.csv'
df = pd.read_csv(path, header = None,
                 names = ['UserID', 'Game', 'Action', 'Hours', 'Not Needed'])
df.head()


# In[ ]:


# Creating a new variable 'Hours Played' and code it as previously described.
df['Hours_Played'] = df['Hours'].astype('float32')

df.loc[(df['Action'] == 'purchase') & (df['Hours'] == 1.0), 'Hours_Played'] = 0


# In[ ]:


# Sort the df by User ID, games, and hours played
# Drop the duplicated records, and unnecessary columns
df.UserID = df.UserID.astype('int')
df = df.sort_values(['UserID', 'Game', 'Hours_Played'])

clean_df = df.drop_duplicates(['UserID', 'Game'], keep = 'last').drop(['Action', 'Hours', 'Not Needed'], axis = 1)

# every transaction is represented by only one record now
clean_df.head()


# In[ ]:


n_users = len(clean_df.UserID.unique())
n_games = len(clean_df.Game.unique())

print('There are {0} users and {1} games in the data'.format(n_users, n_games))


# In[ ]:


# If we build a matrix of users x games, how many cells in the matrix will be filled?
sparsity = clean_df.shape[0] / float(n_users * n_games)
print('{:.2%} of the user-item matrix is filled'.format(sparsity))


# In[ ]:


# Here 
user_counter = Counter()
for user in clean_df.UserID.tolist():
    user_counter[user] +=1

game_counter = Counter()
for game in clean_df.Game.tolist():
    game_counter[game] += 1


# In[ ]:


# Create the dictionaries to convert user and games to idx and back
user2idx = {user: i for i, user in enumerate(clean_df.UserID.unique())}
idx2user = {i: user for user, i in user2idx.items()}

game2idx = {game: i for i, game in enumerate(clean_df.Game.unique())}
idx2game = {i: game for game, i in game2idx.items()}


# In[ ]:


# Convert the user and games to idx
user_idx = clean_df['UserID'].apply(lambda x: user2idx[x]).values
game_idx = clean_df['gameIdx'] = clean_df['Game'].apply(lambda x: game2idx[x]).values
hours = clean_df['Hours_Played'].values


# ## Converting the data to a user x game matrix
# Since we do not have explicit indicators in this dataset, we fill the users by games matrix with simple preferred (1) or not (0). This (preference) matrix  indicates that the user purchased and/or played the game with a 1, whereas 0 means no interaction between the user and the game. 
# 
# Now, one obvious quesiton is what if the users dislike their purchases? If all purchases are represented by ones, how do we find out the ones that users actually regret buying? Well, we handle this situation by constructing a **confidence matrix**. This confidence matrix has the same dimension as the preference matrix and is populated with __*hours played*__. Intuitively, it means that the more time a user spent playing the game, we have more *confidence* in that the user actually *liked/preferred* the game. 

# In[ ]:


# Using a sparse matrix will be more memory efficient and necessary for larger dataset, 
# but this works for now.

zero_matrix = np.zeros(shape = (n_users, n_games)) # Create a zero matrix
user_game_pref = zero_matrix.copy()
user_game_pref[user_idx, game_idx] = 1 # Fill the matrix will preferences (bought)

user_game_interactions = zero_matrix.copy()
# Fill the confidence with (hours played)
# Added 1 to the hours played so that we have min. confidence for games bought but not played.
user_game_interactions[user_idx, game_idx] = hours + 1 


# ## Validation
# To examine the effectiveness of the recommender system, I used top-k precision as my evaluation metric (k = 5, in this case). In order to implement this evaluation metric, I need to first identify users who have bought more than k games and then mask k preferences from the training set. This will bias the validation process towards users with higher number of purchases. However, it makes the problem easier to handle and cold start problem of recommender system is another topic that requires a lot more in depth analysis on its own.

# In[ ]:


k = 5

# Count the number of purchases for each user
purchase_counts = np.apply_along_axis(np.bincount, 1, user_game_pref.astype(int))
buyers_idx = np.where(purchase_counts[:, 1] >= 2 * k)[0] #find the users who purchase 2 * k games
print('{0} users bought {1} or more games'.format(len(buyers_idx), 2 * k))


# In[ ]:


test_frac = 0.2 # Let's save 10% of the data for validation and 10% for testing.
test_users_idx = np.random.choice(buyers_idx,
                                  size = int(np.ceil(len(buyers_idx) * test_frac)),
                                  replace = False)


# In[ ]:


val_users_idx = test_users_idx[:int(len(test_users_idx) / 2)]
test_users_idx = test_users_idx[int(len(test_users_idx) / 2):]


# In[ ]:


# A function used to mask the preferences data from training matrix
def data_process(dat, train, test, user_idx, k):
    for user in user_idx:
        purchases = np.where(dat[user, :] == 1)[0]
        mask = np.random.choice(purchases, size = k, replace = False)
        
        train[user, mask] = 0
        test[user, mask] = dat[user, mask]
    return train, test


# In[ ]:


train_matrix = user_game_pref.copy()
test_matrix = zero_matrix.copy()
val_matrix = zero_matrix.copy()

# Mask the train matrix and create the validation and test matrices
train_matrix, val_matrix = data_process(user_game_pref, train_matrix, val_matrix, val_users_idx, k)
train_matrix, test_matrix = data_process(user_game_pref, train_matrix, test_matrix, test_users_idx, k)


# In[ ]:


# let's take a look at what was actually accomplised
# You can see the test matrix preferences are masked in the train matrix
test_matrix[test_users_idx[0], test_matrix[test_users_idx[0], :].nonzero()[0]]


# In[ ]:


train_matrix[test_users_idx[0], test_matrix[test_users_idx[0], :].nonzero()[0]]


# ## Tensorflow Implementation
# I implemented the implicit CF in tensorflow because I want to get more familiar with it. You can do this with only scipy and numpy. In tensorflow, you have to define the computation graph first and then actually carry out the calculations.

# In[ ]:


tf.reset_default_graph() # Create a new graphs

pref = tf.placeholder(tf.float32, (n_users, n_games))  # Here's the preference matrix
interactions = tf.placeholder(tf.float32, (n_users, n_games)) # Here's the hours played matrix
users_idx = tf.placeholder(tf.int32, (None))


# Instead of directly multiplying the hours played matrix with the preference matrix, we want to add a confidence parameter. We can think of it as how much weight we should give to these interactions. The [original paper](http://yifanhu.net/PUB/cf.pdf) recommends setting the parameter to 40, but I found the result to be less than ideal. Instead, I sampled it from a uniform distribution and use gradient descent to find the optimal value.

# In[ ]:


n_features = 30 # Number of latent features to be extracted

# The X matrix represents the user latent preferences with a shape of user x latent features
X = tf.Variable(tf.truncated_normal([n_users, n_features], mean = 0, stddev = 0.05))

# The Y matrix represents the game latent features with a shape of game x latent features
Y = tf.Variable(tf.truncated_normal([n_games, n_features], mean = 0, stddev = 0.05))

# Here's the initilization of the confidence parameter
conf_alpha = tf.Variable(tf.random_uniform([1], 0, 1))


# ## User and Item Bias
# Two other model parameters that I did not mention above are user and item biases. Intuitively, we would expect some users play games at a faster pace than the others. Similarly, we also expect some games would take less time to complete. User and item biases allow us to express these intuitions in a statistical model where we expect there are systematic differences in how users interact with the games and how games are played.
# 
# One thing to note is that both biases are not necessary to build the recommender system, however, I found including these parameters improved the recommender system.

# In[ ]:


# Initialize a user bias vector
user_bias = tf.Variable(tf.truncated_normal([n_users, 1], stddev = 0.2))

# Concatenate the vector to the user matrix
# Due to how matrix algebra works, we also need to add a column of ones to make sure
# the resulting calculation will take into account the item biases.
X_plus_bias = tf.concat([X, 
                         #tf.convert_to_tensor(user_bias, dtype = tf.float32),
                         user_bias,
                         tf.ones((n_users, 1), dtype = tf.float32)], axis = 1)


# In[ ]:


# Initialize the item bias vector
item_bias = tf.Variable(tf.truncated_normal([n_games, 1], stddev = 0.2))

# Cocatenate the vector to the game matrix
# Also, adds a column one for the same reason stated above.
Y_plus_bias = tf.concat([Y, 
                         tf.ones((n_games, 1), dtype = tf.float32),
                         item_bias],
                         axis = 1)


# In[ ]:


# Here, we finally multiply the matrices together to estimate the predicted preferences
pred_pref = tf.matmul(X_plus_bias, Y_plus_bias, transpose_b=True)

# Construct the confidence matrix with the hours played and alpha paramter
conf = 1 + conf_alpha * interactions


# The cost of the model would be the squared sum of predicted preferences and actual preferences. This cost is modified by the conference matrix. Finally, l2-regularization is also added to avoid overfitting the training set.

# In[ ]:


cost = tf.reduce_sum(tf.multiply(conf, tf.square(tf.subtract(pref, pred_pref))))
l2_sqr = tf.nn.l2_loss(X) + tf.nn.l2_loss(Y) + tf.nn.l2_loss(user_bias) + tf.nn.l2_loss(item_bias)
lambda_c = 0.01
loss = cost + lambda_c * l2_sqr


# In[ ]:


lr = 0.05
optimize = tf.train.AdagradOptimizer(learning_rate = lr).minimize(loss)


# In[ ]:


# This is a function that helps to calculate the top k precision 
def top_k_precision(pred, mat, k, user_idx):
    precisions = []
    
    for user in user_idx:
        rec = np.argsort(-pred[user, :]) # Found the top recommendation from the predictions
        
        top_k = rec[:k]
        labels = mat[user, :].nonzero()[0]
        
        precision = len(set(top_k) & set(labels)) / float(k) # Calculate the precisions from actual labels
        precisions.append(precision)
    return np.mean(precisions) 


# Here's the actual training. I calculate the validation precision after every 10 iterations. With higher number of iteration seems to overfit so I settled with 80.

# In[ ]:


iterations = 80
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for i in range(iterations):
        sess.run(optimize, feed_dict = {pref: train_matrix,
                                        interactions: user_game_interactions})
        
        if i % 10 == 0:
            mod_loss = sess.run(loss, feed_dict = {pref: train_matrix,
                                                   interactions: user_game_interactions})            
            mod_pred = pred_pref.eval()
            train_precision = top_k_precision(mod_pred, train_matrix, k, val_users_idx)
            val_precision = top_k_precision(mod_pred, val_matrix, k, val_users_idx)
            print('Iterations {0}...'.format(i),
                  'Training Loss {:.2f}...'.format(mod_loss),
                  'Train Precision {:.3f}...'.format(train_precision),
                  'Val Precision {:.3f}'.format(val_precision)
                )

    rec = pred_pref.eval()
    test_precision = top_k_precision(rec, test_matrix, k, test_users_idx)
    print('\n')
    print('Test Precision{:.3f}'.format(test_precision))


# The test top-k precision is not that high but I did not spend a lot of time optimizing the hyperparameters so there is probably a lot more room for improvement. Something for you to try if you are interested.

# ## Examples
# Below I print out a few examples of recommendations accompanied with the actual purchases of the users. Some recommendations made more sense than the others but overall precision is fairly low.

# In[ ]:


n_examples = 5
users = np.random.choice(test_users_idx, size = n_examples, replace = False)
rec_games = np.argsort(-rec)


# In[ ]:


for user in users:
    print('Recommended Games for {0} are ...'.format(idx2user[user]))
    purchase_history = np.where(train_matrix[user, :] != 0)[0]
    recommendations = rec_games[user, :]

    
    new_recommendations = recommendations[~np.in1d(recommendations, purchase_history)][:k]
    
    print('We recommend these games')
    print(', '.join([idx2game[game] for game in new_recommendations]))
    print('\n')
    print('The games that the user actually purchased are ...')
    print(', '.join([idx2game[game] for game in np.where(test_matrix[user, :] != 0)[0]]))
    print('\n')
    print('Precision of {0}'.format(len(set(new_recommendations) & set(np.where(test_matrix[user, :] != 0)[0])) / float(k)))
    print('--------------------------------------')
    print('\n')


# ## Final Words
# Beside optimizing the hyperparameters, I did not spend a lot of time cleaning the data. Some of the games are more like DLCs and variations of the game. Consolidating some of these games may yield better results. 
# 
# Like I said, there are actually packages out there that can build the implicit CF for you more efficiently, but I find it to be informative to implement the algorithm on my own and I would recommend anyone who is learning the topic to do so as well.

# In[ ]:




