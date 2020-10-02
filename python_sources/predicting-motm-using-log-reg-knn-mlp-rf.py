#!/usr/bin/env python
# coding: utf-8

# ## This notebook reads in data from the FIFA World Cup 2018, cleans the data, generates a new feature and tests logistic regression, KNN, MLP and random forests to classify the man of the match

# In[ ]:


import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.neural_network import MLPClassifier as mlp
from sklearn.model_selection import GridSearchCV as gs
from sklearn.ensemble import RandomForestClassifier as rf
from sklearn.externals import joblib

import seaborn as sns
import matplotlib.pyplot as plt
#%matplotlib inline


# In[ ]:


def model_output(grid_model):
    """Outputs the best mean cross validation score, the test set accuracy and the parameters of a gridsearch"""
    print('Best mean CV accuracy: ', grid_model.best_score_)
    print('Holdout test set accuracy: ', accuracy_score(grid_model.best_estimator_.predict(xtest), ytest))
    print('Best parameters: ', grid_model.best_params_)


# ### Read in data about the FIFA World Cup 2018

# In[ ]:


fifa = pd.read_csv('../input/FIFA 2018 Statistics.csv')
fifa.head()


# ### Each two rows represent a single game: one row for each team

# ### How many missing values are there, and where?

# In[ ]:


heat_map = sns.heatmap(fifa.isnull(), yticklabels = False,
            cbar = False,
            cmap = 'viridis')


# ### Drop columns where we can't impute missing values ('Own goal Time', '1st Goal') or that intuitively don't have predictive power (such as 'Date')

# In[ ]:


fifa = fifa.drop(['1st Goal', 'Own goal Time', 'Date', 'Team', 'Opponent', 'Round'], axis = 1)


# ### Clean up missing values in own goals (0 will replace NaN since that means no own goals were scored) and dummy encode our two categorical variables

# In[ ]:


fifa['Own goals'] = fifa['Own goals'].fillna(0).astype(int)
fifa['PSO'] = pd.get_dummies(fifa.PSO).Yes
fifa['Man of the Match'] = pd.get_dummies(fifa['Man of the Match']).Yes


# ### Create a column to indicate who won the match. By groups of two rows, decide who got the maximum score within a game and assign 1 for winners and 0 for losers

# In[ ]:


game_group = [n for n in range(len(fifa)//2)]
game_group = np.repeat(game_group, 2)

fifa['winner'] = fifa.groupby(game_group)['Goal Scored'].transform(lambda x: x == max(x))
fifa['winner'] = fifa['winner'].map({True: 1, False: 0})


# ### Separate the outcome variable from the predictors

# In[ ]:


fifa_x = fifa.drop('Man of the Match', axis = 1)
fifa_y = fifa['Man of the Match']


# ### What kind of relationships exist within the data? Looking at distributions for each continuous variable, separated by winners and losers of Man of the Match

# In[ ]:


f = plt.figure(figsize = (20, 15))

## Add a density plot for each of the continuous predictors
for i in range(0, 15):
    f.add_subplot(4, 4, i + 1)
    fifa.iloc[:, i].groupby(fifa_y).plot(kind = 'kde', title = fifa.columns[i])


# ### The distribution 'Goal Scored' and 'Ball Possession %' differ based on which team won Man of the Match. We now scale our data to feed it into classification models, expecting to see these variable relationships come into play

# In[ ]:


scaler = StandardScaler()
fifa_x = scaler.fit_transform(fifa_x)
fifa_x = pd.DataFrame(fifa_x)


# ### Split the data into 1/3 test and 2/3 train

# In[ ]:


xtrain, xtest, ytrain, ytest = train_test_split(fifa_x, 
                                                fifa_y, 
                                                random_state = 42, 
                                                test_size = .33,
                                                stratify = fifa_y)


# ### First try logistic regression and evaluate its performance on the test set

# In[ ]:


log_reg = LogisticRegression()
log_reg.fit(xtrain, ytrain)


# In[ ]:


preds = log_reg.predict(xtest)
print(confusion_matrix(preds, ytest))
print('Test accuracy: ', accuracy_score(preds, ytest))


# ### Next try KNN using values of K from 1 to the number of training rows

# In[ ]:


errors_knn = pd.DataFrame(columns = ['k_value', 'train_acc', 'test_acc'])

for n in range(1, len(xtrain)):
    knn_clf = knn(n_neighbors = n)
    knn_clf.fit(X = xtrain, y = ytrain)
    
    preds = knn_clf.predict(xtrain)
    errors_knn.loc[n, 'train_acc'] = accuracy_score(preds, ytrain)
    
    preds = knn_clf.predict(xtest)
    errors_knn.loc[n, 'test_acc'] = accuracy_score(preds, ytest)


# In[ ]:


errors_knn.plot(title = 'K Value Selection')
plt.xlabel('K')
plt.ylabel('Accuracy')
plt.show()


# In[ ]:


preds = knn_clf.predict(xtest)
print(confusion_matrix(preds, ytest))
print('Test accuracy: ', errors_knn.test_acc.max())


# ### With the optimal value of K, KNN outperforms logistic regression on the test set. We next test a simple neural network, or multilayer perceptron, using gridsearch for its parameters

# In[ ]:


mlp_clf = mlp()

grid = {'hidden_layer_sizes': [(10, 10), (20, 10), (30, 10),
                               (10, 20), (20, 20), (30, 20),
                               (10, 30), (20, 30), (30, 30)], 
        'max_iter': [1000, 2000], 
        'learning_rate_init': [1e-10, 1e-5, 1e-3, 1e-2, 1e-1],
        'random_state': [420]}
grid_mlp = gs(mlp_clf, grid, cv = 10)

grid_mlp.fit(xtrain, ytrain)


# In[ ]:


model_output(grid_mlp)


# ### We use cross validation in our gridsearch and are able to use the test set properly. Comparing to KNN's test accuracy, the optimal MLP's cross-validated accuracy is slightly lower

# ### Lastly, we test a random forest with different numbers of trees, depths and splitting criteria using gridsearch

# In[ ]:


rand_for = rf()

grid = {'n_estimators': [5, 10, 15, 20, 30, 50, 100, 200, 500],
        'max_depth' : [None, 2, 3, 5, 10, 20],
        'criterion': ['gini', 'entropy'],
        'random_state' : [69]}

grid_rand = gs(rand_for, grid, cv = 10)

grid_rand.fit(xtrain, ytrain)


# In[ ]:


model_output(grid_rand)


# ### The random forest performed the best (looking at CV error)! Let's see which variables played a role in our best predictor

# In[ ]:


feat_imp_rf = pd.DataFrame({'Feature' : fifa.drop('Man of the Match', axis = 1).columns,
                            'Importance' : grid_rand.best_estimator_.feature_importances_})

feat_imp_rf.set_index('Feature', inplace = True)
feat_imp_rf.sort_values('Importance', inplace = True)

feat_imp_rf.plot(kind = 'barh', legend = None, title = 'Feature Importance')
plt.show()


# ### Save the model

# In[ ]:


joblib.dump(grid_rand, 'fifa_rf.pkl')


# ### Random forests had the best performance with a cross validation score of 89.4%. It allowed us to understand that the winner of the match is the most important predictor of the Man of the Match, followed by the number of goals scored. 
# 
# ### It should be noted that the random forest didn't generalize well to the final test set, scoring only 76%. Other features may need to be mined from the data to correct for this. 
# 
# ### Further work could include training additional model types, generating new features or properly implementing CV in the logisitic and KNN classifiers.
