#!/usr/bin/env python
# coding: utf-8

# Hi, this is my first notebook. I'm trying to help newbies like myself to get started with machine learning and simple classifications problems. 
# 
# Let's start by importing the dataset, and verify if there is any column with missing values. 

# In[ ]:


import pandas
import numpy as np

train_set = pandas.read_csv("../input/train.csv")
test_set = pandas.read_csv("../input/test.csv")
train_set = train_set.drop('id',axis=1)
print(train_set.describe())


# As you can see the count is the same for every numerical column in the dataset, and the range of values is between 0 and 1. 
# The dataset given has the label column called 'type' which has values who belong to a category. Our next step is to convert string categories in integer values.

# In[ ]:


train_set['type'], categories = train_set['type'].factorize()
print(train_set.describe())


# As you can see the 'type' column now appears as a numerical set of values. 
# 
# You can also see that a second variable called categories was used. This variable contains the information that will allow us to revert this factorization process. 
# 
# To gain a little more insights about the data we can make a simple plot to help us.

# In[ ]:


import matplotlib.pyplot as plt
fig = plt.figure(figsize=(20,10))
aux_plot = fig.add_subplot(111)
fig.colorbar(aux_plot.matshow(train_set.corr()))

aux_plot.set_xticklabels(train_set.columns)
aux_plot.set_yticklabels(train_set.columns)

plt.show()


# This image is really useful and will help us to analyze features and their impact on the 'type' column. 
# As you can see the 'type' column has a high value of negative correlation with columns 'color', 'has_soul' and 'rotting_flesh'. Although the correlation with the 'hair_length' is not very big. 
# 
# Analyzing the correlation between other features it's possible to see that 'has_soul' and 'hair_length' have an interesting value as well as the column 'has_soul' with 'rotting_flesh'. 
# Lets then try to extract some new features from here. We can, for example, multiply 'has_soul' times 'hair_length' to obtain a new feature much more correlated to the 'type' column.
# 
# Lets then create a class responsible for this process.

# In[ ]:


from sklearn.base import BaseEstimator, TransformerMixin

class CreateExtraFeatures(BaseEstimator,TransformerMixin):
    def __init__(self):pass

    def fit(self,X,y=None):
        return self
    def transform(self,X,y=None):
        X['hair_soul'] = X['hair_length'] * X['has_soul']
        X['flesh_soul'] = X['rotting_flesh'] * X['has_soul']
        return np.c_[X]


# In this class we are creating 2 new features, combining 3 existing ones. 
# 
# As we want to be able to easily append more transformers from Scikit-Learn it is a good approach to create a pipeline. In this case, 3 pipelines, one for numerical features, other for categorical features and a third to join both of this pipelines. 
# 
# For numerical features, it is enough to use only our custom transformer, since there aren't missing values and the data is already between 0 and 1. 
# 
# For categorical features we will use OneHotEncoder from Scikit-Learn, to factorize string values and to create a dense matrix with values assigned to 1 being Hot. 

# In[ ]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
pipeline_num = Pipeline([
    ("extra_feat",CreateExtraFeatures())
])

pipeline_cat = Pipeline([
    ("categorical_encoder", OneHotEncoder(sparse=False))
])

from sklearn.pipeline import FeatureUnion

full_pipeline = FeatureUnion([
    ("pip,num",pipeline_num),
    ("pip_cat",pipeline_cat)
])


# Before calling the fit function of the pipeline we should notice that we want this to apply to the test set too. So we are going to join them, knowing that the first 371 set of elements are from the training set.

# In[ ]:


X_train = train_set.drop('type',axis=1)
y_train = train_set.get('type')
X_train= X_train.append(test_set)

num_attributes = ["bone_length","rotting_flesh","hair_length","has_soul"]
cat_attributes = ["color"]
X_train= full_pipeline.fit_transform(X_train[num_attributes],X_train[cat_attributes].values)

X_test = X_train[371:]
X_train = X_train[:371]


# Finally, having the dataset prepared we can now start thinking about our neural network. 
# Scikit-Learn as an easy to use implementation of a neural network, so it will be our choice. 
# To define a most capable combination of Hyperparameters it is a good practice to use something like GridSearch with a cross-validation evaluation to ensure that the best model is chosen.
# 

# In[ ]:


from sklearn.neural_network import MLPClassifier

nn_clf = MLPClassifier(max_iter=3000)

from sklearn.model_selection import GridSearchCV

grid_params = [{"hidden_layer_sizes":range(3,20), "activation":['identity', 'logistic', 'tanh', 'relu'], "solver":["lbfgs","sgd","adam"],"learning_rate":["adaptive"]}]
grid_search = GridSearchCV(nn_clf,param_grid=grid_params,cv=3,verbose=0)


# We can then fit it to our dataset and find the best combinations of parameters and the best score. 
# Let's see!

# In[ ]:


grid_search.fit(X_train,y_train)

print(grid_search.best_estimator_)
print(grid_search.best_score_)


# As you can see the algorithm has around 74.9% of accuracy.
#  
# Last step is to create the submission file.

# In[ ]:


y_pred = grid_search.predict(X_test)

submissions = pandas.DataFrame(y_pred, index=test_set.id,columns=["type"])
submissions["type"] = categories[submissions["type"]]
submissions.to_csv('./submission.csv', index=True)

