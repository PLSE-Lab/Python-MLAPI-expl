#!/usr/bin/env python
# coding: utf-8

# # Hot to track you Data Science experiments with neptune.ml
# ![](https://neptune.ml/wp-content/uploads/2018/08/Company-Header-Neptune.ml_-2-e1560327936998.png)
# 
# *Important disclaimer: I am not an owner nor a developer of the presented service. I only use it for my own projects (including Kaggle competitions) because I've found it very useful.*
# 
# * Have you ever faced a situation when some of your experiments have finished, you got the results, but you completely forgot what exactly you have changed?
# * Maybe you want to run several of experiments to check multiple ideas and see the results in a convenient form?
# * Or you want to collaborate with your teammates more effectively?
# 
# If your answer is 'yes' to any of those questions then you will like a [neptune](https://neptune.ml/) project.
# 
# In this kernel I will show you how to set it up and use in your kaggle competitions.

# 1. [Registering and installing neptune client](#1)
# 2. [Setting up a project](#2)
# 3. [Running an experiment](#3)
# 4. [Track parameters](#4)
# 5. [Track images](#5)
# 6. [Track artifacts](#6)
# 7. [Monitoring resources](#7)
# 8. [Conclusion](#8)

# <a id="1"></a>
# ## Registering and installing neptune client
# 
# I think there is no need to describe how to register in neptune, just go to the project page and click 'Sign up'. Google, Facebook and Github SSO works as well.
# 
# Next step is to install a neptune client. In order to do this enable an Internet access in your kernel's settings.
# ![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F1696976%2Ffc475a6556225863484d04e9ed1baecb%2Fneptune_kernel_1.png?generation=1564558569759774&alt=media)
# 
# Then run *pip install neptune-client*

# In[ ]:


get_ipython().system('pip install neptune-client')


# <a id="2"></a>
# ## Setting up a project
# Neptune has a good documentation let alone an interface is intuitive and easy to use. But I still will guide you with your first experiment.
# 
# First of all we need to create a project for our experiments. Let's do it.
# 
# Go to 'Projects' in the upper-left corner and then click a 'New project' button. You will see a modal window with some fields. Let's fill them and proceed.
# 
# ![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F1696976%2Fbd37ce0d4cda1c6908f808941b59f358%2Fneptune_kernel_2.png?generation=1564558568867977&alt=media)

# Let's get familiar with an interface. 
# 
# ![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F1696976%2F7fbbbc2d193fcb0a30ea2438b5fc90c6%2Fneptune_kernel_3.png?generation=1564558568764085&alt=media)
# 
# We have 4 tabs at the top:
# * Wiki
# * Notebooks
# * Experiments
# * Settings
# 
# Wiki is a README and comments for you project
# 
# Notebooks contains all of the notebooks you are tracking. If you are using a Jupyter notebook (which I bet you are) then you can install a jupyter extension called *neptune-notebooks* and integrate it. After that by simply clicking on one button you will save a notebook checkpoint to your project and then you can keep working without a fear to remove some cell or rewrite a code in it. You will always have a backup. The outputs of the cells are being saved as well.
# 
# ![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F1696976%2Ff7d9882ddc9c732be20bcd438d9677d9%2Fneptune_kernel_4.png?generation=1564564644071941&alt=media)
# 
# Unfortunately kaggle kernels does not support such integration but you can upvote my [feature request](https://www.kaggle.com/product-feedback/101200#583902) for it.

# <a id="3"></a>
# ## Running an experiment
# 
# It is time to create and run our first experiment. To track you experiment first you need to initialize it using neptune.init() method with your token.
# 
# You can obtain a token by clicking your user icon in the upper-right corner and selecting 'Get API Token'
# 
# ![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F1696976%2F03662a1b9f79b4da78120f1f390313c8%2Fneptune_kernel_5.png?generation=1564558561773321&alt=media)
# 
# Next I will create a small dataset with a synthetic data using sklearn.datasets.make_classification and run our first experiment.

# In[ ]:


import neptune
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# Initializing a neptune. First argument is your-user-name/project-name, second argument is your API Token.
# It is strongly recommended to store it in an environment variable NEPTUNE_API_TOKEN
neptune.init('kaggle-presentation/kaggle', api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5tbCIsImFwaV9rZXkiOiIwNTM2NjM2OS1mY2YxLTQyNWQtODQyZi03NWQ5NDhhMWI3YWYifQ==')

# Creating a dataset
X, y = make_classification()
# Splitting it into training and testing
X_train, X_test, y_train, y_test = X[:70], X[70:], y[:70], y[70:]

# Creating experiment in the project defined above
neptune.create_experiment()

# Fitting a model
lr = LogisticRegression(solver='lbfgs')
lr.fit(X_train, y_train)
auc = roc_auc_score(y_test, lr.predict_proba(X_test)[:, 1])

# Sending a metric to the experiment
neptune.send_metric('AUC', auc)

# Stop the experiment
neptune.stop()


# As an output we have an experiment ID and a link to it. Let's see how the experiment results look in neptune interface. 
# 
# Here is a result on the dashboard:
# 
# ![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F1696976%2Fc3c9eb7e9965ee7317ca0fcd350c770e%2Fneptune_kernel_6.png?generation=1564559613195430&alt=media)
# 
# We called our metric 'AUC', but this is not a default column in neptune, so in order to see it we need to add it the dasboard.
# 
# ![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F1696976%2F9324a9242a9789c6d9fbbd92a9f3d06f%2Fneptune_kernel_7.png?generation=1564559774143612&alt=media)
# 

# <a id="4"></a>
# ## Track parameters
# 
# Another useful feature is an ability to save the parameters of the model for your experiment. Here is how you can do it.

# In[ ]:


from sklearn.ensemble import RandomForestClassifier

# Dictionary with parameters
params = {'n_estimators':10,
          'criterion': 'gini',
          'max_depth': 5,
          'min_samples_split': 10,
          'min_samples_leaf': 2,
          'random_state': 47}

# This time we are sending parameters
neptune.create_experiment(params=params)

clf = RandomForestClassifier(**params)
clf.fit(X_train, y_train)
auc = roc_auc_score(y_test, lr.predict_proba(X_test)[:, 1])

# Sending a metric to the experiment
neptune.send_metric('AUC', auc)

# Stop the experiment
neptune.stop()


# You can now find this parameters in the experiment's page if you follow the link above.
# 
# ![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F1696976%2F9ee8902a6a74a207a3d475c205414f4a%2Fneptune_kernel_8.png?generation=1564561096062206&alt=media)
# 
# As for dashboard - you should select parameters you want to be displayed. You can then sort and filter your experiments by any of them.

# <a id="5"></a>
# ## Track images
# 
# Another thing you can log is an images. If you are more visual person this might be really helpful feature for you. Let's, for example, train another model and...

# In[ ]:


from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt


# Dictionary with parameters
params = {'hidden_layer_sizes': (200,),
          'activation': 'relu',
          'max_iter': 500,
          'learning_rate': 'adaptive',
          'random_state': 47}

neptune.create_experiment(params=params)

clf = MLPClassifier(**params)
clf.fit(X_train, y_train)
auc = roc_auc_score(y_test, lr.predict_proba(X_test)[:, 1])

# Sending a metric to the experiment
neptune.send_metric('AUC', auc)

plt.plot(clf.loss_curve_)
plt.savefig('loss_curve.png')
neptune.send_image('loss_curve', 'loss_curve.png')
neptune.stop()


# An image is now available in 'Logs' section of the experiment's page.
# 
# ![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F1696976%2F545fec8350e6c060d49eac18b0d33ddb%2Fneptune_kernel_9.png?generation=1564562627500744&alt=media)

# <a id="6"></a>
# ## Track artifacts
# 
# You can also send some artifacts. For example lets dump the model to pkl and send it to neptune. It is going to be stored in 'Artifacts' section.
# 
# Also we can use neptune experiment in more pythonic way.

# In[ ]:


import joblib

with neptune.create_experiment():
    clf = LogisticRegression(solver='lbfgs', random_state=47)
    clf.fit(X_train, y_train)
    auc = roc_auc_score(y_test, lr.predict_proba(X_test)[:, 1])
    neptune.send_metric('AUC', auc)
    joblib.dump(clf, 'logistic_regression.pkl')
    neptune.send_artifact('logistic_regression.pkl')


# <a id="7"></a>
# ## Monitoring resources
# 
# Neptun client has another cool feature - by default it tracks all the resources usage metrics, such as CPU, Memory and GPU usage. They can be found in the 'Monitoring' section of the experiment's page. 
# 
# For example you can see how your model utilizes a GPU during the experiment.
# 
# ![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F1696976%2F0946f5f0d667c85296a0cddfe23497e8%2Fneptune_kernel_10.png?generation=1564563636462867&alt=media)

# <a id="8"></a>
# ## Conclusion
# 
# I have described only basic features and functionality of neptune and if you like it then don't hesitate to explore this servis on our own. Trust me there is much more cool stuff to see.
