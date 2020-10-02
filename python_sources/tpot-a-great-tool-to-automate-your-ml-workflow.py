#!/usr/bin/env python
# coding: utf-8

# # What is TPOT?
# A Python Automated Machine Learning tool that optimizes machine learning pipelines using genetic programming. 
# 
# **[Click here to read the documentation](http://epistasislab.github.io/tpot/)**
# 
# Source: [Github](https://github.com/EpistasisLab/tpot)
# 
# TPOT will automate the most tedious part of machine learning by intelligently exploring thousands of possible pipelines to find the best one for your data.
# 
# <img src="https://raw.githubusercontent.com/EpistasisLab/tpot/master/images/tpot-ml-pipeline.png">
# 
# Once TPOT is finished searching (or you get tired of waiting), it provides you with the Python code for the best pipeline it found so you can tinker with the pipeline from there.
# 
# <img src="https://raw.githubusercontent.com/EpistasisLab/tpot/master/images/tpot-pipeline-example.png">
# 
# **It is a high-level library built on top of numpy, pandas, tqdm, sklearn, DEAP etc. so it is very easy to use.**
# 

# # Basically, it automatically provides you with the best suited cum optimized model for any given problem using genetic programming.
# 
# **Well, it's like you train the TPOT model using desired data and it will provide you with the best suited model for the data from the scikit-learn library as it has been built on top of it.  This model is highly optimized and brings about the best results. In past versions of this kernel, models like Random Forest classifier, Decision Tree classifier and XGBoost classifier. Yes, different models for the same data because every time you run the kernel, due to TPOT's genetic programming approach, different models can come up every time.**
# 
# # How do genetic algorithms work over generations?
# 
# <img src="http://www.jade-cheng.com/au/coalhmm/optimization/gadiagram.svg">
# 
# 
# Here, I will use the iris dataset to give you an example.

# In[ ]:


# Importing libraries 
from tpot import TPOTClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


# In[ ]:


# Loading data
iris = load_iris()
iris.data[0:5], iris.target


# In[ ]:


# Splitting data into training and test set
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target,
                                                    train_size=0.75, test_size=0.25)
X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[ ]:


tpot = TPOTClassifier(generations=8, population_size=50, verbosity=2)
tpot.fit(X_train, y_train)
print("Accuracy is {}%".format(tpot.score(X_test, y_test)*100))


# **You can also export the optimized model as output in a .py file. Check the output section to view the file and see the chosen model.**
# 
# **Due to genetic programming, the resulting model can be different every time you run the model**

# In[ ]:


tpot.export('tpot_iris_pipeline.py')


# In[ ]:




