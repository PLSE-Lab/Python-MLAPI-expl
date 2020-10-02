#!/usr/bin/env python
# coding: utf-8

# ![](https://www.python.org/static/img/python-logo@2x.png)   ![](https://brocku.ca/goodman/wp-content/uploads/primary-site/sites/6/centre-for-business-analytics-logo.png?x59852) 
# 
# # Brock University 5P11 extra materials (Python)
# 
# # About Python
# Python is an interpreted, high-level, general-purpose programming language. Created by Guido van Rossum and first released in 1991, Python has a design philosophy that emphasizes code readability, notably using significant whitespace. It provides constructs that enable clear programming on both small and large scales.[26] Van Rossum led the language community until stepping down as the leader in July 2018. --wikipedia.
# 
# # Install python
# <img src="https://www.anaconda.com/wp-content/uploads/2018/06/cropped-Anaconda_horizontal_RGB-1-600x102.png" height=200>
# In Brock University's labs, they have python(anaconda both 2 and 3) installed. I will skip this process in the class. Just introduce some software here.
# ### If you want to install Python on your machine, my recommendation is to install anaconda 3. GO to https://www.anaconda.com/distribution/ [Anaconda official website](https://www.anaconda.com/distribution/) download python 3.x **distribution** version choose your machine type. (Suggest you select the environment option.)
# <img src="https://cdn-images-1.medium.com/max/1250/1*7a9zVyGP3iMXu9aB4e_Vhw.png" height=500 width=500>
# 
# ## Additional options
# ### Jupyter Notebook
# <img src="https://jupyter.org/assets/main-logo.svg" height=200, width=200>
# Most popular IDE for Python. If you are using anaconda, this package is already included.
# 
# ### Pycharm
# ![](https://upload.wikimedia.org/wikipedia/commons/thumb/a/a1/PyCharm_Logo.svg/192px-PyCharm_Logo.svg.png)
# Install Pycharm form [Pycharm website](https://www.jetbrains.com/pycharm/download/#section=windows) dowload community version (Free!) Install on your machine
# ### PS:how to fix Interpreter field is empty in pycharm 
# [Youtobe Vedio](https://www.youtube.com/watch?v=ypSSGgKAjhc)
# 
# ### Kaggle
# ![](https://upload.wikimedia.org/wikipedia/commons/thumb/7/7c/Kaggle_logo.png/200px-Kaggle_logo.png)
# You can run your script from the kaggle website by creating a kernel (Jupyter Notebook environment). [Kaggle Website](https://www.kaggle.com/)
# 
# ### Colab
# <img src="https://miro.medium.com/max/1086/1*g_x1-5iYRn-SmdVucceiWw.png" width="200" height="200">
# A wonderful python runtime platform on web, it provides free python environment (you don't need python installed on your machine), even free GPU and TPU.
# ps: don't forget save your result to local machine or Gdrive, otherwise it will gone for ever.
# 
# ### repl.it
# <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/b/b2/Repl.it_logo.svg/330px-Repl.it_logo.svg.png" width="200" height="200">
# Another good platform that you can run more than 20 different launguage online for free, without install any software locally. You can also save and share your code.
# https://repl.it/upgrade/ZackDai
# 
# ### PyPI
# ![](https://pypi.org/static/images/logo-large.72ad8bf1.svg)
# The official place to find python libraries. [Pypi website](https://pypi.org/)
# 
# ### Github
# ![](https://avatars1.githubusercontent.com/u/9919?s=200&v=4)
# The world's leading software development platform. [Github website](https://github.com/)
# 
# ### Spyder
# <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/7/7e/Spyder_logo.svg/1024px-Spyder_logo.svg.png"  width="200" height="200">
# Another IDE inside anaconda package.
# 
# ### VScode
# <img src="https://res.cloudinary.com/duninnjce/image/upload/w_600,q_auto,f_auto/vs-code-icon.png" width="200" height="200">
# Free IDE provided by microsoft, can run most programming languages. Lite and powerful
# 
# 

# In[ ]:





# # Useful materials to learn
# 
# ### Medium
# https://medium.com/
# A place to read and write big ideas and important stories
# 
# ### Free ebooks for python
# https://medium.mybridge.co/19-free-ebooks-to-learn-programming-with-python-8f6f0ad4a7f8
# 
# ### what-is-predictive-analytic? (massive tools introduced at the end)
# https://www.predictiveanalyticstoday.com/what-is-predictive-analytics/
# 
# ### tutorialspoint
# https://www.tutorialspoint.com/python_pandas/python_pandas_groupby.htm
# 
# ### dataCamp (Learn lauguage and get free certificates!)
# https://www.datacamp.com/
# 
# ### Stack Overflow (The biggest cummunity of programmers)
# https://stackoverflow.com/
# 
# 

# # Run simple codes here

# In[ ]:


# load Iris data from web
import pandas as pd
data = pd.read_csv('../input/Iris.csv')  

# read data from csv file, you can change '../input/Iris.csv' into your path for example ('c:/data/abc.csv')


# In[ ]:


print(data.columns)  # show the titles of the table


# In[ ]:


data.head()


# In[ ]:


data.drop('Id',axis=1, inplace=True) # delete useless column 'Id'


# In[ ]:


data.plot() # simple visualization


# In[ ]:


from pandas.plotting import scatter_matrix

scatter_matrix(data) # show the relationship between columns


# In[ ]:


data1 = pd.get_dummies(data) # create dummpy variable
data1 = data1.sample(frac=1) # random shuffle the data
data1.head(10) # show first 10 columns


# In[ ]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

x_train,  x_validate, y_train, y_validate = train_test_split(data.iloc[:,:4], data.iloc[:,4:], test_size=0.3) 

# split the data into train set and validate set, x is input varriables, y is the target value


# In[ ]:


y_train.columns # Check the target value name


# In[ ]:


model = DecisionTreeClassifier(max_depth=5, max_leaf_nodes=10, min_samples_split=5) # create a decision tree model
model.fit(x_train, y_train) # train the model on train set data
y_return = model.predict(x_train)  # get model result from x_train
print('accuracy: ', accuracy_score(y_train, y_return))  # show the train score


# ### pretty good train result, lets see the validation result

# In[ ]:


y_predict = model.predict(x_validate) # predict on validate data
print('validate accuracy: ', accuracy_score(y_validate, y_predict))  # show the score on validate data set


# ### Good result, but you can do better with better model!
# Note: the result might a little bit different as the random splitting procedure.

# In[ ]:


print(confusion_matrix(y_validate, y_predict))  # Show the confusion_matrix

