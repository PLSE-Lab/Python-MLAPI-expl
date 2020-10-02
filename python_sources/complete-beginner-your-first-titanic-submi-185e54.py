#!/usr/bin/env python
# coding: utf-8

# ## 1. Process the data
# 
# ### Load data

# In[ ]:


#Load data
import pandas as pd

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

#Drop features we are not going to use
train = train.drop(['Name','SibSp','Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'],axis=1)
test = test.drop(['Name','SibSp','Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'],axis=1)

#Look at the first 3 rows of our training data
train.head(5)


# Our data has the following columns:
# - PassengerId - Each passenger's id
# - Survived - Whether the passenger survived or not (1 - yes, 0 - no)
# - Pclass - The passenger class: (1st class - 1, 2nd class - 2, third class - 3)
# - Sex - Each passenger's sex
# - Age - Each passenger's age

# ### Prepare the data to be read by our algorithm

# In[ ]:


#Convert ['male','female'] to [1,0] so that our decision tree can be built
for df in [train,test]:
    df['Sex_binary']=df['Sex'].map({'male':1,'female':0})
    
#Fill in missing age values with 0 (presuming they are a baby if they do not have a listed age)
train['Age'] = train['Age'].fillna(0)
test['Age'] = test['Age'].fillna(0)

#Select feature column names and target variable we are going to use for training
features = ['Pclass','Age','Sex_binary']
target = 'Survived'

#Look at the first 3 rows (we have over 800 total rows) of our training data.; 
#This is input which our classifier will use as an input.
train[features].head(5)


# In[ ]:


#Display first 3 target variables
train[target].head(3).values


# # 2. Create and fit the decision tree
# 
# This tree is definitely going to overfit our data. When you get to the challenge stage, you can return here and tune hyperparameters in this cell. For example, you could reduce the maximum depth of the tree to 3 by setting max_depth=3 with the following command:
# >clf = DecisionTreeClassifier(max_depth=3)
# 
# To change multiple hyperparameters, seperate out the parameters with a comma. For example, to change the learning rate and minimum samples per leaf and the maximum depth fill in the parentheses with the following:
# >clf = DecisionTreeClassifier(max_depth=3,min_samples_leaf=2)
# 
# The other parameters are listed below.
# You can also access the list of parameters by reading the [documentation](http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier) for decision tree classifiers. Another way to access the parameters is to place your cursor in between the parentheses and then press shift-tab.
# 

# In[ ]:


from sklearn.tree import DecisionTreeClassifier

#Create classifier object with default hyperparameters
clf = DecisionTreeClassifier(max_depth=3)  

#Fit our classifier using the training features and the training target values
clf.fit(train[features],train[target]) 


# ### Visualize default tree (optional)
# This is not a necessary step, but it shows you how complex the tree is when you don't restrict it. To complete this visualization section you must be going through the code on your computer.

# In[ ]:


#Create decision tree ".dot" file

#Remove each '#' below to uncomment the two lines and export the file.
#from sklearn.tr#ee import export_graphviz
#export_graphviz(#clf,out_file='titanic_tree.dot',feature_names=features,rounded=True,filled=True,class_names=['Survived','Did not Survive'])


# Note, if you want to generate a new tree png, you need to open terminal (or command prompt) after running the cell above. Navigate to the directory where you have this notebook and the type the following command.
# >dot -Tpng titanic_tree.dot -o titanic_tree.png<br><br>

# In[ ]:


#Display decision tree

#Blue on a node or leaf means the tree thinks the person did not survive
#Orange on a node or leaf means that tree thinks that the person did survive

#In Chrome, to zoom in press control +. To zoom out, press control -. If you are on a Mac, use Command.

#Remove each '#' below to run the two lines below.
#from IPython.core.display import Image, display
#display(Image('titanic_tree.png', width=1900, unconfined=True))


# # 3. Make Predictions

# In[ ]:


#Make predictions using the features from the test data set
predictions = clf.predict(test[features])

#Display our predictions - they are either 0 or 1 for each training instance 
#depending on whether our algorithm believes the person survived or not.
predictions


# # 4. Create csv to upload to Kaggle

# In[ ]:


#Create a  DataFrame with the passengers ids and our prediction regarding whether they survived or not
submission = pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':predictions})

#Visualize the first 5 rows
submission.head()


# In[ ]:


import seaborn as sns

sns.lmplot('PassengerId','Survived',data=submission)


# In[ ]:


#Convert DataFrame to a csv file that can be uploaded
#This is saved in the same directory as your notebook
filename = 'Titanic Predictions 1.csv'

submission.to_csv(filename,index=False)

print('Saved file: ' + filename)

