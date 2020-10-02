#!/usr/bin/env python
# coding: utf-8

# **Content Of Kernel**
# * Data Arrange and Train,Test
# * Logistic Regression Classification
# * KKN Classification
# * Support Vector Machine
# * Naive Bayes
# * Decision Tree Classification
# * Random Forest Classification
# * Confusion Matrix
# * Comparison of Algorithm Performance

# **Chapter 0 : Data Arrange and Train,Test**

# In[ ]:


import pandas as pd 

data = pd.read_csv("../input/wisconsin-breast-cancer-cytology/BreastCancer2.csv") # reading data from input file
data= data.drop(["id"],axis=1) # remove useless feature 


# In[ ]:


data.head() #data features and class


# In[ ]:


x = data.drop(["class"],axis = 1) # x consist only features
y = data.loc[:,"class"] # y consist only class

print(x.iloc[0:5])
print(y.iloc[0:5])


# In[ ]:


# AttributeError: 'tuple' object has no attribute 'fit' Therefore we have to transform numpy array
x = data.drop(["class"],axis = 1).values 
y = data.loc[:,"class"].values


# In[ ]:


from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.33,random_state=42) # create a %33 test data from orginal data

print("x train shape:",x_train.shape)
print("x test shape:",x_test.shape)
print("y train shape:",y_train.shape)
print("y test shape:",y_test.shape)


# **Chapter 1 : Logistic Classification **
# 

# In[ ]:


from IPython.display import Image
Image("../input/logistic-shematic/lrs.png")


# Image take from [github](https://rasbt.github.io/mlxtend/user_guide/classifier/LogisticRegression/)

# *  x : Features of data - Train part
# *  w : weight - coefficient 
# *  net input fuction : z = w0x0 + w1x1 + ..... + wnxn
# *  Activation function : Sigmoid fuction

# In[ ]:


from IPython.display import Image
Image("../input/function/sigmoid.png")


# Image take from [github](https://rasbt.github.io/mlxtend/user_guide/classifier/LogisticRegression/)

# * Unit Step function : Binary output
# * Error : Backward propagation
# * More detail  in the my previous kernel : [Machine learning Review](https://www.kaggle.com/zayon5/machine-learning-review?scriptVersionId=5819035)

# In[ ]:


from sklearn.linear_model import LogisticRegressionCV # Use the sklearn module 
lrc = LogisticRegressionCV()
lrc.fit(x_train,y_train) 
print("logistic score",lrc.score(x_test,y_test)*100)


# **Chapter 2 : KKN classification**
# KNN Algorithm is relying on feature similarity: 
# 
# 

# In[ ]:


from IPython.display import Image
Image("../input/knnclass/KNN circle.png")


# The green solid circle is test sample should be classified either to the first class of blue squares or to the second class of red triangles. If k = 3 (solid line circle) it is assigned to the second class because there are 2 triangles and only 1 square inside the inner circle. If k = 5 (dashed line circle) it is assigned to the first class (3 squares vs. 2 triangles inside the outer circle).

# In[ ]:


from  sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5) # K values 
knn.fit(x_train,y_train)
print("KNN score",knn.score(x_test,y_test)*100)


# In[ ]:


neighbors_list = []

for i in range(1,15):
    knn1 = KNeighborsClassifier(n_neighbors = i)
    knn1.fit(x_train,y_train)
    neighbors_list.append(knn1.score(x_test,y_test))
print(neighbors_list) # find finest k value for classification


# **Chapter 3 : Support Vector Machine**
# * Support vector machine(SVM) use for regression and classification, but mostly use for classification.  Datas are ploted as a point in n-dimensional space. Points positioning the particular coordinate due to their value.Then, draw line or plane(depens on dimension) for separate the two classes. Choose the line or plane which has maximum margin(distance called as margin). 

# In[ ]:


from IPython.display import Image
Image("../input/support-vector/SVM.png")


# H1 does not separate the classes. H2 does, but only with a small margin. H3 separates them with the maximum margin.

# In[ ]:


from sklearn.svm import SVC
svc = SVC(random_state=1,gamma=0.22)
svc.fit(x_train,y_train)
print("SVC score",svc.score(x_test,y_test))


# **Chapter 4 : Naive Bayes**
# * Navie Bayes algorithm  predict unknown data to classificate and do this process faster than other algorithms. It works on Bayes theorem of probability

# In[ ]:


from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(x_train,y_train)
print("Naive Boyes score",nb.score(x_test,y_test))


# **Chapter 5 : Decision Tree Classification**
# * Decision Tree Classification split the plot into subplot which incluede tree models where points can take values. Each split create leaves which present in the trees represent class label.
# 

# In[ ]:


from IPython.display import Image
Image("../input/treedesi/decision tree_LI.jpg")


# Image taken from [Slideshare](https://www.slideshare.net/marinasantini1/lecture02-machine-learning)

# 

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(x_train,y_train)
print("Decision Tree Score",dt.score(x_test,y_test))


# **Chapter 6 : Random Forest Classification**
# * This classification alghorithm is ensemble learning model. It take sum of the decision tree to one algorithmic process.

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators = 10, random_state=1)
rf.fit(x_train,y_train)
print("Random Forest Classification Score",rf.score(x_test,y_test))


# **Chapter 7 : Confusion Matrix**
# * Confusion matrix create a table which present visualization of performance of an algorithm

# In[ ]:


from IPython.display import Image
Image("../input/confuison/matrix.png")


# Image take from [Codeproject](http://lh3.ggpht.com/_qIDcOEX659I/SzjW6wGbmyI/AAAAAAAAAtY/Nls9tSN6DgU/contingency_thumb%5B3%5D.png?imgmax=800)

# In[ ]:


from sklearn.metrics import confusion_matrix
y_prediction = rf.predict(x_test) 
cm = confusion_matrix(y_true=y_test,y_pred=y_prediction)#actual value -->y_true
print("Confusion Matrix ",cm)


# In[ ]:


# Vizualition of Confusion Matrix
import matplotlib.pyplot as plt
import seaborn as sns

f,ax = plt.subplots(figsize=(5,5))
sns.heatmap(cm,annot = True,fmt=".0f")
plt.xlabel("Prediction Value")
plt.ylabel("True Value")
plt.show()


# **Comparison of Algorithm Performance**

# In[ ]:


print("Logistic Regression Classification score :",lrc.score(x_test,y_test)*100)
print("KNN score :",knn.score(x_test,y_test)*100)
print("SVC score :",svc.score(x_test,y_test)*100)
print("Naive Boyes score :",nb.score(x_test,y_test)*100)
print("Decision Tree Score :",dt.score(x_test,y_test)*100)
print("Random Forest Classification Score :",rf.score(x_test,y_test)*100)


# 

# I hope this kernel help you to understand this consept.
# 
# Please comment if something wrong
# 
# Thank you for effort.
# 
# 
# 
# 
