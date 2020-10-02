#!/usr/bin/env python
# coding: utf-8

# ## Iris Species Dataset                                                                                                                       

# ## Please star/ upvote if you find it helpful.

# ## CONTENTS::

# [ **1 ) Importing Various Modules**](#content1)

#  [ **2 ) Loading the Dataset**](#content2)

#  [ **3 )Exploring the Dataset**](#content3)

#  [ **4 ) Preparing the Data**](#content4)

#  [ **5 ) Modelling**](#content5)

#  [ **6 ) Comparing Different Algortihms**](#content6)

# In[ ]:





# <a id="content1"></a>
# ## 1 ) Importing Various Modules

# In[ ]:


# Ignore  the warnings
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

# data visualisation and manipulation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

#configure
# sets matplotlib to inline and displays graphs below the corressponding cell.
get_ipython().run_line_magic('matplotlib', 'inline')

#scikit-learn.
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler,LabelEncoder 


# <a id="content2"></a>
# ## 2 ) Loading the Dataset

# In[ ]:


train=pd.read_csv(r'../input/Iris.csv')


# In[ ]:


df=train.copy()


# In[ ]:


df.head(10)


# <a id="content3"></a>
# ## 3 ) Exploring the Dataset

# In[ ]:


df.shape


# The dataset has 150 rows and 5 columns out of which the 'Species' is our target variable which we want to predict. 

# In[ ]:


df.columns # names of all coumns.


# Since the data frame is already indexed we will drop the 'Id' column.

# In[ ]:


df.drop(['Id'],axis=1,inplace=True)


# In[ ]:


df.index # indices of rows.


# #### Now we can check if any column has any null or 'Nan' values or not.

# In[ ]:


df.isnull().any()


# This shows that there aren't any 'Nan' values in any column.

# In[ ]:


msno.matrix(df) # just one final time to visualize.


# In[ ]:


for col in df.columns:
    print("Number of values in column " ,col," : ",df[col].count())


# In[ ]:


df.describe()


# This shows the different statistical quantities like mean, median etc.. of all the numeric columns in the data frame.

# ####  VISUALIZING THE DISTRIBUTIION OF FEATURES.

# In[ ]:


def plot(feature):
    fig,axes=plt.subplots(1,2)
    sns.boxplot(data=df,x=feature,ax=axes[0])
    sns.distplot(a=df[feature],ax=axes[1],color='#ff4125')
    fig.set_size_inches(15,5)


# In[ ]:


plot('SepalLengthCm')


# In[ ]:


plot('SepalWidthCm')


# In[ ]:


plot('PetalLengthCm')


# In[ ]:


plot('PetalWidthCm')


# In[ ]:


sns.factorplot(data=df,x='Species',kind='count')


# **VISULAIZING THE FEATURES AGAINST EACH OTHER (by a scatter plot)'**

# In[ ]:


g = sns.PairGrid(df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm',
       'Species']], hue = "Species")
g = g.map(plt.scatter).add_legend()


# <a id="content4"></a>
# ## 4 ) Preparing the Data

# #### LABEL ENCODING THE TARGET

# Since the algorithms accept only numeric values we will encode the 'Species' column using the Labelencoder() from scikit learn.

# In[ ]:


le=LabelEncoder()
df['Species']=le.fit_transform(df['Species'])


# #### NORMALIZING FEATURES 

# Normalizing the features give better performance. Hence I have normalized all the features by scaling them to 0 mean and a unit standard deviation.

# In[ ]:


scaler=StandardScaler()
scaled_df=scaler.fit_transform(df.drop('Species',axis=1))
X=scaled_df
Y=df['Species'].as_matrix()


# In[ ]:


df.head(10)


# #### SPLITTING INTO TRAINING & VALIDATION SETS.

# In[ ]:


x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.20,random_state=42)


# <a id="content5"></a>
# ## 5 ) MODELLING

# #### LOGISTIC REGRESSION

# In[ ]:


clf_lr=LogisticRegression(C=10)
clf_lr.fit(x_train,y_train)
pred=clf_lr.predict(x_test)
print(accuracy_score(pred,y_test))


# #### kNN

# In[ ]:


clf_knn=KNeighborsClassifier()
clf_knn.fit(x_train,y_train)
pred=clf_knn.predict(x_test)
print(accuracy_score(pred,y_test))


# #### Linear Support Vector Machine(SVM)

# In[ ]:


clf_svm_lin=LinearSVC()
clf_svm_lin.fit(x_train,y_train)
pred=clf_svm_lin.predict(x_test)
print(accuracy_score(pred,y_test))


# #### SVM (with 'rbf' kernel)

# In[ ]:


clf_svm=SVC()
clf_svm.fit(x_train,y_train)
pred=clf_svm.predict(x_test)
print(accuracy_score(pred,y_test))


# <a id="content6"></a>
# ## 6 ) COMPARING DIFFERENT ALGORITHMS

# In[ ]:


models=[LogisticRegression(),LinearSVC(),SVC(),KNeighborsClassifier()]
model_names=['LogisticRegression','LinearSVM','rbfSVM','KNearestNeighbors']

acc=[]
d={}

for model in range(len(models)):
    clf=models[model]
    clf.fit(x_train,y_train)
    pred=clf.predict(x_test)
    acc.append(accuracy_score(pred,y_test))
     
d={'Modelling Algo':model_names,'Accuracy':acc}


# In[ ]:


acc_frame=pd.DataFrame(d)
acc_frame


# In[ ]:


sns.factorplot(data=acc_frame,y='Modelling Algo',x='Accuracy',kind='bar',size=5,aspect=1.5)


# 

# ##  THE END !!!

# ## Please star/upvote if you liked it.

# In[ ]:




