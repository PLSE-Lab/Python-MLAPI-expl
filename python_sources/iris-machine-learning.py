#!/usr/bin/env python
# coding: utf-8

# ### Iris Species

# ### Objectives:
# Classify Iris flowers based on the length and width measurements of their sepals and petals.

# ### Import the Libaries

# In[ ]:


import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# ### Import Dataset

# In[ ]:


ir=pd.read_csv("/kaggle/input/iris/Iris.csv")


# ### Show Random Rows

# In[ ]:


ir.sample(10)


# ### Cleaning the Dataset

# In[ ]:


#Knowing data types and information

ir.info()


# ##### There are no columns with incomplete data, therefore I will not drop any of them

# In[ ]:


ir.drop(['Id'],axis=1,inplace=True)


# In[ ]:


# Show random rows
ir.sample(10)


# ##### I will drop the column ID as it contains no information

# In[ ]:


#Checking NaN values 
ir.isna().any()


# ##### There are no NaN values, so we won't drop or change values for any of the columns

# ##### From the above we can start processing the data

# In[ ]:


## Data description
ir.describe()


# ### Data Exploration

# In[ ]:


# data exploration 
# check class distributions
import plotly.graph_objects as go
fig = go.Figure(data=[
    go.Pie(labels=['Iris-setosa','Iris-virginica', 'Iris-versicolor'],
           values=ir['Species'].value_counts())
])
fig.update_layout(title_text='class distributions')
fig.show()


# #### From the above it is clear that there is no bias in the data, all are equally distributed.

# In[ ]:


# first we plot histogram of numerical fatures
num_features=ir[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
num_features.hist(bins=50,figsize=(20,15))
plt.show()


# #### From the previous it is clear that:
# - The SepalLengthCm and SepalWidthCM are normally distributed. This decreases the possibility of any outliers in this dataset.
# - The PetalLengthCm and PetalWidthCm are skewed to the right which means there are possibilities for outliers. Therefore, I will apply box plot to apply the test of extreme valuew

# ### Test of Extreme Values: Box Plot

# In[ ]:


sns.boxplot(x='Species', y='PetalLengthCm', data=ir)


# #### From the above we can conclude the following:
# - There is an outlier for the Iris Versicolor
# - There is a big gap between the median of Setosa median and that of Versicolo and virginica. 
# - When checking the Setosa species values, it is clear that the least possible petal  length is 1.3 and the maximum possible value is 1.9 cm. But 25% of the species are below 1.4 cm ( but won't be less than 1.3), and the upper 25% will be above 1.7cm (but not more than 1.9cm)
# - When checking the Virginica species values, it is clear that the least possible petal  length is 4.5 and the maximum possible value is 7 cm. But 25% of the species are below 5 cm ( but won't be less than 4.5), and the upper 25% will be above 6cm (but not more than 7cm)
# - When checking the Versicolor species values, it is clear that the least possible petal  length is 3.5 and the maximum possible value is 5.5 cm. But 25% of the species are below 4 cm ( but won't be less than 3.5), and the upper 25% will be above 5.5 cm (but not more than 7cm)
# - This means that 50% of the observations of the setosa species exist between 1.4 and 1.7, Virginica species exist between 5 and 6, and finally Versicolor between 4 and 5.5cm.
# - There is no kind of overlapping between the three species in the petalLengthCm.
# 

# In[ ]:


sns.boxplot(x='Species', y='PetalWidthCm', data=ir)


# #### From the above we can conclude the following:
# - There are outlier for the Iris Setosa.
# - There is a big gap between the median of Setosa median and that of Versicolo and virginica. 
# - When checking the Setosa species values, it is clear that the least possible petal  width is 0.1 and the maximum possible value is 0.4 cm. But 25% of the species are between 0.1 and 0.3, and the upper 25% will be between 0.3 and 0.4 cm.
# - When checking the Virginica species values, it is clear that the least possible petal  width is 1.4 and the maximum possible value is 2.5 cm. But 25% of the species are between 1.4 and 1.9, while the upper 25% will be between 2.3 and 2.5 cm.
# - When checking the Versicolor species values, it is clear that the least possible petal  width is 1 and the maximum possible value is 1.8 cm. But 25% of the species are 1 and 1.3 cm, while the upper 25% will be between 1.5 and 1.8 cm
# - This means that 50% of the observations of the setosa species exist between 0.3 and 0.4, Virginica species exist between 1.4 and 2.3cm, and finally Versicolor between 1.3 and 1.5cm.
# - There is no kind of overlapping between the three species in the petalWidthCm.
# 

# ### Feature Engineering

# In[ ]:


ir['Species'].value_counts()


# In[ ]:


ir.sample(10)


# In[ ]:


# we need to one hot encode all categorical features
label=ir['Species']
ir.drop(['Species'],inplace=True,axis=1)
ir=pd.get_dummies(ir)
ir.sample(10)


# ### Model Training

# In[ ]:


# we will SVM
from sklearn.svm import SVC #import svm as classifier
from sklearn.model_selection import train_test_split

xtrain,xtest,ytrain,ytest=train_test_split(ir,label,test_size=0.25)


# In[ ]:


# train model
svm = SVC(class_weight='balanced') # create new svm classifier with default parameters
svm.fit(xtrain,ytrain)


# ### Model Evaluation

# In[ ]:


from sklearn.metrics import accuracy_score
predictions = svm.predict(xtest) # test model against test set
preds_train=svm.predict(xtrain)
print("Model Acurracy in testing = {}".format(accuracy_score(ytest, predictions))) # print test accuracy
print("Model Acurracy in train = {}".format(accuracy_score(ytrain, preds_train))) # print train accuracy


# In[ ]:


# confusion matrix
from sklearn.metrics import plot_confusion_matrix # only valid in sklearn versions above or equal scikit-learn==0.22.0
plot_confusion_matrix(svm, xtest, ytest)
plt.show()


# In[ ]:


# evaluate performance on train set
plot_confusion_matrix(svm, xtrain, ytrain)
plt.show()


# In[ ]:





# In[ ]:




