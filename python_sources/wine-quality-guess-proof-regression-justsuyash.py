#!/usr/bin/env python
# coding: utf-8

# ## Wine Quality Prediction: ( Challenge at the end )
# 
# ### Date : 14-02-2020
# ### @justsuyash (linkedin)
For more information, read [Cortez et al., 2009].
Input variables (based on physicochemical tests):
1 - fixed acidity
2 - volatile acidity
3 - citric acid
4 - residual sugar
5 - chlorides
6 - free sulfur dioxide
7 - total sulfur dioxide
8 - density
9 - pH
10 - sulphates
11 - alcohol
Output variable (based on sensory data):
12 - quality (score between 0 and 10)
# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

# data visualisation and manipulation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns

#import the necessary modelling algos.
from sklearn.decomposition import PCA

#classifiaction.
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC,SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

#model selection
from sklearn.model_selection import train_test_split,cross_validate
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

#preprocessing
from sklearn.preprocessing import StandardScaler

#evaluation metrics
from sklearn.metrics import mean_squared_log_error,mean_squared_error, r2_score,mean_absolute_error # for regression
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score  # for classification


# In[ ]:


data = pd.read_csv("/kaggle/input/red-wine-quality-cortez-et-al-2009/winequality-red.csv")


# In[ ]:


data['quality'].unique()


# In[ ]:


sp = data['quality'].value_counts()
sp = pd.DataFrame(sp)
sp.T


# In[ ]:


sns.barplot(x = sp.index, y=sp['quality'])
plt.xlabel("Quality Score")
plt.ylabel("Count")


# In[ ]:


plt.figure(figsize = (16,7))
sns.set(font_scale=1.2)
sns.heatmap(data.corr(), annot=True, linewidths=0.5, cmap='YlGnBu')


# ## We see that there are a lot of corelated variables like :
#1) 'fixed acidity' is corealted to 'citric acid' 
#2) 'free sulphur' and 'total sulphur dioxide' are corealted
#3) 'ph' and 'acidity' are negatively corelated(obviously)
# ## But we are not dropping them as we will be doing PCA  and If N variables are highly correlated than they will all load out on the SAME Principal Component (Eigenvector) and hence we do not need to remove them
#1) Use the PCA, and interpret it according to what variables load out on it
#2) Choose one of the highly correlated variables as identified as those that all load onto the same variable and analyse only it.
# In[ ]:


data_cleaned = data


# In[ ]:


data_cleaned.isnull().sum()


# In[ ]:


data_cleaned.describe()


# ##   Lets See an interesting result :

# ## As the kernel suggests and like many people have done as well, we divide the quality in two bins.

# In[ ]:


data_try  = data_cleaned


# In[ ]:


bins = (2, 6.5, 8)
group_names = ['bad', 'good']
data_try['category'] = pd.cut(data_try['quality'], bins = bins, labels = group_names)


# ### We are looking at a highly imbalanced dataset and hence lets make a guess that all wines are bad and lets see the results

# In[ ]:


data_try['category'].value_counts()


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


data_cleaned.columns


# In[ ]:


x1 = data_try[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
       'pH', 'sulphates', 'alcohol']]

y1 = data_try['category']


# In[ ]:


x_train_dummy,x_test_dummy,y_train_dummy,y_test_dummy = train_test_split(x1,y1,test_size = 0.3, random_state=42)


# #  Creating a dummy prediction of all bad quality:
# 
# 
# This is being done in light to get a becnchmark of how good should our model be, I am creating  a dummy prediction in which I have predicted all qualities as bad, as the kernel decscription has told us to. ( Like bin(2,6.5,8)).

# In[ ]:


y_test_dummy.count()


# In[ ]:


y_dummy_predict = []
y_dummy_predict = ['bad']*y_test_dummy.count()


# In[ ]:


accuracy_score(y_dummy_predict,y_test_dummy)


# In[ ]:


print(classification_report(y_test_dummy, y_dummy_predict))


# ##  So we See the problem with most kernels here they achieve an accuracy of 80-88% and are just slightly better or worse than our guess. This is due to the imbalance in data, even if the prediction goes upto 95% its not doing a whole lot of good

# ##  What we will try do instead is consider wines above 5 as 'good' quality wines, and below it as 'bad' quality wines this would lead to somewhat equal distribution and even if we want a extremely good quality wine we would have reduced our options to taste test by a little less than 50%.

# In[ ]:


data_cleaned.info()


# In[ ]:


quality =  data_cleaned['quality'].values

category_balanced = []

for num in quality:
    if num<=5:
        category_balanced.append('bad')
    elif num>=6:
        category_balanced.append('good')  


# In[ ]:


category_balanced  = pd.DataFrame(data=category_balanced,columns=['category_balanced'])


# In[ ]:


category_balanced.isnull().sum()


# In[ ]:


data_cleaned = pd.concat([data_cleaned,category_balanced],axis=1)


# In[ ]:


data_cleaned = data_cleaned.dropna()


# In[ ]:


data_cleaned['category_balanced'].value_counts()


# In[ ]:


x = data_cleaned[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
       'pH', 'sulphates', 'alcohol']]
y = data_cleaned['category_balanced']


# ### We are scaling the data as we do not know the units for each field :

# In[ ]:


scl = StandardScaler()


# In[ ]:


x = scl.fit_transform(x)


# ### Lets perform PCA and see what we get

# In[ ]:


pca = PCA()


# In[ ]:


x_pca = pca.fit_transform(x)


# In[ ]:


plt.figure(figsize=(10,10))
plt.plot(np.cumsum(pca.explained_variance_ratio_), 'ro-')


# ### So we see that 8 features explain about 99% of the variablity so we will use 8 features

# In[ ]:


pca_new = PCA(n_components=8)


# In[ ]:


x_pca_8 = pca_new.fit_transform(x)


# In[ ]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3, random_state=420)


# In[ ]:


x_train.shape


# In[ ]:


x_test.shape


# In[ ]:


y_train.shape


# In[ ]:


y_test.shape


# In[ ]:


models=[LogisticRegression(),SVC(),SVC(kernel='rbf'),KNeighborsClassifier(),RandomForestClassifier(),
        DecisionTreeClassifier(),GradientBoostingClassifier()]
model_names=['LogisticRegression','LinearSVM','rbfSVM','KNearestNeighbors','RandomForestClassifier','DecisionTree',
             'GradientBoostingClassifier','GaussianNB']

acc=[]
d={}

for model in range(len(models)):
    clf=models[model]
    clf.fit(x_train,y_train)
    pred=clf.predict(x_test)
    acc.append(accuracy_score(pred,y_test))
     
d={'Modelling Algo':model_names,'Accuracy':acc}
d


# ###  Let's now have a look at what outcome would a guess work have come up with :

# In[ ]:


print(classification_report(y_test,y_dummy_predict ))


# ##  So we that, it would have only got an accuracy of 20% and now we have something close to 80% with hyperparameter tuning we can get even better result bit I will leave this kernel at this point, Please fell fee to do hyperparameter tuning and other adjustments like dropping corelated columns and share the result with me.

# Think like a Data Scientist and So you become!

# In[ ]:





# # Challenge :
# 
# I have tried removing outliers and it slightly degrades my results, if anyone can remove the outliers and get a better result let me know. 
# 
# Find me on linkedin - @justsuyash

# In[ ]:




