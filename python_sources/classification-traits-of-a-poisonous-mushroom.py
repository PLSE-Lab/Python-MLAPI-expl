#!/usr/bin/env python
# coding: utf-8

#    # <center>Classification of poisonous mushroom</center>
# ![](https://kennettmushrooms.com/wp-content/uploads/2017/05/fungi-funnys-570x285.jpg)

# ## About  Mushroom dataset 

# First column is a classifier
# 0. Class : edible e, poisonous p
# 
# Rest of the columns are 
# 1. cap-shape: bell=b,conical=c,convex=x,flat=f, knobbed=k,sunken=s 
# 2. cap-surface: fibrous=f,grooves=g,scaly=y,smooth=s 
# 3. cap-color: brown=n,buff=b,cinnamon=c,gray=g,green=r, pink=p,purple=u,red=e,white=w,yellow=y 
# 4. bruises?: bruises=t,no=f 
# 5. odor: almond=a,anise=l,creosote=c,fishy=y,foul=f, musty=m,none=n,pungent=p,spicy=s 
# ....

# ## Approach

# 1. Separate X and y variables 
# 2. Use Label encoder to replace text data
# 3. Design multicolumn one hot encoder
# 4. Predict results
# 5. Review optimal K 
# 6. Finding traits of poisonous Mushroom

# Notes : 
# - There are several columns of categorical variables. We need to avoid dummy variable Trap
# - Find a way to rename the columns after one hot encoder operation is done

# In[ ]:


import pandas as pd
import numpy as np


# ## 1. Separate X and y  variables

# In[ ]:


add = "../input/mushrooms.csv"
data = pd.read_csv(add)


# In[ ]:


# seperating X vaules from y values
X= data.iloc[:,1:]
y = data.iloc[:,0]


# ## 2.Use Label encoder to replace text data

# In[ ]:


from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
d = defaultdict (LabelEncoder)
Xfit = X.apply(lambda x: d[x.name].fit_transform(x))


# In[ ]:


le_y = LabelEncoder()
yfit = le_y.fit_transform(y)
# for x in Xfit.columns:
#     print(x)
#     print(Xfit[x].value_counts())


# ## 3. Design Multi-column One Hot encoder 

# - Need to avoid dummy variable trap
# - Using the "d" the defaultdictionary to rename columns after one hot encoder
# - appending new columns after encoding into "final" variable

# In[ ]:


import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import OneHotEncoder
ohc = defaultdict (OneHotEncoder)
# Xfit_ohc = Xfit.apply(lambda x: ohc[x.name].fit_transform(x))
final = pd.DataFrame()

for i in range(22):
    # transforming the columns using One hot encoder
    Xtemp_i = pd.DataFrame(ohc[Xfit.columns[i]].fit_transform(Xfit.iloc[:,i:i+1]).toarray())
   
    #Naming the columns as per label encoder
    ohc_obj  = ohc[Xfit.columns[i]]
    labelEncoder_i= d[Xfit.columns[i]]
    Xtemp_i.columns= Xfit.columns[i]+"_"+labelEncoder_i.inverse_transform(ohc_obj.active_features_)
    
    # taking care of dummy variable trap
    X_ohc_i = Xtemp_i.iloc[:,1:]
    
    #appending the columns to final dataframe
    final = pd.concat([final,X_ohc_i],axis=1)


# In[ ]:


final.shape


# In[ ]:


final.head(20)


# ###  Compare final vs data 

# In[ ]:


final[1:4]


# In[ ]:


data[1:4]


# ## 4. Predict results

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(final, yfit, test_size = 0.1, random_state = 0)


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
classifier =  KNeighborsClassifier(n_neighbors=30,p=2, metric='minkowski')
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)


# In[ ]:


from sklearn.metrics import confusion_matrix
cm= confusion_matrix(y_test,y_pred)
cm


# In[ ]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)


# In[ ]:


classif =  KNeighborsClassifier(n_neighbors=200,p=2, metric='minkowski')
classif.fit(X_train,y_train)
y_pred = classif.predict(X_test)
accuracy_score(y_test,y_pred)


# ## 5. Review optimal K 

# In[ ]:


from sklearn.model_selection import cross_val_score

# creating odd list of K for KNN
myList = list(range(1,200))

# empty list that will hold cv scores
cv_scores = []

# perform 10-fold cross validation
for k in myList[::2]:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')
    cv_scores.append(scores.mean())


# In[ ]:


from matplotlib import pyplot as plt


# In[ ]:


# changing to misclassification error
MSE = [1 - x for x in cv_scores]

# determining best k
optimal_k = myList[::2][MSE.index(min(MSE))]
print ("The optimal number of neighbors is %d" % optimal_k)

# plot misclassification error vs k
plt.plot(myList[::2], MSE)
plt.xlabel('Number of Neighbors K')
plt.ylabel('Misclassification Error')
plt.show()


# KNN provides us a power classifier but finding optimal value of K is very critical. We can take additional test data sets and measure performance with current K = 30 to calibrate and avoid overfit
# 

# ## 6.  Finding traits of poisonous mushrooms
# ( Feature importance in K-NN )

# ##  If I am in the Jungle : How do I survive on mushroom without K-NN
# 
# To answer the above question. We need to find the most significant feature 

# In[ ]:


n_features = final.shape[1]
clf = KNeighborsClassifier()
feature_score = []

for i in range(n_features):
    X_feature= np.reshape(final.iloc[:,i:i+1],-1,1)
    scores = cross_val_score(clf, X_feature, yfit)
    feature_score.append(scores.mean())
    print('%40s        %g' % (final.columns[i], scores.mean()))


# ## The 5 most important factors : to determine poisonous or not

# In[ ]:


feat_imp = pd.Series(data = feature_score, index = final.columns)
feat_imp.sort_values(ascending=False, inplace=True)
feat_imp[feat_imp>0.7]


#  Question from the Jungle : <b>Should I eat or not ? considering factors</b>
# 
# Answer : Need to deep dive to figure out the positive or negative correlation !
# 
# 

# In[ ]:


columns_imp = feat_imp[feat_imp>0.7].index.values
final_Xy= pd.concat([final,pd.DataFrame(yfit,columns=['class'])], axis=1)
grouped = final_Xy.groupby('class')


# In[ ]:


# Edible group of mushrooms
grouped.get_group(0)[columns_imp].sum()


# In[ ]:


# Poisonous group of mushrooms
grouped.get_group(1)[columns_imp].sum()


# ## Final Conclusion

#    # <center>For more clarity on parts of a mushroom</center>
#  
# ![Parts of mushroom](https://infovisual.info/storage/app/media/01/img_en/024%20Mushroom.jpg)
# 
# Now, it is pretty clear that all these factors indicate a poisonous mushroom.
# 
#  <b> DO NOT EAT A MUSHROOM if : </b>
# 1. <b> Odor is foul </b>
# 2. <b> Stalk surface above ring is  silky </b>
# 3. <b> Stalk surface below ring is  silky </b>
# 4. <b> Gill size is narrow </b>
# 5. <b> Spore prints are chocolatey in color </b>
