#!/usr/bin/env python
# coding: utf-8

# Attribute Information: (class attribute has been moved to last column)
#    #  Attribute                     Domain
#    -- -----------------------------------------
#    1. Sample code number            id number
#    2. Clump Thickness               1 - 10
#    3. Uniformity of Cell Size       1 - 10
#    4. Uniformity of Cell Shape      1 - 10
#    5. Marginal Adhesion             1 - 10
#    6. Single Epithelial Cell Size   1 - 10
#    7. Bare Nuclei                   1 - 10
#    8. Bland Chromatin               1 - 10
#    9. Normal Nucleoli               1 - 10
#   10. Mitoses                       1 - 10
# >   11. Class:                        (2 for benign, 4 for malignant)

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

import os
print(os.listdir("../input"))


# In[ ]:


labels = ['id', 'Clump Thickness', 'Uniformity of Cell Size', 
       'Uniformity of Cell Shape', 'Marginal Adhesion', 
       'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin',
       'Normal Nucleoli', 'Mitoses', 'Class']
df = pd.read_csv('../input/breast-cancer-wisconsin.data.txt')
df.columns = labels
#data.set_index('id', inplace=True)
df.head()


# In[ ]:


df.columns[1:10]


# # Data Preprocessing

# In[ ]:


print(np.where(df.isnull()))
print(df.describe())


# In[ ]:


df.info()


# In[ ]:


df['Bare Nuclei'].describe()


# In[ ]:


df['Bare Nuclei'].value_counts()


# ## Assigning the 0 to '?' in Bare Nuclei Column

# In[ ]:


bare_index = df[df['Bare Nuclei'] == '?'].index
b = np.array(bare_index)


# In[ ]:


df.loc[b,'Bare Nuclei'] = 0


# In[ ]:


df['Class'].value_counts()


# ### 0.0 is assign to 2 class & 0.1 is assign to 4 class

# In[ ]:


df['Class'] = df['Class'] / 2 - 1


# In[ ]:


df['Class'].value_counts()


# ## Get the Features and Target variable

# In[ ]:


features = df[['Clump Thickness', 'Uniformity of Cell Size', 
       'Uniformity of Cell Shape', 'Marginal Adhesion', 
       'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin',
       'Normal Nucleoli', 'Mitoses']]
target = df['Class']


# ## Standard Scaler

# In[ ]:


from sklearn.preprocessing import StandardScaler


# In[ ]:


scaler = StandardScaler()


# In[ ]:


scaler_feature = scaler.fit_transform(features)


# Scaled value Assign to DataFrame

# In[ ]:


df_feature = pd.DataFrame(scaler_feature,columns=df.columns[1:10])


# In[ ]:


df_feature.iloc[1:3]


# In[ ]:


X = df_feature
y = target


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


# ### Split dataset into 4 parts X_train,X_test,y_train,y_test

# In[ ]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)


# 
# ## Applying ELBOW Method : for finding K value

# In[ ]:


error_rate = []

for i in range(1,40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    y_pred = knn.predict(X_test)
    error_rate.append(np.mean(y_pred != y_test))


# In[ ]:


plt.figure(figsize=(10,5))
plt.xlabel('K')
plt.ylabel('Error Rate')
plt.title('Error Rate vs. K')
plt.plot(range(1,40),error_rate,color='blue',marker='o',markerfacecolor='pink',markersize=5,ls='--')


# Above plot we take the k value is 24 beacuse the after 24 all value can be continues mean no fluctuation

# # KNN ALGORITHM 

# In[ ]:


knn = KNeighborsClassifier(n_neighbors=24)


# ### Train data fit in model

# In[ ]:


knn.fit(X_train,y_train)


# ### Let's do Predict

# In[ ]:


y_pred = knn.predict(X_test)


# In[ ]:


from sklearn.metrics import classification_report,confusion_matrix


# In[ ]:


print(classification_report(y_test,y_pred))


# In[ ]:


tn,fp,fn,tp = confusion_matrix(y_test,y_pred).ravel()
print(tn,fp,fn,tp)


# ## Accuracy of Model

# In[ ]:


from sklearn.metrics import accuracy_score


# In[ ]:


# inbuilt function
print('Accuracy Score :',accuracy_score(y_test,y_pred))


# In[ ]:


#own calculation
total = sum(sum(confusion_matrix(y_test,y_pred)))
accuracy = (tn + tp)/total
print("Accuracy : ",accuracy)


# # Find Sensitivity and Specificity

# In[ ]:


# recall(ACTUAL) and sensitivity (TPR)
sensitivity = tp / (tp + fn)
print('Sensitivity : ', sensitivity )


# In[ ]:


#precision(PREDICT) and specificity (TNR)
specificity = tn /(tn + fp )
print('Specificity : ', specificity)


# Please is there any mistake or any imporvment then comment here. 
# Please leave me a comment and upvote the kernel if you liked.
# 
# Thank you for your time.

# In[ ]:




