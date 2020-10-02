#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score


# In[ ]:


train_benign = pd.read_csv("../input/dm-assignment-3/train_benign.csv")
train_malware = pd.read_csv("../input/dm-assignment-3/train_malware.csv")
test_set = pd.read_csv("../input/dm-assignment-3/Test_data.csv")


# In[ ]:


#Benign is 0
#Malware is 1

train_benign["Label"] = 0
train_malware["Label"] = 1

train_set = train_benign.append(train_malware)


# In[ ]:


#drop constant columns
list_to_drop = train_set.columns[train_set.nunique() <= 1].values

train_set_dropped = train_set.drop(columns = list_to_drop)
train_set_dropped = train_set_dropped.sample(frac=1).reset_index(drop=True)
test_set_dropped = test_set.drop(columns = list_to_drop)

test_set_dropped = test_set_dropped.drop(columns = ["Unnamed: 1809"])


# In[ ]:


print(train_set_dropped.info())
print(test_set_dropped.info())


# In[ ]:


train_set_dropped.info()


# In[ ]:


train_set_dropped.loc[:,"1":"31"].info()


# In[ ]:


#No null values
train_set_dropped.isnull().values.any()


# In[ ]:


train_set_dropped.loc[:,"1":].describe()


# In[ ]:


#Correlation matrix

train_set_dropped.loc[:,"1":"14"].corr()


# In[ ]:


f = plt.figure(figsize=(19, 15))
plt.matshow(train_set_dropped.loc[:,"1":"14"].corr(), fignum=f.number)
plt.xticks(range(14), train_set_dropped.columns[1:14], fontsize=14, rotation=45)
plt.yticks(range(14), train_set_dropped.columns[1:14], fontsize=14)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)
plt.title('Correlation Matrix', fontsize=16)


# In[ ]:


corr = train_set_dropped.loc[:,"1":"14"].corr()
corr.style.background_gradient(cmap='coolwarm')


# In[ ]:


corr = train_set_dropped.loc[:,"1":"20"].corr()
corr.style.background_gradient(cmap='coolwarm').set_precision(2)


# In[ ]:


train_x = train_set_dropped.drop(columns = ["Label"])
train_y = train_set_dropped["Label"]


# In[ ]:


test_x = test_set_dropped


# In[ ]:


#pca

pca = PCA(n_components = 0.999, svd_solver="full")
pca.fit(train_x)


# In[ ]:


pca.n_components_


# In[ ]:


pca_two = PCA(n_components=2)
principalComponents = pca_two.fit_transform(train_x)
principalComponents_df = pd.DataFrame(data = principalComponents,columns = ['principal component 1', 'principal component 2'])
plt.figure()
plt.figure(figsize=(10,10))
plt.xticks(fontsize=12)
plt.yticks(fontsize=14)
plt.xlabel('Principal Component - 1',fontsize=20)
plt.ylabel('Principal Component - 2',fontsize=20)
plt.title("Principal Component Analysis",fontsize=20)
targets = ['Label 1', 'Label 2']
colors = ['r', 'g']
for target, color in zip(targets,colors):
    indicesToKeep = train_set_dropped['Label'] == target
    plt.scatter(principalComponents_df.loc[indicesToKeep, 'principal component 1']
               , principalComponents_df.loc[indicesToKeep, 'principal component 2'], c = color, s = 50)

plt.legend(targets,prop={'size': 15})


# In[ ]:


print(train_x.info())


# In[ ]:


print(test_x.info())


# In[ ]:


reduced_train_x = pca.transform(train_x)


# In[ ]:


reduced_test_x = pca.transform(test_x)


# In[ ]:


#KNN

from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


#elbow method

x_axis = []
y_axis = []

for i in range(1, 30):
    print(i)
    knn = KNeighborsClassifier(n_neighbors=i, metric='euclidean')
    knn.fit(reduced_train_x, train_y.values)
    y_pred = knn.predict(reduced_train_x)
    y_true = train_y.values
    y_axis.append(accuracy_score(y_true, y_pred))
    x_axis.append(i)


# In[ ]:


plt.xlabel("epochs")
plt.ylabel("accuracy_score")
plt.plot(x_axis, y_axis)

#considering n_neighbours to be 1 now


# In[ ]:


knn = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
knn.fit(reduced_train_x, train_y.values)
y_pred = knn.predict(reduced_train_x)
y_true = train_y.values


# In[ ]:


print(len(reduced_train_x))
print(len(train_y.values))


# In[ ]:


print(confusion_matrix(y_true, y_pred))

accuracy_score(y_true, y_pred)


# In[ ]:


test_y_pred = knn.predict(reduced_test_x)
print(test_y_pred)


# In[ ]:


print(len(reduced_test_x))


# In[ ]:


test_x


# In[ ]:


from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(random_state=0)
lr.fit(reduced_train_x, train_y.values)


# In[ ]:


y_pred = lr.predict(reduced_train_x)
y_true = train_y.values


# In[ ]:


print(confusion_matrix(y_true, y_pred))

accuracy_score(y_true, y_pred)


# In[ ]:


test_y_pred = lr.predict(reduced_test_x)


# In[ ]:


#DT
from sklearn.tree import DecisionTreeClassifier

decision_tree = DecisionTreeClassifier(random_state=0, max_depth = 8)
decision_tree.fit(reduced_train_x, train_y.values)


# In[ ]:


y_pred = decision_tree.predict(reduced_train_x)
y_true = train_y.values


# In[ ]:


print(confusion_matrix(y_true, y_pred))

accuracy_score(y_true, y_pred)


# In[ ]:


test_y_pred = lr.predict(reduced_test_x)


# In[ ]:


from sklearn.ensemble import AdaBoostClassifier

clf = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=1),
    n_estimators=200
)

clf.fit(reduced_train_x, train_y.values)
predictions = clf.predict(reduced_train_x)
print(confusion_matrix(train_y.values, predictions))
accuracy_score(train_y.values, predictions)


# In[ ]:


test_y_pred = clf.predict(reduced_test_x)


# In[ ]:


final_df = pd.DataFrame({'Class': test_y_pred})
final_df["FileName"] = 0
final_df = final_df[["FileName", "Class"]]

for index, row in final_df.iterrows():
    final_df.at[index, "FileName"] = index+1


# In[ ]:


# change output file name here
final_df.to_csv("/kaggle/working/sub3_raw_adaboost.csv", index = False)


# In[ ]:


from IPython.display import HTML
import pandas as pd
import numpy as np
import base64
def create_download_link(df, title = "Download CSV file", filename = "data.csv"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode())
    payload = b64.decode()
    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'
    html = html.format(payload=payload,title=title,filename=filename)
    return HTML(html)
create_download_link(final_df)


# In[ ]:




