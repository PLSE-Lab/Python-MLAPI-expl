#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# **Taking Dataset as input unpickling .pkl file**

# In[ ]:


import pickle
import pandas as pd
import numpy as np
import seaborn as sns
import sys
sys.path.append("../tools/")
original = "/kaggle/input/maindata/final_project_dataset.pkl"
destination = "final_dataset.pkl"

content = ''
outsize = 0
with open(original, 'rb') as infile:
    content = infile.read()
with open(destination, 'wb') as output:
    for line in content.splitlines():
        outsize += len(line) + 1
        output.write(line + str.encode('\n'))

enron = pickle.load(open("final_dataset.pkl", "rb"))
df=pd.DataFrame.from_dict(enron)
df=df.transpose()
print(df.info())


# **Converting columns of object datatypes to respective datatypes**

# In[ ]:


df['salary'] = df['salary'].replace("NaN",0)
df['salary']=df['salary'].astype(int)
df['to_messages'] = df['to_messages'].replace("NaN",0)
df['to_messages']=df['to_messages'].astype(int)
df['deferral_payments'] = df['deferral_payments'].replace("NaN",0)
df['deferral_payments']=df['deferral_payments'].astype(int)
df['total_payments'] = df['total_payments'].replace("NaN",0)
df['total_payments']=df['total_payments'].astype(int)
df['loan_advances'] = df['loan_advances'].replace("NaN",0)
df['loan_advances']=df['loan_advances'].astype(int)
df['bonus'] = df['bonus'].replace("NaN",0)
df['bonus']=df['bonus'].astype(int)
df['email_address'] = df['email_address'].replace("NaN","")
df['email_address']=df['email_address'].astype(str)
df['restricted_stock_deferred'] = df['restricted_stock_deferred'].replace("NaN",0)
df['restricted_stock_deferred']=df['restricted_stock_deferred'].astype(int)
df['deferred_income'] = df['deferred_income'].replace("NaN",0)
df['deferred_income']=df['deferred_income'].astype(int)
df['total_stock_value'] = df['total_stock_value'].replace("NaN",0)
df['total_stock_value']=df['total_stock_value'].astype(int)
df['from_poi_to_this_person'] = df['from_poi_to_this_person'].replace("NaN",0)
df['from_poi_to_this_person']=df['from_poi_to_this_person'].astype(int)
df['exercised_stock_options'] = df['exercised_stock_options'].replace("NaN",0)
df['exercised_stock_options']=df['exercised_stock_options'].astype(int)
df['from_messages'] = df['from_messages'].replace("NaN",0)
df['from_messages']=df['from_messages'].astype(int)
df['other'] = df['other'].replace("NaN",0)
df['other']=df['other'].astype(int)
df['from_this_person_to_poi'] = df['from_this_person_to_poi'].replace("NaN",0)
df['from_this_person_to_poi']=df['from_this_person_to_poi'].astype(int)
df['long_term_incentive'] = df['long_term_incentive'].replace("NaN",0)
df['long_term_incentive']=df['long_term_incentive'].astype(int)
df['shared_receipt_with_poi'] = df['shared_receipt_with_poi'].replace("NaN",0)
df['shared_receipt_with_poi']=df['shared_receipt_with_poi'].astype(int)
df['restricted_stock'] = df['restricted_stock'].replace("NaN",0)
df['restricted_stock']=df['restricted_stock'].astype(int)
df['director_fees'] = df['director_fees'].replace("NaN",0)
df['director_fees']=df['director_fees'].astype(int)
df['expenses'] = df['expenses'].replace("NaN",0)
df['expenses']=df['expenses'].astype(int)
df['poi'] = df['poi'].replace("NaN",False)
df['poi']=df['poi'].astype(bool)
print(df.info())


# **Identifying the outlier with graph using indexes as labels. Here TOTAL is as outlier**

# In[ ]:


sns.set(rc={'figure.figsize':(15,10)})
trace1 = sns.scatterplot(
    x=df.salary,
    y=df.bonus,
    data=df
)
for line in range(0,df.shape[0]):
     trace1.text(df.salary[line]+0.2, df.bonus[line], df.index[line], horizontalalignment='left', size='large', color='black', weight='semibold')


# **Removing TOTAL being an outlier and plotting the rest**

# In[ ]:


df.drop(['TOTAL'], axis = 0, inplace= True)
trace2 = sns.scatterplot(
    x=df.salary,
    y=df.bonus,
    data=df,
    sizes=(2000,20000)
)
for line in range(0,df.shape[0]):
     trace2.text(df.salary[line]+0.2, df.bonus[line], df.index[line], horizontalalignment='left', size='small', color='black', weight='semibold')


# **Just to check whether there is any other index like TOTAL which is not a NAME**

# **FOUND one as THE TRAVEL AGENCY IN THE PARK**

# In[ ]:


for i in df.index:
    print(i)
df1=df.transpose()
print("\n\nTHE TRAVEL AGENCY IN THE PARK")
print(df1['THE TRAVEL AGENCY IN THE PARK'])
df.drop(['THE TRAVEL AGENCY IN THE PARK'], axis = 0, inplace= True)


# **USING existing featues to generate new featues**

# In[ ]:


df["fraction_from_poi"] = df["from_poi_to_this_person"].divide(df["to_messages"], fill_value = 0)

df["fraction_to_poi"] = df["from_this_person_to_poi"].divide(df["from_messages"], fill_value = 0)

df["fraction_from_poi"] = df["fraction_from_poi"].fillna(0.0)
df["fraction_to_poi"] = df["fraction_to_poi"].fillna(0.0)

print(df.info())


# **Below two plots show the difference of feature engineering**

# In[ ]:


ax1 = sns.scatterplot(x=df["from_poi_to_this_person"], y=df["from_this_person_to_poi"],data=df,size=df.index,hue=df.index,sizes=(20,200))


# In[ ]:


ax2 = sns.scatterplot(x=df["fraction_from_poi"], y=df["fraction_to_poi"],data=df,size=df.index,hue=df.index,sizes=(20,200))


# In[ ]:


def featureFormat(dictionary, features, remove_NaN=True, remove_all_zeroes=True, remove_any_zeroes=False, sort_keys=False):
    return_list = []
    if isinstance(sort_keys, str):
        import pickle
        keys = pickle.load(open(sort_keys, "rb"))
    elif sort_keys:
        keys = sorted(dictionary.keys())
    else:
        keys = dictionary.keys()

    for key in keys:
        tmp_list = []
        for feature in features:
            try:
                dictionary[key][feature]
            except KeyError:
                print("error: key ", feature, " not present")
                return
            value = dictionary[key][feature]
            if value == "NaN" and remove_NaN:
                value = 0
            tmp_list.append(float(value))
        append = True
        if features[0] == 'poi':
            test_list = tmp_list[1:]
        else:
            test_list = tmp_list
        if remove_all_zeroes:
            append = False
            for item in test_list:
                if item != 0 and item != "NaN":
                    append = True
                    break
        if remove_any_zeroes:
            if 0 in test_list or "NaN" in test_list:
                append = False
        if append:
            return_list.append(np.array(tmp_list))
    return np.array(return_list)


def targetFeatureSplit(data):
    target = []
    features = []
    for item in data:
        target.append(item[0])
        features.append(item[1:])

    return target, features


# **Performing Feature selection to find out which features stand out and can be used in classifier for best accuracy**

# In[ ]:


my_dataset = df.to_dict('index')

from sklearn.feature_selection import SelectKBest, f_classif

features_list = ["poi", "bonus", "exercised_stock_options", "expenses", "other", "restricted_stock", "salary", 
                  "shared_receipt_with_poi", "total_payments", "total_stock_value", "fraction_to_poi",
                 "fraction_from_poi","loan_advances","long_term_incentive"]

data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

# Perform feature selection
selector = SelectKBest(f_classif, k=5)
selector.fit(features, labels)

# Get the raw p-values for each feature, and transform from p-values into scores
scores = -np.log10(selector.pvalues_)

ax = sns.barplot(x=features_list[1:], y=scores)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90, horizontalalignment='right')


# **Apply Feature scaling using MinMaxScaler to all features**

# In[ ]:


from sklearn.preprocessing import MinMaxScaler
X = df.drop(columns = ["poi", "to_messages", "email_address"])
temp = df.drop(columns = ["poi", "to_messages", "email_address"])
y = df["poi"]
scaler = MinMaxScaler()
X =  scaler.fit_transform(X)
X = pd.DataFrame(X)
X.columns = temp.columns
y = y.reset_index()
y = y.drop(columns = "index")
X = pd.concat([X, y], axis = 1)
df = X


# **Using only required featues and dropping all unneccessary featues & Splitting the features using train_test_split**

# In[ ]:


from sklearn.model_selection import train_test_split
scaler = MinMaxScaler()
y = df["poi"]
X = df[["bonus","exercised_stock_options","salary","total_stock_value","fraction_to_poi","long_term_incentive"]]
features_list=["bonus","exercised_stock_options","salary","total_stock_value","fraction_to_poi","long_term_incentive"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,random_state=1)
print(len(y_test))


# **Using Random Forest Classifier**

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
clf = RandomForestClassifier()
pred = clf.fit(X_train, y_train).predict(X_test)
acc=accuracy_score(pred,y_test)
print("Random Forest Classifier",acc)


# **Using AdaBoost Classifier**

# In[ ]:


from sklearn.ensemble import AdaBoostClassifier
clf = AdaBoostClassifier()
clf.fit(X_train, y_train)
pred=clf.predict(X_test)
acc=accuracy_score(pred,y_test)
print("AdaBoost",acc)


# **Using Gaussian Naive Bayes Classifier**

# In[ ]:


from sklearn.naive_bayes import GaussianNB
clf=GaussianNB()
clf.fit(X_train, y_train)
pred=clf.predict(X_test)
acc=accuracy_score(pred,y_test)
print("G NB",acc)


# **Using Decision Tree Classifier**

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
clf1 = DecisionTreeClassifier()
clf1.fit(X_train, y_train)
pred1=clf1.predict(X_test)
acc=accuracy_score(pred1,y_test)
print("Decision Tree",acc)


# **Using KNN classifier**

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
clf2=KNeighborsClassifier(n_neighbors=6)
clf2.fit(X_train, y_train)
pred2=clf.predict(X_test)
acc=accuracy_score(pred2,y_test)
print("KNN",acc)


# **Converting output to pickle file format**

# In[ ]:


import pickle
pickle.dump(clf1, open("my_classifier.pkl", "wb") )
pickle.dump(pred1, open("my_dataset.pkl", "wb") )
pickle.dump(features_list, open("my_feature_list.pkl", "wb") )

