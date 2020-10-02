#!/usr/bin/env python
# coding: utf-8

# **Predicting the approximate  price of cars using KNN, SVM, Decision Tree**

# First of all we import the considerable libraries we need

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import os


# In[ ]:


# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore')


# kaggle prerequisite to read data in notebook

# In[ ]:


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# Reading data

# In[ ]:


df = pd.read_csv('../input/used-cars-data-pakistan/OLX_Car_Data_CSV.csv',encoding='unicode_escape')


# **Preprocessing:**

# In[ ]:


df.head()


# In[ ]:


print(f"The value counts of Car Brands:\n{df['Brand'].value_counts()}")


# We have 303 model of cars

# In[ ]:


df['Model'].nunique()


# In[ ]:


df['Transaction Type'].unique()


# Showing the heatmap of null values

# In[ ]:


sns.heatmap(pd.DataFrame(df.isnull().sum()),annot=True,
            cmap=sns.color_palette("cool"),linewidth=1)


# In[ ]:


df[df.isnull().any(axis=1)].shape


# replacing null values the most common values

# In[ ]:


df = df.fillna(df.mode().iloc[0])


# In[ ]:


df.info()


# **IMPORTANT**: Now that we'v got rid of nulls, we start to categorize our features but as you know there are lots of unique values in the 'Brand','Model','Registered City' that may ruin our processing for classification, so we change the values that have low value counts(lower that 2% of all values) as 'Othres'.
# 

# In[ ]:


# Categorical Features
to_drop = []
cat_features = ['Brand','Model','Registered City']
down_limit = 0.02 * len(df)
for feature in cat_features:
    unique_values = df[feature].unique()
    
    # Fill low frequents with 'Others'
    to_drop = [val for val in unique_values if (list(df[feature]).count(val) < down_limit)]
    print('\n', to_drop, 'are now Other')
    
    #replace low count values with 'Others' 
    df[feature].mask(df[feature].isin(to_drop), 'Others', inplace=True)
    
    # To categorical
    # Creating one-hot features using 'get_dummies' function
    temp_df = pd.get_dummies(df[feature], prefix=feature, dtype=np.float64)
    df = pd.concat([df, temp_df], axis=1).drop([feature], axis=1)
    print('{} Categorized'.format(feature), '\n')


# In[ ]:


df.columns


# In[ ]:


df.head()


# In[ ]:


sns.distplot(df['Price'])


# In[ ]:


f, (ax1, ax2) = plt.subplots(1,2, sharey=True, figsize=(15,3))
ax1.scatter(df['Year'], df['Price'])
ax1.set_title('Price vs Year')
ax2.scatter(df['KMs Driven'], df['Price'])
ax2.set_title('Price vs KMs Driven')


# create other features as one-hot features

# In[ ]:


data_with_dummies = pd.get_dummies(df)
data_with_dummies.head()


# In[ ]:


data_with_dummies.columns


# **The Price of cars are not categorical objects so I make the price into equal periods **

# before that we remove outlier data

# In[ ]:


len(df.loc[df["Price"] > 2000000].index)


# In[ ]:


data_with_dummies.drop(data_with_dummies[data_with_dummies['Price'] > 2000000].index,inplace=True)
data_with_dummies.info()


# In[ ]:


from pylab import rcParams

rcParams['figure.figsize'] = 12.7, 8.27
g = sns.kdeplot(data_with_dummies['Price'], color="red", shade = True)
g.set_xlabel("Price")
g.set_ylabel("Frequency")
plt.title('Distribution of Price',size = 20)


# In[ ]:


data_with_dummies["Price Period"] = pd.cut(data_with_dummies.Price,
                                    5,
                                    labels=['Very cheap',
                                            'Cheap', 'Normal',
                                            'Expensive', 'Very expensive'])

data_with_dummies.head()


# removing price column for good

# In[ ]:


data_with_dummies = data_with_dummies.drop(axis=1,columns='Price')
data_with_dummies.head()


# In[ ]:


data_with_dummies['Price Period'].unique()


# In[ ]:


sns.catplot(x="Price Period", y="Year", data=data_with_dummies);
plt.xticks(rotation=45)


# shuffling data

# In[ ]:


data_with_dummies = data_with_dummies.sample(frac=1).reset_index(drop=True)


# seprating target with data

# In[ ]:


df_Target = data_with_dummies.iloc[:,35:36]
col_no_price = data_with_dummies.drop(columns='Price Period')


# Starting to run algorithms:

# Create train and test data(we use 80% of data as training sample)

# In[ ]:


X = np.array(col_no_price.values.tolist())
y = np.array(df_Target.values.tolist())
print(X)
print(y)


# In[ ]:



from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.2,
                                                    random_state = 0)

print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)


# In[ ]:


from sklearn.tree import DecisionTreeClassifier
import sklearn.metrics as metrics

tree_model = DecisionTreeClassifier()
tree_model.fit(X_train, Y_train)
print("training accuracy :", tree_model.score(X_train, Y_train))
print("testing accuracy :", tree_model.score(X_test, Y_test))


# You can change the max_depth of the model in 'DecisionTreeClassifier'

# In[ ]:


print(tree_model.tree_.max_depth)


# In[ ]:


from sklearn.tree import export_graphviz
import graphviz

dot_file = export_graphviz(tree_model,out_file=None,filled=True
                           ,feature_names=col_no_price.columns,rounded=True,max_depth=5)

graphviz.Source(dot_file)


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, Y_train)
print("testing accuracy :",knn.score(X_train, Y_train))
print("testing accuracy :",knn.score(X_test, Y_test))


# In[ ]:


from sklearn.svm import SVC

svc_model = SVC(kernel = "rbf", C = 1.0, gamma = 0.1)
svc_model.fit(X_train, Y_train)
print("training accuracy :", svc_model.score(X_train, Y_train))
print("testing accuracy :", svc_model.score(X_test, Y_test))


# In[ ]:


from sklearn.ensemble import RandomForestClassifier

RF = RandomForestClassifier(n_estimators=9)
RF.fit(X_train, Y_train)
print("training accuracy :", RF.score(X_train, Y_train))
print("testing accuracy :", RF.score(X_test, Y_test))


# As you see the scores are satisfying, now we do the Final evaluation on each model:

# In[ ]:


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
pred = knn.predict(X_test)
print(confusion_matrix(Y_test, pred))
print(classification_report(Y_test, pred))


# In[ ]:


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
pred = svc_model.predict(X_test)
print(confusion_matrix(Y_test, pred))
print(classification_report(Y_test, pred))


# In[ ]:


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
pred = tree_model.predict(X_test)
print(confusion_matrix(Y_test, pred))
print(classification_report(Y_test, pred))


# In[ ]:


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
pred = RF.predict(X_test)
print(confusion_matrix(Y_test, pred))
print(classification_report(Y_test, pred))


# In[ ]:




