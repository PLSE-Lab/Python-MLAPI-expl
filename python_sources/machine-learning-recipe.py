#!/usr/bin/env python
# coding: utf-8

# # load your data 

# In[ ]:


df = pd.read_csv(filepath)
df


# # discover your data 
# ## define target 
# ## checking missing values, and value counts 

# In[ ]:


target_col = 'col'
print(df[target_col].unique())
print('-----------------------')
print(df[target_col].value_counts())
print('-----------------------')
print(df.dtypes)
print('-----------------------')
print(df.isna().sum())


# # prepare X, y and split data 

# In[ ]:


y = df[target_col]
X = df.drop(columns=[target_col])
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)


# # Hyperparameter range 

# In[ ]:


para = list(range(100, 601, 100))
print(para)


# # build model + parameter tuning 

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
results = {}
for n in para:
    print('para=', n)
    model = RandomForestClassifier(n_estimators=n)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    accu = accuracy_score(y_true=y_test, y_pred=preds)
    f1 = f1_score(y_true=y_test, y_pred=preds, average='micro')
    print(classification_report(y_true=y_test, y_pred=preds))
    print('--------------------------')
    results[n] = f1


# # show results 

# In[ ]:


import matplotlib.pylab as plt
# sorted by key, return a list of tuples
lists = sorted(results.items()) 
p, a = zip(*lists) # unpack a list of pairs into two tuples
plt.plot(p, a)
plt.show()


# # select best parameters 

# In[ ]:


best_para = max(results, key=results.get)
print('best para', best_para)
print('value', results[best_para])



# # write predcitions 

# In[ ]:


sub_df.to_csv('sub.csv', index=False)

