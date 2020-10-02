#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
plt.rcParams['figure.figsize'] = (18,8)


# In[ ]:


from sklearn import preprocessing
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import average_precision_score


# In[ ]:


import tensorflow as tf


# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


def processing(pd_df):
    a = pd_df['Class']
    b = pd_df.drop(['Time','Class'],axis = 1)
    return a,b


# In[ ]:


credit_fraud = pd.read_csv('creditcard.csv') # loading csv dataset into Notebook


# In[ ]:


credit_fraud.head()


# In[ ]:


credit_fraud.info()


# In[ ]:


plt.plot(credit_fraud[credit_fraud['Class']==0]['V1'],credit_fraud[credit_fraud['Class']==0]['V2'],'ro')
plt.plot(credit_fraud[credit_fraud['Class']==1]['V1'],credit_fraud[credit_fraud['Class']==1]['V2'],'bo')
plt.xlabel('1st Component')
plt.ylabel('2nd Component')


# In[ ]:


plt.hist(credit_fraud['Amount'], bins=50)# the only known and understandable colunm is Amount, it is extremely skewed.  


# In[ ]:


target = credit_fraud['Class'] # seperate target from the rest of the data


# In[ ]:


credit_fraud_analysis = credit_fraud.drop(['Time','Class'],axis = 1)


# In[ ]:


x = credit_fraud_analysis
y = target


# In[ ]:


from sklearn.model_selection import train_test_split     #seperate imbalanced dataset into training and testing, !stratify parameter)
X_train, X_test, y_train, y_test = train_test_split(x, y,
                                                stratify=y, 
                                                test_size=0.30)


# In[ ]:


sns.heatmap(X_train.corr())


# In[ ]:


logit = LogisticRegressionCV(Cs = 10, cv = 10, scoring = 'average_precision')
logit.fit(preprocessing.normalize(X_train), y_train)


# In[ ]:


logit.scores_


# In[ ]:


a = logit.predict(preprocessing.normalize(X_test))


# In[ ]:


tn, fp, fn, tp = confusion_matrix(a, y_test).ravel()
(tn, fp, fn, tp)


# In[ ]:


accuracy = (tp + tn)/(tn + fp + fn + tp)
TPR = tp/(tp+fn)
FPR = fp/(fp+tn)
FNR = fn/(tp+fn)
precision = tp/(tp+fp)


# In[ ]:


print(accuracy, TPR, FPR, FNR, precision)


# In[ ]:


forrest = RandomForestClassifier(max_depth = 5, n_estimators = 50)
from sklearn.model_selection import cross_val_score


# In[ ]:


score = cross_val_score(forrest,preprocessing.normalize(X_train), y_train, cv = 10)


# In[ ]:


forrest.fit(preprocessing.normalize(X_train), y_train)


# In[ ]:


result = forrest.predict(preprocessing.normalize(X_test))
confusion_matrix(result, y_test)


# In[ ]:


result = forrest.predict(preprocessing.normalize(X_test))
tn, fp, fn, tp = confusion_matrix(result, y_test).ravel()
print(accuracy, TPR, FPR, FNR, precision)


# In[ ]:





# In[ ]:


credit_fraud_analysis = preprocessing.normalize(credit_fraud_analysis)


# In[ ]:


credit_fraud_analysis = pd.DataFrame(credit_fraud_analysis)


# In[ ]:


credit_fraud_analysis.columns = credit_fraud.columns[1:30]


# In[ ]:


credit_fraud_analysis.head()


# In[ ]:


sns.boxplot(data = credit_fraud_analysis)


# In[ ]:


feature_columns = [tf.feature_column.numeric_column("x", shape=[29])]


# In[ ]:


classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                          hidden_units=[10, 20, 10],
                                          n_classes=2,
                                          model_dir="/tmp/credit_fraud")


# In[ ]:


train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": np.array(X_train)},
      y=np.array(y_train),
      num_epochs=None,
      shuffle=True)


# In[ ]:


classifier.train(input_fn=train_input_fn, steps=5000)


# In[ ]:


test_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": np.array(X_test)},
      y=np.array(y_test),
      num_epochs=1,
      shuffle=False)


# In[ ]:


accuracy_score = classifier.evaluate(input_fn=test_input_fn)


# In[ ]:


predictions = list(classifier.predict(input_fn=test_input_fn))
predicted_classes = [p["classes"] for p in predictions]


# In[ ]:


result = pd.DataFrame(predicted_classes)
c = result[0].apply(lambda x: int(x))


# In[ ]:


np.sum(c)


# In[ ]:


np.sum(y_test)

