#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df=pd.read_csv('../input/Kaggle_Training_Dataset.csv')


# In[ ]:


df.head(3)


# In[ ]:


df.columns


# In[ ]:


df=pd.read_csv('../input/Kaggle_Training_Dataset.csv')


# In[ ]:


df.head(2)


# In[ ]:


df.columns


# In[ ]:


df.isnull().sum()


# In[ ]:


df.lead_time.unique()


# In[ ]:


df[df.lead_time ==12].head(3)


# In[ ]:


ob=df.dtypes[df.dtypes=='object'].index
df[ob].head(3)
Y_train=df['went_on_backorder']


# In[ ]:


#hot encoding
from sklearn.preprocessing import LabelEncoder
ec=LabelEncoder()
for col in ob:
    df[col]=ec.fit_transform(df[col])


# In[ ]:


df.head(3)


# In[ ]:


df=df.drop(['sku'],axis=1)


# In[ ]:


df=df.drop(['lead_time'],axis=1)


# In[ ]:


df.head(3)


# In[ ]:


corr=df.corr()
corr = (corr)
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
sns.plt.title('Heatmap of Correlation Matrix')


# In[ ]:


# GETTING Correllation matrix

corr_mat=df.corr(method='pearson')
plt.figure(figsize=(22,23))
sns.heatmap(corr_mat,vmax=1,square=True,annot=True,cmap='cubehelix') 


# In[ ]:


#Principle component analysis
from sklearn.decomposition import PCA
pca = PCA()
pa=pca.fit_transform(df)
pa


# In[ ]:


df.columns


# In[ ]:


covariance=pca.get_covariance()
explained_variance=pca.explained_variance_
explained_variance


# In[ ]:


with plt.style.context('dark_background'):
    plt.figure(figsize=(6, 4))
    
    plt.bar(range(21), explained_variance, alpha=0.5, align='center',
            label='individual explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.legend(loc='best')
    plt.tight_layout()


# In[ ]:


df.columns


# In[ ]:


df=df.fillna(0)


# In[ ]:


plt.figure()
df['went_on_backorder'].value_counts().plot(kind = 'bar')
plt.ylabel("Count")
plt.title('Went on backorder? (0=No, 1=Yes)')


# In[ ]:


plt.figure()
df['went_on_backorder'].value_counts().plot(kind = 'pie')
plt.ylabel("Count")
plt.title('Went on backorder? (0=No, 1=Yes)')


# In[ ]:


sns.countplot(x='went_on_backorder',data=df)


# In[ ]:


Y=df['went_on_backorder']
X=df.drop(['went_on_backorder'],axis=1)


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1, random_state = 1)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier(n_jobs = -1)
RF.fit(X_train, Y_train)
Y_pred = RF.predict(X_test)


# In[ ]:


RF.score(X_test,Y_test)


# In[ ]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve, auc, precision_recall_curve
print('Random Forest Classifier accuracy: %0.3f'% accuracy_score(Y_pred, Y_test))


# In[ ]:


print("Random Forest Classifier report \n", classification_report(Y_pred, Y_test))


# In[ ]:


def roc_curve_acc(Y_test, Y_pred, method):
    false_positive_rate, true_positive_rate, thresholds = roc_curve(Y_test, Y_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    plt.title('Receiver Operating Characteristic')
    plt.plot(false_positive_rate, true_positive_rate, color='cyan' ,label='%s AUC = %0.3f'% (method, roc_auc))
    plt.legend(loc = 'lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([-0.1,1.2])
    plt.ylim([-0.1,1.2])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')


# In[ ]:


print('roc_auc_score: %0.3f'% roc_auc_score(Y_pred, Y_test))


# In[ ]:





# In[ ]:


roc_curve_acc(Y_test, Y_pred, "RF")


# In[ ]:


from sklearn.linear_model import LogisticRegression
LR=LogisticRegression()
LR.fit(X_train, Y_train)
Y_pred_LR = LR.predict(X_test)


# In[ ]:


LR.score(X_test,Y_test)


# In[ ]:


print('Logistic Regrassor Classifier accuracy: %0.3f'% accuracy_score(Y_pred_LR, Y_test))


# In[ ]:


print("Logistic Regrassor Classifier report \n", classification_report(Y_pred_LR, Y_test))


# In[ ]:


roc_curve_acc(Y_test, Y_pred_LR, "LR")


# In[ ]:


cnf_matrix = confusion_matrix(Y_test, Y_pred_LR)
cnf_matrix


# In[ ]:


sns.heatmap(cnf_matrix)


# In[ ]:


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[ ]:


class_names=(0,1)


# In[ ]:


plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')


# In[ ]:


#It shows Random forest is good.

