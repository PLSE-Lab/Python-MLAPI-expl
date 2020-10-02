#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Let's start by importing the data

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from IPython.display import display
from matplotlib import pyplot as plt
plt.style.use = ('seaborn')
# Load the dataset
in_file = '../input/Cleaned_data.csv'
full_data = pd.read_csv(in_file)

# Print the first few entries of the data
display(full_data.head())


# In[ ]:


#lets display some stats
main_data = full_data.drop('Severity', axis =1)

column_names = list(main_data.columns.values)
y=0
for x in main_data:
    #min value 
    s=np.amin(main_data[x])
    t=column_names[y]
    print('{} min: {:.2f}'.format(t,s))
    
    #max value
    p=np.amax(main_data[x])
    print('{} max: {:.2f}'.format(t,p))
    
    #range value
    s=p-s
    print('{} range: {:.2f}'.format(t,s))
    
    #mean value
    s=np.mean(main_data[x])
    print('{} mean: {:.2f}'.format(t,s))
    
    #q1 value
    s=np.percentile(main_data[x],25)
    print('{} q1: {:.2f}'.format(t,s))
    
    #q3 value
    s=np.percentile(main_data[x],75)
    print('{} q3: {:.2f}'.format(t,s))
    
    #std value
    s=np.std(main_data[x])
    print('{} std: {:.2f}'.format(t,s))
    
    y=y+1


# So we have found that there is one row which has a BI-RAD of 55 which is outside of the specified 1-5 values so we will need to delete that row of data

# In[ ]:


print(main_data.iloc[257])


# In[ ]:


#we have found that the row with BI-RADS = 55 is row 257 so lets delete that row
main_data = main_data.drop(257,axis=0)
main_data = main_data.reset_index(drop=True)
print(main_data[250:260])
full_data = full_data.drop(257,axis=0)
full_data = full_data.reset_index(drop=True)
print(full_data[250:260])


# In[ ]:


plt.plot(main_data['BI-RADS'])


# From this plot we can understand that most of the mammograms have been identified as 5 on the BI-RADS scale. Next we will take a look if age has any correlation with BI-RADS.

# In[ ]:


plt.scatter(main_data['Age'],main_data['BI-RADS'])


# From this visualisation, I think we can confidently say that age has no distinct correlation with the BI-RADS scale as across the whole range of ages we can see BI-RAD values of 4 and 5. 

# In[ ]:


from sklearn import cross_validation
x_train,x_test,y_train,y_test= cross_validation.train_test_split(main_data,full_data['Severity'],
                                                                 test_size=0.3,random_state=1)


# In[ ]:


from sklearn.model_selection import StratifiedShuffleSplit

X = main_data.values
Y = full_data['Severity'].values
sss= StratifiedShuffleSplit(n_splits=1,test_size=0.3,train_size=0.7,random_state=1)
for train_index,test_index in sss.split(X,Y)
    x_train2, x_test2 = X[train_index],X[test_index]
    y_train2, y_test2 = Y[train_index],Y[test_index]


# In[ ]:


from sklearn.metrics import confusion_matrix
import itertools
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
def compute_cnf(classifier,x_test,y_test,setno):
    print ("Set "+setno)
    cnf_matrix = confusion_matrix(classifier.predict(x_test),y_test)
    np.set_printoptions(precision=2)
    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=['Benign','Malignant'],
                      title='Confusion matrix, without normalization')

    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=['Benign','Malignant'], normalize=True,
                      title='Normalized confusion matrix')

    plt.show()


# In[ ]:


from sklearn.metrics import f1_score as fscorer

def f1_score(classifier,x_test,y_test):
    return fscorer(classifier.predict(x_test),y_test)


# In[ ]:


from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier(random_state=0)
classifier.fit(x_train,y_train)
print ('Set 1:'+classifier.score(x_test,y_test))
print("Set 1 F1:", f1_score(classifier,x_test,y_test))
print ("Set 2:"+classifier.score(x_test,y_test))
print("Set 2 F1:", f1_score(classifier,x_test,y_test))

# Compute and plot confusion matrix for the best shuffled set
if classifier.score(x_test,y_test)> classifier.score(x_test2,y_test2):
    compute_cnf(classifier,x_test,y_test,1)
else:
    compute_cnf(classifier,x_test2,y_test2,2)


# In[ ]:


from sklearn.tree import ExtraTreeClassifier

classifier = ExtraTreeClassifier(random_state=0)
classifier.fit(x_train,y_train)
print (classifier.score(x_test,y_test))
compute_cnf(classifier,x_test,y_test)


# In[ ]:


from sklearn.svm import SVC

classifier = SVC(kernel='poly',random_state=0)
classifier.fit(x_train,y_train)
print (classifier.score(x_test,y_test))
compute_cnf(classifier,x_test,y_test)


# In[ ]:


classifier = SVC(kernel='sigmoid',random_state=0)
classifier.fit(x_train,y_train)
print (classifier.score(x_test,y_test))
compute_cnf(classifier,x_test,y_test)


# In[ ]:


classifier = SVC(kernel='linear',random_state=0)
classifier.fit(x_train,y_train)
print (classifier.score(x_test,y_test))
compute_cnf(classifier,x_test,y_test)


# In[ ]:


classifier = SVC(kernel='rbf',random_state=0)
classifier.fit(x_train,y_train)
print (classifier.score(x_test,y_test))
compute_cnf(classifier,x_test,y_test)


# In[ ]:


from sklearn.svm import LinearSVC

classifier = LinearSVC(random_state=0)
classifier.fit(x_train,y_train)
print (classifier.score(x_test,y_test))
compute_cnf(classifier,x_test,y_test)


# In[ ]:


from sklearn.svm import NuSVC

classifier = NuSVC(random_state=0)
classifier.fit(x_train,y_train)
print (classifier.score(x_test,y_test))
compute_cnf(classifier,x_test,y_test)


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(algorithm='ball_tree')
classifier.fit(x_train,y_train)
print (classifier.score(x_test,y_test))
compute_cnf(classifier,x_test,y_test)


# In[ ]:


classifier = KNeighborsClassifier(algorithm='kd_tree')
classifier.fit(x_train,y_train)
print (classifier.score(x_test,y_test))
compute_cnf(classifier,x_test,y_test)


# In[ ]:


classifier = KNeighborsClassifier(algorithm='brute')
classifier.fit(x_train,y_train)
print (classifier.score(x_test,y_test))
compute_cnf(classifier,x_test,y_test)


# In[ ]:


from sklearn.gaussian_process import GaussianProcessClassifier

classifier = GaussianProcessClassifier(random_state=0)
classifier.fit(x_train,y_train)
print (classifier.score(x_test,y_test))
compute_cnf(classifier,x_test,y_test)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(random_state=0)
classifier.fit(x_train,y_train)
print (classifier.score(x_test,y_test))
compute_cnf(classifier,x_test,y_test)


# In[ ]:


from sklearn.ensemble import AdaBoostClassifier

classifier = AdaBoostClassifier(random_state=0)
classifier.fit(x_train,y_train)
print (classifier.score(x_test,y_test))
compute_cnf(classifier,x_test,y_test)


# In[ ]:


from sklearn.ensemble import BaggingClassifier

classifier = BaggingClassifier(random_state=0)
classifier.fit(x_train,y_train)
print (classifier.score(x_test,y_test))
compute_cnf(classifier,x_test,y_test)


# In[ ]:


from sklearn.ensemble import ExtraTreesClassifier

classifier = ExtraTreesClassifier(random_state=0)
classifier.fit(x_train,y_train)
print (classifier.score(x_test,y_test))
print (cm(classifier.predict(x_test),y_test))
compute_cnf(classifier,x_test,y_test)


# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier

classifier = GradientBoostingClassifier(loss='exponential',random_state=0)
classifier.fit(x_train,y_train)
print (classifier.score(x_test,y_test))
print (cm(classifier.predict(x_test),y_test))
compute_cnf(classifier,x_test,y_test)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(random_state=0)
classifier.fit(x_train,y_train)
print (classifier.score(x_test,y_test))
print (cm(classifier.predict(x_test),y_test))
compute_cnf(classifier,x_test,y_test)


# In[ ]:


from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()
classifier.fit(x_train,y_train)
print (classifier.score(x_test,y_test))
print (cm(classifier.predict(x_test),y_test))
compute_cnf(classifier,x_test,y_test)


# In[ ]:


from sklearn.naive_bayes import MultinomialNB

classifier = MultinomialNB()
classifier.fit(x_train,y_train)
print (classifier.score(x_test,y_test))
print (cm(classifier.predict(x_test),y_test))
compute_cnf(classifier,x_test,y_test)


# In[ ]:


from sklearn.naive_bayes import BernoulliNB

classifier = BernoulliNB()
classifier.fit(x_train,y_train)
print (classifier.score(x_test,y_test))
print (cm(classifier.predict(x_test),y_test))
compute_cnf(classifier,x_test,y_test)


# In[ ]:


#lets try and predict what the BI-RAD values are from the rest of the data
f_data = main_data.drop('BI-RADS',axis=1)
x_train,x_test,y_train,y_test= cross_validation.train_test_split(f_data,full_data['BI-RADS'],
                                                                 test_size=0.3,random_state=1)


# In[ ]:




