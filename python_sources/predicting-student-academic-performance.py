#!/usr/bin/env python
# coding: utf-8

# # Predicting Student Academic Performance
# ## an exploration in data visualiation and machine learning efffectiveness
# #### The goal of this project was to examine a number of ML algorithms that were capable of adjusting to categorical data and attempt to predict student performance. Some parts about our problem that make it unique are:  There are 3 classes and most of our data is categorical data and not purely quantitative. Our goal with this was to perform some initial data visualzation and to determine which classifier handles this data the best.
# ##### Our project used the Kaggle.com dataset found [here](https://www.kaggle.com/aljarah/xAPI-Edu-Data).
#  ## Reading in the data

# In[1]:


import pandas as pd    # a wonderful dataframe to work with
import numpy as np     # adding a number of mathematical and science functions
import seaborn as sns  # a very easy to use statistical data visualization package
import matplotlib.pyplot as plt # a required plotting tool
import warnings
# sklearn is a big source of pre-written and mostly optimized ML algorithms.
# Here we use their Decision trees, Support Vector Machines, and the classic Perceptron. 
from sklearn import preprocessing, svm   
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier
#ignore warnings
warnings.filterwarnings('ignore')


data = pd.read_csv('../input/xAPI-Edu-Data.csv')
data.head()


# In[2]:


data.tail()


# ## Data Fields
# <table>
#     <tr>
#     <th>Data Field</th>
#     <th>Description</th>
#     </tr>
#     <tr>
#     <th>gender</th>
#     <td>The student's gender.</td>
#     </tr>
#     <tr>
#     <th>NationalITy</th>
#     <td>The student's nationality.</td>
#     </tr>
#     <tr>
#     <th>PlaceofBirth</th>
#     <td>The student's country of birth.</td>
#     </tr>
#     <tr>
#     <th>StageID</th>
#     <td>Educational level student belongs to (Elementary, Middle, or High School).</td>
#     </tr>
#     <tr>
#     <th>GradeID</th>
#     <td>The grade year of the student.</td>
#     </tr>
#     <tr>
#     <th>SectionID</th>
#     <td>The classroom the student is in.</td>
#     </tr>
#     <tr>
#     <th>Topic</th>
#     <td>The topic of the course.</td>
#     </tr>
#     <tr>
#     <th>Semester</th>
#     <td>The semester of the school year.   (F for Fall, S for Spring)</td>
#     </tr>
#     <tr>
#     <th>Relation</th>
#     <td>The parent responsible for student.</td>
#     </tr>
#     <tr>
#     <th>raisedhands</th>
#     <td>How many times the student raises his/her hand on classroom</td>
#     </tr>
#     <tr>
#     <th>VisITedResources</th>
#     <td>How many times the student visits a course content</td>
#     </tr>
#     <tr>
#     <th>AnnouncementsView</th>
#     <td>How many times the student checks the new announcements</td>
#     </tr>
#     <tr>
#     <th>Discussion</th>
#     <td>How many times the student participate on discussion groups</td>
#     </tr>
#     <tr>
#     <th>ParentAnsweringSurvey</th>
#     <td>Parent answered the surveys which are provided from school or not</td>
#     </tr>
#     <tr>
#     <th>ParentschoolSatisfaction</th>
#     <td>Whether or not the parents were satisfied. "Good" or "Bad". Oddly this was not null for parents who did not answer the survey. It is unclear how this value was filled in.</td>
#     </tr>
#     <tr>
#     <th>StudentAbsenceDays</th>
#     <td>Whether or not a student was absent for more than 7 days</td>
#     </tr>
#     <tr>
#     <th>Class</th>
#     <th>Our classification field. 'L' is for students who got a failing percentage (Less than 69%), 'M' for students who got a low passing grade (Between 70% and 89%), and 'H' for students who achieved high marks in their course (90% to 100%)</th>
#     </tr>
#     </table>
#     
# ## Preliminary Data Visuialization
# #### Our goal with our data visuialization is to get an idea of the shape of our data and to see if we can easily identify any possible outliers. Because this is primarily categorical data we look mostly at countplots of the datafields and our classes. We also look to see if any of our data is unclear or redundant.

# In[3]:


ax = sns.countplot(x='Class', data=data, order=['L', 'M', 'H'])
for p in ax.patches:
    ax.annotate('{:.2f}%'.format((p.get_height() * 100) / len(data)), (p.get_x() + 0.24, p.get_height() + 2))
plt.show()


# In[4]:


fig, axarr  = plt.subplots(2,figsize=(10,10))
sns.countplot(x='gender', data=data, order=['M','F'], ax=axarr[0])
sns.countplot(x='gender', hue='Class', data=data, order=['M', 'F'],hue_order = ['L', 'M', 'H'], ax=axarr[1])
plt.show()


# In[5]:


fig, axarr  = plt.subplots(2,figsize=(10,10))
sns.countplot(x='NationalITy', data=data, ax=axarr[0])
sns.countplot(x='NationalITy', hue='Class', data=data,hue_order = ['L', 'M', 'H'], ax=axarr[1])
plt.show()


# In[6]:


fig, axarr  = plt.subplots(2,figsize=(10,10))
sns.countplot(x='PlaceofBirth', data=data, ax=axarr[0])
sns.countplot(x='PlaceofBirth', hue='Class', data=data,hue_order = ['L', 'M', 'H'], ax=axarr[1])
plt.show()


# In[7]:


fig, axarr  = plt.subplots(2,figsize=(10,10))
sns.countplot(x='StageID', data=data, ax=axarr[0])
sns.countplot(x='StageID', hue='Class', data=data, hue_order = ['L', 'M', 'H'], ax=axarr[1])
plt.show()


# In[8]:


fig, axarr  = plt.subplots(2,figsize=(10,10))
sns.countplot(x='GradeID', data=data, order=['G-02', 'G-04', 'G-05', 'G-06', 'G-07', 'G-08', 'G-09', 'G-10', 'G-11', 'G-12'], ax=axarr[0])
sns.countplot(x='GradeID', hue='Class', data=data, order=['G-02', 'G-04', 'G-05', 'G-06', 'G-07', 'G-08', 'G-09', 'G-10', 'G-11', 'G-12'], hue_order = ['L', 'M', 'H'], ax=axarr[1])
plt.show()


# #### Looking at these results, Grades 5, 9, and 10 have very few counts. In addition to that, no 5th grade students pass and no 9th grade students achieve high marks. Perhaps these are outliers?

# In[9]:


#Students in Grade 5
data.loc[data['GradeID'] == 'G-05']


# In[10]:


#Students in Grade 9
data.loc[data['GradeID'] == 'G-09']


# #### After looking at the rows themselves, The grade 5 students appear to have similar data to all other students who did not pass (missed more than 7 days, low numerical values, no school survey, etc.)
# #### And again, after examining the data for the grade 9 students it also looks like what we would likely come to expect for each category.
# #### We will look a bit further at these later.

# In[11]:


fig, axarr  = plt.subplots(2,figsize=(10,10))
sns.countplot(x='SectionID', data=data, order=['A', 'B', 'C'], ax = axarr[0])
sns.countplot(x='SectionID', hue='Class', data=data, order=['A', 'B', 'C'],hue_order = ['L', 'M', 'H'], ax = axarr[1])
plt.show()


# In[12]:


fig, axarr  = plt.subplots(2,figsize=(10,10))
sns.countplot(x='Topic', data=data, ax = axarr[0])
sns.countplot(x='Topic', hue='Class', data=data,hue_order = ['L', 'M', 'H'], ax = axarr[1])
plt.show()


# #### An interesting thing to note is that no Geology students fail. We will look into this in a second.

# In[13]:


fig, axarr  = plt.subplots(2,figsize=(10,10))
sns.countplot(x='Semester', data=data, ax = axarr[0])
sns.countplot(x='Semester', hue='Class', data=data,hue_order = ['L', 'M', 'H'], ax = axarr[1])
plt.show()


# In[14]:


fig, axarr  = plt.subplots(2,figsize=(10,10))
sns.countplot(x='Relation', data=data, ax = axarr[0])
sns.countplot(x='Relation', hue='Class', data=data,hue_order = ['L', 'M', 'H'], ax = axarr[1])
plt.show()


# #### Just looking at this there seems to be a correlation betwen students who have mothers as their primary caregiver and students who are less likely to fail.
# 
# ### Next, we take a look at our measurable data. The recorded number of times a student: Raised their hand, Visited the course's resources, Viewed the online course's Anouncement's page, and Visited the Discussion pages. For easier visual comparison, we plot these together:

# In[15]:


sns.pairplot(data, hue="Class", diag_kind="kde", hue_order = ['L', 'M', 'H'], markers=["o", "s", "D"])
plt.show()


# In[16]:


data.groupby('Topic').median()


# #### Here we can see part of the likely reason why the all of the geology students pass. They have far higher median numerical values than most other courses.

# In[17]:


data.groupby('GradeID').median()


# #### Here, looking at the median data again we can see part of the likely reason why the 5th and 9th grade students performed as they did as well.

# In[18]:


fig, axarr  = plt.subplots(2,figsize=(10,10))
sns.countplot(x='ParentAnsweringSurvey', data=data, order=['Yes', 'No'], ax = axarr[0])
sns.countplot(x='ParentAnsweringSurvey', hue='Class', data=data, order=['Yes', 'No'], hue_order = ['L', 'M', 'H'], ax = axarr[1])
plt.show()


# #### Looking at this graph brings a number of questions regarding the causation of this to mind. Were the paents more likely to answer the survey because their student did well, or did the students performance influence the responses? Unfortunately, like many times,  this is one of the questions that arises while looking at data visualizations that we just don't have access to the answer with the data.

# In[19]:


fig, axarr  = plt.subplots(2,figsize=(10,10))
sns.countplot(x='ParentschoolSatisfaction', data=data, order=['Good', 'Bad'], ax = axarr[0])
sns.countplot(x='ParentschoolSatisfaction', hue='Class', data=data, order=['Good', 'Bad'],hue_order = ['L', 'M', 'H'], ax = axarr[1])
plt.show()


# #### The same kind of causation questions arise when looking at the result of the parent's satisfaction with the school.

# In[20]:


fig, axarr  = plt.subplots(2,figsize=(10,10))
sns.countplot(x='StudentAbsenceDays', data=data, order=['Under-7', 'Above-7'], ax = axarr[0])
sns.countplot(x='StudentAbsenceDays', hue='Class', data=data, order=['Under-7', 'Above-7'],hue_order = ['L', 'M', 'H'], ax = axarr[1])
plt.show()


# #### StudentAbsenceDays seems to have a strong correlation with our Class variable. Very few students who missed more than 7 days managed to achieve high marks and very few students who missed less than 7 days failed their course.
# 
# ## Preprocessing the Data
# #### Our goal with prerocessing is to change our numerical fields that have a value like GradeID to a numerical only value in a way that we preserve that distance in a meningful way. Additionally, we want to assign our three classes to numerical outcomes with a preserved distance. There are a couple of ways to do this. We went with setting L = -1, M = 0, and H = 1. Additionally, you could set each to the middle value of their category on the 100% scale (L = 34.5, M = 79.5, and H = 95). We chose to preserve the distance between the categorical values. Additionally, we decided to scale our numerical fields so that they would be more meaningful when compared together. For this we used scikit learn's built in pre-processing scaling ability.

# In[21]:


# Translate GradeID from categorical to numerical
gradeID_dict = {"G-01" : 1,
                "G-02" : 2,
                "G-03" : 3,
                "G-04" : 4,
                "G-05" : 5,
                "G-06" : 6,
                "G-07" : 7,
                "G-08" : 8,
                "G-09" : 9,
                "G-10" : 10,
                "G-11" : 11,
                "G-12" : 12}

data = data.replace({"GradeID" : gradeID_dict})

class_dict = {"L" : -1,
                "M" : 0,
                "H" : 1}
data = data.replace({"Class" : class_dict})

# Scale numerical fields
data["GradeID"] = preprocessing.scale(data["GradeID"])
data["raisedhands"] = preprocessing.scale(data["raisedhands"])
data["VisITedResources"] = preprocessing.scale(data["VisITedResources"])
data["AnnouncementsView"] = preprocessing.scale(data["AnnouncementsView"])
data["Discussion"] = preprocessing.scale(data["Discussion"])

# Use dummy variables for categorical fields
data = pd.get_dummies(data, columns=["gender",
                                     "NationalITy",
                                     "PlaceofBirth",
                                     "SectionID",
                                     "StageID",
                                     "Topic",
                                     "Semester",
                                     "Relation",
                                     "ParentAnsweringSurvey",
                                     "ParentschoolSatisfaction",
                                     "StudentAbsenceDays"])

# Show preprocessed data
data.head()


# #### One of the primary methods of handling categorical data is to convert fields with many values into binary "dummy" variables. This ensures that our algorithms dont interpret a distance in a column with many possible categories. In our case, a good example of this is our Nationality column. It has 16 possible values and simply converting the values to a number would imply some distance between one or another which isn't something that makes sense in this case. As you can see we go from having 17 columns (16 variables and 1 class) to having 64 (63 variables and 1 class). Creating dummy variables like this can definitely increase the complexity of a problem, but most of them are very sparsely populated. Which becomes important with the ML methods we use.
# 
# #### Now that things have been preprocessed a bit, we can take a look at the correlations between fields.

# In[22]:


corr = data.corr()
corr.iloc[[5]]


# #### A row of the correlation matrix looking only at our Classes and which attributes have a correlation to them. As we can see, StudentAbsenceDays has a strong correlation like we expected as do other columns. One important thing to note the values of columns that were converted into simple binary dummy pairs will always have an 'equal' correlation. (for example gender_F and gender_M)
# #### Listing our 8 highest correlated fields: Visited Resources, Student Absence Days, Raised Hands, Announcement Views, Survey Answered, Relation, Parent Satisfaction, Discussion, Gender, and Semester.
# 

# ## Configuring the Perceptron Classifier

# In[23]:


perc = Perceptron(n_iter=100, eta0=0.1, random_state=15)


# ## Split Data, Train, and Test - Perceptron

# In[24]:


results = []
predMiss = []

for _ in range(1000):
    # Randomly sample our training data
    data_train = data.sample(frac=0.7)
    # train data without label
    data_train_X = data_train.loc[:, lambda x: [l for l in data if l != "Class"]]
    # labels of train data 
    data_train_Y = data_train.loc[:, lambda x: "Class"]

    # The rest is test data
    data_test = data.loc[~data.index.isin(data_train.index)]
    # Test data without label
    data_test_X = data_test.loc[:, lambda x: [l for l in data if l != "Class"]]
    # labels of test data 
    data_test_Y = data_test.loc[:, lambda x: "Class"]

    # Train svm
    perc.fit(data_train_X, data_train_Y)
    predMiss.append((data_test_Y != perc.predict(data_test_X)).sum())
    # Score the mean accuracy on the test data and append results in a list
    results.append(perc.score(data_test_X, data_test_Y))

# Convert results to an array and look at the minimum and the average
predErr = np.hstack(predMiss)
Final = np.hstack(results)
print('Minimum Accuracy Score:   %.8f' % Final[Final.argmin()])
print('Maximum Accuracy Score:   %.8f' % Final[Final.argmax()])
print('Average Accuracy Score:   %.8f' % np.average(Final))
print('Minimum Prediction Misses:   %d' % predErr[predErr.argmin()])
print('Maximum Prediction Misses:   %d' % predErr[predErr.argmax()])
print('Average Prediction Misses:   %.2f' % np.average(predErr))


# ## Configuring  the SVM Classifiers

# In[25]:


# Create the radial basis function kernel version of a Support Vector Machine classifier
rbf_clf = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
              decision_function_shape='ovo', degree=3, gamma='auto', kernel='rbf',
              max_iter=-1, probability=False, random_state=None, shrinking=True,
              tol=0.001, verbose=False)
# Create the linear kernel version of a Support Vector Machine classifier
lin_clf = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
              decision_function_shape='ovo', degree=3, gamma='auto', kernel='linear',
              max_iter=-1, probability=False, random_state=None, shrinking=True,
              tol=0.001, verbose=False)
# Create the polynomial kernel version of a Support Vector Machine classifier
poly_clf = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
              decision_function_shape='ovo', degree=3, gamma='auto', kernel='poly',
              max_iter=-1, probability=False, random_state=None, shrinking=True,
              tol=0.001, verbose=False)
# Create the sigmoid kernel version of a Support Vector Machine classifier
sig_clf = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
              decision_function_shape='ovo', degree=3, gamma='auto', kernel='sigmoid',
              max_iter=-1, probability=False, random_state=None, shrinking=True,
              tol=0.001, verbose=False)


# ##  Split Data, Train, and Test - SVMs

# In[26]:


res_rbf = []
predMiss_rbf = []
res_lin = []
predMiss_lin = []
res_poly = []
predMiss_poly = []
res_sig = []
predMiss_sig = []

for _ in range(1000):
    # Randomly sample our training data
    data_train = data.sample(frac=0.7)
    # train data without label
    data_train_X = data_train.loc[:, lambda x: [l for l in data if l != "Class"]]
    # labels of train data 
    data_train_Y = data_train.loc[:, lambda x: "Class"]

    # The rest is test data
    data_test = data.loc[~data.index.isin(data_train.index)]
    # Test data without label
    data_test_X = data_test.loc[:, lambda x: [l for l in data if l != "Class"]]
    # labels of test data 
    data_test_Y = data_test.loc[:, lambda x: "Class"]

    # Train svms
    rbf_clf.fit(data_train_X, data_train_Y)
    lin_clf.fit(data_train_X, data_train_Y)
    poly_clf.fit(data_train_X, data_train_Y)
    sig_clf.fit(data_train_X, data_train_Y)
    
    #Sum the prediction misses. Since this is a smaller dataset, 
    predMiss_rbf.append((data_test_Y != rbf_clf.predict(data_test_X)).sum())
    predMiss_lin.append((data_test_Y != lin_clf.predict(data_test_X)).sum())
    predMiss_poly.append((data_test_Y != poly_clf.predict(data_test_X)).sum())
    predMiss_sig.append((data_test_Y != sig_clf.predict(data_test_X)).sum())
    # Score the mean accuracy on the test data and append results in a list
    res_rbf.append(rbf_clf.score(data_test_X, data_test_Y))
    res_lin.append(lin_clf.score(data_test_X, data_test_Y))
    res_poly.append(poly_clf.score(data_test_X, data_test_Y))
    res_sig.append(sig_clf.score(data_test_X, data_test_Y))

# Convert results and prediction lists to an array and look at the minimums and the averages
predErr_rbf = np.hstack(predMiss_rbf)
Final_rbf = np.hstack(res_rbf)
predErr_lin = np.hstack(predMiss_lin)
Final_lin = np.hstack(res_lin)
predErr_poly = np.hstack(predMiss_poly)
Final_poly = np.hstack(res_poly)
predErr_sig = np.hstack(predMiss_sig)
Final_sig = np.hstack(res_sig)


print('RBF Minimum Accuracy Score:   %.8f' % Final_rbf[Final_rbf.argmin()])
print('RBF Maximum Accuracy Score:   %.8f' % Final_rbf[Final_rbf.argmax()])
print('RBF Average Accuracy Score:   %.8f' % np.average(Final_rbf))
print('------------------------------------------------')
print('Linear Minimum Accuracy Score:   %.8f' % Final_lin[Final_lin.argmin()])
print('Linear Maximum Accuracy Score:   %.8f' % Final_lin[Final_lin.argmax()])
print('Linear Average Accuracy Score:   %.8f' % np.average(Final_lin))
print('------------------------------------------------')
print('Polynomial Minimum Accuracy Score:   %.8f' % Final_poly[Final_poly.argmin()])
print('Polynomial Maximum Accuracy Score:   %.8f' % Final_poly[Final_poly.argmax()])
print('Polynomial Average Accuracy Score:   %.8f' % np.average(Final_poly))
print('------------------------------------------------')
print('Sigmoid Minimum Accuracy Score:   %.8f' % Final_sig[Final_sig.argmin()])
print('Sigmoid Maximum Accuracy Score:   %.8f' % Final_sig[Final_sig.argmax()])
print('Sigmoid Average Accuracy Score:   %.8f' % np.average(Final_sig))
print('================================================')
#print('Minimum Prediction Misses:   %d' % predErr[predErr.argmin()])
#print('Maximum Prediction Misses:   %d' % predErr[predErr.argmax()])
print('RBF Average Prediction Misses:   %.2f' % np.average(predErr_rbf))
print('Linear Average Prediction Misses:   %.2f' % np.average(predErr_lin))
print('Polynomial Average Prediction Misses:   %.2f' % np.average(predErr_poly))
print('Sigmoid Average Prediction Misses:   %.2f' % np.average(predErr_sig))


# ## Configuring the Decision Tree Classifiers

# In[27]:


tree3 = DecisionTreeClassifier(random_state=56, criterion='gini', max_depth=3)
tree5 = DecisionTreeClassifier(random_state=56, criterion='gini', max_depth=5)


# ##  Split Data, Train, and Test - Decision Trees

# In[30]:


results_3 = []
results_5 = []
predMiss_3 = []
predMiss_5 = []


for _ in range(1000):
    # Randomly sample our training data
    data_train = data.sample(frac=0.7)
    # train data without label
    data_train_X = data_train.loc[:, lambda x: [l for l in data if l != "Class"]]
    # labels of train data 
    data_train_Y = data_train.loc[:, lambda x: "Class"]

    # The rest is test data
    data_test = data.loc[~data.index.isin(data_train.index)]
    # Test data without label
    data_test_X = data_test.loc[:, lambda x: [l for l in data if l != "Class"]]
    # labels of test data 
    data_test_Y = data_test.loc[:, lambda x: "Class"]

    # Train svm
    tree3.fit(data_train_X, data_train_Y)
    tree5.fit(data_train_X, data_train_Y)
    #Sum the prediction misses. Since this is a smaller dataset,
    predMiss_3.append((data_test_Y != tree3.predict(data_test_X)).sum())
    predMiss_5.append((data_test_Y != tree5.predict(data_test_X)).sum())
    # Score the mean accuracy on the test data and append results in a list
    results_3.append(tree3.score(data_test_X, data_test_Y))
    results_5.append(tree5.score(data_test_X, data_test_Y))

# Convert results to an array and look at the minimum and the average
predErr_3 = np.hstack(predMiss_3)
predErr_5 = np.hstack(predMiss_5)
Final_3 = np.hstack(results_3)
Final_5 = np.hstack(results_5)
print('3-depth Tree Minimum Accuracy Score:   %.8f' % Final_3[Final_3.argmin()])
print('3-depth Tree Maximum Accuracy Score:   %.8f' % Final_3[Final_3.argmax()])
print('3-depth Tree Average Accuracy Score:   %.8f' % np.average(Final_3))
print('------------------------------------------------')
print('5-depth Tree Minimum Accuracy Score:   %.8f' % Final_5[Final_5.argmin()])
print('5-depth Tree Maximum Accuracy Score:   %.8f' % Final_5[Final_5.argmax()])
print('5-depth Tree Average Accuracy Score:   %.8f' % np.average(Final_5))
#print('Minimum Prediction Misses:   %d' % predErr[predErr.argmin()])
#print('Maximum Prediction Misses:   %d' % predErr[predErr.argmax()])
#print('Average Prediction Misses:   %.2f' % np.average(predErr))


# ## Final results
# <table>
#     <tr>
#     <th>Algorithm</th>
#     <td>Perceptron</td>
#     <td>SVM (rbf)</td>
#     <td>SVM (linear)</td>
#     <td>SVM (polynomial (1))</td>
#     <td>SVM (polynomial (2))</td>
#     <td>SVM (polynomial (3))</td>
#     <td>SVM (polynomial (5))</td>
#     <td>SVM (sigmoid)</td>
#     <td>Random Forest (depth = 3)</td>
#     <td>Random Forest (depth = 5)</td>
#     </tr>
#     <tr>
#     <th>Average Accuracy</th>
#     <td>0.64736806</td>
#     <td>0.74331250</td>
#     <td>0.75625000</td>
#     <td>0.73275000</td>
#     <td>0.60676389</td>
#     <td>0.43888194</td>
#     <td>0.43865278</td>
#     <td>0.72772222</td>
#     <td>0.68082639</td>
#     <td>0.71702083</td>
#     </tr>
#     </table>
# #### As we can see from the table, a SVM with a linear kernel actually ends up handling the data the best with a 75.62% accuracy.

# In[ ]:




