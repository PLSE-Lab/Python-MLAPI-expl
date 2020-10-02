#!/usr/bin/env python
# coding: utf-8

# # Heartbeat classification from ECG morphology using Machine learning.
# 
# ## Motivation
# 
# Acording to [Wikipedia](https://en.wikipedia.org/wiki/Heart_arrhythmia) 
# Arrhythmia affects millions of people in the world. In Europe and North America, as of 2014, atrial fibrillation affects about 2% to 3% of the population. Atrial fibrillation and atrial flutter resulted in 112,000 deaths in 2013, up from 29,000 in 1990. Sudden cardiac death is the cause of about half of deaths due to cardiovascular disease and about 15% of all deaths globally. About 80% of sudden cardiac death is the result of ventricular arrhythmias. Arrhythmias may occur at any age but are more common among older people. Arrhythmias are coused by problems with the electrical conduction system of the heart. A number of tests can help with diagnosis including an electrocardiogram (ECG) and Holter monitor. Regarding ECG, the diagnosis is based on the carefully analysis that a specialized doctor perform on the shape and structure of the independent heartbeats. This process is tedious and requires time. 
# ![ecg](./pics/ecg.png)
# 
# In this work, we aim to classify the heart beats extracted from an ECG using machine learning, based only on the lineshape (morphology) of the individual heartbeats. The goal would be to develop a method that automatically detects anomallies and help for the prompt diagnosis of arrythmia.
# 
# ## Data
# 
# The original data comes from the [MIT-BIH Arrythmia database](https://physionet.org/content/mitdb/1.0.0/). Some details of the dataset are briefly summarized below:
# 
# + 48.5 hour excerpts of two-channel ambulatory ECG recordings
# + 48 subjects studied by the BIH Arrhythmia Laboratory between 1975 and 1979.
# + 23 recordings randomly selected from a set of 4000 24-hour ambulatory ECG recordings collected from a mixed at Boston's Beth Israel Hospital.
# + 25 recordings were selected from the same set to include less common but clinically significant arrhythmias.
# + Two or more cardiologists independently annotated each record (approximately 110,000 annotations in all).
# 
# Although the one that its currently being used here is taken from [kaggle](https://www.kaggle.com/alexandrefarb/mitbih-arrhythmia-database-de-chazal-class-labels). In this dataset the single heartbeats from the ECG were extracted using the [Pam-Tompkins algorithm](https://en.wikipedia.org/wiki/Pan-Tompkins_algorithm). Each row of the dataset represents a QRS complex as the one schematically shown below:
# <img src="./pics/qrs.png" width="300">
# These QRS are taken from the MLII lead from the ECG. As observed in the firts figure above, there is also the V1 lead, which is not used in this work.
# For further details on how the data was generated, the interested can read the original paper by [Chazal et al.](https://www.ncbi.nlm.nih.gov/pubmed/15248536)
# 
# The different arrythmia classes are:
# 
# 0. Normal
# 1. Supraventricular ectopic beat
# 2. Ventricular ectopic beat
# 3. Fusion Beat
# 
# This means that we are facing a multi-class classification problem with four classes. 
# 
# ## Strategy.
# 
# To achieve our goal we will go the following way:
# 
# 1. Data standardisation
# 2. Selection of three promising ML algorithms.
# 3. Fine tunning of the best models
# 4. Model comparison
# 5. Build  a Neural Network
# 6. Compare 
# 
# Rather important, in order to evaluate the performance of our models is to choose the appropiate metrics. In this case we will be checking the confusion matrix and the f1 score with macro averaging.
# 
# 
# ## Data Loading and first insights.
# 

# In[ ]:


import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import VotingClassifier

import seaborn as sn

hbeat_signals = pd.read_csv("../input/mitbih-arrhythmia-database-de-chazal-class-labels/DS1_signals.csv", header=None)
hbeat_labels = pd.read_csv("../input/mitbih-arrhythmia-database-de-chazal-class-labels//DS1_labels.csv", header=None)

print("+"*50)
print("Signals Info:")
print("+"*50)
print(hbeat_signals.info())
print("+"*50)
print("Labels Info:")
print("+"*50)
print(hbeat_labels.info())
print("+"*50)


# In[ ]:


hbeat_signals.head()


# Let's have a look at how the data looks like for the different types of heartbeats.

# In[ ]:


# Collect data of different hheartbeats in different lists
#class 0
cl_0_idx = hbeat_labels[hbeat_labels[0] == 0].index.values
cl_N = hbeat_signals.iloc[cl_0_idx]
#class 1
cl_1_idx = hbeat_labels[hbeat_labels[0] == 1].index.values
cl_S = hbeat_signals.iloc[cl_1_idx]
#class 2
cl_2_idx = hbeat_labels[hbeat_labels[0] == 2].index.values
cl_V = hbeat_signals.iloc[cl_2_idx]
#class 3
cl_3_idx = hbeat_labels[hbeat_labels[0] == 3].index.values
cl_F = hbeat_signals.iloc[cl_3_idx]

# make plots for the different hbeat classes
plt.subplot(221)
for n in range(3):
    cl_N.iloc[n].plot(title='Class N (0)', figsize=(10,8))
plt.subplot(222)
for n in range(3):
    cl_S.iloc[n].plot(title='Class S (1)')
plt.subplot(223)
for n in range(3):
    cl_V.iloc[n].plot(title='Class V (2)')
plt.subplot(224)
for n in range(3):
    cl_F.iloc[n].plot(title='Class F (3)')


# In[ ]:


#check if missing data
print("Column\tNr of NaN's")
print('+'*50)
for col in hbeat_signals.columns:
    if hbeat_signals[col].isnull().sum() > 0:
        print(col, hbeat_signals[col].isnull().sum()) 


# This means that there are no missing values to fill. We can now proceed to check if there are some correlations on the data.

# In[ ]:


joined_data = hbeat_signals.join(hbeat_labels, rsuffix="_signals", lsuffix="_labels")

#rename columns
joined_data.columns = [i for i in range(180)]+['class']


# In[ ]:


#get correlaction matrix
corr_matrix = joined_data.corr()


# In[ ]:


print('+'*50)
print('Top 10 high positively correlated features')
print('+'*50)
print(corr_matrix['class'].sort_values(ascending=False).head(10))
print('+'*50)
print('Top 10 high negatively correlated features')
print('+'*50)
print(corr_matrix['class'].sort_values().head(10))


# Let's plot some of these features to see how they correlate.

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')

from pandas.plotting import scatter_matrix

#Take features with the larges correlations
features = [79,80,78,77]
scatter_matrix(joined_data[features], figsize=(20,15), c =joined_data['class'], alpha=0.5);


# The correlation among the selected features is strongly linear. 
# 
# We now check if the data is balanced, this means, if there is a balenced amount of data for the diferent classes we want to study.

# In[ ]:


print('-'*20)
print('Class\t %')
print('-'*20)
print(joined_data['class'].value_counts()/len(joined_data))
joined_data.hist('class');
print('-'*20)


# The above plot shows that the dataset is quite unbalanced. There is very few data for classes 1, 2 and 3, whereas class 1 is about 90% of the total dataset. 
# 
# Now since the data is rather clean and without missing values we can start preparing our data to do some machine learning.
# 
# # Machine learning 
# 
# We now first produce some test and train data from our dataset. Be carefull that there is not many data for instances with classes 1 to 3. This means that we have to split the data not randomly, but trying to keep data with these clases in the train and test data. Hence, we will perform a stratified data split.
# 
# For instance, lets check, once more, which percentage of data of each class we have in our data set.

# In[ ]:


print("class\t%")
joined_data['class'].value_counts()/len(joined_data)


# In[ ]:


from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2,random_state=42)

for train_index, test_index in split.split(joined_data, joined_data['class']):
    strat_train_set = joined_data.loc[train_index]
    strat_test_set = joined_data.loc[test_index]    


# Let's now check if the train data fulfills the stratified conditions after the split was done.

# In[ ]:


print("class\t%")
strat_train_set['class'].value_counts()/len(strat_train_set)


# Nice, we see that the amount of data with classes 0 to 3 in the train set maps to those from the original data.
# 
# We are ready to pick some ML models to start training with our data.
# We will use a brute force approach in the sense that we will try several models at once. For each model, we will do a 5-fold cross validation and depending on its metrics we will choose the best among them. To do that we will write a simple function that takes a list of models, and perfom the cross validation for each and prints its metrics, i.e., confusion matrix, precission, recall and f1 score.

# In[ ]:


def compare_conf_matrix_scores(models, X, y):
    """
    This function compares predictive scores and confusion matrices fro different ML algorithms
    """
    
    for i, model in enumerate(models):

        # perform Kfold cross-validation returning prediction scores of each test fold.
        labels_train_pred = cross_val_predict(model, X, y, cv=5)
        print('+'*50)
        print('Model {} Confusion matrix'.format(i+1))
        print('+'*50)
        print(confusion_matrix(y, labels_train_pred))
        print('+'*50)

        prec_score = precision_score(y, labels_train_pred, average='macro')
        rec_score = recall_score(y, labels_train_pred, average='macro')
        f1_sc = f1_score(y, labels_train_pred, average='macro')
        print('Precision score: {}\nRecall Score: {}\nf1 score: {}'.format(prec_score,rec_score, f1_sc))
    print('+'*50)
    
#produce labels and features sets for the training stage
strat_features_train = strat_train_set.drop('class', 1)
strat_labels_train = strat_train_set['class']


# We will firts start with the following models:
# 
# 1. One vs. One classifier (OVO)
# 2. Random Forest
# 3. linear Support Vector Machine 
# 4. Support Vector Machine (SVM)
# 
# Note that we are at the moment not doing any standardization. We will do that after this step, just to compare if there is an improvement by standardizing the data.

# In[ ]:


#initiate ML the classifiers

# one versus one clasifier
ova_clf = OneVsOneClassifier(SGDClassifier(random_state=42, n_jobs=-1))

#random forest
forest_clf = RandomForestClassifier(random_state=42, n_jobs=-1)

#Support vector machines
svm_clf = LinearSVC(random_state=42)
svc = SVC(decision_function_shape='ovo', random_state=42, max_iter=1000)

warnings.filterwarnings('ignore')

compare_conf_matrix_scores([ova_clf, forest_clf, svm_clf, svc], strat_features_train, strat_labels_train)


# The confusion matrix clearly tell us that these model are not classifying the data correctly. For instance, for model 1, we can see that class 3 has not been correctly classified at all, whereas class 1 is poorly classified (25 from 181 were correctly classified). The precision and recall scores clearly tell us that our model just classifies correctly about 40% of the instances for classes other than N (0). 
# 
# The most promissing model is model 2, i.e., Random Forest. It has the best scores and its confusion matrix eshibits reasonable predictions.
# 
# Let us now try what happends when we standardize the features and try again with these models.

# In[ ]:


#initialize standardscaler instance
scaler = StandardScaler()

#standarized data, i.e,  substract mean and devides by variance
std_features = scaler.fit_transform(strat_features_train)


# Before fitting models, lets have a look how does the data look like after standardization. The following figure shows the same plots as the one we did at the begining, but using the data after using StandardScaler.

# In[ ]:


# make plots for the different hbeat classes (standarized)
fig = plt.figure(figsize=(10,8))
plt.subplot(221)
plt.plot(figsize=(10,8))
x = np.linspace(0,179,180)
for n in range(3):
    plt.title('Class N (0)')
    plt.plot(x, std_features[cl_0_idx[n]])
plt.subplot(222)
for n in range(3):
    plt.title('Class S (1)')
    plt.plot(x, std_features[cl_1_idx[n]])
plt.subplot(223)
for n in range(3):
    plt.title('Class V (2)')
    plt.plot(x, std_features[cl_2_idx[n]])
plt.subplot(224)
for n in range(3):
    plt.title('Class F (3)')
    plt.plot(x, std_features[cl_3_idx[n]])


# After standardization it makes also sense to have a look at some of the features to see if we can observe some clustering, patterns or characteristics of the data dependending on the classes.

# In[ ]:


fig= plt.figure(figsize=(25,15))
for n in range(9):
    plt.subplot(3,3,n+1)
    scatter = plt.scatter(std_features[:,(n+1)*10],std_features[:,-1*(n+1)*5], alpha=0.5, c=strat_labels_train)
    plt.xlabel('Feat. {}'.format((n+1)*10))
    plt.ylabel('Feat. {}'.format(180-1*(n+1)*5))
plt.rc('font', size=20)
plt.rc('legend', fontsize=20)
#plt.rc('axes', labelsize=40)
#plt.legend(*scatter.legend_elements(), loc="best", title="Classes");


# Apparently, as observed from these fplots, there are some sort of clustering for the 0 and 2 classes. also class 4 seem to show a particular pattern. This is good, since it means that in principle a good ML algorithm might learn something from these features.
# 
# Let us now check again our firts four models with the standardized data and see if they improve.

# In[ ]:


warnings.filterwarnings('ignore')

compare_conf_matrix_scores([ova_clf, forest_clf, svm_clf, svc], std_features, strat_labels_train)


# So the standard scaler improves the ovo_clf and the Linear-SVM. The SVC model also has an important improvemnet. In the case of the random forest it does not change much, however,  random forest is still the best model so far. Lets try another couple of models, for instance:
# 
# 1. k-Nearest Neighbors (KNN)
# 2. Gaussian Naive Bayes (GNB)
# 3. Stochastic Gradient Descent (SGD)

# In[ ]:


# K nearest neighbors
knn_clf = KNeighborsClassifier(n_jobs=-1)

#Gaussian Naive Bayes
gnb_clf = GaussianNB()

#Stochastic gradient classifier
sgd_clf = SGDClassifier(n_jobs=-1,random_state=42)

compare_conf_matrix_scores([knn_clf, gnb_clf, sgd_clf], std_features, strat_labels_train)


# It seem that our KNN model performs reasonably good with a f1 score of 0.89, which is similar to the one we got with the randome forest classifier (0.83). Actually, the KNN model does a much better job in classifying instances of class 1 and 3. Also the SVM model is a promissing one with a  0.78 f1 score. Therrfore we will choose **Random Forest**, **KNN** and **SVM** as the models to use to do our classification task.
# 
# Let us now try to fine tune these three models and see if we can improve their performances.
# 
# ## Model fine tunning via GridSearch cross validation.
# 
# Since SVM model takes much more time to optimze via grid searchcv, we first start with the random forest and KNN models.

# In[ ]:


#parameter grid
forest_param_grid = {'n_estimators': [50,100,200,300], 'max_depth':[2,4,8]}
knn_param_grid = {'n_neighbors':[2,4,8,10], 'weights':['uniform', 'distance']}

warnings.filterwarnings('ignore')

#initialize classifiers
forest = RandomForestClassifier(random_state=42, n_jobs=-1)
knn = KNeighborsClassifier(n_jobs=-1)

#initialize grid search
forest_grid_search = GridSearchCV(forest, forest_param_grid, cv=5, scoring="f1_macro")
knn_grid_search = GridSearchCV(knn, knn_param_grid, cv=5, scoring="f1_macro")

#fit classifiers using gridsearch
forest_grid_search.fit(std_features, strat_labels_train)
knn_grid_search.fit(std_features, strat_labels_train)

print("+"*50)
print("Model\t\tBest params\t\tBest score")
print("-"*50)
print("Random Forest\t\t", forest_grid_search.best_params_, forest_grid_search.best_score_)
print("-"*50)
print("KNN\t\t", knn_grid_search.best_params_, knn_grid_search.best_score_)
print("+"*50)


# Further fine tunning of the Random forest model.

# In[ ]:


forest_param_grid = {'n_estimators': [158,160,162], 'max_depth':[83,80,87]}
forest_grid_search = GridSearchCV(forest, forest_param_grid, cv=5, scoring="f1_macro")
forest_grid_search.fit(std_features, strat_labels_train)

print("+"*50)
print('Model\t\tBest params\t\tBest score')
print("-"*50)
print("Random Forest\t\t", forest_grid_search.best_params_, forest_grid_search.best_score_)
print("+"*50)


# Lets fine tune the SVM model.

# In[ ]:


#parameter grid
svc_param_grid = {'C':[10], 'gamma':[0.1,1,10]}

warnings.filterwarnings('ignore')

#initialize classifiers
svc = SVC(kernel='rbf',decision_function_shape='ovo',random_state=42, max_iter = 500)

#initialize grid search
svc_grid_search = GridSearchCV(svc, svc_param_grid, cv=3, scoring="f1_macro")

#fit classifiers using gridsearch
svc_grid_search.fit(std_features, strat_labels_train)

print("+"*50)
print('Model\t\tBest params\t\tBest score')
print("-"*50)
print("SVC\t\t", svc_grid_search.best_params_, svc_grid_search.best_score_)
print("+"*50)


# After some fine tuning of the random forest we have increase its efficiency from 0.82 to 0.85. Also the KNN model improved from 0.89 to 0.9, whereas the SVM model improved from 0.78 to 0.91. Let us now check the confusion matrix from these improved models.

# In[ ]:


best_forest = forest_grid_search.best_estimator_
best_knn = knn_grid_search.best_estimator_
best_svc = svc_grid_search.best_estimator_

compare_conf_matrix_scores([best_forest, best_knn, best_svc], std_features, strat_labels_train)


# Nice, now we can try these models with our test data. but first we have to standardize the test data using the same scaler we used for the train data.

# In[ ]:


#init scaler
scaler = StandardScaler()

#fit scaler to train data
scaler.fit(strat_features_train)

#produce labels and features sets for the test stage
strat_features_test = strat_test_set.drop('class', 1)
strat_labels_test = strat_test_set['class']

#transform the test data
std_features_test = scaler.transform(strat_features_test)

#predict values for the test data
forest_pred = best_forest.predict(std_features_test)
knn_pred = best_knn.predict(std_features_test)
svc_pred = best_svc.predict(std_features_test)

#determine f1 score
forest_f1 = f1_score(strat_labels_test, forest_pred, average='macro')
knn_f1 = f1_score(strat_labels_test, knn_pred, average='macro')
svc_f1 = f1_score(strat_labels_test, svc_pred, average='macro')

#determine confusion matrix
print('+'*50)
print('Random Forest Confusion matrix (f1 score: {})'.format(forest_f1))
print('+'*50)
print(confusion_matrix(strat_labels_test, forest_pred))
print('+'*50)
print('KNN Confusion matrix (f1 score: {})'.format(knn_f1))
print('+'*50)
print(confusion_matrix(strat_labels_test, knn_pred))
print('+'*50)
print('SVC Confusion matrix (f1 score: {})'.format(svc_f1))
print('+'*50)
print(confusion_matrix(strat_labels_test, svc_pred))


# Although not perfect, the models do a pretty decent job to classify the different classes of heartbeats.
# 
# ## Ensemble model
# 
# One last thing we can try is to merge these three models using assemble in order to maximize the classification power. This we do by using a Voting classifier and set the voting Hyperparameter to 'hard'. This basically means that, for a given input, each model will give classification, i.e., 0, 1, 2 or 3 and the classification with the higher votes wins.

# In[ ]:


#initialize ensemble
ensemble=VotingClassifier(estimators=[('Random Forest', best_forest), ('KNN', best_knn), ('SVC', best_svc)], voting='hard')

#fit ensemble
ensemble.fit(std_features,strat_labels_train)

compare_conf_matrix_scores([ensemble], std_features, strat_labels_train)


# The ensemble classifier performs better that KNN and random forest, although SVM still does a better job by its own. Lets check it with test data.

# In[ ]:


#predict values for the test data
ensemble_pred = ensemble.predict(std_features_test)

#determine f1 score
ensemble_f1 = f1_score(strat_labels_test, ensemble_pred, average='macro')

#determine confusion matrix
print('+'*50)
print('Ensemble Confusion matrix (f1 score: {})'.format(ensemble_f1))
print('+'*50)
print(confusion_matrix(strat_labels_test, ensemble_pred))
print('+'*50)


# This esemble model in particular seems not to perform better than the SVM model for instance. Although they are really close.
# 
# Just to have a closer look on the differences of these four models, let us make plots on how these models draw their decision boundaries. 

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt

h = 0.155 # step size in the mesh

# we create an instance Classifier and fit the data. We are picking features 70 and 145.
X, y = std_features[:,[70,145]], strat_labels_train

#initialize classifiers
clf1 = RandomForestClassifier(max_depth=83,n_estimators=158, n_jobs=-1,random_state=42)
clf2 = KNeighborsClassifier(n_jobs=-1, n_neighbors=4,weights='distance')
clf3 = SVC(C=10,decision_function_shape='ovo', gamma=0.1, kernel='rbf', max_iter=500, random_state=42)
clf4 = ensemble=VotingClassifier(estimators=[('Random Forest', clf1), ('KNN', clf2), ('SVC', clf3)], voting='hard')
#fit classifiers
clf1.fit(X,y)
clf2.fit(X,y)
clf3.fit(X,y)
clf4.fit(X,y)

# Plot the decision boundary. For that, we will assign a color to each point in the mesh [x_min, x_max]x[y_min, y_max].
x_min, x_max = X[:, 0].min(), X[:, 0].max()
y_min, y_max = X[:, 1].min(), X[:, 1].max()
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

#fig titles
tt =['Random Forest (depth=83)', 'KNN (k=4)','Kernel (RBF) SVM', 'Hard Voting']

fig= plt.figure(figsize=(20,15))

for idx, clf in enumerate([clf1, clf2, clf3, clf4]):

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    #plot decision boundary
    plt.subplot(2,2,idx+1)
    plt.pcolormesh(xx, yy, Z, alpha=0.2)

    # Plot training points
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.5)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xlabel('Feat. 70')
    plt.ylabel('Feat. 145')
    plt.title(tt[idx])
#    plt.legend(*scatter.legend_elements(), loc="best", title="Classes");    


# ## Neural network
# 
# To finish, we will now train an Artifitial Neural Network (ANN) using TensorFlow and Keras. We will then build a ANN with an input layer, two hidden layers, one with 200 and the other one with 100 neurones. And an output layer with four neurons, one for each class. For the Hidden layers we will use a ReLu activation funtion, whereas for the output layer we will use a softmax function. 

# In[ ]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD

#define parameters
batch_size = len(std_features)//300

#build model
model = keras.Sequential([
    keras.layers.Dense(200, activation='relu', input_shape=(180,)),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(4, activation='softmax')
])

#transform the test data
strat_features_test = strat_test_set.drop('class', 1)
strat_labels_test = strat_test_set['class']

#standardize 
std_features_test = scaler.transform(strat_features_test)

#change labels to categorical, requaried to use 'categorical_crossentropy'
categorical_labels_train = to_categorical(strat_labels_train, num_classes=None)
categorical_labels_test = to_categorical(strat_labels_test, num_classes=None)

#stochastic gradient descent optimizer
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

#compile model
model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])

#fit model and get scores
model.fit(std_features, categorical_labels_train,epochs=20,batch_size=batch_size)

score = model.evaluate(std_features_test, categorical_labels_test, batch_size=batch_size)


# In[ ]:


# check metrics on the test data
y_pred1 = model.predict(std_features_test)
y_pred = np.argmax(y_pred1, axis=1)

f1_sc = f1_score(strat_labels_test, y_pred , average="macro")
conf_mat = confusion_matrix(strat_labels_test, y_pred)

# Print f1, precision, and recall scores
print('+'*50)
print('Neural Network Confusion matrix (f1 score: {})'.format(f1_sc))
print('+'*50)
print(conf_mat)
print('+'*50)


# ## Conclusions and outlook
# 
# We have perform a multi-class classification task on a dataset consisting of ECG heartbeats, we have mapped several classifier models and selected the three among seven which showed better classification performance as using n-fold cross validation. These three models were further improved by optimizing some of their hyperparameters via GridSearch cross validation. From these procedure we have found that the SVM model was the one that performed the best with a f1 score of 0.92. 
# 
# In order to see if we can do better, we have merged these three models into one ensemble model using hard voting classifier. However, this did not outperform the SVM model. Finally, we built and trained an ANN which showed a very nice performance, however, still below that of the SVM model.
# 
# Although, the work here done looks rather promising we have to consider that the dataset is highly unbalanced. There is not enough data for classes 1 to 3. It would be good to have more representation of these classes in the dataset, since this will increase the classification performance of the models. One way to solve this problem would be to try some data augmentation procedure. For instance, by determining the principal components of these classes and with these randomly generate data that can be used. Also, the use of the V1 data from the ECG in order to increase the amount of features would help, together with introduction of other important features like area under the QRS curve, peak maxima and minim, inflection points, etc.
# 

# In[ ]:


#best knn model n_neighbors = 4, weights='distance', n_jobs=-1
#best random forest max_depth=83, n_estimators=158, n_jobs=-1
best_knn, best_forest, best_svc

