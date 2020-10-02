#!/usr/bin/env python
# coding: utf-8

# **<h2>AMAZON Fine Food Reviews Data & Sentimal Analysis</h2>
# <h2>Exploratory Data Analysis</h2>
# <h4>Sentiment Analysis </h4>
# <h5>Using Naive Bayes</h5>
# <h5>Logestic Regression by default using l2 Regularizer</h5>
# <h5>Logestic Regression by using l1 Regularizer</h5>
# <h5>Examining features of Data by Using Feature Engineering</h5>
# <h5> Linear SVM Using Hinge loss</h5>
# <h5>calculate Metrics(Confusion Metrics,Precision, Recall, F1 Score,Log loss, Accuracy) for each model</h5>
# <h5>Cross Validation and Grid Search for finding optimal regularizer in logistic Regression</h5>
# <h5>K-Fold Cross Validation For logistic Regression</h5>
# <h5>Comparing all the Models</h5>

# In[ ]:


# Import Data Set 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split

data = pd.read_csv("../input/Reviews.csv")

df= pd.DataFrame(data)
data.head()


# In[ ]:


df.columns


# In[ ]:


# Remove reviews of  Rating=3
df1 = df[df.Score != 3]
(df1.shape)


# In[ ]:


# Converting Rating to Binary class i.e high rating=1, low rating=0
df1['Score']=df1['Score'].apply(lambda x: 1 if (x > 3)  else 0)
df1.head()


# <h1> a.) Positive Reviews are Common </h1>
# <h4> 84% data belongs to positive rating class and 15.6% Reviews are corresponds to negative class rating</h4>

# In[ ]:


# Count Reviews belonged to High Rating and Low Rating
df2= df1.groupby('Score').size()
df2


# In[ ]:


# Calculate Percentage of High Rating and low Rating
per= df2/sum(df2)*100
per


# In[ ]:


# plot Histogram between percentage of rating 
df2.plot(kind='bar',title='Label Distribution')
plt.xlabel('rating')
plt.ylabel('values')
# plt.legend()
plt.show()


# <h2>b.)positive reviews are shorter in length</h2>

# In[ ]:


# Calculate Length of High rating Review V/S Low Level Review
df3=df.iloc[:,[6,8]]
df3 = df[df.Score != 3]
df3['Score']=df3['Score'].apply(lambda x: 1 if (x > 3)  else 0)
df3=df3.iloc[:,[6,8]]
df3['Length'] = df3['Summary'].str.len()
df3=df3.iloc[:,[0,2]]
df3=df3.groupby('Score')['Length'].mean()
df3.head()


# In[ ]:


# Plot the Graph between Length of High rating Review V/S Low Level Review
df3.plot(kind='bar',color='g',title='positive Reviews have shorter length')
plt.xlabel('rating')
plt.ylabel('avg length of Summary')
# plt.legend()
plt.show()


# <h4>c.) Longer reviews are more helpful, we consider review is helpul if  are given by atleast one person, otherwise Excluded </h4>

# In[ ]:


# Calculate Ratio to find Reviews are Helpfull or not corresponding to Length of the review 
df = df[df.Score != 3]
df['Score']=df['Score'].apply(lambda x: 1 if (x > 3)  else 0)
df['Length'] = df['Summary'].str.len()
df4= df[df.HelpfulnessDenominator!=0]
df4['ratio'] = df4['HelpfulnessNumerator']/df4['HelpfulnessDenominator']
df4['ratio']=df4['ratio'].apply(lambda x: 0 if (x < 0.5)  else 1)
df4=df4.groupby('ratio')['Length'].mean()
df4.head()


# In[ ]:


# Plot Histogram between Helpfulness ratio and Length of the Summary
df4.plot(kind='bar',color='g',title='longer reviews are helpul')
plt.xlabel('ratio')
plt.ylabel('avg length of Summary')
# plt.legend()
plt.show()


# <h4>d.) Despite being more common and shorter, positive reviews are found more helpful.
# that is if review is positive rated then it is also helpfull, count those Rows </h4>

# In[ ]:


# Calculate Review is high rated and helpul are more than low rated length Reviews
df5= df1[df1.HelpfulnessDenominator!=0]
df5['ratio'] = df5['HelpfulnessNumerator']/df5['HelpfulnessDenominator']
df5['ratio']=df5['ratio'].apply(lambda x: 0 if (x < 0.5)  else 1)
df5['que'] = df5.apply(lambda x : 1 if ((x['Score'] and x['ratio']) ==1) else 0, axis=1)
# print df5.head(3)
df5= df5.groupby('que').size()
df5


# In[ ]:


# Plot to show that Positive Reviews are Longer in length
df5.plot(kind='bar',color=['g','r'],title='longer reviews are helpul')
plt.xlabel('Rating')
plt.ylabel('Helpullness')
# plt.legend()
plt.show()


# In[ ]:


#take only three required columns(Score,Summary, Text)
df6=df.iloc[:,[6,8,9]]
df7 = df6[df.Score != 3]
df7['Score']=df7['Score'].apply(lambda x: 1 if (x > 3)  else 0)
df7.Score.value_counts()


# In[ ]:


# for model building consider 1,2,3 star rating as 0(Low Rating) and 4,5 included as 1(High  Rating)
df8 = data[pd.notnull(data.Summary)]
df8['Score']=df8['Score'].apply(lambda x: 1 if (x > 3)  else 0)
print (df8.shape)
df8.head()


# In[ ]:


# how to define X and y (from the Review  data) for use with COUNTVECTORIZER
X = df8.Summary+df8.Text
y = df8.Score
print(X.shape)
print(y.shape)


# In[ ]:


# split X and y into training and testing sets
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# <h3> Vectorizing DataSet</h3>

# In[ ]:


# Create instantiation for CounterVectorizer
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer()


# In[ ]:


X_train_dtm = vect.fit_transform(X_train)
X_train_dtm
print(type(X_train_dtm))
print(X_train_dtm.shape)
print(X_train_dtm[10,:])


# In[ ]:


# transform testing data (using fitted vocabulary) into a document-term matrix
X_test_dtm = vect.transform(X_test)
X_test_dtm


# <h3>Model Building Using Naive Bayes</h3>

# In[ ]:


# Create instantiation of Multinomial Naive bayes and from that check about various parameters
#by default laplace smooting i.e. alpha=1.0
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()


# In[ ]:


# train the model using X_train_dtm (timing it with an IPython "magic command")
# to know how much time this command will take for execution
get_ipython().run_line_magic('time', 'nb.fit(X_train_dtm, y_train)')


# In[ ]:


# Visulize the effects on term matrix after fitted vocabulary in the model
print(type(X_train_dtm))
print(X_train_dtm.shape)
print(X_test_dtm)


# In[ ]:


# Predict function is used to predict to which class Test Review Belongs to.
y_pred_class = nb.predict(X_test_dtm)
print(y_pred_class)


# In[ ]:


# print the confusion matrix
from sklearn.metrics import confusion_matrix
from sklearn import metrics
gb=metrics.confusion_matrix(y_test, y_pred_class)
print(gb)


# In[ ]:


#Plot of Confusion Metric
import matplotlib as plt
from sklearn.metrics import confusion_matrix
import pylab as pl
pl.matshow(gb)
# cm = metrics.confusion_matrix(y_test, y_pred_class, labels=['FAKE', 'REAL'])
pl.title('Confusion matrix')
pl.colorbar()
pl.show()


# <h3> Accuracy =TP+TN/TP+TN+FP+FN </h3>

# In[ ]:


# calculate accuracy of class predictions
from sklearn import metrics
acc= metrics.accuracy_score(y_test, y_pred_class)
acc


# <h4>Classification Error, Counts the Off  Diagonal elements
#  FP+FN / (TP+TN+FP+FN)</h4>

# In[ ]:


#eqivalent to 1-accuracy
error=1-acc
error


# <h2> Precision-Recall</h2>
# <h5>Recall- True Positive Rate or Senstivity or probability of detection
# TPR=TP/TP+FN. It defines what fraction of all positive instances does the classifier correctly identify as positive</h5>

# In[ ]:


#Recall From above Confusion Metric 
recall=(gb[1,1]+0.0)/sum(gb[1,:])
recall


# <h5>Precision- 
# TPR=TP/TP+FP. It defines what fraction of positive predictions are correct</h5>

# In[ ]:


#precision From above Confusion Metric
pre=(gb[1,1]+0.0)/sum(gb[:,1])
print(pre)


# 
# <h3>F1- Score Combining Precision and Recall in to a Single number
# F1= (2*Precision * Recall)/(Precision+Recall) </h3>

# In[ ]:


# caculating F1 Score By using HP i.e 
#F1=2*TP/2*TP+FP+FN
F1=(2*pre*recall)/(pre+recall)
F1


# <h2>Evaluation Metrics for Binary Classification By Using sklearn Lib</h2>

# In[ ]:


# from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
# Accuracy must be greater than 84% in this case because without any model text belong to positive class
# is 84%, this is imbalanced data set
print('Accuracy', metrics.accuracy_score(y_test, y_pred_class))
print('Recall',metrics.recall_score(y_test,y_pred_class))
print('Precision' ,metrics.precision_score(y_test,y_pred_class))
print('F1-Score',metrics.f1_score(y_test,y_pred_class))


# In[ ]:


from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred_class,target_names=['Negative','Positive']))


# In[ ]:


# print message text for the false positives (positive review incorrectly classified as negative)
X_test[(y_pred_class==1)&(y_test==0)].count


# In[ ]:


# calculate predicted probabilities for X_test_dtm 
# We predict the class-membership probability of the samples via the predict_proba method.
y_pred_prob = nb.predict_proba(X_test_dtm)[:,1]
y_pred_prob


# In[ ]:


# calculate AUC with probabilities values
roc_auc=metrics.roc_auc_score(y_test, y_pred_prob)
roc_auc


# In[ ]:


# calculate AUC without probabilities values
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import random
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(false_positive_rate, true_positive_rate)
print('ROC Rate', roc_auc)
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate, true_positive_rate, 'g',
label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# <h4>Compute Log Loss</h4>

# In[ ]:


from sklearn.metrics import log_loss
log_error=log_loss(y_test, y_pred_prob)
log_error


# <h3>Logestic Regression</h3>

# In[ ]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()


# In[ ]:


# train the model using X_train_dtm
get_ipython().run_line_magic('time', 'logreg.fit(X_train_dtm, y_train)')


# In[ ]:


# make class predictions for X_test_dtm
y1_pred_class = logreg.predict(X_test_dtm)
y1_pred_class


# In[ ]:


# print the confusion matrix
cm= metrics.confusion_matrix(y_test, y1_pred_class)
cm


# In[ ]:


# Plot confusion Metric
import matplotlib as plt
from sklearn.metrics import confusion_matrix
import pylab as pl
pl.matshow(cm)
pl.title('Confusion matrix')
pl.colorbar()
pl.show()


# In[ ]:


# calculate predicted probabilities for X_test_dtm (well calibrated)
y1_pred_prob = logreg.predict_proba(X_test_dtm)[:,1]
y1_pred_prob


# In[ ]:


# Calculating ROC Rate
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y1_pred_prob)
roc_lg = auc(false_positive_rate, true_positive_rate)
print('ROC Rate', roc_lg)


# In[ ]:


# PLot AUC For Logistic Regression
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import random
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate, true_positive_rate, 'b',
label='AUC = %0.2f'% roc_lg)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# <h4>Computing Logg Loss</h4>

# In[ ]:


# With Probabilities
from sklearn.metrics import log_loss
log_error=log_loss(y_test, y1_pred_prob)
log_error


# <h4>Evaluation Metrics for Logestic Regression</h4>

# In[ ]:


# calculate accuracy,precision,recall,F1 score
print('Accuracy', metrics.accuracy_score(y_test, y1_pred_class))
print('Recall', metrics.recall_score(y_test,y1_pred_class))
print('Precision', metrics.precision_score(y_test,y1_pred_class))
print('F1 Score', metrics.f1_score(y_test,y1_pred_class))


# In[ ]:


#  classification_report
print(classification_report(y_test,y1_pred_class,target_names=['Negative','Positive']))


# <h4>Logestic Regression through L1 Regularization</h4>

# In[ ]:


from sklearn.linear_model import LogisticRegression
logreg1 = LogisticRegression(penalty='l1',C=1)
get_ipython().run_line_magic('time', 'logreg1.fit(X_train_dtm, y_train)')


# In[ ]:


# make class predictions for X_test_dtm
y2_pred_class = logreg1.predict(X_test_dtm)


# In[ ]:


# print the confusion matrix
cml1= metrics.confusion_matrix(y_test, y2_pred_class)
cml1


# In[ ]:


# Plot confusion Metric
import matplotlib as plt
from sklearn.metrics import confusion_matrix
import pylab as pl
pl.matshow(cml1)
pl.title('Confusion matrix')
pl.colorbar()
pl.show()


# In[ ]:


y2_pred_prob = logreg.predict_proba(X_test_dtm)[:,1]
y2_pred_prob


# In[ ]:


# Calculating ROC Rate
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y2_pred_prob)
roc_lg = auc(false_positive_rate, true_positive_rate)
print('ROC Rate', roc_lg)


# In[ ]:


from matplotlib import pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate, true_positive_rate, 'g',
label='AUC = %0.2f'% roc_lg)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[ ]:


# Calculating ROC Rate
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y2_pred_prob)
roc_lg = auc(false_positive_rate, true_positive_rate)
print('ROC Rate', roc_lg)


# In[ ]:


# calculate accuracy,precision,recall,F1 score
print('Accuracy', metrics.accuracy_score(y_test, y2_pred_class))
print('Recall', metrics.recall_score(y_test,y2_pred_class))
print('Precision', metrics.precision_score(y_test,y2_pred_class))
print('F1 Score', metrics.f1_score(y_test,y2_pred_class))


# <h4> computing Log Loss</h4>

# In[ ]:


# With Probabilities
from sklearn.metrics import log_loss
log_error=log_loss(y_test, y2_pred_prob)
log_error


# <h2>Examining Model</h2>

# In[ ]:


# store the vocabulary of X_train
X_train_tokens = vect.get_feature_names()
len(X_train_tokens)


# In[ ]:


# examine the first 50 tokens
print(X_train_tokens[0:50])


# In[ ]:


# Naive Bayes counts the number of times each token appears in each class
# row 1 represents number of times Low rated reviews token appear and row 2corresponds to high rated Reviews tokens
# Leading underscore is Scikit learn convention that the functionality learns while fitting, use by (_)
nb.feature_count_


# In[ ]:


# rows represent classes, columns represent tokens
nb.feature_count_.shape


# In[ ]:


# number of times each token appears across all Negative REVIEWS
# here we just slicing above feature count
neg_token_count = nb.feature_count_[0, :]
neg_token_count


# In[ ]:


# number of times each token appears across all POSITIVE REVIEWS
pos_token_count = nb.feature_count_[1, :]
pos_token_count


# In[ ]:


# Create a DataFrame of tokens with their separate Negative and Positive Reviews
tokens = pd.DataFrame({'token':X_train_tokens, 'Negative':neg_token_count, 'Positive':pos_token_count}).set_index('token')
tokens.head()


# In[ ]:


# examine 5 random DataFrame rows
# It shows the number of times a word appear in neg and pos class
tokens.sample(50, random_state=5)


# In[ ]:


# Naive Bayes counts the number of observations in each class
# it shows that our data is trained on 93426 neg words and 332895 pos word
# In Class_count_  (_)underscore is used because this method is available after model is fitted
nb.class_count_


# In[ ]:


# add 1 to neg and pos counts to avoid dividing by 0 
tokens['Negative'] = tokens.Negative+ 1
tokens['Positive'] = tokens.Positive + 1
tokens.sample(5, random_state=6)


# In[ ]:


# convert the negative  and positive counts into frequencies
tokens['Negative']= tokens.Negative / nb.class_count_[0]
tokens['Positive'] = tokens.Positive / nb.class_count_[1]
tokens.sample(5, random_state=6)


# In[ ]:


# calculate the ratio of Positive-to-Negative for each token
tokens['Positive_ratio'] = tokens.Positive / tokens.Negative
tokens.sample(5, random_state=6)


# <h4>Print 100 top positive features</h4>

# In[ ]:


# examine the DataFrame sorted by Positive_rate
top=tokens.sort_values('Positive_ratio', ascending=False)
print(type(top))
print(top.shape)
print(top.head(100))


# <h3> Linear SVM by using SGD</h3>

# In[ ]:


# fit train data into Model
from sklearn.linear_model import SGDClassifier
clf = SGDClassifier(loss="hinge", penalty="l2")
get_ipython().run_line_magic('time', 'clf.fit(X_train_dtm, y_train)')


# In[ ]:


# to know everything about instance that we have created
get_ipython().run_line_magic('pinfo', 'clf')


# In[ ]:


# Visulize the effects on term matrix after fitted vocabulary in the model
print(type(X_train_dtm))
print(X_train_dtm.shape)


# In[ ]:


# Predict the class label on test class
ys_pred_class = clf.predict(X_test_dtm)
print(ys_pred_class.shape)


# In[ ]:


# print the confusion matrix
from sklearn.metrics import confusion_matrix
from sklearn import metrics
csvm=metrics.confusion_matrix(y_test, ys_pred_class)
csvm


# In[ ]:


#Plot of Confusion Metric
import matplotlib as plt
from sklearn.metrics import confusion_matrix
import pylab as pl
pl.matshow(csvm)
# cm = metrics.confusion_matrix(y_test, y_pred_class, labels=['FAKE', 'REAL'])
pl.title('Confusion matrix')
pl.colorbar()
pl.show()


# <h3>Evaluation Metrics for SVM By Using sklearn Lib</h3>

# In[ ]:


print('Accuracy', metrics.accuracy_score(y_test, ys_pred_class))
print('Recall',metrics.recall_score(y_test,ys_pred_class))
print('Precision' ,metrics.precision_score(y_test,ys_pred_class))
print('F1-Score',metrics.f1_score(y_test,ys_pred_class))


# In[ ]:


from sklearn.metrics import classification_report
print(classification_report(y_test,ys_pred_class,target_names=['Negative','Positive']))


# In[ ]:


# calculate AUC
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import random
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, ys_pred_class)
roc_auc = auc(false_positive_rate, true_positive_rate)
print('ROC Rate', roc_auc)


# In[ ]:


# plot AU-ROC Curve
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate, true_positive_rate, 'g',
label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[ ]:


from sklearn.metrics import log_loss
log_error=log_loss(y_test, ys_pred_class)
log_error


# <h2> K- fold Cross Validation for k=3 </h2>

# In[ ]:


from sklearn import linear_model, datasets
from sklearn.cross_validation import cross_val_score


# In[ ]:


from sklearn.cross_validation import cross_val_score
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
Xlg = vect.fit_transform(X)
Xlg


# In[ ]:


# 3-fold cross-validation with c=100 for logestic regression
logreg = LogisticRegression(C=100)
get_ipython().run_line_magic('time', "scores = cross_val_score(logreg, Xlg, y, cv=3, scoring='accuracy')")
print(scores)


# In[ ]:


# use average accuracy as an estimate of out-of-sample accuracy
print(scores.mean())


# In[ ]:


# 2-fold cross-validation with c=0.05 for logestic regression
logreg = LogisticRegression(penalty='l1',C=0.05)
get_ipython().run_line_magic('time', "scores = cross_val_score(logreg, Xlg, y, cv=2, scoring='accuracy')")
print(scores)


# In[ ]:


# search for an optimal value of Lambda for Logistic Regression C=1/Lambda
logrg = LogisticRegression(penalty='l1',C=0.05)
L_range = list(range(1,5))
L_scores = []
for l in L_range:
    logreg = LogisticRegression(C=l)
    scores = cross_val_score(logrg, Xlg, y, cv=3, scoring='accuracy')
    L_scores.append(scores.mean())
print(L_scores)


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# plot the value of Lambda for Logestic Regression(x-axis) versus the cross-validated accuracy (y-axis)
plt.plot(L_range, L_scores)
plt.xlabel('Value of Reverse of regularizer for C ')
plt.ylabel('Cross-Validated Accuracy')


# ## Tuning  Parameter using GridSearchCV

# In[ ]:


from sklearn.grid_search import GridSearchCV


# In[ ]:


# define the parameter values that should be searched
l_range = list(range(1,5))
print(l_range)


# In[ ]:


# create a parameter grid: map the parameter names to the values that should be searched
param_grid=dict(C=l_range)
print(param_grid)


# In[ ]:


# instantiate the grid
get_ipython().run_line_magic('time', "grid = GridSearchCV(logrg, param_grid, cv=3, scoring='accuracy')")


# In[ ]:


# fit the grid with data
grid.fit(Xlg, y)


# In[ ]:


# view the complete results (list of named tuples)
grid.grid_scores_


# In[ ]:


# create a list of the mean scores only
grid_mean_scores = [result.mean_validation_score for result in grid.grid_scores_]
print(grid_mean_scores)


# <h3> Comparing all the Models on the basis of Metrics</h3>

# Parameters|Naive Bayes | Logestic Regression|Loges Regr. L1|Lr SVM
# ------------| -------------
# **TIME**| **472 msec** |7min 28s|1min 30s|2.06 s
# **Accuracy**|0.89829|0.92677|**0.9273**|0.89829
# **Recall**|0.93117|**0.9639**|0.9635|0.9311
# **Precision**|0.9380|0.9434|**0.94454**|0.9380
# **F1-Score**|0.93458|0.95358|**0.9539**|0.93458
# **AU-ROC**|0.9350|0.95765|**0.95766**|0.86230
# **logloss** |0.478633|0.2103|0.2103|
# **TN**|24403 |24817|**24953**|24220
# **TP**|103251|**106884**|106837|105221
# **FP**|6822|6408|** 6272**|7005
# **FN**| 7631|**3998**|4045|5661
# #### By using Cross validation, it is found that for C=3 Accuracy is maximum for Logestic Regression 
# #### Naive Bayes is faster among others Linear models.
# 
