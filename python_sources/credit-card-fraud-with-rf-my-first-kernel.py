#!/usr/bin/env python
# coding: utf-8

# **Tackling the Credit Card Fraud dataset with Random Forest**
# 
# Before I start, I would like to mention that this is my first kernel on Kaggle. Not everything is perfect, but I hope that my analysis brings some insight into the problem and help others that are also starting at Kaggle.
# 
# In this kernel, I investigate the credit card fraud dataset and ultimately opt for working with a random forest model.
# This dataset provides us with data in the form of PCA components, the amounts associated with each transaction, the time at which they occurred (in this kernel, we have not explored this variable, but based on other kernels, it does not seem to play a key role), and whether a given transaction is fraudulent or not.
# In the first part of this kernel, we do a uni/bi-variate analysis to determine what PCA components might be more relevant in drawing a line between fraudulent and non-fraudulent transactions. This is used to reduce the dimensionality of this problem.
# 
# In the second part, we look at the variable amount in more detail. We find that quite a few transactions, fraudulent or not, correspond to an amount of 0. We eliminated these transactions on the basis that they do not aggregate any immediate financial value. Perhaps given more context for the problem, it would make sense to retain these transactions.
# 
# Our investigation also reveals that the only a small subset of the fraudulent transactions (about 150 of the total 492 fraudulent transactions) account for about 95% of the total amount loss due to frauds. We felt like it would be better to create a separate model to account for the small-amount transactions. This seemly led to small improvements in the recall metric for fraudulent transactions without affecting the precision much.

# In[ ]:


import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn import metrics
from sklearn import neighbors
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import preprocessing
from sklearn import decomposition
from sklearn import ensemble
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
import scipy
from scipy.stats import spearmanr
from scipy.stats import ks_2samp


# In[ ]:


#importing the dataset 
rawdf = pd.read_csv("../input/creditcard.csv", delimiter = ",")
rawdf.head()


# In[ ]:


rawdf.shape


# Time seems to represent the time at which the transaction happened. I did not investigate this variable but you should check other kernels for more details on it.
# 
# V1, V2, ..., V28 represent PCA components associated with the true data, which cannot be disclosured.
# Amount represents the amount involved in a given transaction.
# 
# Class = 0 is the label for non-fraudulent transactions. 
# Class = 1 is the label for fraudulent transactions.
# 
# Let's check how many of each we have.

# In[ ]:


n_fraud = rawdf.Class[rawdf.Class==1].count()
n_legit = rawdf.Class[rawdf.Class==0].count()

print(n_fraud, n_legit)


# **First Step: Univariate and Bivariate analysis**
# Let's prepare some variables to make the data exploration easier

# In[ ]:


#Looking at the dataset overall statistics

#We will get into more detail about some of the statistics later. In particular, we care about the variable amount.
rawdf.describe()


# In[ ]:


PCA_list = rawdf.columns[1:29] #contains the labels V1, V2, ..., V28
PCA_index = np.arange(1,29) #vector with numbers 1, 2, 3, ..., 28


# In[ ]:


plt.plot(PCA_index[0:10], rawdf[PCA_list[0:10]].std(), 'o')
plt.plot(PCA_index[10:20], rawdf[PCA_list[10:20]].std(), 'o')
plt.plot(PCA_index[20:29], rawdf[PCA_list[20:29]].std(), 'o')
plt.ylabel("standard deviation")
plt.xlabel("PCA component")
plt.xticks(PCA_index)
plt.tight_layout()

#This plot just confirms our expectation that the standard deviation is decreasing for higher PCA components.


# In[ ]:


#Let's take a look at the first few components to have a feeling for bivariate distributions

PCA_shortlist = list(PCA_list[0:4])
PCA_shortlist.append('Class')
sns.pairplot(rawdf[PCA_shortlist], hue='Class')
plt.show()


# We do see some tendency for fraudulent transactions to form their own cluster, though it certainly overlaps with the clusters of non-fraudulent transactions. The hope is that we might be able to draw a line between the different labels in the multi-dimensional space.
# Let us now look at the histograms for various PCA components.

# In[ ]:


#histograms for various PCA components

components = ["V1", "V2", "V3", "V4", "V5"]

for PCA_comp in components:
    
    plt.figure(figsize=(8,3))
    plt.subplot(1,2,1)
    rawdf[rawdf.Class == 1][PCA_comp].hist(label="fraud", color="blue", density='True')
    rawdf[rawdf.Class == 0][PCA_comp].hist(label="not fraud", color="orange", density='True')
    plt.title(PCA_comp)
    plt.legend()
    
    plt.subplot(1,2,2)
    sns.kdeplot(rawdf[rawdf.Class == 1][PCA_comp], color='blue')
    sns.kdeplot(rawdf[rawdf.Class == 0][PCA_comp], color='orange')
    plt.tight_layout()
    plt.title(PCA_comp)


# From the plots above we can see that there are noticeable deviations for the PCA components for various fraudulent transactions.
# Aiming at dimensionality reduction, let's take a look at higher PCA components to see if the effect is diminished.
# In what follows below, we have added components that we think will be ultimately not so relevant, but it is easy to adjust the code to show the remaining components.

# In[ ]:


components = ["V6", "V8", "V13", "V15", "V19", "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28"]

for PCA_comp in components:
    
    plt.figure(figsize=(8,3))
    plt.subplot(1,2,1)
    rawdf[rawdf.Class == 1][PCA_comp].hist(label="fraud", color="blue", density='True')
    rawdf[rawdf.Class == 0][PCA_comp].hist(label="not fraud", color="orange", density='True')
    plt.title(PCA_comp)
   # sns.kdeplot(rawdf[rawdf.Class == 1][PCA_comp], color="blue")
   # sns.kdeplot(rawdf[rawdf.Class == 0][PCA_comp], color='orange')
    plt.legend()
    
    plt.subplot(1,2,2)
    sns.kdeplot(rawdf[rawdf.Class == 1][PCA_comp], color='blue')
    sns.kdeplot(rawdf[rawdf.Class == 0][PCA_comp], color='orange')
    plt.tight_layout()
    plt.title(PCA_comp)


# The plots above suggest cutting out components starting from V19 and above. We can later readjust to verify improvements. There are also some random components that do not seem to contribute much. V13 and V15 do not seem very useful. Neither do V6 and V8.
# I looked at correlations between PCA components and the class labels with the loop below. It was fruitless: the imbalance makes every correlation be simply too small. There might be some workaround, but I decide not to pursue this any further, leaving the results here for the sake of clarity and completeness.

# In[ ]:



for component in PCA_list[:3]:
    print("The current component is "+component)
    print(spearmanr(rawdf[component], rawdf["Class"]))


# The problem with the above is that, since the dataset is vastly dominated by class 0, the correlation is very weak. What we will do instead is to quantify the difference between the distributions for every PCA component when class = 0 and when class = 1.
# I decided next to compute KS statistic. In practice, that turned out not to be so useful either, but let's see what we get.

# In[ ]:



#Null hypothesis: two samples are drawn from the same distribution.
for component in PCA_list[:3]:
        print("The current component is "+component)
        print(ks_2samp(rawdf[rawdf.Class==1][component], rawdf[rawdf.Class==0][component]))


# The KS metric was also not that helpful. It is simply stating that, given the two samples (class 0 and 1), we know that they are coming from different distributions, and every single PCA component is capable of confirming that (you can easily adapt the loop above to look at all components).
# 
# That said, in a real situation, what matters is not a sample, but a single point. Thus, the KS metric ends up not being so helpful in practice.
# Though I have not quantified this on here, what matters is that a point will be difficult to classify when it resides in an area where two distributions associated with labels 0 and 1, respectively, overlap too much. Thus, I eliminate components for which that happens. For now, this procedure will have to be based on visual inspection. It is worth considering whether this could be made more rigorous at some point.
# Before we proceed, let us look at the variable 'Amount'.
# 
# I realized after some attempts that it is better to limit ourselves to amounts smaller than 1000 at first because most transactions are concentrated in this region.

# In[ ]:


#histogram for transaction amounts below 1000
crit_amount = 1000

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
rawdf[rawdf.Class==1]['Amount'][rawdf.Amount < crit_amount].hist(label="fraud", color="blue", density='True')
plt.legend()
plt.subplot(1,2,2)
rawdf[rawdf.Class==0]['Amount'][rawdf.Amount < crit_amount].hist(label="not fraud", color="orange", density='True')
plt.legend()

plt.figure()
sns.distplot(rawdf[rawdf.Class==1].Amount[rawdf.Amount < crit_amount], label='fraud')
sns.distplot(rawdf[rawdf.Class==0].Amount[rawdf.Amount < crit_amount], label='not fraud')
plt.legend()


# A very tiny portion of the dataset (about 1%) corresponds to transactions about the critical value of 1000
# Strangely enough, quite a few transactions correspond to the amount 0. We will drop those transactions from the dataset at this point.

# In[ ]:


rawdf = rawdf[rawdf.Amount>0]
print(rawdf.Amount[rawdf.Class==1].count())
#this is the updated number of fraudulent transactions


# In[ ]:


print("Non-fraudulent transactions, amount above 1000: "+
      str(rawdf[rawdf.Class==0].Amount[rawdf.Amount > crit_amount].count()))
print("Fraudulent transactions, amount above 1000: "+
       str(rawdf[rawdf.Class==1].Amount[rawdf.Amount > crit_amount].count()))


# We would like to determine what transaction range is responsible for most of the losses.

# In[ ]:


print(rawdf.Amount.groupby(rawdf.Class).describe())

print(rawdf.Amount.groupby(rawdf.Class).sum())


# From the above, we can see the total amount associated with class 0 and with class 1. We can also see that the vast majority of transactions is below about the value of 100 (about 75%). However, we do not care so much about the number of transactions, but the amount lost from such transactions combined. 
# 
# We will see next, for example, that even though the low-amount transactions outnumber the high-amount ones, the 150 transactions with the highest amounts are responsible for 95% of all losses. Hence, it is much more important to focus on those.
# Let's see how losses stack up next to confirm what we just said.

# In[ ]:


amounts_fraud = np.asarray(rawdf.Amount[rawdf.Class==1]) #array with the amounts of fraudulent transactions
amounts_fraud.sort() #amount sorted in ascending order
cum_amounts_fraud = np.cumsum(amounts_fraud) #array with the cumulative sum 
#plt.style.use('ggplot')

total = cum_amounts_fraud[-1] #total amount, used to normalize the cumulative sum

plt.figure()
plt.grid()
plt.plot(cum_amounts_fraud/total)
plt.ylabel("Normalized Cumulative amount")
plt.xlabel("Number of transactions (starting from the lowest)")


# In[ ]:


#A few more useful numbers to have in mind (we did not pick these numbers randomly; they are based on the plot above)

print("Fraction associated with the lowest 300 transactions: "+ 
      str(amounts_fraud[0:300].sum()/total)) 

print("Fraction associated with the highest 100 transactions: "+
      str(amounts_fraud[365:].sum()/total)) 

print("Amount at which the 300th transaction starts: "+
      str(amounts_fraud[300]))

print("Amount lost associated with the 10 highest transactions: "+
      str(amounts_fraud[455:].sum()/total))

print("Amount associated with the 10 most expensive transactions:" )
print(str(amounts_fraud[455:]))


# From the above, we conclude that the 300 transactions with the lowest amounts account for less than 10% of the losses. It might be worth eliminating them at some point and treating the remaining transactions separately.
# 
# About 100 transactions account for 80% of the losses (the ones between 365 and 465).
# 
# The 300th transaction starts at about an amount of 88. It provides with a number to further reduce the dataset if we decide to cut transactions below that.
# Finally, the 10 most expensive transactions are responsible for almost 25% of the losses. This is a very large number, and given that we have only 10 points in this region, it might be difficult to develop an algorithm that would work well in this sector. It might be worth looking at those points alone to see if they would stand out somehow as an expansion on this kernel. In the last line, we see that those more expensive transactions begin at amount > 996, approximately.

# In[ ]:


#Let's check if the dataset balance improves in the region starting from amount = 88 and above. 
rawdf.Amount[rawdf.Amount > 88][rawdf.Class==1].count()/rawdf.Amount[rawdf.Amount > 88].count()


# The imbalance has improved slightly (the original ratio was something like 0.0017).
# In what follows, we will concentrate ourselves on this sector with amounts above 88.
# 
# From now on, I will tackle this dataset with the method of random forest. I did some preliminary investigation before opting for random forest, but I am going to skip that part here since it was not necessarily an exhaustive investigation. There might be better starting options.

# **Data transformation step**
# We start by dropping some components that seemed somewhat unimportant according to the histograms we had at the beginning.
# Also, we keep only amounts above 88, as previously discussed.

# In[ ]:


df = rawdf[(rawdf.Amount > 88)].copy()
cols_to_drop = ["V6", "V8", "V13", "V15", "V19", "V20", "V21", "V22", "V23", "V24", "V25", "V26", 
                "V27", "V28", "Time", "Amount"]
df.drop(cols_to_drop, inplace=True, axis=1)
df.head()


# In[ ]:


Y = df.Class 
X = df.drop("Class", axis=1)


# n what follows next, we will try four different strategies: 
# * Random forest, assigning class weights to labels 0 and 1 
# * Oversampling (SMOTE) 
# * Undersampling (random) 
# * Combined oversampling and undersampling
# 
# I ultimately find that it is worth simply using the imbalanced dataset, but assigning weights to labels 0 and 1 for class.
# I did not notice substantial gains to justify the usage of these fancier strategies. In particular, undersampling can do quite some damage to the precision score.
# 
# We start by splitting the dataset into training and test set. 30% will be left for test.

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                stratify=Y, 
                                                test_size=0.3)


# Next we do oversampling and undersampling. This must be done on the training set; not on the test set. 
# In the case of oversampling, it is important for the test set not to be used to generate extra samples, otherwise it will make the model biased towards the test set.
# 
# In the case of undersampling, the idea is that we still would like to see how the model can perform on an imbalanced dataset, even though we balanced it by eliminating samples.

# In[ ]:


#Oversampling 
from imblearn.over_sampling import SMOTE
oversampler = SMOTE(n_jobs=-1)
X_over, Y_over = oversampler.fit_sample(X_train, np.ravel(Y_train))

#undersampling
from imblearn.under_sampling import RandomUnderSampler
undersampler = RandomUnderSampler()
X_under, Y_under = undersampler.fit_sample(X_train, np.ravel(Y_train))

#combining both
from imblearn.combine import SMOTEENN
combined = SMOTEENN()
X_comb, Y_comb = combined.fit_sample(X_train, Y_train)


# Now we prepare one for model for each strategy:
# -imbalanced dataset, label 0 has weight 1, label 1 has weight 100
# -oversampled dataset
# -undersampled dataset
# -combined over/under dataset

# In[ ]:


#different classifiers for different strategies
rf = ensemble.RandomForestClassifier(class_weight={0:1,1:100})
rf_over = ensemble.RandomForestClassifier()
rf_under = ensemble.RandomForestClassifier()
rf_comb = ensemble.RandomForestClassifier()

#fitting the four models
rf.fit(X_train, Y_train)
rf_over.fit(X_over, Y_over)
rf_under.fit(X_under, Y_under)
rf_comb.fit(X_comb, Y_comb)

#predictions for the test set associated with the four models
Y_pred = rf.predict(X_test)
Y_over_pred = rf_over.predict(X_test)
Y_under_pred = rf_under.predict(X_test)
Y_comb_pred = rf_comb.predict(X_test)


# Let's now get some predictions for the test set and looking at the classification reports for each

# In[ ]:



print("Imbalanced Sample")
print(classification_report(Y_test, Y_pred))

print("Oversampling")
print(classification_report(Y_test, Y_over_pred))

print("Undersampling")
print(classification_report(Y_test, Y_under_pred))

print("Combined")
print(classification_report(Y_test, Y_comb_pred))


# Conclusions:
# -Oversampling seems to have a performance that is always very close to that obtained with the imbalanced data. However, it reduces the precision by about 10-20% (the precise value varies from run to run). Choosing between these two approaches will depend on the desired level of precision.
# -Undersampling is often capable of pushing the recall higher, but hurts precision substantially.
# -The combined method seems to do more or less the same as oversampling.
# -Overall, it seems like it might be better to invest some time in the imbalanced dataset. Before we proceed to do that, let's also take a look at the confusion matrices.

# In[ ]:


#Function to create a nice plot of the confusion matrix
#This function was NOT created by me. It was part of a code an instructor presented during some class
#I am assuming that there is no ownership over it, but I thought I'd make clear that I did not code this function

import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[ ]:


plt.figure(figsize=(6,3))
cm = confusion_matrix(Y_test, Y_pred)
plot_confusion_matrix(cm, [0,1])
plt.title("random forest without original sample")

plt.figure(figsize=(6,3))
cm = confusion_matrix(Y_test, Y_over_pred)
plot_confusion_matrix(cm, [0,1])
plt.title("random forest with oversampling")

plt.figure(figsize=(6,3))
cm = confusion_matrix(Y_test, Y_under_pred)
plot_confusion_matrix(cm, [0,1])
plt.title("random forest with undersampling")

plt.figure(figsize=(6,3))
cm = confusion_matrix(Y_test, Y_comb_pred)
plot_confusion_matrix(cm, [0,1])
plt.title("random forest with combined over+under")


# In[ ]:


#Looking at the most important features.
feature_importances = pd.DataFrame(rf.feature_importances_,
                                   index = X_train.columns,
                                    columns=['importance']).sort_values('importance',         
                                                                        ascending=True)
feature_importances

#It is important to realize that the precise order of these features may vary from run to run.
#However, some tend to often show up in low tiers, and I eliminated a few based on that.
#One can always go back and add them back to see if they would improve performance.

#In my run, I decided to eliminate the following features:


# In[ ]:


#Let's start by reducing our space even further
X_train.drop(["V2","V1", "V11", "V18", "V5", "V9", "V7", "V4"], axis=1, inplace=True)
X_test.drop(["V2","V1", "V11", "V18", "V5", "V9", "V7", "V4"], axis=1, inplace=True)


# In[ ]:


#confirming that we dropped the right columns
X_train.head(3)


# Let's now do some fine-tuning with GridSearch

# In[ ]:



param_grid = {"max_depth": [2,3],
              "max_features": [2,3],
              "n_estimators": [80,100,150,200],
              "class_weight": [{0:1, 1:100}]
              }

cv_method = StratifiedKFold(n_splits=5, shuffle=True)

rf_grid = GridSearchCV(estimator = ensemble.RandomForestClassifier(),
                       param_grid = param_grid,
                       cv = cv_method,
                       scoring = 'recall')

#The goal here is to maximize the metric recall.


# I am being modest in the range of parameters considered above.
# That is partly because training the various models was taking very long.
# Another reason is that I did not see this level of fine-tuning actually affecting performance dramatically.
# Finally, I did some preliminary, a bit more expansive investigation of the range of parameters.
# In doing so, I realized that more often than not, max_depth and max_features, for example, should be kept to a minimum.

# In[ ]:


#using gridsearch to fit
rf_grid.fit(X_train, np.ravel(Y_train))


# In[ ]:


rf_grid.best_params_


# The line above gives us the best parameters obtained by grid search. As I said before, the values will vary from run to run. After all, the very train/test splitting is random (I am aware that it can be made not random, but I chose not to).
# I found that the following set works fairly well, though it is probably not going to be the same given by a random run:
# class_weight: {0: 1, 1: 100},
#  max_depth: 2,
#  max_features: 2,
#  n_estimators': 100.
#  
#  Finally, let's finish this kernel with the classification report and confusion matrix for this model

# In[ ]:


Y_grid_pred = rf_grid.predict(X_test)
print(classification_report(Y_test, Y_grid_pred))

plt.figure(figsize=(6,3))
cm = confusion_matrix(Y_test, Y_grid_pred)
plot_confusion_matrix(cm, [0,1])


# The next step would be to look at transactions below the amount of 88, but we will leave that for another time, since they account for about 5% of the amount lost.
# (That said, they correspond to about 350 transactions that still need to be classified.)
# 
# On my runs, it seemed liked this focus on the more expensive transactions led to some improvement of the recall metric, but it would be interesting to quantify this further and make sure that this is not just an effect of model variance.

# I got some requests for adding a ROC-AUC curve to my kernel, so I am updating the file with some lines to do just that.

# In[ ]:


Y_probs = rf_grid.predict_proba(X_test)
pos_probs = list(map(lambda l: l[1], Y_probs))
fpr, tpr, thresholds = roc_curve(Y_test, pos_probs, pos_label=1)
roc_auc = auc(fpr, tpr)
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()


# 

# 

# 

# 

# 
