#!/usr/bin/env python
# coding: utf-8

# # Is (H)odor important?
# # Mushroom Classification: safe to eat or deadly poison?

# # 1. Introduction and objectives
# 
# Our objectives are:
# * We want to train some models to solve the classification problem (Edible / Poisonous). 
# * We want to create a feature relevance ranking, that will help us to understand what information is more helpful in this problem.
# * We want to have fun!
# 
# In order to train and evaluate our models we need to define some scores. We propose the following metrics:
# * Accuracy
# * F-score 
# * Area Under the Curve (AUC_ROC)
# 
# We will validate the parameters of our models using the F-score, that is calculated using Precision and Recall. We will consider that the positives are cases of Poisonous mushrooms, which means that false negatives are critical (deadly, indeed) and should be avoided. On the other hand, having a big number of false positives is not so problematic, because an Edible mushroom is not supposed to kill anybody. Consequently, we need to have a Recall close to 1, and a Precision as high as possible. We will focus on the F1-score, that will compute the harmonic mean of both metrics.

# # 2. Preprocessing stage and models
# We begin loading our files. The available variables (or columns) are categorical features, which means that we need to assign numerical indexes to each one of them if we want to use sklearn. We will use LabelEncoder().
# 
# As the dataset has a small number of samples we also compute the histogram for each feature. It can be useful to determine the number of labels of each one of the features, and also to check out if some of them are mono-class. In that case, they can be removed to save some computational time in later calculations.
# 

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
# File loading
import pandas as pd
import numpy as np
from sklearn import preprocessing

data = pd.read_csv('../input/mushrooms.csv');

columns = data.columns


#encoder = preprocessing.LabelEncoder()
#encoder.fit(np.unique(data))
encoder_labels = []
histo = [];
for ind in np.arange(0, len(columns)):
    encoder = preprocessing.LabelEncoder()
    data[columns[ind]] = encoder.fit_transform(data[columns[ind]])
    #data[columns[ind]] = encoder.transform(data[columns[ind]])
    
    # We save the histograms, just in case.
    dummy_histo, dummy = np.histogram(data[columns[ind]], bins=len(np.unique(data[columns[ind]])))
   
    
    # If we find a feature with a single label we just remove it.
    if len(dummy_histo) == 1:
        del data[columns[ind]]
        print('Removing feature:'+repr(columns[ind]))
    else:
        histo.append(dummy_histo);
        encoder_labels.append(encoder)
            
columns = data.columns


# In[ ]:


# We can check that both classes (edible and poissonous) have a similar number of samples.
print('N=0: ', histo[0][0])
print('N=1: ', histo[0][1])


# Before we craft any new feature we are going to train some basic models, linear and non-linear. We will consider logistic regression and Random Forest classifiers.
# 
# First, we divide our dataset into train and test sets. We are going to validate some of the parameters during training, so we will also consider a validation set.

# In[ ]:


from sklearn.cross_validation import train_test_split
labels = columns[columns != 'class']

X_train, X_test, y_train, y_test = train_test_split(data[labels], data['class'], test_size=0.20, random_state=42);

from sklearn.grid_search import GridSearchCV


# ## 2.1. Logistic regressor - Training

# In[ ]:


from sklearn.linear_model import LogisticRegression

parameters = { 
    'penalty':['l1','l2'],
    'C':[.001,.01,.1, 1, 10, 100, 1000],              
    }
machine = LogisticRegression(random_state = 44, n_jobs = -1, class_weight = 'balanced')

clf = GridSearchCV(machine, parameters, n_jobs = -1, scoring = 'f1', cv = 5)  # scoring='roc_auc'
clf.fit(X_train, y_train)
clf.grid_scores_


# We observe that using a high penalty (C) produces a higher f-score. We need to consider that this penalty is applied over the loss-function of the classifier, which means that applying regularization is not critical for this particular dataset. On the other hand, both l1-norm and l2-norm produce similar scores.
# 
# We choose the best trained model, which will be used during the test evaluation.

# In[ ]:


model_log_reg = clf.best_estimator_ 


# ## 2.2. Random Forest - Training

# In[ ]:


from sklearn.ensemble import RandomForestClassifier

parameters = { 
    'n_estimators':[10,20],
    'max_features':[.3, 1],
    'max_depth':[1, 3, 5, 10, None],
    'min_samples_leaf':[100, 10, 1]
    }
machine = RandomForestClassifier(random_state = 44, n_jobs = -1, class_weight = 'balanced')

clf = GridSearchCV(machine, parameters, n_jobs = -1, scoring = 'f1', cv = 5)  # scoring='roc_auc'
clf.fit(X_train, y_train);


# This is a non-linear model, where we train several decision trees in order to classify our features. From the validation process we can depict the next information:
# * The number of estimators is a critical parameter if we apply an extremely low depth, such as max_depth = [1]. In fact, we can increase by 0.11 the mean f-score simply using more trees. However, it increases the standard deviation.
# * If we do not apply any restriction to the maximum depth the f-score tends to increases until it reaches the maximum f-score. This happens when we set max_depth = [10, None].
# * The minimum number of samples per leaf to split is not a critical parameter in this problem. In fact, we could consider that it begins to have some relevance when trees are deep enough, which happens because the training stops before the tree has some complexity (bigger depth), where we would have a lower amount of samples to train.
# * Setting max_features = [1] states that the system will use all the features in order to train each tree. However, this configuration only proved to be useful when the depth was max_depth = 1. In the rest of our models using max_features = 0.3 improved the generalization capabilities of our system.
# 
# We validated all the parameters directly because the dataset has a small number of samples and features. This could not be so easily feasible with a more complex dataset. On the other hand, we still need to compute a test score in order to verify if the system is truly able to generalize.

# In[ ]:


clf.grid_scores_


# In[ ]:


model_RF = clf.best_estimator_;


# # 3. Test results

# In this section we will compute different scores using the two best models that we trained previously.

# In[ ]:


from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
def compute_scores(y_test, y_pred):
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)
    
    print('Accuracy = '+repr(acc))
    print('F-score =  '+repr(f1))
    print('AUC = '+repr(auc))
    
    return acc, f1, auc

print('\n********\nLOGISTIC REGRESSION')
acc_log, f1_log, auc_log = compute_scores(y_test, model_log_reg.predict(X_test));
print('\n********\nRANDOM FOREST')
acc_RF, f1_RF, auc_RF = compute_scores(y_test, model_RF.predict(X_test));


# The results showed that we can compute the highest scores using Random Forest, at least with the metrics that we considered in our analysis. However, we can observe that the Logistic Regressor produced remarkably high scores, being all of them really close to 1.0. It could mean that the dataset is almost-linearly separable.
# 
# We will perform now an analysis of the features that formed our dataset, and we will try to draw some conclusions.

# # 4. Feature analysis

# In this section we will try to draw conclusions using the feature relevance ranking produced by our Random Forest model.

# In[ ]:


import matplotlib.pyplot as plt
plt.stem(model_RF.feature_importances_)
plt.ylabel('Feature weight')
plt.xlabel('Feature index')


# From the previous figure we observe that there exist five particularly high features, which are (descending relevance):
# * 'odor' [4]
# * 'gill-size' [7]
# * 'spore-print-color' [18]
# * 'population' [19]
# * 'stalk-surface-above-ring' [11]
# 
# However, we should take into account that none of them is relevant enough (let's say, a relevance of ~0.75) to solve our classification problem by itself.
# 
# There are others with relatively high scores, but we will focus on these ones.

# In[ ]:


top_idx = [ 4,7,18,19,11];


# We are going to analyze if there exist any linear relationship between those features and the labels, as our classification models suggest.

# In[ ]:


dummy_top = [0, 5,8,19,20,12];
data[columns[dummy_top]].corr()


# We observe that there is no linear relation amongst the selected features nor the output classes, which means that our top 5 features can be considered to be linearly independent. However, as we verified with our Logistic Regression model it does not mean that there is no linear model able to produce good results.

# We are going to check what information is more relevant about edible mushrooms, and we will repeat this analysis for the Poisonous ones. We will pay attention to our top 5 features, and will compare histograms for both classes.

# In[ ]:


def plot_histogram(data, feature_index):
    feature = labels[feature_index]

    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.ticker import FormatStrFormatter
    
    # EDIBLE MUSHROOMS
    samples = data[data['class'] == 0];  # Edible mushrooms
    samples = samples[labels[top_idx]];
    #plt.hist(samples['odor'],label='aaa')


    #fig, ax = plt.subplots()
    fig = plt.figure()
    ax = plt.subplot(111)
    counts, bins, patches = ax.hist(samples[feature], facecolor='blue', edgecolor='gray',
                                    bins = np.arange(0,np.max(data[feature])+2),
                                    range=(0,np.max(data[feature])))
    print('Counts E: ',counts);
    
    ax.set_xticks(bins)
    # Label the raw counts and the percentages below the x-axis...
    bin_centers = 0.5 * np.diff(bins) + bins[:-1]
    for count, x in zip(encoder_labels[feature_index+1].classes_, bin_centers):
        # Label the raw counts
        ax.annotate(count, xy=(x, 0), xycoords=('data', 'axes fraction'),
            xytext=(0, -18), textcoords='offset points', va='top', ha='center')

    plt.title('Histogram comparison. Feature: "'+feature+'".')
    
    # POISSONOUS MUSHROOMS
    samples = data[data['class'] == 1];  # Edible mushrooms
    samples = samples[labels[top_idx]];

    
    counts, bins, patches = ax.hist(samples[feature], facecolor='red', edgecolor='gray',
                                    bins = np.arange(0,np.max(data[feature])+2),
                                    range=(0,np.max(data[feature])),
                                    alpha=0.75
                                   )
    ax.set_xticks(bins)

    # Label the raw counts and the percentages below the x-axis...
    bin_centers = 0.5 * np.diff(bins) + bins[:-1]
    for count, x in zip(encoder_labels[feature_index+1].classes_, bin_centers):
        # Label the raw counts
        ax.annotate(count, xy=(x, 0), xycoords=('data', 'axes fraction'),
            xytext=(0, -18), textcoords='offset points', va='top', ha='center')
    
    plt.ylabel('# Counts')
        
    plt.legend(['Edible','Poissonous'])
    
    print('Counts P: ',counts);


# ## 4.1. Feature #1 : 'odor'
# The most relevant feature according to our previous results is 'odor'. In the following figure we compare the histograms of edible and Poisonous mushrooms considering this particular feature. The labels (or tags) used for this feature are the following:
# *  almond = a
# * anise = l
# * creosote = c
# * fishy = y
# * foul = f
# * musty = m
# * none = n
# * pungent = p
# * spicy = s
# 
# If we analyze the two histograms we observe that there exist certain odors that are exclusive of edible and Poisonous mushrooms.
# 
# First, we will focus on Poisonous ones (red color). The histogram depicts that there are 6 kinds of odors that are characteristic of Poisonous mushrooms, whose tags are: ['c', 'f', 'm', 'p', 's', 'y']. Consequently, if we find a mushroom whose smell belongs to that group, it is likely to be Poisonous (at least if we only consider the samples of our dataset). However, we do not know whether a bigger dataset could provide different results or not. 
# 
# When we check the Edible ones features we observe that the most characteristic odors are ['a', 'l','n']. The two first labels are exclusive of Edible mushrooms, but the tag ['n'] appears in both classes. We have several examples of Edible mushrooms that have this feature value (almost 3500) in comparison with Poisonous ones. However, we should not forget those numbers could change if we had more samples. Consequently, if a mushroom is labeled as ['n'] we should take into account other features apart from 'odor'.
# 

# In[ ]:


plot_histogram(data, top_idx[0])


# ## 4.2. Feature #2 : 'gill-size'
# This particular feature is binary, and the labels are the following ones:
# * broad = b
# * narrow = n
# 
# If we analyze the histogram of each class (E or P), we observe that Edible mushroom tend to have broad gills. However, we should take into account that there are certain Poissonous ones whose gill has this particular size. Consequently, if the gill is broad it would not be recommended to eat the mushroom, as the number of Poissonous samples is significatively high.
# 
# On the other hand, we observe that a small number of Edible samples have a narrow gill. If we consider their Poissonous counterpart we find out that almost a half of the samples are poissonous. Consequently, according to our data samples a narrow-gilled mushroom is likely to be poissonous.
# 
# In fact, this feature is not recommended to be used alone if we want to determine if a mushroom is Poissonous, essentially because the histogram of those mushrooms is almost uniform.

# In[ ]:


plot_histogram(data, top_idx[1])


# ## 4.3. Feature #3 : 'spore-print-color'
# The labels of this feature are the following:
# * black = k
# * brown = n
# * buff = b
# * chocolate = h
# * green = r
# * orange = o
# * purple = u
# * white = w
# * yellow = y
# 
# In this case histograms show that there are plenty of uncertainties. If we focus on the feature labels ['b', 'o', 'r', 'u', 'y'] we observe that we have a small number of samples compared with the rest of the labels, but they are exclusive or certain kind of mushrooms.
# 
# The labels ['h', 'w'] are strongly related with Poisonous mushrooms, in a similar fashion to ['k', 'n'] with Edible ones. However,  it is recommended to consider other features to make a final decision.

# In[ ]:


plot_histogram(data, top_idx[2])


# ## 4.4. Feature #4 : 'population'
# The population indicates if mushrooms of a certain kind are grouped or not, and how dense are those groups. The values of this feature are the following:
# * abundant = a
# * clustered = c
# * numerous = n
# * scattered = s
# * several = v
# * solitary = y
# 
# Two values, ['a', 'n'], are common in Edible mushrooms, and the rest can be found on Edible and Poisonous ones. In fact, Poisonous examples have a tendency to groups with several ['v'] members. This characteristic could be used as a warning signal, essentially because it is a visual feature where you do not need to know about the details.  Beware if there are several mushrooms of the same kind.

# In[ ]:


plot_histogram(data, top_idx[3])


# ## 4.5. Feature #5 : 'stalk-surface-above-ring'
# 
# The values of this feature are:
# * fibrous = f
# * scaly = y
# * silky = k
# * smooth = s
# 
# First, we observe that there is a small number of samples of ['y']. Secondly, the histogram shows that ['k'] is a property common in Poisonous specimens, even if there exist examples that are Edible. The values ['f', 's'] are slightly more usual in Edible samples, but there is a good number of Poisonous specimens that share these features. 
# 
# Consequently, 'stalk-surface-above-ring' is a feature that can be apparently useful when it takes certain values, but cannot be considered alone if we want to develop a good model.

# In[ ]:


plot_histogram(data, top_idx[4])


# # 5. Conclusions

# From this dataset we can make the following conclusions:
# * This dataset has a relatively small number of samples and features (8124, 22). Consequently, if we wanted to develop a real model we would have needed a bigger amount of samples, because a wrong result can be dangerous for people. 
# * Our linear model performance was pretty close to the maximum of every score that we set. Consequently, we could assume that our features can be separated using linear boundaries. The non-linear model produces the highest scores possible, which means that despite the linearity level of the dataset it is necessary to introduce non-linearities for a perfect fitting.
# * The most relevant feature is 'odor'. In fact, if we check out the histogram of this feature we observe that there is only one situation where 'odor' can produce wrong answers: the value ['n'], which comes from 'None'. Consequently, if there is no odor a mushroom might be Poisonous, but according to the histogram this behaviour is unusual in our dataset. The rest of the top 5 features are less discriminant, but in some cases provide other useful information. For example, 'population' is a visual feature that can help to quickly determine if there is any risk when you find a specimen or a group of them: if they are 'clustered', 'scattered', 'several' or 'solitary' then the specimen might be Poisonous. The real question is: how many specimens does a group need to be tagged as 'several', 'numerous', etc?
# 
