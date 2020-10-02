#!/usr/bin/env python
# coding: utf-8

# # Introduction
# When trying to make good predictions from a data set, there are two basic things we can do: (1) preprocessing our data (normalizing, re-sampling, feature engineering, etc.) and (2) running that data through an appropriate and well-tuned classifier. This kernel is going to focus on the former. Specifically, what are some preprocessing techniques we can try to see if they improve the results, and how can we be careful about making sure they're actually producing *real* improvement (and not just over-fitting based on noise).
# 
# To do that, we're going to build a test harness that lets us quickly test whether a new technique is an improvement over baseline, and then try out 5 different techniques for improving our results:
# 
# 1. Outlier removal
# 2. Upsampling
# 3. Stemming
# 4. n-grams
# 5. Stop-word removal
# 
# The main take-away here shouldn't be whether any given technique works or not, but rather how we can systematically check whether techniques work, and understand (1) why things might not work and (2) when we might use a technique even though it doesn't seem to produce an improvement.

# # Setup
# First, let's import all the modules we need, neatly and alphabetized like we're trying to impress our boss. Let's also turn off deprecation and future warnings, because that's easier to than writing better code that doesn't throw warnings. Ha.

# In[ ]:


import json
import re
import warnings

from imblearn import pipeline as imblearn_pipeline
from imblearn import FunctionSampler
from imblearn.over_sampling import SMOTE, RandomOverSampler
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import pandas as pd
from scipy.stats import chisquare
from sklearn import (
    ensemble,
    feature_extraction,
    model_selection,
    multiclass,
    pipeline,
    preprocessing,
    svm,
)

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)


# Next, let's define a function to load and preprocess our data. We're not doing much preprocessing in this function itself (we'll save the heavy-lifting for pipelines), just converting the ingredient lists to comma-separated strings, replacing hyphens, and removing non-letter characters.
# 
# Notice we also added three flags: **target** (which let's us encode a target variable with a LabelEncoder, but only for our training set), **subsamp** (which let's us subsample the data for speedy investigation), and **stem** (which lets us reduce each word in our list to a simpler stem). This will allow us to more speedily process/test data without changing a bunch of code.
# 
# We'll also call our function, taking a subsample of the training data (but not stemming yet).

# In[ ]:


def load_and_process_data(path, target=None, subsamp=False, stem=False):
    """
    Loads and preprocesses either the training or test data sets
    :param path: The path to the data
    :param target: The name of the target variable, if any
    :param subsamp: Whether to sub-sample the data for speed and profit
    :return: A cleaned up dataframe and a LabelEncoder (or None, if no target is supplied)
    """

    with open(path, 'r') as fp:
        data = json.load(fp)

    df = pd.DataFrame(data)
    
    # Replace hyphens with spaces for consistency
    df['text'] = df.ingredients.apply(lambda x: ','.join([w.replace('-', ' ') for w in x]))

    # Remove everything except letters and spaces 
    nonalphas = re.compile('[^a-z ]')
    df['text'] = df.text.apply(lambda x: ','.join([nonalphas.sub('', w) for w in x.split(',')]))
    
    if stem:
        # Stem everything down to correct spelling issues and increase consistency
        stemmer = LancasterStemmer()
        df['text'] = df.text.apply(lambda x: ','.join([' '.join([stemmer.stem(w) for w in i.split(' ')]) for i in x.split(',')]))
    
    if target:
        le = preprocessing.LabelEncoder()
        df[target] = le.fit_transform(df[target])
    else:
        le = None
        
    # Sub-sample flag lets us create a smaller, balanced version of the data set for speedy
    # investigation. Not balancing classes because we want to see, e.g., if the upsamplers 
    # would be helpful on the full (unbalanced) training set
    if subsamp:
        df = df.sample(5000, random_state=3)

    # Reshuffle the dataframe just in case...
    df = df.sample(frac=1.0, random_state=12).reset_index()

    return df, le


train_path = '../input/train.json'
target = 'cuisine'

train, le = load_and_process_data(
    train_path,
    target=target,
    subsamp=True
)


# # Getting a Baseline
# Next, we instantiate our classifiers. Here, we're going to use an SVM because it seems to be the best performing single model (not too suprising; SVMs tend to peform well on text problems).
# 
# Our parameters are courtesy of Shivam Bansal in [this kernel](https://www.kaggle.com/shivamb/tf-idf-with-ovr-svm-what-s-cooking).
# 
# Notice how high the **C** is! That might make it perform well on the text, but would probably make it prone to overfitting if we add more features or get too high dimensional... Note also that the **max_iter** is -1, meaning it'll run forever if it can't converge on a solution. I've found that with the right (or wrong) preprocessing, it sometimes times out the kernel.
# 
# We'll then toss it in a pipeline with a TF-IDF vectorizer with nothing fancy about it and see what happens...

# In[ ]:


# Pre-tuned parameters courtesy of Shivam Bansal
# https://www.kaggle.com/shivamb/tf-idf-with-ovr-svm-what-s-cooking
svm_clf = svm.SVC(
    C=100,  # penalty parameter
    kernel='rbf',  # kernel type, rbf working fine here
    degree=3,  # default value
    gamma=1,  # kernel coefficient
    coef0=1,  # change to 1 from default value of 0.0
    shrinking=True,  # using shrinking heuristics
    tol=0.001,  # stopping criterion tolerance
    probability=True,  # no need to enable probability estimates
    cache_size=200,  # 200 MB cache size
    class_weight=None,  # all classes are treated equally
    verbose=False,  # print the logs
    max_iter=-1,  # no limit, let it run
    decision_function_shape=None,  # will use one vs rest explicitly
    random_state=None
)

clf = pipeline.Pipeline(steps=[
    ('vec', feature_extraction.text.TfidfVectorizer()),
    ('clf', svm_clf),
])
clf = multiclass.OneVsRestClassifier(clf, n_jobs=-1)


# In[ ]:


score_base = model_selection.cross_val_score(
    clf,
    train['text'],
    train[target],
    cv=3,
)

print(f'Training score on base case: {score_base.mean()}')


# That's not a bad result, especially given our subsample size! (Training on the full data set should get you in the ballpark of ~0.80% accuracy).

# # Exploring
# Now the fun part: Exploring what might increase our accuracy. To do this, we'll need two things: a base_case and a test_case and a way of deciding whether the difference in accuracy between the two is meaningful or noise.
# 
# To do that, we'll create a little function to use the chi-squared test to find the p-value (the probability that the difference would occur by random chance, rather than real differences in the data), which will help us decide whether the difference is likely signal or likely noise.
# 
# By convention, legitimate scientistific inquiry uses p < 0.05 as the cut-off for signal vs noise. If the p-value is less than 0.05, there's only a 5% chance that the difference would arise purely from noise, and therefore we can be reasonably comfortable that we're looking at something meaningful.
# 
# But since we're just pursuing general lines of inquiry, we're going to take a higher p-value of 0.1 as the cut-off for our none-too-important experiments. Meaning: if the difference between the two classifiers produces a p-value of greater than 0.1, we're going to discount it. Otherwise, we're going to pursue that inquiy.
# 
# Note: A higher or lower p-value might be appropriate here. Higher would be appropriate since we're just pursuing general lines of inquiry, not trying to draw scientific conclusions. A little noise never hurt anyone, except taken too far it becomes a recipe for over-fitting. A lower p-value might be appropriate if we're running dozens of experiments, because otherwise we're going to run into some [look-elsewhere effect](https://en.wikipedia.org/wiki/Look-elsewhere_effect).

# In[ ]:


def check_difference(a1, s1, a2, s2):
    """
    Given the accuracy and sample-size of two different classifiers, finds the chi-squared statistic and p-value
    :param a1: Accuracy % of base-case
    :param s1: Training examples in base-case
    :param a2: Accuracy % of test-case
    :param s2: Training examples in test-case
    :return: x (chi-squared statistic) and p (p-value)
    """

    observed = np.array([a1 * s1, a2 * s2])
    expected = np.array([a1 * s1, a1 * s2])
    x, p = chisquare(observed, expected)

    print(f'Training score on base case: {a1}')
    print(f'Training score on test case: {a2}')
    if a2 <= a1:
        print('Oops. No improvement!')
    elif p < 0.1:
        print(f'Statistically significant difference @ p-value of {p}')
    else:
        print(f'Not a statistically significant difference @ p-value of {p}')
    return x, p


# ## Outlier Removal
# First, we'll test whether removing outliers improves our results. To do this, we'll use an isolation forest to spot likely outliers, and remove them from one set of data.
# 
# In terms of specific implementation, we're going to next our outlier removal within our training pipeline. We do this because we want to avoid leakage of data between the training and test sets (we don't want the initial fit of the outlier remover to learn about the test-set outliers), and pipelines will save us a lot of lines of code because they play well with the cross_val_score function. (Otherwise, we'd need to write our own custom cross-validation loop. No thx).
# 
# Instead of an sklearn pipeline, we'll use a similar imblearn pipeline, and basically copy the [example code](http://contrib.scikit-learn.org/imbalanced-learn/dev/auto_examples/plot_outlier_rejections.html#sphx-glr-auto-examples-plot-outlier-rejections-py) from the docs.

# In[ ]:


def outlier_rejection(X, y):
    """This will be our function used to resample our dataset."""
    model = ensemble.IsolationForest(
        n_estimators=1000,
        random_state=3)
    model.fit(X)
    y_pred = model.predict(X)
    return X[y_pred == 1], y[y_pred == 1]


reject_sampler = FunctionSampler(func=outlier_rejection)
clf_no_outliers = imblearn_pipeline.Pipeline(steps=[
    ('vec', feature_extraction.text.TfidfVectorizer()),
    ('outlier_removal', FunctionSampler(func=outlier_rejection)),
    ('clf', svm_clf),
])
clf_no_outliers = multiclass.OneVsRestClassifier(clf_no_outliers, n_jobs=-1)


# Then we'll score a classifier on the training data sans outliers...

# In[ ]:


score_no_outliers = model_selection.cross_val_score(
    clf_no_outliers,
    train['text'],
    train[target],
    cv=3,
)


# And compare the results! Because we've got our outlier-remover within a pipeline and then nested within a cross-validation loop, we don't know for sure how big the training data set is after removing outliers. But I'm gonna guess is about 80% the size of the original training set, based on earlier tests. (Spoiler alert: it won't matter because the accuracy is going to drop).

# In[ ]:


a1, s1, a2, s2 = score_base.mean(), len(train['text']), score_no_outliers.mean(), len(train['text']) * 0.8
x, p = check_difference(a1, s1, a2, s2)


# Slight drop. No dice there. Probably that means that there are not a lot of outliers in the data, and all we're doing with outlier removal is reducing our number of training examples. (Per the discussion above re: pipelines, note that if we don't use a pipeline and train on the whole data set, we actually get a modest but not statistically significant improvement... maybe due to information leakage?)

# ## Upsampling
# Next up: re-balancing our classes.
# 
# As others have pointed out, our classes are quite imbalanced. E.g., we have less than 500 examples of Brazilian food, but over 4000 examples of Korean food! So even a very stupid classifier would be smart to just default to "Korean" whenever it's confused.
# 
# Here, down-sampling (removing training examples from over-represented classes) is not a great option because it'd leave us with a lot less training data. Instead, we're going to up-sample our data. Upsampling is a technique that basically "adds" training examples to under-represented classes by duplicating existing or synthesizing new training examples.
# 
# We're going to use [SMOTE](http://contrib.scikit-learn.org/imbalanced-learn/stable/generated/imblearn.over_sampling.SMOTE.html), a **s**ynthetic **m**inority **o**ver-**s**amplilng **te**chnique (see what I did there?). SMOTE essentially synthesizes new training examples using characteristics from existing training examples. You can read up about it [here](https://arxiv.org/pdf/1106.1813.pdf), or check out the docs for imblearn, which have pretty pictures.
# 
# I picked SMOTE mostly because I like the name. But it's worth testing [different methods](http://contrib.scikit-learn.org/imbalanced-learn/stable/over_sampling.html#a-practical-guide).

# Anyway, instantiate a SMOTE pipeline (notice we're using a different pipeline than the native sklearn pipeline here)...

# In[ ]:


smote_clf = imblearn_pipeline.Pipeline(steps=[
    ('vec', feature_extraction.text.TfidfVectorizer()),
    ('up_sample', SMOTE(kind='regular')),
    ('clf', svm_clf),
])
smote_clf = multiclass.OneVsRestClassifier(smote_clf, n_jobs=-1)


# ...and run our test (using the original training data, not our outlier removed training data).

# In[ ]:


score_smote = model_selection.cross_val_score(
    smote_clf,
    train['text'],
    train[target],
    cv=3,
)


# And check our difference:

# In[ ]:


a1, s1, a2, s2 = score_base.mean(), len(train['text']), score_smote.mean(), len(train['text']) * 4
x, p = check_difference(a1, s1, a2, s2)


# SMOTE produced a very slight improvement, but with a p-value of 0.92, it's so small that it's virtually impossible to say it's due to anything but chance. (Again, we're not sure how much the upsampling increases the training set size... but based on earlier estimates, it's about an 4x increase. But again, it doesn't actually matter because our score went down).

# But let's also try a simpler method: randomly duplicating training examples from the minority class. We actually should have run this one first, maybe. It'll give us a baseline for whether the imbalanced classes are even an issue for our classifier, without any risk that our method for synthesizing new training examples is synthesizing training examples that are actually confusing our classifier.

# In[ ]:


rand_up_clf = imblearn_pipeline.Pipeline(steps=[
    ('vec', feature_extraction.text.TfidfVectorizer()),
    ('up_sample', RandomOverSampler(random_state=3)),
    ('clf', svm_clf),
])
rand_up_clf = multiclass.OneVsRestClassifier(rand_up_clf, n_jobs=-1)

score_rand_up = model_selection.cross_val_score(
    rand_up_clf,
    train['text'],
    train[target],
    cv=3,
)

a1, s1, a2, s2 = score_base.mean(), len(train['text']), score_rand_up.mean(), len(train['text']) * 4
x, p = check_difference(a1, s1, a2, s2)


# Again, no help there. So it looks like the imbalance of the classes is not a huge deal here. And actually, that's kind of good, since the imblearn pipeline doesn't play super well with certain sklearn features, so sticking to a pure sklearn pipeline gives us more flexibility for other tests we might want to run!

# ## Stemming
# Let's try stemming next.
# 
# Stemming is a method for reducing words to simple "stem." So, for instance, "powered sugar" becomes "powd sug." Stemming does't produce the Queen's English, but it's quite fast compared to the alternative (lemmatization). Plus our TF-IDF vectorizer doesn't care.
# 
# Stemming is potentially useful for a few reasos. First, it might help with typos where words have had their endings cut off (there are quite a few in the data set on visual inspection). E.g., "powdered", "powder", and "powde" all stem down to "powd". Second, it wil reduce our total vocabulary (we go from 2 words, "powdered" and "powder", to just 1, "powd", thereby reducing dimensionality). Third, obviously, it will help our classifier see that "chicken" and "chickens" are probably the same ingredient, so it can interpret that data correctly.
# 
# In terms of implemtation, we don't need to worry about data leakage because the stemming algorithm doesn't rely on the characteristics of the data. So we're just going to use our "stem" flag to create a new, stemmed data set. (You can also use a pipeline if you want!)

# In[ ]:


train_stemmed, le = load_and_process_data(
    train_path,
    target=target,
    subsamp=True,
    stem=True
)

score_stemmed = model_selection.cross_val_score(
    clf,
    train_stemmed['text'],
    train_stemmed[target],
    cv=3,
)

a1, s1, a2, s2 = score_base.mean(), len(train['text']), score_stemmed.mean(), len(train_stemmed['text'])
x, p = check_difference(a1, s1, a2, s2)


# Dang. No improvement there. (Using other seeds, I've gotten different results, with some p-values as low as ~0.25...). But it looks like stemming doesn't hurt much, so I might do it for speed. Specifically:

# In[ ]:


v = feature_extraction.text.TfidfVectorizer()
ar = v.fit_transform(train['text'])
print(ar.shape)

ar_stemmed = v.fit_transform(train_stemmed['text'])
print(ar_stemmed.shape)


# On our little sample, stemming reduces our features from ~2000 to ~1700. That's likely to speed up our classifier and reduce the chances of over-fitting, even if it doesn't produce a verifiable improvement. So:

# In[ ]:


base_score = score_stemmed
train = train_stemmed


# ## 1-Grams vs. N-Grams
# Well, what about generating n-grams (phrases of n length)? We've got ingredient lists such as "tomato paste, fresh coriander, garam masala." Until now, we've just been vectorizing each word separately ([tomato, paste, fresh, coriander, garam, masala]). But what if we tokenize the whole ingredient ([tomato paste, fresh coriander, garam masala]).
# 
# The theory here is that "fish sauce" together encodes more data than "fish" and "sauce" separately, which is probably correct. But there are two big downsides to using n-grams instead of 1-grams here:
# 1. "fresh coriander" is probably no different than just "coriander", but vectorizing them separately means the classifier can't tell they're the same ingredient. Instead, I'd rather vectorize them as ['fresh', 'coriander'] and let the model figure out the value, if any, of the word "fresh".
# 2. n-grams will very likely create a much higher dimensional space than 1-grams, which will increase the chance of over-fitting, especially given our C of 100.

# In[ ]:


def tokenizer(x):
    # We're using a function rather than a lambda because the lambda function causes 
    # problems with the pipeline. Not sure why. Pickle traceback pickle something.
    return [i.strip() for i in x.split(',') if i.strip()]


t_clf = pipeline.Pipeline(steps=[
    ('vec', feature_extraction.text.TfidfVectorizer(tokenizer=tokenizer)),
    ('clf', svm_clf),
])
t_clf = multiclass.OneVsRestClassifier(t_clf, n_jobs=-1)

score_t = model_selection.cross_val_score(
    t_clf,
    train['text'],
    train[target],
    cv=3,
)

a1, s1, a2, s2 = score_base.mean(), len(train['text']), score_t.mean(), len(train['text'])
x, p = check_difference(a1, s1, a2, s2)


# Looks like our suspicions were correct. That's a big drop in accuracy. Whatever new information we got from using n-grams instead of 1-grams, it was probably offset by the two potential problems we identified. n-grams roughly duble the number of features:

# In[ ]:


v = feature_extraction.text.TfidfVectorizer(tokenizer=tokenizer)
print(v.fit_transform(train['text']).shape)


# So let's stick with 1-grams.

# ## Stop-Word Removal
# Let's try one last test: removing stop-words. 
# 
# Stop-words are words that don't typically convey useful information, such as "and", "or", various pronouns, etc. But obviously our stop-word set for food is going to be different than our stop-word set for regular speech. So I went through the data and found a list of frequently occuring words that I'd say are good candidates for food-ish stop-words:

# In[ ]:


stop_words = [
    'low', 'less', 'non', 'nonfat', 'reduced', 'sodium', 'fat', 'boneless', 'skinless', 'fresh', 'gluten', 'extra',
    'virgin', 'free', 'range', 'dried', 'large', 'dry', 'fine', 'coarse', 'unsalted', 'chopped', 'minced', 'ground',
    'peeled', 'plain', 'cold', 'warm', 'freshly', 'light', 'shredded', 'diced', 'cooked', 'cheese', 'ice',
    'unsweetened', 'sweetened', 'unsweet', 'semisweet', 'crushed', 'organic', 'halves', 'halved', 'unbleached',
    'slivered', 'unsalted',
]

stemmer = LancasterStemmer()
stop_words = [stemmer.stem(x) for x in stop_words]


# (Note we want to stem them if we're using the stemmed data). Let's feed these into our TF-IDF vectorizer and see the difference:

# In[ ]:


clf_stop = pipeline.Pipeline(steps=[
    ('vec', feature_extraction.text.TfidfVectorizer(stop_words=stop_words)),
    ('clf', svm_clf),
])
clf_stop = multiclass.OneVsRestClassifier(clf_stop, n_jobs=-1)

score_stop = model_selection.cross_val_score(
    clf_stop,
    train['text'],
    train[target],
    cv=3,
)

a1, s1, a2, s2 = score_base.mean(), len(train['text']), score_stop.mean(), len(train['text'])
x, p = check_difference(a1, s1, a2, s2)


# Here, we see a slight improvement, but not a statistically significant one. Maybe it is irrelevant whether food is minced, diced, ground, or chopped, but it's also not too hard to imagine that different regions would use different preparatory techniques, etc. But more industrious kagglers than I might want to play around with the list of stop-words and see what they can see.
# 
# At any rate, I propose we remove stop-words because it reduces dimensionality a little, and doesn't seem to hurt. Specifically:

# In[ ]:


v = feature_extraction.text.TfidfVectorizer(stop_words=stop_words)
print(v.fit_transform(train['text']).shape)


# # Conclusion
# Of the five preprocessing techniques we explored, none produced a statistically significant improved. However, several techniques did reduce our dimensionality without materially affecting our accuracy, which should improve our training times and make our classifier more robust against over-fitting... so we've gained some useful insights.
# 
# More generally, though, we've got a little test harness we can use to quickly investigate whether a given technique improves our accuracy, similar to what we might be able to do with a parameter search, but with a bit more flexibility. That should allow us to quickly investigate different lines of inquiry, and shut them down if they don't seem to be going anywhere.
# 
# In terms of next steps, here are two concrete preprocessing suggestions:
# 
# 1. **Adding meta-features** (e.g., # of ingredients, # of different types of special characters, etc). My initial exploration suggested meta-features didn't help (likely due to over-fitting), but maybe I didn't find the right meta-feature. Best practice for that would a [FeatureUnion pipeline](http://scikit-learn.org/stable/auto_examples/hetero_feature_union.html#sphx-glr-auto-examples-hetero-feature-union-py).
# 2. **Combining 1-grams and n-grams** (e.g., feeding both 1-grams and n-ngrams into a classifier). Again, I didn't find this to help, but if you were aggressive about identifying and dumping junk 1-grams and junk n-grams, and tried to keep the TF-IDF's vocabulary at around the same number of words, I could see that helping. 
# 
# Until next time, I hope that this was helpful for getting the creative juices flowing.
