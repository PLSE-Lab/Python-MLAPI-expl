#!/usr/bin/env python
# coding: utf-8

# # Overview
# 
# I completed this project as part of my Data Science course at the Flatiron School. Check it out! https://flatironschool.com/
# 
# This project uses a Kaggle dataset to predict gene classifications. In this dataset, we are given multiple genetic variants and various properties of each. Expert raters at different laboratories rated these variants based on their perceived clinical classifications, with ratings ranging from Benign to Pathogenic. The target variable is whether the raters have clinical classifications that are concordant, meaning that they are in the same clinical category.
# 
# I approach this with the OMESN framework. Data cleaning turns out to be the most substantial part of this project. In the end, I test a few different modeling approaches and present the results of the highest scoring model.

# ## Initialization

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# # Obtaining the Data
# 
# For this project, I downloaded the dataset from the Kaggle page as a csv.

# In[3]:


df = pd.read_csv('../input/clinvar_conflicting.csv')
df.head()


# In[ ]:


df.info()


# # Scrubbing the Data
# 
# There seem to be a number of feilds with missing data and incorrect types. In this section, I scrub the dataset squeaky-clean.

# ## Very Low Incidence Features
# 
# Here I drop features with under 600 entries (1% of dataset).

# In[ ]:


df = df.drop(['CLNDISDBINCL', 'CLNDNINCL', 'CLNSIGINCL', 'SSR', 'DISTANCE', 'MOTIF_NAME', 'MOTIF_POS', 'HIGH_INF_POS', 'MOTIF_SCORE_CHANGE'], axis = 1)


# ## Low Incidence Features
# 
# Here I dichotomize features that are present for less than half the dataset, 1 indicating that data are present, 0 otherwise.

# In[ ]:


for var in ['CLNVI', 'INTRON', 'BAM_EDIT', 'SIFT', 'PolyPhen', 'BLOSUM62']:
    df[var] = df[var].apply(lambda x: 1 if x == x else 0).astype('category')
    print(df[var].value_counts())


# ## Target: CLASS
# 
# The CLASS vartible is the target variable, which indicates whether there were conflicting submissions.

# In[ ]:


df = df.rename({'CLASS': 'target'}, axis = 1)
df['target'] = df['target'].astype('category')
df['target'].value_counts()


# ## CHROM
# 
# This variable captures the chromosome on which the variant is located. This should be a categorical variable. Strangely, there are two "16"s in this list, which should be combined. I do this by converting it to a strip and striping the spaces before making it into a category.

# In[ ]:


df['CHROM'].value_counts()


# In[ ]:


df['CHROM'] = df['CHROM'].astype('str').apply(lambda x: x.strip())
df['CHROM'] = df['CHROM'].astype('category')


# ## POS
# 
# This variable captures position of the gene on the chromosome. Will need to treat this with care in analysis, since it depends on CHROM.

# In[ ]:


df['POS'].describe()


# ## REF, ALT, Allele
# 
# These variables are for capture variant alleles - should be categorical.

# In[ ]:


for var in ['REF', 'ALT', 'Allele']:
    print(df[var].value_counts()[0:10])


# There are a lot of low-frequency categories - I will lump them together into an "other" category.

# In[ ]:


for var in ['REF', 'ALT', 'Allele']:
    df[var] = df[var].apply(lambda x: 'O' if x not in ['A', 'C', 'G', 'T'] else x).astype('category')


# ## AF_ESP, AF_EXAC, and AF_TGP
# 
# These variables capture the allele frequency as found in other datasets. They are almost all zero, so I dichotomize them into zero vs non-zero.

# In[ ]:


df[['AF_ESP', 'AF_EXAC', 'AF_TGP']].describe()


# In[ ]:


df[['AF_ESP', 'AF_EXAC', 'AF_TGP']].hist()


# In[ ]:


df['AF_ESP'] = df['AF_ESP'].apply(lambda x: 1 if x > 0 else 0).astype('category')
df['AF_EXAC'] = df['AF_EXAC'].apply(lambda x: 1 if x > 0 else 0).astype('category')
df['AF_TGP'] = df['AF_TGP'].apply(lambda x: 1 if x > 0 else 0).astype('category')


# ## CLNDISDB
# 
# This variable contains IDs for diseases in other databases. This variable has a large number of values, so it will be difficult to use it. I see that different values for this variable often contain the same identifiedrs, making the values arguable not unique (e.g. 'MedGen:CN169374' appears in multiple values). I choose to drop it.

# In[ ]:


print(len(df['CLNDISDB'].unique()))
df['CLNDISDB'].value_counts()[0:10]


# In[ ]:


df = df.drop('CLNDISDB', axis = 1)


# ## CLNDN
# 
# This captures the preferred disease name using the identifiers from CLNDISDB. This may be cleaner than the other variable, and is probably important for prediction, so I will attempt to clean it.

# In[ ]:


print(len(df['CLNDN'].unique()))
df['CLNDN'].value_counts()[0:20]


# Each value is a list of diseases. It seems like I could clean this by creating dummy variables for specific common diseases in each list. I will create dummies for the top 100 diseases.

# In[ ]:


name_df = df['CLNDN'].str.split(pat = '|', expand = True)
name_df.head()
top_100_dn = name_df.apply(pd.value_counts).sum(axis=1).sort_values(ascending = False)[0:100]
print(top_100_dn[0:10])

top_100_dn_list = list(top_100_dn.index)
print(top_100_dn_list[0:10])


# In[ ]:


for dn in top_100_dn_list:
    df[dn] = df['CLNDN'].apply(lambda x: 1 if dn in x else 0).astype('category')
df = df.drop('CLNDN', axis = 1)


# In[ ]:


print(df.columns)


# ## CLNHGVS
# 
# This variable is all unique values that I don't understand related to HGVS expression. I choose to drop it.

# In[ ]:


print(len(df['CLNHGVS'].unique()))
df = df.drop('CLNHGVS', axis = 1)


# ## CLNVC
# 
# This variant type variable is almost all one value - I will turn it into a categorical variable by consolidating low-incidence types.

# In[ ]:


print(df['CLNVC'].value_counts())


# In[ ]:


clnvc_types = ['single_nucleotide_variant', 'Deletion', 'Duplication']
df['CLNVC'] = df['CLNVC'].apply(lambda x: x if x in clnvc_types else 'Other').astype('category')


# ## MC
# 
# Molecular consequence is a categorical variable, need to clean up rare values. Since values are lists of consequences, I will do this similarly to how I did it for the names, splitting up the series and coding dummies.

# In[ ]:


df['MC'].value_counts()[0:10]


# In[ ]:


name_df = df['MC'].str.split(pat = '[|,]', expand = True)
name_df.head()
top_mc = name_df.apply(pd.value_counts).sum(axis=1).sort_values(ascending = False)[0:20]
print(top_mc)

top_mc_list = [x for x in list(top_mc.index) if 'SO:' not in x]
print(top_mc_list)


# In[ ]:


df['MC'] = df['MC'].fillna('unknown')
for mc in top_mc_list:
    df[mc] = df['MC'].apply(lambda x: 1 if mc in x else 0).astype('category')
    print(df[mc].value_counts())
df = df.drop('MC', axis = 1)


# ## ORIGIN
# 
# Here is the description: "Allele origin. One or more of the following values may be added: 0 - unknown; 1 - germline; 2 - somatic; 4 - inherited; 8 - paternal; 16 - maternal; 32 - de-novo; 64 - biparental; 128 - uniparental; 256 - not-tested; 512 - tested-inconclusive; 1073741824 - other" Since almost all have origin 1 (germline), I will recode this to have 0 for all other values to make it a dummy variable.

# In[ ]:


df['ORIGIN'] = df['ORIGIN'].fillna(0).apply(lambda x: 1 if x == 1.0 else 0).astype('category')


# ## Consequence
# 
# This variable is similar to MC, but with slightly different values. I'm not sure why. I will use it to update the MC dummy variables from before.

# In[ ]:


name_df = df['Consequence'].str.split(pat = '&', expand = True)
name_df.head()
top_mc = name_df.apply(pd.value_counts).sum(axis=1).sort_values(ascending = False)
print(top_mc[0:20])


# In[ ]:


for mc in top_mc_list:
    mc2 = mc + '2'
    df[mc2] = df['Consequence'].apply(lambda x: 1 if mc in x else 0).astype('category')
    df[mc] = df[[mc, mc2]].apply(lambda x: max(x[mc], x[mc2]), axis = 1).astype('category')
    print(df[mc].value_counts())
    df=df.drop(mc2, axis = 1)
df = df.drop('Consequence', axis = 1)


# ## IMPACT
# 
# Categorical variable capturing variant impact

# In[ ]:


df['IMPACT'].value_counts()


# In[ ]:


df['IMPACT'] = df['IMPACT'].astype('category')


# ## SYMBOL
# 
# This variable is the Gene symbol/ID. It has many values - I will make it categorical, but only keep the top 100 values, recoding the rest as "Other".

# In[ ]:


len(df['SYMBOL'].unique())


# In[ ]:


df['SYMBOL'].value_counts()[0:10]


# In[ ]:


top_100_symb = df['SYMBOL'].value_counts()[0:100].index
df['SYMBOL'] = df['SYMBOL'].apply(lambda x: x if x in top_100_symb else 'Other').astype('category')


# In[ ]:


df['SYMBOL'].value_counts()[0:100]


# ## Feature
# 
# This is an ID associated with gene name - deleting due to redundancy

# In[ ]:


df = df.drop('Feature', axis = 1)


# ## Feature_type and BIOTYPE
# 
# These features have little information (almost all records have same value), so I drop them.

# In[ ]:


for var in ['Feature_type', 'BIOTYPE']:
    print(df[var].value_counts())
    df = df.drop(var, axis = 1)


# ## EXON
# 
# This captures the relative exon number. Given the very large numbers of unique values, I choose to drop it.

# In[ ]:


len(df['EXON'].unique())


# In[ ]:


df = df.drop('EXON', axis = 1)


# ## cDNA_position, CDS_position, Protein_position
# 
# These represent relative positions of the base pair in various ways. These are all distance measures, which I think are irrelevant to the problem at hand, and difficult to clean so I drop them.

# In[ ]:


df = df.drop(['cDNA_position', 'CDS_position', 'Protein_position'], axis = 1)


# ## Amino_acids, Codons
# 
# These have a large number of unique values, so I drop them.

# In[ ]:


df = df.drop(['Amino_acids', 'Codons'], axis = 1)


# ## STRAND
# 
# Categorical: defined as + (forward) or - (reverse)

# In[ ]:


df['STRAND'].value_counts()


# In[ ]:


df['STRAND'] = df['STRAND'].fillna(df['STRAND'].mode())
df['STRAND'] = df['STRAND'].astype('category')


# ## LoFtool
# 
# Numeric variable: Loss of Function tolerance score for loss of function variants. Will fill missing values with median.

# In[ ]:


df['LoFtool'] = df['LoFtool'].fillna(df['LoFtool'].median())


# ## CADD_PHRED, CADD_RAW
# 
# Different scores of deleteriousness - I keep them and fill missing values with medians.

# In[ ]:


df['CADD_PHRED'] = df['CADD_PHRED'].fillna(df['CADD_PHRED'].median())


# In[ ]:


df['CADD_RAW'] = df['CADD_RAW'].fillna(df['CADD_RAW'].median())


# ## Scaling numeric variables

# In[ ]:


from sklearn.preprocessing import StandardScaler

num_var_list = ['POS', 'LoFtool', 'CADD_PHRED', 'CADD_RAW']
scl = StandardScaler()
df[num_var_list] = scl.fit_transform(df[num_var_list])


# ## Separate target and features

# In[ ]:


target = df['target']
features = df.drop('target', axis = 1)


# # Exploring the Data

# In[ ]:


#Original columns
list(df.columns[0:23])


# In[ ]:


df.iloc[:, 0:23].info()


# In[ ]:


#Original feature set
orig_feat = list(features.columns[0:22])
orig_feat_cat = [x for x in orig_feat if x not in num_var_list]


# ## Numeric Variables
# 
# In this section I explore the numeric variables in the dataset. I find that there is a high correlation between CADD_PHRED and CADD_RAW, so I choose to drop one of them. I drop CADD_RAW due to the long right tail.

# In[ ]:


features[num_var_list].describe()


# In[ ]:


features[num_var_list].hist()


# In[ ]:


plt.figure(figsize=(8,8))
sns.heatmap(features[num_var_list].corr(),
            vmin=0,
            vmax=1,
            cmap='YlGnBu',
            annot=np.round(features[num_var_list].corr(), 2))


# In[ ]:


features = features.drop('CADD_RAW', axis = 1)


# ## Categorical Variables
# 
# Associations between categorical variables can be difficult to visualize. I use Cramer's V to understand the associations between each pair of categorical features, adapting this code: https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9

# In[ ]:


import scipy.stats as ss

def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x,y)
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
    rcorr = r-((r-1)**2)/(n-1)
    kcorr = k-((k-1)**2)/(n-1)
    return np.sqrt(phi2corr/min((kcorr-1),(rcorr-1)))


# In[ ]:


num_feat = len(orig_feat_cat)
cat_corr_arr = np.empty((num_feat, num_feat))
for i, row in enumerate(orig_feat_cat):
    for j, col in enumerate(orig_feat_cat):
        #print((i, j))
        cat_corr_arr[i, j] = cramers_v(features[row], features[col])
print(cat_corr_arr[0:5, 0:5])


# In[ ]:


plt.figure(figsize=(16, 14))
sns.heatmap(cat_corr_arr,
            vmin=0,
            vmax=1,
            cmap='YlGnBu',
            xticklabels = orig_feat_cat,
            yticklabels = orig_feat_cat,
            annot=np.round(cat_corr_arr, 2))


# I choose to drop the Allele, IMPACT, SYMBOL and PolyPhen variables due to high correlations.

# In[ ]:


features = features.drop(['Allele', 'IMPACT', 'SYMBOL', 'PolyPhen'], axis = 1)


# # Modeling
# 
# For this problem, I choose to use multiple classifiers to see how they compare. I start with a dummy classifier as a baseline for comparison. Then I proceed to Random Forest Classifier, Naive Bayes, and AdaBoost. I will test the effects of various parameter spefications on model performance.

# In[ ]:


from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.model_selection import GridSearchCV, train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, make_scorer


# In[ ]:


features = pd.get_dummies(features, drop_first = True)
print(features.columns)


# ## Evaluation Metric
# 
# For this problem I choose the F1 score, which balances precision and recall. Since there are fewer positive cases than negative one, I want to include recall as part of my metric, but I want to include precision as well to avoid over-classification.

# In[ ]:


f1_scorer = make_scorer(f1_score)


# ## Dummy
# 
# The F1 score for the Dummy classifier is 0.253, providing a point of comparison for other models.

# In[ ]:


dm_clf = DummyClassifier(random_state = 42)
mean_dm_cv_score = cross_val_score(dm_clf, features, target, scoring = f1_scorer, cv = 3).mean()
print("Mean Cross Validation F1 Score for Dummy Classifier: {:.3}".format(mean_dm_cv_score))


# ## Naive Bayes
# 
# Gaussian Naive Bayes doesn't seem like a natural fit, given that there are many one-hot encoded variables in this dataset, but I am curious whether its performance is better than the Dummy classifier - at 0.341 it does seem to be better. Bernoulli Naive Bayes does even better still.

# In[ ]:


gnb_clf = GaussianNB()
mean_gnb_cv_score = cross_val_score(gnb_clf, features, target, scoring = f1_scorer, cv = 3).mean()
print("Mean Cross Validation F1 Score for Gaussian Naive Bayes Classifier: {:.3}".format(mean_gnb_cv_score))


# In[ ]:


bnb_clf = BernoulliNB()
mean_bnb_cv_score = cross_val_score(bnb_clf, features, target, scoring = f1_scorer, cv = 3).mean()
print("Mean Cross Validation F1 Score for Bernoulli Naive Bayes Classifier: {:.3}".format(mean_bnb_cv_score))


# ## AdaBoost
# 
# I decide to fit AdaBooost next using decision tree and logistic regression classifiers. These models provide no improvement over Naive Bayes.

# In[ ]:


adb_clf = AdaBoostClassifier(base_estimator = DecisionTreeClassifier(), random_state = 42)
mean_adb_cv_score = cross_val_score(adb_clf, features, target, scoring = f1_scorer, cv = 3).mean()
print("Mean Cross Validation F1 Score for AdaBoost Decision Tree Classifier: {:.3}".format(mean_adb_cv_score))


# In[ ]:


adb_clf = AdaBoostClassifier(base_estimator = LogisticRegression(solver = 'lbfgs'), random_state = 42)
mean_adb_cv_score = cross_val_score(adb_clf, features, target, scoring = f1_scorer, cv = 3).mean()
print("Mean Cross Validation F1 Score for AdaBoost Logistic Regression Classifier: {:.3}".format(mean_adb_cv_score))


# ## XGBoost
# 
# I next decide to use XGBoost, a popular boosting algorithm. This does not seem to improve performance.

# In[ ]:


import xgboost as xgb
xgb_clf = xgb.XGBClassifier(seed = 123)
mean_xgb_cv_score = cross_val_score(xgb_clf, features, target, scoring = f1_scorer, cv = 3).mean()
print("Mean Cross Validation F1 Score for XGBoost Classifier: {:.3}".format(mean_xgb_cv_score))


# ## Random Forest
# 
# Lastly I fit a Random Forest model, which has an F1 score of 0.212.

# In[ ]:


rf_clf = RandomForestClassifier(n_estimators = 100, random_state = 42)
mean_rf_cv_score = cross_val_score(rf_clf, features, target, scoring = f1_scorer, cv = 3).mean()
print("Mean Cross Validation F1 Score for Random Forest Classifier: {:.3}".format(mean_rf_cv_score))


# ## Bernoulli Naive Bayes Tuning
# 
# Here I use grid search and random search to tune the highest performing model: Bernoulli Naive Bayes. The best hyperparameters yeild an F1 score of 0.437. This is not great, but substantially better than the dummy model.

# In[ ]:


bnb_param_grid = {
'alpha': [0.1, 0.5, 1, 2, 5],
'fit_prior': [True, False]
}


# In[ ]:


import time
start = time.time()
bnb_grid_search = GridSearchCV(bnb_clf, bnb_param_grid, scoring = f1_scorer, cv = 3)
bnb_grid_search.fit(features, target)

print("Cross Validation F1 Score: {:.3}".format(bnb_grid_search.best_score_))
print("Total Runtime for Grid Search on Bernoulli Naive Bayes: {:.4} seconds".format(time.time() - start))
print("")
print("Optimal Parameters: {}".format(bnb_grid_search.best_params_))


# # Interpreting the Model
# 
# Below I explore the final model to better understand the important properties of the model.

# ## Performance
# 
# Looking at the performance, I see that the unbalanced nature of the classes seem to yeild a fair amount of misclassification. Specifically, a number of cases where experts agreed were classified as being cases of disagreement (the upper right of the confusion matrix).

# In[ ]:


best_bnb = bnb_grid_search.best_estimator_


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(features, target, test_size = 0.33, random_state = 42)
best_bnb.fit(X_train, y_train)
y_hat_test = best_bnb.predict(X_test) 
bnb_confusion_matrix = confusion_matrix(y_test, y_hat_test)
print(bnb_confusion_matrix)
bnb_classification_report = classification_report(y_test, y_hat_test)
print(bnb_classification_report)


# ## Feature Probabilities
# 
# While Naive Bayes doesn't yeild feature importances, I am able to look at which features have the largest difference in predicted probability of being present between the two target classes. The allele frequency variables jump out as having the largest differences, as do genes associated with unknown disease variants.

# In[ ]:


feat_df = pd.DataFrame()
feat_df['prob_0'] = np.exp(best_bnb.feature_log_prob_[0])
feat_df['prob_1'] = np.exp(best_bnb.feature_log_prob_[1])
feat_df.index = features.columns
feat_df.head()


# In[ ]:


feat_df['ave_prob'] = feat_df.apply(lambda x: (x[0] + x[1])/2, axis = 1)
feat_df['prob_diff'] = feat_df.apply(lambda x: np.abs(x[0] - x[1]), axis = 1)
feat_df.head()


# In[ ]:


feat_df.sort_values('prob_diff', ascending=False).head(10)


# # Conclusion
# 
# In this analysis, I find that I am able to predict when experts will disagree about gene severity moderately well, with an F1 score of 0.437 for my final Bernoulli Naive Bayes model. This is a notable improvement over the dummy model, with F1 score of 0.253. This model can be used to prioritize research on gene variants with debatable severity. However, there is still a fair amount of misclassification, specifically with concurrences being classified as disagreements more often than warranted. Future analysis could look for ways to better balance the overall accuracy with the recall of the model.
