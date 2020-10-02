#!/usr/bin/env python
# coding: utf-8

# This is a first draft of an analysis of the Genetic Variant Classification dataset. Comments and feedback are welcome!
# 
# **Methodology:**
#  - Initially the data is cleaned: NaN entries are either droped or imputed, and most numerical data is normalized to lie in the range [0,1]
#  - Several new features are engineered, the most novel being a 'weight' feature for the CLNDN, MC and Consequence columns; see below for further details.
#  - The dataset is then balanced (initially ~75% of data belongs to class 0) and categorical data is encoded numerically (only a selection of the columns are used in this first draft).
#  - A RandomForest classifier is repeatedly applied to splits of the dataset (90% train, 10% test) and accuracy statistics are collected (see "Findings" below).
# 
# **Findings:**
# 
#    Following [this kernel](https://www.kaggle.com/vasilyb/clinvar-identifying-conflicting-genetic-variants), we use an (averaged) AUROC score for measuring the efficacy the RandomForestClassifier. Iterating over various samples of the processed data yields an average AUROC score greater than 0.8. By adjusting the RandomForestClassifier, it seems possible to obtain a TPR / FPR ratio of 0.8 / 0.2.
#    
# **ToDo List:**
# 1. Incorporate other key features into the model, e.g.  cDNA / CDS / Protein_positions, Amion_Acids, Exon, Codons
# 2. Hypertune the parameters for the RandomForestClassifier to search for more optimal prediction models
# 3. Apply other common ML algorithms

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
from scipy.stats import binom
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_curve, auc, f1_score, recall_score, precision_score, classification_report
from sklearn.model_selection import train_test_split

verbose = False

#alpha is a parameter to control how much of the final dataset is used in training the random forest classifier
alpha = 0.9

#n_models determines how many times to build and test a classifier from the final dataset
n_models = 20

#this is a random parameter to control the random state of the model, for reproducibility.
rnd=0


# The "relative_weights" function in the next code box is used to engineer several new features in the dataset. Here, the procedure is briefly described:
# 
#         The columns CLNDN, MC, and Consequence of the given dataset are 'multi-entires', in the sense that each entry contains a string that lists (possibly several) sub-features of the particular entry in the data set. These sub-features are first collected into a dictionary each entry of which is of the form 
# 
#         "sub-feature" : [overall count of sub-frequency, count of sub-frequency in class 1]
# 
#         Each sub-feature is then assigned a class, 0 or 1, based on whether rows with sub-feature were more often 0 or 1 (strategy is winner-take-all; the borderline case of equal frequency will not matter, see the expression for predictive power below). Sub-features are also assigned a weight, a measure of their predictive power, given by
# 
#         rho = count of sub-feature in class 1 / overall count of sub-feature
#         predictive power = (2 rho - 1)^2
# 
#         Thus, if the sub-feature appeared equally often in both classes, the predictive power is 0, while if it appeared more often in one class or the other, the predictive power is close to 1. Finally, we use the sub-feature class to weight each row using the formula
# 
#         weight = (1 / # sub-features in this row) * Sum [(-1)^ (sub-feature class) * predictive power]
# 
#         Here the sum is taken over all sub-features appearing in this row. The effect of this is that, if a row contains sub-features, the majority of which belong to class 0, then the row is assigned a weight near 1, while if the majority of the sub-features in a row belong to class 1, the row is assigned weight -1. Weights near 0 mean that the sub-features appearing in the given row are relatively balanced between the two classes.
# 
# The remaining helper functions in the next code box are just for convenience in pre-processing other parts of the dataset.

# In[ ]:


def relative_weights(df, class_name, col_name, delimiter, synonyms ={}, reduce = False):
    #df is the input dataframe
    #col_name is the name of the column whose relative weights we want to compute
    #delimiter is how the entries in col_name are divided into classes
    #synonyms lists any synonyms among words that may appear in col_name entries, e.g. not provided = not specified

    counts = {}
    for index, row in df.iterrows():
        #extract the words for this row
        new_words = row[col_name].split(delimiter)
        #make synonym substitutions and remove any duplicates
        new_words = list(set([synonyms.get(j,j) for j in new_words]))
        for n in new_words:
            #for existing words, we increase the overall count and the class frequency
            #if the word is new, create an entry [1, (class of this row)]
            if n in counts:
                counts[n][0] +=1
                counts[n][1] += row[class_name]
            else:
                counts[n] = [1,row[class_name]]
    n_entries = df.shape[0]
    weights = dict([(key, (np.power((2*counts[key][1]/counts[key][0] -1),2), np.floor(0.5 + counts[key][1]/counts[key][0]))) 
                      for key in counts])

    #create a subfunction to get the weight of a given entry
    #this will allow us to use a lambda function to map over rows of the dataframe
    def get_weight(row, dictionary, delimiter_sub, syns):
        kwds = row.split(delimiter_sub)
        #make synonym substitutions
        kwds = list(set([syns.get(j,j) for j in kwds]))
        return np.sum([np.power((-1),dictionary[j][1]) *dictionary[j][0] for j in kwds])/len(kwds)
        
    return df[col_name].apply(lambda x: get_weight(x, weights, delimiter,synonyms))

#for converting Chromosome numbers to strings
def to_int(s):
    try:
        return int(s)
    except ValueError:
        if s == 'X':
            return 23
        else:
            return 24
    
#for shortening base-pair entries
def string_to_short(s):
    if len(s)>1:
        return 'L'
    else:
        return s
    
#Extract training and testing data from a dataset
def get_train_test(df, alpha = 0.9, clss=None):
    train = df.sample(frac = 0.9, axis = 0)
    test = df[~df.index.isin(train.index)]
    
    cols_not_class = [j for j in df.columns.values]
    if clss in cols_not_class:
        cols_not_class.remove(clss)
    return train[cols_not_class], train[clss], test[cols_not_class], test[clss]


# The next code box contains the initial pre-processing of the data. A few remarks are in order:
# 1. We begin by dropping columns which have 50% of their entries as NaN; it seems difficult to use such sparse data in making predictions.
# 2. Some entries in the CHROM column contain numeric entries, some alphabetic (e.g. 22 vs. '22'). All are converted to integers (and the extra entires, 'X' and 'MT' are assigned numbers 23 and 24 respectively.
# 3. The columns REF, ALT, and Allele contain base-pair sequences. We add new features marking the lengths of the base-pairs appearing (as was done in [this kernel](https://www.kaggle.com/vasilyb/clinvar-identifying-conflicting-genetic-variants)), and we also modify entries in REF, ALT, and Allele with long base-pair sequences (length > 1) to an entry 'L'.
# 4. LoFtool, ORIGIN, CADD_PHRED, CADD_RAW are numeric with missing values that we impute; for ORIGIN we use the median, since its entries are all integers, while for the other columns we impute using the mean.
# 5. The data in POS seems to carry data about where, in a given chromosome, the mutation occurs. Since different chromosomes have different lengths, we normalize entries in the POS column to take values between 0 and 1, so that the scale of entries in POS will not bias the model.
# 6. Data in the IMPACT column has categorical descriptions that seem to imply an order (HIGH - MEDIUM - LOW), so we convert these to evenly-spaced floats in [0,1].

# In[ ]:


data = pd.read_csv("../input/clinvar_conflicting.csv")

print("Processing data...")

#drop NA values up to 50%
data = data.dropna(axis=1, thresh=.5*data.shape[0])

#convert entries in the Chromosome column to integers
data.CHROM = pd.Series(data.CHROM.map(lambda x: to_int(x)), index = data.CHROM.index)

#drop rows with few NA features in certain columns
data = data.drop(data[ (data['STRAND'].isna()) | (data['BIOTYPE'].isna()) | (data['SYMBOL'].isna())].index, axis = 0)

#'Feature Type' has a unique entry, so we drop this feature
data = data.drop(['Feature_type'], axis = 1)

#start looking at single / multiple base entries. We see most entries are single bases
single_base = ['T','A','G','C','-']

I = data[~data.REF.isin(single_base)].index
I=I.union(data[~data.ALT.isin(single_base)].index)
I=I.union(data[~data.Allele.isin(single_base)].index)

#We'll address this in two ways: we'll add a feature that counts the length of the bases involved in the REF, ALT, Allele category
#We'll also condense the longer base sequences into a category 'L' -- this allows the distinctions between G,C,T,A,- to still appear in thse features    
data['REF_LEN'] = pd.Series(data['REF'].apply(lambda x: len(x)), index = data['REF'].index)
data['ALT_LEN'] = pd.Series(data['ALT'].apply(lambda x: len(x)), index = data['ALT'].index)
data['Allele_LEN'] = pd.Series(data['Allele'].apply(lambda x: len(x)), index = data['Allele'].index)
    
data['REF'] = pd.Series(data['REF'].apply(string_to_short), data['REF'].index)
data['ALT'] = pd.Series(data['ALT'].apply(string_to_short), data['ALT'].index)
data['Allele'] = pd.Series(data['Allele'].apply(string_to_short), data['Allele'].index)

#Here we start with some imputation
imp_mean = SimpleImputer(strategy = 'mean')
imp_med = SimpleImputer(strategy = 'median')

data['ORIGIN'] = pd.Series(imp_med.fit_transform(data.ORIGIN.values.reshape((-1,1))).flatten(), index = data.ORIGIN.index)
data['LoFtool'] = pd.Series(imp_mean.fit_transform(data.LoFtool.values.reshape((-1,1))).flatten(), index = data.LoFtool.index)
data['CADD_PHRED'] = pd.Series(imp_mean.fit_transform(data.CADD_PHRED.values.reshape((-1,1))).flatten(), index = data.CADD_PHRED.index)
data['CADD_RAW'] = pd.Series(imp_mean.fit_transform(data.CADD_RAW.values.reshape((-1,1))).flatten(), index = data.CADD_RAW.index)

#Finally, we normalize numerical columns
cols_to_normalize = ['AF_ESP', 'AF_EXAC', 'AF_TGP', 'LoFtool', 'CADD_PHRED', 'CADD_RAW']
for c in cols_to_normalize:
    data[c] = pd.Series((lambda x: (x-x.min())/(x.max() - x.min()))(data[c]), index = data[c].index)

#We also normalize the position factor
#within each chromosome, we rescale the position to a numerical value between 0 and 1 representing its relative position in the chromosome
#We'll use the largest position for a given chromosome as a proxy for its actual maximal length; this needs to be replaced with actual maximal lengths
Maxes = [data[data['CHROM']==j].POS.max() for j in data.CHROM.unique()]
data.POS = data.apply(lambda x: x.POS / Maxes[x.CHROM-1], axis = 1)
    
#replace NaN entries in the Codons, Amino_Acids, and MC columns with a class 'unkn'.
data.Codons.fillna('unkn', inplace=True)
data.Amino_acids.fillna('unkn', inplace=True)
data.MC.fillna('unkn', inplace=True)

#convert the impact scores to numerical values
impact_dict = {"MODIFIER":0, "LOW": 0.33333, "MODERATE": 0.66667, "HIGH": 1}
data["IMPACT"] = data["IMPACT"].apply(lambda x:impact_dict[x])
    
#create the weights column
data['CLNDN_WTS'] = relative_weights(data, "CLASS", "CLNDN", "|", {"not_specified" : "not_provided"})
data['MC_WTS'] = relative_weights(data, "CLASS", "MC", ",")
data['Consequence_WTS'] = relative_weights(data, "CLASS", "Consequence", "&")
    
    
print("Finished!")
#Some summary info about data so far
if verbose:
    for c in data.columns:
        print("Column: ", c, "\n")
        print(data[c].unique())
        print("\n")
    sns.countplot(x="CLASS", data = data)


# We include a separate plot showing the result of the new feature engineered using the relative_weights function (the plots for MC_WTS and Consqeuence_WTS are essentially the same, so we plot them together).

# In[ ]:


pd.Series(sorted(data.CLNDN_WTS.values, reverse = True)).plot()


# In[ ]:


pd.Series(sorted(data.MC_WTS.values, reverse = True)).plot()
pd.Series(sorted(data.Consequence_WTS.values, reverse = True)).plot()


# The modeling occurs in the next code box. A few notes are in order:
# 1. Only some of the columns are used. There are others that likely contain useful information; this is just a first draft and future updates will take these other columns into account.
# 2. The data turns out to be fairly imbalanced (nearly 75% of entries are class 0); here the data is balanced by selecting a subset of class 0 entries containing as many elements as there are in the class 1 entries, so that the balanced dataset has 

# In[ ]:


#select features for the model
features_columns = ["CHROM", "POS", "REF", "ALT", "AF_ESP", "AF_EXAC", "AF_TGP", "CLNVC", "ORIGIN", "CLASS", "Allele", "IMPACT", "STRAND",
                   "LoFtool", "CADD_PHRED", "CADD_RAW", "REF_LEN", "ALT_LEN","Allele_LEN", "CLNDN_WTS", "MC_WTS", "Consequence_WTS"]

#here we extract a balanced dataset:
data_balanced = pd.concat([data[data['CLASS']==0].sample(n=data[data['CLASS']==1].shape[0], axis = 0), data[data['CLASS']==1]])

#encode categorical data and extract train / test data
data_bal_encoded = pd.get_dummies(data_balanced[features_columns])
features = data_bal_encoded[data_bal_encoded.columns.drop("CLASS")]
labels = data_bal_encoded.CLASS

#model_parameters
a=[150, 15]
clf = RandomForestClassifier(n_estimators = a[0], min_samples_split = a[1])

plt.figure()

AUCS = []
f1s =[]

for i in range(n_models):
    #split and fit the data
    features_train, features_test, labels_train, labels_test = train_test_split(features, labels, train_size = alpha, test_size = 1-alpha)
    roc_probs = clf.fit(features_train, labels_train).predict_proba(features_test)
    y_pred = clf.predict(features_test)
    
    #feedback from this model
    print("Model %i trained"%(i+1))
    if verbose: 
        print(classification_report(labels_test, y_pred))
        print("Model %i AUROC score: %0.3f"%((i+1), auc(fpr,tpr)))
    
    #capture data
    fpr, tpr, threshs = roc_curve(labels_test.values, roc_probs[:,1])
    plt.plot(fpr, tpr)
    AUCS.append(auc(fpr,tpr))
    
plt.plot([0,1],[0,1], linestyle = '--')
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.suptitle("AUROC Curves")
plt.title("Average AUROC: %0.3f; Standard Deviation: %0.3f"%(np.mean(AUCS), np.std(AUCS)))
plt.show()

