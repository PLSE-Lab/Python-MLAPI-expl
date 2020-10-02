#!/usr/bin/env python
# coding: utf-8

# Welcome to my kernel for the prediction of conflicts in classification of genetic variants! I'm relatively new to data science, so I welcome any constructive feedback you'd like to offer. This is an updated version of the kernel, after I did some debugging of some errors in use of PCA.
# # Feature Selection and Engineering
# 
# 
# First I'd like to examine the most promising columns of the data set to determine which features would most likely be relevant to prediction of conflicts, and convert them to workable formats.
# 
# The reference point I'll be using to ascertain the "relevant" features is to check the base rate of conflict: the prior probability of any variant having a conflicting classification is 25.21%. Comparing this rate to the rate of conflict among samples with a given value for a binary feature - which, as we'll see, are the predominant useable features in this dataset - should help illuminate which features will have predictive power. This isn't perfect, since of course some patterns may arise only among subsets of samples filtered by *multiple* features,  but if we retain enough features for our training set, such interactions will likely be revealed in the machine learning models themselves.
# 
# One very important lesson I learned far too late - if you're going to use cross-validation, split your data into the k folds *before* you select your features! This has been detailed before in sources such as [this one](https://github.com/mottalrd/cross-validation-done-wrong/blob/master/Cross%20Validation%20done%20wrong.ipynb), but it's actually an embarrassingly simple concept. If you choose your features based on their correlations with the classification variable in the entire dataset, or the distribution of that classification variable among samples of a given feature value in the entire dataset, then you're including information that will bias your model when you evaluate its performance on different folds of the training set! This consideration led me to exclude certain chromosome dummy variables whose conflict rate estimates became less significant when I corrected this potential source of bias.

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore') # A warning about F1 score appears every time recall for one
                                  # class is 0; it's unnecessary to keep seeing this.

df = pd.read_csv('../input/clinvar_conflicting.csv', dtype={0: object, 38: str, 40: object})
print(df['CLASS'].describe())
df.head()


# (Credit to Kevin Arvai's EDA Kernel.) We convert the "x/y" format of EXON to an absolute value. In addition, I'm recoding this into a binary variable for exon status, since leaving exon number as a numerical variable would not be terribly enlightening. Since there are so many genes represented in this dataset, I restrict the dummy variables to those with a frequency of at least 1000 (here again, Kevin's kernel was helpful in providing a crosstab plot of conflict rate and frequency by gene). I also create dummy variables for IMPACT.

# In[ ]:


df.EXON.fillna('0', inplace=True)
df['variant_exon'] = df.EXON.apply(lambda x: 0 if x == '0' else 1)

dummies = pd.get_dummies(df.SYMBOL)
df = pd.concat([df, dummies], axis=1)

dummies = pd.get_dummies(df.IMPACT)
df = pd.concat([df, dummies], axis=1)

dummies = pd.get_dummies(df.CHROM)
dummies.columns = ['CHROM_' + str(col) for col in dummies.columns]
df = pd.concat([df, dummies], axis=1)

df.head()


# The file lists the consequence of the variant in two forms, but 'Consequence' has more information. 

# In[ ]:


print(df['Consequence'].unique())
print(df['CLNVC'].unique())


# Examining the values of this column, we see that the consequences are separated by '&'.  We want to create dummy variables for each consequence, using similar methods as in Kevin Arvai's kernel. Let's do the same for CLNVC.

# In[ ]:


Conseq_list = df.Consequence.dropna().str.split('&')
Cons_dummies = pd.get_dummies(Conseq_list.apply(pd.Series).stack()).sum(level=0)
Cons_dummies = Cons_dummies.reindex(index=Conseq_list.index)
df = df.join(Cons_dummies).drop(columns=['Consequence'])

var_list = df.CLNVC.dropna()
var_dummies = pd.get_dummies(var_list.apply(pd.Series).stack()).sum(level=0)
var_dummies = var_dummies.reindex(index=var_list.index)
df = df.join(var_dummies).drop(columns=['CLNVC'])

df.head()


# Strangely enough, intron status is included in Consequence as well. Let's make a dummy variable for intron status to compare with this later. Also, while "ORIGIN" is numeric, the numbers are codes, so we'll need dummy variables for it too. I encode missing values as 0 because 0 originally refers to variants of unknown origin, so these should be comparable. Finally, we parse the Amino Acid column.

# In[ ]:


df.INTRON.fillna('0', inplace=True)
df['variant_intron'] = df.INTRON.apply(lambda x: 0 if x == '0' else 1)

df.ORIGIN = df.ORIGIN.fillna(0)
dummies = pd.get_dummies(df.ORIGIN)
dummies.columns = ['ORIGIN_' + str(col) for col in dummies.columns]
df = pd.concat([df, dummies], axis=1)

dummies = pd.get_dummies(df.Amino_acids)
dummies.columns = [str(col) for col in dummies.columns]
df = pd.concat([df, dummies], axis=1)


# In[ ]:


df.head()


# Below, I first separate out the test set from training, then construct a list of 10 training and validation sets. The use of "StratifiedKFold" ensures the rate of conflict in each fold is approximately equal to the base rate, as confirmed below. 

# In[ ]:


from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

X = df.drop('CLASS', axis=1)
y = df['CLASS']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1, stratify=y)

folds = StratifiedKFold(n_splits = 10, random_state = 1337)

X_train_sets = []
X_val_sets = []
y_train_sets = []
y_val_sets = []
for train_index, val_index in folds.split(X_train,y_train):
    X_train_sets.append(X_train.iloc[train_index,:])
    y_train_sets.append(y_train.iloc[train_index])
    X_val_sets.append(X_train.iloc[val_index,:])
    y_val_sets.append(y_train.iloc[val_index])


# In[ ]:


print([np.mean(fold) for fold in y_train_sets])
print([np.mean(fold) for fold in y_val_sets])


# To determine the rate of conflicting classifications for variants where a column takes on a certain value, I constructed the following helper function. I apply this function to genes with a frequency of at least 652 (1% of the entire dataset), exon status, and the IMPACT variable. For the purposes of this analysis, if the aveage conflict rate across all 10 folds differs from the base rate by 3 percentage points or more, I'll consider it a feature worth including in the model testing process.

# In[ ]:


def conflict_rate(col_name, value):
    results = []
    for fold_index in range(10):
        data = X_train_sets[fold_index]
        subset = data[col_name] == value
        results.append(np.mean(y_train_sets[fold_index][subset]))
    return np.mean(results), np.std(results)
        
df.head()


# In[ ]:


for i in list(df['SYMBOL'].value_counts().index.values):
    if df['SYMBOL'].value_counts()[i] < 652:
        break
    print(i, str(conflict_rate('SYMBOL',i)))


# All of the high-frequency genes actually have skewed conflict rates.
# 
# Since there are so many genes represented among these variants, it seems a shame to waste the data on the rarer genes. I construct a new variable that tells whether the given variant is in *any* of the genes with a conflict rate 3% above or below the base rate.

# In[ ]:


high_gene = []
low_gene = []

for i in df['SYMBOL'].unique():
    if conflict_rate('SYMBOL',i)[0] > 0.2821:
        high_gene.append(i)
    if conflict_rate('SYMBOL',i)[0] < 0.2221:
        low_gene.append(i)


# In[ ]:


for j in range(10):
    X_train_sets[j]['HIGH_SYMBOL'] = X_train_sets[j]['SYMBOL'].isin(high_gene).apply(int)
    X_train_sets[j]['LOW_SYMBOL'] = X_train_sets[j]['SYMBOL'].isin(low_gene).apply(int)
    X_val_sets[j]['HIGH_SYMBOL'] = X_val_sets[j]['SYMBOL'].isin(high_gene).apply(int)
    X_val_sets[j]['LOW_SYMBOL'] = X_val_sets[j]['SYMBOL'].isin(low_gene).apply(int)


# In[ ]:


print(conflict_rate('HIGH_SYMBOL',1))
print(conflict_rate('LOW_SYMBOL',1))


# In[ ]:


for j in range(10):
    print(X_train_sets[j]['HIGH_SYMBOL'].value_counts())
    print(X_train_sets[j]['LOW_SYMBOL'].value_counts())


# In[ ]:


print(conflict_rate('IMPACT','HIGH')) #very low
print(conflict_rate('IMPACT','MODERATE')) #close to average
print(conflict_rate('IMPACT','MODIFIER')) #close to average
print(conflict_rate('IMPACT','LOW')) #close to average
print(conflict_rate('variant_exon',0)) #close to average
print(conflict_rate('variant_exon',1)) #close to average


# Unsurprisingly, we see that HIGH-IMPACT variants have a much lower rate of conflicting classification than the base. However, we also see that when the conflict rates of exons and introns are compared, they don't differ substantially. A natural next question is whether the number of HIGH-IMPACT variants is large enough that this different misclassification rate might make a significant difference. We see below that the number of HIGH-IMPACT variants is relatively low, but hardly negligible - 4752 out of 65188.

# In[ ]:


print(df['IMPACT'].value_counts())


# Next let's check chromosomes.

# In[ ]:


for i in df['CHROM'].unique():
    print(i + ":")
    print(conflict_rate('CHROM',i))
    print(sum(df['CHROM'] == i))
    print("\n")


# For chromosome 2, this isn't an exceptional deviation, which shouldn't be surprising since the frequency of Chrom 2 among the variants is so high anyway - thus if Chrom 2 had a drastically large rate of conflicting classifications, the base rate of conflict probably wouldn't be very different unless every other chromosome had exceptionally low conflict rates.
# 
# However, some of the rarer chromosome variants, we see, have significantly different conflict rates! Chromosomes 5, 7, 11, 13, 14, 15, 20, 21, 22, and MT (mitochondrial) stand out. MT is too rare to be worth including, but the rest may be useful predictors.
# 
# Looking at the 'MC' crosstab from Kevin Arvai's kernel, 'intron_variant' seems to be promising. Let's check conflict rates for variants with and without each Consequence.

# In[ ]:


for i in list(df.columns.unique()[2401:2424]):
    print(i + ":")
    print("\t" + str(conflict_rate(i,1)))
    print("\t" + str(conflict_rate(i,0)))
    print("\t" + str(np.sum(df[i] == 1)))
    print("\n")


# Here we see a very low conflict rate for "frameshift_variant" and "stop_gained", which are somewhat rare, but still have frequencies above 1000. More common are splice_region_variants, which have an abnormally high conflict rate.
# 
# While SNVs dominate the CLNVC category, it might be worth checking if the rarer classes have skewed proportions of conflicts.

# In[ ]:


print(conflict_rate('single_nucleotide_variant',1)) #close to average
print(conflict_rate('Deletion',1)) #relatively low
print(np.sum(df['Deletion'] == 1))
print(conflict_rate('Duplication',1)) #slightly low
print(np.sum(df['Duplication'] == 1))


# Based on these lower conflict rates and frequencies above 1000, Deletion and Duplication are worth including as features.
# 
# We see that intron variants, according to the Consequence column, have a slightly higher conflict rate (although this isn't so significant that it meets the threshold I've set). Why is it that the conflict rate we see for introns by this metric is so different from what we found in the EXON column? Let's examine our variant_intron dummy variable.

# In[ ]:


print(df[df['variant_exon'] == 0].shape)
print(df[df['variant_intron'] == 1].shape)
print(df[np.logical_and(df['variant_intron'] == 0, df['variant_exon'] == 0)].shape)
print(df[np.logical_and(df['intron_variant'] == 0, df['variant_exon'] == 0)].shape)
print(df[df['intron_variant'] == 0].shape)
print(df[np.logical_and(df['variant_intron'] == 1, df['variant_exon'] == 0)].shape)
print(df.shape)
print(df[df['intron_variant'] == 1].shape)


# More variants have non-missing data in the INTRON column than have "1" for a value in the intron_variant column. To get to the bottom of this, we have to examine the intersection of cases with "0" for variant_exon and "1" for variant_intron. This number is smaller than the number fulfilling either of these conditions separately, which is unexpected unless some variants are neither introns nor exons. Evidently at most 133 variants cannot be classified as either, based on the recoded EXON and INTRON columns, although some may simply be missing cases. Unfortunately, there's no CLNVC category for exons, so we can't do a direct comparison.
# 
# There are a lot more variants with missing EXON values *and* a "0" for intron_variant (as opposed to variant_intron, which was made from the INTRON column), so I'm inclined to think that every variant with non-missing INTRON data must actual be an intron, and it's more likely that the cases labeled 0 in intron_variant but 1 in INTRON are false negatives. So the conflict rate for variant_intron will be more useful:

# In[ ]:


print(conflict_rate('variant_intron',1))
print(conflict_rate('variant_intron',0))


# These tell basically the same story as the dummy variable for EXON status; it seems that the intron/exon distinction will not be a useful predictor of conflicts.
# 
# For SNVs, REF and ALT might be worth checking out, but alas we see that the conflict rates don't differ meaningfully between the nucleotides.

# In[ ]:


print(conflict_rate('REF','A')) #about average
print(conflict_rate('REF','T')) #about average
print(conflict_rate('REF','G')) #about average     
print(conflict_rate('REF','C')) #about average
print(conflict_rate('ALT','A')) #about average
print(conflict_rate('ALT','T')) #about average
print(conflict_rate('ALT','G')) #about average     
print(conflict_rate('ALT','C')) #about average


# What about SIFT and PolyPhen?

# In[ ]:


print(conflict_rate('SIFT','tolerated_low_confidence')) #about average
print(conflict_rate('SIFT','deleterious_low_confidence')) #about average
print(conflict_rate('SIFT','tolerated')) #about average
print(conflict_rate('SIFT','deleterious')) #about average

print(conflict_rate('PolyPhen','benign')) #about average
print(conflict_rate('PolyPhen','probably_damaging')) #about average
print(conflict_rate('PolyPhen','possibly_damaging')) #about average


# Nothing useful from either of these, alas. It's surprising that the low-confidence cases don't have a larger conflict rate; if anything, it's the opposite! But not enough that these features would be helpful in a model.
# 
# Kevin's kernel showed visually that allele frequencies were negatively correlated with classification conflict, but let's quantify this. First, I examine the distribution of the non-transformed allele frequency column in one of the test sets (as an example) versus log-transformed and min-max-scaled allele frequency.

# In[ ]:


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

logfreq = np.log(np.clip(X_train_sets[0]['AF_ESP'],1e-16, None))
scaled_logfreq = scaler.fit_transform(logfreq.values.reshape(-1,1))

plt.hist(X_train_sets[0]['AF_ESP'], bins = 100)
plt.show()

plt.hist(scaled_logfreq, bins = 100)
plt.show()


# The log-transformed distribution is still very skewed, but plausibly this transformed feature may be more amenable to learning.

# In[ ]:


for j in range(10):
    X_train_sets[j]['AF_ESP'] = np.log(np.clip(X_train_sets[j]['AF_ESP'],1e-16, None))
    X_train_sets[j]['AF_ESP'] = scaler.fit_transform(X_train_sets[j]['AF_ESP'].values.reshape(-1,1))
    X_train_sets[j]['AF_EXAC'] = np.log(np.clip(X_train_sets[j]['AF_EXAC'],1e-16, None))
    X_train_sets[j]['AF_EXAC'] = scaler.fit_transform(X_train_sets[j]['AF_EXAC'].values.reshape(-1,1))
    X_train_sets[j]['AF_TGP'] = np.log(np.clip(X_train_sets[j]['AF_TGP'],1e-16, None))
    X_train_sets[j]['AF_TGP'] = scaler.fit_transform(X_train_sets[j]['AF_TGP'].values.reshape(-1,1))
    X_val_sets[j]['AF_ESP'] = np.log(np.clip(X_val_sets[j]['AF_ESP'],1e-16, None))
    X_val_sets[j]['AF_ESP'] = scaler.fit_transform(X_val_sets[j]['AF_ESP'].values.reshape(-1,1))
    X_val_sets[j]['AF_EXAC'] = np.log(np.clip(X_val_sets[j]['AF_EXAC'],1e-16, None))
    X_val_sets[j]['AF_EXAC'] = scaler.fit_transform(X_val_sets[j]['AF_EXAC'].values.reshape(-1,1))
    X_val_sets[j]['AF_TGP'] = np.log(np.clip(X_val_sets[j]['AF_TGP'],1e-16, None))
    X_val_sets[j]['AF_TGP'] = scaler.fit_transform(X_val_sets[j]['AF_TGP'].values.reshape(-1,1))
    X_test['AF_TGP'] = np.log(np.clip(X_test['AF_TGP'],1e-16, None))
    X_test['AF_TGP'] = scaler.fit_transform(X_test['AF_TGP'].values.reshape(-1,1))
    data_y = y_train_sets[j]
    print(np.corrcoef(data_y, X_train_sets[j]['AF_ESP']))
    print(np.corrcoef(data_y, X_train_sets[j]['AF_EXAC']))
    print(np.corrcoef(data_y, X_train_sets[j]['AF_TGP']))
    print("\n")


# Now, what to make of the "Indel"/"SNV" dichotomy? In Kevin's kernel we saw that the length of REF or ALT could be used to classify variants as indels or SNVs. But we also see these two categories as dummy variables from CLNVC, and it turns out that not every variant is classified as one or the other in this set of columns. Rather, "Indel" is evidently used to mark variants that are either insertions or deletions, but *not* definitely known to be one or the other. Unfortunately, while we do indeed see that Indels are less likely to have conflicts than SNVs, they're exceptionally rare; almost all of the variants in the dataset are SNVs.

# In[ ]:


print(np.sum(df['Indel']))
print(np.sum(df['Insertion']))
print(np.sum(df['Deletion']))
print(np.sum(df['Indel']) + np.sum(df['single_nucleotide_variant']) +
     np.sum(df['Insertion']) + np.sum(df['Deletion']))
print(df.shape)
print(df['Indel'].isnull().sum() + df['single_nucleotide_variant'].isnull().sum())
print(df[np.logical_and(df['Indel'] == 1, df['Deletion'] == 1)].shape)
print(df[np.logical_and(df['Indel'] == 1, df['Insertion'] == 1)].shape)

print(conflict_rate('Indel',1))
print(conflict_rate('single_nucleotide_variant',1))
print(conflict_rate('Insertion',1))


# Next let's consider 'ORIGIN'. While this column is numeric, the numbers are codes, so we need dummy variables here.

# In[ ]:


for i in range(len(list(df['ORIGIN'].unique()))):
    value = list(df['ORIGIN'].unique())[i]
    print(str(value) + ":")
    print(conflict_rate('ORIGIN',value))
    print(sum(df['ORIGIN'] == value))
    print("\n")

df.head()


# Given that the encoding of missing values as equivalent to unknown origin is reasonable, "0" is promising. Let's check Feature_type and BIOTYPE:

# In[ ]:


print(list(df['Feature_type'].unique()))
print(conflict_rate('Feature_type','Transcript'))
print(np.sum(df['Feature_type'] == 'Transcript'))
print(conflict_rate('Feature_type','MotifFeature'))

print(list(df['BIOTYPE'].unique()))
print(conflict_rate('BIOTYPE','protein_coding'))
print(np.sum(df['BIOTYPE'] == 'protein_coding'))
print(conflict_rate('BIOTYPE','misc_RNA'))
print(np.sum(df['BIOTYPE'] == 'misc_RNA'))


# No such luck.
# 
# Next we check amino acids. The only notable case (not exceptionally rare, yet the conflict rate is much different) is R/H. No notable patterns for STRAND.

# In[ ]:


for i in list(df['Amino_acids'].value_counts().index.values):
    if df['Amino_acids'].value_counts()[i] < 652:
        break
    print(i, str(conflict_rate('Amino_acids',i)))
    print(df['Amino_acids'].value_counts()[i])

print(conflict_rate('STRAND',1))
print(conflict_rate('STRAND',-1))


# In[ ]:


high_feature = []
low_feature = []

for i in df['Feature'].unique():
    if conflict_rate('Feature',i)[0] > 0.2821:
        high_feature.append(i)
    if conflict_rate('Feature',i)[0] < 0.2221:
        low_feature.append(i)


# In[ ]:


for j in range(10):
    X_train_sets[j]['HIGH_FEATURE'] = X_train_sets[j]['Feature'].isin(high_feature).apply(int)
    X_train_sets[j]['LOW_FEATURE'] = X_train_sets[j]['Feature'].isin(low_feature).apply(int)
    X_val_sets[j]['HIGH_FEATURE'] = X_val_sets[j]['Feature'].isin(high_feature).apply(int)
    X_val_sets[j]['LOW_FEATURE'] = X_val_sets[j]['Feature'].isin(low_feature).apply(int)


# In[ ]:


print(conflict_rate('HIGH_FEATURE',1))
print(conflict_rate('LOW_FEATURE',1))


# In[ ]:


for j in range(10):
    print(X_train_sets[j]['HIGH_FEATURE'].value_counts())
    print(X_train_sets[j]['LOW_FEATURE'].value_counts())


# **Summary of insights from EDA:**
# 
# 1) Unbalanced problem: prior probability of any variant having a conflicting classification is 25.21%.
# 
# 2) Introns and exons don't have appreciably different rates of conflicts.
# 
# 3) Chromosome-to-chromosome rates of conflict may differ from the base rate by as much as 6%.
# 
# 4) Gene-to-gene rates of conflict among some of the most common rates can differ from base rate by as much as 14% (!).
# 
# 5) Some relatively rare yet non-negligible Consequences, clinical variant classifications, and amino acids have significantly
# deviant conflict rates.
# 
# 6) High-impact variants have a much lower conflict rate than the prior.
# 
# 7) Allele frequency is mildly negatively correlated with conflict in classification.

# In[ ]:


for i in range(10):
    X_train_sets[i] = X_train_sets[i][['TTN', 'BRCA2', 'BRCA1',
            'ATM','APC','MSH6','LDLR','PALB2','NF1','HIGH',
           'CHROM_5', 'CHROM_7', 'CHROM_11', 'CHROM_13',
           'CHROM_14', 'CHROM_15', 'CHROM_20',
            'CHROM_21','CHROM_22', 'ORIGIN_0.0',
           'stop_gained', 'splice_region_variant',
           'frameshift_variant', 'Deletion', 'Duplication',
           'AF_TGP','AF_EXAC','AF_ESP','R/Q','R/H','G',
            'R/C','A/T','E/K','R/W','V/I','I/V','R/*',
            'HIGH_SYMBOL','LOW_SYMBOL','HIGH_FEATURE','LOW_FEATURE']]
    X_train_sets[i] = np.array(X_train_sets[i])


# In[ ]:


for i in range(10):
    y_train_sets[i] = np.array(y_train_sets[i])


# In[ ]:


for i in range(10):
    X_val_sets[i] = X_val_sets[i][['TTN', 'BRCA2', 'BRCA1',
            'ATM','APC','MSH6','LDLR','PALB2','NF1','HIGH',
           'CHROM_5', 'CHROM_7', 'CHROM_11', 'CHROM_13',
           'CHROM_14', 'CHROM_15', 'CHROM_20',
            'CHROM_21','CHROM_22', 'ORIGIN_0.0',
           'stop_gained', 'splice_region_variant',
           'frameshift_variant', 'Deletion', 'Duplication',
           'AF_TGP','AF_EXAC','AF_ESP','R/Q','R/H','G',
            'R/C','A/T','E/K','R/W','V/I','I/V','R/*',
            'HIGH_SYMBOL','LOW_SYMBOL','HIGH_FEATURE','LOW_FEATURE']]
    X_val_sets[i] = np.array(X_val_sets[i])


# In[ ]:


for i in range(10):
    y_val_sets[i] = np.array(y_val_sets[i])


# In[ ]:


X_test['HIGH_SYMBOL'] = X_test['SYMBOL'].isin(high_gene).apply(int)
X_test['LOW_SYMBOL'] = X_test['SYMBOL'].isin(low_gene).apply(int)
X_test['HIGH_FEATURE'] = X_test['Feature'].isin(high_feature).apply(int)
X_test['LOW_FEATURE'] = X_test['Feature'].isin(low_feature).apply(int)

X_test = X_test[['TTN', 'BRCA2', 'BRCA1',
            'ATM','APC','MSH6','LDLR','PALB2','NF1','HIGH',
           'CHROM_5', 'CHROM_7', 'CHROM_11', 'CHROM_13',
           'CHROM_14', 'CHROM_15', 'CHROM_20',
            'CHROM_21','CHROM_22', 'ORIGIN_0.0',
           'stop_gained', 'splice_region_variant',
           'frameshift_variant', 'Deletion', 'Duplication',
           'AF_TGP','AF_EXAC','AF_ESP','R/Q','R/H','G',
            'R/C','A/T','E/K','R/W','V/I','I/V','R/*',
            'HIGH_SYMBOL','LOW_SYMBOL','HIGH_FEATURE','LOW_FEATURE']]
X_test = np.array(X_test)
y_test = np.array(y_test)


# Our feature matrix has all dummy variables except for allele frequency. First we verify that no imputation of missing values is required.

# In[ ]:


for i in range(10):
    print(sum(np.isnan(X_train_sets[i])))
    print(sum(np.isnan(y_train_sets[i])))
    print(sum(np.isnan(X_val_sets[i])))
    print(sum(np.isnan(y_val_sets[i])))


# All good. Fortunately, all the features are between 0 and 1 already, so feature scaling isn't necessary. Below I reformat the target value arrays such that the models will work on them.

# In[ ]:


for i in range(10):
    y_train_sets[i] = np.ravel(y_train_sets[i])
    y_val_sets[i] = np.ravel(y_val_sets[i])

y_test = np.ravel(y_test)


# Also, let's investigate the results of PCA on this set of features. As I'm new to PCA, my intuition is that using principal components for our models may come at the cost of interpretability, but we can still check which features are most strongly correlated with the PCs retained for the analysis.

# In[ ]:


from sklearn.decomposition import PCA
for i in range(10):
    pca = PCA(n_components = None)
    pca.fit(X_train_sets[i])
    print(pca.explained_variance_ratio_)
    print(np.cumsum(pca.explained_variance_ratio_))
    print("\n")


# The first two PCs stand out, but they still only explains about 53% of variance. I will examine how performance varies with the number of PCs included in the analysis, separately for each model.
# 
# Now I want to assess the performance of a variety of models, based on F1 score, which is evidently ideal for a binary imbalanced problem like this. I'll also examine the area under the ROC curve (AUC) for each model, although this evidently has limitations for imbalanced problems.
# 
# But what is the F1 score to beat in this case? At the very least, our model should outperform a prediction of "no conflict" for every variant.

# In[ ]:


from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_auc_score

"""
The following helper function returns the average F1 score and area under the ROC curve for
the given classifier's performance across the 10 folds. The "search" and "use_pca" arguments
will come into play when I try Randomized Search for hyperparameter tuning, as well as PCA.
"""

def evaluate(classifier, search=False, use_pca=False):
    scores = []
    roc = []
    param_sets = []
    for i in range(10):
        X = X_train_sets[i]
        y = y_train_sets[i]
        X_val = X_val_sets[i]
        y_val = y_val_sets[i]
        if use_pca == True:
            X = pca.fit_transform(X)
            X_val = pca.fit_transform(X_val)
        classifier.fit(X, y)
        val = classifier.predict(X_val)
        scores.append(f1_score(y_val, val, average='weighted'))
        roc.append(roc_auc_score(y_val, val, average='weighted'))
        if search == True:
            param_sets.append(classifier.best_params_)
    return(np.mean(scores), np.mean(roc), param_sets)

base = DummyClassifier(strategy = 'most_frequent')
evaluate(base)


# This means that if a model can achieve an F1 score of no better than about 0.64, it's no improvement on a naive prediction rule based on the base rate of conflicts.
# 
# # Logistic Regression
# 
# The logistic regression classifier's performance may vary based on the value of C (inverse of the regularization hyperparameter).

# In[ ]:


from sklearn.linear_model import LogisticRegression

mean_scores = []
mean_roc = []

C_vals = [10**k for k in range(-3,3)]

for j in C_vals:
    lr = LogisticRegression(C=j, random_state=1337,
                            solver = 'lbfgs')
    mean_scores.append(evaluate(lr)[0])
    mean_roc.append(evaluate(lr)[1])
    
print(mean_scores)
print(mean_roc)

# [0.6400306147844548, 0.6400306147844548, 0.6400665625355366, 0.6400571456896468, 0.6399836690743743, 0.6399930586282129]
# [0.5, 0.5, 0.500025205587336, 0.5000123883609838, 0.49993591386823893, 0.4999487310945911]


# In[ ]:


nums_pca = [3, 5, 10, 20]


# In[ ]:


mean_scores = []
mean_roc = []

for n in nums_pca:
    pca = PCA(n_components = n)
    lr = LogisticRegression(C=0.001, random_state=1337,
                           solver = 'lbfgs')
    mean_scores.append(evaluate(lr,use_pca=True)[0])
    mean_roc.append(evaluate(lr,use_pca=True)[1])
    
print(mean_scores)
print(mean_roc)


# The logistic regression algorithm seems to basically emulate the dummy classifier. PCA doesn't help.
# 
# # Decision Trees
# 
# How will adjusting the hyperparameters of a decision tree affect performance? Here I try both Bayesian optimization and randomized search to tune the hyperparameters. Bayesian optimization searches for the best hyperparameters by using information from previously tested hyperparameters, rather than simply checking a random combination of hyperparameters from a given search space. We need to define an objective function to be minimized for this search method, so the natural choice is [1 - average F1 score across the 10 folds].

# In[ ]:


from hyperopt import hp, tpe, Trials, fmin, STATUS_OK
from hyperopt.pyll.stochastic import sample
from sklearn.tree import DecisionTreeClassifier

def objective_tree(hyperparameters):
    for parameter_name in ['min_samples_split','min_samples_leaf']:
        hyperparameters[parameter_name] = int(hyperparameters[parameter_name])
        
    model = DecisionTreeClassifier(**hyperparameters, random_state = 1337)
    
    score = evaluate(model)[0]
    loss = 1 - score
    
    return {'loss': loss, 'hyperparameters': hyperparameters,
            'status': STATUS_OK, 'score': score}
        
space_tree = {
    'criterion': hp.choice('criterion', ['gini', 'entropy']),
    'splitter': hp.choice('splitter', ['best', 'random']),
    'max_depth': hp.quniform('max_depth', 1, 38, 1),
    'min_samples_split': hp.quniform('min_samples_split', 2,60,2),
    'min_samples_leaf': hp.quniform('min_samples_leaf', 1,30,2),
    'max_features': hp.choice('max_features', ['sqrt', 'log2', None]),
}


# I apply the optimization algorithm to the decision tree, given the search space "space_tree," and print the best hyperparameters. Note that for hyperparameters that are strings, "fmin" returns the indices of those strings in the lists I defined with "hp.choice."

# In[ ]:


trials = Trials()

best = fmin(objective_tree, space_tree, algo=tpe.suggest, max_evals=100,
            trials = trials, rstate = np.random.RandomState(1337))

print(best)


# In[ ]:


tree = DecisionTreeClassifier(criterion = 'gini',
                               max_depth = 10,
                               max_features = None,
                               min_samples_leaf = 8,
                               min_samples_split = 14,
                               splitter = 'best',
                               random_state = 1337)

evaluate(tree)

# (0.7451884300842948, 0.6255220843318451, [])


# In[ ]:


scores = [trials.results[i]['score'] for i in range(100)]

plt.scatter(range(1,101), scores)
plt.xlabel('Iteration')
plt.ylabel('F1 Score')
plt.show()


# This is miles ahead of the dummy classifier and logistic regression! I should note that these results are in stark contrast to those I observed before I included the HIGH_SYMBOL, LOW_SYMBOL, HIGH_FEATURE, and LOW_FEATURE features. This goes to show how far the right feature engineering can go when the fanciest algorithms seem to be failing. While this model fares much better than the baseline, as we can see from the scatterplot, the optimization process doesn't seem to be improving the F1 scores significantly.

# In[ ]:


from sklearn.model_selection import RandomizedSearchCV


# In[ ]:


max_features = [None, 'sqrt','log2']
min_samples_split = [2, 5, 8, 10, 20, 40, 60]
min_samples_leaf = [1, 2, 4, 10, 20, 30]
max_depth = range(5,39)
criterion = ['gini','entropy']
splitter = ['best','random']

random_grid = {'max_depth': max_depth,
               'max_features': max_features,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'criterion': criterion,
               'splitter': splitter}

tree = DecisionTreeClassifier(random_state = 1337)

tree_random = RandomizedSearchCV(estimator = tree,
            param_distributions = random_grid,
            n_iter = 20, cv = 5, verbose=2,
            random_state=1337, n_jobs = -1)


# In[ ]:


evaluate(tree_random, search=True)

"""
(0.7354725283328227,
 0.6081079683838219,
 [{'splitter': 'best',
   'min_samples_split': 10,
   'min_samples_leaf': 10,
   'max_features': 'sqrt',
   'max_depth': 30,
   'criterion': 'entropy'},
  {'splitter': 'best',
   'min_samples_split': 5,
   'min_samples_leaf': 10,
   'max_features': 'sqrt',
   'max_depth': 35,
   'criterion': 'gini'},
  {'splitter': 'best',
   'min_samples_split': 10,
   'min_samples_leaf': 10,
   'max_features': 'sqrt',
   'max_depth': 30,
   'criterion': 'entropy'},
  {'splitter': 'best',
   'min_samples_split': 10,
   'min_samples_leaf': 10,
   'max_features': 'sqrt',
   'max_depth': 30,
   'criterion': 'entropy'},
  {'splitter': 'best',
   'min_samples_split': 8,
   'min_samples_leaf': 30,
   'max_features': 'sqrt',
   'max_depth': 29,
   'criterion': 'entropy'},
  {'splitter': 'best',
   'min_samples_split': 40,
   'min_samples_leaf': 4,
   'max_features': 'sqrt',
   'max_depth': 37,
   'criterion': 'entropy'},
  {'splitter': 'best',
   'min_samples_split': 8,
   'min_samples_leaf': 30,
   'max_features': 'sqrt',
   'max_depth': 29,
   'criterion': 'entropy'},
  {'splitter': 'best',
   'min_samples_split': 8,
   'min_samples_leaf': 30,
   'max_features': 'sqrt',
   'max_depth': 29,
   'criterion': 'entropy'},
  {'splitter': 'best',
   'min_samples_split': 5,
   'min_samples_leaf': 10,
   'max_features': 'sqrt',
   'max_depth': 35,
   'criterion': 'gini'},
  {'splitter': 'best',
   'min_samples_split': 8,
   'min_samples_leaf': 30,
   'max_features': 'sqrt',
   'max_depth': 29,
   'criterion': 'entropy'}])
"""


# The randomized search tree is worse than the Bayesian-optimized one, which at least is a sign that Bayesian optimization is doing its job. Let's try the "optimized" parameters from the Bayesian method on our PCA-transformed folds.

# In[ ]:


mean_scores = []
mean_roc = []

tree = DecisionTreeClassifier(criterion = 'gini',
                               max_depth = 10,
                               max_features = None,
                               min_samples_leaf = 8,
                               min_samples_split = 14,
                               splitter = 'best',
                               random_state = 1337)

for n in nums_pca:
    pca = PCA(n_components = n)
    mean_scores.append(evaluate(tree, use_pca = True)[0])
    mean_roc.append(evaluate(tree, use_pca = True)[1])
    
print(mean_scores)
print(mean_roc)

# [0.7079295409357134, 0.7060316153950137, 0.687911303148617, 0.6929085669646577]
# [0.5945664151449178, 0.5962103241099056, 0.5795035473571721, 0.5796663760671968]


# Unfortunately, PCA does not improve performance.
# 
# Let's try a Bayesian search with PCs - it's possible that these hyperparameters aren't optimal because they were derived from an analysis *without* PCA.

# In[ ]:


pca = PCA(n_components = 3)

def objective_tree_pca(hyperparameters):
    for parameter_name in ['min_samples_split','min_samples_leaf']:
        hyperparameters[parameter_name] = int(hyperparameters[parameter_name])
        
    model = DecisionTreeClassifier(**hyperparameters, random_state = 1337)
    
    score = evaluate(model, use_pca = True)[0]
    loss = 1 - score
    
    return {'loss': loss, 'hyperparameters': hyperparameters,
            'status': STATUS_OK, 'score': score}
        
space_tree_pca = {
    'criterion': hp.choice('criterion', ['gini', 'entropy']),
    'splitter': hp.choice('splitter', ['best', 'random']),
    'max_depth': hp.quniform('max_depth', 1, 38, 1),
    'min_samples_split': hp.quniform('min_samples_split', 2,60,2),
    'min_samples_leaf': hp.quniform('min_samples_leaf', 1,30,2),
    'max_features': hp.choice('max_features', ['sqrt', 'log2', None]),
}


# In[ ]:


trials = Trials()

best = fmin(objective_tree_pca, space_tree_pca, algo=tpe.suggest, max_evals=20,
            trials = trials, rstate = np.random.RandomState(1337))

print(best)


# In[ ]:


tree = DecisionTreeClassifier(criterion = 'gini',
                               max_depth = 14,
                               max_features = None,
                               min_samples_leaf = 28,
                               min_samples_split = 28,
                               splitter = 'best',
                               random_state = 1337)

evaluate(tree, use_pca = True)

# (0.7158981294197934, 0.6043733286889215, [])


# Better, although still not as good as the best decision tree without PCA.

# # K-Nearest Neighbors
# 
# KNN classifiers are particularly subject to the curse of dimensionality, so I'll try PCA here.

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


pca = PCA(n_components = 3)

mean_scores = []
mean_roc = []
kays = range(3,8)

for k in kays:
    knn = KNeighborsClassifier(n_neighbors=k, 
                           p=2, metric='minkowski')
    mean_scores.append(evaluate(knn, use_pca = True)[0])
    mean_roc.append(evaluate(knn, use_pca = True)[1])


# In[ ]:


print(mean_scores)
print(mean_roc)

# [0.6985008656214247, 0.6921237746265237, 0.6836164859591516, 0.6917872861068209, 0.6923423518570587]
# [0.5895092398891508, 0.5582714075330364, 0.5753057104798376, 0.5593956211510343, 0.5761970657842401]


# This model would not be easy to interpret given the huge number of dimensions we have. None of the F1 scores or AUCs surpasses that of the best decision tree, although it's not a bad model.
# 
# # Stochastic Gradient Descent Classifier

# In[ ]:


from sklearn.linear_model import SGDClassifier

def objective_sgd(hyperparameters):
    for parameter_name in ['alpha','l1_ratio']:
        hyperparameters[parameter_name] = float(hyperparameters[parameter_name])

    model = SGDClassifier(**hyperparameters, shuffle = False,
                          n_jobs = -1, random_state = 1337)
    
    score = evaluate(model)[0]
    loss = 1 - score
    
    return {'loss': loss, 'hyperparameters': hyperparameters,
            'status': STATUS_OK, 'score': score}
        
space_sgd = {
    'loss': hp.choice('loss', ['log', 'hinge']),
    'penalty': hp.choice('penalty', ['l1', 'l2']),
    'alpha': hp.loguniform('alpha', np.log(0.0001),
                                    np.log(0.1)),
    'l1_ratio': hp.quniform('l1_ratio', 0.0, 1.0, 0.1),
}


# In[ ]:


trials = Trials()

best = fmin(objective_sgd, space_sgd, algo=tpe.suggest, max_evals=50,
            trials = trials, rstate = np.random.RandomState(1337))

print(best)

# {'alpha': 0.0003075800442327615, 'l1_ratio': 0.0, 'loss': 0, 'penalty': 1}


# In[ ]:


scores = [trials.results[i]['score'] for i in range(50)]

plt.scatter(range(1,51), scores)
plt.show()


# In[ ]:


sgd = SGDClassifier(alpha = 0.0003075800442327615,
                    l1_ratio = 0.0, loss = 'log',
                    penalty = 'l2', shuffle = False,
                    n_jobs = -1, random_state = 1337)

evaluate(sgd)


# In[ ]:


mean_scores = []
mean_roc = []

sgd = SGDClassifier(alpha = 0.0001543133712738374,
                    l1_ratio = 1.0, loss = 'log',
                    penalty = 'l1', shuffle = False,
                    n_jobs = -1, random_state = 1337)

for n in nums_pca:
    pca = PCA(n_components = n)
    mean_scores.append(evaluate(sgd, use_pca = True)[0])
    mean_roc.append(evaluate(sgd, use_pca = True)[1])
    
print(mean_scores)
print(mean_roc)

# [0.6400306147844548, 0.6510323586028245, 0.6577273095976571, 0.6673763711628204]
# [0.5, 0.5068256298736168, 0.5136870162693423, 0.523999170417162]


# The stochatic gradient descent classifier doesn't come close to beating our best decision tree.
# 
# # Random Forest
# 
# Unfortunately random forests are something of a black box, but we can still inspect feature importances if this model succeeds. Attempting a Bayesian optimization search for the best number of estimators, keeping the best hyperparameters from our decision tree constant, was unsuccessful due to runtime limits. First let's just look at the best decision tree hyperparameters applied to an RF.

# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


rf = RandomForestClassifier(n_estimators = 20, random_state = 1337,
                            criterion = 'gini',
                            max_depth = 10,
                            max_features = None,
                            min_samples_leaf = 8,
                            min_samples_split = 14,
                            n_jobs = -1)

evaluate(rf)

# (0.7441793448895122, 0.6199220302462896, [])


# Almost as good as the best decision tree, but not quite. Let's try the default hyperparameters, with the exception of n_estimators.

# In[ ]:


rf = RandomForestClassifier(n_estimators = 20, random_state = 1337,
                            n_jobs = -1)

evaluate(rf)

# (0.7316282555436648, 0.6103295171382959, [])


# This is worse, alas, although impressive for a default. What if we try to optimize the same hyperparameters as we did for the decision tree? The runtime for this process is rather impractical, even with a GPU and with an upper bound of 10 iterations (at least for my machine). I limited the number of estimators to 10 for this.

# In[ ]:


def objective_rf(hyperparameters):
    for parameter_name in ['min_samples_split','min_samples_leaf']:
        hyperparameters[parameter_name] = int(hyperparameters[parameter_name])

    model = RandomForestClassifier(**hyperparameters, n_estimators = 10,
                                   n_jobs = -1, random_state = 1337)
    
    score = evaluate(model)[0]
    loss = 1 - score
    
    return {'loss': loss, 'hyperparameters': hyperparameters,
            'status': STATUS_OK, 'score': score}
        
space_rf = {
     'criterion': hp.choice('criterion', ['gini', 'entropy']),
    'max_depth': hp.quniform('max_depth', 1, 38, 1),
    'min_samples_split': hp.quniform('min_samples_split', 2,60,2),
    'min_samples_leaf': hp.quniform('min_samples_leaf', 1,30,2),
    'max_features': hp.choice('max_features', ['sqrt', 'log2', None])
}


# In[ ]:


trials = Trials()

best = fmin(objective_rf, space_rf, algo=tpe.suggest, max_evals=10,
            trials = trials, rstate = np.random.RandomState(1337))

print(best)

# {'criterion': 1, 'max_depth': 11.0, 'max_features': 2,
# 'min_samples_leaf': 16.0, 'min_samples_split': 26.0}


# In[ ]:


rf = RandomForestClassifier(n_estimators = 10, random_state = 1337,
                            n_jobs = -1, criterion = 'entropy',
                           max_depth = 11, max_features = None,
                           min_samples_leaf = 16, min_samples_split = 26)

evaluate(rf)

# (0.7460403119906623, 0.62293146426257, [])


# Even with a modest number of trees, the Bayesian optimized random forest performs slightly better than the decision tree!

# In[ ]:


def objective_rf_pca(hyperparameters):
    for parameter_name in ['min_samples_split','min_samples_leaf']:
        hyperparameters[parameter_name] = int(hyperparameters[parameter_name])

    model = RandomForestClassifier(**hyperparameters, n_estimators = 10,
                                   n_jobs = -1, random_state = 1337)
    
    score = evaluate(model, use_pca = True)[0]
    loss = 1 - score
    
    return {'loss': loss, 'hyperparameters': hyperparameters,
            'status': STATUS_OK, 'score': score}
        
space_rf_pca = {
     'criterion': hp.choice('criterion', ['gini', 'entropy']),
    'max_depth': hp.quniform('max_depth', 1, 38, 1),
    'min_samples_split': hp.quniform('min_samples_split', 2,60,2),
    'min_samples_leaf': hp.quniform('min_samples_leaf', 1,30,2),
    'max_features': hp.choice('max_features', ['sqrt', 'log2', None])
}


# In[ ]:


pca = PCA(n_components = 3)

trials = Trials()

best = fmin(objective_rf_pca, space_rf_pca, algo=tpe.suggest, max_evals=10,
            trials = trials, rstate = np.random.RandomState(1337))

print(best)


# In[ ]:


rf = RandomForestClassifier(n_estimators = 10, random_state = 1337,
                            n_jobs = -1, criterion = 'entropy',
                           max_depth = 11, max_features = None,
                           min_samples_leaf = 16, min_samples_split = 26)

evaluate(rf, use_pca=True)

# (0.7188045621044724, 0.6024729342981416, [])


# Not bad, but still not in the league even of the default RF.

# # Gradient Boosting

# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier


# In[ ]:


gb = GradientBoostingClassifier(random_state = 1337,
                            max_depth = 10,
                            max_features = None,
                            min_samples_leaf = 8,
                            min_samples_split = 14,
                            n_estimators = 20)

evaluate(gb)


# In[ ]:


gb = GradientBoostingClassifier(random_state = 1337,
                            max_depth = 11,
                            max_features = None,
                            min_samples_leaf = 16,
                            min_samples_split = 26,
                            n_estimators = 20)

pca = PCA(n_components = 3)

evaluate(gb, use_pca = True)


# Exploring the gradient boosting model in more depth would be ideal, but executing it just once with 20 estimators takes significant runtime. Considering the reputation of this algorithm, it would likely match if not surpass the best decision tree given a larger number of iterations to work with. Still,  it achieves a decent F1 score.

# In[ ]:


def objective_gb_pca(hyperparameters):
    for parameter_name in ['min_samples_split','min_samples_leaf']:
        hyperparameters[parameter_name] = int(hyperparameters[parameter_name])

    model = GradientBoostingClassifier(**hyperparameters, n_estimators = 20,
                                   random_state = 1337)
    
    score = evaluate(model, use_pca = True)[0]
    loss = 1 - score
    
    return {'loss': loss, 'hyperparameters': hyperparameters,
            'status': STATUS_OK, 'score': score}
        
space_gb_pca = {
    'max_depth': hp.quniform('max_depth', 1, 38, 1),
    'min_samples_split': hp.quniform('min_samples_split', 2,60,2),
    'min_samples_leaf': hp.quniform('min_samples_leaf', 1,30,2),
    'max_features': hp.choice('max_features', ['sqrt', 'log2', None])
}


# In[ ]:


pca = PCA(n_components = 3)

trials = Trials()

best = fmin(objective_gb_pca, space_gb_pca, algo=tpe.suggest, max_evals=10,
            trials = trials, rstate = np.random.RandomState(1337))

print(best)


# In[ ]:


gb = GradientBoostingClassifier(random_state = 1337,
                            max_depth = 20,
                            max_features = None,
                            min_samples_leaf = 28,
                            min_samples_split = 24,
                            n_estimators = 50)

pca = PCA(n_components = 3)

evaluate(gb, use_pca = True)


# # Naive Bayes
# 
# Since almost all of the features are binary, a Bernoulli Naive Bayes classifier might be well-suited to this problem. We'll need to binarize the allele frequencies - the threshold is a hyperparameter we can tune.

# In[ ]:


from sklearn.naive_bayes import BernoulliNB

def objective_nb(hyperparameters):
    for parameter_name in ['alpha','binarize']:
        hyperparameters[parameter_name] = float(hyperparameters[parameter_name])

    model = BernoulliNB(**hyperparameters)
    
    score = evaluate(model)[0]
    loss = 1 - score
    
    return {'loss': loss, 'hyperparameters': hyperparameters,
            'status': STATUS_OK, 'score': score}
        
space_nb = {
    'alpha': hp.quniform('alpha', 0.0, 1.0, 0.1),
    'binarize': hp.quniform('binarize', 0.1, 0.9, 0.1),
}


# In[ ]:


trials = Trials()

best = fmin(objective_nb, space_nb, algo=tpe.suggest, max_evals=50,
            trials = trials, rstate = np.random.RandomState(1337))

print(best)


# In[ ]:


nb = BernoulliNB(alpha = 0.1, binarize = 0.9)

evaluate(nb)

# (0.7107953936780353, 0.6710188402129263, [])


# With very little computation time, the Naive Bayes classifier achieves a decent F1 score, but it's not comparable to the decision tree. Let's try PCA on this.

# In[ ]:


mean_scores = []
mean_roc = []

nb = BernoulliNB(alpha = 1.0, binarize = 0.9)

for n in nums_pca:
    pca = PCA(n_components = n)
    mean_scores.append(evaluate(sgd, use_pca = True)[0])
    mean_roc.append(evaluate(sgd, use_pca = True)[1])
    
print(mean_scores)
print(mean_roc)


# In[ ]:


def objective_nb_pca(hyperparameters):
    for parameter_name in ['alpha','binarize']:
        hyperparameters[parameter_name] = float(hyperparameters[parameter_name])

    model = BernoulliNB(**hyperparameters)
    
    score = evaluate(model, use_pca = True)[0]
    loss = 1 - score
    
    return {'loss': loss, 'hyperparameters': hyperparameters,
            'status': STATUS_OK, 'score': score}
        
space_nb_pca = {
    'alpha': hp.quniform('alpha', 0.0, 1.0, 0.1),
    'binarize': hp.quniform('binarize', 0.1, 0.9, 0.1),
}


# In[ ]:


pca = PCA(n_components = 3)

trials = Trials()

best = fmin(objective_nb_pca, space_nb_pca, algo=tpe.suggest, max_evals=50,
            trials = trials, rstate = np.random.RandomState(1337))

print(best)


# In[ ]:


nb = BernoulliNB(alpha = 0.7000000000000001, binarize = 0.4)

evaluate(nb, use_pca = True)


# This doesn't help. Nonetheless, as a final analysis, we can see if an ensemble of our most promising models will surpass all of them.
# 
# # Majority Vote Ensembles

# In[ ]:


from sklearn.ensemble import VotingClassifier

tree = DecisionTreeClassifier(criterion = 'gini',
                               max_depth = 10,
                               max_features = None,
                               min_samples_leaf = 8,
                               min_samples_split = 14,
                               splitter = 'best',
                               random_state = 1337)

nb = BernoulliNB(alpha = 0.1, binarize = 0.9)

knn = KNeighborsClassifier(n_neighbors=3, 
                           p=2, metric='minkowski')


# In[ ]:


majority = VotingClassifier(estimators=[('tree', tree),
                            ('nb', nb), ('knn', knn)],
                            voting='soft', weights=[2,1,1])

evaluate(majority)

# (0.745711670887099, 0.650576691374489, [])


# The majority vote ensembles don't quite surpass the best random forest in terms of F1 score, although we see the best performance in terms of AUROC. Tinkering with weights for the votes revealed that performance increased most apparently when the decision tree was up-weighted - suggesting that these three models are probably failing on similar examples.

# # Conclusions
# 
# Despite facing some brick walls in the form of computation time for the hyperparameter optimization of our ensemble methods, this analysis has shown that some sizeable improvements upon the most basic prediction rule can be achieved, for the prediction of conflicting classifications of these variants. In updates of this kernel I'd like to examine the feature/PC importances more closely. However, based on the comparison of these results with those from iterations in which I did not include HIGH_SYMBOL, LOW_SYMBOL, HIGH_FEATURE, or LOW_FEATURE, it's evident that identifying the genes and feature IDs with either high or low conflict rates, and marking which variants include match any of such genes/feature IDs, can provide significant predictive power. I include these lists for reference: 

# In[ ]:


print(high_feature, "\n", low_feature, "\n", high_gene, "\n", low_gene)


# Final score on the test set: 0.748! The cross-validation procedure was evidently conservative.

# In[ ]:


rf = RandomForestClassifier(n_estimators = 10, random_state = 1337,
                            n_jobs = -1, criterion = 'entropy',
                           max_depth = 11, max_features = None,
                           min_samples_leaf = 16, min_samples_split = 26)

pca = PCA(n_components = 3)

rf.fit(X_train_sets[0], y_train_sets[0])
print(rf.score(X_test, y_test))

