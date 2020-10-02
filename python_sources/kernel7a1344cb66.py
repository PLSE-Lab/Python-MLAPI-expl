import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import re
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Open tsv file
for dirname, _, filenames in os.walk('/kaggle/input/BitterSweet'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
path = r"/kaggle/input/BitterSweet/"
sweet_train_filename = "sweet-train.tsv"
sweet_test_filename = "sweet-test.tsv"
# bitter_filename = "bitter-train.tsv"
sweet_train = pd.read_csv(path+sweet_train_filename, sep='\t')
sweet_test= pd.read_csv(path+sweet_test_filename, sep='\t')

# bitter = pd.read_csv(path+bitter_filename, sep='\t')

# Define function to count number of cyclic patterns in the molecule
def count_digits(string):
    return sum(item.isdigit() for item in string)

def count_atoms(symbol, data):
    return data['SMILES'].str.lower().str.count(symbol)

# Build columns with corresponding atoms
atoms = ["c", "n", "s", "o", "cl", "\=", "\@", "\[", "\("]
data = [sweet_train, sweet_test]
for d in data:
    for atom in atoms:
        d[atom] = count_atoms(atom, d)
    d['c'] = d['c'] - d['cl']
    d['Cyclic'] = d['SMILES'].apply(count_digits) / 2
    d.rename(columns={'\=': 'DoubleBonds',
                     '\(': 'SideChain',
                     '\@': 'Chirality',
                      '\[': 'Ions'},
            inplace=True)

# print(bitter['cyclic'][1])
#
# Visualise dependencies between sweet taste and a factor

sweet_train.groupby('c').mean()
# bitter.groupby('c').mean()

sweet_train['Sweet'] = sweet_train['Sweet'].astype(int)
sweet_train.groupby('Sweet').size().plot.bar()
plt.savefig('Taste_Sweet')

pd.crosstab(sweet_train.c, sweet_train.Sweet).plot(kind='bar')
plt.savefig('C_Sweet')
# pd.crosstab(bitter.c, bitter.Bitter).plot(kind='bar')
# plt.savefig('C_Bitter')

pd.crosstab(sweet_train.SideChain, sweet_train.Sweet).plot(kind='bar')
plt.savefig('SideChain_Sweet')
# pd.crosstab(bitter.SideChain, bitter.Bitter).plot(kind='bar')
# plt.savefig('SideChain_Bitter')

pd.crosstab(sweet_train.DoubleBonds, sweet_train.Sweet).plot(kind='bar')
plt.savefig('DoubleBonds_Sweet')
# pd.crosstab(bitter.DoubleBonds, bitter.Bitter).plot(kind='bar')
# plt.savefig('DoubleBonds_Bitter')

pd.crosstab(sweet_train.Cyclic, sweet_train.Sweet).plot(kind='bar')
plt.savefig('Cyclic_Sweet')
# pd.crosstab(bitter.Cyclic, bitter.Bitter).plot(kind='bar')
# plt.savefig('Cyclic_Bitter')

pd.crosstab(sweet_train.n, sweet_train.Sweet).plot(kind='bar')
plt.savefig('N_Sweet')
# pd.crosstab(bitter.n, bitter.Bitter).plot(kind='bar')
# plt.savefig('N_Bitter')

pd.crosstab(sweet_train.c, sweet_train.Sweet).plot(kind='bar')
plt.savefig('C_Sweet')
# pd.crosstab(bitter.c, bitter.Bitter).plot(kind='bar')
# plt.savefig('C_Bitter')

pd.crosstab(sweet_train.Ions, sweet_train.Sweet).plot(kind='bar')
plt.savefig('Ions_Sweet')
# pd.crosstab(bitter.Ions, bitter.Bitter).plot(kind='bar')
# plt.savefig('Ions_Bitter')

pd.crosstab(sweet_train.Chirality, sweet_train.Sweet).plot(kind='bar')
plt.savefig('Chirality_Sweet')
# pd.crosstab(bitter.Chirality, bitter.Bitter).plot(kind='bar')
# plt.savefig('Chirality_Bitter')

pd.crosstab(sweet_train.cl, sweet_train.Sweet).plot(kind='bar')
plt.savefig('cl_Sweet')
# pd.crosstab(bitter.cl, bitter.Bitter).plot(kind='bar')
# plt.savefig('cl_Bitter')

pd.crosstab(sweet_train.s, sweet_train.Sweet).plot(kind='bar')
plt.savefig('S_Sweet')
# pd.crosstab(bitter.s, bitter.Bitter).plot(kind='bar')
# plt.savefig('S_Bitter')

pd.crosstab(sweet_train.o, sweet_train.Sweet).plot(kind='bar')
plt.savefig('O_Sweet')
# pd.crosstab(bitter.o, bitter.Bitter).plot(kind='bar')
# plt.savefig('O_Bitter')

# build a boxplot
sweet_train.boxplot(column=['c', 'o', 'n', 's', ], by='Sweet')
plt.savefig('CONS_boxplot')
sweet_train.boxplot(column=['cl', 'DoubleBonds', 'Chirality', 'Ions'], by='Sweet')
plt.savefig('Cl_Doub_Chir_Ions')
sweet_train.boxplot(column=['SideChain', 'Cyclic'], by='Sweet')
plt.savefig('Side_Cyclic')
# Where the distribution has many outliers, try using log transformation iin order to bring it closer to the normal distribution
sweet_train['C_log_value'] = np.log(sweet_train['c'])
sweet_train.boxplot(column=['C_log_value'], by='Sweet')
plt.savefig('C_log_box')

# Conclusion: Use all the factors besides chirality

# Introduce dummy variables

# factors=['c','n','o','s','cl','DoubleBonds','Chirality','SideChain','Cyclic', 'Ions']
for d in data:
    d['C>20'] = (d['c']>=20)
    d['C>20'] = d['C>20'].astype(int)
    d['DoubleBonds'] = (d['DoubleBonds']>=4).astype(int)
    d['Ions>0'] = (d['Ions']>0).astype(int)

"""
for factor in factors:
    factors_list='factor'+'_'+factor
    factors_list = pd.get_dummies(sweet[factor], prefix=factor)
    sweet1=sweet.join(factors_list)
    sweet=sweet1

factors=['c','n','o','s','cl','DoubleBonds','Chirality','SideChain','Cyclic', 'Ions']
sweet_factors=sweet.columns.values.tolist()
to_keep=[i for i in sweet_factors if i not in factors]

sweet_final=sweet[to_keep]
sweet_final.columns.values

"""
# Recursive Feature Elimination
from sklearn.feature_selection import RFE
from  sklearn.linear_model import LogisticRegression

feature_cols = ['c', 'o', 'n', 's', 'cl', 'DoubleBonds', 'Ions', 'SideChain', 'Cyclic']
X_train = sweet_train[feature_cols] #Features
y_train= sweet_train.Sweet #Target variable

logreg = LogisticRegression()
rfe = RFE(logreg, 20)
fre = rfe.fit(X_train, y_train.values.ravel())
print(rfe.support_)
print(rfe.ranking_)


# Logistic model implementation

import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression

feature_cols = ['c', 'o', 'n', 's', 'cl', 'DoubleBonds', 'Ions', 'SideChain', 'Cyclic']
X_train = sweet_train[feature_cols] #Features
y_train= sweet_train.Sweet #Target variable

X_test = sweet_test[feature_cols]
y_test = sweet_test['Sweet']


logreg = LogisticRegression(C=1e10)
logreg.fit(X_train,y_train)
y_pred=logreg.predict(X_test)

result=logreg.fit(X_train,y_train)

import statsmodels.api as sm
logit_model=sm.Logit(y_train,X_train, C=1e0)
result1=logit_model.fit()
print(result1.summary2())

results_file = open("Logit_results_C1e10", "w")
results_file.write(format(result1.summary2()))
results_file.close()


# Logistic regression model fitting

from sklearn.linear_model import LogisticRegression
from sklearn  import metrics

logreg = LogisticRegression(C=1e10)
logreg.fit(X_train, y_train)
results2 = logreg.fit(X_train, y_train)

results2_file = open("Logreg_results", "w")
results2_file.write(format(results2))
results2_file.close()

# Predicting the test results and calculating the accuracy

y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set:{:.2f}'.format(logreg.score(X_test, y_test)))

# Confusion matrix
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)

# Compute precision, recall, F-measure and support
from sklearn.metrics import classification_report
# print(classification_report(y_test, y_pred))
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))

# ROC Curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
# y_pred_proba = logreg.predict_proba(X_test)[::,1]
logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
# auc = metrics.roc_auc_score(y_test, y_pred_proba)
fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])

# plt.plot(fpr,tpr,label="data 1, auc="+str(auc))

# plt.show()

plt.figure()
plt.plot(fpr, tpr, label="auc="+str(logit_roc_auc))
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
# plt.legend(loc="lower right")
plt.legend(loc=4)
plt.savefig('Log_ROC')
plt.show()
my_submission = pd.DataFrame({'Id': sweet_test['SMILES'], 'Sweet_Taste': y_pred})
my_submission.to_csv('submission.csv', index=False)