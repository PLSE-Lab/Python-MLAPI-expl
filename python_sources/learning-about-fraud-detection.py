# imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import average_precision_score
from xgboost.sklearn import XGBClassifier
from xgboost import plot_importance, to_graphviz

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

pd.set_option('display.max_columns', 11)

# import data

df = pd.read_csv('../input/PS_20174392719_1491204439457_log.csv')
df = df.rename(columns={
    'oldbalanceOrg': 'oldBalanceOrig', 
    'newbalanceOrig': 'newBalanceOrig',
    'oldbalanceDest': 'oldBalanceDest',
    'newbalanceDest': 'newBalanceDest'
})

"""
    Explanatory Data Analysis (EDA)
        This section intents to gain insights into the dataset
        
    Insights found:
        - Fraudulent transactions only occurs in TRANSFER and CASH_OUT types
        - isFlaggedFraud, nameOrig and nameDest features are meaningless
"""

# which types of transactions are fraudulent? how much fraudulent transactions each type have?

print('\n The types of fraudulent transactions are {}'.format(
    list(df.loc[df.isFraud == 1].type.drop_duplicates().values)
))

dfFraudTransfer = df.loc[(df.isFraud == 1) & (df.type == 'TRANSFER')]
dfFraudCashout = df.loc[(df.isFraud == 1) & (df.type == 'CASH_OUT')]

print('The number of fraudulent TRANFERs = {}'.format(len(dfFraudTransfer)))
print('The number of fraudulent CASH_OUTs = {}'.format(len(dfFraudCashout)))

# what determines whether the feature isFlaggedFraud gets set or not?

print('The type of transactions in which isFlaggedFraud is set: {}'.format(
    list(df.loc[df.isFlaggedFraud == 1].type.drop_duplicates()
)))

dfTransfer = df.loc[df.type == 'TRANSFER']
dfFlagged = df.loc[df.isFlaggedFraud == 1]
dfNotFlagged = df.loc[df.isFlaggedFraud == 0]

print('Min amount transacted when isFlaggedFraud is set = {}'.format(
    dfFlagged.amount.min()    
))
print('Max amount transacted in a TRANSFER where isFlaggedFraud is not set = {}'.format(
    dfTransfer.loc[dfTransfer.isFlaggedFraud == 0].amount.max()    
))

print('The number of TRANSFERs where isFlaggedFraud = 0, yet oldBalanceDest = 0 and newBalanceDest = 0: {}'.format(
    len(dfTransfer.loc[(dfTransfer.isFlaggedFraud == 0) & (dfTransfer.oldBalanceDest == 0) & (dfTransfer.newBalanceDest == 0)])    
))

print('Min, Max of oldBalanceOrig for isFlaggedFraud = 1 TRANSFERs: {}'.format(
    [round(dfFlagged.oldBalanceOrig.min()), round(dfFlagged.oldBalanceOrig.max())]    
))
print('Min, Max of oldBalanceOrig for isFlaggedFraud = 0 TRANSFERs where oldBalanceOrig = newBalanceOrig: {}'.format(
    [dfTransfer.loc[(dfTransfer.isFlaggedFraud == 0) & (dfTransfer.oldBalanceOrig == dfTransfer.newBalanceOrig)].oldBalanceOrig.min(),
    dfTransfer.loc[(dfTransfer.isFlaggedFraud == 0) & (dfTransfer.oldBalanceOrig == dfTransfer.newBalanceOrig)].oldBalanceOrig.max()]    
))

print('Have originators of transactions flagged as fraud transacted more than once? {}'.format(
    (dfFlagged.nameOrig.isin(pd.concat([dfNotFlagged.nameOrig, dfNotFlagged.nameDest]))).any()    
))
print('Have destinations for transactions flagged as fraud initiated other transactions? {}'.format(
    (dfFlagged.nameDest.isin(dfNotFlagged.nameOrig)).any()
))
print('How many destination accounts of transactions flagged as fraud have been destination accounts more than once? {}'.format(
    sum(dfFlagged.nameDest.isin(dfNotFlagged.nameDest))    
))

# are expected merchant accounts accordingly labelled?

print('Are there any merchants among originator accounts for CASH_IN transactions? {}'.format(
    (df.loc[df.type == 'CASH_IN'].nameOrig.str.contains('M')).any()
))

print('Are there any merchants among destination accounts for CASH_OUT transactions? {}'.format(
    (df.loc[df.type == 'CASH_OUT'].nameDest.str.contains('M')).any()    
))

print('Are there merchants among any originator accounts? {}'.format(
    df.nameOrig.str.contains('M').any()
))
print('Are there any transactions having merchants among destination accounts other than the PAYMENT type? {}'.format(
    (df.loc[df.nameDest.str.contains('M')].type != 'PAYMENT').any()    
))

# are there account labels common to fraudulent TRANSFERs and CASH_OUTs?

print('Within fraudulent transactions, are there destinations for TRANSFERs that are also originators for CASH_OUTs? {}'.format(
    (dfFraudTransfer.nameDest.isin(dfFraudCashout.nameOrig)).any()    
))

dfNotFraud = df.loc[df.isFraud == 0]

print('Fraudulent TRANSFERs whose destination accounts are originators of genuine CASH_OUTs: \n\n{}'.format(
    (dfFraudTransfer.loc[dfFraudTransfer.nameDest.isin(dfNotFraud.loc[dfNotFraud.type == 'CASH_OUT'].nameOrig.drop_duplicates())])    
))

print('Fraudulent TRANSFER to C423543548 occured at step = 486 whereas genuine CASH_OUT from this account occured earlier at step = {}'.format(
    dfNotFraud.loc[(dfNotFraud.type == 'CASH_OUT') & (dfNotFraud.nameOrig == 'C423543548')].step.values
))


"""
    Data cleaning
        This section intents to clear data based on insights found in previous section
"""

X = df.loc[(df.type == 'TRANSFER') | (df.type == 'CASH_OUT')]

randomState = 5
np.random.seed(randomState)

Y = X['isFraud']
del X['isFraud']

X = X.drop(['nameOrig', 'nameDest', 'isFlaggedFraud'], axis=1)

X.loc[X.type == 'TRANSFER', 'type'] = 0
X.loc[X.type == 'CASH_OUT', 'type'] = 1
X.type = X.type.astype(int)

# imputation of latent missing values

Xfraud = X.loc[Y == 1]
XnonFraud = X.loc[Y == 0]

print('The fraction of fraudulent transactions with \'oldBalanceDest\' = \'newBalanceDest\' = 0 although the transacted \'amount\' is non-zero is: {0:.4g}%'.format(
    len(Xfraud.loc[(Xfraud.oldBalanceDest == 0) & (Xfraud.newBalanceDest == 0) & (Xfraud.amount)]) / (1.0 * len(Xfraud)) * 100    
))
print('The fraction of genuine transactions with \'oldBalanceDest\' = \'newBalanceDest\' = 0 although the transacted \'amount\' is non-zero is: {0:.4g}%'.format(
    len(XnonFraud.loc[(XnonFraud.oldBalanceDest == 0) & (XnonFraud.newBalanceDest == 0) & (XnonFraud.amount)]) / (1.0 * len(XnonFraud)) * 100    
))

X.loc[(X.oldBalanceDest == 0) & (X.newBalanceDest == 0) & (X.amount != 0), ['oldBalanceDest', 'newBalanceDest']] = - 1

X.loc[(X.oldBalanceOrig == 0) & (X.newBalanceOrig == 0) & (X.amount != 0), ['oldBalanceOrig', 'newBalanceOrig']] = np.nan


"""
    Feature-engineering
        Generate new features to obtain best performance from the ML algorithm
"""

X['errorBalanceOrig'] = X.newBalanceOrig + X.amount - X.oldBalanceOrig
X['errorBalanceDest'] = X.oldBalanceDest + X.amount - X.newBalanceDest


"""
    Data visualization
        In this section we're gonna visualize the differences between fraudulent and genuine transactions to confirm that the data contains enough information
"""

limit = len(X)

def plotStrip(x, y, hue, figsize=(14, 9)):
    fig = plt.figure(figsize=figsize)
    colours = plt.cm.tab10(np.linspace(0, 1, 9))
    with sns.axes_style('ticks'):
        ax = sns.stripplot(x, y, hue=hue, jitter=0.4, marker='.', size=4, palette=colours)
        ax.set_xlabel('')
        ax.set_xticklabels(['genuine', 'fraudulent'], size=16)
        for axis in ['top', 'bottom', 'left', 'right']:
                ax.spines[axis].set_linewidth(2)
                
        handles, labels = ax.get_legend_handles_labels()
        plt.legend(handles, ['Transfer', 'Cash out'], bbox_to_anchor=(1, 1), loc=2, borderaxespad=0, fontsize=16)
    return ax

# dispersion over time

ax = plotStrip(Y[:limit], X.step[:limit], X.type[:limit])
ax.set_ylabel('time [hour]', size=16)
ax.set_title('Striped vs. homogenous fingerprints of genuine and fraudulent transactions over time', size=20)

# dispersion over amount

ax = plotStrip(Y[:limit], X.amount[:limit], X.type[:limit], figsize = (14, 9))
ax.set_ylabel('amount', size = 16)
ax.set_title('Same-signed fingerprints of genuine and fraudulent transactions over amount', size = 18);

# dispersion over error in balance in destination accounts

ax = plotStrip(Y[:limit], - X.errorBalanceDest[:limit], X.type[:limit], figsize = (14, 9))
ax.set_ylabel('- errorBalanceDest', size = 16)
ax.set_title('Opposite polarity fingerprints over the error in destination account balances', size = 18);

# separating out genuine from fraudulent transactions

x = 'errorBalanceDest'
y = 'step'
z = 'errorBalanceOrig'
zOffset = 0.02
limit = len(X)

sns.reset_orig()

fig = plt.figure(figsize = (10, 12))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X.loc[Y == 0, x][:limit], X.loc[Y == 0, y][:limit], \
  -np.log10(X.loc[Y == 0, z][:limit] + zOffset), c = 'g', marker = '.', s = 1, label = 'genuine')
    
ax.scatter(X.loc[Y == 1, x][:limit], X.loc[Y == 1, y][:limit], \
  -np.log10(X.loc[Y == 1, z][:limit] + zOffset), c = 'r', marker = '.', s = 1, label = 'fraudulent')

ax.set_xlabel(x, size = 16); 
ax.set_ylabel(y + ' [hour]', size = 16); 
ax.set_zlabel('- log$_{10}$ (' + z + ')', size = 16)
ax.set_title('Error-based features separate out genuine and fraudulent transactions', size = 20)

plt.axis('tight')
ax.grid(1)

noFraudMarker = mlines.Line2D([], [], linewidth = 0, color='g', marker='.',
                          markersize = 10, label='genuine')
fraudMarker = mlines.Line2D([], [], linewidth = 0, color='r', marker='.',
                          markersize = 10, label='fraudulent')

plt.legend(handles = [noFraudMarker, fraudMarker], bbox_to_anchor = (1.20, 0.38 ), frameon = False, prop={'size': 16})


"""
    Machine learning to detect fraud in skewed data
"""

print('skew = {0:.3g}%'.format(len(Xfraud) / float(len(X)) * 100))

















