import os
import gc
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
DATA_DIR = '../input/'

W, H = 9, 9
train = pd.read_csv(DATA_DIR+'train_{}_{}_mat.csv'.format(W, H))
test = pd.read_csv(DATA_DIR+'test_{}_{}_mat.csv'.format(W, H))

X = train.iloc[:, 1:].values
y = train.iloc[:, 0].values
X_test = test.iloc[:, 1:].values
y_test = test.iloc[:, 0].values

X = X / X.max().max().astype(np.float32)
X_test = X_test / X_test.max().max().astype(np.float32)

fit_bias = False
show_heatmap = True

if fit_bias:
    X = np.hstack([X, np.ones((X.shape[0], 1))])
    X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])

from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.1, random_state=42)

#####################################################
########   Original Logistic Regression   ###########
#####################################################

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score
sk_lr = LogisticRegression(
    n_jobs=4, solver='sag', fit_intercept=False
)
sk_lr.fit(X_train, y_train)

w_ori = sk_lr.coef_
print('Original weight matrix shape', w_ori.shape)

# print('sk lr valid score', sk_lr.score(X_valid, y_valid))
# print('sk lr test score', sk_lr.score(X_test, y_test))
print('sk lr valid score', accuracy_score(y_valid, np.argmax(X_valid@w_ori.T, 1)))
print('sk lr test score', accuracy_score(y_test, np.argmax(X_test@w_ori.T, 1)))

#####################################################
##########   Approximate Quantization   #############
#####################################################

L = 10
alphas = np.abs(w_ori).max(0) / L
print('Scaling factor array shape', alphas.shape)
n_pos_appr = np.round( (L+w_ori/alphas)/2 ).astype('int')
w_aq = (2 * n_pos_appr - L) * alphas
print('Equivalent Quantized Weights shape', w_aq.shape)

print('aq lr valid score', accuracy_score(y_valid, np.argmax(X_valid@w_aq.T, 1)))
print('aq lr test score', accuracy_score(y_test, np.argmax(X_test@w_aq.T, 1)))

#####################################################
###############   Show Figures   ####################
#####################################################

if show_heatmap:
    f = plt.figure(figsize=[16, 4])
    plt.subplot(121)
    sns.heatmap(w_ori, vmin=-12, vmax=12, cbar=True)
    plt.title('Original Weights')
    plt.subplot(122)
    sns.heatmap(w_aq, vmin=-12, vmax=12, cbar=True)
    plt.title('Equivalent Quantized Weights')
    plt.tight_layout()
    plt.show()
    plt.savefig('Compare_heatmap.png')

    f = plt.figure(figsize=[8, 4])
    sns.distplot(w_ori.ravel())
    sns.distplot(w_aq.ravel())
    plt.legend(['Original', 'ApprQuant'])
    plt.tight_layout()
    plt.show()
    plt.savefig('W_dist.png')
    
#####################################################
#######   Approximate Quantization log1p   ##########
#####################################################

eps = 1e-17
w_ori_sign = np.sign(w_ori + eps) # prevent 0 np.sign(0)->0
w_ori_abs = np.abs(w_ori)
w_ori_abs_log1p = np.log1p(w_ori_abs)
L = 10
alphas_log1p = w_ori_abs_log1p.max(0) / L
print('Scaling factor array shape', alphas_log1p.shape)
n_pos_appr_log1p = np.round( (L+w_ori_abs_log1p/alphas_log1p)/2 ).astype('int')
w_aq_log1p_abs = (2 * n_pos_appr_log1p - L) * alphas_log1p
w_aq_log1p = w_ori_sign*np.expm1( w_aq_log1p_abs )
print('Equivalent Quantized Weights_log1p shape', w_aq_log1p.shape)

print('aq lr_log1p valid score', accuracy_score(y_valid, np.argmax(X_valid@w_aq_log1p.T, 1)))
print('aq lr_log1p test score', accuracy_score(y_test, np.argmax(X_test@w_aq_log1p.T, 1)))

#####################################################
############   Show Figures log1p   #################
#####################################################

if show_heatmap:
    f = plt.figure(figsize=[16, 4])
    plt.subplot(121)
    sns.heatmap(w_ori, cbar=True)
    plt.title('Original Weights')
    plt.subplot(122)
    sns.heatmap(w_aq_log1p, cbar=True)
    plt.title('Equivalent Quantized Weights')
    plt.tight_layout()
    plt.show()
    plt.savefig('Compare_heatmap_log1p.png')
    
    f = plt.figure(figsize=[8, 4])
    sns.distplot(w_ori.ravel())
    sns.distplot( w_aq_log1p.ravel())
    plt.legend(['Original', 'ApprQuant_log1p'])
    plt.tight_layout()
    plt.show()
    plt.savefig('W_log1p_dist.png')

#####################################################
###############   Show W_binary   ###################
#####################################################

print('Equivalent Quantized Weights shape', w_aq.shape)
w_bin = np.zeros((w_aq.shape[0]*L, w_aq.shape[1]))
for i_row in range(n_pos_appr.shape[0]):
    for i_col in range(n_pos_appr.shape[1]):
        start_row = i_row*L
        end_row = i_row*L+L
        num_positive = n_pos_appr[i_row, i_col]
        w_bin[start_row:start_row+num_positive, i_col] = 1
        w_bin[start_row+num_positive:end_row, i_col] = -1
f = plt.figure(figsize=[8, 4])
sns.heatmap(w_bin, cbar=False)
plt.savefig('w_bin heatmap.png')
print('Quantized Weights shape L=%d times larger'%L, w_bin.shape)





















    