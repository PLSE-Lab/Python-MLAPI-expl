### thanks to nirajverma who explained 
### how he preprocessed the data

import numpy as np
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib as mpl
import matplotlib.pyplot as plt

DATA = "../input/HR_comma_sep.csv"

#loading data
df = pd.read_csv(DATA)

# feature names
columns_names=df.columns.tolist()

# print (df.head())

df['sales'].unique()

groupby_sales=df.groupby('sales').mean() # use mean
# print groupby_sales

IT=groupby_sales['satisfaction_level'].IT
RandD=groupby_sales['satisfaction_level'].RandD
accounting=groupby_sales['satisfaction_level'].accounting
hr=groupby_sales['satisfaction_level'].hr
management=groupby_sales['satisfaction_level'].management
marketing=groupby_sales['satisfaction_level'].marketing
product_mng=groupby_sales['satisfaction_level'].product_mng
sales=groupby_sales['satisfaction_level'].sales
support=groupby_sales['satisfaction_level'].support
technical=groupby_sales['satisfaction_level'].technical


# drop sales and salary
df_drop=df.drop(labels=['sales','salary'],axis=1)

cols = df_drop.columns.tolist()

cols.insert(0, cols.pop(cols.index('left')))# move targer col to the leftmost

df_drop = df_drop.reindex(columns= cols)

X = df_drop.iloc[:,1:6].values # only take the first 6 features according to PCA results
y = df_drop.iloc[:,0].values

# print (y)

from sklearn.preprocessing import StandardScaler
X_std = StandardScaler().fit_transform(X)

# print (X_std)

from sklearn.decomposition import PCA

# use PCA to analyze which feature can be droped
# pca = PCA().fit(X_std)
# plt.plot(np.cumsum(pca.explained_variance_ratio_))
# plt.xlim(0,7,1)
# plt.xlabel('Number of components')
# plt.ylabel('Cumulative explained variance')

# fig1 = plt.gcf()
# # plt.show()
# # plt.draw()
# fig1.savefig('test.png', dpi=100)



from sklearn.model_selection import train_test_split

# prepare training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size=0.20, random_state=42)

from sklearn import svm

# use SVC to train and predict
clf = svm.SVC(kernel='rbf')
clf.fit(X_train, y_train)

print('The accuracy of the svm classifier on training data is {:.2f} out of 1'.format(clf.score(X_train, y_train)))

print('The accuracy of the svm classifier on test data is {:.2f} out of 1'.format(clf.score(X_test, y_test)))
