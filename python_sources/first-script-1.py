import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split 
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

# The competition datafiles are in the directory ../input
# Read competition data files:
train = pd.read_csv("../input/train.csv")
test  = pd.read_csv("../input/test.csv")

# Write to the log:
print("Training set has {0[0]} rows and {0[1]} columns".format(train.shape))
print("Test set has {0[0]} rows and {0[1]} columns".format(test.shape))
# Any files you write to the current directory get shown as outputs

target = train.label
df = train.drop('label',axis=1)

#df = df.append(test)
#print(df.shape)

pca = PCA(n_components=200)
pca.fit(df)
X = pca.transform(df)
print ("Explained Variance Ratio Sum of 200: ",sum(pca.explained_variance_ratio_[:200]))

X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=0.25, random_state=11)
print("X_train has {0[0]} rows and {0[1]} columns".format(X_train.shape))

clf = KNeighborsClassifier(n_neighbors=50)
clf.fit(X_train, y_train)
print("Score: ", clf.score(X_test, y_test))

y = clf.predict(PCA(n_components=200).fit_transform(test))
oFile = open('submission.csv','w')
oFile.write('ImageId,Label\n')
for i in range(0,len(y)):
    oFile.write(str(i+1) + ',' + str(y[i]) + '\n')
oFile.close()
#submission_df = pd.DataFrame(y, columns=['Labels'])
#submission_df.to_csv('submission.csv')