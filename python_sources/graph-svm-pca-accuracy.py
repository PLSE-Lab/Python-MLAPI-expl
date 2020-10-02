import sklearn.svm as svm
import sklearn.decomposition as decomposition
import sklearn.cross_validation as cross_validation
import matplotlib.pyplot as plt
import pandas as pd
import numpy
# The competition datafiles are in the directory ../input
# Read competition data files:
train = pd.read_csv("../input/train.csv")
test  = pd.read_csv("../input/test.csv")
labels = numpy.array(train.ix[:,0].values.astype("int32"))
train = numpy.array(train.ix[:,1:].values.astype("float32"))

components = range(5,50,10)
print(components)

scores = list()
scores_std = list()

#Train and perform PCA with 5,15,...45 components
for COMPONENT_NUM in components:
    print("PCA: ", COMPONENT_NUM)
    pca = decomposition.PCA(n_components=COMPONENT_NUM, whiten=True)
    pca.fit(train)
    pca_train = pca.transform(train)
    pca_test = pca.transform(test)
    print("Train SVM" )
    svc = svm.SVC()
    score = cross_validation.cross_val_score(svc, pca_train, labels)
    scores.append(numpy.mean(score))
    scores_std.append(numpy.std(score))

#Output Scores
sc_array = numpy.array(scores)
std_array = numpy.array(scores_std)
print('Score: ', sc_array)
print('Std  : ', std_array)

#Plot scores
plt.plot(components, scores)
plt.plot(components, sc_array + std_array, 'b--')
plt.plot(components, sc_array - std_array, 'b--')
plt.ylabel('CV score')
plt.xlabel('# of trees')
plt.savefig('cv_trees.png')


print("Done!")