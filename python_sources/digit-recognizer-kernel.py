import pandas,numpy
path_file='../input/train.csv'
path_file2='../input/test.csv'
df = pandas.read_csv(path_file)
df2=pandas.read_csv(path_file2)
df_target=df['label']
del df['label']

df_all = pandas.get_dummies(pandas.concat([df,df2]))

from sklearn.decomposition import PCA
from sklearn import preprocessing

pca_dims = PCA()
pca_dims.fit(df_all)
cumsum = numpy.cumsum(pca_dims.explained_variance_ratio_)
d = numpy.argmax(cumsum >= 0.95) + 1
pca = PCA(n_components=d)
df_all = pca.fit_transform(df_all)
df_all=preprocessing.scale(df_all)

df=df_all[:42000]
df2=df_all[42000:]

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC,SVC
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
ImageId=list(range(1,df2.shape[0]+1))

classifier_dict={'KNeighborsClassifier':KNeighborsClassifier,'LinearSVC':LinearSVC,'SVC':SVC,\
'RandomForestClassifier':RandomForestClassifier,'ExtraTreesClassifier':ExtraTreesClassifier,\
'AdaBoostClassifier':AdaBoostClassifier,'GradientBoostingClassifier':GradientBoostingClassifier,\
'DecisionTreeClassifier':DecisionTreeClassifier,'MLPClassifier':MLPClassifier}

for key,value in classifier_dict.items():
    classifier=value()
    classifier.fit(df, df_target)
    predict=classifier.predict(df2)
    array=numpy.column_stack((ImageId,predict))
    numpy.savetxt(key+'.txt',array,fmt='%i,%i',header='ImageId,Label',comments='')