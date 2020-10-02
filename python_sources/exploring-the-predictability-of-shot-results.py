# Playing with the shot_logs.csv dataset to test out various techniques for
# predicting the shot result from a selection of the variables contained within it

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

#Import all necessary libraries and classes
import numpy as np
import pandas as pd
import sklearn.linear_model
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import logistic
from sklearn import model_selection, metrics, base
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from itertools import combinations

df = pd.read_csv('../input/shot_logs.csv')
#Change capitals to lower case in column names
df.columns = map(str.lower, df.columns)
df.head()

df.count()

#Check level of duplication of dataset
#By game_id, player_id, period, game_clock
df['dup'] = df.duplicated(['game_id','player_id','period','game_clock'])
df = df.sort_values(['game_id','player_id','period','game_clock'])
df[['game_id','player_id','period','game_clock','dup']].head(50)
df['dup'].sum()

#Locate example duplicates via index value
df.loc[[28670,28671]]

#Level of dataset looks like it should be 1 row per game_id, player_id, period, game_clock
#However, there are duplicates despite this with opposite shot_result values
#For the purposes of the following exercise, I will just drop any duplicates at this level
df = df.drop_duplicates(['game_id','player_id','period','game_clock'])

#To ensure independence of observations, I will continue by looking only at the first shot that each
#player makes per game

df = df.drop_duplicates(['game_id','player_id'],keep='first')
df['dup'] = df.duplicated(['game_id','player_id'])
df['dup'].sum()
df.count()

#Check unique values of certain variables
print(df['location'].unique())
print(df['w'].unique())
print(df['shot_result'].unique())
print(df['dribbles'].unique())

#Count of shot_result
print(df.groupby('shot_result').size())
print(df.groupby('shot_result').size() / df.shape[0])

#Scatter plotting dribbles vs shot_dist - splitting by shot_result
plt.scatter(df[['dribbles']['shot_result' == 'missed']],df[['shot_dist']['shot_result' == 'missed']],color='red',marker='o')
plt.scatter(df[['dribbles']['shot_result' == 'made']],df[['shot_dist']['shot_result' == 'made']],color='blue',marker = 'x')
plt.xlabel('Dribbles')
plt.ylabel('Shot Distance')
plt.show()

#Scatter plotting close_def_dist vs shot_dist - splitting by shot_result
plt.scatter(df[['close_def_dist']['shot_result' == 'missed']],df[['shot_dist']['shot_result' == 'missed']],color='red',marker='o')
plt.scatter(df[['close_def_dist']['shot_result' == 'made']],df[['shot_dist']['shot_result' == 'made']],color='blue',marker = 'x')
plt.xlabel('Closest Defender Distance')
plt.ylabel('Shot Distance')
plt.show()

#Scatter plotting close_def_dist vs shot_dist - splitting by location
plt.scatter(df[['close_def_dist']['location' == 'A']],df[['shot_dist']['LOCATION' == 'A']])
plt.scatter(df[['close_def_dist']['location' == 'A']],df[['shot_dist']['LOCATION' == 'H']])
plt.xlabel('Closest Defender Distance')
plt.ylabel('Shot Distance')
plt.show()

#Histograms
plt.hist(df[['shot_dist']['shot_result' == 'missed']],bins=15)
plt.show()
plt.hist(df[['shot_dist']['shot_result' == 'made']],bins=15)
plt.show()
#The 2 distributions above are identical which suggests that shot_dist will be a weak predictor of
#shot result

#Plot distribution of shot_result
vals = df['shot_result'].unique()
numVals = np.arange(len(df['shot_result'].unique()))
cntByVal = df.groupby('shot_result').size()
plt.bar(numVals,cntByVal, align='center')
plt.xticks(numVals,vals)
plt.xlabel('Shot Result')
plt.ylabel('Number of Records')
plt.show()

#Iteration 1 - Predict shot result in terms of shot distance

#Data preparation

#check for nulls - No nulls
print(df['shot_dist'].isnull().sum())
print(df['shot_result'].isnull().sum())

#Create dataset with explanatory variable
X = df['shot_dist']

#Change dependent variable to binary numeric types
df['shot_result_bin'] = np.where(df['shot_result'] == 'made',1,0)
y = df['shot_result_bin']

#Create training and test datasets
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.7, random_state=1)

#Re-shape the output series to have 2 dimensions, as passing 1D arrays to LogisticRegression() is deprecated
print(np.shape(X_train))
print(np.shape(y_train))
X_train = X_train.reshape((len(X_train),1))
X_test = X_test.reshape((len(X_test),1))

#Logistic Regression model
lr = logistic.LogisticRegression()

lr.fit(X_train, y_train)

#Check the values of the model coefficient and intercept
print(lr.intercept_)
print(lr.coef_)

predTrain = lr.predict(X_train)
predTest = lr.predict(X_test)

#Accuracy score

print('Training Accuracy %.2f' % metrics.accuracy_score(y_train,predTrain))
print('Test Accuracy %.2f' % metrics.accuracy_score(y_test,predTest))
#0.02% less accuracy in the test

#Get arrays for False Postive Rate and True Positive Rate and the corresponding thresholds for the values 
#in these arrays
fpr, tpr, thresholds = metrics.roc_curve(y_train, predTrain)

#Get AUC statistic
auc = metrics.auc(fpr,tpr)

print('False Positive Rate: ',fpr)
print('Frue Positive Rate: ',tpr)
print('Area Under the Curve: ',auc)

#Plot the ROC Curve
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

#Score the datasets 
probTrain = lr.predict_proba(X_train)
probTest = lr.predict_proba(X_test)

#check the distribution of probability values - Second column is the probabilities for the positive outcome
plt.hist(probTrain[:,1],color = 'blue')
plt.show()
plt.hist(probTest[:,1],color = 'red')
plt.show()
#The probability scores provided by the model do not exceed much close to 0.6 and the accuracy is similar
#So this is a weak predictor of shot result

#Try running the algorithm several times for different values of the regularization parameter
#Logistic Regression model
weights, params, models, scores = [], [], [], []
#for c in np.arange(-5,5): - Negative values not currently allowed by the kernal but do work on my machine
for c in np.arange(0,5):
        lr = logistic.LogisticRegression(C=10**c, random_state=0)
        lr.fit(X_train, y_train)
        predTrain = lr.predict(X_train)
        accuracy = metrics.accuracy_score(y_train,predTrain)
        weights.append(lr.coef_[0])
        params.append(10**c)
        models.append(lr)
        scores.append(accuracy)
weights = np.array(weights)
plt.plot(params, weights, label='Shot Distance')
plt.xlabel('Regularization Parameter')
plt.ylabel('Weight Coefficient')
plt.legend(loc='upper left')
plt.xscale('log')
plt.show()

#Plot the accuracy of the models with the different parameter values
#Accuracy has an optimum across the range of values
plt.plot(params,scores,label='Accuracy')
plt.xlabel('Regularization Parameter')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.xscale('log')
plt.show()

#Now look at a list of potential predictor variables
#Apply SBS (Sequential Backward Selection) to find the optimal feature list for predicting shot_result

#Define the SBS class
class SBS():
    def __init__(self, estimator, k_features, scoring=metrics.accuracy_score,
                 test_size=0.25, random_state=1):
        self.scoring = scoring
        self.estimator = base.clone(estimator)
        self.k_features = k_features
        self.test_size = test_size
        self.random_state = random_state

    def fit(self, X, y):
        
        X_train, X_test, y_train, y_test = \
                model_selection.train_test_split(X, y, test_size=self.test_size, 
                                                 random_state=self.random_state)

        dim = X_train.shape[1]
        self.indices_ = tuple(range(dim))
        self.subsets_ = [self.indices_]
        score = self._calc_score(X_train, y_train, 
                                 X_test, y_test, self.indices_)
        self.scores_ = [score]

        while dim > self.k_features:
            scores = []
            subsets = []

            for p in combinations(self.indices_, r=dim-1):
                score = self._calc_score(X_train, y_train, 
                                         X_test, y_test, p)
                scores.append(score)
                subsets.append(p)

            best = np.argmax(scores)
            self.indices_ = subsets[best]
            self.subsets_.append(self.indices_)
            dim -= 1

            self.scores_.append(scores[best])
        self.k_score_ = self.scores_[-1]

        return self

    def transform(self, X):
        return X[:, self.indices_]

    def _calc_score(self, X_train, y_train, X_test, y_test, indices):
        self.estimator.fit(X_train[:, indices], y_train)
        y_pred = self.estimator.predict(X_test[:, indices])
        score = self.scoring(y_test, y_pred)
        return score

#Add the full list of features to the dataset
X = df[['shot_dist','close_def_dist','dribbles']]

#Create training and test datasets
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.7, random_state=1)

#Standardize the variables
stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)
print(X_train.head())
print(X_train_std[0:5,])

#Check the distributions of the standardized variables
pd.DataFrame(X_train_std[:,0]).plot(kind='kde')
plt.show()
pd.DataFrame(X_train_std[:,1]).plot(kind='kde')
plt.show()
pd.DataFrame(X_train_std[:,2]).plot(kind='kde')
plt.show()

#Instantiate the class
sbs = SBS(lr, k_features = 1)
sbs.fit(X_train_std, y_train)

#plotting performance of feature subsets
k_feat = [len(k) for k in sbs.subsets_]
plt.plot(k_feat, sbs.scores_, marker='o')
plt.ylabel('Accuracy')
plt.xlabel('Number of features')
plt.grid()
plt.tight_layout()
plt.show()
#Accuracy is greatest with all 3 variables

#Apply dimensionality reduction via PCA (Principal Components Analysis)
#1. Standardize the full-featured dataset
#Done above
#2. Construct the covariance matrix
cov_mat = np.cov(X_train_std.T)
#3. Decompose the covariance matrix into its eigenvectors and eigenvalues
eigenvals, eigenvecs = np.linalg.eig(cov_mat)
#4. Select the top 2 eignvectors that correspond to the largest eigenvalues 
eigen_pairs = [(np.abs(eigenvals[i]),eigenvecs[:,i]) for i in range(len(eigenvals))]
eigen_pairs.sort(reverse=True)
#5. Construct a projection matrix from these top 2 eigenvectors
w = np.hstack((eigen_pairs[0][1][:, np.newaxis],
               eigen_pairs[1][1][:, np.newaxis]))
#np.newaxis transposes the vectors from horizontal to vertical
print('Matrix W: \n', w)
#6. Transform the input dataset using the projection matrix
X_train_pca = np.dot(X_train_std,w)
#Using the same projection matrix on the test as created with the training dataset
X_test_pca = np.dot(X_test_std,w)
lr.fit(X_train_pca, y_train)
predTrain = lr.predict(X_train_pca)
predTest = lr.predict(X_test_pca)

print(metrics.accuracy_score(predTrain,y_train))
print(metrics.accuracy_score(predTest,y_test))

#PCA version 2
#Using the sklearn short way

#Instantiate the PCA class
pca = PCA(n_components=2)

lr = logistic.LogisticRegression()

X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)

lr.fit(X_train_pca,y_train)

predTrain = lr.predict(X_train_pca)
predTest = lr.predict(X_test_pca)

print(metrics.accuracy_score(predTrain,y_train))
print(metrics.accuracy_score(predTest,y_test))